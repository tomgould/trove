"""
Code chunker - orchestrates code file collection and parsing.
"""

import dataclasses
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .parser import CodeParser
from .git_utils import detect_repos, walk_code_files, RepoInfo
from .extractors import CodeChunk

logger = logging.getLogger(__name__)


def _process_file_worker(args: tuple) -> list[dict]:
    """Worker function for ProcessPoolExecutor — runs in a separate process.

    Each worker creates its own CodeParser (tree-sitter Language objects aren't picklable).
    """
    file_path_str, repo_info_dict, lang_config = args
    from pathlib import Path as _Path
    from .parser import CodeParser as _CodeParser
    from .git_utils import RepoInfo as _RepoInfo

    file_path = _Path(file_path_str)
    repo_info = None
    if repo_info_dict:
        repo_info = _RepoInfo(
            repo_root=_Path(repo_info_dict["repo_root"]),
            repo_name=repo_info_dict["repo_name"],
            remote_url=repo_info_dict.get("remote_url"),
            current_branch=repo_info_dict.get("current_branch"),
        )

    try:
        parser = _CodeParser(lang_config)
        chunks = parser.parse_file(file_path, repo_info)
        return [chunk.to_dict() for chunk in chunks]
    except Exception as e:
        logging.getLogger(__name__).warning(f"Worker failed on {file_path}: {e}")
        return []


class CodeChunker:
    """
    High-level interface for chunking code from a directory.

    Handles:
    - Repository detection
    - File collection with filtering
    - Parallel parsing
    - Chunk aggregation
    """

    def __init__(self, config: dict = None):
        """
        Initialize the code chunker.

        Args:
            config: Code configuration from client.yaml
                {
                    "enabled": true,
                    "paths": ["src/", "lib/"],
                    "languages": {...},
                    "parallel_workers": 4,
                    "max_file_size_kb": 500,
                    "git": {"detect_repos": true, ...}
                }
        """
        self.config = config or {}
        self.parser = CodeParser(self.config.get("languages", {}))

    def get_extensions_for_languages(self) -> set[str]:
        """Get all file extensions to index based on language config."""
        extensions = set()
        languages = self.config.get("languages", {})

        for lang_name, lang_config in languages.items():
            if isinstance(lang_config, dict) and "extensions" in lang_config:
                for ext in lang_config["extensions"]:
                    # Ensure extension starts with dot
                    if not ext.startswith("."):
                        ext = f".{ext}"
                    extensions.add(ext.lower())

        # If no languages configured, use all supported
        if not extensions:
            extensions = self.parser.get_supported_extensions()

        return extensions

    def get_exclude_patterns(self) -> list[str]:
        """Collect all exclude patterns from config."""
        patterns = []
        languages = self.config.get("languages", {})

        for lang_config in languages.values():
            if isinstance(lang_config, dict) and "exclude" in lang_config:
                patterns.extend(lang_config["exclude"])

        return patterns

    def collect_files(self, base_paths: list[Path], repos: list[RepoInfo] = None) -> list[Path]:
        """
        Collect all code files from the given paths.

        Uses `git ls-files` for git repos (near-instant) with os.scandir fallback.

        Args:
            base_paths: List of directories to scan
            repos: Detected git repos (enables git ls-files fast path)

        Returns:
            List of file paths to process
        """
        extensions = self.get_extensions_for_languages()
        exclude_patterns = self.get_exclude_patterns()
        # Also include top-level exclude_patterns from code config
        top_level_excludes = self.config.get("exclude_patterns", [])
        if top_level_excludes:
            exclude_patterns.extend(top_level_excludes)
        max_file_size = self.config.get("max_file_size_kb", 500)
        follow_symlinks = self.config.get("git", {}).get("follow_symlinks", True)

        files = []
        for i, base_path in enumerate(base_paths):
            if not base_path.exists():
                logger.warning(f"Code path does not exist: {base_path}")
                continue

            logger.info(f"Scanning path {i + 1}/{len(base_paths)}: {base_path}")
            path_count = 0
            for file_path in walk_code_files(
                base_path,
                extensions,
                exclude_patterns,
                follow_symlinks=follow_symlinks,
                max_file_size_kb=max_file_size,
                repos=repos,
            ):
                files.append(file_path)
                path_count += 1
            logger.info(f"  Found {path_count} files in {base_path.name}")

        logger.info(f"Collected {len(files)} code files from {len(base_paths)} paths")
        return files

    def detect_repositories(self, base_paths: list[Path]) -> list[RepoInfo]:
        """
        Detect git repositories in the given paths.

        Args:
            base_paths: List of directories to scan

        Returns:
            List of detected RepoInfo objects
        """
        if not self.config.get("git", {}).get("detect_repos", True):
            return []

        repos = []
        for base_path in base_paths:
            if base_path.exists():
                repos.extend(detect_repos(base_path))

        logger.info(f"Detected {len(repos)} git repositories")
        for repo in repos:
            logger.debug(f"  - {repo.repo_name} at {repo.repo_root}")

        return repos

    def chunk_files(
        self,
        file_paths: list[Path],
        repos: list[RepoInfo] = None,
        parallel: bool = True,
    ) -> list[dict]:
        """
        Parse files and return chunks ready for indexing.

        Args:
            file_paths: List of files to process
            repos: List of detected repositories
            parallel: Whether to use parallel processing

        Returns:
            List of chunk dictionaries ready for indexing
        """
        repos = repos or []
        import os as _os
        default_workers = min(_os.cpu_count() or 4, 8)
        workers = self.config.get("parallel_workers", default_workers)

        logger.info(f"Chunking {len(file_paths)} files (parallel={parallel}, workers={workers})...")
        if parallel and workers > 1 and len(file_paths) > 10:
            chunks = self._chunk_parallel(file_paths, repos, workers)
        else:
            chunks = self._chunk_sequential(file_paths, repos)

        logger.info(f"Extracted {len(chunks)} code chunks from {len(file_paths)} files")
        return chunks

    def _chunk_sequential(
        self,
        file_paths: list[Path],
        repos: list[RepoInfo],
    ) -> list[dict]:
        """Process files sequentially."""
        from .git_utils import find_repo_for_file

        all_chunks = []
        for file_path in file_paths:
            repo_info = find_repo_for_file(file_path, repos) if repos else None
            chunks = self.parser.parse_file(file_path, repo_info)
            all_chunks.extend(chunk.to_dict() for chunk in chunks)

        return all_chunks

    def _chunk_parallel(
        self,
        file_paths: list[Path],
        repos: list[RepoInfo],
        workers: int,
    ) -> list[dict]:
        """Process files in parallel using separate processes (bypasses GIL)."""
        from .git_utils import find_repo_for_file

        # Prepare picklable args for each file
        lang_config = self.config.get("languages", {})
        work_items = []
        for fp in file_paths:
            repo_info = find_repo_for_file(fp, repos) if repos else None
            if repo_info:
                repo_dict = {
                    "repo_root": str(repo_info.repo_root),
                    "repo_name": repo_info.repo_name,
                    "remote_url": repo_info.remote_url,
                    "current_branch": repo_info.current_branch,
                }
            else:
                repo_dict = None
            work_items.append((str(fp), repo_dict, lang_config))

        all_chunks = []
        done = 0
        total = len(work_items)

        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process_file_worker, item): item
                    for item in work_items
                }
                for future in as_completed(futures):
                    try:
                        chunks = future.result()
                        all_chunks.extend(chunks)
                    except Exception as e:
                        item = futures[future]
                        logger.warning(f"Failed to process {item[0]}: {e}")
                    done += 1
                    if done % 5000 == 0:
                        logger.info(f"Chunked {done}/{total} files ({len(all_chunks)} chunks so far)...")
        except Exception as e:
            logger.warning(f"ProcessPool failed ({e}), falling back to threads")
            all_chunks = self._chunk_parallel_threads(file_paths, repos, workers)

        return all_chunks

    def _chunk_parallel_threads(
        self,
        file_paths: list[Path],
        repos: list[RepoInfo],
        workers: int,
    ) -> list[dict]:
        """Fallback: process files in parallel using threads (GIL-bound)."""
        from .git_utils import find_repo_for_file

        all_chunks = []

        def process_file(file_path: Path) -> list[dict]:
            repo_info = find_repo_for_file(file_path, repos) if repos else None
            chunks = self.parser.parse_file(file_path, repo_info)
            return [chunk.to_dict() for chunk in chunks]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_file, fp): fp
                for fp in file_paths
            }

            done = 0
            total = len(futures)
            for future in as_completed(futures):
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    file_path = futures[future]
                    logger.warning(f"Failed to process {file_path}: {e}")
                done += 1
                if done % 5000 == 0:
                    logger.info(f"Chunked {done}/{total} files ({len(all_chunks)} chunks so far)...")

        return all_chunks

    def chunk_directory(self, base_path: Path) -> list[dict]:
        """
        Convenience method to chunk all code in a directory.

        Args:
            base_path: Directory to process

        Returns:
            List of chunk dictionaries ready for indexing
        """
        return self.chunk_paths([base_path])

    def chunk_paths(self, paths: list[Path]) -> list[dict]:
        """
        Process multiple paths and return all chunks.

        Args:
            paths: List of directories or files to process

        Returns:
            List of chunk dictionaries ready for indexing
        """
        # Separate files and directories
        directories = [p for p in paths if p.is_dir()]
        files = [p for p in paths if p.is_file()]

        # Detect repos BEFORE collecting files (enables git ls-files fast path)
        repos = self.detect_repositories(directories)

        # Collect files from directories (uses git ls-files for repos)
        collected_files = self.collect_files(directories, repos=repos)

        # Combine with directly specified files
        all_files = collected_files + files

        # Chunk all files
        return self.chunk_files(all_files, repos)
