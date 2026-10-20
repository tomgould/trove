"""
Main code parser - coordinates language-specific extractors.
"""

import logging
from pathlib import Path
from typing import Optional

from .extractors import EXTRACTOR_MAP, CodeChunk
from .git_utils import RepoInfo, find_repo_for_file

logger = logging.getLogger(__name__)


class CodeParser:
    """
    Parse source code files and extract chunks using tree-sitter.

    Automatically selects the appropriate extractor based on file extension.
    """

    def __init__(self, language_config: dict = None):
        """
        Initialize the code parser.

        Args:
            language_config: Language configuration from client.yaml
                {
                    "python": {"extensions": [".py"], "exclude": [...]},
                    "php": {"extensions": [".php", ".module"], "exclude": [...]},
                    ...
                }
        """
        self.language_config = language_config or {}
        self._extractors = {}

    def get_extractor(self, file_path: Path):
        """Get the appropriate extractor for a file based on its extension."""
        ext = file_path.suffix.lower()

        if ext not in EXTRACTOR_MAP:
            return None

        # Cache extractor instances
        if ext not in self._extractors:
            extractor_class = EXTRACTOR_MAP[ext]
            self._extractors[ext] = extractor_class()

        return self._extractors[ext]

    def parse_file(
        self,
        file_path: Path,
        repo_info: Optional[RepoInfo] = None,
    ) -> list[CodeChunk]:
        """
        Parse a single source file and extract code chunks.

        Args:
            file_path: Path to the source file
            repo_info: Optional repository information

        Returns:
            List of CodeChunk objects
        """
        extractor = self.get_extractor(file_path)
        if extractor is None:
            logger.debug(f"No extractor for file: {file_path}")
            return []

        try:
            source_code = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return []

        # Skip empty files
        if not source_code.strip():
            return []

        try:
            chunks = extractor.extract(source_code, str(file_path))
        except Exception as e:
            logger.warning(f"Failed to parse file {file_path}: {e}")
            return []

        # Add repo info to all chunks
        if repo_info:
            for chunk in chunks:
                chunk.repo_name = repo_info.repo_name
                chunk.repo_remote = repo_info.remote_url
                chunk.repo_branch = repo_info.current_branch
                chunk.repo_relative_path = repo_info.get_relative_path(file_path)

        return chunks

    def parse_files(
        self,
        file_paths: list[Path],
        repos: list[RepoInfo] = None,
    ) -> list[CodeChunk]:
        """
        Parse multiple source files.

        Args:
            file_paths: List of file paths to parse
            repos: List of detected repositories for metadata

        Returns:
            List of all extracted CodeChunk objects
        """
        repos = repos or []
        all_chunks = []

        for file_path in file_paths:
            repo_info = find_repo_for_file(file_path, repos) if repos else None
            chunks = self.parse_file(file_path, repo_info)
            all_chunks.extend(chunks)

        return all_chunks

    def get_supported_extensions(self) -> set[str]:
        """Get all supported file extensions."""
        extensions = set(EXTRACTOR_MAP.keys())

        # Add any custom extensions from config
        for lang_config in self.language_config.values():
            if "extensions" in lang_config:
                extensions.update(lang_config["extensions"])

        return extensions
