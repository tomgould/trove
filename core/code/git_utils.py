"""
Git repository detection and metadata extraction.

Handles:
- Detecting repo boundaries by .git presence
- Extracting repo name, remote URL, branch
- Symlink handling with cycle detection
"""

import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class RepoInfo:
    """Information about a detected git repository."""

    repo_root: Path
    repo_name: str
    remote_url: Optional[str]
    current_branch: Optional[str]

    def get_relative_path(self, file_path: Path) -> str:
        """Get path relative to repo root."""
        try:
            return str(file_path.relative_to(self.repo_root))
        except ValueError:
            return str(file_path)


def get_remote_url(repo_root: Path) -> Optional[str]:
    """Get origin remote URL for a git repository."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.debug(f"Failed to get remote URL for {repo_root}: {e}")
    return None


def get_current_branch(repo_root: Path) -> Optional[str]:
    """Get current branch name for a git repository."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            if branch:
                return branch

        # Fallback for detached HEAD - get short SHA
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return f"detached:{result.stdout.strip()}"

    except Exception as e:
        logger.debug(f"Failed to get branch for {repo_root}: {e}")
    return None


def extract_repo_name(remote_url: Optional[str], repo_root: Path) -> str:
    """
    Extract repository name from remote URL or directory name.

    Examples:
        git@github.com:anthropics/anthropic-sdk-python.git -> anthropic-sdk-python
        https://github.com/anthropics/anthropic-sdk-python.git -> anthropic-sdk-python
        /path/to/my-project -> my-project
    """
    if remote_url:
        # Match: git@host:org/repo.git or https://host/org/repo.git
        match = re.search(r"[/:]([^/]+?)(?:\.git)?$", remote_url)
        if match:
            return match.group(1)

    # Fallback to directory name
    return repo_root.name


def _make_repo_info(repo_root: Path) -> RepoInfo:
    """Create a RepoInfo for a discovered repo root."""
    remote_url = get_remote_url(repo_root)
    repo_name = extract_repo_name(remote_url, repo_root)
    current_branch = get_current_branch(repo_root)
    return RepoInfo(
        repo_root=repo_root,
        repo_name=repo_name,
        remote_url=remote_url,
        current_branch=current_branch,
    )


def detect_repos(base_path: Path) -> list[RepoInfo]:
    """
    Detect git repositories whose .git directory is directly under base_path.

    Only detects repos that are fully contained within the configured paths.
    Does NOT walk up from symlink targets to find parent repos — if a user
    symlinks to a subdirectory of a repo, only that subdirectory is indexed
    (via normal os.scandir walk), not the entire repo.

    Args:
        base_path: Directory to search for repositories

    Returns:
        List of RepoInfo objects
    """
    repos = []
    seen_roots = set()

    # Handle symlinks - resolve to real path for the base
    try:
        base_path = base_path.resolve()
    except OSError:
        logger.warning(f"Cannot resolve path: {base_path}")
        return repos

    def _add_repo(repo_root: Path):
        """Add a repo if not already seen."""
        try:
            real_root = repo_root.resolve()
        except OSError:
            return
        if real_root in seen_roots:
            return
        seen_roots.add(real_root)
        info = _make_repo_info(repo_root)
        repos.append(info)
        logger.debug(f"Detected repo: {info.repo_name} at {repo_root}")

    # Scan top-level directories in parallel for .git dirs
    logger.info(f"Scanning for .git directories under {base_path}...")

    # Check if base_path itself is a repo
    if (base_path / ".git").is_dir():
        _add_repo(base_path)

    # Collect top-level entries to scan in parallel
    top_dirs = []
    try:
        for entry in os.scandir(base_path):
            if entry.is_dir(follow_symlinks=True) and entry.name != ".git":
                top_dirs.append(Path(entry.path))
    except OSError:
        pass

    if top_dirs:
        def _scan_dir(dir_path: Path) -> list[Path]:
            """Find all .git dirs under a single top-level directory."""
            found = []
            try:
                for git_dir in dir_path.rglob(".git"):
                    if git_dir.is_dir():
                        found.append(git_dir.parent)
            except OSError:
                pass
            return found

        workers = min(len(top_dirs), os.cpu_count() or 8)
        logger.info(f"Parallel repo scan: {len(top_dirs)} top-level dirs, {workers} workers")
        git_dir_count = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_scan_dir, d): d for d in top_dirs}
            for future in as_completed(futures):
                for repo_root in future.result():
                    git_dir_count += 1
                    _add_repo(repo_root)
    else:
        git_dir_count = 0

    logger.info(f"Repo scan complete: found {len(repos)} repos from {git_dir_count} .git directories")
    return repos


def find_repo_for_file(file_path: Path, repos: list[RepoInfo]) -> Optional[RepoInfo]:
    """
    Find which repository a file belongs to.

    Args:
        file_path: Path to the file
        repos: List of detected repositories

    Returns:
        RepoInfo if file is in a repo, None otherwise
    """
    try:
        file_path = file_path.resolve()
    except OSError:
        return None

    # Find the repo with the longest matching path (most specific)
    best_match = None
    best_length = 0

    for repo in repos:
        try:
            repo_root = repo.repo_root.resolve()
            if file_path.is_relative_to(repo_root):
                if len(str(repo_root)) > best_length:
                    best_match = repo
                    best_length = len(str(repo_root))
        except (OSError, ValueError):
            continue

    return best_match


# Global excludes - these are ALWAYS skipped
GLOBAL_EXCLUDES = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "vendor",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    ".tox",
    ".eggs",
    "*.egg-info",
    ".cache",
    ".idea",
    ".vscode",
}


def should_skip_directory(dir_name: str) -> bool:
    """Check if a directory should be skipped (global excludes)."""
    return dir_name in GLOBAL_EXCLUDES or dir_name.endswith(".egg-info")


def list_repo_files(
    repo_root: Path,
    extensions: set[str],
    exclude_patterns: list[str] = None,
    max_file_size_kb: int = 500,
) -> list[Path]:
    """
    List code files in a git repo using `git ls-files` (reads git index, near-instant).

    Falls back to None if git command fails (caller should use walk_code_files).

    Args:
        repo_root: Root of the git repository
        extensions: Set of file extensions to include
        exclude_patterns: Glob patterns to exclude
        max_file_size_kb: Skip files larger than this

    Returns:
        List of file paths, or None if git ls-files failed
    """
    import fnmatch

    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
    except Exception as e:
        logger.debug(f"git ls-files failed for {repo_root}: {e}")
        return None

    exclude_patterns = exclude_patterns or []
    files = []

    for line in result.stdout.splitlines():
        if not line:
            continue

        path = repo_root / line

        # Extension filter
        if path.suffix.lower() not in extensions:
            continue

        # Exclude patterns
        excluded = False
        path_str = str(path)
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path.name, pattern):
                excluded = True
                break
            if pattern.endswith("/**"):
                dir_pattern = pattern[:-3]
                if fnmatch.fnmatch(path_str, dir_pattern):
                    excluded = True
                    break
        if excluded:
            continue

        # File size check (stat only filtered files)
        if max_file_size_kb > 0:
            try:
                size_kb = path.stat().st_size / 1024
                if size_kb > max_file_size_kb:
                    continue
            except OSError:
                continue

        files.append(path)

    return files


def _is_child_repo(repo: RepoInfo, base_path: Path) -> bool:
    """Check if repo root is a child of base_path."""
    try:
        repo_resolved = repo.repo_root.resolve()
        base_resolved = base_path.resolve()
        return repo_resolved.is_relative_to(base_resolved) and repo_resolved != base_resolved
    except (OSError, ValueError):
        return False


def walk_code_files(
    base_path: Path,
    extensions: set[str],
    exclude_patterns: list[str] = None,
    follow_symlinks: bool = True,
    max_file_size_kb: int = 500,
    repos: list[RepoInfo] = None,
) -> Iterator[Path]:
    """
    Walk directory tree and yield code files.

    For git repositories, uses `git ls-files` (near-instant) instead of
    recursive directory walking. Falls back to os.scandir walk for non-git dirs.

    Handles:
    - Git fast path via `git ls-files` for known repos
    - Symlink following with cycle detection
    - Global excludes (node_modules, .git, etc.)
    - Extension filtering
    - File size limits

    Args:
        base_path: Starting directory
        extensions: Set of file extensions to include (e.g., {".py", ".js"})
        exclude_patterns: Additional glob patterns to exclude
        follow_symlinks: Whether to follow symbolic links
        max_file_size_kb: Skip files larger than this (0 = no limit)
        repos: List of detected git repos (enables git ls-files fast path)

    Yields:
        Paths to code files
    """
    import fnmatch

    repos = repos or []
    seen_real_paths = set()
    exclude_patterns = exclude_patterns or []

    # Try git fast path: if base_path is inside a repo, use git ls-files
    for repo in repos:
        try:
            base_resolved = base_path.resolve()
            repo_resolved = repo.repo_root.resolve()
            if base_resolved == repo_resolved or base_resolved.is_relative_to(repo_resolved):
                files = list_repo_files(
                    repo.repo_root, extensions, exclude_patterns, max_file_size_kb,
                )
                if files is not None:
                    # If base_path is a subdirectory of repo, filter to that subtree
                    if base_resolved != repo_resolved:
                        files = [f for f in files if f.resolve().is_relative_to(base_resolved)]
                    if files:
                        logger.info(f"git ls-files: {len(files)} files from {repo.repo_name}")
                        yield from files
                        return
                    # 0 files after filtering — likely a symlink scenario
                    # where repo was found inside base_path via symlink but
                    # files resolve to paths outside base_path. Fall through
                    # to os.scandir walk which follows symlinks correctly.
                    logger.debug(
                        f"git ls-files: 0 files from {repo.repo_name} after "
                        f"filtering to {base_resolved}, falling back to walk"
                    )
        except (OSError, ValueError):
            continue

    # Check for child directories that are repos — use git ls-files for each
    repo_roots_used = set()
    if repos:
        child_repos = [r for r in repos if _is_child_repo(r, base_path)]
        if child_repos:
            logger.info(f"Collecting files from {len(child_repos)} child repos via parallel git ls-files...")
            workers = min(len(child_repos), os.cpu_count() or 8)

            def _ls_repo(repo):
                try:
                    files = list_repo_files(
                        repo.repo_root, extensions, exclude_patterns, max_file_size_kb,
                    )
                    return repo, files
                except (OSError, ValueError):
                    return repo, None

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_ls_repo, r): r for r in child_repos}
                done = 0
                for future in as_completed(futures):
                    repo, files = future.result()
                    done += 1
                    if files is not None:
                        if done % 50 == 0 or done == len(child_repos):
                            logger.info(f"git ls-files: {done}/{len(child_repos)} repos done ({repo.repo_name}, {len(files)} files)")
                        yield from files
                        try:
                            repo_roots_used.add(repo.repo_root.resolve())
                        except OSError:
                            pass
            logger.info(f"Parallel git ls-files complete: {len(repo_roots_used)} repos, {done} total")

    def _should_exclude(path_str: str, name: str) -> bool:
        """Check if path matches any exclude pattern."""
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
            if fnmatch.fnmatch(name, pattern):
                return True
            if pattern.endswith("/**"):
                dir_pattern = pattern[:-3]
                if fnmatch.fnmatch(path_str, dir_pattern):
                    return True
        return False

    def walk(dir_path: str):
        """Recursive walk using os.scandir for speed."""
        try:
            with os.scandir(dir_path) as it:
                for entry in it:
                    try:
                        name = entry.name

                        if entry.is_dir(follow_symlinks=follow_symlinks):
                            # Skip global excludes
                            if should_skip_directory(name):
                                continue

                            # Skip repos already handled via git ls-files
                            if repo_roots_used:
                                try:
                                    entry_resolved = Path(entry.path).resolve()
                                    if entry_resolved in repo_roots_used:
                                        continue
                                except OSError:
                                    pass

                            # Check exclude patterns
                            if exclude_patterns and _should_exclude(entry.path, name):
                                continue

                            # Symlink cycle detection
                            if follow_symlinks and entry.is_symlink():
                                try:
                                    real = Path(entry.path).resolve()
                                except (RuntimeError, OSError):
                                    continue
                                if real in seen_real_paths:
                                    continue
                                seen_real_paths.add(real)

                            yield from walk(entry.path)

                        elif entry.is_file(follow_symlinks=follow_symlinks):
                            # Extension filter
                            _, ext = os.path.splitext(name)
                            if ext.lower() not in extensions:
                                continue

                            # Exclude patterns
                            if exclude_patterns and _should_exclude(entry.path, name):
                                continue

                            # File size check (uses cached stat from scandir)
                            if max_file_size_kb > 0:
                                try:
                                    stat = entry.stat(follow_symlinks=follow_symlinks)
                                    if stat.st_size / 1024 > max_file_size_kb:
                                        continue
                                except OSError:
                                    continue

                            yield Path(entry.path)

                    except OSError:
                        continue
        except PermissionError:
            logger.debug(f"Permission denied: {dir_path}")
        except OSError as e:
            logger.debug(f"Error accessing {dir_path}: {e}")

    yield from walk(str(base_path))
