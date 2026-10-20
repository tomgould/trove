"""
Incremental indexing support for code files.

Tracks file hashes and modification times to enable efficient
re-indexing of only changed files.
"""

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# State file name
CODE_INDEX_STATE_FILE = "code_index_state.json"


@dataclass
class FileState:
    """State of an indexed file."""

    file_hash: str
    mtime: float
    chunk_ids: list[str] = field(default_factory=list)


class CodeIndexState:
    """
    Track indexed code files for incremental updates.

    Stores file hashes and modification times to detect changes.
    """

    def __init__(self, index_path: Path):
        """
        Initialize the index state.

        Args:
            index_path: Path to the index directory
        """
        self.index_path = Path(index_path)
        self.state_file = self.index_path / CODE_INDEX_STATE_FILE
        self.state: dict[str, FileState] = {}
        self._load_state()

    def _load_state(self):
        """Load state from disk."""
        if self.state_file.exists():
            try:
                size_mb = self.state_file.stat().st_size / (1024 * 1024)
                if size_mb > 1:
                    logger.info(f"Loading code index state ({size_mb:.1f} MB)...")
                with open(self.state_file, "r") as f:
                    data = json.load(f)

                for file_path, state_data in data.items():
                    self.state[file_path] = FileState(
                        file_hash=state_data["file_hash"],
                        mtime=state_data["mtime"],
                        chunk_ids=state_data.get("chunk_ids", []),
                    )

                logger.info(f"Loaded code index state: {len(self.state)} files")
            except Exception as e:
                logger.warning(f"Failed to load code index state: {e}")
                self.state = {}

    def save_state(self):
        """Save state to disk."""
        try:
            self.index_path.mkdir(parents=True, exist_ok=True)

            data = {
                file_path: {
                    "file_hash": state.file_hash,
                    "mtime": state.mtime,
                    "chunk_ids": state.chunk_ids,
                }
                for file_path, state in self.state.items()
            }

            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved code index state: {len(self.state)} files")
        except Exception as e:
            logger.warning(f"Failed to save code index state: {e}")

    def clear(self):
        """Clear all state."""
        self.state = {}
        if self.state_file.exists():
            self.state_file.unlink()

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file."""
        try:
            content = file_path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        except Exception:
            return ""

    def is_file_changed(self, file_path: Path) -> bool:
        """
        Check if a file has changed since last index.

        Returns True if:
        - File is not in state (new file)
        - File mtime changed AND hash changed
        """
        file_str = str(file_path)

        if file_str not in self.state:
            return True

        state = self.state[file_str]

        # Quick check: mtime
        try:
            current_mtime = file_path.stat().st_mtime
            if current_mtime == state.mtime:
                return False
        except OSError:
            return True

        # mtime changed, check hash
        current_hash = self.get_file_hash(file_path)
        return current_hash != state.file_hash

    def get_changes(
        self,
        current_files: list[Path],
        workers: int = 0,
    ) -> tuple[list[Path], list[Path], list[str]]:
        """
        Compare current files against stored state.

        Uses parallel workers for stat/hash checks (I/O bound).

        Args:
            current_files: List of current code file paths
            workers: Number of parallel workers for file checking

        Returns:
            Tuple of (new_or_modified_files, unchanged_files, deleted_file_paths)
        """
        if workers <= 0:
            workers = min(os.cpu_count() or 8, 32)

        logger.info(f"Building file sets ({len(current_files)} current files)...")
        current_set = {str(f) for f in current_files}
        indexed_set = set(self.state.keys())

        # Find deleted files
        deleted = indexed_set - current_set
        logger.info(f"Set comparison done: {len(deleted)} deleted files detected")

        # Categorize current files — parallel stat/hash checks
        new_or_modified = []
        unchanged = []
        checked = 0
        total = len(current_files)

        if total > 100 and workers > 1:
            logger.info(f"Checking {total} files for changes ({workers} workers)...")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_file = {
                    executor.submit(self.is_file_changed, f): f
                    for f in current_files
                }
                for future in as_completed(future_to_file):
                    f = future_to_file[future]
                    try:
                        if future.result():
                            new_or_modified.append(f)
                        else:
                            unchanged.append(f)
                    except Exception:
                        new_or_modified.append(f)
                    checked += 1
                    if checked % 10000 == 0:
                        logger.info(f"Change detection: {checked}/{total} files checked...")
        else:
            for file_path in current_files:
                if self.is_file_changed(file_path):
                    new_or_modified.append(file_path)
                else:
                    unchanged.append(file_path)

        logger.info(
            f"Code index changes: {len(new_or_modified)} modified, "
            f"{len(unchanged)} unchanged, {len(deleted)} deleted"
        )

        return new_or_modified, unchanged, list(deleted)

    def update_file(
        self,
        file_path: Path,
        chunk_ids: list[str],
    ):
        """
        Update state for a file after indexing.

        Args:
            file_path: Path to the indexed file
            chunk_ids: IDs of chunks created from this file
        """
        try:
            self.state[str(file_path)] = FileState(
                file_hash=self.get_file_hash(file_path),
                mtime=file_path.stat().st_mtime,
                chunk_ids=chunk_ids,
            )
        except Exception as e:
            logger.warning(f"Failed to update state for {file_path}: {e}")

    def remove_file(self, file_path: str) -> list[str]:
        """
        Remove a file from state.

        Args:
            file_path: Path string of the file to remove

        Returns:
            List of chunk IDs that should be deleted from index
        """
        if file_path in self.state:
            chunk_ids = self.state[file_path].chunk_ids
            del self.state[file_path]
            return chunk_ids
        return []

    def get_chunk_ids_for_file(self, file_path: str) -> list[str]:
        """Get chunk IDs for a file."""
        if file_path in self.state:
            return self.state[file_path].chunk_ids
        return []

    def get_all_chunk_ids(self) -> list[str]:
        """Get all chunk IDs from all files."""
        ids = []
        for state in self.state.values():
            ids.extend(state.chunk_ids)
        return ids


def compute_chunk_id(chunk: dict) -> str:
    """
    Compute a stable ID for a code chunk.

    Based on file path, symbol name, line numbers, and content hash.
    """
    metadata = chunk.get("metadata", {})
    source = metadata.get("source_file", "")
    symbol = metadata.get("symbol_name", "")
    symbol_type = metadata.get("symbol_type", "")
    line_start = metadata.get("line_start", 0)

    # Include content hash for uniqueness when files have similar structure
    content = chunk.get("content", "")
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]

    # Create a unique key
    key = f"{source}:{symbol_type}:{symbol}:{line_start}:{content_hash}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]
