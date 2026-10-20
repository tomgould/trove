"""
Progress tracking and resumable ingestion.

Provides:
- IngestionProgress: Track progress of an ingestion run
- IngestionCheckpoint: Save/restore state for resumable ingestion
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = ".ingest_checkpoint.json"


@dataclass
class IngestionProgress:
    """Track ingestion progress with ETA estimation."""

    total_files: int
    processed_files: int = 0
    total_chunks: int = 0
    skipped_files: int = 0
    failed_files: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    @property
    def percent(self) -> float:
        """Completion percentage (0-100)."""
        if self.total_files == 0:
            return 100.0
        return (self.processed_files / self.total_files) * 100.0

    @property
    def eta_seconds(self) -> float | None:
        """Estimated seconds remaining, or None if not enough data."""
        if self.processed_files == 0:
            return None
        elapsed = time.time() - self.start_time
        rate = self.processed_files / elapsed
        remaining = self.total_files - self.processed_files
        return remaining / rate if rate > 0 else None

    @property
    def status(self) -> str:
        """Current status string."""
        if self.processed_files >= self.total_files:
            if self.failed_files:
                return "completed_with_errors"
            return "completed"
        return "in_progress"

    def format_eta(self) -> str:
        """Format ETA as human-readable string."""
        eta = self.eta_seconds
        if eta is None:
            return "calculating..."
        minutes, seconds = divmod(int(eta), 60)
        if minutes > 0:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"


@dataclass
class IngestionCheckpoint:
    """Save/restore ingestion state for resumable ingestion."""

    client_name: str
    processed_files: list[str]
    total_files: list[str]
    chunks_indexed: int = 0
    timestamp: str = ""

    def save(self, path: Path) -> None:
        """Save checkpoint to JSON file."""
        data = {
            "client_name": self.client_name,
            "processed_files": self.processed_files,
            "total_files": self.total_files,
            "chunks_indexed": self.chunks_indexed,
            "timestamp": self.timestamp or time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved checkpoint: {len(self.processed_files)}/{len(self.total_files)} files")

    @classmethod
    def load(cls, path: Path) -> "IngestionCheckpoint | None":
        """Load checkpoint from JSON file, or None if not found."""
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(
                client_name=data["client_name"],
                processed_files=data.get("processed_files", []),
                total_files=data.get("total_files", []),
                chunks_indexed=data.get("chunks_indexed", 0),
                timestamp=data.get("timestamp", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    @staticmethod
    def checkpoint_path(client_index_path: Path) -> Path:
        """Get the checkpoint file path for a client."""
        return client_index_path / CHECKPOINT_FILENAME

    @staticmethod
    def remove(path: Path) -> None:
        """Remove checkpoint file."""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
