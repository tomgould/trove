"""Base interface for source connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SyncResult:
    """Result of a source sync operation."""

    source_type: str
    files_added: int = 0
    files_updated: int = 0
    files_removed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return self.files_added + self.files_updated + self.files_removed

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class BaseSourceConnector(ABC):
    """Abstract base class for source connectors."""

    @abstractmethod
    def sync(self, target_dir: str) -> SyncResult:
        """Sync this source, writing files to the target directory."""
        ...

    @abstractmethod
    def status(self) -> dict:
        """Return current sync status (last sync time, file count, etc.)."""
        ...
