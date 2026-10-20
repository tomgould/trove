"""
Ingestion job management for MCP tools.

Provides thread-safe job tracking with max 1 concurrent ingestion job.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from core.ingest_progress import IngestionProgress

logger = logging.getLogger(__name__)


@dataclass
class IngestionJob:
    """Represents a single ingestion job."""

    job_id: str
    client_name: str
    job_type: str  # "file", "url", "reindex"
    status: str = "pending"  # "pending", "running", "completed", "failed"
    progress: Optional[IngestionProgress] = None
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Serialize job to dict for MCP responses."""
        d = {
            "job_id": self.job_id,
            "client_name": self.client_name,
            "job_type": self.job_type,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "age_seconds": int(time.time() - self.created_at),
        }
        if self.progress:
            d["progress"] = {
                "percent": round(self.progress.percent, 1),
                "processed_files": self.progress.processed_files,
                "total_files": self.progress.total_files,
                "total_chunks": self.progress.total_chunks,
                "eta": self.progress.format_eta(),
                "failed_files": self.progress.failed_files,
                "status": self.progress.status,
            }
        return d


class IngestionJobManager:
    """Thread-safe manager for ingestion jobs. Max 1 concurrent job."""

    def __init__(self, max_concurrent: int = 1):
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestionJob] = {}
        self._max_concurrent = max_concurrent
        self._running_count = 0

    def _count_running(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "running")

    def create_job(self, client_name: str, job_type: str) -> IngestionJob:
        """Create a new job. Raises if max concurrent jobs reached."""
        with self._lock:
            if self._count_running() >= self._max_concurrent:
                raise RuntimeError(
                    f"Max concurrent ingestion jobs ({self._max_concurrent}) reached. "
                    f"Wait for the current job to complete."
                )
            job_id = str(uuid.uuid4())[:8]
            job = IngestionJob(
                job_id=job_id,
                client_name=client_name,
                job_type=job_type,
            )
            self._jobs[job_id] = job
            return job

    def start_job(self, job_id: str) -> None:
        """Mark a job as running."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "running"

    def complete_job(self, job_id: str, result: str) -> None:
        """Mark a job as completed."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "completed"
                job.result = result

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark a job as failed."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "failed"
                job.error = error

    def update_progress(self, job_id: str, progress: IngestionProgress) -> None:
        """Update job progress."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.progress = progress

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def get_client_jobs(self, client_name: str) -> list[IngestionJob]:
        """Get all jobs for a client, newest first."""
        jobs = [j for j in self._jobs.values() if j.client_name == client_name]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        """Remove completed/failed jobs older than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        with self._lock:
            to_remove = [
                jid for jid, j in self._jobs.items()
                if j.status in ("completed", "failed") and j.created_at < cutoff
            ]
            for jid in to_remove:
                del self._jobs[jid]
            return len(to_remove)
