"""
Adaptive resource management for Trove ingestion.

Probes CPU, RAM, and GPU to recommend concurrency settings.
Leaves configurable headroom (default 20%) for other processes.
"""

import logging
import os
import warnings
from typing import Optional

import psutil

logger = logging.getLogger(__name__)


class ResourceManager:
    """Probe system resources and recommend ingestion concurrency settings."""

    def __init__(self, headroom: float = 0.05):
        """
        Args:
            headroom: Fraction of resources to leave free (0.0 to 0.5). Default 0.05.
        """
        self.headroom = max(0.0, min(0.5, headroom))
        self._gpu_info: Optional[dict] = None
        self._gpu_probed = False

    def probe_cpu(self) -> dict:
        """Return CPU core counts with headroom applied."""
        total = os.cpu_count() or 4
        available = max(2, int(total * (1.0 - self.headroom)))
        return {"total_cores": total, "available_cores": available}

    def probe_ram(self) -> dict:
        """Return RAM in GB with headroom applied."""
        mem = psutil.virtual_memory()
        total_gb = round(mem.total / (1024 ** 3), 1)
        available_gb = round(mem.available / (1024 ** 3), 1)
        usable_gb = round(available_gb * (1.0 - self.headroom), 1)
        return {"total_gb": total_gb, "available_gb": available_gb, "usable_gb": usable_gb}

    def probe_gpu(self) -> Optional[dict]:
        """Return GPU VRAM info in MB, or None if no GPU / pynvml unavailable."""
        if self._gpu_probed:
            return self._gpu_info
        self._gpu_probed = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                from pynvml import (
                    nvmlInit,
                    nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetMemoryInfo,
                    nvmlDeviceGetUtilizationRates,
                )
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            util = nvmlDeviceGetUtilizationRates(handle)
            self._gpu_info = {
                "total_mb": info.total // (1024 ** 2),
                "used_mb": info.used // (1024 ** 2),
                "free_mb": info.free // (1024 ** 2),
                "utilization_pct": util.gpu,
            }
            return self._gpu_info
        except Exception as e:
            logger.debug(f"GPU probe failed (no GPU or pynvml not installed): {e}")
            self._gpu_info = None
            return None

    def recommend_parse_workers(self, file_count: int) -> int:
        """Recommend number of parse workers based on file count and available CPU."""
        cpu = self.probe_cpu()
        cores = cpu["available_cores"]

        if file_count < 500:
            return min(4, cores)
        elif file_count < 10000:
            return min(int(cores * 0.6), 16)
        else:
            return cores

    def recommend_embed_batch(self, model_dim: int) -> int:
        """Recommend embedding batch size based on available VRAM and model dimension."""
        gpu = self.probe_gpu()
        if gpu is None:
            return 64  # CPU fallback — conservative

        free_mb = gpu["free_mb"]
        usable_mb = free_mb * (1.0 - self.headroom)

        # Estimate: ~0.5MB per 512 chunks for 384d, ~1MB for 768d
        mb_per_batch_of_512 = (model_dim / 384) * 0.5
        if mb_per_batch_of_512 <= 0:
            return 256

        batches_that_fit = usable_mb / mb_per_batch_of_512
        recommended = int(batches_that_fit * 512)

        return max(32, min(recommended, 1024))

    def classify_client(self, file_count: int, chunk_count: int) -> str:
        """Classify client as small/medium/large based on content size."""
        if chunk_count < 5000:
            return "small"
        elif chunk_count < 50000:
            return "medium"
        else:
            return "large"

    def summary(self) -> str:
        """Return a human-readable summary of system resources."""
        cpu = self.probe_cpu()
        ram = self.probe_ram()
        gpu = self.probe_gpu()
        lines = [
            f"CPU: {cpu['available_cores']}/{cpu['total_cores']} cores (headroom={self.headroom:.0%})",
            f"RAM: {ram['usable_gb']:.1f}GB usable / {ram['total_gb']:.1f}GB total",
        ]
        if gpu:
            lines.append(
                f"GPU: {gpu['free_mb']}MB free / {gpu['total_mb']}MB total, "
                f"{gpu['utilization_pct']}% utilized"
            )
        else:
            lines.append("GPU: not available")
        return " | ".join(lines)
