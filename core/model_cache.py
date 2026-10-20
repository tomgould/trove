"""
Model cache — singleton that keeps embedding models and rerankers in memory.

Avoids redundant GPU loads when multiple clients use the same model.
"""

import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_instance: Optional["ModelCache"] = None


def get_model_cache() -> "ModelCache":
    """Get the singleton ModelCache instance."""
    global _instance
    if _instance is None:
        _instance = ModelCache()
    return _instance


class ModelCache:
    """Cache for embedding models and cross-encoder rerankers."""

    def __init__(self):
        self._embedders: dict[str, object] = {}  # key: "model_name:device"
        self._rerankers: dict[str, object] = {}

    def get_embedder(self, model_name: str, device: str = "cuda"):
        """Get or load a SentenceTransformer model."""
        key = f"{model_name}:{device}"
        if key not in self._embedders:
            from sentence_transformers import SentenceTransformer
            logger.info(f"ModelCache: loading embedder {model_name} on {device}")
            self._embedders[key] = SentenceTransformer(model_name, device=device)
        return self._embedders[key]

    def get_reranker(self, model_name: str, device: str = "cuda"):
        """Get or load a CrossEncoder reranker model."""
        key = f"{model_name}:{device}"
        if key not in self._rerankers:
            from sentence_transformers import CrossEncoder
            logger.info(f"ModelCache: loading reranker {model_name} on {device}")
            self._rerankers[key] = CrossEncoder(model_name, device=device)
        return self._rerankers[key]

    def clear(self):
        """Release all cached models and free GPU memory."""
        count = len(self._embedders) + len(self._rerankers)
        self._embedders.clear()
        self._rerankers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if count:
            logger.info(f"ModelCache: cleared {count} cached models")
