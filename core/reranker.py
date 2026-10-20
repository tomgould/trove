"""
Configurable reranker backends for cross-encoder re-ranking.

Supports:
- Local CrossEncoder (default, same as before)
- API-based reranker (Cohere/vLLM-compatible POST /v1/rerank)
- Fallback reranker (API with auto-fallback to local on failure)
"""

import logging
from typing import Optional, Protocol, runtime_checkable

import requests

from core.plugins.interfaces import BaseReranker

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranker backends (kept for backwards compatibility)."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score query-document pairs. Returns list of relevance scores."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class LocalReranker(BaseReranker):
    """Local CrossEncoder reranker (sentence-transformers)."""

    name = "local"

    def __init__(self, model: str, device: str = None):
        import torch
        from sentence_transformers import CrossEncoder

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._model = CrossEncoder(model, device=device)
        except RuntimeError as e:
            if device == "cuda":
                logger.warning(f"GPU unavailable for reranker ({e}), using CPU")
                self._model = CrossEncoder(model, device="cpu")
            else:
                raise

        self._device = device
        logger.info(f"LocalReranker initialized: {model} on {device}")

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None


class APIReranker(BaseReranker):
    """API-based reranker (Cohere/vLLM-compatible POST /v1/rerank)."""

    name = "api"

    def __init__(
        self,
        endpoint: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        logger.info(f"APIReranker initialized: {self.endpoint}")

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        query = pairs[0][0]
        documents = [doc for _, doc in pairs]

        payload = {
            "query": query,
            "documents": documents,
        }
        if self.model:
            payload["model"] = self.model

        response = requests.post(
            f"{self.endpoint}/v1/rerank",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        # Cohere format: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        results = data.get("results", [])
        # Sort by index to match input order
        results.sort(key=lambda r: r.get("index", 0))
        return [r.get("relevance_score", 0.0) for r in results]

    def close(self) -> None:
        pass


class FallbackReranker(BaseReranker):
    """API reranker with automatic fallback to local on failure."""

    name = "fallback"

    def __init__(
        self,
        api_reranker: APIReranker,
        local_model: str,
        device: str = None,
    ):
        self._api = api_reranker
        self._local: Optional[LocalReranker] = None
        self._local_model = local_model
        self._device = device
        self._api_failed = False
        logger.info("FallbackReranker initialized (API primary, local fallback)")

    def _get_local(self) -> LocalReranker:
        if self._local is None:
            logger.info("Initializing local fallback reranker...")
            self._local = LocalReranker(self._local_model, self._device)
        return self._local

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._api_failed:
            return self._get_local().predict(pairs)

        try:
            return self._api.predict(pairs)
        except Exception as e:
            logger.warning(f"API reranker failed ({e}), falling back to local")
            self._api_failed = True
            return self._get_local().predict(pairs)

    def close(self) -> None:
        self._api.close()
        if self._local is not None:
            self._local.close()


def create_reranker(
    backend: str = "local",
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    device: str = None,
) -> Reranker:
    """
    Factory function to create a reranker.

    Args:
        backend: "local" or "api"
        model: Model name (for local) or API model name
        endpoint: API endpoint URL (required for "api" backend)
        api_key: API key for authentication
        device: Device for local model ("cuda", "cpu", or None for auto)

    Returns:
        Reranker instance
    """
    if backend == "api":
        if not endpoint:
            raise ValueError("endpoint is required for API reranker backend")
        api = APIReranker(endpoint=endpoint, model=model, api_key=api_key)
        # Wrap in FallbackReranker for resilience
        return FallbackReranker(api, local_model=model, device=device)
    else:
        return LocalReranker(model=model, device=device)
