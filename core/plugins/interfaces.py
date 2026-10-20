"""
Abstract base classes for all Trove extension points.

Each interface defines the contract a plugin must implement. Built-in
implementations use these same interfaces — plugins are first-class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class BaseEmbedder(ABC):
    """Interface for embedding models.

    Converts text into dense vector representations for similarity search.
    """

    name: str = ""
    dimension: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Override this if the model uses a different prefix for queries
        vs passages (e.g. E5 models with "query: " prefix).

        Default implementation delegates to embed().
        """
        return self.embed([text])[0]

    def close(self) -> None:
        """Release model resources."""
        pass


class BaseChunker(ABC):
    """Interface for document chunking strategies.

    Splits parsed document elements into chunks suitable for embedding.
    """

    name: str = ""

    @abstractmethod
    def chunk(
        self,
        elements: list,
        max_chars: int = 1500,
        overlap: int = 100,
        **kwargs,
    ) -> list[str]:
        """Chunk document elements into text segments.

        Args:
            elements: Parsed document elements (from a parser).
            max_chars: Maximum characters per chunk.
            overlap: Character overlap between chunks.
            **kwargs: Strategy-specific parameters.

        Returns:
            List of chunk text strings.
        """
        ...


class BaseParser(ABC):
    """Interface for file parsers.

    Parses document files into a list of elements with metadata.
    """

    name: str = ""
    supported_extensions: list[str] = []

    @abstractmethod
    def parse(self, path: Path) -> list:
        """Parse a document file into elements.

        Args:
            path: Path to the document file.

        Returns:
            List of parsed elements. Each element should support str()
            for text content and have a .metadata attribute (or dict).
        """
        ...

    def can_parse(self, path: Path) -> bool:
        """Check if this parser supports the given file type.

        Default checks against supported_extensions.
        """
        if not self.supported_extensions:
            return True
        return path.suffix.lower() in self.supported_extensions


class BaseReranker(ABC):
    """Interface for reranker models.

    Scores query-document pairs for relevance after initial retrieval.
    """

    name: str = ""

    @abstractmethod
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score query-document pairs for relevance.

        Args:
            pairs: List of (query, document_text) tuples.

        Returns:
            List of relevance scores, one per pair.
        """
        ...

    def close(self) -> None:
        """Release model resources."""
        pass


class BaseLLMProvider(ABC):
    """Interface for LLM providers.

    Generates text responses from language models.
    """

    name: str = ""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> str:
        """Generate a response from the language model.

        Args:
            messages: Conversation messages, each with "role" and "content".
            system_prompt: Optional system instructions.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text response.
        """
        ...


class BaseMergeStrategy(ABC):
    """Interface for search result merge strategies.

    Combines results from BM25 keyword search and vector semantic search.
    """

    name: str = ""

    @abstractmethod
    def merge(
        self,
        bm25_results: list[tuple[dict, float]],
        vector_results: list[tuple[dict, float]],
        config: dict[str, Any],
    ) -> list[dict]:
        """Merge BM25 and vector search results.

        Args:
            bm25_results: List of (chunk_dict, score) from BM25.
            vector_results: List of (chunk_dict, score) from vector search.
            config: Search configuration dict with weights, normalization, etc.

        Returns:
            Merged list of chunk dicts sorted by combined score.
            Each chunk should have _bm25_score, _vector_score,
            and _combined_score fields.
        """
        ...
