"""
Plugin registry for Trove.

Stores registered plugins by extension point and name. Plugins register
via register_* methods; core code looks up plugins via get_*.
"""

from __future__ import annotations

import logging
from typing import Optional, Type

from core.plugins.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseLLMProvider,
    BaseMergeStrategy,
    BaseParser,
    BaseReranker,
)

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Central registry for all Trove plugins.

    Plugins are registered by type and name. Built-in implementations
    are registered at startup alongside any user plugins.
    """

    def __init__(self):
        self._embedders: dict[str, Type[BaseEmbedder]] = {}
        self._chunkers: dict[str, Type[BaseChunker]] = {}
        self._parsers: dict[str, Type[BaseParser]] = {}
        self._rerankers: dict[str, Type[BaseReranker]] = {}
        self._llm_providers: dict[str, Type[BaseLLMProvider]] = {}
        self._merge_strategies: dict[str, Type[BaseMergeStrategy]] = {}

    # --- Registration ---

    def register_embedder(self, cls: Type[BaseEmbedder]) -> None:
        """Register an embedder plugin class."""
        self._register("embedder", self._embedders, cls)

    def register_chunker(self, cls: Type[BaseChunker]) -> None:
        """Register a chunker plugin class."""
        self._register("chunker", self._chunkers, cls)

    def register_parser(self, cls: Type[BaseParser]) -> None:
        """Register a parser plugin class."""
        self._register("parser", self._parsers, cls)

    def register_reranker(self, cls: Type[BaseReranker]) -> None:
        """Register a reranker plugin class."""
        self._register("reranker", self._rerankers, cls)

    def register_llm_provider(self, cls: Type[BaseLLMProvider]) -> None:
        """Register an LLM provider plugin class."""
        self._register("llm_provider", self._llm_providers, cls)

    def register_merge_strategy(self, cls: Type[BaseMergeStrategy]) -> None:
        """Register a merge strategy plugin class."""
        self._register("merge_strategy", self._merge_strategies, cls)

    # --- Lookup ---

    def get_embedder(self, name: str) -> Optional[Type[BaseEmbedder]]:
        """Get an embedder class by name."""
        return self._embedders.get(name)

    def get_chunker(self, name: str) -> Optional[Type[BaseChunker]]:
        """Get a chunker class by name."""
        return self._chunkers.get(name)

    def get_parser(self, name: str) -> Optional[Type[BaseParser]]:
        """Get a parser class by name."""
        return self._parsers.get(name)

    def get_reranker(self, name: str) -> Optional[Type[BaseReranker]]:
        """Get a reranker class by name."""
        return self._rerankers.get(name)

    def get_llm_provider(self, name: str) -> Optional[Type[BaseLLMProvider]]:
        """Get an LLM provider class by name."""
        return self._llm_providers.get(name)

    def get_merge_strategy(self, name: str) -> Optional[Type[BaseMergeStrategy]]:
        """Get a merge strategy class by name."""
        return self._merge_strategies.get(name)

    # --- Listing ---

    def list_embedders(self) -> list[str]:
        """List registered embedder names."""
        return sorted(self._embedders.keys())

    def list_chunkers(self) -> list[str]:
        """List registered chunker names."""
        return sorted(self._chunkers.keys())

    def list_parsers(self) -> list[str]:
        """List registered parser names."""
        return sorted(self._parsers.keys())

    def list_rerankers(self) -> list[str]:
        """List registered reranker names."""
        return sorted(self._rerankers.keys())

    def list_llm_providers(self) -> list[str]:
        """List registered LLM provider names."""
        return sorted(self._llm_providers.keys())

    def list_merge_strategies(self) -> list[str]:
        """List registered merge strategy names."""
        return sorted(self._merge_strategies.keys())

    def list_all(self) -> dict[str, list[str]]:
        """List all registered plugins by type."""
        return {
            "embedders": self.list_embedders(),
            "chunkers": self.list_chunkers(),
            "parsers": self.list_parsers(),
            "rerankers": self.list_rerankers(),
            "llm_providers": self.list_llm_providers(),
            "merge_strategies": self.list_merge_strategies(),
        }

    # --- Internal ---

    def _register(
        self,
        type_name: str,
        store: dict[str, Type],
        cls: Type,
    ) -> None:
        """Register a plugin class, checking for duplicates."""
        name = getattr(cls, "name", None)
        if not name:
            raise ValueError(
                f"Plugin class {cls.__name__} must define a 'name' attribute"
            )
        if name in store:
            existing = store[name]
            logger.warning(
                "Overriding %s plugin '%s': %s -> %s",
                type_name,
                name,
                existing.__name__,
                cls.__name__,
            )
        store[name] = cls
        logger.debug("Registered %s plugin: %s (%s)", type_name, name, cls.__name__)
