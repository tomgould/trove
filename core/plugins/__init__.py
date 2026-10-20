"""
Plugin system for Trove.

Provides abstract base classes, a plugin registry, and a loader
for discovering and loading custom extensions.
"""

from core.plugins.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseLLMProvider,
    BaseMergeStrategy,
    BaseParser,
    BaseReranker,
)
from core.plugins.registry import PluginRegistry

__all__ = [
    "BaseChunker",
    "BaseEmbedder",
    "BaseLLMProvider",
    "BaseMergeStrategy",
    "BaseParser",
    "BaseReranker",
    "PluginRegistry",
]
