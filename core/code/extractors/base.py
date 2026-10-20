"""
Base extractor class for code parsing.

All language-specific extractors inherit from this class.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a chunk of code extracted from a source file."""

    content: str
    source_file: str
    language: str
    symbol_type: str  # function, method, class, module, etc.
    symbol_name: str
    line_start: int
    line_end: int

    # Optional fields
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
    parent_module: Optional[str] = None
    decorators: list[str] = field(default_factory=list)
    visibility: str = "public"  # public, private, protected

    # Repo info (populated later)
    repo_name: Optional[str] = None
    repo_remote: Optional[str] = None
    repo_branch: Optional[str] = None
    repo_relative_path: Optional[str] = None

    # Language-specific extras
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for indexing."""
        metadata = {
            "chunk_type": "code",
            "source_file": self.source_file,
            "language": self.language,
            "symbol_type": self.symbol_type,
            "symbol_name": self.symbol_name,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "visibility": self.visibility,
        }

        # Add optional fields if present
        if self.signature:
            metadata["signature"] = self.signature
        if self.docstring:
            metadata["docstring"] = self.docstring
        if self.parent_class:
            metadata["parent_class"] = self.parent_class
        if self.parent_module:
            metadata["parent_module"] = self.parent_module
        if self.decorators:
            metadata["decorators"] = self.decorators

        # Add repo info if present
        if self.repo_name:
            metadata["repo_name"] = self.repo_name
        if self.repo_remote:
            metadata["repo_remote"] = self.repo_remote
        if self.repo_branch:
            metadata["repo_branch"] = self.repo_branch
        if self.repo_relative_path:
            metadata["repo_relative_path"] = self.repo_relative_path

        # Add language-specific extras
        metadata.update(self.extras)

        return {
            "content": self.content,
            "metadata": metadata,
        }


class BaseExtractor(ABC):
    """Base class for language-specific code extractors."""

    # Override in subclasses
    language: str = "unknown"
    tree_sitter_language: str = "unknown"

    def __init__(self):
        """Initialize the extractor with tree-sitter parser."""
        self._parser = None
        self._language = None

    @property
    def parser(self):
        """Lazy-load the tree-sitter parser."""
        if self._parser is None:
            import tree_sitter_languages
            self._parser = tree_sitter_languages.get_parser(self.tree_sitter_language)
        return self._parser

    def parse(self, source_code: str) -> "tree_sitter.Tree":  # noqa: F821 (optional dependency, never evaluated)
        """Parse source code into an AST."""
        return self.parser.parse(bytes(source_code, "utf-8"))

    @abstractmethod
    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """
        Extract code chunks from source code.

        Args:
            source_code: The source code content
            file_path: Path to the source file (for metadata)

        Returns:
            List of CodeChunk objects
        """
        pass

    def get_node_text(self, node, source_bytes: bytes) -> str:
        """Get the text content of an AST node."""
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def get_node_line_range(self, node) -> tuple[int, int]:
        """Get the line range (1-indexed) of an AST node."""
        return (node.start_point[0] + 1, node.end_point[0] + 1)

    def find_children_by_type(self, node, type_name: str) -> list:
        """Find all direct children of a specific type."""
        return [child for child in node.children if child.type == type_name]

    def find_child_by_type(self, node, type_name: str):
        """Find the first direct child of a specific type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def find_descendants_by_type(self, node, type_name: str) -> list:
        """Find all descendants of a specific type (iterative)."""
        results = []
        stack = [node]

        while stack:
            n = stack.pop()
            if n.type == type_name:
                results.append(n)
            # Add children in reverse so leftmost is processed first
            stack.extend(reversed(n.children))

        return results

    def extract_docstring(self, node, source_bytes: bytes) -> Optional[str]:
        """
        Extract docstring from a node.

        Override in language-specific extractors for different docstring formats.
        """
        return None

    def determine_visibility(self, name: str, node=None) -> str:
        """
        Determine visibility (public/private/protected) from naming conventions.

        Override in language-specific extractors for language-specific rules.
        """
        if name.startswith("__") and not name.endswith("__"):
            return "private"
        elif name.startswith("_"):
            return "protected"
        return "public"
