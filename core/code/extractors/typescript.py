"""
TypeScript code extractor using tree-sitter.

Extends JavaScript extractor with TypeScript-specific features:
interfaces, type aliases, enums, and type annotations.
"""

import logging
from typing import Optional

from .base import CodeChunk
from .javascript import JavaScriptExtractor

logger = logging.getLogger(__name__)


class TypeScriptExtractor(JavaScriptExtractor):
    """Extract code chunks from TypeScript source files."""

    language = "typescript"
    tree_sitter_language = "typescript"

    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract functions, classes, interfaces, and types from TypeScript."""
        # Use tsx grammar for .tsx files (typescript grammar hangs on JSX)
        original_lang = self.tree_sitter_language
        if file_path.endswith('.tsx'):
            self.tree_sitter_language = "tsx"
            self._parser = None  # Force parser reload

        try:
            return self._extract_impl(source_code, file_path)
        finally:
            if self.tree_sitter_language != original_lang:
                self.tree_sitter_language = original_lang
                self._parser = None

    def _extract_impl(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Implementation of TypeScript extraction."""
        # Get base JavaScript chunks
        chunks = super().extract(source_code, file_path)

        # Fix language for all chunks
        for chunk in chunks:
            chunk.language = self.language

        source_bytes = bytes(source_code, "utf-8")

        try:
            tree = self.parse(source_code)
        except Exception as e:
            logger.warning(f"Failed to parse TypeScript file {file_path}: {e}")
            return chunks

        root = tree.root_node

        # Extract interfaces
        for interface_node in self.find_descendants_by_type(root, "interface_declaration"):
            chunk = self._extract_interface(interface_node, source_bytes, file_path)
            if chunk:
                chunks.append(chunk)

        # Extract type aliases
        for type_node in self.find_descendants_by_type(root, "type_alias_declaration"):
            chunk = self._extract_type_alias(type_node, source_bytes, file_path)
            if chunk:
                chunks.append(chunk)

        # Extract enums
        for enum_node in self.find_descendants_by_type(root, "enum_declaration"):
            chunk = self._extract_enum(enum_node, source_bytes, file_path)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _extract_interface(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract a TypeScript interface."""
        name_node = self.find_child_by_type(node, "type_identifier")
        if not name_node:
            return None

        name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)

        # Get JSDoc
        docstring = self._get_jsdoc(node, source_bytes)

        # Get extends
        extends = self._get_interface_extends(node, source_bytes)

        # Get property names
        properties = self._get_interface_properties(node, source_bytes)

        content = self.get_node_text(node, source_bytes)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="interface",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            visibility="public",
            extras={"extends": extends, "properties": properties},
        )

    def _extract_type_alias(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract a TypeScript type alias."""
        name_node = self.find_child_by_type(node, "type_identifier")
        if not name_node:
            return None

        name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)

        # Get JSDoc
        docstring = self._get_jsdoc(node, source_bytes)

        content = self.get_node_text(node, source_bytes)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="type_alias",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            visibility="public",
        )

    def _extract_enum(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract a TypeScript enum."""
        name_node = self.find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)

        # Get JSDoc
        docstring = self._get_jsdoc(node, source_bytes)

        # Get enum members
        members = self._get_enum_members(node, source_bytes)

        content = self.get_node_text(node, source_bytes)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="enum",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            visibility="public",
            extras={"members": members},
        )

    def _get_interface_extends(self, node, source_bytes: bytes) -> list[str]:
        """Get interfaces that this interface extends."""
        extends = []
        extends_clause = self.find_child_by_type(node, "extends_type_clause")
        if extends_clause:
            for type_id in self.find_descendants_by_type(extends_clause, "type_identifier"):
                extends.append(self.get_node_text(type_id, source_bytes))
        return extends

    def _get_interface_properties(self, node, source_bytes: bytes) -> list[str]:
        """Get property names from an interface."""
        properties = []
        body = self.find_child_by_type(node, "object_type") or self.find_child_by_type(node, "interface_body")
        if body:
            for prop in self.find_descendants_by_type(body, "property_signature"):
                name_node = self.find_child_by_type(prop, "property_identifier")
                if name_node:
                    properties.append(self.get_node_text(name_node, source_bytes))
        return properties

    def _get_enum_members(self, node, source_bytes: bytes) -> list[str]:
        """Get member names from an enum."""
        members = []
        body = self.find_child_by_type(node, "enum_body")
        if body:
            for member in self.find_children_by_type(body, "enum_assignment"):
                name_node = self.find_child_by_type(member, "property_identifier")
                if name_node:
                    members.append(self.get_node_text(name_node, source_bytes))
            # Also check for simple identifiers
            for member in self.find_children_by_type(body, "property_identifier"):
                name = self.get_node_text(member, source_bytes)
                if name not in members:
                    members.append(name)
        return members
