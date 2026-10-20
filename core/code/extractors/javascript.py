"""
JavaScript code extractor using tree-sitter.

Extracts functions, classes, arrow functions, and exports.
"""

import logging
from typing import Optional

from .base import BaseExtractor, CodeChunk

logger = logging.getLogger(__name__)


class JavaScriptExtractor(BaseExtractor):
    """Extract code chunks from JavaScript source files."""

    language = "javascript"
    tree_sitter_language = "javascript"

    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract functions, classes, and exports from JavaScript code."""
        chunks = []
        source_bytes = bytes(source_code, "utf-8")

        try:
            tree = self.parse(source_code)
        except Exception as e:
            logger.warning(f"Failed to parse JavaScript file {file_path}: {e}")
            return chunks

        root = tree.root_node

        # Track what we've extracted to avoid duplicates
        extracted_ranges = set()

        # Extract classes
        for class_node in self.find_descendants_by_type(root, "class_declaration"):
            class_chunks = self._extract_class(class_node, source_bytes, file_path)
            chunks.extend(class_chunks)
            extracted_ranges.add((class_node.start_byte, class_node.end_byte))

        # Extract top-level function declarations
        for func_node in self.find_descendants_by_type(root, "function_declaration"):
            if not self._is_within_ranges(func_node, extracted_ranges):
                chunk = self._extract_function(func_node, source_bytes, file_path)
                if chunk:
                    chunks.append(chunk)
                    extracted_ranges.add((func_node.start_byte, func_node.end_byte))

        # Extract arrow functions assigned to variables
        # tsx grammar uses lexical_declaration instead of variable_declaration
        for node_type in ("variable_declaration", "lexical_declaration"):
            for var_decl in self.find_descendants_by_type(root, node_type):
                if not self._is_within_ranges(var_decl, extracted_ranges):
                    chunk = self._extract_arrow_function(var_decl, source_bytes, file_path)
                    if chunk:
                        chunks.append(chunk)
                        extracted_ranges.add((var_decl.start_byte, var_decl.end_byte))

        # Extract named exports
        for export_node in self.find_descendants_by_type(root, "export_statement"):
            if not self._is_within_ranges(export_node, extracted_ranges):
                chunk = self._extract_export(export_node, source_bytes, file_path, extracted_ranges)
                if chunk:
                    chunks.append(chunk)

        return chunks

    def _is_within_ranges(self, node, ranges: set) -> bool:
        """Check if a node is within any of the given ranges."""
        for start, end in ranges:
            if node.start_byte >= start and node.end_byte <= end:
                return True
        return False

    def _extract_function(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_class: str = None,
    ) -> Optional[CodeChunk]:
        """Extract a function declaration."""
        # Get function name
        name_node = self.find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)

        # Get parameters
        params_node = self.find_child_by_type(node, "formal_parameters")
        signature = self.get_node_text(params_node, source_bytes) if params_node else "()"

        # Get JSDoc comment
        docstring = self._get_jsdoc(node, source_bytes)

        # Get content
        content = self.get_node_text(node, source_bytes)

        symbol_type = "method" if parent_class else "function"

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type=symbol_type,
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            signature=signature,
            docstring=docstring,
            parent_class=parent_class,
            visibility=self.determine_visibility(name),
        )

    def _extract_arrow_function(
        self,
        var_decl,
        source_bytes: bytes,
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract an arrow function assigned to a variable."""
        # Find the variable declarator
        declarator = self.find_child_by_type(var_decl, "variable_declarator")
        if not declarator:
            return None

        # Get the name
        name_node = self.find_child_by_type(declarator, "identifier")
        if not name_node:
            return None

        # Check if value is an arrow function
        arrow_func = self.find_child_by_type(declarator, "arrow_function")
        if not arrow_func:
            return None

        name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(var_decl)

        # Get parameters
        params_node = self.find_child_by_type(arrow_func, "formal_parameters")
        if params_node:
            signature = self.get_node_text(params_node, source_bytes)
        else:
            # Single parameter without parens
            param = arrow_func.children[0] if arrow_func.children else None
            if param and param.type == "identifier":
                signature = f"({self.get_node_text(param, source_bytes)})"
            else:
                signature = "()"

        # Get JSDoc
        docstring = self._get_jsdoc(var_decl, source_bytes)

        # Get content
        content = self.get_node_text(var_decl, source_bytes)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="arrow_function",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            signature=signature,
            docstring=docstring,
            visibility=self.determine_visibility(name),
        )

    def _extract_class(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract a class and its methods."""
        chunks = []

        # Get class name
        name_node = self.find_child_by_type(node, "identifier")
        if not name_node:
            return chunks

        class_name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)

        # Get JSDoc
        docstring = self._get_jsdoc(node, source_bytes)

        # Get superclass
        extends_clause = self.find_child_by_type(node, "class_heritage")
        base_class = None
        if extends_clause:
            extends_name = self.find_descendant_identifier(extends_clause, source_bytes)
            if extends_name:
                base_class = extends_name

        # Get method names
        method_names = self._get_method_names(node, source_bytes)

        # Build class overview
        class_def = f"class {class_name}"
        if base_class:
            class_def += f" extends {base_class}"

        class_content = class_def
        if docstring:
            class_content = f"/**\n * {docstring}\n */\n{class_content}"
        if method_names:
            class_content += f"\n\n// Methods: {', '.join(method_names)}"

        class_chunk = CodeChunk(
            content=class_content,
            source_file=file_path,
            language=self.language,
            symbol_type="class",
            symbol_name=class_name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            visibility="public",
            extras={"base_class": base_class, "methods": method_names},
        )
        chunks.append(class_chunk)

        # Extract methods
        body = self.find_child_by_type(node, "class_body")
        if body:
            for method_node in self.find_children_by_type(body, "method_definition"):
                method_chunk = self._extract_method(method_node, source_bytes, file_path, class_name)
                if method_chunk:
                    chunks.append(method_chunk)

        return chunks

    def _extract_method(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_class: str,
    ) -> Optional[CodeChunk]:
        """Extract a class method."""
        # Get method name
        name_node = self.find_child_by_type(node, "property_identifier")
        if not name_node:
            return None

        name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)

        # Get parameters
        params_node = self.find_child_by_type(node, "formal_parameters")
        signature = self.get_node_text(params_node, source_bytes) if params_node else "()"

        # Get JSDoc
        docstring = self._get_jsdoc(node, source_bytes)

        # Get content
        content = self.get_node_text(node, source_bytes)

        # Check for static/async
        is_static = any(c.type == "static" for c in node.children)
        is_async = any(c.type == "async" for c in node.children)

        extras = {}
        if is_static:
            extras["static"] = True
        if is_async:
            extras["async"] = True

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="method",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            signature=signature,
            docstring=docstring,
            parent_class=parent_class,
            visibility=self.determine_visibility(name),
            extras=extras,
        )

    def _extract_export(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        extracted_ranges: set,
    ) -> Optional[CodeChunk]:
        """Extract named exports."""
        # Check for default export with function
        func_decl = self.find_child_by_type(node, "function_declaration")
        if func_decl and not self._is_within_ranges(func_decl, extracted_ranges):
            chunk = self._extract_function(func_decl, source_bytes, file_path)
            if chunk:
                chunk.extras["exported"] = True
                chunk.extras["default"] = "default" in self.get_node_text(node, source_bytes)
                return chunk

        # Check for export default class
        class_decl = self.find_child_by_type(node, "class_declaration")
        if class_decl:
            # Will be handled separately
            return None

        return None

    def _get_jsdoc(self, node, source_bytes: bytes) -> Optional[str]:
        """Get JSDoc comment preceding a node."""
        prev_sibling = node.prev_sibling
        while prev_sibling:
            if prev_sibling.type == "comment":
                comment = self.get_node_text(prev_sibling, source_bytes)
                if comment.startswith("/**"):
                    return self._clean_jsdoc(comment)
            elif prev_sibling.type not in ("decorator",):
                break
            prev_sibling = prev_sibling.prev_sibling
        return None

    def _clean_jsdoc(self, jsdoc: str) -> str:
        """Clean up a JSDoc comment."""
        lines = jsdoc.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if line.startswith("/**"):
                line = line[3:].strip()
            if line.endswith("*/"):
                line = line[:-2].strip()
            if line.startswith("*"):
                line = line[1:].strip()
            if line and not line.startswith("@"):
                cleaned.append(line)
        return "\n".join(cleaned)

    def _get_method_names(self, node, source_bytes: bytes) -> list[str]:
        """Get all method names in a class."""
        names = []
        body = self.find_child_by_type(node, "class_body")
        if body:
            for method_node in self.find_children_by_type(body, "method_definition"):
                name_node = self.find_child_by_type(method_node, "property_identifier")
                if name_node:
                    names.append(self.get_node_text(name_node, source_bytes))
        return names

    def find_descendant_identifier(self, node, source_bytes: bytes) -> Optional[str]:
        """Find the first identifier descendant."""
        for child in node.children:
            if child.type == "identifier":
                return self.get_node_text(child, source_bytes)
            result = self.find_descendant_identifier(child, source_bytes)
            if result:
                return result
        return None

    def determine_visibility(self, name: str, node=None) -> str:
        """Determine visibility from naming convention."""
        if name.startswith("_"):
            return "private"
        if name.startswith("#"):
            return "private"  # ES2022 private fields
        return "public"
