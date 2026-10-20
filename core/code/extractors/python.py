"""
Python code extractor using tree-sitter.

Extracts functions, methods, classes, and module-level constants.
"""

import logging
from typing import Optional

from .base import BaseExtractor, CodeChunk

logger = logging.getLogger(__name__)


class PythonExtractor(BaseExtractor):
    """Extract code chunks from Python source files."""

    language = "python"
    tree_sitter_language = "python"

    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract functions, classes, and methods from Python code."""
        chunks = []
        source_bytes = bytes(source_code, "utf-8")

        try:
            tree = self.parse(source_code)
        except Exception as e:
            logger.warning(f"Failed to parse Python file {file_path}: {e}")
            return chunks

        root = tree.root_node

        # Extract module docstring
        module_docstring = self._get_module_docstring(root, source_bytes)

        # Extract top-level functions
        for node in self.find_children_by_type(root, "function_definition"):
            chunk = self._extract_function(node, source_bytes, file_path)
            if chunk:
                chunks.append(chunk)

        # Extract classes and their methods
        for class_node in self.find_children_by_type(root, "class_definition"):
            class_chunks = self._extract_class(class_node, source_bytes, file_path)
            chunks.extend(class_chunks)

        # Extract module-level constants (assignments at top level)
        for node in self.find_children_by_type(root, "expression_statement"):
            chunk = self._extract_constant(node, source_bytes, file_path)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _get_module_docstring(self, root, source_bytes: bytes) -> Optional[str]:
        """Extract module-level docstring if present."""
        if root.children:
            first_stmt = root.children[0]
            if first_stmt.type == "expression_statement":
                expr = self.find_child_by_type(first_stmt, "string")
                if expr:
                    docstring = self.get_node_text(expr, source_bytes)
                    return self._clean_docstring(docstring)
        return None

    def _extract_function(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_class: str = None,
    ) -> Optional[CodeChunk]:
        """Extract a function or method definition."""
        # Get function name
        name_node = self.find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)

        # Get decorators
        decorators = self._get_decorators(node, source_bytes)

        # Get parameters (signature)
        params_node = self.find_child_by_type(node, "parameters")
        signature = self.get_node_text(params_node, source_bytes) if params_node else "()"

        # Get return type annotation if present
        return_type = self._get_return_type(node, source_bytes)
        if return_type:
            signature = f"{signature} -> {return_type}"

        # Get docstring
        docstring = self._get_function_docstring(node, source_bytes)

        # Get full content
        content = self.get_node_text(node, source_bytes)

        # Determine symbol type
        symbol_type = "method" if parent_class else "function"

        # Determine visibility
        visibility = self.determine_visibility(name)

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
            decorators=decorators,
            visibility=visibility,
        )

    def _extract_class(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract a class and all its methods."""
        chunks = []

        # Get class name
        name_node = self.find_child_by_type(node, "identifier")
        if not name_node:
            return chunks

        class_name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)

        # Get decorators
        decorators = self._get_decorators(node, source_bytes)

        # Get base classes
        bases = self._get_base_classes(node, source_bytes)

        # Get class docstring
        docstring = self._get_class_docstring(node, source_bytes)

        # Create class overview chunk
        # Include class definition line and docstring, not full body
        class_def_line = self._get_class_definition_line(node, source_bytes)
        method_names = self._get_method_names(node, source_bytes)

        class_content = class_def_line
        if docstring:
            class_content += f'\n    """{docstring}"""'
        if method_names:
            class_content += f"\n\n    # Methods: {', '.join(method_names)}"

        class_chunk = CodeChunk(
            content=class_content,
            source_file=file_path,
            language=self.language,
            symbol_type="class",
            symbol_name=class_name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            decorators=decorators,
            visibility=self.determine_visibility(class_name),
            extras={"base_classes": bases, "methods": method_names},
        )
        chunks.append(class_chunk)

        # Extract methods
        body = self.find_child_by_type(node, "block")
        if body:
            for method_node in self.find_children_by_type(body, "function_definition"):
                method_chunk = self._extract_function(
                    method_node, source_bytes, file_path, parent_class=class_name
                )
                if method_chunk:
                    chunks.append(method_chunk)

        return chunks

    def _extract_constant(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract module-level constant assignments."""
        # Look for assignment expressions
        assign = self.find_child_by_type(node, "assignment")
        if not assign:
            return None

        # Get the left side (name)
        left = assign.children[0] if assign.children else None
        if not left or left.type != "identifier":
            return None

        name = self.get_node_text(left, source_bytes)

        # Only extract UPPER_CASE constants
        if not name.isupper() and not name.startswith("_"):
            return None

        line_start, line_end = self.get_node_line_range(node)
        content = self.get_node_text(node, source_bytes)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="constant",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            visibility="public" if not name.startswith("_") else "private",
        )

    def _get_decorators(self, node, source_bytes: bytes) -> list[str]:
        """Get decorators for a function or class."""
        decorators = []

        # Look for decorator nodes that precede the function/class
        parent = node.parent
        if parent:
            idx = None
            for i, child in enumerate(parent.children):
                if child == node:
                    idx = i
                    break

            if idx is not None:
                # Check preceding nodes for decorators
                for i in range(idx - 1, -1, -1):
                    prev = parent.children[i]
                    if prev.type == "decorator":
                        dec_text = self.get_node_text(prev, source_bytes)
                        decorators.insert(0, dec_text)
                    elif prev.type not in ("comment",):
                        break

        return decorators

    def _get_return_type(self, node, source_bytes: bytes) -> Optional[str]:
        """Get return type annotation from function definition."""
        # Look for type annotation after parameters
        for child in node.children:
            if child.type == "type":
                return self.get_node_text(child, source_bytes)
        return None

    def _get_function_docstring(self, node, source_bytes: bytes) -> Optional[str]:
        """Get docstring from a function definition."""
        body = self.find_child_by_type(node, "block")
        if body and body.children:
            first_stmt = body.children[0]
            if first_stmt.type == "expression_statement":
                string_node = self.find_child_by_type(first_stmt, "string")
                if string_node:
                    docstring = self.get_node_text(string_node, source_bytes)
                    return self._clean_docstring(docstring)
        return None

    def _get_class_docstring(self, node, source_bytes: bytes) -> Optional[str]:
        """Get docstring from a class definition."""
        body = self.find_child_by_type(node, "block")
        if body and body.children:
            first_stmt = body.children[0]
            if first_stmt.type == "expression_statement":
                string_node = self.find_child_by_type(first_stmt, "string")
                if string_node:
                    docstring = self.get_node_text(string_node, source_bytes)
                    return self._clean_docstring(docstring)
        return None

    def _get_base_classes(self, node, source_bytes: bytes) -> list[str]:
        """Get base classes from class definition."""
        bases = []
        arg_list = self.find_child_by_type(node, "argument_list")
        if arg_list:
            for child in arg_list.children:
                if child.type == "identifier":
                    bases.append(self.get_node_text(child, source_bytes))
                elif child.type == "attribute":
                    bases.append(self.get_node_text(child, source_bytes))
        return bases

    def _get_class_definition_line(self, node, source_bytes: bytes) -> str:
        """Get just the class definition line (not the body)."""
        # Find the colon that ends the class definition line
        for child in node.children:
            if child.type == ":":
                end_byte = child.end_byte
                return source_bytes[node.start_byte:end_byte + 1].decode("utf-8")

        # Fallback: just get class name and bases
        name_node = self.find_child_by_type(node, "identifier")
        name = self.get_node_text(name_node, source_bytes) if name_node else "Unknown"
        return f"class {name}:"

    def _get_method_names(self, node, source_bytes: bytes) -> list[str]:
        """Get all method names in a class."""
        names = []
        body = self.find_child_by_type(node, "block")
        if body:
            for method_node in self.find_children_by_type(body, "function_definition"):
                name_node = self.find_child_by_type(method_node, "identifier")
                if name_node:
                    names.append(self.get_node_text(name_node, source_bytes))
        return names

    def _clean_docstring(self, docstring: str) -> str:
        """Clean up a docstring (remove quotes, normalize whitespace)."""
        # Remove surrounding quotes
        if docstring.startswith('"""') or docstring.startswith("'''"):
            docstring = docstring[3:-3]
        elif docstring.startswith('"') or docstring.startswith("'"):
            docstring = docstring[1:-1]

        # Normalize whitespace
        lines = docstring.strip().split("\n")
        if len(lines) > 1:
            # Find minimum indentation (excluding first line)
            min_indent = float("inf")
            for line in lines[1:]:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)

            if min_indent < float("inf"):
                lines = [lines[0]] + [
                    line[min_indent:] if len(line) > min_indent else line
                    for line in lines[1:]
                ]

        return "\n".join(lines).strip()
