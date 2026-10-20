"""
PHP code extractor using tree-sitter.

Extracts functions, methods, classes, interfaces, and traits.
Includes Drupal hook detection for .module, .install files.
"""

import logging
import re
from typing import Optional

from .base import BaseExtractor, CodeChunk

logger = logging.getLogger(__name__)

# Drupal hook patterns
DRUPAL_HOOK_PATTERN = re.compile(r"^(\w+)_(hook_\w+|preprocess_\w+|theme_\w+|form_\w+_alter|.+_alter)$")
DRUPAL_HOOK_COMMENT = re.compile(r"Implements\s+hook_(\w+)\(\)")


class PHPExtractor(BaseExtractor):
    """Extract code chunks from PHP source files."""

    language = "php"
    tree_sitter_language = "php"

    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract functions, classes, and methods from PHP code."""
        chunks = []
        source_bytes = bytes(source_code, "utf-8")

        # Determine if this is a Drupal file
        is_drupal = any(file_path.endswith(ext) for ext in [".module", ".install", ".theme", ".inc", ".profile"])

        try:
            tree = self.parse(source_code)
        except Exception as e:
            logger.warning(f"Failed to parse PHP file {file_path}: {e}")
            return chunks

        root = tree.root_node

        # Find the program node (PHP files have a wrapper)
        program = root
        if root.type == "program":
            program = root

        # Extract top-level functions
        for node in self.find_descendants_by_type(program, "function_definition"):
            # Skip methods (they'll be extracted with their class)
            if self._is_method(node):
                continue
            chunk = self._extract_function(node, source_bytes, file_path, is_drupal=is_drupal)
            if chunk:
                chunks.append(chunk)

        # Extract classes
        for class_node in self.find_descendants_by_type(program, "class_declaration"):
            class_chunks = self._extract_class(class_node, source_bytes, file_path)
            chunks.extend(class_chunks)

        # Extract interfaces
        for interface_node in self.find_descendants_by_type(program, "interface_declaration"):
            chunk = self._extract_interface(interface_node, source_bytes, file_path)
            if chunk:
                chunks.append(chunk)

        # Extract traits
        for trait_node in self.find_descendants_by_type(program, "trait_declaration"):
            trait_chunks = self._extract_trait(trait_node, source_bytes, file_path)
            chunks.extend(trait_chunks)

        return chunks

    def _is_method(self, node) -> bool:
        """Check if a function definition is a class method."""
        parent = node.parent
        while parent:
            if parent.type in ("class_declaration", "trait_declaration", "interface_declaration"):
                return True
            if parent.type == "declaration_list":
                return True
            parent = parent.parent
        return False

    def _extract_function(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_class: str = None,
        is_drupal: bool = False,
    ) -> Optional[CodeChunk]:
        """Extract a function or method definition."""
        # Get function name
        name_node = self.find_child_by_type(node, "name")
        if not name_node:
            return None

        name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)

        # Get visibility modifier
        visibility = self._get_visibility(node, source_bytes)

        # Get parameters
        params_node = self.find_child_by_type(node, "formal_parameters")
        signature = self.get_node_text(params_node, source_bytes) if params_node else "()"

        # Get return type
        return_type = self._get_return_type(node, source_bytes)
        if return_type:
            signature = f"{signature}: {return_type}"

        # Get docblock
        docstring = self._get_docblock(node, source_bytes)

        # Detect Drupal hook
        symbol_type = "method" if parent_class else "function"
        extras = {}

        if is_drupal and not parent_class:
            hook_type = self._detect_drupal_hook(name, docstring)
            if hook_type:
                symbol_type = "hook"
                extras["hook_type"] = hook_type

        # Get full content
        content = self.get_node_text(node, source_bytes)

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
            visibility=visibility,
            extras=extras,
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
        name_node = self.find_child_by_type(node, "name")
        if not name_node:
            return chunks

        class_name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)

        # Get docblock
        docstring = self._get_docblock(node, source_bytes)

        # Get base class and interfaces
        base_class = self._get_base_class(node, source_bytes)
        interfaces = self._get_interfaces(node, source_bytes)

        # Get method names
        method_names = self._get_method_names(node, source_bytes)

        # Build class overview
        class_def = f"class {class_name}"
        if base_class:
            class_def += f" extends {base_class}"
        if interfaces:
            class_def += f" implements {', '.join(interfaces)}"

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
            extras={
                "base_class": base_class,
                "interfaces": interfaces,
                "methods": method_names,
            },
        )
        chunks.append(class_chunk)

        # Extract methods
        decl_list = self.find_child_by_type(node, "declaration_list")
        if decl_list:
            for method_node in self.find_children_by_type(decl_list, "method_declaration"):
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
        name_node = self.find_child_by_type(node, "name")
        if not name_node:
            return None

        name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)

        # Get visibility
        visibility = self._get_visibility(node, source_bytes)

        # Get parameters
        params_node = self.find_child_by_type(node, "formal_parameters")
        signature = self.get_node_text(params_node, source_bytes) if params_node else "()"

        # Get return type
        return_type = self._get_return_type(node, source_bytes)
        if return_type:
            signature = f"{signature}: {return_type}"

        # Get docblock
        docstring = self._get_docblock(node, source_bytes)

        # Get content
        content = self.get_node_text(node, source_bytes)

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
            visibility=visibility,
        )

    def _extract_interface(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract an interface definition."""
        name_node = self.find_child_by_type(node, "name")
        if not name_node:
            return None

        name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)
        docstring = self._get_docblock(node, source_bytes)
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
        )

    def _extract_trait(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract a trait and its methods."""
        chunks = []

        name_node = self.find_child_by_type(node, "name")
        if not name_node:
            return chunks

        trait_name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)
        docstring = self._get_docblock(node, source_bytes)

        # Get method names
        method_names = self._get_method_names(node, source_bytes)

        content = f"trait {trait_name}"
        if docstring:
            content = f"/**\n * {docstring}\n */\n{content}"
        if method_names:
            content += f"\n\n// Methods: {', '.join(method_names)}"

        trait_chunk = CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="trait",
            symbol_name=trait_name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            visibility="public",
            extras={"methods": method_names},
        )
        chunks.append(trait_chunk)

        # Extract methods
        decl_list = self.find_child_by_type(node, "declaration_list")
        if decl_list:
            for method_node in self.find_children_by_type(decl_list, "method_declaration"):
                method_chunk = self._extract_method(method_node, source_bytes, file_path, trait_name)
                if method_chunk:
                    chunks.append(method_chunk)

        return chunks

    def _get_visibility(self, node, source_bytes: bytes) -> str:
        """Get visibility modifier (public/private/protected)."""
        for child in node.children:
            if child.type == "visibility_modifier":
                return self.get_node_text(child, source_bytes).lower()
        return "public"

    def _get_return_type(self, node, source_bytes: bytes) -> Optional[str]:
        """Get return type annotation."""
        # Look for return type after colon
        seen_colon = False
        for child in node.children:
            if child.type == ":":
                seen_colon = True
            elif seen_colon and child.type in ("named_type", "primitive_type", "nullable_type", "union_type"):
                return self.get_node_text(child, source_bytes)
        return None

    def _get_docblock(self, node, source_bytes: bytes) -> Optional[str]:
        """Get PHPDoc block preceding a node."""
        # Look for comment before the node
        prev_sibling = node.prev_sibling
        while prev_sibling:
            if prev_sibling.type == "comment":
                comment = self.get_node_text(prev_sibling, source_bytes)
                if comment.startswith("/**"):
                    return self._clean_docblock(comment)
            elif prev_sibling.type not in ("attribute_list",):
                break
            prev_sibling = prev_sibling.prev_sibling
        return None

    def _clean_docblock(self, docblock: str) -> str:
        """Clean up a PHPDoc block."""
        # Remove /** and */
        lines = docblock.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if line.startswith("/**"):
                line = line[3:].strip()
            if line.endswith("*/"):
                line = line[:-2].strip()
            if line.startswith("*"):
                line = line[1:].strip()
            if line:
                cleaned.append(line)
        return "\n".join(cleaned)

    def _get_base_class(self, node, source_bytes: bytes) -> Optional[str]:
        """Get base class from extends clause."""
        extends = self.find_child_by_type(node, "base_clause")
        if extends:
            name = self.find_child_by_type(extends, "name")
            if name:
                return self.get_node_text(name, source_bytes)
        return None

    def _get_interfaces(self, node, source_bytes: bytes) -> list[str]:
        """Get interfaces from implements clause."""
        interfaces = []
        impl_clause = self.find_child_by_type(node, "class_interface_clause")
        if impl_clause:
            for name_node in self.find_children_by_type(impl_clause, "name"):
                interfaces.append(self.get_node_text(name_node, source_bytes))
        return interfaces

    def _get_method_names(self, node, source_bytes: bytes) -> list[str]:
        """Get all method names in a class/trait."""
        names = []
        decl_list = self.find_child_by_type(node, "declaration_list")
        if decl_list:
            for method_node in self.find_children_by_type(decl_list, "method_declaration"):
                name_node = self.find_child_by_type(method_node, "name")
                if name_node:
                    names.append(self.get_node_text(name_node, source_bytes))
        return names

    def _detect_drupal_hook(self, func_name: str, docstring: Optional[str]) -> Optional[str]:
        """
        Detect if a function is a Drupal hook.

        Returns the hook type (e.g., "hook_cron", "hook_form_alter") if detected.
        """
        # Check docstring for "Implements hook_X()"
        if docstring:
            match = DRUPAL_HOOK_COMMENT.search(docstring)
            if match:
                return f"hook_{match.group(1)}"

        # Check function name pattern
        match = DRUPAL_HOOK_PATTERN.match(func_name)
        if match:
            # The second group is the hook type
            return match.group(2)

        return None
