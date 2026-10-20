"""
CSS code extractor using tree-sitter.

Extracts selectors, variables (custom properties), mixins, and keyframes.
Handles CSS, SCSS, SASS, and LESS.
"""

import logging
import re
from typing import Optional

from .base import BaseExtractor, CodeChunk

logger = logging.getLogger(__name__)


class CSSExtractor(BaseExtractor):
    """Extract code chunks from CSS source files."""

    language = "css"
    tree_sitter_language = "css"

    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract selectors, variables, and mixins from CSS."""
        chunks = []
        source_bytes = bytes(source_code, "utf-8")

        # Determine CSS flavor
        is_scss = file_path.endswith((".scss", ".sass"))
        is_less = file_path.endswith(".less")

        try:
            tree = self.parse(source_code)
        except Exception as e:
            logger.warning(f"Failed to parse CSS file {file_path}: {e}")
            # Fallback to regex-based extraction
            return self._extract_fallback(source_code, file_path)

        root = tree.root_node

        # Extract rule sets (selectors)
        for rule in self.find_descendants_by_type(root, "rule_set"):
            chunk = self._extract_rule(rule, source_bytes, file_path)
            if chunk:
                chunks.append(chunk)

        # Extract keyframes
        for keyframes in self.find_descendants_by_type(root, "keyframes_statement"):
            chunk = self._extract_keyframes(keyframes, source_bytes, file_path)
            if chunk:
                chunks.append(chunk)

        # Extract media queries
        for media in self.find_descendants_by_type(root, "media_statement"):
            chunk = self._extract_media_query(media, source_bytes, file_path)
            if chunk:
                chunks.append(chunk)

        # For SCSS/LESS, also extract with regex for mixins and variables
        if is_scss or is_less:
            scss_chunks = self._extract_scss_extras(source_code, file_path)
            chunks.extend(scss_chunks)

        return chunks

    def _extract_rule(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract a CSS rule set."""
        # Get selectors
        selectors_node = self.find_child_by_type(node, "selectors")
        if not selectors_node:
            return None

        selectors = self.get_node_text(selectors_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)

        # Clean up selector for symbol name
        symbol_name = selectors.split(",")[0].strip()  # Use first selector
        symbol_name = re.sub(r"\s+", " ", symbol_name)  # Normalize whitespace

        content = self.get_node_text(node, source_bytes)

        # Check if this contains CSS variables
        has_variables = "--" in content

        extras = {}
        if has_variables:
            variables = re.findall(r"(--[\w-]+):", content)
            if variables:
                extras["variables"] = variables

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="selector",
            symbol_name=symbol_name,
            line_start=line_start,
            line_end=line_end,
            extras=extras,
        )

    def _extract_keyframes(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract a @keyframes rule."""
        # Get keyframes name
        name_node = self.find_child_by_type(node, "keyframes_name")
        if not name_node:
            return None

        name = self.get_node_text(name_node, source_bytes)
        line_start, line_end = self.get_node_line_range(node)
        content = self.get_node_text(node, source_bytes)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="keyframes",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
        )

    def _extract_media_query(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract a @media rule."""
        line_start, line_end = self.get_node_line_range(node)
        content = self.get_node_text(node, source_bytes)

        # Extract media query condition
        query_match = re.search(r"@media\s*([^{]+)", content)
        query = query_match.group(1).strip() if query_match else "unknown"

        # Truncate long content
        if len(content) > 1500:
            content = content[:1500] + "\n/* truncated */"

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="media_query",
            symbol_name=f"@media {query[:50]}",
            line_start=line_start,
            line_end=line_end,
            extras={"query": query},
        )

    def _extract_scss_extras(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract SCSS/LESS specific features using regex."""
        chunks = []
        lines = source_code.split("\n")

        # Extract SCSS variables ($var)
        var_pattern = re.compile(r"^(\$[\w-]+)\s*:\s*([^;]+);", re.MULTILINE)
        for match in var_pattern.finditer(source_code):
            var_name = match.group(1)
            var_value = match.group(2)
            line_num = source_code[:match.start()].count("\n") + 1

            chunks.append(CodeChunk(
                content=match.group(0),
                source_file=file_path,
                language="scss",
                symbol_type="variable",
                symbol_name=var_name,
                line_start=line_num,
                line_end=line_num,
                extras={"value": var_value.strip()},
            ))

        # Extract mixins (@mixin name)
        mixin_pattern = re.compile(r"@mixin\s+([\w-]+)\s*(\([^)]*\))?\s*\{", re.MULTILINE)
        for match in mixin_pattern.finditer(source_code):
            mixin_name = match.group(1)
            params = match.group(2) or "()"
            line_num = source_code[:match.start()].count("\n") + 1

            # Find the closing brace (simplified)
            start_idx = match.end() - 1
            brace_count = 1
            end_idx = start_idx + 1
            while brace_count > 0 and end_idx < len(source_code):
                if source_code[end_idx] == "{":
                    brace_count += 1
                elif source_code[end_idx] == "}":
                    brace_count -= 1
                end_idx += 1

            content = source_code[match.start():end_idx]
            end_line = source_code[:end_idx].count("\n") + 1

            chunks.append(CodeChunk(
                content=content,
                source_file=file_path,
                language="scss",
                symbol_type="mixin",
                symbol_name=mixin_name,
                line_start=line_num,
                line_end=end_line,
                signature=params,
            ))

        # Extract functions (@function name)
        func_pattern = re.compile(r"@function\s+([\w-]+)\s*(\([^)]*\))\s*\{", re.MULTILINE)
        for match in func_pattern.finditer(source_code):
            func_name = match.group(1)
            params = match.group(2)
            line_num = source_code[:match.start()].count("\n") + 1

            # Find the closing brace
            start_idx = match.end() - 1
            brace_count = 1
            end_idx = start_idx + 1
            while brace_count > 0 and end_idx < len(source_code):
                if source_code[end_idx] == "{":
                    brace_count += 1
                elif source_code[end_idx] == "}":
                    brace_count -= 1
                end_idx += 1

            content = source_code[match.start():end_idx]
            end_line = source_code[:end_idx].count("\n") + 1

            chunks.append(CodeChunk(
                content=content,
                source_file=file_path,
                language="scss",
                symbol_type="function",
                symbol_name=func_name,
                line_start=line_num,
                line_end=end_line,
                signature=params,
            ))

        return chunks

    def _extract_fallback(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Fallback regex-based extraction when tree-sitter fails."""
        chunks = []

        # Extract rule sets
        rule_pattern = re.compile(r"([^{}]+)\{([^{}]*)\}", re.MULTILINE)
        for match in rule_pattern.finditer(source_code):
            selector = match.group(1).strip()
            if not selector or selector.startswith("@"):
                continue

            line_num = source_code[:match.start()].count("\n") + 1
            end_line = source_code[:match.end()].count("\n") + 1

            # Clean up selector
            selector = re.sub(r"\s+", " ", selector)
            symbol_name = selector.split(",")[0].strip()

            chunks.append(CodeChunk(
                content=match.group(0),
                source_file=file_path,
                language=self.language,
                symbol_type="selector",
                symbol_name=symbol_name,
                line_start=line_num,
                line_end=end_line,
            ))

        return chunks
