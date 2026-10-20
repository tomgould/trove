"""
HTML code extractor using tree-sitter.

Extracts components, templates, and significant HTML structures.
Also handles Vue and Svelte single-file components.
"""

import logging
import re
from typing import Optional

from .base import BaseExtractor, CodeChunk

logger = logging.getLogger(__name__)


class HTMLExtractor(BaseExtractor):
    """Extract code chunks from HTML source files."""

    language = "html"
    tree_sitter_language = "html"

    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract components and templates from HTML."""
        chunks = []
        source_bytes = bytes(source_code, "utf-8")

        # Detect file type
        is_vue = file_path.endswith(".vue")
        is_svelte = file_path.endswith(".svelte")

        if is_vue:
            return self._extract_vue(source_code, file_path)
        elif is_svelte:
            return self._extract_svelte(source_code, file_path)

        try:
            tree = self.parse(source_code)
        except Exception as e:
            logger.warning(f"Failed to parse HTML file {file_path}: {e}")
            return chunks

        root = tree.root_node

        # Extract template elements (custom elements, web components)
        for element in self.find_descendants_by_type(root, "element"):
            chunk = self._extract_element(element, source_bytes, file_path)
            if chunk:
                chunks.append(chunk)

        # If no significant elements found, create a document chunk
        if not chunks and source_code.strip():
            chunks.append(self._create_document_chunk(source_code, file_path))

        return chunks

    def _extract_element(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract a significant HTML element."""
        # Get tag name
        start_tag = self.find_child_by_type(node, "start_tag")
        if not start_tag:
            return None

        tag_name_node = self.find_child_by_type(start_tag, "tag_name")
        if not tag_name_node:
            return None

        tag_name = self.get_node_text(tag_name_node, source_bytes)

        # Only extract custom elements (contain hyphen) or significant structural elements
        significant_tags = {"template", "slot", "component", "main", "article", "section", "header", "footer", "nav", "form"}
        is_custom_element = "-" in tag_name
        is_significant = tag_name.lower() in significant_tags

        if not is_custom_element and not is_significant:
            return None

        line_start, line_end = self.get_node_line_range(node)

        # Get id and class attributes
        attrs = self._get_attributes(start_tag, source_bytes)
        element_id = attrs.get("id")
        element_class = attrs.get("class")

        # Build symbol name
        symbol_name = tag_name
        if element_id:
            symbol_name = f"{tag_name}#{element_id}"
        elif element_class:
            first_class = element_class.split()[0] if element_class else ""
            if first_class:
                symbol_name = f"{tag_name}.{first_class}"

        content = self.get_node_text(node, source_bytes)

        # Truncate very long content
        if len(content) > 2000:
            content = content[:2000] + "\n<!-- truncated -->"

        symbol_type = "component" if is_custom_element else "template"

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type=symbol_type,
            symbol_name=symbol_name,
            line_start=line_start,
            line_end=line_end,
            extras={"tag": tag_name, "id": element_id, "class": element_class},
        )

    def _get_attributes(self, start_tag, source_bytes: bytes) -> dict[str, str]:
        """Get attributes from a start tag."""
        attrs = {}
        for attr in self.find_children_by_type(start_tag, "attribute"):
            name_node = self.find_child_by_type(attr, "attribute_name")
            value_node = self.find_child_by_type(attr, "quoted_attribute_value") or \
                         self.find_child_by_type(attr, "attribute_value")

            if name_node:
                name = self.get_node_text(name_node, source_bytes)
                value = ""
                if value_node:
                    value = self.get_node_text(value_node, source_bytes).strip('"\'')
                attrs[name] = value

        return attrs

    def _create_document_chunk(self, source_code: str, file_path: str) -> CodeChunk:
        """Create a chunk for the entire document."""
        lines = source_code.split("\n")
        content = source_code if len(source_code) < 2000 else source_code[:2000] + "\n<!-- truncated -->"

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="document",
            symbol_name=file_path.split("/")[-1],
            line_start=1,
            line_end=len(lines),
        )

    def _extract_vue(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract from Vue single-file component."""
        chunks = []
        lines = source_code.split("\n")

        # Extract component name from filename
        component_name = file_path.split("/")[-1].replace(".vue", "")

        # Find template section
        template_match = re.search(r"<template[^>]*>(.*?)</template>", source_code, re.DOTALL)
        if template_match:
            template_content = template_match.group(0)
            template_start = source_code[:template_match.start()].count("\n") + 1

            chunks.append(CodeChunk(
                content=template_content[:2000] if len(template_content) > 2000 else template_content,
                source_file=file_path,
                language="vue",
                symbol_type="template",
                symbol_name=f"{component_name}:template",
                line_start=template_start,
                line_end=template_start + template_content.count("\n"),
            ))

        # Find script section
        script_match = re.search(r"<script[^>]*>(.*?)</script>", source_code, re.DOTALL)
        if script_match:
            script_content = script_match.group(0)
            script_start = source_code[:script_match.start()].count("\n") + 1

            chunks.append(CodeChunk(
                content=script_content[:3000] if len(script_content) > 3000 else script_content,
                source_file=file_path,
                language="vue",
                symbol_type="component",
                symbol_name=component_name,
                line_start=script_start,
                line_end=script_start + script_content.count("\n"),
            ))

        # Find style section
        style_match = re.search(r"<style[^>]*>(.*?)</style>", source_code, re.DOTALL)
        if style_match:
            style_content = style_match.group(0)
            style_start = source_code[:style_match.start()].count("\n") + 1

            chunks.append(CodeChunk(
                content=style_content[:1000] if len(style_content) > 1000 else style_content,
                source_file=file_path,
                language="vue",
                symbol_type="style",
                symbol_name=f"{component_name}:style",
                line_start=style_start,
                line_end=style_start + style_content.count("\n"),
            ))

        return chunks

    def _extract_svelte(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract from Svelte single-file component."""
        chunks = []

        # Extract component name from filename
        component_name = file_path.split("/")[-1].replace(".svelte", "")

        # Find script section
        script_match = re.search(r"<script[^>]*>(.*?)</script>", source_code, re.DOTALL)
        script_content = script_match.group(0) if script_match else ""

        # The rest is template
        template_content = source_code
        if script_match:
            template_content = source_code[:script_match.start()] + source_code[script_match.end():]

        # Remove style section from template
        style_match = re.search(r"<style[^>]*>(.*?)</style>", template_content, re.DOTALL)
        if style_match:
            template_content = template_content[:style_match.start()] + template_content[style_match.end():]

        template_content = template_content.strip()

        if script_content:
            script_start = source_code[:script_match.start()].count("\n") + 1 if script_match else 1
            chunks.append(CodeChunk(
                content=script_content[:3000] if len(script_content) > 3000 else script_content,
                source_file=file_path,
                language="svelte",
                symbol_type="component",
                symbol_name=component_name,
                line_start=script_start,
                line_end=script_start + script_content.count("\n"),
            ))

        if template_content:
            chunks.append(CodeChunk(
                content=template_content[:2000] if len(template_content) > 2000 else template_content,
                source_file=file_path,
                language="svelte",
                symbol_type="template",
                symbol_name=f"{component_name}:template",
                line_start=1,
                line_end=template_content.count("\n") + 1,
            ))

        return chunks
