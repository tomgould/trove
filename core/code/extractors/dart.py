"""
Dart code extractor using regex-based parsing.

tree-sitter-dart is not available as a pip package, so this extractor
uses regex patterns + brace counting to extract symbols from Dart source files.

Extracts: classes (abstract/sealed/base/final/interface), mixins, extensions,
enums, typedefs, top-level functions, class methods, and doc comments.
"""

import logging
import re
from typing import Optional

from .base import BaseExtractor, CodeChunk

logger = logging.getLogger(__name__)

# Declaration patterns
_CLASS_RE = re.compile(
    r"^(?:abstract\s+|sealed\s+|base\s+|final\s+|interface\s+)*class\s+(\w+)",
    re.MULTILINE,
)
_MIXIN_RE = re.compile(r"^mixin\s+(\w+)", re.MULTILINE)
_EXTENSION_RE = re.compile(r"^extension\s+(\w+)\s+on\s+(\S+)", re.MULTILINE)
_ENUM_RE = re.compile(r"^enum\s+(\w+)", re.MULTILINE)
_TYPEDEF_RE = re.compile(
    r"^typedef\s+(\w+)\s*(?:<[^>]*>)?\s*=", re.MULTILINE
)
# Top-level function: return-type name( — but NOT inside a class body.
# Matches patterns like: void main(, Future<void> build(, String? _helper(
_FUNCTION_RE = re.compile(
    r"^(?:\w+(?:<[^>]*>)?\??\s+)+(\w+)\s*(?:<[^>]*>)?\s*\(", re.MULTILINE
)

# Doc comment patterns
_DOC_SINGLE_RE = re.compile(r"^\s*///(.*)$")
_DOC_BLOCK_START_RE = re.compile(r"^\s*/\*\*")
_DOC_BLOCK_END_RE = re.compile(r"\*/\s*$")

# Method pattern (inside class bodies) — same as function but used at indent
_METHOD_RE = re.compile(
    r"^\s+(?:static\s+|@override\s+|@protected\s+)*"
    r"(?:\w+(?:<[^>]*>)?\??\s+)+(\w+)\s*(?:<[^>]*>)?\s*\(",
    re.MULTILINE,
)


def _find_brace_end(source: str, start: int) -> int:
    """Find the position after the matching closing brace.

    Args:
        source: Full source text.
        start: Position of the opening '{'.

    Returns:
        Position after the matching '}', or len(source) if unmatched.
    """
    depth = 0
    i = start
    in_string = None  # Track string delimiters
    prev_char = ""

    while i < len(source):
        ch = source[i]

        # Handle string literals (skip their content)
        if in_string:
            if ch == in_string and prev_char != "\\":
                in_string = None
        elif ch in ("'", '"'):
            in_string = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1

        prev_char = ch
        i += 1

    return len(source)


def _extract_doc_comment(lines: list[str], decl_line_idx: int) -> Optional[str]:
    """Extract doc comment immediately above a declaration line.

    Supports both ``/// triple-slash`` and ``/** block */`` styles.
    """
    if decl_line_idx <= 0:
        return None

    collected: list[str] = []
    idx = decl_line_idx - 1

    # Check for /// style first
    while idx >= 0:
        line = lines[idx].strip()
        m = _DOC_SINGLE_RE.match(lines[idx])
        if m:
            collected.append(m.group(1).strip())
            idx -= 1
        else:
            break

    if collected:
        collected.reverse()
        return "\n".join(collected)

    # Check for /** block */ style
    idx = decl_line_idx - 1
    while idx >= 0:
        line = lines[idx].strip()
        if not line:
            idx -= 1
            continue
        # Scan backwards from decl to find the block comment
        if _DOC_BLOCK_END_RE.search(line) or line == "*/":
            # Found end of block comment, scan backwards for start
            end_idx = idx
            while idx >= 0:
                if _DOC_BLOCK_START_RE.match(lines[idx].strip()):
                    # Extract the block comment content
                    block_lines = lines[idx : end_idx + 1]
                    cleaned = []
                    for bl in block_lines:
                        bl = bl.strip()
                        if bl.startswith("/**"):
                            bl = bl[3:].strip()
                        if bl.endswith("*/"):
                            bl = bl[:-2].strip()
                        if bl.startswith("*"):
                            bl = bl[1:].strip()
                        if bl:
                            cleaned.append(bl)
                    return "\n".join(cleaned) if cleaned else None
                idx -= 1
        break

    return None


def _line_of_offset(source: str, offset: int) -> int:
    """Return the 1-indexed line number for a byte offset."""
    return source[:offset].count("\n") + 1


class DartExtractor(BaseExtractor):
    """Extract code chunks from Dart source files using regex parsing."""

    language = "dart"
    tree_sitter_language = "dart"  # Not actually used

    @property
    def parser(self):
        """Override to avoid loading tree-sitter (no Dart grammar available)."""
        return None

    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract classes, functions, mixins, enums, etc. from Dart code."""
        chunks: list[CodeChunk] = []
        lines = source_code.split("\n")

        # Track extracted byte ranges to avoid duplicates
        extracted_ranges: list[tuple[int, int]] = []

        # 1. Extract classes (including abstract/sealed/base/final/interface)
        for m in _CLASS_RE.finditer(source_code):
            class_chunks = self._extract_class(
                m, source_code, lines, file_path
            )
            if class_chunks:
                chunks.extend(class_chunks)
                # Find the brace range for this class
                brace_pos = source_code.find("{", m.start())
                if brace_pos != -1:
                    end_pos = _find_brace_end(source_code, brace_pos)
                    extracted_ranges.append((m.start(), end_pos))

        # 2. Extract mixins
        for m in _MIXIN_RE.finditer(source_code):
            if not self._in_extracted_range(m.start(), extracted_ranges):
                chunk = self._extract_mixin(m, source_code, lines, file_path)
                if chunk:
                    chunks.append(chunk)
                    brace_pos = source_code.find("{", m.start())
                    if brace_pos != -1:
                        end_pos = _find_brace_end(source_code, brace_pos)
                        extracted_ranges.append((m.start(), end_pos))

        # 3. Extract extensions
        for m in _EXTENSION_RE.finditer(source_code):
            if not self._in_extracted_range(m.start(), extracted_ranges):
                chunk = self._extract_extension(
                    m, source_code, lines, file_path
                )
                if chunk:
                    chunks.append(chunk)
                    brace_pos = source_code.find("{", m.start())
                    if brace_pos != -1:
                        end_pos = _find_brace_end(source_code, brace_pos)
                        extracted_ranges.append((m.start(), end_pos))

        # 4. Extract enums
        for m in _ENUM_RE.finditer(source_code):
            if not self._in_extracted_range(m.start(), extracted_ranges):
                chunk = self._extract_enum(m, source_code, lines, file_path)
                if chunk:
                    chunks.append(chunk)
                    brace_pos = source_code.find("{", m.start())
                    if brace_pos != -1:
                        end_pos = _find_brace_end(source_code, brace_pos)
                        extracted_ranges.append((m.start(), end_pos))

        # 5. Extract typedefs (single-line, no brace body)
        for m in _TYPEDEF_RE.finditer(source_code):
            if not self._in_extracted_range(m.start(), extracted_ranges):
                chunk = self._extract_typedef(m, source_code, lines, file_path)
                if chunk:
                    chunks.append(chunk)

        # 6. Extract top-level functions (not inside any class/mixin/etc.)
        for m in _FUNCTION_RE.finditer(source_code):
            if not self._in_extracted_range(m.start(), extracted_ranges):
                # Verify it's truly at column 0 (top-level)
                line_start_pos = source_code.rfind("\n", 0, m.start()) + 1
                if m.start() == line_start_pos:
                    chunk = self._extract_top_level_function(
                        m, source_code, lines, file_path
                    )
                    if chunk:
                        chunks.append(chunk)
                        brace_pos = source_code.find("{", m.start())
                        if brace_pos != -1:
                            end_pos = _find_brace_end(source_code, brace_pos)
                            extracted_ranges.append((m.start(), end_pos))

        return chunks

    def _in_extracted_range(
        self, offset: int, ranges: list[tuple[int, int]]
    ) -> bool:
        """Check if an offset falls within any already-extracted range."""
        for start, end in ranges:
            if start <= offset < end:
                return True
        return False

    def _extract_class(
        self,
        match: re.Match,
        source: str,
        lines: list[str],
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract a class declaration + its methods."""
        chunks: list[CodeChunk] = []
        class_name = match.group(1)
        decl_line = match.group(0).strip()
        decl_line_idx = _line_of_offset(source, match.start()) - 1

        # Find opening brace
        brace_pos = source.find("{", match.start())
        if brace_pos == -1:
            return chunks

        end_pos = _find_brace_end(source, brace_pos)
        class_body = source[brace_pos + 1 : end_pos - 1]

        line_start = decl_line_idx + 1
        line_end = _line_of_offset(source, end_pos - 1)

        # Get doc comment
        docstring = _extract_doc_comment(lines, decl_line_idx)

        # Detect modifiers from the full declaration line
        full_decl = source[match.start() : source.find("\n", match.start())]
        extras: dict = {}
        if "abstract" in full_decl.split("class")[0]:
            extras["abstract"] = True
        if "sealed" in full_decl.split("class")[0]:
            extras["sealed"] = True

        # Detect extends/implements/with from declaration
        decl_to_brace = source[match.start() : brace_pos].strip()
        base_class = None
        extends_m = re.search(r"\bextends\s+(\w+)", decl_to_brace)
        if extends_m:
            base_class = extends_m.group(1)
            extras["base_class"] = base_class

        with_m = re.findall(r"\bwith\s+([\w,\s]+?)(?:\s*(?:implements|{)|\s*$)", decl_to_brace)
        if with_m:
            mixins = [m.strip() for m in with_m[0].split(",") if m.strip()]
            if mixins:
                extras["mixins"] = mixins

        implements_m = re.search(r"\bimplements\s+([\w,\s]+?)(?:\s*{|\s*$)", decl_to_brace)
        if implements_m:
            interfaces = [i.strip() for i in implements_m.group(1).split(",") if i.strip()]
            if interfaces:
                extras["implements"] = interfaces

        # Extract method names for overview
        method_names = self._find_method_names(class_body)

        # Build class overview content
        overview = decl_to_brace
        if docstring:
            overview = f"/// {docstring}\n{overview}"
        if method_names:
            overview += f"\n\n// Methods: {', '.join(method_names)}"

        visibility = self._dart_visibility(class_name)

        class_chunk = CodeChunk(
            content=overview,
            source_file=file_path,
            language=self.language,
            symbol_type="class",
            symbol_name=class_name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            visibility=visibility,
            extras=extras,
        )
        chunks.append(class_chunk)

        # Extract individual methods
        method_chunks = self._extract_methods(
            class_body, brace_pos + 1, source, lines, file_path, class_name
        )
        chunks.extend(method_chunks)

        return chunks

    def _extract_methods(
        self,
        body: str,
        body_offset: int,
        source: str,
        lines: list[str],
        file_path: str,
        parent_class: str,
    ) -> list[CodeChunk]:
        """Extract methods from a class/mixin body."""
        chunks: list[CodeChunk] = []
        extracted: list[tuple[int, int]] = []

        for m in _METHOD_RE.finditer(body):
            rel_start = m.start()
            if any(s <= rel_start < e for s, e in extracted):
                continue

            method_name = m.group(1)
            abs_start = body_offset + rel_start

            # Find method body
            brace_pos = body.find("{", rel_start)
            if brace_pos == -1:
                # Could be abstract method ending with ;
                semi_pos = body.find(";", rel_start)
                if semi_pos != -1:
                    method_text = body[rel_start : semi_pos + 1].strip()
                    line_start = _line_of_offset(source, abs_start)
                    line_end = _line_of_offset(
                        source, body_offset + semi_pos
                    )
                    decl_line_idx = line_start - 1
                    docstring = _extract_doc_comment(lines, decl_line_idx)

                    chunks.append(
                        CodeChunk(
                            content=method_text,
                            source_file=file_path,
                            language=self.language,
                            symbol_type="method",
                            symbol_name=method_name,
                            line_start=line_start,
                            line_end=line_end,
                            docstring=docstring,
                            parent_class=parent_class,
                            visibility=self._dart_visibility(method_name),
                        )
                    )
                continue

            end_pos = _find_brace_end(body, brace_pos)
            method_text = body[rel_start:end_pos].strip()
            extracted.append((rel_start, end_pos))

            line_start = _line_of_offset(source, abs_start)
            line_end = _line_of_offset(source, body_offset + end_pos - 1)
            decl_line_idx = line_start - 1
            docstring = _extract_doc_comment(lines, decl_line_idx)

            # Detect static/async
            extras: dict = {}
            first_line = method_text.split("\n")[0]
            if "static " in first_line:
                extras["static"] = True
            if "async " in first_line or "async*" in first_line:
                extras["async"] = True

            # Extract signature (up to opening brace)
            sig_end = method_text.find("{")
            signature = method_text[:sig_end].strip() if sig_end != -1 else method_text.split("\n")[0]

            chunks.append(
                CodeChunk(
                    content=method_text,
                    source_file=file_path,
                    language=self.language,
                    symbol_type="method",
                    symbol_name=method_name,
                    line_start=line_start,
                    line_end=line_end,
                    signature=signature,
                    docstring=docstring,
                    parent_class=parent_class,
                    visibility=self._dart_visibility(method_name),
                    extras=extras,
                )
            )

        return chunks

    def _extract_mixin(
        self,
        match: re.Match,
        source: str,
        lines: list[str],
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract a mixin declaration."""
        name = match.group(1)
        decl_line_idx = _line_of_offset(source, match.start()) - 1

        brace_pos = source.find("{", match.start())
        if brace_pos == -1:
            return None

        end_pos = _find_brace_end(source, brace_pos)
        content = source[match.start() : end_pos]

        docstring = _extract_doc_comment(lines, decl_line_idx)

        # Extract "on" constraint
        decl_to_brace = source[match.start() : brace_pos].strip()
        extras: dict = {}
        on_m = re.search(r"\bon\s+(\w+)", decl_to_brace)
        if on_m:
            extras["on_type"] = on_m.group(1)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="mixin",
            symbol_name=name,
            line_start=decl_line_idx + 1,
            line_end=_line_of_offset(source, end_pos - 1),
            docstring=docstring,
            visibility=self._dart_visibility(name),
            extras=extras,
        )

    def _extract_extension(
        self,
        match: re.Match,
        source: str,
        lines: list[str],
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract an extension declaration."""
        name = match.group(1)
        on_type = match.group(2)
        decl_line_idx = _line_of_offset(source, match.start()) - 1

        brace_pos = source.find("{", match.start())
        if brace_pos == -1:
            return None

        end_pos = _find_brace_end(source, brace_pos)
        content = source[match.start() : end_pos]

        docstring = _extract_doc_comment(lines, decl_line_idx)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="extension",
            symbol_name=name,
            line_start=decl_line_idx + 1,
            line_end=_line_of_offset(source, end_pos - 1),
            docstring=docstring,
            visibility=self._dart_visibility(name),
            extras={"on_type": on_type},
        )

    def _extract_enum(
        self,
        match: re.Match,
        source: str,
        lines: list[str],
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract an enum declaration."""
        name = match.group(1)
        decl_line_idx = _line_of_offset(source, match.start()) - 1

        brace_pos = source.find("{", match.start())
        if brace_pos == -1:
            return None

        end_pos = _find_brace_end(source, brace_pos)
        content = source[match.start() : end_pos]

        docstring = _extract_doc_comment(lines, decl_line_idx)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="enum",
            symbol_name=name,
            line_start=decl_line_idx + 1,
            line_end=_line_of_offset(source, end_pos - 1),
            docstring=docstring,
            visibility=self._dart_visibility(name),
        )

    def _extract_typedef(
        self,
        match: re.Match,
        source: str,
        lines: list[str],
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract a typedef declaration."""
        name = match.group(1)
        decl_line_idx = _line_of_offset(source, match.start()) - 1

        # Typedef ends at ;
        semi_pos = source.find(";", match.start())
        if semi_pos == -1:
            return None

        content = source[match.start() : semi_pos + 1]
        docstring = _extract_doc_comment(lines, decl_line_idx)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="typedef",
            symbol_name=name,
            line_start=decl_line_idx + 1,
            line_end=_line_of_offset(source, semi_pos),
            docstring=docstring,
            visibility=self._dart_visibility(name),
        )

    def _extract_top_level_function(
        self,
        match: re.Match,
        source: str,
        lines: list[str],
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract a top-level function."""
        name = match.group(1)
        decl_line_idx = _line_of_offset(source, match.start()) - 1

        brace_pos = source.find("{", match.start())
        # Could be => arrow function or just declaration ending in ;
        semi_pos = source.find(";", match.start())

        if brace_pos != -1 and (semi_pos == -1 or brace_pos < semi_pos):
            end_pos = _find_brace_end(source, brace_pos)
            content = source[match.start() : end_pos]
            line_end = _line_of_offset(source, end_pos - 1)
        elif semi_pos != -1:
            content = source[match.start() : semi_pos + 1]
            line_end = _line_of_offset(source, semi_pos)
        else:
            return None

        docstring = _extract_doc_comment(lines, decl_line_idx)

        # Extract signature (up to opening brace or semicolon)
        sig_end = content.find("{")
        if sig_end == -1:
            sig_end = content.find(";")
        signature = content[:sig_end].strip() if sig_end and sig_end != -1 else content.split("\n")[0]

        extras: dict = {}
        first_line = content.split("\n")[0]
        if "async " in first_line or "async*" in first_line:
            extras["async"] = True

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="function",
            symbol_name=name,
            line_start=decl_line_idx + 1,
            line_end=line_end,
            signature=signature,
            docstring=docstring,
            visibility=self._dart_visibility(name),
            extras=extras,
        )

    def _find_method_names(self, body: str) -> list[str]:
        """Find method names in a class body."""
        names: list[str] = []
        for m in _METHOD_RE.finditer(body):
            name = m.group(1)
            if name not in names:
                names.append(name)
        return names

    def _dart_visibility(self, name: str) -> str:
        """Determine Dart visibility from naming convention.

        In Dart, names starting with ``_`` are library-private.
        """
        if name.startswith("_"):
            return "private"
        return "public"

    def determine_visibility(self, name: str, node=None) -> str:
        """Override base class visibility for Dart conventions."""
        return self._dart_visibility(name)
