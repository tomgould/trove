"""
Code extractors and code-aware search helpers.
"""

import pytest


class TestDartExtractor:
    """Tests for the regex-based Dart code extractor."""

    def _extract(self, code: str, path: str = "test.dart"):
        from core.code.extractors.dart import DartExtractor

        return DartExtractor().extract(code, path)

    def test_simple_class(self):
        code = "class MyWidget {\n  int x = 0;\n}\n"
        chunks = self._extract(code)
        assert any(c.symbol_type == "class" and c.symbol_name == "MyWidget" for c in chunks)

    def test_private_class(self):
        code = "class _InternalState {\n  bool _active = false;\n}\n"
        chunks = self._extract(code)
        cls = [c for c in chunks if c.symbol_type == "class"][0]
        assert cls.symbol_name == "_InternalState"
        assert cls.visibility == "private"

    def test_abstract_class(self):
        code = "abstract class BaseRepository {\n  Future<void> fetch();\n}\n"
        chunks = self._extract(code)
        cls = [c for c in chunks if c.symbol_type == "class"][0]
        assert cls.symbol_name == "BaseRepository"
        assert cls.extras.get("abstract") is True

    def test_sealed_class(self):
        code = "sealed class Result {\n}\n"
        chunks = self._extract(code)
        cls = [c for c in chunks if c.symbol_type == "class"][0]
        assert cls.extras.get("sealed") is True

    def test_class_with_extends(self):
        code = "class HomePage extends StatelessWidget {\n  Widget build() {\n    return Container();\n  }\n}\n"
        chunks = self._extract(code)
        cls = [c for c in chunks if c.symbol_type == "class"][0]
        assert cls.extras.get("base_class") == "StatelessWidget"

    def test_class_with_implements(self):
        code = "class UserRepo implements Repository {\n  void save() {}\n}\n"
        chunks = self._extract(code)
        cls = [c for c in chunks if c.symbol_type == "class"][0]
        assert "Repository" in cls.extras.get("implements", [])

    def test_class_with_mixins(self):
        code = "class MyApp extends StatelessWidget with TickerProviderMixin {\n  void run() {}\n}\n"
        chunks = self._extract(code)
        cls = [c for c in chunks if c.symbol_type == "class"][0]
        assert "TickerProviderMixin" in cls.extras.get("mixins", [])

    def test_method_extraction(self):
        code = (
            "class Counter {\n"
            "  int _count = 0;\n"
            "\n"
            "  void increment() {\n"
            "    _count++;\n"
            "  }\n"
            "\n"
            "  int get count => _count;\n"
            "}\n"
        )
        chunks = self._extract(code)
        methods = [c for c in chunks if c.symbol_type == "method"]
        assert any(m.symbol_name == "increment" for m in methods)
        for m in methods:
            assert m.parent_class == "Counter"

    def test_private_method(self):
        code = "class Foo {\n  void _helper() {\n    print('hi');\n  }\n}\n"
        chunks = self._extract(code)
        methods = [c for c in chunks if c.symbol_type == "method"]
        priv = [m for m in methods if m.symbol_name == "_helper"]
        assert len(priv) == 1
        assert priv[0].visibility == "private"

    def test_static_method(self):
        code = "class Utils {\n  static String format(String s) {\n    return s.trim();\n  }\n}\n"
        chunks = self._extract(code)
        methods = [c for c in chunks if c.symbol_type == "method"]
        assert any(m.symbol_name == "format" and m.extras.get("static") for m in methods)

    def test_mixin_extraction(self):
        code = "mixin Loggable on Object {\n  void log(String msg) {\n    print(msg);\n  }\n}\n"
        chunks = self._extract(code)
        mixin_chunks = [c for c in chunks if c.symbol_type == "mixin"]
        assert len(mixin_chunks) == 1
        assert mixin_chunks[0].symbol_name == "Loggable"
        assert mixin_chunks[0].extras.get("on_type") == "Object"

    def test_extension_extraction(self):
        code = "extension StringExt on String {\n  bool get isBlank => trim().isEmpty;\n}\n"
        chunks = self._extract(code)
        ext = [c for c in chunks if c.symbol_type == "extension"]
        assert len(ext) == 1
        assert ext[0].symbol_name == "StringExt"
        assert ext[0].extras.get("on_type") == "String"

    def test_enum_extraction(self):
        code = "enum Color {\n  red,\n  green,\n  blue,\n}\n"
        chunks = self._extract(code)
        enums = [c for c in chunks if c.symbol_type == "enum"]
        assert len(enums) == 1
        assert enums[0].symbol_name == "Color"

    def test_typedef_extraction(self):
        code = "typedef JsonMap = Map<String, dynamic>;\n"
        chunks = self._extract(code)
        tds = [c for c in chunks if c.symbol_type == "typedef"]
        assert len(tds) == 1
        assert tds[0].symbol_name == "JsonMap"

    def test_top_level_function(self):
        code = "void main() {\n  runApp(MyApp());\n}\n"
        chunks = self._extract(code)
        funcs = [c for c in chunks if c.symbol_type == "function"]
        assert any(f.symbol_name == "main" for f in funcs)

    def test_async_function(self):
        code = "Future<void> fetchData() async {\n  await http.get(url);\n}\n"
        chunks = self._extract(code)
        funcs = [c for c in chunks if c.symbol_type == "function"]
        assert any(f.symbol_name == "fetchData" and f.extras.get("async") for f in funcs)

    def test_triple_slash_doc_comment(self):
        code = "/// A simple counter widget.\nclass Counter {\n}\n"
        chunks = self._extract(code)
        cls = [c for c in chunks if c.symbol_type == "class"][0]
        assert cls.docstring is not None
        assert "simple counter" in cls.docstring

    def test_block_doc_comment(self):
        code = "/**\n * Manages user authentication.\n */\nclass AuthService {\n}\n"
        chunks = self._extract(code)
        cls = [c for c in chunks if c.symbol_type == "class"][0]
        assert cls.docstring is not None
        assert "authentication" in cls.docstring

    def test_stateless_widget_pattern(self):
        code = (
            "class MyHomePage extends StatelessWidget {\n"
            "  const MyHomePage({super.key});\n"
            "\n"
            "  @override\n"
            "  Widget build(BuildContext context) {\n"
            "    return Scaffold(\n"
            "      body: Center(child: Text('Hello')),\n"
            "    );\n"
            "  }\n"
            "}\n"
        )
        chunks = self._extract(code)
        cls = [c for c in chunks if c.symbol_type == "class"][0]
        assert cls.symbol_name == "MyHomePage"
        assert cls.extras.get("base_class") == "StatelessWidget"
        methods = [c for c in chunks if c.symbol_type == "method"]
        assert any(m.symbol_name == "build" for m in methods)

    def test_stateful_widget_pattern(self):
        code = (
            "class CounterPage extends StatefulWidget {\n"
            "  const CounterPage({super.key});\n"
            "\n"
            "  @override\n"
            "  State<CounterPage> createState() => _CounterPageState();\n"
            "}\n"
            "\n"
            "class _CounterPageState extends State<CounterPage> {\n"
            "  int _count = 0;\n"
            "\n"
            "  void _increment() {\n"
            "    setState(() { _count++; });\n"
            "  }\n"
            "\n"
            "  @override\n"
            "  Widget build(BuildContext context) {\n"
            "    return Text('$_count');\n"
            "  }\n"
            "}\n"
        )
        chunks = self._extract(code)
        classes = [c for c in chunks if c.symbol_type == "class"]
        assert len(classes) == 2
        names = {c.symbol_name for c in classes}
        assert "CounterPage" in names
        assert "_CounterPageState" in names

    def test_empty_file(self):
        chunks = self._extract("")
        assert chunks == []

    def test_import_only_file(self):
        code = "import 'package:flutter/material.dart';\n\nimport 'dart:async';\n"
        chunks = self._extract(code)
        assert chunks == []

    def test_visibility_detection(self):
        from core.code.extractors.dart import DartExtractor

        ext = DartExtractor()
        assert ext.determine_visibility("MyClass") == "public"
        assert ext.determine_visibility("_private") == "private"
        assert ext.determine_visibility("publicMethod") == "public"
        assert ext.determine_visibility("_internalHelper") == "private"

    def test_multiple_constructs(self):
        """Test a file with multiple different construct types."""
        code = (
            "typedef Callback = void Function(int);\n"
            "\n"
            "enum Status { active, inactive }\n"
            "\n"
            "mixin Printable on Object {\n"
            "  void printSelf() {\n"
            "    print(this);\n"
            "  }\n"
            "}\n"
            "\n"
            "class MyClass {\n"
            "  void doWork() {}\n"
            "}\n"
            "\n"
            "void main() {\n"
            "  print('hello');\n"
            "}\n"
        )
        chunks = self._extract(code)
        types = {c.symbol_type for c in chunks}
        assert "typedef" in types
        assert "enum" in types
        assert "mixin" in types
        assert "class" in types
        assert "function" in types

    def test_method_names_in_class_overview(self):
        code = (
            "class Service {\n"
            "  void start() {}\n"
            "  void stop() {}\n"
            "  Future<void> restart() async {}\n"
            "}\n"
        )
        chunks = self._extract(code)
        cls = [c for c in chunks if c.symbol_type == "class"][0]
        assert "start" in cls.content
        assert "stop" in cls.content

    def test_language_metadata(self):
        code = "class Foo {\n}\n"
        chunks = self._extract(code)
        assert all(c.language == "dart" for c in chunks)

    def test_line_numbers(self):
        code = "// line 1\n// line 2\nclass Foo {\n  void bar() {}\n}\n"
        chunks = self._extract(code)
        cls = [c for c in chunks if c.symbol_type == "class"][0]
        assert cls.line_start == 3
        assert cls.line_end >= 5

    def test_nested_braces_in_method(self):
        """Brace counting should handle nested blocks correctly."""
        code = (
            "class Logic {\n"
            "  void process(List<int> items) {\n"
            "    for (var i in items) {\n"
            "      if (i > 0) {\n"
            "        print(i);\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n"
        )
        chunks = self._extract(code)
        methods = [c for c in chunks if c.symbol_type == "method"]
        assert len(methods) == 1
        assert methods[0].symbol_name == "process"
        # Content should include all nested braces
        assert "for (var i in items)" in methods[0].content

    def test_string_with_braces_does_not_confuse_parser(self):
        code = (
            "class Formatter {\n"
            "  String format() {\n"
            '    return "value: ${compute()}";\n'
            "  }\n"
            "}\n"
        )
        chunks = self._extract(code)
        methods = [c for c in chunks if c.symbol_type == "method"]
        assert len(methods) == 1


###############################################################################
# Code search feature tests
###############################################################################

class TestExtractCodeSearchTerms:
    """Tests for HybridSearcher._extract_code_search_terms."""

    def _make_searcher(self):
        """Create a minimal searcher without full init."""
        from core.search import HybridSearcher
        searcher = HybridSearcher.__new__(HybridSearcher)
        return searcher

    def test_extracts_camel_case(self):
        s = self._make_searcher()
        terms = s._extract_code_search_terms("The HybridSearcher class handles queries")
        assert "HybridSearcher" in terms

    def test_extracts_snake_case(self):
        s = self._make_searcher()
        terms = s._extract_code_search_terms("Call build_rag_prompt to create messages")
        assert "build_rag_prompt" in terms

    def test_extracts_backtick_code(self):
        s = self._make_searcher()
        terms = s._extract_code_search_terms("Use `search_with_quality()` for results")
        assert "search_with_quality()" in terms

    def test_extracts_file_references(self):
        s = self._make_searcher()
        # The regex captures the extension group ("py"), which is added as a term
        # Also "core/search" is found as a path-like pattern
        terms = s._extract_code_search_terms("See core/search.py and utils/helper.tsx for details")
        # Should extract something from file references
        assert len(terms) > 0

    def test_deduplicates_and_limits(self):
        s = self._make_searcher()
        # Many repeated terms
        content = " ".join(["HybridSearcher"] * 20 + [f"term_{i}_value" for i in range(20)])
        terms = s._extract_code_search_terms(content)
        assert len(terms) <= 10
        assert len(terms) == len(set(t.lower() for t in terms))

    def test_skips_short_terms(self):
        s = self._make_searcher()
        terms = s._extract_code_search_terms("a `b` or `cd` and longer_term")
        assert "b" not in terms
        assert "cd" not in terms
        assert "longer_term" in terms


class TestPrioritizeCodeChunks:
    """Tests for HybridSearcher.prioritize_code_chunks."""

    def _make_searcher(self):
        from core.search import HybridSearcher
        return HybridSearcher.__new__(HybridSearcher)

    def test_boosts_code_chunks(self):
        s = self._make_searcher()
        chunks = [
            {"content": "docs", "metadata": {"chunk_type": "document"}, "_rerank_score": 0.8},
            {"content": "code", "metadata": {"chunk_type": "code"}, "_rerank_score": 0.5},
        ]
        result = s.prioritize_code_chunks(chunks, boost=0.5)
        # Code chunk (0.5 + 0.5 = 1.0) should now be first
        assert result[0]["metadata"]["chunk_type"] == "code"
        assert result[0]["_rerank_score"] == pytest.approx(1.0)
        assert result[0]["_code_boosted"] is True

    def test_does_not_boost_documents(self):
        s = self._make_searcher()
        chunks = [
            {"content": "docs", "metadata": {"chunk_type": "document"}, "_rerank_score": 0.8},
        ]
        result = s.prioritize_code_chunks(chunks, boost=0.5)
        assert result[0]["_rerank_score"] == pytest.approx(0.8)
        assert "_code_boosted" not in result[0]

    def test_boosts_combined_score_when_no_rerank(self):
        s = self._make_searcher()
        chunks = [
            {"content": "code", "metadata": {"chunk_type": "code"}, "_combined_score": 0.3},
        ]
        result = s.prioritize_code_chunks(chunks, boost=0.2)
        assert result[0]["_combined_score"] == pytest.approx(0.5)

    def test_empty_chunks(self):
        s = self._make_searcher()
        assert s.prioritize_code_chunks([]) == []


class TestPrioritizeDataChunks:
    """Tests for HybridSearcher.prioritize_data_chunks."""

    def _make_searcher(self):
        from core.search import HybridSearcher
        return HybridSearcher.__new__(HybridSearcher)

    def test_spreadsheets_ranked_above_text(self):
        s = self._make_searcher()
        chunks = [
            {"content": "some text", "metadata": {"source_file": "guide.pdf"}, "_rerank_score": 0.9},
            {"content": "data rows", "metadata": {"source_file": "reference.xlsx"}, "_rerank_score": 0.5},
        ]
        result = s.prioritize_data_chunks(chunks)
        # Spreadsheet reference should be first (tier 0) despite lower score
        assert result[0]["metadata"]["source_file"] == "reference.xlsx"

    def test_reference_patterns_deprioritized(self):
        s = self._make_searcher()
        chunks = [
            {"content": "refer to appendix B for details", "metadata": {"source_file": "intro.pdf"}, "_rerank_score": 0.9},
            {"content": "actual data here\nrow1\nrow2", "metadata": {"source_file": "data.pdf"}, "_rerank_score": 0.3},
        ]
        result = s.prioritize_data_chunks(chunks)
        # "refer to appendix" content goes to tier 4
        assert result[0]["metadata"]["source_file"] == "data.pdf"

    def test_empty_chunks(self):
        s = self._make_searcher()
        assert s.prioritize_data_chunks([]) == []


class TestApplyCodeFilters:
    """Tests for HybridSearcher._apply_code_filters."""

    def _make_searcher(self):
        from core.search import HybridSearcher
        return HybridSearcher.__new__(HybridSearcher)

    def _sample_chunks(self):
        return [
            {"content": "def foo()", "metadata": {"chunk_type": "code", "language": "python", "symbol_type": "function", "repo_name": "trag"}},
            {"content": "A guide to search", "metadata": {"chunk_type": "document"}},
            {"content": "class Bar {}", "metadata": {"chunk_type": "code", "language": "typescript", "symbol_type": "class", "repo_name": "bsc"}},
        ]

    def test_no_filters_returns_all(self):
        s = self._make_searcher()
        chunks = self._sample_chunks()
        result = s._apply_code_filters(chunks)
        assert len(result) == 3

    def test_filter_by_chunk_type(self):
        s = self._make_searcher()
        chunks = self._sample_chunks()
        result = s._apply_code_filters(chunks, chunk_types=["code"])
        assert len(result) == 2
        assert all(c["metadata"]["chunk_type"] == "code" for c in result)

    def test_filter_by_language(self):
        s = self._make_searcher()
        chunks = self._sample_chunks()
        result = s._apply_code_filters(chunks, chunk_types=["code"], languages=["python"])
        assert len(result) == 1
        assert result[0]["metadata"]["language"] == "python"

    def test_filter_by_symbol_type(self):
        s = self._make_searcher()
        chunks = self._sample_chunks()
        result = s._apply_code_filters(chunks, chunk_types=["code"], symbol_types=["class"])
        assert len(result) == 1
        assert result[0]["metadata"]["symbol_type"] == "class"

    def test_filter_by_repo(self):
        s = self._make_searcher()
        chunks = self._sample_chunks()
        result = s._apply_code_filters(chunks, repos=["trag"])
        # trag repo chunk + document (no repo = included)
        assert any(c["metadata"].get("repo_name") == "trag" for c in result)

    def test_combined_filters(self):
        s = self._make_searcher()
        chunks = self._sample_chunks()
        result = s._apply_code_filters(chunks, chunk_types=["code"], languages=["typescript"], repos=["bsc"])
        assert len(result) == 1
        assert result[0]["metadata"]["language"] == "typescript"

    def test_empty_chunks(self):
        s = self._make_searcher()
        assert s._apply_code_filters([], chunk_types=["code"]) == []
