"""
Plugin interfaces, registry and loader.
"""

import pytest
from pathlib import Path


# ============================================================
# Plugin System Tests
# ============================================================

class TestPluginInterfaces:
    """Test abstract base class enforcement for plugin interfaces."""

    def test_base_embedder_is_abstract(self):
        from core.plugins.interfaces import BaseEmbedder
        with pytest.raises(TypeError):
            BaseEmbedder()

    def test_base_chunker_is_abstract(self):
        from core.plugins.interfaces import BaseChunker
        with pytest.raises(TypeError):
            BaseChunker()

    def test_base_parser_is_abstract(self):
        from core.plugins.interfaces import BaseParser
        with pytest.raises(TypeError):
            BaseParser()

    def test_base_reranker_is_abstract(self):
        from core.plugins.interfaces import BaseReranker
        with pytest.raises(TypeError):
            BaseReranker()

    def test_base_llm_provider_is_abstract(self):
        from core.plugins.interfaces import BaseLLMProvider
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_base_merge_strategy_is_abstract(self):
        from core.plugins.interfaces import BaseMergeStrategy
        with pytest.raises(TypeError):
            BaseMergeStrategy()

    def test_concrete_embedder_can_instantiate(self):
        from core.plugins.interfaces import BaseEmbedder

        class DummyEmbedder(BaseEmbedder):
            name = "dummy"
            dimension = 3

            def embed(self, texts):
                return [[0.1, 0.2, 0.3]] * len(texts)

        e = DummyEmbedder()
        assert e.name == "dummy"
        assert e.dimension == 3
        result = e.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 3

    def test_embedder_embed_query_delegates(self):
        from core.plugins.interfaces import BaseEmbedder

        class DummyEmbedder(BaseEmbedder):
            name = "dummy"
            dimension = 2

            def embed(self, texts):
                return [[float(len(t)), 0.0] for t in texts]

        e = DummyEmbedder()
        result = e.embed_query("hi")
        assert result == [2.0, 0.0]

    def test_embedder_close_is_optional(self):
        from core.plugins.interfaces import BaseEmbedder

        class DummyEmbedder(BaseEmbedder):
            name = "dummy"
            dimension = 1
            def embed(self, texts):
                return [[0.0]] * len(texts)

        e = DummyEmbedder()
        e.close()  # should not raise

    def test_concrete_chunker(self):
        from core.plugins.interfaces import BaseChunker

        class DummyChunker(BaseChunker):
            name = "dummy"
            def chunk(self, elements, max_chars=1500, overlap=100, **kwargs):
                return [str(e) for e in elements]

        c = DummyChunker()
        result = c.chunk(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_concrete_parser(self):
        from core.plugins.interfaces import BaseParser

        class DummyParser(BaseParser):
            name = "dummy"
            supported_extensions = [".txt"]
            def parse(self, path):
                return [{"content": "parsed"}]

        p = DummyParser()
        assert p.can_parse(Path("doc.txt"))
        assert not p.can_parse(Path("doc.pdf"))
        result = p.parse(Path("doc.txt"))
        assert result == [{"content": "parsed"}]

    def test_parser_can_parse_no_extensions(self):
        from core.plugins.interfaces import BaseParser

        class UniversalParser(BaseParser):
            name = "universal"
            supported_extensions = []
            def parse(self, path):
                return []

        p = UniversalParser()
        assert p.can_parse(Path("anything.xyz"))

    def test_concrete_reranker(self):
        from core.plugins.interfaces import BaseReranker

        class DummyReranker(BaseReranker):
            name = "dummy"
            def predict(self, pairs):
                return [1.0] * len(pairs)

        r = DummyReranker()
        scores = r.predict([("query", "doc1"), ("query", "doc2")])
        assert scores == [1.0, 1.0]
        r.close()  # should not raise

    def test_concrete_llm_provider(self):
        from core.plugins.interfaces import BaseLLMProvider

        class DummyLLM(BaseLLMProvider):
            name = "dummy"
            def generate(self, messages, system_prompt=None, max_tokens=2000):
                return "generated response"

        llm = DummyLLM()
        result = llm.generate([{"role": "user", "content": "hi"}])
        assert result == "generated response"

    def test_concrete_merge_strategy(self):
        from core.plugins.interfaces import BaseMergeStrategy

        class DummyMerge(BaseMergeStrategy):
            name = "dummy"
            def merge(self, bm25_results, vector_results, config):
                all_chunks = {}
                for chunk, score in bm25_results + vector_results:
                    cid = chunk["id"]
                    if cid not in all_chunks:
                        all_chunks[cid] = chunk.copy()
                        all_chunks[cid]["_combined_score"] = score
                return sorted(all_chunks.values(),
                              key=lambda c: c["_combined_score"], reverse=True)

        m = DummyMerge()
        bm25 = [({"id": "a", "content": "x"}, 0.5)]
        vec = [({"id": "b", "content": "y"}, 0.8)]
        result = m.merge(bm25, vec, {})
        assert result[0]["id"] == "b"  # higher score first


class TestPluginRegistry:
    """Test plugin registry registration and lookup."""

    def test_register_and_get_embedder(self):
        from core.plugins.interfaces import BaseEmbedder
        from core.plugins.registry import PluginRegistry

        class MyEmbedder(BaseEmbedder):
            name = "my_embedder"
            dimension = 128
            def embed(self, texts):
                return [[0.0] * 128] * len(texts)

        reg = PluginRegistry()
        reg.register_embedder(MyEmbedder)
        assert reg.get_embedder("my_embedder") is MyEmbedder
        assert reg.get_embedder("nonexistent") is None

    def test_register_and_get_all_types(self):
        from core.plugins.interfaces import (
            BaseChunker, BaseEmbedder, BaseLLMProvider,
            BaseMergeStrategy, BaseParser, BaseReranker,
        )
        from core.plugins.registry import PluginRegistry

        class E(BaseEmbedder):
            name = "e"
            dimension = 1
            def embed(self, texts): return [[0.0]] * len(texts)
        class C(BaseChunker):
            name = "c"
            def chunk(self, elements, **kw): return []
        class P(BaseParser):
            name = "p"
            def parse(self, path): return []
        class R(BaseReranker):
            name = "r"
            def predict(self, pairs): return []
        class L(BaseLLMProvider):
            name = "l"
            def generate(self, messages, **kw): return ""
        class M(BaseMergeStrategy):
            name = "m"
            def merge(self, bm25, vec, cfg): return []

        reg = PluginRegistry()
        reg.register_embedder(E)
        reg.register_chunker(C)
        reg.register_parser(P)
        reg.register_reranker(R)
        reg.register_llm_provider(L)
        reg.register_merge_strategy(M)

        assert reg.get_embedder("e") is E
        assert reg.get_chunker("c") is C
        assert reg.get_parser("p") is P
        assert reg.get_reranker("r") is R
        assert reg.get_llm_provider("l") is L
        assert reg.get_merge_strategy("m") is M

    def test_list_registered_plugins(self):
        from core.plugins.interfaces import BaseEmbedder, BaseLLMProvider
        from core.plugins.registry import PluginRegistry

        class E1(BaseEmbedder):
            name = "alpha"
            dimension = 1
            def embed(self, texts): return []
        class E2(BaseEmbedder):
            name = "beta"
            dimension = 1
            def embed(self, texts): return []
        class L(BaseLLMProvider):
            name = "gamma"
            def generate(self, messages, **kw): return ""

        reg = PluginRegistry()
        reg.register_embedder(E1)
        reg.register_embedder(E2)
        reg.register_llm_provider(L)

        assert reg.list_embedders() == ["alpha", "beta"]
        assert reg.list_llm_providers() == ["gamma"]
        assert reg.list_chunkers() == []

    def test_list_all(self):
        from core.plugins.registry import PluginRegistry
        reg = PluginRegistry()
        result = reg.list_all()
        assert set(result.keys()) == {
            "embedders", "chunkers", "parsers",
            "rerankers", "llm_providers", "merge_strategies",
        }
        for v in result.values():
            assert v == []

    def test_duplicate_name_overrides(self):
        from core.plugins.interfaces import BaseEmbedder
        from core.plugins.registry import PluginRegistry

        class E1(BaseEmbedder):
            name = "same"
            dimension = 1
            def embed(self, texts): return []
        class E2(BaseEmbedder):
            name = "same"
            dimension = 2
            def embed(self, texts): return []

        reg = PluginRegistry()
        reg.register_embedder(E1)
        reg.register_embedder(E2)
        # E2 should override E1
        assert reg.get_embedder("same") is E2

    def test_missing_name_raises(self):
        from core.plugins.interfaces import BaseEmbedder
        from core.plugins.registry import PluginRegistry

        class BadEmbedder(BaseEmbedder):
            # name not set (empty string from base class)
            dimension = 1
            def embed(self, texts): return []

        reg = PluginRegistry()
        with pytest.raises(ValueError, match="must define a 'name' attribute"):
            reg.register_embedder(BadEmbedder)


class TestPluginLoader:
    """Test plugin discovery and loading from directories."""

    def test_discover_empty_dir(self, tmp_path):
        from core.plugins.loader import discover_plugins
        result = discover_plugins(tmp_path)
        assert result == []

    def test_discover_nonexistent_dir(self, tmp_path):
        from core.plugins.loader import discover_plugins
        result = discover_plugins(tmp_path / "nonexistent")
        assert result == []

    def test_discover_py_files(self, tmp_path):
        from core.plugins.loader import discover_plugins
        (tmp_path / "embedder.py").write_text("# plugin")
        (tmp_path / "chunker.py").write_text("# plugin")
        (tmp_path / "__init__.py").write_text("# skip")
        (tmp_path / ".hidden.py").write_text("# skip")
        result = discover_plugins(tmp_path)
        names = [p.name for p in result]
        assert "embedder.py" in names
        assert "chunker.py" in names
        assert "__init__.py" not in names
        assert ".hidden.py" not in names

    def test_discover_package_dirs(self, tmp_path):
        from core.plugins.loader import discover_plugins
        pkg = tmp_path / "my_plugin"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("# plugin package")
        # Dir without __init__ should be skipped
        (tmp_path / "not_a_plugin").mkdir()
        result = discover_plugins(tmp_path)
        assert len(result) == 1
        assert result[0].name == "my_plugin"

    def test_load_plugin_file(self, tmp_path):
        from core.plugins.loader import load_plugin
        from core.plugins.registry import PluginRegistry

        plugin_code = '''
from core.plugins.interfaces import BaseEmbedder

class TestEmbedder(BaseEmbedder):
    name = "test_file_embedder"
    dimension = 5
    def embed(self, texts):
        return [[0.0] * 5] * len(texts)

def register(registry):
    registry.register_embedder(TestEmbedder)
'''
        plugin_file = tmp_path / "test_embedder.py"
        plugin_file.write_text(plugin_code)

        reg = PluginRegistry()
        result = load_plugin(plugin_file, reg)
        assert result is True
        assert reg.get_embedder("test_file_embedder") is not None

    def test_load_plugin_package(self, tmp_path):
        from core.plugins.loader import load_plugin
        from core.plugins.registry import PluginRegistry

        pkg = tmp_path / "test_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text('''
from core.plugins.interfaces import BaseLLMProvider

class TestLLM(BaseLLMProvider):
    name = "test_pkg_llm"
    def generate(self, messages, system_prompt=None, max_tokens=2000):
        return "test"

def register(registry):
    registry.register_llm_provider(TestLLM)
''')

        reg = PluginRegistry()
        result = load_plugin(pkg, reg)
        assert result is True
        assert reg.get_llm_provider("test_pkg_llm") is not None

    def test_load_plugin_no_register_function(self, tmp_path):
        from core.plugins.loader import load_plugin
        from core.plugins.registry import PluginRegistry

        (tmp_path / "no_register.py").write_text("x = 1\n")
        reg = PluginRegistry()
        result = load_plugin(tmp_path / "no_register.py", reg)
        assert result is False

    def test_load_plugin_import_error(self, tmp_path):
        from core.plugins.loader import load_plugin
        from core.plugins.registry import PluginRegistry

        (tmp_path / "broken.py").write_text("import nonexistent_module_xyz\n")
        reg = PluginRegistry()
        result = load_plugin(tmp_path / "broken.py", reg)
        assert result is False

    def test_load_plugin_register_raises(self, tmp_path):
        from core.plugins.loader import load_plugin
        from core.plugins.registry import PluginRegistry

        (tmp_path / "bad_register.py").write_text('''
def register(registry):
    raise RuntimeError("registration failed")
''')
        reg = PluginRegistry()
        result = load_plugin(tmp_path / "bad_register.py", reg)
        assert result is False

    def test_load_plugins_full(self, tmp_path):
        from core.plugins.loader import load_plugins
        from core.plugins.registry import PluginRegistry

        # Create two valid plugins and one broken
        (tmp_path / "good1.py").write_text('''
from core.plugins.interfaces import BaseReranker

class R1(BaseReranker):
    name = "r1"
    def predict(self, pairs): return [0.5] * len(pairs)

def register(registry):
    registry.register_reranker(R1)
''')
        (tmp_path / "good2.py").write_text('''
from core.plugins.interfaces import BaseChunker

class C1(BaseChunker):
    name = "c1"
    def chunk(self, elements, **kw): return ["chunk"]

def register(registry):
    registry.register_chunker(C1)
''')
        (tmp_path / "broken.py").write_text("raise ImportError('nope')\n")

        reg = load_plugins(tmp_path)
        assert reg.get_reranker("r1") is not None
        assert reg.get_chunker("c1") is not None
        assert reg.list_rerankers() == ["r1"]
        assert reg.list_chunkers() == ["c1"]

    def test_load_plugins_no_dir(self, tmp_path):
        from core.plugins.loader import load_plugins
        reg = load_plugins(tmp_path / "nonexistent")
        assert reg.list_all() == {
            "embedders": [], "chunkers": [], "parsers": [],
            "rerankers": [], "llm_providers": [], "merge_strategies": [],
        }

    def test_load_plugins_with_existing_registry(self, tmp_path):
        from core.plugins.interfaces import BaseEmbedder
        from core.plugins.loader import load_plugins
        from core.plugins.registry import PluginRegistry

        # Pre-register something
        class PreExisting(BaseEmbedder):
            name = "pre"
            dimension = 1
            def embed(self, texts): return []

        reg = PluginRegistry()
        reg.register_embedder(PreExisting)

        # Add a plugin
        (tmp_path / "new.py").write_text('''
from core.plugins.interfaces import BaseEmbedder
class New(BaseEmbedder):
    name = "new"
    dimension = 1
    def embed(self, texts): return []
def register(registry):
    registry.register_embedder(New)
''')

        result = load_plugins(tmp_path, registry=reg)
        assert result is reg  # same registry
        assert reg.list_embedders() == ["new", "pre"]
