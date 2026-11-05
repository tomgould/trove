"""
Hybrid searcher, tokenizer, BM25 and search caches.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock
import tempfile
import shutil


class TestHybridSearcher:
    """Tests for HybridSearcher."""

    @pytest.fixture
    def temp_index(self):
        """Create temporary index directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_searcher_initialization(self, temp_index):
        """Test searcher can be initialized."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)

        assert searcher.bm25_weight == 0.5
        assert searcher.vector_weight == 0.5
        assert searcher.bm25_index is None  # Not built yet

    def test_index_and_search(self, temp_index):
        """Test indexing chunks and searching."""
        from core.search import HybridSearcher
        from sentence_transformers import SentenceTransformer

        searcher = HybridSearcher(index_path=temp_index)

        # Create test chunks
        chunks = [
            {"id": "1", "content": "Python is a programming language", "metadata": {"source": "test"}},
            {"id": "2", "content": "JavaScript runs in the browser", "metadata": {"source": "test"}},
            {"id": "3", "content": "Rust is known for memory safety", "metadata": {"source": "test"}},
        ]

        # Create embeddings
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()

        # Index
        searcher.index_chunks(chunks, embeddings)

        # Verify stats
        stats = searcher.get_stats()
        assert stats["total_chunks"] == 3
        assert stats["bm25_indexed"] == 3

        # Search
        results = searcher.search("programming language", top_k=2)

        assert len(results) <= 2
        # Python chunk should rank highly
        found_python = any("Python" in r["content"] for r in results)
        assert found_python


class TestEnhancedTokenizer:
    """Tests for EnhancedTokenizer with stemming and synonyms."""

    def test_basic_tokenization(self):
        """Test tokenization without stemming."""
        from core.search import EnhancedTokenizer

        tokenizer = EnhancedTokenizer()
        tokens = tokenizer.tokenize("The quick brown fox jumps over the lazy dog")

        # Should remove stopwords
        assert "the" not in tokens
        assert "over" not in tokens
        # Should keep content words
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_porter_stemming(self):
        """Test Porter stemmer reduces words to stems."""
        from core.search import EnhancedTokenizer

        tokenizer = EnhancedTokenizer(stemmer="porter")
        tokens = tokenizer.tokenize("countries jurisdictions requirements")

        # Porter stems these words
        assert "countri" in tokens  # countries -> countri
        assert "jurisdict" in tokens  # jurisdictions -> jurisdict
        assert "requir" in tokens  # requirements -> requir

    def test_synonym_expansion(self):
        """Test synonym expansion in queries."""
        from core.search import EnhancedTokenizer

        synonyms = {
            "aml": ["anti-money laundering", "financial crime"],
            "kyc": ["know your customer"],
        }
        tokenizer = EnhancedTokenizer(synonyms=synonyms)

        # Query tokenization should expand synonyms
        tokens = tokenizer.tokenize_query("AML requirements")

        assert "aml" in tokens
        # Should include expanded terms
        assert "anti" in tokens or "money" in tokens or "laundering" in tokens

    def test_config_hash(self):
        """Test that config hash changes with settings."""
        from core.search import EnhancedTokenizer

        tok1 = EnhancedTokenizer()
        tok2 = EnhancedTokenizer(stemmer="porter")
        tok3 = EnhancedTokenizer(synonyms={"aml": ["anti-money laundering"]})

        # Different configs should have different hashes
        assert tok1.get_config_hash() != tok2.get_config_hash()
        assert tok1.get_config_hash() != tok3.get_config_hash()


class TestBM25Cache:
    """Tests for BM25 index caching."""

    @pytest.fixture
    def temp_index(self):
        """Create temporary index directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_cache_path(self, temp_index):
        """Test BM25 cache path generation."""
        from core.search import HybridSearcher, BM25_CACHE_FILENAME

        searcher = HybridSearcher(index_path=temp_index)
        cache_path = searcher._get_bm25_cache_path()

        assert cache_path == Path(temp_index) / BM25_CACHE_FILENAME

    def test_cache_save_and_load(self, temp_index):
        """Test BM25 cache save and load."""
        from core.search import HybridSearcher

        # Create and index some data
        searcher = HybridSearcher(index_path=temp_index)
        chunks = [
            {"id": "1", "content": "Test document one", "metadata": {"source": "test1"}},
            {"id": "2", "content": "Test document two", "metadata": {"source": "test2"}},
        ]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        # Cache should be saved
        cache_path = searcher._get_bm25_cache_path()
        assert cache_path.exists()

        # Close and create new searcher - should load from cache
        searcher.close()
        searcher2 = HybridSearcher(index_path=temp_index)

        assert searcher2.bm25_index is not None
        assert len(searcher2.bm25_chunks) == 2

    def test_cache_invalidation_on_content_change(self, temp_index):
        """Test that cache is invalidated when content changes."""
        from core.search import HybridSearcher

        # Create initial index
        searcher = HybridSearcher(index_path=temp_index)
        chunks = [{"id": "1", "content": "Original content", "metadata": {"source": "test"}}]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        original_hash = searcher._compute_corpus_hash()
        searcher.close()

        # Add more content
        searcher2 = HybridSearcher(index_path=temp_index)
        new_chunks = [{"id": "2", "content": "New content", "metadata": {"source": "test2"}}]
        new_embeddings = searcher2.embedder.encode([c["content"] for c in new_chunks]).tolist()
        searcher2.index_chunks(new_chunks, new_embeddings)

        new_hash = searcher2._compute_corpus_hash()

        # Hash should be different after adding content
        assert original_hash != new_hash

    def test_ephemeral_no_cache(self, temp_index):
        """Test that ephemeral searchers don't use cache."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, ephemeral=True)
        chunks = [{"id": "1", "content": "Test", "metadata": {"source": "test"}}]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        # Cache should not be created for ephemeral
        cache_path = searcher._get_bm25_cache_path()
        assert not cache_path.exists()


class TestErrorHandling:
    """Tests for search pipeline error handling."""

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_normalize_all_zero_scores(self, temp_index):
        """Test that normalize handles all-zero scores without division by zero."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)

        # Simulate all-zero BM25 results
        bm25_results = [
            ({"id": "1", "content": "a", "metadata": {}}, 0.0),
            ({"id": "2", "content": "b", "metadata": {}}, 0.0),
        ]
        vector_results = [
            ({"id": "1", "content": "a", "metadata": {}}, 0.5),
        ]

        # Should not raise
        merged = searcher._merge_results(bm25_results, vector_results)
        assert len(merged) == 2

        # BM25 scores should all be 0
        for chunk in merged:
            assert chunk["_bm25_score"] == 0.0

    def test_search_with_empty_index(self, temp_index):
        """Test search returns empty list on empty index."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        results = searcher.search("test query")
        assert results == []


# ---------------------------------------------------------------------------
# Stage 2: MCP Enhancement Tests
# ---------------------------------------------------------------------------

class TestSearcherCache:
    """Tests for SearcherCache LRU cache."""

    def test_init_default(self):
        """Test default cache initialization."""
        from mcp_server.server import SearcherCache

        cache = SearcherCache()
        assert cache.max_size == 3
        assert cache.current is None
        assert cache.current_name is None

    def test_init_custom_size(self):
        """Test custom cache size."""
        from mcp_server.server import SearcherCache

        cache = SearcherCache(max_size=5)
        assert cache.max_size == 5

    def test_get_miss(self):
        """Test get returns None for missing key."""
        from mcp_server.server import SearcherCache

        cache = SearcherCache()
        assert cache.get("nonexistent") is None

    def test_put_and_get(self):
        """Test put and get round-trip."""
        from mcp_server.server import SearcherCache
        from unittest.mock import MagicMock

        cache = SearcherCache(max_size=3)
        mock_client = MagicMock()
        mock_searcher = MagicMock()

        cache.put("test", mock_client, mock_searcher)
        result = cache.get("test")

        assert result is not None
        assert result[0] is mock_client
        assert result[1] is mock_searcher

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from mcp_server.server import SearcherCache
        from unittest.mock import MagicMock

        cache = SearcherCache(max_size=2)

        # Fill cache
        clients = {}
        searchers = {}
        for name in ["a", "b"]:
            clients[name] = MagicMock()
            searchers[name] = MagicMock()
            cache.put(name, clients[name], searchers[name])

        # Add third item - should evict "a" (LRU)
        client_c = MagicMock()
        searcher_c = MagicMock()
        cache.put("c", client_c, searcher_c)

        assert cache.get("a") is None  # evicted
        assert cache.get("b") is not None  # still there
        assert cache.get("c") is not None  # just added
        searchers["a"].close.assert_called_once()  # close was called on evicted

    def test_lru_access_order(self):
        """Test that get updates LRU access order."""
        from mcp_server.server import SearcherCache
        from unittest.mock import MagicMock

        cache = SearcherCache(max_size=2)

        for name in ["a", "b"]:
            cache.put(name, MagicMock(), MagicMock())

        # Access "a" to make it recently used
        cache.get("a")

        # Add "c" - should evict "b" (now LRU, not "a")
        cache.put("c", MagicMock(), MagicMock())

        assert cache.get("a") is not None  # accessed recently, not evicted
        assert cache.get("b") is None  # evicted
        assert cache.get("c") is not None  # just added

    def test_current_property(self):
        """Test current property tracks current client."""
        from mcp_server.server import SearcherCache
        from unittest.mock import MagicMock

        cache = SearcherCache()
        mock_client = MagicMock()
        mock_searcher = MagicMock()

        cache.put("test", mock_client, mock_searcher)
        cache.current_name = "test"

        assert cache.current is not None
        assert cache.current[0] is mock_client

    def test_current_none_when_not_set(self):
        """Test current is None when no current name set."""
        from mcp_server.server import SearcherCache
        from unittest.mock import MagicMock

        cache = SearcherCache()
        cache.put("test", MagicMock(), MagicMock())
        # current_name not set
        assert cache.current is None

    def test_close_all(self):
        """Test close_all cleans up everything."""
        from mcp_server.server import SearcherCache
        from unittest.mock import MagicMock

        cache = SearcherCache()
        searchers = []
        for name in ["a", "b"]:
            s = MagicMock()
            searchers.append(s)
            cache.put(name, MagicMock(), s)

        cache.current_name = "a"
        cache.close_all()

        for s in searchers:
            s.close.assert_called_once()
        assert cache.current is None
        assert cache.current_name is None
        assert cache.get("a") is None

    def test_put_replaces_existing(self):
        """Test that putting same name replaces and closes old searcher."""
        from mcp_server.server import SearcherCache
        from unittest.mock import MagicMock

        cache = SearcherCache()
        old_searcher = MagicMock()
        cache.put("test", MagicMock(), old_searcher)

        new_searcher = MagicMock()
        cache.put("test", MagicMock(), new_searcher)

        old_searcher.close.assert_called_once()
        result = cache.get("test")
        assert result[1] is new_searcher


class TestSearchPresets:
    """Tests for search mode presets."""

    def test_preset_definitions(self):
        """Test all expected presets are defined."""
        from mcp_server.server import SEARCH_PRESETS

        expected = {"precise", "broad", "code_first", "exhaustive", "diverse"}
        assert set(SEARCH_PRESETS.keys()) == expected

    def test_precise_preset(self):
        """Test precise preset has high BM25 weight."""
        from mcp_server.server import SEARCH_PRESETS

        p = SEARCH_PRESETS["precise"]
        assert p["bm25_weight"] == 0.7
        assert p["vector_weight"] == 0.3
        assert p["_search_params"]["top_k"] == 10

    def test_broad_preset(self):
        """Test broad preset has high vector weight."""
        from mcp_server.server import SEARCH_PRESETS

        p = SEARCH_PRESETS["broad"]
        assert p["bm25_weight"] == 0.3
        assert p["vector_weight"] == 0.7
        assert p["_search_params"]["top_k"] == 30

    def test_apply_and_restore(self):
        """Test _apply_preset and _restore_preset cycle."""
        from mcp_server.server import _apply_preset, _restore_preset
        from unittest.mock import MagicMock

        searcher = MagicMock()
        searcher.bm25_weight = 0.5
        searcher.vector_weight = 0.5
        searcher.merge_strategy = "weighted"
        searcher.mmr_lambda = None

        saved, params = _apply_preset(searcher, "precise")

        # Attributes should be changed
        assert searcher.bm25_weight == 0.7
        assert searcher.vector_weight == 0.3
        assert params["top_k"] == 10

        # Saved should contain originals
        assert saved["bm25_weight"] == 0.5
        assert saved["vector_weight"] == 0.5

        # Restore
        _restore_preset(searcher, saved)
        assert searcher.bm25_weight == 0.5
        assert searcher.vector_weight == 0.5

    def test_unknown_mode_returns_empty(self):
        """Test unknown mode returns empty dicts."""
        from mcp_server.server import _apply_preset
        from unittest.mock import MagicMock

        searcher = MagicMock()
        saved, params = _apply_preset(searcher, "nonexistent")
        assert saved == {}
        assert params == {}


class TestInputValidation:
    """Tests for search query sanitization."""

    def _make_searcher(self):
        """Create a minimal HybridSearcher mock for testing _sanitize_query."""
        from core.search import HybridSearcher
        # We need to test _sanitize_query without a full index
        # Create a minimal instance by using __new__ to skip __init__
        searcher = object.__new__(HybridSearcher)
        return searcher

    def test_sanitize_empty_query(self):
        """Empty/whitespace queries return None."""
        searcher = self._make_searcher()
        assert searcher._sanitize_query("") is None
        assert searcher._sanitize_query("   ") is None
        assert searcher._sanitize_query(None) is None

    def test_sanitize_long_query(self):
        """Queries >10000 chars are truncated."""
        searcher = self._make_searcher()
        long_query = "a" * 15_000
        result = searcher._sanitize_query(long_query)
        assert result is not None
        assert len(result) == 10_000

    def test_sanitize_control_chars(self):
        """Null bytes and control chars are removed."""
        searcher = self._make_searcher()
        result = searcher._sanitize_query("hello\x00world\x07test")
        assert result == "helloworld test" or result == "helloworldtest"
        assert "\x00" not in result
        assert "\x07" not in result

    def test_sanitize_normal_query(self):
        """Normal query passes through unchanged."""
        searcher = self._make_searcher()
        result = searcher._sanitize_query("What is hybrid search?")
        assert result == "What is hybrid search?"


class TestSearchIntegration:
    """Integration test using the real trag client."""

    def test_search_integration(self):
        """Load real trag client, run a search, verify results."""
        from core.client import load_client
        from core.search import HybridSearcher

        try:
            client = load_client("trag")
        except Exception:
            pytest.skip("trag client not available")

        search_config = client.search_config
        embedding_config = client.embedding_config

        try:
            searcher = HybridSearcher(
                index_path=str(client.index_path),
                collection_name=client.machine_name,
                embedding_config=embedding_config,
                tokenizer_config=client.tokenization_config,
                bm25_weight=search_config.get("bm25_weight", 0.5),
                vector_weight=search_config.get("vector_weight", 0.5),
                min_score=search_config.get("min_score"),
                merge_strategy=search_config.get("merge_strategy", "weighted"),
                rrf_k=search_config.get("rrf_k", 60),
                mmr_lambda=search_config.get("mmr_lambda"),
            )
        except Exception:
            pytest.skip("trag client index not available")

        # Verify the searcher has data
        stats = searcher.get_stats()
        if stats.get("total_chunks", 0) == 0:
            pytest.skip("trag client has no indexed data")

        results = searcher.search("hybrid search", top_k=5)

        assert isinstance(results, list)
        assert len(results) > 0, "Expected at least one result for 'hybrid search'"

        # Verify result structure
        for chunk in results:
            assert "content" in chunk
            assert "metadata" in chunk
            assert "id" in chunk


class TestEmbeddingBenchmark:
    """Tests for 9.3: Embedding model benchmark script."""

    def test_benchmark_format_results(self):
        """Test formatting benchmark results as table."""
        from scripts.benchmark_embeddings import format_results_table
        results = [
            {
                "model": "all-MiniLM-L6-v2",
                "dimension": 384,
                "mrr": 0.85,
                "hit_at_5": 0.95,
                "hit_at_10": 0.98,
                "ndcg": 0.80,
                "precision_at_5": 0.60,
                "embed_time_per_chunk_ms": 1.2,
                "total_time_s": 10.5,
                "num_chunks": 100,
                "num_queries": 5,
            },
        ]
        table = format_results_table(results)
        assert "all-MiniLM-L6-v2" in table
        assert "0.850" in table

    def test_benchmark_format_empty_results(self):
        """Test formatting empty results."""
        from scripts.benchmark_embeddings import format_results_table
        assert format_results_table([]) == "No results."

    def test_benchmark_run_single_model(self):
        """Test benchmarking a single model with mock data."""
        from scripts.benchmark_embeddings import benchmark_model
        from core.evaluation import TestCase
        import numpy as np

        chunks = [
            {"id": "1", "content": "Python is a programming language.", "metadata": {}},
            {"id": "2", "content": "JavaScript runs in the browser.", "metadata": {}},
            {"id": "3", "content": "Rust focuses on memory safety.", "metadata": {}},
        ]
        eval_cases = [
            {
                "query": "programming language",
                "expected_chunks": ["Python"],
                "test_case": TestCase(id="t1", query="programming language", expected_chunks=["Python"]),
            },
        ]

        # Use the real (small) model for testing
        result = benchmark_model("all-MiniLM-L6-v2", chunks, eval_cases)

        assert result["model"] == "all-MiniLM-L6-v2"
        assert result["dimension"] == 384
        assert "mrr" in result
        assert "embed_time_per_chunk_ms" in result
        assert result["num_chunks"] == 3
        assert result["num_queries"] == 1

    def test_benchmark_missing_eval_dataset(self):
        """Test benchmark warns when no eval dataset exists."""
        from scripts.benchmark_embeddings import load_client_data
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.path = Path("/nonexistent/path")
        mock_client.machine_name = "fake"

        with pytest.raises(FileNotFoundError):
            load_client_data(mock_client)


class TestSearchCache:
    """Tests for search result caching."""

    def test_cache_put_and_get(self):
        """Test basic cache put and get."""
        from core.search import SearchCache

        cache = SearchCache(max_size=10, ttl=300)
        cache.put("key1", ["result1"])
        assert cache.get("key1") == ["result1"]

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from core.search import SearchCache

        cache = SearchCache(max_size=10, ttl=300)
        assert cache.get("nonexistent") is None

    def test_cache_ttl_expiry(self):
        """Test cache entries expire after TTL."""
        import time as _time
        from core.search import SearchCache

        cache = SearchCache(max_size=10, ttl=0.1)  # 100ms TTL
        cache.put("key1", ["result1"])
        assert cache.get("key1") == ["result1"]
        _time.sleep(0.15)
        assert cache.get("key1") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when max_size exceeded."""
        from core.search import SearchCache

        cache = SearchCache(max_size=2, ttl=300)
        cache.put("key1", ["r1"])
        cache.put("key2", ["r2"])
        cache.put("key3", ["r3"])  # Should evict key1
        assert cache.get("key1") is None
        assert cache.get("key2") == ["r2"]
        assert cache.get("key3") == ["r3"]

    def test_cache_lru_access_refreshes(self):
        """Test accessing an entry refreshes its LRU position."""
        from core.search import SearchCache

        cache = SearchCache(max_size=2, ttl=300)
        cache.put("key1", ["r1"])
        cache.put("key2", ["r2"])
        cache.get("key1")  # Refresh key1
        cache.put("key3", ["r3"])  # Should evict key2 (not key1)
        assert cache.get("key1") == ["r1"]
        assert cache.get("key2") is None

    def test_cache_clear(self):
        """Test cache clear removes all entries."""
        from core.search import SearchCache

        cache = SearchCache(max_size=10, ttl=300)
        cache.put("key1", ["r1"])
        cache.put("key2", ["r2"])
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0
        assert cache.get("key1") is None

    def test_cache_disabled_when_ttl_zero(self):
        """Test cache is effectively disabled when TTL is 0."""
        from core.search import SearchCache

        cache = SearchCache(max_size=10, ttl=0)
        cache.put("key1", ["r1"])
        assert cache.get("key1") is None  # Immediately expired

    def test_cache_size_property(self):
        """Test cache size tracking."""
        from core.search import SearchCache

        cache = SearchCache(max_size=10, ttl=300)
        assert cache.size == 0
        cache.put("key1", ["r1"])
        assert cache.size == 1
        cache.put("key2", ["r2"])
        assert cache.size == 2

    def test_cache_config_defaults(self):
        """Test cache config defaults in DEFAULT_CONFIG."""
        from core.client import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["search"]["cache_ttl"] == 300
        assert DEFAULT_CONFIG["search"]["cache_max_size"] == 128

    def test_client_search_config_includes_cache(self):
        """Test Client.search_config includes cache keys."""
        from core.client import Client
        from pathlib import Path

        client = Client(Path("/tmp/fake"), {"machine_name": "test"})
        cfg = client.search_config
        assert cfg["cache_ttl"] == 300
        assert cfg["cache_max_size"] == 128

    def test_cache_config_in_known_search_keys(self):
        """Test cache keys are in _KNOWN_SEARCH_KEYS."""
        from core.client import _KNOWN_SEARCH_KEYS

        assert "cache_ttl" in _KNOWN_SEARCH_KEYS
        assert "cache_max_size" in _KNOWN_SEARCH_KEYS

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_searcher_has_cache(self, temp_index):
        """Test HybridSearcher initializes with a cache."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, cache_ttl=300, cache_max_size=64)
        assert searcher._cache is not None
        assert searcher._cache.max_size == 64
        assert searcher._cache.ttl == 300

    def test_searcher_cache_disabled(self, temp_index):
        """Test cache is disabled when ttl=0."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, cache_ttl=0)
        assert searcher._cache.ttl == 0

    def test_search_quality_cache_fields(self):
        """Test SearchQuality has cache_hit and cache_size fields."""
        from core.search import SearchQuality

        q = SearchQuality()
        assert q.cache_hit is False
        assert q.cache_size == 0

    def test_cache_key_generation(self, temp_index):
        """Test cache key includes query and config params."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        key1 = searcher._build_cache_key("test query", 10, None)
        key2 = searcher._build_cache_key("test query", 10, None)
        key3 = searcher._build_cache_key("different query", 10, None)
        key4 = searcher._build_cache_key("test query", 20, None)
        assert key1 == key2  # Same params = same key
        assert key1 != key3  # Different query = different key
        assert key1 != key4  # Different top_k = different key

    def test_cache_key_includes_filters(self, temp_index):
        """Test cache key differs when filters are applied."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        key_no_filter = searcher._build_cache_key("test", 10, None)
        key_python = searcher._build_cache_key("test", 10, None, languages=["python"])
        key_js = searcher._build_cache_key("test", 10, None, languages=["javascript"])
        key_code = searcher._build_cache_key("test", 10, None, chunk_types=["code"])
        assert key_no_filter != key_python  # Filter vs no filter
        assert key_python != key_js  # Different language filter
        assert key_no_filter != key_code  # Chunk type filter

    def test_cache_invalidated_on_rebuild(self, temp_index):
        """Test cache is cleared when BM25 index is rebuilt."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, cache_ttl=300)
        searcher._cache.put("test_key", ["results"])
        assert searcher._cache.size == 1
        searcher._rebuild_bm25_index(save_cache=False)
        assert searcher._cache.size == 0


###############################################################################
# Search bug fix tests
###############################################################################

class TestSearchBugFix:
    """Tests for the get_response → generate bug fix in search.py."""

    def test_hyde_calls_generate_not_get_response(self):
        """HyDE calls llm_provider.generate(), not get_response()."""
        from core.search import HybridSearcher
        from unittest.mock import MagicMock

        mock_provider = MagicMock()
        mock_provider.generate.return_value = "A hypothetical document about testing."

        # Verify that generate is the method called
        # (If the code still called get_response, this mock wouldn't return anything useful)
        searcher = HybridSearcher.__new__(HybridSearcher)
        searcher.llm_provider = mock_provider

        result = searcher._generate_hypothetical_document("test query")

        mock_provider.generate.assert_called_once()
        assert not hasattr(mock_provider.generate, 'get_response') or not mock_provider.get_response.called
        assert result == "A hypothetical document about testing."

    def test_multi_query_calls_generate_not_get_response(self):
        """Multi-query lambda calls generate(), not get_response()."""
        # Verify the source code directly — the lambda should call .generate()
        import inspect
        from core.search import HybridSearcher
        source = inspect.getsource(HybridSearcher._multi_query_search)
        assert "self.llm_provider.generate(" in source
        assert "self.llm_provider.get_response(" not in source


###############################################################################
# merge_search_results tests
###############################################################################

class TestMergeSearchResults:
    """Tests for the merge_search_results utility function."""

    def test_merges_main_and_conversation_results(self):
        """Basic merge of two result sets."""
        from core.search import merge_search_results

        main = [
            {"id": "a", "content": "main A", "_rerank_score": 0.9},
            {"id": "b", "content": "main B", "_rerank_score": 0.7},
        ]
        conv = [
            {"id": "c", "content": "conv C", "_rerank_score": 0.5},
        ]

        merged = merge_search_results(main, conv)
        assert len(merged) == 3
        # Sorted by score descending
        assert merged[0]["id"] == "a"

    def test_deduplicates_by_id(self):
        """Duplicate IDs across main and conv keep only main version."""
        from core.search import merge_search_results

        main = [{"id": "x", "content": "main X", "_rerank_score": 0.9}]
        conv = [{"id": "x", "content": "conv X", "_rerank_score": 0.3}]

        merged = merge_search_results(main, conv)
        assert len(merged) == 1
        assert merged[0]["content"] == "main X"

    def test_boosts_conversation_results(self):
        """Conversation results get a score boost."""
        from core.search import merge_search_results

        main = [{"id": "a", "_rerank_score": 0.5}]
        conv = [{"id": "b", "_rerank_score": 0.5}]

        merged = merge_search_results(main, conv, boost_conversation=0.2)
        conv_result = [r for r in merged if r["id"] == "b"][0]
        assert conv_result["_rerank_score"] == pytest.approx(0.7)
        assert conv_result["_from_conversation"] is True

    def test_empty_inputs(self):
        """Works with empty lists."""
        from core.search import merge_search_results

        assert merge_search_results([], []) == []
        assert len(merge_search_results([{"id": "a", "_rerank_score": 0.5}], [])) == 1
        assert len(merge_search_results([], [{"id": "a", "_rerank_score": 0.5}])) == 1

    def test_chunks_without_id_ignored(self):
        """Chunks without 'id' key are skipped."""
        from core.search import merge_search_results

        main = [{"content": "no id", "_rerank_score": 0.9}]
        conv = [{"content": "also no id", "_rerank_score": 0.5}]

        merged = merge_search_results(main, conv)
        assert len(merged) == 0

    def test_boosts_combined_score_too(self):
        """Boost also applies to _combined_score when present."""
        from core.search import merge_search_results

        conv = [{"id": "a", "_combined_score": 0.4}]
        merged = merge_search_results([], conv, boost_conversation=0.1)
        assert merged[0]["_combined_score"] == pytest.approx(0.5)


###############################################################################
# SearchCache edge case tests
###############################################################################

class TestSearchCacheEdgeCases:
    """Additional edge case tests for SearchCache."""

    def test_put_overwrites_existing_key(self):
        from core.search import SearchCache
        cache = SearchCache(max_size=10, ttl=300)
        cache.put("key1", "value1")
        cache.put("key1", "value2")
        assert cache.get("key1") == "value2"
        assert cache.size == 1

    def test_lru_eviction_order(self):
        """Least recently used item is evicted when cache is full."""
        from core.search import SearchCache
        cache = SearchCache(max_size=3, ttl=300)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # Access 'a' to make it recently used
        cache.get("a")
        # Add 'd' — should evict 'b' (least recently used)
        cache.put("d", 4)
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("d") == 4

    def test_ttl_expiry(self):
        """Expired entries return None."""
        from core.search import SearchCache
        import time as time_mod

        cache = SearchCache(max_size=10, ttl=0.01)  # 10ms TTL
        cache.put("key", "value")
        time_mod.sleep(0.02)
        assert cache.get("key") is None

    def test_clear(self):
        from core.search import SearchCache
        cache = SearchCache(max_size=10, ttl=300)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    def test_size_property(self):
        from core.search import SearchCache
        cache = SearchCache(max_size=10, ttl=300)
        assert cache.size == 0
        cache.put("a", 1)
        assert cache.size == 1
        cache.put("b", 2)
        assert cache.size == 2


# ───────────────────────────────────────────────────────────────────────
# Performance & Quality Hardening Tests (Pre-Stage improvements)
# ───────────────────────────────────────────────────────────────────────


class TestCorpusHash:
    """Tests for BM25 corpus hash computation."""

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_corpus_hash_deterministic(self, temp_index):
        """Same docs produce same hash regardless of insertion order."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        chunks_a = [
            {"id": "a", "content": "Alpha document", "metadata": {"source": "a"}},
            {"id": "b", "content": "Beta document", "metadata": {"source": "b"}},
        ]
        embeddings_a = searcher.embedder.encode([c["content"] for c in chunks_a]).tolist()
        searcher.index_chunks(chunks_a, embeddings_a)
        hash1 = searcher._compute_corpus_hash()
        searcher.close()

        # Create new index with same docs in reverse order
        tmpdir2 = tempfile.mkdtemp()
        try:
            searcher2 = HybridSearcher(index_path=tmpdir2)
            chunks_b = list(reversed(chunks_a))
            embeddings_b = searcher2.embedder.encode([c["content"] for c in chunks_b]).tolist()
            searcher2.index_chunks(chunks_b, embeddings_b)
            hash2 = searcher2._compute_corpus_hash()
            searcher2.close()
            assert hash1 == hash2
        finally:
            shutil.rmtree(tmpdir2)

    def test_corpus_hash_changes_on_content_change(self, temp_index):
        """Modifying a doc changes the hash."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        chunks = [{"id": "1", "content": "Original text here", "metadata": {"source": "test"}}]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)
        hash_before = searcher._compute_corpus_hash()

        # Update the same doc with different content
        updated_emb = searcher.embedder.encode(["Modified text here"]).tolist()
        searcher.collection.upsert(
            ids=["1"], documents=["Modified text here"],
            embeddings=updated_emb, metadatas=[{"source": "test"}],
        )
        searcher._last_corpus_hash = None
        hash_after = searcher._compute_corpus_hash()
        assert hash_before != hash_after

    def test_corpus_hash_cached_in_last_corpus_hash(self, temp_index):
        """_compute_corpus_hash stores result in _last_corpus_hash."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        chunks = [{"id": "1", "content": "Test content", "metadata": {"source": "test"}}]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)
        searcher._last_corpus_hash = None

        result = searcher._compute_corpus_hash()
        assert searcher._last_corpus_hash == result
        assert result != ""

    def test_bm25_cache_doc_count_mismatch_skips_hash(self, temp_index):
        """Fast-path rejection when doc count doesn't match."""
        from core.search import HybridSearcher, TOKENIZER_VERSION

        searcher = HybridSearcher(index_path=temp_index)
        chunks = [
            {"id": "1", "content": "First document", "metadata": {"source": "a"}},
            {"id": "2", "content": "Second document", "metadata": {"source": "b"}},
        ]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        # Manually write a cache with wrong doc count using the internal save
        # then tamper with the saved file
        import json as _json_test
        cache_path = searcher._get_bm25_cache_path()
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = _json_test.load(f)
        cache_data["doc_count"] = 999
        with open(cache_path, "w", encoding="utf-8") as f:
            _json_test.dump(cache_data, f)

        # _load_bm25_cache should return False without computing hash
        searcher.bm25_index = None
        searcher.bm25_chunks = []
        searcher._last_corpus_hash = None
        result = searcher._load_bm25_cache()
        assert result is False
        # Hash should NOT have been computed (fast path)
        assert searcher._last_corpus_hash is None


class TestBM25SkipRebuild:
    """Tests for skipping BM25 rebuild when cache is valid."""

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_bm25_skip_rebuild_when_cache_valid(self, temp_index):
        """Calling _rebuild_bm25_index with valid cache skips work."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        chunks = [
            {"id": "1", "content": "Document one test", "metadata": {"source": "a"}},
            {"id": "2", "content": "Document two test", "metadata": {"source": "b"}},
        ]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        # Cache is now saved. Clear in-memory state.
        searcher.bm25_index = None
        searcher.bm25_chunks = []

        # Rebuild should load from cache (no actual rebuild)
        searcher._rebuild_bm25_index(save_cache=True)
        assert searcher.bm25_index is not None
        assert len(searcher.bm25_chunks) == 2

    def test_rebuild_bm25_skips_cache_check_when_flagged(self, temp_index):
        """skip_cache_check=True forces a full rebuild."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        chunks = [{"id": "1", "content": "Test document", "metadata": {"source": "test"}}]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        # Even with a valid cache, skip_cache_check=True forces rebuild
        rebuild_called = False
        original_load = searcher._load_bm25_cache

        def tracking_load():
            nonlocal rebuild_called
            rebuild_called = True
            return original_load()

        searcher._load_bm25_cache = tracking_load
        searcher._rebuild_bm25_index(save_cache=True, skip_cache_check=True)
        assert not rebuild_called  # Cache load should have been skipped

    def test_rebuild_proceeds_when_no_cache(self, temp_index):
        """Rebuild proceeds normally when no cache file exists."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        chunks = [{"id": "1", "content": "Fresh document", "metadata": {"source": "test"}}]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        # Delete cache
        searcher.clear_bm25_cache()
        searcher.bm25_index = None
        searcher.bm25_chunks = []

        # Rebuild should proceed and recreate cache
        searcher._rebuild_bm25_index(save_cache=True)
        assert searcher.bm25_index is not None
        assert len(searcher.bm25_chunks) == 1
        assert searcher._get_bm25_cache_path().exists()


class TestCorpusHashSampling:
    """Tests for BM25 corpus hash sampling on large corpora."""

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_small_corpus_uses_full_hash(self, temp_index):
        """Corpora below threshold use full hash (existing behavior)."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        chunks = [
            {"id": "1", "content": "Doc one", "metadata": {"source": "a"}},
            {"id": "2", "content": "Doc two", "metadata": {"source": "b"}},
        ]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)
        h = searcher._compute_corpus_hash()
        assert h != ""
        assert len(h) == 32  # MD5 hex digest

    def test_corpus_hash_includes_count(self, temp_index):
        """Hash changes when doc count changes (even with sampling)."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        chunks = [{"id": "1", "content": "Doc one", "metadata": {"source": "a"}}]
        emb = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, emb)
        hash1 = searcher._compute_corpus_hash()

        # Add another doc
        chunks2 = [{"id": "2", "content": "Doc two", "metadata": {"source": "b"}}]
        emb2 = searcher.embedder.encode([c["content"] for c in chunks2]).tolist()
        searcher.index_chunks(chunks2, emb2)
        searcher._last_corpus_hash = None
        hash2 = searcher._compute_corpus_hash()

        assert hash1 != hash2
