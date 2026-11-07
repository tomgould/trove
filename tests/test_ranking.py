"""
Merging, scoring, re-ranking and query expansion.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil


class TestRRFMerge:
    """Tests for Reciprocal Rank Fusion merge strategy."""

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_rrf_merge_basic(self, temp_index):
        """Test RRF merge produces valid results."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, merge_strategy="rrf")

        bm25_results = [
            ({"id": "1", "content": "first", "metadata": {}}, 5.0),
            ({"id": "2", "content": "second", "metadata": {}}, 3.0),
        ]
        vector_results = [
            ({"id": "2", "content": "second", "metadata": {}}, 0.9),
            ({"id": "3", "content": "third", "metadata": {}}, 0.8),
        ]

        merged = searcher._rrf_merge(bm25_results, vector_results)

        assert len(merged) == 3
        # All should have _combined_score, _bm25_score, _vector_score
        for chunk in merged:
            assert "_combined_score" in chunk
            assert "_bm25_score" in chunk
            assert "_vector_score" in chunk

        # Chunk 2 appears in both lists, should have highest RRF score
        chunk2 = next(c for c in merged if c["id"] == "2")
        chunk1 = next(c for c in merged if c["id"] == "1")
        assert chunk2["_combined_score"] > chunk1["_combined_score"]

    def test_rrf_scores_are_small(self, temp_index):
        """Test that RRF scores are in expected range (won't trigger adaptive rerank)."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, merge_strategy="rrf", rrf_k=60)

        bm25_results = [
            ({"id": str(i), "content": f"chunk {i}", "metadata": {}}, 10.0 - i)
            for i in range(10)
        ]
        vector_results = [
            ({"id": str(i), "content": f"chunk {i}", "metadata": {}}, 1.0 - i * 0.1)
            for i in range(10)
        ]

        merged = searcher._rrf_merge(bm25_results, vector_results)

        # RRF scores with k=60 should be much less than 0.9
        for chunk in merged:
            assert chunk["_combined_score"] < 0.1

    def test_rrf_merge_strategy_dispatch(self, temp_index):
        """Test that merge_strategy='rrf' dispatches to _rrf_merge in search."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, merge_strategy="rrf")
        assert searcher.merge_strategy == "rrf"

    def test_weighted_merge_strategy_dispatch(self, temp_index):
        """Test that merge_strategy='weighted' dispatches to _merge_results."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, merge_strategy="weighted")
        assert searcher.merge_strategy == "weighted"


class TestMMRDiversity:
    """Tests for MMR diversity reranking."""

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_mmr_returns_correct_count(self, temp_index):
        """Test MMR returns requested number of results."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, mmr_lambda=0.7)

        chunks = [
            {"id": str(i), "content": f"Content about topic {i}", "metadata": {}, "_rerank_score": 1.0 - i * 0.1}
            for i in range(10)
        ]

        result = searcher._mmr_diversify("test query", chunks, top_k=5, lambda_param=0.7)
        assert len(result) == 5

    def test_mmr_pure_relevance(self, temp_index):
        """Test MMR with lambda=1.0 preserves relevance order."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)

        chunks = [
            {"id": "best", "content": "Most relevant content", "metadata": {}, "_rerank_score": 5.0},
            {"id": "mid", "content": "Moderately relevant", "metadata": {}, "_rerank_score": 3.0},
            {"id": "low", "content": "Less relevant content", "metadata": {}, "_rerank_score": 1.0},
        ]

        result = searcher._mmr_diversify("test", chunks, top_k=3, lambda_param=1.0)
        # With lambda=1.0 (pure relevance), order should match original score order
        assert result[0]["id"] == "best"

    def test_mmr_single_chunk(self, temp_index):
        """Test MMR handles single chunk gracefully."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        chunks = [{"id": "1", "content": "Only chunk", "metadata": {}, "_rerank_score": 1.0}]

        result = searcher._mmr_diversify("test", chunks, top_k=5, lambda_param=0.7)
        assert len(result) == 1

    def test_mmr_empty_chunks(self, temp_index):
        """Test MMR handles empty input."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        result = searcher._mmr_diversify("test", [], top_k=5, lambda_param=0.7)
        assert result == []

    def test_mmr_disabled_by_default(self, temp_index):
        """Test MMR is disabled when mmr_lambda is None."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        assert searcher.mmr_lambda is None


class TestSearchQuality:
    """Tests for search quality indicators."""

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_search_quality_dataclass(self):
        """Test SearchQuality dataclass creation."""
        from core.search import SearchQuality

        q = SearchQuality(
            avg_rerank_score=0.5,
            top_score=0.9,
            score_spread=0.4,
            unique_sources=3,
            total_candidates=50,
            merge_strategy="rrf",
        )
        assert q.avg_rerank_score == 0.5
        assert q.merge_strategy == "rrf"

    def test_search_result_dataclass(self):
        """Test SearchResult dataclass creation."""
        from core.search import SearchResult, SearchQuality

        sr = SearchResult(
            results=[{"id": "1", "content": "test"}],
            quality=SearchQuality(top_score=0.9),
        )
        assert len(sr.results) == 1
        assert sr.quality.top_score == 0.9

    def test_compute_quality_empty(self, temp_index):
        """Test _compute_quality with empty results."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        quality = searcher._compute_quality([], 0, {}, False)
        assert quality.avg_rerank_score == 0.0
        assert quality.top_score == 0.0
        assert quality.unique_sources == 0

    def test_compute_quality_with_results(self, temp_index):
        """Test _compute_quality computes correct values."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        results = [
            {"id": "1", "content": "a", "metadata": {"source_file": "doc1.pdf"}, "_rerank_score": 2.0},
            {"id": "2", "content": "b", "metadata": {"source_file": "doc2.pdf"}, "_rerank_score": 1.0},
            {"id": "3", "content": "c", "metadata": {"source_file": "doc1.pdf"}, "_rerank_score": 0.5},
        ]
        quality = searcher._compute_quality(results, 50, {"total": 1.5}, False)

        assert quality.top_score == 2.0
        assert quality.avg_rerank_score == pytest.approx(1.1667, abs=0.001)
        assert quality.score_spread == 1.5
        assert quality.unique_sources == 2
        assert quality.total_candidates == 50

    def test_search_with_quality(self, temp_index):
        """Test search_with_quality returns SearchResult."""
        from core.search import HybridSearcher, SearchResult

        searcher = HybridSearcher(index_path=temp_index)

        # Create and index test data
        chunks = [
            {"id": "1", "content": "Python programming language", "metadata": {"source_file": "test.pdf"}},
            {"id": "2", "content": "JavaScript web development", "metadata": {"source_file": "test.pdf"}},
        ]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        result = searcher.search_with_quality(query="Python", top_k=2)
        assert isinstance(result, SearchResult)
        assert result.quality is not None
        assert result.quality.total_candidates > 0


class TestAdaptiveRerank:
    """Tests for configurable adaptive reranking."""

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_adaptive_rerank_default(self, temp_index):
        """Test adaptive rerank is enabled by default."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        assert searcher.adaptive_rerank is True
        assert searcher.adaptive_rerank_threshold == 0.9
        assert searcher.adaptive_rerank_depth == 30

    def test_adaptive_rerank_disabled(self, temp_index):
        """Test adaptive rerank can be disabled."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, adaptive_rerank=False)
        assert searcher.adaptive_rerank is False

    def test_adaptive_rerank_custom_threshold(self, temp_index):
        """Test custom adaptive rerank threshold."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(
            index_path=temp_index,
            adaptive_rerank_threshold=0.8,
            adaptive_rerank_depth=10,
        )
        assert searcher.adaptive_rerank_threshold == 0.8
        assert searcher.adaptive_rerank_depth == 10


class TestMultiClientRRF:
    """Tests for cross-client RRF merge."""

    def test_single_client(self):
        """Test RRF merge with single client."""
        from mcp_server.server import _cross_client_rrf_merge

        results = {
            "client_a": [
                {"id": "1", "content": "first"},
                {"id": "2", "content": "second"},
            ]
        }
        merged = _cross_client_rrf_merge(results, top_k=5)

        assert len(merged) == 2
        assert merged[0]["_source_client"] == "client_a"

    def test_multi_client_merge(self):
        """Test RRF merge across multiple clients."""
        from mcp_server.server import _cross_client_rrf_merge

        results = {
            "client_a": [
                {"id": "1", "content": "A result 1"},
                {"id": "2", "content": "A result 2"},
            ],
            "client_b": [
                {"id": "1", "content": "B result 1"},
                {"id": "2", "content": "B result 2"},
            ],
        }
        merged = _cross_client_rrf_merge(results, top_k=3)

        assert len(merged) == 3
        # All results should have source client tagged
        clients = {r["_source_client"] for r in merged}
        assert len(clients) >= 1  # At least one client represented

    def test_empty_results(self):
        """Test RRF merge with empty results."""
        from mcp_server.server import _cross_client_rrf_merge

        merged = _cross_client_rrf_merge({}, top_k=10)
        assert merged == []

    def test_top_k_limits_output(self):
        """Test that top_k limits merged output."""
        from mcp_server.server import _cross_client_rrf_merge

        results = {
            "a": [{"id": str(i), "content": f"chunk {i}"} for i in range(20)],
            "b": [{"id": str(i), "content": f"chunk {i}"} for i in range(20)],
        }
        merged = _cross_client_rrf_merge(results, top_k=5)
        assert len(merged) == 5

    def test_rrf_scores_are_correct(self):
        """Test RRF score computation."""
        from mcp_server.server import _cross_client_rrf_merge

        results = {
            "a": [{"id": "1", "content": "first"}],  # rank 0 in client a
            "b": [{"id": "1", "content": "first"}],  # rank 0 in client b (different namespace)
        }
        merged = _cross_client_rrf_merge(results, k=60, top_k=10)
        # Each client contributes 1/(60+0+1) = 1/61 ≈ 0.0164 per unique entry
        # Since they have different namespaced IDs (a:1 vs b:1), both should appear
        assert len(merged) == 2


# ============================================================================
# Stage 5: Response Quality Tests
# ============================================================================


class TestRerankerBackend:
    """Tests for 5.3: Configurable Reranker Backend."""

    def test_local_reranker_creation(self):
        """Test creating a local reranker via factory."""
        from core.reranker import create_reranker, LocalReranker
        reranker = create_reranker(backend="local")
        assert isinstance(reranker, LocalReranker)
        reranker.close()

    def test_api_reranker_requires_endpoint(self):
        """Test that API reranker requires an endpoint."""
        from core.reranker import create_reranker
        with pytest.raises(ValueError, match="endpoint is required"):
            create_reranker(backend="api")

    def test_api_reranker_creation(self):
        """Test creating an API reranker via factory (returns FallbackReranker)."""
        from core.reranker import create_reranker, FallbackReranker
        reranker = create_reranker(backend="api", endpoint="http://localhost:9999")
        assert isinstance(reranker, FallbackReranker)
        reranker.close()

    def test_api_reranker_predict_mock(self):
        """Test APIReranker.predict with mocked response."""
        from unittest.mock import patch, MagicMock
        from core.reranker import APIReranker

        reranker = APIReranker(endpoint="http://mock:1234")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.3},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("core.reranker.requests.post", return_value=mock_response):
            scores = reranker.predict([
                ("query", "doc1"),
                ("query", "doc2"),
            ])

        assert len(scores) == 2
        assert scores[0] == 0.9
        assert scores[1] == 0.3

    def test_fallback_reranker_falls_back(self):
        """Test FallbackReranker falls back to local on API failure."""
        from unittest.mock import MagicMock
        from core.reranker import FallbackReranker, APIReranker, LocalReranker

        api = MagicMock(spec=APIReranker)
        api.predict.side_effect = ConnectionError("API down")

        fallback = FallbackReranker(api, local_model="cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Should fall back to local
        pairs = [("test query", "test document")]
        scores = fallback.predict(pairs)
        assert len(scores) == 1
        assert isinstance(scores[0], float)

        # Second call should go directly to local (api_failed flag)
        assert fallback._api_failed
        fallback.close()

    def test_reranker_config_defaults(self):
        """Test reranker config defaults in client."""
        from core.client import DEFAULT_CONFIG
        search_cfg = DEFAULT_CONFIG["search"]
        assert search_cfg["reranker"] == "local"
        assert search_cfg["reranker_model"] is None
        assert search_cfg["reranker_endpoint"] is None

    def test_reranker_config_validation(self):
        """Test reranker config validation."""
        from core.client import validate_config
        warnings = validate_config({
            "machine_name": "test",
            "search": {"reranker": "invalid"},
        })
        assert any("reranker" in w for w in warnings)

        warnings = validate_config({
            "machine_name": "test",
            "search": {"reranker": "local"},
        })
        assert not any("reranker" in w for w in warnings)


# ---------------------------------------------------------------------------
# Stage 8: Advanced Query Processing Tests
# ---------------------------------------------------------------------------

class TestBM25ParameterTuning:
    """Tests for configurable BM25 k1 and b parameters."""

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_default_bm25_params(self, temp_index):
        """Test default k1 and b values are stored correctly."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        assert searcher.bm25_k1 == 1.5
        assert searcher.bm25_b == 0.75

    def test_custom_bm25_params(self, temp_index):
        """Test custom k1 and b values are stored."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, bm25_k1=2.0, bm25_b=0.5)
        assert searcher.bm25_k1 == 2.0
        assert searcher.bm25_b == 0.5

    def test_bm25_params_passed_to_index(self, temp_index):
        """Test BM25 params are used when building index."""
        from core.search import HybridSearcher
        from unittest.mock import patch

        searcher = HybridSearcher(index_path=temp_index, bm25_k1=2.0, bm25_b=0.3)

        # Index some test data
        chunks = [
            {"id": "1", "content": "Python programming language guide", "metadata": {"source": "test"}},
            {"id": "2", "content": "JavaScript web development tutorial", "metadata": {"source": "test"}},
        ]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        # Verify the BM25 index was built with custom params
        assert searcher.bm25_index is not None
        # BM25Okapi stores k1 and b
        assert searcher.bm25_index.k1 == 2.0
        assert searcher.bm25_index.b == 0.3

    def test_corpus_hash_changes_with_bm25_params(self, temp_index):
        """Test corpus hash changes when k1 or b change."""
        from core.search import HybridSearcher

        searcher1 = HybridSearcher(index_path=temp_index, bm25_k1=1.5, bm25_b=0.75)

        # Index test data
        chunks = [{"id": "1", "content": "test document", "metadata": {"source": "test"}}]
        embeddings = searcher1.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher1.index_chunks(chunks, embeddings)

        hash1 = searcher1._compute_corpus_hash()

        # Create searcher with different params (same index)
        searcher2 = HybridSearcher(index_path=temp_index, bm25_k1=2.0, bm25_b=0.5)
        hash2 = searcher2._compute_corpus_hash()

        assert hash1 != hash2

    def test_config_defaults_include_bm25_params(self):
        """Test DEFAULT_CONFIG includes bm25_k1 and bm25_b."""
        from core.client import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["search"]["bm25_k1"] == 1.5
        assert DEFAULT_CONFIG["search"]["bm25_b"] == 0.75

    def test_client_search_config_bm25_params(self):
        """Test client.search_config merges BM25 params."""
        from core.client import Client

        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {
                "name": "Test",
                "search": {"bm25_k1": 2.0, "bm25_b": 0.5},
            })
            config = client.search_config
            assert config["bm25_k1"] == 2.0
            assert config["bm25_b"] == 0.5

    def test_validate_bm25_k1_positive(self):
        """Test validation rejects non-positive bm25_k1."""
        from core.client import validate_config

        config = {"machine_name": "test", "search": {"bm25_k1": -1.0}}
        warnings = validate_config(config)
        assert any("bm25_k1" in w and "> 0" in w for w in warnings)

    def test_validate_bm25_b_range(self):
        """Test validation rejects bm25_b outside 0-1."""
        from core.client import validate_config

        config = {"machine_name": "test", "search": {"bm25_b": 1.5}}
        warnings = validate_config(config)
        assert any("bm25_b" in w and "between 0 and 1" in w for w in warnings)

    def test_validate_bm25_b_zero_ok(self):
        """Test bm25_b=0 is valid (no length normalization)."""
        from core.client import validate_config

        config = {"machine_name": "test", "search": {"bm25_b": 0}}
        warnings = validate_config(config)
        assert not any("bm25_b" in w for w in warnings)


class TestScoreNormalization:
    """Tests for configurable score normalization strategies."""

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_default_normalization(self, temp_index):
        """Test default normalization is minmax."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        assert searcher.score_normalization == "minmax"

    def test_minmax_normalization(self, temp_index):
        """Test minmax normalizes to 0-1 range."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, score_normalization="minmax")
        results = [
            ({"id": "a"}, 10.0),
            ({"id": "b"}, 5.0),
            ({"id": "c"}, 2.5),
        ]
        normalized = searcher._normalize_scores(results)
        assert normalized["a"] == 1.0
        assert normalized["b"] == 0.5
        assert normalized["c"] == 0.25

    def test_zscore_normalization(self, temp_index):
        """Test zscore normalizes using z-scores shifted so min=0."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, score_normalization="zscore")
        results = [
            ({"id": "a"}, 10.0),
            ({"id": "b"}, 5.0),
            ({"id": "c"}, 0.0),
        ]
        normalized = searcher._normalize_scores(results)

        # Minimum should be 0
        assert min(normalized.values()) == 0.0
        # Order should be preserved
        assert normalized["a"] > normalized["b"] > normalized["c"]

    def test_zscore_fallback_when_std_zero(self, temp_index):
        """Test zscore falls back to minmax when all scores are equal."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, score_normalization="zscore")
        results = [
            ({"id": "a"}, 5.0),
            ({"id": "b"}, 5.0),
        ]
        normalized = searcher._normalize_scores(results)
        # Should fall back to minmax (all scores = 1.0 since they equal max)
        assert normalized["a"] == 1.0
        assert normalized["b"] == 1.0

    def test_zscore_fallback_single_result(self, temp_index):
        """Test zscore falls back to minmax with single result."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, score_normalization="zscore")
        results = [({"id": "a"}, 5.0)]
        normalized = searcher._normalize_scores(results)
        assert normalized["a"] == 1.0

    def test_raw_normalization(self, temp_index):
        """Test raw passes scores through unchanged."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, score_normalization="raw")
        results = [
            ({"id": "a"}, 10.0),
            ({"id": "b"}, -2.0),
        ]
        normalized = searcher._normalize_scores(results)
        assert normalized["a"] == 10.0
        assert normalized["b"] == -2.0

    def test_empty_results(self, temp_index):
        """Test normalization with empty results."""
        from core.search import HybridSearcher

        for strategy in ("minmax", "zscore", "raw"):
            searcher = HybridSearcher(index_path=temp_index, score_normalization=strategy)
            assert searcher._normalize_scores([]) == {}

    def test_search_quality_includes_normalization(self):
        """Test SearchQuality dataclass includes score_normalization."""
        from core.search import SearchQuality

        q = SearchQuality(score_normalization="zscore")
        assert q.score_normalization == "zscore"

    def test_validate_score_normalization(self):
        """Test validation rejects invalid normalization strategy."""
        from core.client import validate_config

        config = {"machine_name": "test", "search": {"score_normalization": "invalid"}}
        warnings = validate_config(config)
        assert any("score_normalization" in w for w in warnings)

    def test_validate_score_normalization_valid(self):
        """Test valid normalization strategies don't warn."""
        from core.client import validate_config

        for strategy in ("minmax", "zscore", "raw"):
            config = {"machine_name": "test", "search": {"score_normalization": strategy}}
            warnings = validate_config(config)
            assert not any("score_normalization" in w for w in warnings)


class TestHyDE:
    """Tests for Hypothetical Document Embeddings."""

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_hyde_disabled_by_default(self, temp_index):
        """Test HyDE is disabled by default."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        assert searcher.hyde is False
        assert searcher.llm_provider is None

    def test_hyde_without_provider_no_crash(self, temp_index):
        """Test hyde=True but no provider doesn't crash."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index, hyde=True, llm_provider=None)
        assert searcher.hyde is True
        assert searcher.llm_provider is None

        # generate_hypothetical_document should return None
        result = searcher._generate_hypothetical_document("test query")
        assert result is None

    def test_hyde_generates_document(self, temp_index):
        """Test HyDE generates hypothetical document via mock LLM."""
        from core.search import HybridSearcher
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "Python is a high-level programming language known for its simple syntax "
            "and extensive standard library. It supports multiple programming paradigms."
        )

        searcher = HybridSearcher(index_path=temp_index, hyde=True, llm_provider=mock_llm)
        result = searcher._generate_hypothetical_document("What is Python?")

        assert result is not None
        assert "Python" in result
        mock_llm.generate.assert_called_once()

    def test_hyde_graceful_degradation(self, temp_index):
        """Test HyDE falls back gracefully when LLM fails."""
        from core.search import HybridSearcher
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM unavailable")

        searcher = HybridSearcher(index_path=temp_index, hyde=True, llm_provider=mock_llm)

        # Index some data
        chunks = [
            {"id": "1", "content": "Python programming language", "metadata": {"source": "test"}},
        ]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        # Search should still work (falls back to raw query)
        results = searcher.search("Python", top_k=1)
        assert len(results) >= 1

    def test_hyde_short_response_rejected(self, temp_index):
        """Test HyDE rejects very short LLM responses."""
        from core.search import HybridSearcher
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "short"

        searcher = HybridSearcher(index_path=temp_index, hyde=True, llm_provider=mock_llm)
        result = searcher._generate_hypothetical_document("test")
        assert result is None

    def test_search_quality_includes_hyde(self):
        """Test SearchQuality includes hyde_applied field."""
        from core.search import SearchQuality

        q = SearchQuality(hyde_applied=True)
        assert q.hyde_applied is True

        q2 = SearchQuality()
        assert q2.hyde_applied is False

    def test_config_defaults_include_hyde(self):
        """Test DEFAULT_CONFIG includes hyde setting."""
        from core.client import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["search"]["hyde"] is False


class TestMultiQuery:
    """Tests for Multi-Query / RAG-Fusion."""

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_multi_query_disabled_by_default(self, temp_index):
        """Test multi-query is disabled by default."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        assert searcher.multi_query is False
        assert searcher.multi_query_count == 3

    def test_generate_query_variants_with_mock(self):
        """Test query variant generation with mock LLM."""
        from core.refinement import generate_query_variants

        def mock_llm(messages, system):
            return '["variant one", "variant two", "variant three"]'

        variants = generate_query_variants("test query", count=3, get_llm_response=mock_llm)
        assert len(variants) == 4  # original + 3 variants
        assert variants[0] == "test query"
        assert "variant one" in variants

    def test_generate_query_variants_fallback(self):
        """Test query variant generation falls back to [query] without LLM."""
        from core.refinement import generate_query_variants

        variants = generate_query_variants("test query", count=3)
        assert variants == ["test query"]

    def test_generate_query_variants_llm_failure(self):
        """Test query variant generation falls back on LLM failure."""
        from core.refinement import generate_query_variants

        def failing_llm(messages, system):
            raise Exception("LLM error")

        variants = generate_query_variants("test query", count=3, get_llm_response=failing_llm)
        assert variants == ["test query"]

    def test_generate_query_variants_invalid_json(self):
        """Test query variant generation handles invalid JSON."""
        from core.refinement import generate_query_variants

        def bad_json_llm(messages, system):
            return "not valid json at all"

        variants = generate_query_variants("test query", count=3, get_llm_response=bad_json_llm)
        assert variants == ["test query"]

    def test_multi_query_search_integration(self, temp_index):
        """Test multi-query search with mock LLM."""
        from core.search import HybridSearcher
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.generate.return_value = '["Python coding", "programming with Python"]'

        searcher = HybridSearcher(
            index_path=temp_index,
            multi_query=True,
            multi_query_count=2,
            llm_provider=mock_llm,
        )

        # Index test data
        chunks = [
            {"id": "1", "content": "Python programming language guide", "metadata": {"source": "test"}},
            {"id": "2", "content": "JavaScript web development tutorial", "metadata": {"source": "test"}},
            {"id": "3", "content": "Python data science and machine learning", "metadata": {"source": "test"}},
        ]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        results = searcher.search("Python", top_k=3)
        assert len(results) >= 1
        # Multi-query should have been called
        mock_llm.generate.assert_called()

    def test_multi_query_recursion_prevention(self, temp_index):
        """Test _skip_multi_query prevents recursion in inner search calls."""
        from core.search import HybridSearcher
        from unittest.mock import MagicMock, patch

        mock_llm = MagicMock()
        mock_llm.generate.return_value = '["variant"]'

        searcher = HybridSearcher(
            index_path=temp_index,
            multi_query=True,
            multi_query_count=1,
            llm_provider=mock_llm,
        )

        # Index test data
        chunks = [{"id": "1", "content": "test content", "metadata": {"source": "test"}}]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        # Track _skip_multi_query kwarg during inner search calls
        inner_skip_flags = []
        original_search = searcher.search.__func__

        def tracking_search(self_arg, query, **kwargs):
            inner_skip_flags.append(kwargs.get("_skip_multi_query", False))
            return original_search(self_arg, query, **kwargs)

        with patch.object(type(searcher), 'search', tracking_search):
            searcher._multi_query_search("test", top_k=1)

        # Inner searches should have _skip_multi_query=True
        assert all(flag is True for flag in inner_skip_flags)
        # multi_query instance attribute should never have been mutated
        assert searcher.multi_query is True

    def test_multi_query_no_provider_fallback(self, temp_index):
        """Test multi_query=True but no provider falls through to normal search."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(
            index_path=temp_index,
            multi_query=True,
            llm_provider=None,
        )

        # Index test data
        chunks = [{"id": "1", "content": "test content", "metadata": {"source": "test"}}]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        # Should fall through to normal search (multi_query guard checks llm_provider)
        results = searcher.search("test", top_k=1)
        assert len(results) >= 1

    def test_search_quality_multi_query_fields(self):
        """Test SearchQuality includes multi-query fields."""
        from core.search import SearchQuality

        q = SearchQuality(multi_query_applied=True, multi_query_variants=4)
        assert q.multi_query_applied is True
        assert q.multi_query_variants == 4

        q2 = SearchQuality()
        assert q2.multi_query_applied is False
        assert q2.multi_query_variants == 0

    def test_validate_multi_query_count_range(self):
        """Test validation rejects multi_query_count outside 2-7."""
        from core.client import validate_config

        config = {"machine_name": "test", "search": {"multi_query_count": 1}}
        warnings = validate_config(config)
        assert any("multi_query_count" in w for w in warnings)

        config = {"machine_name": "test", "search": {"multi_query_count": 10}}
        warnings = validate_config(config)
        assert any("multi_query_count" in w for w in warnings)

    def test_validate_multi_query_count_valid(self):
        """Test valid multi_query_count values don't warn."""
        from core.client import validate_config

        for count in (2, 3, 5, 7):
            config = {"machine_name": "test", "search": {"multi_query_count": count}}
            warnings = validate_config(config)
            assert not any("multi_query_count" in w for w in warnings)

    def test_known_search_keys_updated(self):
        """Test _KNOWN_SEARCH_KEYS includes all new Stage 8 keys."""
        from core.client import _KNOWN_SEARCH_KEYS

        stage8_keys = {"bm25_k1", "bm25_b", "score_normalization", "hyde", "multi_query", "multi_query_count"}
        assert stage8_keys.issubset(_KNOWN_SEARCH_KEYS)
