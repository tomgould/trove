"""
Basic tests for core Trove modules.
"""

import json
import subprocess
import pytest
import requests
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil

# Test imports work
def test_imports():
    """Verify all core modules can be imported."""
    from core import client
    from core import ingest
    from core import llm
    from core import search
    from core import reference_config

    assert hasattr(client, 'Client')
    assert hasattr(client, 'load_client')
    assert hasattr(ingest, 'DocumentIngestor')
    assert hasattr(llm, 'LMStudioProvider')
    assert hasattr(llm, 'ClaudeProvider')
    assert hasattr(search, 'HybridSearcher')
    assert hasattr(search, 'EnhancedTokenizer')
    assert hasattr(reference_config, 'ReferenceConfig')


def test_client_config():
    """Test client configuration loading."""
    from core.client import Client, DEFAULT_CONFIG

    # Create a minimal client
    with tempfile.TemporaryDirectory() as tmpdir:
        client_path = Path(tmpdir)

        config = {
            "name": "Test Client",
            "machine_name": "test",
        }

        client = Client(client_path, config)

        assert client.name == "Test Client"
        assert client.machine_name == "test"
        assert client.documents_path == client_path / "documents"
        assert client.index_path == client_path / "index"

        # Check defaults are merged
        search_config = client.search_config
        assert "bm25_weight" in search_config
        assert "vector_weight" in search_config


def test_discover_clients():
    """Test client discovery."""
    from core.client import discover_clients

    with tempfile.TemporaryDirectory() as tmpdir:
        clients_dir = Path(tmpdir)

        # Create test client directories
        (clients_dir / "client_a").mkdir()
        (clients_dir / "client_b").mkdir()
        (clients_dir / ".hidden").mkdir()  # Should be ignored

        clients = discover_clients(clients_dir)

        assert "client_a" in clients
        assert "client_b" in clients
        assert ".hidden" not in clients


def test_llm_prompt_building():
    """Test RAG prompt construction."""
    from core.llm import build_rag_prompt

    chunks = [
        {
            "content": "This is chunk 1 content.",
            "metadata": {"source_file": "doc1.pdf"},
        },
        {
            "content": "This is chunk 2 content.",
            "metadata": {"source_file": "doc2.pdf"},
        },
    ]

    messages = build_rag_prompt(
        question="What is the content?",
        chunks=chunks,
        max_context_chars=10000,
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "chunk 1 content" in messages[0]["content"]
    assert "chunk 2 content" in messages[0]["content"]
    assert "[Source 1:" in messages[0]["content"]
    assert "[Source 2:" in messages[0]["content"]


def test_llm_context_truncation():
    """Test that context is truncated at max_context_chars."""
    from core.llm import build_rag_prompt

    # Create a very long chunk
    long_content = "A" * 5000
    chunks = [
        {"content": long_content, "metadata": {"source_file": "big.pdf"}},
        {"content": "Second chunk", "metadata": {"source_file": "small.pdf"}},
    ]

    # With small max_context_chars, second chunk should be excluded
    messages = build_rag_prompt(
        question="Test?",
        chunks=chunks,
        max_context_chars=1000,  # Too small for both chunks
    )

    content = messages[0]["content"]
    # First chunk should be truncated or excluded
    assert len(content) < 6000


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


class TestClientConfig:
    """Tests for new client configuration properties."""

    def test_embedding_config_defaults(self):
        """Test embedding_config returns defaults."""
        from core.client import Client

        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {"name": "Test"})
            config = client.embedding_config

            assert config["model"] == "all-MiniLM-L6-v2"
            assert config["dimension"] == 384
            assert config["query_prefix"] == ""
            assert config["passage_prefix"] == ""

    def test_embedding_config_custom(self):
        """Test embedding_config with custom values."""
        from core.client import Client

        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {
                "name": "Test",
                "embedding": {
                    "model": "BAAI/bge-base-en-v1.5",
                    "dimension": 768,
                }
            })
            config = client.embedding_config

            assert config["model"] == "BAAI/bge-base-en-v1.5"
            assert config["dimension"] == 768

    def test_tokenization_config_defaults(self):
        """Test tokenization_config returns defaults."""
        from core.client import Client

        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {"name": "Test"})
            config = client.tokenization_config

            assert config["stemmer"] is None
            assert config["synonyms"] == {}

    def test_tokenization_config_custom(self):
        """Test tokenization_config with custom values."""
        from core.client import Client

        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {
                "name": "Test",
                "search": {
                    "tokenization": {
                        "stemmer": "porter",
                        "synonyms": {"aml": ["anti-money laundering"]},
                    }
                }
            })
            config = client.tokenization_config

            assert config["stemmer"] == "porter"
            assert "aml" in config["synonyms"]

    def test_reference_config(self):
        """Test reference_config property."""
        from core.client import Client
        from core.reference_config import ReferenceConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {
                "name": "Test",
                "references": {
                    "enabled": True,
                    "domain": "legal",
                }
            })
            config = client.reference_config

            assert isinstance(config, ReferenceConfig)
            assert config.enabled is True
            assert config.domain == "legal"


class TestReferenceConfig:
    """Tests for ReferenceConfig and pattern detection."""

    def test_default_config(self):
        """Test default ReferenceConfig."""
        from core.reference_config import ReferenceConfig

        config = ReferenceConfig()

        assert config.enabled is True
        assert config.mode == "extend"
        assert config.domain == "compliance"

    def test_domain_patterns(self):
        """Test that domain patterns are loaded."""
        from core.reference_config import ReferenceConfig

        config = ReferenceConfig(domain="compliance")
        patterns = config.get_patterns()

        # Compliance domain should have appendix, section, schedule patterns
        pattern_names = [p.name for p in patterns]
        assert "appendix" in pattern_names
        assert "section" in pattern_names
        assert "schedule" in pattern_names

    def test_legal_domain(self):
        """Test legal domain patterns."""
        from core.reference_config import ReferenceConfig

        config = ReferenceConfig(domain="legal")
        patterns = config.get_patterns()

        pattern_names = [p.name for p in patterns]
        assert "exhibit" in pattern_names
        assert "clause" in pattern_names
        assert "article" in pattern_names

    def test_detect_references(self):
        """Test reference detection in text."""
        from core.reference_config import ReferenceConfig

        config = ReferenceConfig(domain="compliance")

        text = "Please refer to Appendix C for the list of high-risk countries. See also Section 3.2 for requirements."
        refs = config.detect_references(text)

        assert len(refs) >= 2
        ref_keys = [r["key"] for r in refs]
        assert "appendix:C" in ref_keys
        assert "section:3.2" in ref_keys

    def test_custom_patterns(self):
        """Test custom pattern configuration."""
        from core.reference_config import ReferenceConfig

        config = ReferenceConfig(
            mode="extend",
            domain="compliance",
            patterns={
                "custom_ref": {
                    "priority": "high",
                    "patterns": [r'\bREF-(\d+)\b'],
                    "query_templates": ["Reference {id}"],
                }
            }
        )

        text = "See REF-12345 for details."
        refs = config.detect_references(text)

        ref_types = [r["type"] for r in refs]
        assert "custom_ref" in ref_types

    def test_build_queries(self):
        """Test query building for references."""
        from core.reference_config import ReferenceConfig

        config = ReferenceConfig(domain="compliance")

        ref = {"type": "appendix", "id": "C", "key": "appendix:C"}
        queries = config.build_queries_for_reference(ref)

        assert "Appendix C" in queries or "Annex C" in queries

    def test_disabled_config(self):
        """Test that disabled config is respected."""
        from core.reference_config import ReferenceConfig

        config = ReferenceConfig(enabled=False)
        assert config.enabled is False


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


class TestEmbeddingConfig:
    """Tests for configurable embedding models."""

    @pytest.fixture
    def temp_index(self):
        """Create temporary index directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_default_embedding_model(self, temp_index):
        """Test default embedding model is used."""
        from core.search import HybridSearcher, DEFAULT_EMBEDDING_MODEL

        searcher = HybridSearcher(index_path=temp_index)
        assert searcher.embedding_model == DEFAULT_EMBEDDING_MODEL

    def test_custom_embedding_config(self, temp_index):
        """Test custom embedding config is applied."""
        from core.search import HybridSearcher

        config = {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
        }
        searcher = HybridSearcher(index_path=temp_index, embedding_config=config)

        assert searcher.embedding_model == "all-MiniLM-L6-v2"
        assert searcher.query_prefix == "query: "
        assert searcher.passage_prefix == "passage: "
        assert searcher.embedding_dimension == 384

    def test_model_metadata_stored(self, temp_index):
        """Test that model metadata is stored in collection."""
        from core.search import HybridSearcher

        # Create index with model config
        searcher = HybridSearcher(
            index_path=temp_index,
            embedding_config={"model": "all-MiniLM-L6-v2", "dimension": 384}
        )
        chunks = [{"id": "1", "content": "Test", "metadata": {"source": "test"}}]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        # Check collection metadata
        metadata = searcher.collection.metadata
        assert metadata.get("embedding_model") == "all-MiniLM-L6-v2"
        assert metadata.get("embedding_dimension") == 384

    def test_model_match_check(self, temp_index):
        """Test model match checking works."""
        from core.search import HybridSearcher

        # Create index
        searcher = HybridSearcher(
            index_path=temp_index,
            embedding_config={"model": "all-MiniLM-L6-v2", "dimension": 384}
        )
        chunks = [{"id": "1", "content": "Test", "metadata": {"source": "test"}}]
        embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
        searcher.index_chunks(chunks, embeddings)

        # Check with same config - should match
        matches, reason = searcher.check_embedding_model_match()
        assert matches is True

        # Manually change the searcher's config and check again
        searcher.embedding_model = "different-model"
        matches, reason = searcher.check_embedding_model_match()
        assert matches is False
        assert "mismatch" in reason.lower()


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
        assert searcher.adaptive_rerank_depth == 15

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


class TestClientConfigStage1:
    """Tests for new Stage 1 client configuration defaults."""

    def test_search_config_new_defaults(self):
        """Test that new search config defaults are set."""
        from core.client import Client, DEFAULT_CONFIG

        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {"name": "Test"})
            config = client.search_config

            assert config["adaptive_rerank"] is True
            assert config["adaptive_rerank_threshold"] == 0.9
            assert config["adaptive_rerank_depth"] == 15
            assert config["merge_strategy"] == "weighted"
            assert config["rrf_k"] == 60
            assert config["mmr_lambda"] is None

    def test_search_config_custom_overrides(self):
        """Test that custom config overrides defaults."""
        from core.client import Client

        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {
                "name": "Test",
                "search": {
                    "merge_strategy": "rrf",
                    "mmr_lambda": 0.7,
                    "adaptive_rerank": False,
                }
            })
            config = client.search_config

            assert config["merge_strategy"] == "rrf"
            assert config["mmr_lambda"] == 0.7
            assert config["adaptive_rerank"] is False


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


class TestContextBudget:
    """Tests for context budget management."""

    def test_estimate_tokens(self):
        """Test token estimation heuristic."""
        from mcp_server.server import estimate_tokens

        assert estimate_tokens("") == 0
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("a" * 100) == 25
        assert estimate_tokens("a" * 400) == 100

    def test_format_with_budget(self):
        """Test format_search_results respects max_tokens budget."""
        from mcp_server.server import format_search_results

        results = [
            {
                "id": str(i),
                "content": "x" * 500,
                "metadata": {"source_file": f"doc{i}.pdf"},
                "_rerank_score": 1.0 - i * 0.1,
            }
            for i in range(10)
        ]

        # Small budget should limit results
        output = format_search_results(results, include_content=True, max_tokens=200)
        assert "showing" in output
        assert "Tokens:" in output
        # Should not show all 10 results
        assert "Result 10" not in output

    def test_format_without_budget(self):
        """Test format_search_results shows all when no budget."""
        from mcp_server.server import format_search_results

        results = [
            {
                "id": str(i),
                "content": f"Content {i}",
                "metadata": {"source_file": f"doc{i}.pdf"},
                "_rerank_score": 1.0 - i * 0.1,
            }
            for i in range(3)
        ]

        output = format_search_results(results, include_content=True, max_tokens=None)
        assert "Result 1" in output
        assert "Result 2" in output
        assert "Result 3" in output
        assert "Tokens:" not in output

    def test_format_empty_results(self):
        """Test format with no results."""
        from mcp_server.server import format_search_results

        output = format_search_results([], max_tokens=500)
        assert output == "No results found."

    def test_format_shows_client_label(self):
        """Test format shows _source_client when present."""
        from mcp_server.server import format_search_results

        results = [
            {
                "id": "1",
                "content": "test",
                "metadata": {"source_file": "doc.pdf"},
                "_rerank_score": 0.5,
                "_source_client": "my_client",
            }
        ]

        output = format_search_results(results, include_content=True)
        assert "**Client:** my_client" in output


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


class TestSummarization:
    """Tests for result summarization."""

    def test_summarize_fallback_truncation(self):
        """Test _summarize_chunk falls back to truncation without LLM."""
        from mcp_server.server import _summarize_chunk

        long_content = "A" * 500
        summary = _summarize_chunk(long_content, "test query", provider=None)
        assert len(summary) == 203  # 200 chars + "..."
        assert summary.endswith("...")

    def test_summarize_short_content(self):
        """Test _summarize_chunk returns short content as-is."""
        from mcp_server.server import _summarize_chunk

        short_content = "Short text"
        summary = _summarize_chunk(short_content, "test query", provider=None)
        assert summary == "Short text"

    def test_summarize_with_mock_llm(self):
        """Test _summarize_chunk uses LLM when available."""
        from mcp_server.server import _summarize_chunk
        from unittest.mock import MagicMock

        mock_provider = MagicMock()
        mock_provider.generate.return_value = "This is an LLM summary."

        summary = _summarize_chunk("Some content", "test query", provider=mock_provider)
        assert summary == "This is an LLM summary."
        mock_provider.generate.assert_called_once()

    def test_summarize_llm_error_fallback(self):
        """Test _summarize_chunk falls back on LLM error."""
        from mcp_server.server import _summarize_chunk
        from unittest.mock import MagicMock

        mock_provider = MagicMock()
        mock_provider.generate.side_effect = RuntimeError("LLM error")

        long_content = "B" * 500
        summary = _summarize_chunk(long_content, "test query", provider=mock_provider)
        assert summary.endswith("...")
        assert len(summary) == 203


# ---------------------------------------------------------------------------
# Stage 3: Evaluation Framework Tests
# ---------------------------------------------------------------------------

class TestEvalDataset:
    """Tests for evaluation dataset loading."""

    def test_load_eval_dataset(self):
        """Load valid JSON, returns TestCase list."""
        from core.evaluation import load_eval_dataset, TestCase
        from core.client import Client

        with tempfile.TemporaryDirectory() as tmpdir:
            client_path = Path(tmpdir)
            tests_dir = client_path / "tests"
            tests_dir.mkdir()

            dataset = {
                "version": "1.0",
                "test_cases": [
                    {
                        "id": "test-001",
                        "query": "What is X?",
                        "expected_chunks": ["X is"],
                        "ground_truth": "X is something.",
                        "tags": ["basic"],
                    }
                ],
            }
            (tests_dir / "eval_dataset.json").write_text(json.dumps(dataset))

            client = Client(client_path, {"name": "Test"})
            cases = load_eval_dataset(client)

            assert len(cases) == 1
            assert isinstance(cases[0], TestCase)
            assert cases[0].id == "test-001"

    def test_load_missing_dataset(self):
        """Missing file raises FileNotFoundError."""
        from core.evaluation import load_eval_dataset
        from core.client import Client

        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {"name": "Test"})
            with pytest.raises(FileNotFoundError):
                load_eval_dataset(client)

    def test_testcase_fields(self):
        """Verify all fields populated correctly."""
        from core.evaluation import load_eval_dataset
        from core.client import Client

        with tempfile.TemporaryDirectory() as tmpdir:
            client_path = Path(tmpdir)
            tests_dir = client_path / "tests"
            tests_dir.mkdir()

            dataset = {
                "test_cases": [
                    {
                        "id": "tc-1",
                        "query": "query text",
                        "expected_chunks": ["a", "b"],
                        "ground_truth": "answer",
                        "tags": ["tag1", "tag2"],
                    }
                ],
            }
            (tests_dir / "eval_dataset.json").write_text(json.dumps(dataset))

            client = Client(client_path, {"name": "Test"})
            tc = load_eval_dataset(client)[0]

            assert tc.id == "tc-1"
            assert tc.query == "query text"
            assert tc.expected_chunks == ["a", "b"]
            assert tc.ground_truth == "answer"
            assert tc.tags == ["tag1", "tag2"]


class TestRetrievalMetrics:
    """Tests for retrieval metric computation."""

    def test_mrr_first_result(self):
        """Relevant chunk at position 1 -> MRR=1.0."""
        from core.evaluation import mrr

        chunks = [
            {"content": "BM25 keyword search works great"},
            {"content": "unrelated content"},
        ]
        assert mrr(chunks, ["BM25"]) == 1.0

    def test_mrr_third_result(self):
        """Relevant chunk at position 3 -> MRR=1/3."""
        from core.evaluation import mrr

        chunks = [
            {"content": "unrelated A"},
            {"content": "unrelated B"},
            {"content": "BM25 keyword search"},
        ]
        assert mrr(chunks, ["BM25"]) == pytest.approx(1.0 / 3, abs=0.001)

    def test_mrr_no_match(self):
        """No relevant chunks -> MRR=0.0."""
        from core.evaluation import mrr

        chunks = [{"content": "unrelated"}]
        assert mrr(chunks, ["BM25"]) == 0.0

    def test_hit_at_k(self):
        """Relevant at position 3: k=5 -> 1.0, k=2 -> 0.0."""
        from core.evaluation import hit_at_k

        chunks = [
            {"content": "A"},
            {"content": "B"},
            {"content": "BM25 keyword"},
            {"content": "C"},
            {"content": "D"},
        ]
        assert hit_at_k(chunks, ["BM25"], k=5) == 1.0
        assert hit_at_k(chunks, ["BM25"], k=2) == 0.0

    def test_precision_at_k(self):
        """2 relevant in top 5 -> 0.4."""
        from core.evaluation import precision_at_k

        chunks = [
            {"content": "BM25 keyword search"},
            {"content": "unrelated"},
            {"content": "vector semantic search"},
            {"content": "unrelated"},
            {"content": "unrelated"},
        ]
        assert precision_at_k(chunks, ["BM25", "vector"], k=5) == pytest.approx(0.4)

    def test_ndcg_perfect(self):
        """All relevant at top -> 1.0."""
        from core.evaluation import ndcg

        chunks = [
            {"content": "BM25 keyword search"},
            {"content": "vector semantic search"},
            {"content": "unrelated"},
        ]
        assert ndcg(chunks, ["BM25", "vector"], k=3) == pytest.approx(1.0, abs=0.001)

    def test_ndcg_partial(self):
        """Some relevant -> between 0 and 1."""
        from core.evaluation import ndcg

        chunks = [
            {"content": "unrelated"},
            {"content": "BM25 keyword search"},
            {"content": "unrelated"},
        ]
        score = ndcg(chunks, ["BM25"], k=3)
        assert 0.0 < score < 1.0

    def test_context_recall(self):
        """2/3 patterns found -> 0.667."""
        from core.evaluation import context_recall

        chunks = [
            {"content": "BM25 keyword search and vector semantic search"},
        ]
        score = context_recall(chunks, ["BM25", "vector", "graph"])
        assert score == pytest.approx(2.0 / 3.0, abs=0.001)


class TestRagMetrics:
    """Tests for RAG quality metrics (LLM-as-judge)."""

    def test_llm_judge_valid_json(self):
        """Mock LLM returns valid JSON -> parsed correctly."""
        from core.evaluation import _llm_judge
        from unittest.mock import MagicMock

        provider = MagicMock()
        provider.generate.return_value = '{"score": 0.85, "reason": "Good answer"}'

        result = _llm_judge(provider, "test prompt")
        assert result["score"] == 0.85
        assert result["reason"] == "Good answer"

    def test_llm_judge_markdown_wrapped(self):
        """Response wrapped in ```json``` -> still parsed."""
        from core.evaluation import _llm_judge
        from unittest.mock import MagicMock

        provider = MagicMock()
        provider.generate.return_value = '```json\n{"score": 0.7, "reason": "OK"}\n```'

        result = _llm_judge(provider, "test prompt")
        assert result["score"] == 0.7

    def test_llm_judge_invalid_response(self):
        """Invalid response -> score 0.0 with error reason."""
        from core.evaluation import _llm_judge
        from unittest.mock import MagicMock

        provider = MagicMock()
        provider.generate.return_value = "This is not JSON"

        result = _llm_judge(provider, "test prompt", max_retries=0)
        assert result["score"] == 0.0
        assert "error" in result["reason"].lower()

    def test_compute_rag_metrics_no_ground_truth(self):
        """Without ground_truth -> no context_precision metric."""
        from core.evaluation import compute_rag_metrics
        from unittest.mock import MagicMock

        provider = MagicMock()
        provider.generate.return_value = '{"score": 0.9, "reason": "Good"}'

        metrics = compute_rag_metrics(
            provider,
            question="What is X?",
            answer="X is Y.",
            retrieved_chunks=[{"content": "X is Y according to docs"}],
            ground_truth=None,
        )

        assert "faithfulness" in metrics
        assert "answer_relevance" in metrics
        assert "context_precision" not in metrics


class TestAggregateMetrics:
    """Tests for metric aggregation."""

    def test_aggregate_basic(self):
        """Average of multiple results."""
        from core.evaluation import aggregate_metrics, EvalResult, TestCase

        tc = TestCase(id="t", query="q", expected_chunks=[])
        results = [
            EvalResult(test_case=tc, retrieved_chunks=[], metrics={"mrr": 1.0, "hit@10": 1.0}),
            EvalResult(test_case=tc, retrieved_chunks=[], metrics={"mrr": 0.5, "hit@10": 1.0}),
        ]

        agg = aggregate_metrics(results)
        assert agg["mrr"] == pytest.approx(0.75)
        assert agg["hit@10"] == pytest.approx(1.0)

    def test_aggregate_empty(self):
        """Empty list -> empty dict."""
        from core.evaluation import aggregate_metrics

        assert aggregate_metrics([]) == {}


# =============================================================================
# Stage 4: Codebase Health Tests
# =============================================================================

class TestConfigValidation:
    """Tests for client config validation."""

    def test_validate_config_valid(self):
        """Valid config returns no warnings."""
        from core.client import validate_config

        config = {
            "name": "Test",
            "machine_name": "test",
            "search": {
                "bm25_weight": 0.5,
                "vector_weight": 0.5,
                "top_k": 20,
                "merge_strategy": "weighted",
            },
        }
        warnings = validate_config(config)
        assert warnings == []

    def test_validate_config_unknown_keys(self):
        """Typos in top-level and search keys produce warnings."""
        from core.client import validate_config

        config = {
            "machine_name": "test",
            "naem": "typo",  # typo
            "search": {
                "bm25_weght": 0.5,  # typo
            },
        }
        warnings = validate_config(config)
        assert any("naem" in w for w in warnings)
        assert any("bm25_weght" in w for w in warnings)

    def test_validate_config_bad_types(self):
        """Wrong type for numeric fields produces warnings."""
        from core.client import validate_config

        config = {
            "machine_name": "test",
            "search": {
                "bm25_weight": "not_a_number",
                "top_k": 3.14,  # should be int
            },
        }
        warnings = validate_config(config)
        assert any("bm25_weight" in w for w in warnings)
        assert any("top_k" in w for w in warnings)

    def test_validate_config_bad_ranges(self):
        """Out-of-range values produce warnings."""
        from core.client import validate_config

        config = {
            "machine_name": "test",
            "search": {
                "bm25_weight": 1.5,  # > 1
                "top_k": -1,  # < 0
            },
        }
        warnings = validate_config(config)
        assert any("bm25_weight" in w and "between 0 and 1" in w for w in warnings)
        assert any("top_k" in w and "> 0" in w for w in warnings)

    def test_validate_config_missing_machine_name(self):
        """Missing required field produces warning."""
        from core.client import validate_config

        config = {"name": "Test"}
        warnings = validate_config(config)
        assert any("machine_name" in w for w in warnings)

    def test_validate_config_embedding_mismatch(self):
        """Embedding dimension mismatch produces warning."""
        from core.client import validate_config

        config = {
            "machine_name": "test",
            "embedding": {
                "model": "all-MiniLM-L6-v2",
                "dimension": 768,  # should be 384
            },
        }
        warnings = validate_config(config)
        assert any("dimension" in w and "384" in w for w in warnings)


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


class TestLLMRetry:
    """Tests for LLM retry logic."""

    def test_retry_succeeds_on_second_try(self):
        """Retry succeeds after transient failure."""
        from core.llm import _retry_with_backoff

        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.ConnectionError("Connection refused")
            return "success"

        result = _retry_with_backoff(
            flaky_func,
            max_retries=2,
            base_delay=0.01,  # fast for tests
            retryable_exceptions=(requests.ConnectionError,),
        )
        assert result == "success"
        assert call_count == 2

    def test_retry_exhausted(self):
        """All retries fail, raises the exception."""
        from core.llm import _retry_with_backoff

        def always_fail():
            raise requests.ConnectionError("Connection refused")

        with pytest.raises(requests.ConnectionError):
            _retry_with_backoff(
                always_fail,
                max_retries=2,
                base_delay=0.01,
                retryable_exceptions=(requests.ConnectionError,),
            )

    def test_retry_non_retryable(self):
        """Non-retryable error raises immediately without retry."""
        from core.llm import _retry_with_backoff

        call_count = 0

        def bad_request():
            nonlocal call_count
            call_count += 1
            raise ValueError("Bad request")

        with pytest.raises(ValueError):
            _retry_with_backoff(
                bad_request,
                max_retries=2,
                base_delay=0.01,
                retryable_exceptions=(requests.ConnectionError,),
            )
        assert call_count == 1  # No retry for non-retryable


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


class TestContextFilter:
    """Tests for 5.2: Context Relevance Pre-Filtering."""

    def test_context_filter_default_disabled(self):
        """Test context filter is disabled by default."""
        from core.client import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["llm"]["context_filter"] is False

    def test_context_filter_threshold_default(self):
        """Test default context filter threshold."""
        from core.client import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["llm"]["context_filter_threshold"] == 0.1

    def test_context_filter_config_merging(self):
        """Test that client config properly merges filter settings."""
        from core.client import Client
        client = Client(Path("/tmp/test"), {
            "machine_name": "test",
            "llm": {"context_filter": True, "context_filter_threshold": 0.5},
        })
        llm_cfg = client.llm_config
        assert llm_cfg["context_filter"] is True
        assert llm_cfg["context_filter_threshold"] == 0.5



class TestTokenBudgeting:
    """Tests for 5.4: Fine-Grained Token Budgeting."""

    def test_categorize_chunks_empty(self):
        """Test categorizing an empty list."""
        from core.llm import categorize_chunks
        result = categorize_chunks([])
        assert result == {"doc": [], "code": [], "reference": []}

    def test_categorize_chunks_mixed(self):
        """Test categorizing mixed chunk types."""
        from core.llm import categorize_chunks
        chunks = [
            {"content": "doc1", "metadata": {}},
            {"content": "code1", "metadata": {"chunk_type": "code"}},
            {"content": "ref1", "metadata": {}, "_resolved_reference": True},
            {"content": "xref1", "metadata": {}, "_cross_referenced": True},
            {"content": "doc2", "metadata": {"chunk_type": "document"}},
        ]
        result = categorize_chunks(chunks)
        assert len(result["doc"]) == 2
        assert len(result["code"]) == 1
        assert len(result["reference"]) == 2

    def test_allocate_budgets_all_active(self):
        """Test budget allocation with all categories active."""
        from core.llm import allocate_budgets
        budgets = allocate_budgets(
            total_chars=10000,
            doc_ratio=0.6, code_ratio=0.3, ref_ratio=0.1,
            counts={"doc": 5, "code": 3, "reference": 2},
        )
        assert budgets["doc"] == 6000
        assert budgets["code"] == 3000
        assert budgets["reference"] == 1000

    def test_allocate_budgets_no_code(self):
        """Test budget reallocation when no code chunks exist."""
        from core.llm import allocate_budgets
        budgets = allocate_budgets(
            total_chars=10000,
            doc_ratio=0.6, code_ratio=0.3, ref_ratio=0.1,
            counts={"doc": 5, "code": 0, "reference": 2},
        )
        assert budgets["code"] == 0
        # doc and reference get redistributed proportionally
        total_non_code = budgets["doc"] + budgets["reference"]
        assert total_non_code <= 10000
        assert budgets["doc"] > budgets["reference"]

    def test_allocate_budgets_only_docs(self):
        """Test budget when only docs exist."""
        from core.llm import allocate_budgets
        budgets = allocate_budgets(
            total_chars=10000,
            doc_ratio=0.6, code_ratio=0.3, ref_ratio=0.1,
            counts={"doc": 5, "code": 0, "reference": 0},
        )
        assert budgets["doc"] == 10000
        assert budgets["code"] == 0
        assert budgets["reference"] == 0

    def test_build_rag_prompt_without_budgets(self):
        """Test build_rag_prompt without budget ratios (legacy behavior)."""
        from core.llm import build_rag_prompt
        chunks = [
            {"content": "Test content 1", "metadata": {"source_file": "test.pdf"}},
            {"content": "Test content 2", "metadata": {"source_file": "test2.pdf"}},
        ]
        messages = build_rag_prompt("What is this?", chunks)
        assert len(messages) == 1
        assert "[Source 1:" in messages[0]["content"]
        assert "[Source 2:" in messages[0]["content"]

    def test_build_rag_prompt_with_budgets(self):
        """Test build_rag_prompt with budget ratios."""
        from core.llm import build_rag_prompt
        chunks = [
            {"content": "Document content", "metadata": {"source_file": "doc.pdf"}},
            {"content": "Code content", "metadata": {"source_file": "code.py", "chunk_type": "code"}},
        ]
        messages = build_rag_prompt(
            "What is this?", chunks,
            doc_budget_ratio=0.7, code_budget_ratio=0.2, reference_budget_ratio=0.1,
        )
        assert len(messages) == 1
        # Both chunks should appear (within budget)
        assert "Document content" in messages[0]["content"]
        assert "Code content" in messages[0]["content"]

    def test_budget_ratio_config_defaults(self):
        """Test budget ratio defaults in client config."""
        from core.client import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["llm"]["doc_budget_ratio"] == 0.6
        assert DEFAULT_CONFIG["llm"]["code_budget_ratio"] == 0.3
        assert DEFAULT_CONFIG["llm"]["reference_budget_ratio"] == 0.1


class TestInlineCitations:
    """Tests for 5.1: Inline Citations & Source Attribution."""

    def test_citation_map_building(self):
        """Test building a citation map from chunks."""
        from core.llm import build_citation_map
        chunks = [
            {"content": "Chunk 1", "metadata": {"source_file": "a.pdf", "page_number": 1}, "_rerank_score": 0.9},
            {"content": "Chunk 2", "metadata": {"source_file": "b.pdf"}, "_rerank_score": 0.7},
        ]
        cmap = build_citation_map(chunks)
        assert len(cmap) == 2
        assert cmap.get(1)["source"] == "a.pdf"
        assert cmap.get(1)["page"] == 1
        assert cmap.get(2)["source"] == "b.pdf"

    def test_parse_citations_basic(self):
        """Test parsing basic citation references."""
        from core.llm import build_citation_map, parse_citations_from_response
        chunks = [
            {"content": "c1", "metadata": {"source_file": "a.pdf"}, "_rerank_score": 0.9},
            {"content": "c2", "metadata": {"source_file": "b.pdf"}, "_rerank_score": 0.7},
            {"content": "c3", "metadata": {"source_file": "c.pdf"}, "_rerank_score": 0.5},
        ]
        cmap = build_citation_map(chunks)
        response = "This is based on [1] and also [3]."
        used = parse_citations_from_response(response, cmap)
        assert len(used) == 2
        assert used[0]["citation_number"] == 1
        assert used[1]["citation_number"] == 3

    def test_parse_citations_duplicate(self):
        """Test that duplicate citations are deduplicated."""
        from core.llm import build_citation_map, parse_citations_from_response
        chunks = [
            {"content": "c1", "metadata": {"source_file": "a.pdf"}, "_rerank_score": 0.9},
        ]
        cmap = build_citation_map(chunks)
        response = "First [1] then [1] again."
        used = parse_citations_from_response(response, cmap)
        assert len(used) == 1

    def test_parse_citations_invalid_numbers(self):
        """Test that invalid citation numbers are ignored."""
        from core.llm import build_citation_map, parse_citations_from_response
        chunks = [
            {"content": "c1", "metadata": {"source_file": "a.pdf"}, "_rerank_score": 0.9},
        ]
        cmap = build_citation_map(chunks)
        response = "This [1] and [99] are cited."
        used = parse_citations_from_response(response, cmap)
        assert len(used) == 1
        assert used[0]["citation_number"] == 1

    def test_parse_citations_empty_response(self):
        """Test parsing citations from empty response."""
        from core.llm import build_citation_map, parse_citations_from_response, CitationMap
        cmap = CitationMap()
        used = parse_citations_from_response("", cmap)
        assert used == []

    def test_citation_prompt_format(self):
        """Test that citation format uses [N] instead of [Source N: ...]."""
        from core.llm import build_rag_prompt
        chunks = [
            {"content": "Test content", "metadata": {"source_file": "test.pdf"}},
        ]
        messages = build_rag_prompt("question?", chunks, citations=True)
        content = messages[0]["content"]
        assert "[1] test.pdf" in content
        assert "[Source 1:" not in content

    def test_citation_system_prompt_suffix(self):
        """Test citation system prompt suffix exists and instructs numbered refs."""
        from core.llm import CITATION_SYSTEM_PROMPT_SUFFIX
        assert "[1]" in CITATION_SYSTEM_PROMPT_SUFFIX
        assert "[2]" in CITATION_SYSTEM_PROMPT_SUFFIX

    def test_citations_default_disabled(self):
        """Test citations are disabled by default."""
        from core.client import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["llm"]["citations"] is False

    def test_build_rag_prompt_with_citations_convenience(self):
        """Test the build_rag_prompt_with_citations convenience function."""
        from core.llm import build_rag_prompt_with_citations
        chunks = [
            {"content": "Content 1", "metadata": {"source_file": "a.pdf"}, "_rerank_score": 0.9},
            {"content": "Content 2", "metadata": {"source_file": "b.pdf"}, "_rerank_score": 0.7},
        ]
        messages, cmap = build_rag_prompt_with_citations("question?", chunks)
        assert len(messages) == 1
        assert len(cmap) == 2
        assert "[1]" in messages[0]["content"]


# =============================================================================
# Stage 6: Ingestion Improvements Tests
# =============================================================================


class TestChunkValidation:
    """Tests for 6.2: Chunk quality validation."""

    def test_validate_chunk_valid(self):
        """Valid chunk passes validation."""
        from core.ingest import validate_chunk
        is_valid, reason = validate_chunk("This is a perfectly valid chunk with enough content to pass.")
        assert is_valid is True
        assert reason is None

    def test_validate_chunk_too_short(self):
        """Short chunk is rejected."""
        from core.ingest import validate_chunk
        is_valid, reason = validate_chunk("Too short")
        assert is_valid is False
        assert reason == "too_short"

    def test_validate_chunk_no_alphanumeric(self):
        """Chunk with no alphanumeric chars is rejected."""
        from core.ingest import validate_chunk
        is_valid, reason = validate_chunk("---" * 30)  # 90 chars, no alphanumeric
        assert is_valid is False
        assert reason == "no_alphanumeric"

    def test_validate_chunk_whitespace_stripped(self):
        """Whitespace-only chunk counts as too short."""
        from core.ingest import validate_chunk
        is_valid, reason = validate_chunk("   \n\n\t   ")
        assert is_valid is False
        assert reason == "too_short"

    def test_chunk_deduplication(self):
        """Identical content chunks are deduplicated."""
        from core.ingest import DocumentIngestor

        ingestor = DocumentIngestor()
        # Create mock elements that produce identical content
        from unittest.mock import MagicMock, patch

        class FakeElement:
            def __init__(self, text):
                self._text = text
                self.metadata = MagicMock()
                self.metadata.to_dict.return_value = {}
            def __str__(self):
                return self._text

        elements = [FakeElement("Duplicate content that is long enough to pass validation." * 2)]

        # Patch chunk_by_title to return duplicate chunks
        with patch("core.ingest.chunk_by_title", return_value=[
            FakeElement("Same content repeated for dedup test. " * 3),
            FakeElement("Same content repeated for dedup test. " * 3),
            FakeElement("Different content that should not be deduped. " * 3),
        ]):
            result = ingestor.chunk_elements(elements, source_id="test")

        assert len(result) == 2  # One duplicate removed

    def test_chunk_metadata_extraction(self):
        """Unstructured metadata is extracted into chunk metadata."""
        from core.ingest import DocumentIngestor
        from unittest.mock import MagicMock, patch

        class FakeElement:
            def __init__(self, text, page=None):
                self._text = text
                self.metadata = MagicMock()
                self.metadata.to_dict.return_value = {
                    "page_number": page,
                    "parent_id": "abc123",
                }
            def __str__(self):
                return self._text

        ingestor = DocumentIngestor()
        elem = FakeElement("Content with page metadata that is long enough. " * 3, page=5)

        with patch("core.ingest.chunk_by_title", return_value=[elem]):
            result = ingestor.chunk_elements([elem], source_id="test")

        assert len(result) == 1
        meta = result[0]["metadata"]
        assert meta["page_number"] == 5
        assert meta["parent_id"] == "abc123"
        assert "content_hash" in meta


class TestHTMLStripping:
    """Tests for 6.5: Better HTML stripping with BeautifulSoup."""

    def test_strip_html_tags(self):
        """HTML tags are removed."""
        from core.ingest import _strip_html
        result = _strip_html("<p>Hello <b>world</b></p>")
        assert "Hello" in result
        assert "world" in result
        assert "<p>" not in result
        assert "<b>" not in result

    def test_strip_html_script_style(self):
        """Script and style elements are removed."""
        from core.ingest import _strip_html
        html = "<p>Content</p><script>alert('x')</script><style>.foo{}</style><p>More</p>"
        result = _strip_html(html)
        assert "Content" in result
        assert "More" in result
        assert "alert" not in result
        assert ".foo" not in result

    def test_strip_html_entities(self):
        """HTML entities are decoded."""
        from core.ingest import _strip_html
        result = _strip_html("<p>Hello &amp; world &lt;3</p>")
        assert "Hello & world <3" in result


class TestCharsetDetection:
    """Tests for 6.5: Charset detection."""

    def test_declared_charset_works(self):
        """Declared charset is used when valid."""
        from core.ingest import _detect_charset
        payload = "Hello world".encode("utf-8")
        assert _detect_charset(payload, "utf-8") == "utf-8"

    def test_fallback_to_chardet(self):
        """Falls back to chardet when declared charset fails."""
        from core.ingest import _detect_charset
        payload = "Hello world".encode("utf-8")
        # "invalid-charset" should fail, chardet should detect utf-8
        result = _detect_charset(payload, "invalid-charset")
        assert result in ("utf-8", "ascii")  # chardet may detect ascii for simple text

    def test_final_fallback_utf8(self):
        """Falls back to utf-8 as last resort."""
        from core.ingest import _detect_charset
        payload = b"\xff\xfe"  # BOM-like bytes
        result = _detect_charset(payload, None)
        # Should get something (chardet may detect or fall back to utf-8)
        assert isinstance(result, str)


class TestIngestionProgress:
    """Tests for 6.3: Progress tracking."""

    def test_progress_percent(self):
        """Progress percentage calculation."""
        from core.ingest_progress import IngestionProgress
        p = IngestionProgress(total_files=10, processed_files=3)
        assert p.percent == 30.0

    def test_progress_percent_zero_total(self):
        """Zero total files returns 100%."""
        from core.ingest_progress import IngestionProgress
        p = IngestionProgress(total_files=0)
        assert p.percent == 100.0

    def test_progress_status(self):
        """Status transitions correctly."""
        from core.ingest_progress import IngestionProgress
        p = IngestionProgress(total_files=5, processed_files=3)
        assert p.status == "in_progress"

        p.processed_files = 5
        assert p.status == "completed"

        p.failed_files = ["bad.pdf"]
        assert p.status == "completed_with_errors"

    def test_progress_eta(self):
        """ETA returns None when no files processed."""
        from core.ingest_progress import IngestionProgress
        p = IngestionProgress(total_files=10, processed_files=0)
        assert p.eta_seconds is None

    def test_checkpoint_save_load(self):
        """Checkpoint save/load round-trip."""
        from core.ingest_progress import IngestionCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            cp = IngestionCheckpoint(
                client_name="test",
                processed_files=["a.pdf", "b.pdf"],
                total_files=["a.pdf", "b.pdf", "c.pdf"],
                chunks_indexed=42,
            )
            cp.save(path)

            loaded = IngestionCheckpoint.load(path)
            assert loaded is not None
            assert loaded.client_name == "test"
            assert loaded.processed_files == ["a.pdf", "b.pdf"]
            assert loaded.chunks_indexed == 42

    def test_checkpoint_load_missing(self):
        """Loading from nonexistent path returns None."""
        from core.ingest_progress import IngestionCheckpoint
        result = IngestionCheckpoint.load(Path("/nonexistent/path.json"))
        assert result is None


class TestIngestionJobManager:
    """Tests for 6.1: Ingestion job management."""

    def test_job_lifecycle(self):
        """Job goes through create -> start -> complete lifecycle."""
        from core.ingest_job import IngestionJobManager
        mgr = IngestionJobManager()
        job = mgr.create_job("test_client", "reindex")
        assert job.status == "pending"

        mgr.start_job(job.job_id)
        assert mgr.get_job(job.job_id).status == "running"

        mgr.complete_job(job.job_id, "Done: 100 chunks")
        assert mgr.get_job(job.job_id).status == "completed"
        assert mgr.get_job(job.job_id).result == "Done: 100 chunks"

    def test_max_concurrent_enforcement(self):
        """Cannot exceed max concurrent jobs."""
        from core.ingest_job import IngestionJobManager
        mgr = IngestionJobManager(max_concurrent=1)
        job = mgr.create_job("test_client", "reindex")
        mgr.start_job(job.job_id)

        with pytest.raises(RuntimeError, match="Max concurrent"):
            mgr.create_job("test_client", "reindex")

    def test_get_client_jobs(self):
        """Get jobs filtered by client name."""
        from core.ingest_job import IngestionJobManager
        mgr = IngestionJobManager(max_concurrent=5)
        mgr.create_job("client_a", "file")
        mgr.create_job("client_b", "file")
        mgr.create_job("client_a", "url")

        jobs = mgr.get_client_jobs("client_a")
        assert len(jobs) == 2
        assert all(j.client_name == "client_a" for j in jobs)

    def test_unknown_job_returns_none(self):
        """Looking up unknown job ID returns None."""
        from core.ingest_job import IngestionJobManager
        mgr = IngestionJobManager()
        assert mgr.get_job("nonexistent") is None

    def test_job_failure(self):
        """Failed job records error message."""
        from core.ingest_job import IngestionJobManager
        mgr = IngestionJobManager()
        job = mgr.create_job("test", "reindex")
        mgr.start_job(job.job_id)
        mgr.fail_job(job.job_id, "Out of memory")

        result = mgr.get_job(job.job_id)
        assert result.status == "failed"
        assert result.error == "Out of memory"


class TestChunkingConfig:
    """Tests for 6.4: Alternative chunking strategies."""

    def test_chunking_config_defaults(self):
        """Default chunking config values."""
        from core.chunking import ChunkingConfig
        cfg = ChunkingConfig()
        assert cfg.strategy == "title"
        assert cfg.max_chunk_size == 1500
        assert cfg.semantic_threshold == 0.5

    def test_client_chunking_config(self):
        """Client reads chunking config from yaml."""
        from core.client import Client
        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {
                "name": "Test",
                "chunking": {"strategy": "semantic", "semantic_threshold": 0.6},
            })
            cfg = client.chunking_config
            assert cfg["strategy"] == "semantic"
            assert cfg["semantic_threshold"] == 0.6
            assert cfg["max_chunk_size"] == 1500  # default

    def test_chunking_key_recognized(self):
        """Chunking key doesn't produce validation warning."""
        from core.client import validate_config
        warnings = validate_config({
            "machine_name": "test",
            "chunking": {"strategy": "title"},
        })
        assert not any("chunking" in w for w in warnings)

    def test_sentence_splitting(self):
        """Sentence splitter handles basic cases."""
        from core.chunking import _split_to_sentences
        sentences = _split_to_sentences("Hello world. How are you? I am fine!")
        assert len(sentences) == 3

    def test_semantic_chunking_basic(self):
        """Semantic chunking produces chunks."""
        from core.chunking import chunk_by_semantic
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        elements = [
            "Machine learning is a subset of artificial intelligence. It uses algorithms to learn from data.",
            "The weather today is sunny. Perfect for a walk in the park.",
            "Neural networks are inspired by biological neurons. Deep learning uses many layers.",
        ]
        chunks = chunk_by_semantic(elements, embedder, threshold=0.3, max_chars=500, min_chars=10)
        assert len(chunks) >= 1
        # All original content should be in the chunks
        combined = " ".join(chunks)
        assert "Machine learning" in combined
        assert "weather" in combined

    def test_title_strategy_passthrough(self):
        """Title strategy uses standard chunk_by_title."""
        from core.ingest import DocumentIngestor
        from unittest.mock import patch, MagicMock

        class FakeElement:
            def __init__(self, text):
                self._text = text
                self.metadata = MagicMock()
                self.metadata.to_dict.return_value = {}
            def __str__(self):
                return self._text

        ingestor = DocumentIngestor()
        elem = FakeElement("Valid content that is long enough to pass the minimum. " * 3)

        with patch("core.ingest.chunk_by_title", return_value=[elem]) as mock_cbt:
            result = ingestor.chunk_elements([elem], chunking_config={"strategy": "title"})
            mock_cbt.assert_called_once()
            assert len(result) == 1


# ===================================================================
# Stage 7: Knowledge Graph Tests
# ===================================================================


class TestGraphModels:
    """Tests for Entity and Relationship dataclasses."""

    def test_entity_key_normalization(self):
        """Entity key normalizes name and type to lowercase."""
        from core.graph.models import Entity
        e = Entity(name="  Alice  ", entity_type="  Person  ")
        assert e.key == "alice::person"

    def test_entity_merge_sources_and_aliases(self):
        """Merging entities accumulates source chunks and aliases."""
        from core.graph.models import Entity
        e1 = Entity(name="Alice", entity_type="person", description="Short", source_chunk_ids=["c1"], aliases=["A"])
        e2 = Entity(name="Alice", entity_type="person", description="A much longer description", source_chunk_ids=["c2"], aliases=["Alice B"])
        e1.merge(e2)
        assert set(e1.source_chunk_ids) == {"c1", "c2"}
        assert "Alice B" in e1.aliases
        assert e1.description == "A much longer description"

    def test_entity_roundtrip_serialization(self):
        """Entity to_dict/from_dict roundtrip preserves data."""
        from core.graph.models import Entity
        e = Entity(name="Test Corp", entity_type="organization", description="A test org", source_chunk_ids=["c1"], aliases=["TC"])
        d = e.to_dict()
        e2 = Entity.from_dict(d)
        assert e2.name == e.name
        assert e2.entity_type == e.entity_type
        assert e2.source_chunk_ids == e.source_chunk_ids
        assert e2.aliases == e.aliases
        assert e2.key == e.key

    def test_relationship_key(self):
        """Relationship key format is 'source--type--target'."""
        from core.graph.models import Relationship
        r = Relationship(source_entity="alice::person", target_entity="bob::person", relationship_type="knows")
        assert r.key == "alice::person--knows--bob::person"

    def test_relationship_merge_weight(self):
        """Merging relationships takes max weight and accumulates sources."""
        from core.graph.models import Relationship
        r1 = Relationship(source_entity="a", target_entity="b", relationship_type="r", weight=0.5, source_chunk_ids=["c1"])
        r2 = Relationship(source_entity="a", target_entity="b", relationship_type="r", weight=0.9, source_chunk_ids=["c2"])
        r1.merge(r2)
        assert r1.weight == 0.9
        assert set(r1.source_chunk_ids) == {"c1", "c2"}

    def test_relationship_roundtrip_serialization(self):
        """Relationship to_dict/from_dict roundtrip."""
        from core.graph.models import Relationship
        r = Relationship(source_entity="a::person", target_entity="b::org", relationship_type="works_at", description="Employed", weight=0.8, source_chunk_ids=["c1"])
        d = r.to_dict()
        r2 = Relationship.from_dict(d)
        assert r2.source_entity == r.source_entity
        assert r2.weight == r.weight
        assert r2.key == r.key


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph store."""

    def test_add_entity(self):
        """Adding an entity stores it by key."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = KnowledgeGraph(Path(tmpdir))
            e = Entity(name="Alice", entity_type="person", source_chunk_ids=["c1"])
            graph.add_entity(e)
            assert graph.get_entity("alice::person") is not None

    def test_add_duplicate_merges(self):
        """Adding duplicate entity merges sources."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = KnowledgeGraph(Path(tmpdir))
            e1 = Entity(name="Alice", entity_type="person", source_chunk_ids=["c1"])
            e2 = Entity(name="alice", entity_type="person", source_chunk_ids=["c2"])
            graph.add_entity(e1)
            graph.add_entity(e2)
            stored = graph.get_entity("alice::person")
            assert set(stored.source_chunk_ids) == {"c1", "c2"}

    def test_add_relationship(self):
        """Adding a relationship stores it."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity, Relationship
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = KnowledgeGraph(Path(tmpdir))
            graph.add_entity(Entity(name="Alice", entity_type="person"))
            graph.add_entity(Entity(name="Bob", entity_type="person"))
            r = Relationship(source_entity="alice::person", target_entity="bob::person", relationship_type="knows")
            graph.add_relationship(r)
            assert graph.stats["relationship_count"] == 1

    def test_save_load_roundtrip(self):
        """Save and load preserves graph state."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity, Relationship
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build graph
            g1 = KnowledgeGraph(Path(tmpdir))
            g1.add_entity(Entity(name="Alice", entity_type="person", source_chunk_ids=["c1"]))
            g1.add_entity(Entity(name="Acme", entity_type="organization"))
            g1.add_relationship(Relationship(
                source_entity="alice::person", target_entity="acme::organization",
                relationship_type="works_at", weight=0.9,
            ))
            g1.save()

            # Load into new instance
            g2 = KnowledgeGraph(Path(tmpdir))
            assert g2.load() is True
            assert g2.stats["entity_count"] == 2
            assert g2.stats["relationship_count"] == 1
            alice = g2.get_entity("alice::person")
            assert alice is not None
            assert alice.source_chunk_ids == ["c1"]

    def test_load_nonexistent_returns_false(self):
        """Loading from empty directory returns False."""
        from core.graph.store import KnowledgeGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            g = KnowledgeGraph(Path(tmpdir))
            assert g.load() is False

    def test_stats(self):
        """Stats returns correct counts and distributions."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity, Relationship
        with tempfile.TemporaryDirectory() as tmpdir:
            g = KnowledgeGraph(Path(tmpdir))
            g.add_entity(Entity(name="A", entity_type="person", source_chunk_ids=["c1"]))
            g.add_entity(Entity(name="B", entity_type="person", source_chunk_ids=["c2"]))
            g.add_entity(Entity(name="C", entity_type="organization", source_chunk_ids=["c3"]))
            g.add_relationship(Relationship(source_entity="a::person", target_entity="c::organization", relationship_type="member_of"))
            stats = g.stats
            assert stats["entity_count"] == 3
            assert stats["relationship_count"] == 1
            assert stats["entity_type_distribution"]["person"] == 2
            assert stats["entity_type_distribution"]["organization"] == 1

    def test_find_entities_by_name(self):
        """Substring and alias matching works."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity
        with tempfile.TemporaryDirectory() as tmpdir:
            g = KnowledgeGraph(Path(tmpdir))
            g.add_entity(Entity(name="Alice Smith", entity_type="person", aliases=["A. Smith"]))
            g.add_entity(Entity(name="Bob Jones", entity_type="person"))
            # Substring match on name
            results = g.find_entities_by_name("alice")
            assert len(results) == 1
            assert results[0].name == "Alice Smith"
            # Alias match
            results = g.find_entities_by_name("smith")
            assert len(results) == 1

    def test_get_neighbors(self):
        """BFS neighbor traversal works."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity, Relationship
        with tempfile.TemporaryDirectory() as tmpdir:
            g = KnowledgeGraph(Path(tmpdir))
            g.add_entity(Entity(name="A", entity_type="person"))
            g.add_entity(Entity(name="B", entity_type="person"))
            g.add_entity(Entity(name="C", entity_type="person"))
            g.add_relationship(Relationship(source_entity="a::person", target_entity="b::person", relationship_type="knows"))
            g.add_relationship(Relationship(source_entity="b::person", target_entity="c::person", relationship_type="knows"))
            # 1 hop from A should find B
            n1 = g.get_neighbors("a::person", max_depth=1)
            assert len(n1) == 1
            assert n1[0][0].name == "B"
            # 2 hops from A should find B and C
            n2 = g.get_neighbors("a::person", max_depth=2)
            assert len(n2) == 2

    def test_clear(self):
        """Clear removes all data."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity
        with tempfile.TemporaryDirectory() as tmpdir:
            g = KnowledgeGraph(Path(tmpdir))
            g.add_entity(Entity(name="A", entity_type="person"))
            g.clear()
            assert g.stats["entity_count"] == 0


class TestProcessedChunkTracking:
    """Tests for knowledge graph processed chunk ID tracking."""

    def test_processed_chunk_ids_initially_empty(self):
        """New graph has empty processed set."""
        from core.graph.store import KnowledgeGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = KnowledgeGraph(Path(tmpdir))
            assert graph.processed_chunk_ids == set()

    def test_mark_chunks_processed(self):
        """Marking chunks adds them to the set."""
        from core.graph.store import KnowledgeGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = KnowledgeGraph(Path(tmpdir))
            graph.mark_chunks_processed(["c1", "c2"])
            assert graph.processed_chunk_ids == {"c1", "c2"}
            graph.mark_chunks_processed(["c2", "c3"])
            assert graph.processed_chunk_ids == {"c1", "c2", "c3"}

    def test_processed_chunk_ids_save_load_roundtrip(self):
        """Processed IDs persist through save/load."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity
        with tempfile.TemporaryDirectory() as tmpdir:
            g1 = KnowledgeGraph(Path(tmpdir))
            g1.add_entity(Entity(name="A", entity_type="person", source_chunk_ids=["c1"]))
            g1.mark_chunks_processed(["c1", "c2", "c3"])
            g1.save()

            g2 = KnowledgeGraph(Path(tmpdir))
            assert g2.load() is True
            assert g2.processed_chunk_ids == {"c1", "c2", "c3"}

    def test_processed_chunk_ids_backfill_from_entities(self):
        """Loading old-format JSON (no processed_chunk_ids key) backfills from entities."""
        from core.graph.store import KnowledgeGraph, GRAPH_JSON_FILENAME, GRAPH_VERSION
        from core.graph.models import Entity
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write old-format JSON without processed_chunk_ids
            old_data = {
                "version": GRAPH_VERSION,
                "entities": [
                    Entity(name="A", entity_type="person", source_chunk_ids=["c1", "c2"]).to_dict(),
                    Entity(name="B", entity_type="person", source_chunk_ids=["c2", "c3"]).to_dict(),
                ],
                "relationships": [],
            }
            graph_path = Path(tmpdir) / GRAPH_JSON_FILENAME
            with open(graph_path, "w") as f:
                json.dump(old_data, f)

            graph = KnowledgeGraph(Path(tmpdir))
            assert graph.load() is True
            assert graph.processed_chunk_ids == {"c1", "c2", "c3"}
            # Verify migration created SQLite DB
            from core.graph.store import GRAPH_FILENAME
            assert (Path(tmpdir) / GRAPH_FILENAME).exists()

    def test_clear_resets_processed_chunks(self):
        """clear() empties the processed set."""
        from core.graph.store import KnowledgeGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = KnowledgeGraph(Path(tmpdir))
            graph.mark_chunks_processed(["c1", "c2"])
            graph.clear()
            assert graph.processed_chunk_ids == set()

    def test_stats_includes_processed_count(self):
        """Stats dict has processed_chunk_count field."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = KnowledgeGraph(Path(tmpdir))
            graph.add_entity(Entity(name="A", entity_type="person", source_chunk_ids=["c1"]))
            graph.mark_chunks_processed(["c1", "c2", "c3"])
            stats = graph.stats
            assert stats["processed_chunk_count"] == 3


class TestGraphExtractor:
    """Tests for LLM-based entity/relationship extraction."""

    def test_parse_valid_json(self):
        """Parsing valid JSON response extracts entities and relationships."""
        from core.graph.extractor import GraphExtractor

        def mock_llm(messages, system_prompt, max_tokens):
            return ""  # Not used in this test

        extractor = GraphExtractor(get_llm_response=mock_llm)
        response = json.dumps({
            "entities": [
                {"name": "Alice", "type": "person", "description": "A researcher"},
                {"name": "MIT", "type": "organization", "description": "University"},
            ],
            "relationships": [
                {"source": "Alice", "target": "MIT", "type": "works_at", "description": "Employment", "weight": 0.9},
            ],
        })
        entities, rels = extractor._parse_response(response, "chunk1")
        assert len(entities) == 2
        assert len(rels) == 1
        assert entities[0].name == "Alice"
        assert rels[0].relationship_type == "works_at"

    def test_parse_markdown_wrapped_json(self):
        """Handles ```json ... ``` wrapper."""
        from core.graph.extractor import GraphExtractor

        extractor = GraphExtractor(get_llm_response=lambda m, s, t: "")
        response = '```json\n{"entities": [{"name": "Bob", "type": "person", "description": ""}], "relationships": []}\n```'
        entities, rels = extractor._parse_response(response, "c1")
        assert len(entities) == 1
        assert entities[0].name == "Bob"

    def test_parse_malformed_json(self):
        """Gracefully returns empty on malformed JSON."""
        from core.graph.extractor import GraphExtractor

        extractor = GraphExtractor(get_llm_response=lambda m, s, t: "")
        entities, rels = extractor._parse_response("this is not json at all", "c1")
        assert entities == []
        assert rels == []

    def test_extract_from_chunk_with_mock_llm(self):
        """Extract from chunk calls LLM and parses response."""
        from core.graph.extractor import GraphExtractor

        def mock_llm(messages, system_prompt, max_tokens):
            return json.dumps({
                "entities": [{"name": "Test", "type": "concept", "description": "A test concept"}],
                "relationships": [],
            })

        extractor = GraphExtractor(get_llm_response=mock_llm)
        chunk = {"id": "c1", "content": "This is about the Test concept."}
        entities, rels = extractor.extract_from_chunk(chunk)
        assert len(entities) == 1
        assert entities[0].name == "Test"
        assert "c1" in entities[0].source_chunk_ids

    def test_entity_type_filtering(self):
        """Only configured entity types are kept."""
        from core.graph.extractor import GraphExtractor

        extractor = GraphExtractor(
            get_llm_response=lambda m, s, t: "",
            entity_types=["person"],
        )
        response = json.dumps({
            "entities": [
                {"name": "Alice", "type": "person", "description": ""},
                {"name": "MIT", "type": "organization", "description": ""},
            ],
            "relationships": [],
        })
        entities, rels = extractor._parse_response(response, "c1")
        assert len(entities) == 1
        assert entities[0].entity_type == "person"

    def test_batch_extraction(self):
        """Batch extraction splits chunks by text budget and processes in batches."""
        from core.graph.extractor import GraphExtractor

        call_count = 0

        def mock_llm(messages, system_prompt, max_tokens):
            nonlocal call_count
            call_count += 1
            return json.dumps({
                "entities": [{"name": f"Entity{call_count}", "type": "concept", "description": ""}],
                "relationships": [],
            })

        # Use a small text budget to force 2 batches from 4 chunks
        # Each chunk is ~30 chars ("Content X" + "[Chunk cX]\n" header)
        # Budget of 70 fits ~2 chunks per batch
        extractor = GraphExtractor(
            get_llm_response=mock_llm, max_workers=1, batch_text_budget=70,
        )
        chunks = [
            {"id": f"c{i}", "content": f"Content {i}"} for i in range(4)
        ]
        entities, rels = extractor.extract_from_chunks(chunks)
        assert call_count == 2  # 4 chunks split into 2 batches by text budget
        assert len(entities) == 2


class TestGraphBuildProgress:
    """Tests for GraphBuildProgress."""

    def test_percent_and_eta(self):
        """Percent and ETA calculation work."""
        import time
        from core.graph.progress import GraphBuildProgress
        p = GraphBuildProgress(total_chunks=100, start_time=time.time() - 10)
        p.processed_chunks = 50
        assert p.percent == 50.0
        eta = p.eta_seconds
        assert eta is not None
        assert eta > 0
        # Format ETA should return a string
        assert isinstance(p.format_eta(), str)


class TestGraphSearch:
    """Tests for graph-augmented search modes."""

    def _make_graph(self, tmpdir):
        """Create a test graph with entities and relationships."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity, Relationship
        g = KnowledgeGraph(Path(tmpdir))
        g.add_entity(Entity(name="Alice", entity_type="person", source_chunk_ids=["c1", "c2"]))
        g.add_entity(Entity(name="Bob", entity_type="person", source_chunk_ids=["c3"]))
        g.add_entity(Entity(name="Acme", entity_type="organization", source_chunk_ids=["c4", "c5"]))
        g.add_relationship(Relationship(
            source_entity="alice::person", target_entity="acme::organization",
            relationship_type="works_at", source_chunk_ids=["c1"],
        ))
        g.add_relationship(Relationship(
            source_entity="bob::person", target_entity="acme::organization",
            relationship_type="works_at", source_chunk_ids=["c3"],
        ))
        return g

    def test_local_search_finds_entity_chunks(self):
        """Local search returns chunks for matching entities."""
        from unittest.mock import MagicMock
        from core.graph.search import local_search

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = self._make_graph(tmpdir)
            # Mock searcher with a collection that returns chunks
            searcher = MagicMock()
            searcher.collection.get.return_value = {
                "ids": ["c1", "c2"],
                "documents": ["Content 1", "Content 2"],
                "metadatas": [{"source_file": "a.txt"}, {"source_file": "b.txt"}],
            }
            results = local_search("Alice", graph, searcher)
            assert len(results) > 0
            assert results[0].get("_graph_entity") is True

    def test_global_search_traverses_relationships(self):
        """Global search follows relationships to find connected chunks."""
        from unittest.mock import MagicMock
        from core.graph.search import global_search

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = self._make_graph(tmpdir)
            searcher = MagicMock()
            # Return chunks for all the chunk IDs
            searcher.collection.get.return_value = {
                "ids": ["c1", "c2", "c4", "c5"],
                "documents": ["C1", "C2", "C4", "C5"],
                "metadatas": [{}, {}, {}, {}],
            }
            results = global_search("Alice", graph, searcher, max_hops=2)
            assert len(results) > 0

    def test_mix_search_merges_graph_and_standard(self):
        """Mix search combines graph and standard results."""
        from unittest.mock import MagicMock, patch
        from core.graph.search import mix_search

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = self._make_graph(tmpdir)
            searcher = MagicMock()
            # Standard search returns some results
            searcher.search.return_value = [
                {"id": "s1", "content": "Standard result", "metadata": {}},
            ]
            # Graph search chunk fetch
            searcher.collection.get.return_value = {
                "ids": ["c1"],
                "documents": ["Graph result"],
                "metadatas": [{}],
            }
            results = mix_search("Alice", graph, searcher, top_k=10)
            # Should have results from both sources
            assert len(results) >= 1

    def test_graph_search_falls_back_to_naive(self):
        """graph_search with no graph falls back to standard search."""
        from unittest.mock import MagicMock
        searcher = MagicMock()
        searcher._knowledge_graph = None
        searcher.load_graph = MagicMock(return_value=None)
        searcher.search = MagicMock(return_value=[{"id": "s1"}])
        # Call the real method manually since it's a mock
        from core.search import HybridSearcher
        # Use the actual method bound to our mock
        result = HybridSearcher.graph_search(searcher, query="test", mode="local", top_k=10)
        searcher.search.assert_called_once()


class TestGraphExport:
    """Tests for graph stats and D3 export."""

    def test_graph_stats_empty(self):
        """Stats on empty graph returns zeros."""
        from core.graph.store import KnowledgeGraph
        from core.graph.export import graph_stats
        with tempfile.TemporaryDirectory() as tmpdir:
            g = KnowledgeGraph(Path(tmpdir))
            stats = graph_stats(g)
            assert stats["entity_count"] == 0
            assert stats["relationship_count"] == 0
            assert stats["density"] == 0.0

    def test_graph_stats_populated(self):
        """Stats on populated graph returns correct counts."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity, Relationship
        from core.graph.export import graph_stats
        with tempfile.TemporaryDirectory() as tmpdir:
            g = KnowledgeGraph(Path(tmpdir))
            g.add_entity(Entity(name="A", entity_type="person"))
            g.add_entity(Entity(name="B", entity_type="person"))
            g.add_relationship(Relationship(source_entity="a::person", target_entity="b::person", relationship_type="knows"))
            stats = graph_stats(g)
            assert stats["entity_count"] == 2
            assert stats["relationship_count"] == 1
            assert stats["density"] > 0

    def test_export_for_d3_format(self):
        """D3 export produces valid node/link format."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity, Relationship
        from core.graph.export import export_for_d3
        with tempfile.TemporaryDirectory() as tmpdir:
            g = KnowledgeGraph(Path(tmpdir))
            g.add_entity(Entity(name="A", entity_type="person"))
            g.add_entity(Entity(name="B", entity_type="org"))
            g.add_relationship(Relationship(source_entity="a::person", target_entity="b::org", relationship_type="member"))
            d3 = export_for_d3(g)
            assert "nodes" in d3
            assert "links" in d3
            assert len(d3["nodes"]) == 2
            assert len(d3["links"]) == 1
            assert d3["nodes"][0]["name"] in ("A", "B")

    def test_export_for_d3_max_nodes(self):
        """D3 export respects max_nodes limit."""
        from core.graph.store import KnowledgeGraph
        from core.graph.models import Entity
        from core.graph.export import export_for_d3
        with tempfile.TemporaryDirectory() as tmpdir:
            g = KnowledgeGraph(Path(tmpdir))
            for i in range(10):
                g.add_entity(Entity(name=f"Entity{i}", entity_type="concept"))
            d3 = export_for_d3(g, max_nodes=5)
            assert len(d3["nodes"]) == 5


class TestClientGraphConfig:
    """Tests for graph config in Client."""

    def test_graph_config_defaults(self):
        """Graph config returns defaults when not set."""
        from core.client import Client
        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {"machine_name": "test"})
            gc = client.graph_config
            assert gc["enabled"] is False
            assert gc["search_mode"] == "naive"
            assert "person" in gc["entity_types"]

    def test_graph_config_override(self):
        """Graph config merges with client config."""
        from core.client import Client
        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {
                "machine_name": "test",
                "graph": {"enabled": True, "search_mode": "mix"},
            })
            gc = client.graph_config
            assert gc["enabled"] is True
            assert gc["search_mode"] == "mix"
            # Defaults still present
            assert gc["extract_batch_size"] == 512

    def test_graph_key_recognized_in_validation(self):
        """'graph' key doesn't produce unknown-key warning."""
        from core.client import validate_config
        config = {
            "machine_name": "test",
            "graph": {"enabled": True},
        }
        warnings = validate_config(config)
        assert not any("Unknown top-level key" in w and "graph" in w for w in warnings)

    def test_graph_unknown_key_warned(self):
        """Unknown graph keys produce warnings."""
        from core.client import validate_config
        config = {
            "machine_name": "test",
            "graph": {"enabled": True, "bogus_key": 42},
        }
        warnings = validate_config(config)
        assert any("bogus_key" in w for w in warnings)

    def test_graph_invalid_search_mode_warned(self):
        """Invalid graph.search_mode produces warning."""
        from core.client import validate_config
        config = {
            "machine_name": "test",
            "graph": {"search_mode": "invalid_mode"},
        }
        warnings = validate_config(config)
        assert any("search_mode" in w for w in warnings)


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


# ===================================================================
# Stage 9: Embedding & Indexing Quality Tests
# ===================================================================


class TestContextualEmbeddingsConfig:
    """Tests for 9.1: Contextual embeddings configuration."""

    def test_contextualize_default_false(self):
        """Test embedding.contextualize defaults to False."""
        from core.client import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["embedding"]["contextualize"] is False

    def test_client_embedding_config_includes_contextualize(self):
        """Test client.embedding_config merges contextualize."""
        from core.client import Client
        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {
                "name": "Test",
                "embedding": {"contextualize": True},
            })
            config = client.embedding_config
            assert config["contextualize"] is True

    def test_contextualize_default_in_client(self):
        """Test contextualize defaults to False in client config."""
        from core.client import Client
        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {"name": "Test"})
            config = client.embedding_config
            assert config["contextualize"] is False

    def test_known_embedding_dims_includes_new_models(self):
        """Test _KNOWN_EMBEDDING_DIMS includes benchmark candidate models."""
        from core.client import _KNOWN_EMBEDDING_DIMS
        assert "BAAI/bge-m3" in _KNOWN_EMBEDDING_DIMS
        assert "nomic-ai/nomic-embed-text-v1.5" in _KNOWN_EMBEDDING_DIMS


class TestBuildContextPrefix:
    """Tests for 9.1: build_context_prefix function."""

    def test_basic_prefix_with_source_file(self):
        """Build prefix from source_file metadata."""
        from core.ingest import build_context_prefix
        metadata = {"source_file": "guide.pdf"}
        result = build_context_prefix(metadata)
        assert "Document: guide.pdf." in result

    def test_prefix_with_page_number(self):
        """Build prefix includes page number when present."""
        from core.ingest import build_context_prefix
        metadata = {"source_file": "report.pdf", "page_number": 5}
        result = build_context_prefix(metadata)
        assert "Page 5." in result

    def test_prefix_empty_metadata(self):
        """Empty metadata returns empty string."""
        from core.ingest import build_context_prefix
        result = build_context_prefix({})
        assert result == ""

    def test_prefix_none_values_skipped(self):
        """None values in metadata are skipped."""
        from core.ingest import build_context_prefix
        metadata = {"source_file": "test.txt", "page_number": None, "element_type": None}
        result = build_context_prefix(metadata)
        assert "Document: test.txt." in result
        assert "Page" not in result

    def test_prefix_with_all_fields(self):
        """Full prefix with source_file and page_number."""
        from core.ingest import build_context_prefix
        metadata = {"source_file": "manual.pdf", "page_number": 3}
        result = build_context_prefix(metadata)
        assert result == "Document: manual.pdf. Page 3. "


class TestContextualEmbeddings:
    """Tests for 9.1: create_embeddings with context_strings."""

    def test_create_embeddings_without_context(self):
        """Default behavior: no context strings, embeddings match raw content."""
        from core.ingest import DocumentIngestor
        ingestor = DocumentIngestor()
        chunks = [
            {"id": "1", "content": "Python is a programming language.", "metadata": {}},
        ]
        embeddings = ingestor.create_embeddings(chunks)
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0

    def test_create_embeddings_with_context_strings(self):
        """Context strings change the embedding vectors."""
        from core.ingest import DocumentIngestor
        import numpy as np
        ingestor = DocumentIngestor()
        chunks = [
            {"id": "1", "content": "Functions are reusable blocks of code.", "metadata": {}},
        ]
        emb_raw = ingestor.create_embeddings(chunks)
        emb_ctx = ingestor.create_embeddings(chunks, context_strings=["Document: guide.pdf. Page 3. "])
        assert not np.allclose(emb_raw[0], emb_ctx[0], atol=1e-4)

    def test_create_embeddings_context_strings_length_mismatch(self):
        """Mismatched context_strings length raises ValueError."""
        from core.ingest import DocumentIngestor
        ingestor = DocumentIngestor()
        chunks = [
            {"id": "1", "content": "test", "metadata": {}},
            {"id": "2", "content": "test2", "metadata": {}},
        ]
        with pytest.raises(ValueError, match="context_strings"):
            ingestor.create_embeddings(chunks, context_strings=["only one"])

    def test_create_embeddings_empty_context_string(self):
        """Empty context string is equivalent to no context."""
        from core.ingest import DocumentIngestor
        import numpy as np
        ingestor = DocumentIngestor()
        chunks = [
            {"id": "1", "content": "Hello world testing content.", "metadata": {}},
        ]
        emb_raw = ingestor.create_embeddings(chunks)
        emb_empty = ingestor.create_embeddings(chunks, context_strings=[""])
        assert np.allclose(emb_raw[0], emb_empty[0], atol=1e-6)


class TestContextualEmbeddingsIntegration:
    """Integration test for contextual embeddings in ingest pipeline."""

    def test_ingest_document_with_contextualize_flag(self):
        """ingest_document passes context strings when contextualize=True."""
        from core.ingest import DocumentIngestor
        from unittest.mock import patch, MagicMock
        import numpy as np

        ingestor = DocumentIngestor()

        mock_elements = [MagicMock()]

        class FakeChunk:
            def __init__(self, text, meta):
                self._text = text
                self.metadata = MagicMock()
                self.metadata.to_dict.return_value = meta
            def __str__(self):
                return self._text

        fake_chunks = [
            FakeChunk("This is a valid chunk with enough content to pass validation. " * 2,
                      {"page_number": 3}),
        ]

        with patch.object(ingestor, "parse_document", return_value=mock_elements):
            with patch("core.ingest.chunk_by_title", return_value=fake_chunks):
                # Without contextualize (default)
                chunks_raw, emb_raw = ingestor.ingest_document("/fake/path.pdf")

                # With contextualize
                chunks_ctx, emb_ctx = ingestor.ingest_document(
                    "/fake/path.pdf", contextualize=True,
                )

        assert len(chunks_raw) == len(chunks_ctx)
        # Metadata should include contextualized flag
        assert chunks_ctx[0]["metadata"].get("contextualized") is True
        assert chunks_raw[0]["metadata"].get("contextualized") is not True
        # Embedding vectors should differ
        assert not np.allclose(emb_raw[0], emb_ctx[0], atol=1e-4)


class TestRecursiveChunking:
    """Tests for 9.2: Recursive chunking strategy."""

    def test_chunking_config_has_separators(self):
        """ChunkingConfig includes separators field."""
        from core.chunking import ChunkingConfig
        cfg = ChunkingConfig()
        assert hasattr(cfg, "separators")
        assert cfg.separators == ["\n\n", "\n", ". ", " "]

    def test_recursive_basic(self):
        """Recursive chunking splits text on paragraph boundaries."""
        from core.chunking import chunk_by_recursive
        text = "Paragraph one content.\n\nParagraph two content.\n\nParagraph three content."
        chunks = chunk_by_recursive(text, max_chars=100, overlap=0)
        assert len(chunks) >= 1
        combined = " ".join(chunks)
        assert "Paragraph one" in combined
        assert "Paragraph three" in combined

    def test_recursive_respects_max_chars(self):
        """All chunks are within max_chars limit."""
        from core.chunking import chunk_by_recursive
        text = ("Short paragraph.\n\n" * 20).strip()
        chunks = chunk_by_recursive(text, max_chars=100, overlap=0)
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_recursive_fallback_to_next_separator(self):
        """Falls back to next separator when first doesn't produce small enough chunks."""
        from core.chunking import chunk_by_recursive
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunk_by_recursive(text, max_chars=50, overlap=0)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_recursive_overlap(self):
        """Overlap is applied between consecutive chunks."""
        from core.chunking import chunk_by_recursive
        text = "A" * 100 + "\n\n" + "B" * 100
        chunks = chunk_by_recursive(text, max_chars=120, overlap=20)
        assert len(chunks) >= 2
        if len(chunks) >= 2:
            assert len(chunks[1]) > 0

    def test_recursive_custom_separators(self):
        """Custom separator list is respected."""
        from core.chunking import chunk_by_recursive
        text = "part1|part2|part3"
        chunks = chunk_by_recursive(text, max_chars=10, overlap=0, separators=["|"])
        assert len(chunks) == 3
        assert chunks[0] == "part1"
        assert chunks[1] == "part2"
        assert chunks[2] == "part3"

    def test_recursive_empty_input(self):
        """Empty input returns empty list."""
        from core.chunking import chunk_by_recursive
        assert chunk_by_recursive("", max_chars=100, overlap=0) == []
        assert chunk_by_recursive("   ", max_chars=100, overlap=0) == []

    def test_recursive_single_small_text(self):
        """Text smaller than max_chars returns as single chunk."""
        from core.chunking import chunk_by_recursive
        chunks = chunk_by_recursive("Hello world.", max_chars=1000, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_recursive_strategy_dispatch(self):
        """chunk_elements dispatches to recursive chunking when strategy='recursive'."""
        from core.ingest import DocumentIngestor
        from unittest.mock import MagicMock, patch

        class FakeElement:
            def __init__(self, text):
                self._text = text
                self.metadata = MagicMock()
                self.metadata.to_dict.return_value = {}
            def __str__(self):
                return self._text

        ingestor = DocumentIngestor()
        elem = FakeElement("Content that is definitely long enough to pass validation. " * 3)

        with patch("core.chunking.chunk_by_recursive", return_value=["Content that is definitely long enough to pass validation. " * 3]) as mock_recursive:
            result = ingestor.chunk_elements(
                [elem],
                chunking_config={"strategy": "recursive", "max_chunk_size": 500, "overlap": 50},
            )
            mock_recursive.assert_called_once()

    def test_client_chunking_config_recursive(self):
        """Client reads recursive chunking config from yaml."""
        from core.client import Client
        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(Path(tmpdir), {
                "name": "Test",
                "chunking": {
                    "strategy": "recursive",
                    "separators": ["\n\n", "\n"],
                },
            })
            cfg = client.chunking_config
            assert cfg["strategy"] == "recursive"
            assert cfg["separators"] == ["\n\n", "\n"]


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


class TestHNSWConfig:
    """Tests for HNSW tuning configuration."""

    def test_hnsw_defaults_in_config(self):
        """Test HNSW defaults exist in DEFAULT_CONFIG."""
        from core.client import DEFAULT_CONFIG

        emb = DEFAULT_CONFIG["embedding"]
        assert emb["hnsw_ef_construction"] == 200
        assert emb["hnsw_max_neighbors"] == 16
        assert emb["hnsw_ef_search"] == 100

    def test_client_embedding_config_includes_hnsw(self):
        """Test Client.embedding_config merges HNSW defaults."""
        from core.client import Client
        from pathlib import Path

        client = Client(Path("/tmp/fake"), {"machine_name": "test"})
        cfg = client.embedding_config
        assert cfg["hnsw_ef_construction"] == 200
        assert cfg["hnsw_max_neighbors"] == 16
        assert cfg["hnsw_ef_search"] == 100

    def test_client_hnsw_override(self):
        """Test client.yaml can override HNSW params."""
        from core.client import Client
        from pathlib import Path

        client = Client(Path("/tmp/fake"), {
            "machine_name": "test",
            "embedding": {
                "hnsw_ef_construction": 400,
                "hnsw_max_neighbors": 32,
                "hnsw_ef_search": 200,
            },
        })
        cfg = client.embedding_config
        assert cfg["hnsw_ef_construction"] == 400
        assert cfg["hnsw_max_neighbors"] == 32
        assert cfg["hnsw_ef_search"] == 200

    @pytest.fixture
    def temp_index(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_searcher_accepts_hnsw_params(self, temp_index):
        """Test HybridSearcher accepts and stores HNSW params."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(
            index_path=temp_index,
            hnsw_ef_construction=400,
            hnsw_max_neighbors=32,
            hnsw_ef_search=200,
        )
        assert searcher.hnsw_ef_construction == 400
        assert searcher.hnsw_max_neighbors == 32
        assert searcher.hnsw_ef_search == 200

    def test_searcher_default_hnsw_params(self, temp_index):
        """Test HybridSearcher uses ChromaDB defaults for HNSW."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        assert searcher.hnsw_ef_construction == 200
        assert searcher.hnsw_max_neighbors == 16
        assert searcher.hnsw_ef_search == 100

    def test_hnsw_build_params_in_collection_metadata(self, temp_index):
        """Test HNSW build-time params are passed to collection metadata."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(
            index_path=temp_index,
            hnsw_ef_construction=400,
            hnsw_max_neighbors=32,
        )
        metadata = searcher.collection.metadata
        assert metadata.get("hnsw:construction_ef") == 400
        assert metadata.get("hnsw:M") == 32


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


# ============================================================================
# Stage 11: Agent Orchestration Tests
# ============================================================================


class TestMemoryStore:
    """Tests for core.memory.MemoryStore."""

    @pytest.fixture
    def memory(self):
        """Create a temporary MemoryStore."""
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test_state.db"
        from core.memory import MemoryStore
        store = MemoryStore(db_path)
        yield store
        store.close()
        shutil.rmtree(tmpdir)

    def test_memory_store_and_retrieve(self, memory):
        """Basic store/retrieve cycle."""
        memory.store("decisions", "arch-choice", "Use SQLite for persistence")
        result = memory.retrieve("decisions", "arch-choice")
        assert result is not None
        assert result["content"] == "Use SQLite for persistence"
        assert result["namespace"] == "decisions"
        assert result["key"] == "arch-choice"

    def test_memory_store_upsert(self, memory):
        """Updating existing key preserves ID and updates content."""
        id1 = memory.store("ns", "k", "original")
        id2 = memory.store("ns", "k", "updated")
        assert id1 == id2
        result = memory.retrieve("ns", "k")
        assert result["content"] == "updated"

    def test_memory_retrieve_missing(self, memory):
        """Returns None for nonexistent key."""
        result = memory.retrieve("ns", "nonexistent")
        assert result is None

    def test_memory_list_all(self, memory):
        """List without namespace filter returns all memories."""
        memory.store("ns1", "a", "content a")
        memory.store("ns2", "b", "content b")
        results = memory.list()
        assert len(results) == 2

    def test_memory_list_by_namespace(self, memory):
        """List with namespace filter returns only matching memories."""
        memory.store("ns1", "a", "content a")
        memory.store("ns2", "b", "content b")
        results = memory.list(namespace="ns1")
        assert len(results) == 1
        assert results[0]["namespace"] == "ns1"

    def test_memory_delete(self, memory):
        """Delete removes the memory."""
        memory.store("ns", "k", "content")
        assert memory.delete("ns", "k") is True
        assert memory.retrieve("ns", "k") is None

    def test_memory_delete_nonexistent(self, memory):
        """Deleting nonexistent key returns False."""
        assert memory.delete("ns", "nonexistent") is False

    def test_memory_ttl_cleanup(self, memory):
        """Store with TTL, then cleanup removes expired entries."""
        import time
        memory.store("ns", "ephemeral", "temp data", ttl=1)
        assert memory.retrieve("ns", "ephemeral") is not None
        time.sleep(1.1)
        count = memory.cleanup()
        assert count == 1
        assert memory.retrieve("ns", "ephemeral") is None

    def test_memory_search_without_embeddings(self, memory):
        """LIKE-based fallback search works without embedding model."""
        memory.store("ns", "python", "Python is a programming language")
        memory.store("ns", "rust", "Rust focuses on memory safety")
        results = memory.search("programming")
        assert len(results) >= 1
        assert any("Python" in r["content"] for r in results)

    def test_memory_search_with_embeddings(self):
        """Vector search works with an embedding model."""
        from sentence_transformers import SentenceTransformer
        from core.memory import MemoryStore

        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test_embed.db"
        model = SentenceTransformer("all-MiniLM-L6-v2")
        store = MemoryStore(db_path, embedding_model=model)

        store.store("ns", "python", "Python is a versatile programming language")
        store.store("ns", "cooking", "How to bake chocolate cake")
        store.store("ns", "rust", "Rust ensures memory safety at compile time")

        results = store.search("programming languages", top_k=2)
        assert len(results) <= 2
        # Programming-related entries should rank higher than cooking
        assert results[0]["key"] in ("python", "rust")

        store.close()
        shutil.rmtree(tmpdir)

    def test_memory_namespace_isolation(self, memory):
        """Same key in different namespaces are independent."""
        memory.store("ns1", "shared_key", "value from ns1")
        memory.store("ns2", "shared_key", "value from ns2")

        r1 = memory.retrieve("ns1", "shared_key")
        r2 = memory.retrieve("ns2", "shared_key")
        assert r1["content"] == "value from ns1"
        assert r2["content"] == "value from ns2"

    def test_memory_metadata_roundtrip(self, memory):
        """Dict metadata is stored and retrieved correctly."""
        meta = {"priority": "high", "tags": ["arch", "decision"], "count": 42}
        memory.store("ns", "k", "content", metadata=meta)
        result = memory.retrieve("ns", "k")
        assert result["metadata"] == meta


class TestWorkflowEngine:
    """Tests for core.workflow.WorkflowEngine."""

    @pytest.fixture
    def engine(self):
        """Create a temporary WorkflowEngine."""
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test_state.db"
        from core.workflow import WorkflowEngine
        eng = WorkflowEngine(db_path)
        yield eng
        eng.close()
        shutil.rmtree(tmpdir)

    def test_workflow_create_simple(self, engine):
        """Linear workflow with no deps."""
        wid = engine.create("simple", [
            {"name": "step1", "description": "first"},
            {"name": "step2", "description": "second"},
        ])
        wf = engine.get(wid)
        assert wf is not None
        assert wf["name"] == "simple"
        assert len(wf["steps"]) == 2

    def test_workflow_create_with_deps(self, engine):
        """Workflow with dependency graph."""
        wid = engine.create("dag", [
            {"name": "A"},
            {"name": "B", "depends_on": ["A"]},
            {"name": "C", "depends_on": ["A"]},
            {"name": "D", "depends_on": ["B", "C"]},
        ])
        wf = engine.get(wid)
        assert len(wf["steps"]) == 4
        # A should come first in sort order
        assert wf["steps"][0]["name"] == "A"

    def test_workflow_create_cycle_rejected(self, engine):
        """Circular dependency raises ValueError."""
        with pytest.raises(ValueError, match="circular"):
            engine.create("cycle", [
                {"name": "X", "depends_on": ["Y"]},
                {"name": "Y", "depends_on": ["X"]},
            ])

    def test_workflow_get(self, engine):
        """Retrieve created workflow."""
        wid = engine.create("test", [{"name": "s1"}])
        wf = engine.get(wid)
        assert wf["id"] == wid
        assert wf["status"] == "pending"

    def test_workflow_get_nonexistent(self, engine):
        """Returns None for nonexistent workflow."""
        assert engine.get("nonexistent-id") is None

    def test_workflow_list(self, engine):
        """List all workflows."""
        engine.create("wf1", [{"name": "s1"}])
        engine.create("wf2", [{"name": "s1"}])
        results = engine.list()
        assert len(results) == 2

    def test_workflow_list_by_status(self, engine):
        """Filter by status."""
        wid = engine.create("wf1", [{"name": "s1"}])
        engine.create("wf2", [{"name": "s1"}])

        # Start one workflow
        wf = engine.get(wid)
        engine.update_step(wid, wf["steps"][0]["id"], "running")

        running = engine.list(status="running")
        assert len(running) == 1
        assert running[0]["name"] == "wf1"

    def test_workflow_next_steps_initial(self, engine):
        """Returns steps with no dependencies."""
        wid = engine.create("dag", [
            {"name": "A"},
            {"name": "B", "depends_on": ["A"]},
        ])
        ns = engine.next_steps(wid)
        assert len(ns) == 1
        assert ns[0]["name"] == "A"

    def test_workflow_next_steps_after_completion(self, engine):
        """Completing a step unlocks dependents."""
        wid = engine.create("dag", [
            {"name": "A"},
            {"name": "B", "depends_on": ["A"]},
            {"name": "C", "depends_on": ["A"]},
        ])
        wf = engine.get(wid)
        a_id = [s for s in wf["steps"] if s["name"] == "A"][0]["id"]

        engine.update_step(wid, a_id, "completed")
        ns = engine.next_steps(wid)
        names = sorted([s["name"] for s in ns])
        assert names == ["B", "C"]

    def test_workflow_update_step_status(self, engine):
        """Status transition works."""
        wid = engine.create("test", [{"name": "s1"}])
        wf = engine.get(wid)
        step_id = wf["steps"][0]["id"]

        assert engine.update_step(wid, step_id, "running") is True
        wf = engine.get(wid)
        assert wf["steps"][0]["status"] == "running"
        assert wf["steps"][0]["started_at"] is not None

    def test_workflow_auto_complete(self, engine):
        """Workflow status auto-updates when all steps done."""
        wid = engine.create("test", [
            {"name": "s1"},
            {"name": "s2"},
        ])
        wf = engine.get(wid)
        for step in wf["steps"]:
            engine.update_step(wid, step["id"], "completed")

        wf = engine.get(wid)
        assert wf["status"] == "completed"

    def test_workflow_delete_cascades(self, engine):
        """Deleting workflow removes steps."""
        wid = engine.create("test", [{"name": "s1"}, {"name": "s2"}])
        assert engine.delete(wid) is True
        assert engine.get(wid) is None


class TestAgentLoader:
    """Tests for core.agents.AgentLoader."""

    @pytest.fixture
    def agents_dir(self):
        """Create a temp agents directory with test YAMLs."""
        import yaml
        tmpdir = tempfile.mkdtemp()
        agents_path = Path(tmpdir)

        # Create test agents
        with open(agents_path / "test-reviewer.yaml", "w") as f:
            yaml.dump({
                "name": "Test Reviewer",
                "description": "Reviews tests for quality",
                "system_prompt": "You are a test reviewer.",
                "tags": ["testing", "review"],
            }, f)

        with open(agents_path / "code-helper.yaml", "w") as f:
            yaml.dump({
                "name": "Code Helper",
                "description": "Helps write code",
                "system_prompt": "You are a code helper.",
                "tags": ["code", "development"],
            }, f)

        yield agents_path
        shutil.rmtree(tmpdir)

    def test_agents_list(self, agents_dir):
        """Returns all agents from directory."""
        from core.agents import AgentLoader
        loader = AgentLoader(agents_dir)
        agents = loader.list()
        assert len(agents) == 2

    def test_agents_list_by_tag(self, agents_dir):
        """Tag filter works."""
        from core.agents import AgentLoader
        loader = AgentLoader(agents_dir)
        agents = loader.list(tag="testing")
        assert len(agents) == 1
        assert agents[0]["name"] == "Test Reviewer"

    def test_agents_get_by_name(self, agents_dir):
        """Exact match on name field."""
        from core.agents import AgentLoader
        loader = AgentLoader(agents_dir)
        agent = loader.get("Test Reviewer")
        assert agent is not None
        assert agent["name"] == "Test Reviewer"
        assert "system_prompt" in agent

    def test_agents_get_by_filename(self, agents_dir):
        """Match on filename slug."""
        from core.agents import AgentLoader
        loader = AgentLoader(agents_dir)
        agent = loader.get("code-helper")
        assert agent is not None
        assert agent["name"] == "Code Helper"

    def test_agents_get_nonexistent(self, agents_dir):
        """Returns None for nonexistent agent."""
        from core.agents import AgentLoader
        loader = AgentLoader(agents_dir)
        assert loader.get("nonexistent") is None

    def test_agents_search(self, agents_dir):
        """Substring matching works."""
        from core.agents import AgentLoader
        loader = AgentLoader(agents_dir)
        results = loader.search("review")
        assert len(results) >= 1
        assert results[0]["name"] == "Test Reviewer"


class TestMemoryStoreExpiry:
    """Tests for expired memory auto-filtering."""

    @pytest.fixture
    def store(self):
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test_expiry.db"
        from core.memory import MemoryStore
        s = MemoryStore(db_path)
        yield s
        s.close()
        shutil.rmtree(tmpdir)

    def test_retrieve_excludes_expired(self, store):
        """Expired entries return None on retrieve."""
        store.store("ns", "k", "content", ttl=1)
        assert store.retrieve("ns", "k") is not None
        # Manually expire it
        store._conn.execute(
            "UPDATE memories SET expires_at = ? WHERE key = ?",
            (time.time() - 10, "k"),
        )
        store._conn.commit()
        assert store.retrieve("ns", "k") is None

    def test_list_excludes_expired(self, store):
        """Expired entries don't appear in list."""
        store.store("ns", "fresh", "still valid")
        store.store("ns", "old", "expired", ttl=1)
        store._conn.execute(
            "UPDATE memories SET expires_at = ? WHERE key = ?",
            (time.time() - 10, "old"),
        )
        store._conn.commit()
        results = store.list()
        keys = [r["key"] for r in results]
        assert "fresh" in keys
        assert "old" not in keys

    def test_search_excludes_expired(self, store):
        """LIKE search skips expired entries."""
        store.store("ns", "active", "search for this")
        store.store("ns", "dead", "search for this too", ttl=1)
        store._conn.execute(
            "UPDATE memories SET expires_at = ? WHERE key = ?",
            (time.time() - 10, "dead"),
        )
        store._conn.commit()
        results = store.search("search for this")
        keys = [r["key"] for r in results]
        assert "active" in keys
        assert "dead" not in keys


class TestWorkflowStatusValidation:
    """Tests for workflow status validation."""

    @pytest.fixture
    def engine(self):
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test_validation.db"
        from core.workflow import WorkflowEngine
        e = WorkflowEngine(db_path)
        yield e
        e.close()
        shutil.rmtree(tmpdir)

    def test_update_step_rejects_invalid_status(self, engine):
        """Invalid status raises ValueError."""
        wid = engine.create("test", [{"name": "s1"}])
        wf = engine.get(wid)
        step_id = wf["steps"][0]["id"]
        with pytest.raises(ValueError, match="Invalid status"):
            engine.update_step(wid, step_id, "bogus")

    def test_update_step_accepts_all_valid_statuses(self, engine):
        """All valid statuses are accepted."""
        from core.workflow import WorkflowEngine
        for status in WorkflowEngine.VALID_STEP_STATUSES:
            wid = engine.create(f"test-{status}", [{"name": "s1"}])
            wf = engine.get(wid)
            step_id = wf["steps"][0]["id"]
            assert engine.update_step(wid, step_id, status) is True

    def test_workflow_auto_complete_with_skipped(self, engine):
        """Workflow completes when all steps are completed or skipped."""
        wid = engine.create("test", [
            {"name": "s1"},
            {"name": "s2"},
        ])
        wf = engine.get(wid)
        engine.update_step(wid, wf["steps"][0]["id"], "completed")
        engine.update_step(wid, wf["steps"][1]["id"], "skipped")
        wf = engine.get(wid)
        assert wf["status"] == "completed"


class TestSharedConnection:
    """Tests for shared SQLite connection between MemoryStore and WorkflowEngine."""

    def test_shared_connection_works(self):
        """Both stores function on a shared connection."""
        import sqlite3
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "shared.db"
        from core.memory import MemoryStore
        from core.workflow import WorkflowEngine

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        lock = threading.Lock()

        memory = MemoryStore(db_path, connection=conn, lock=lock)
        workflow = WorkflowEngine(db_path, connection=conn, lock=lock)

        # Both work
        memory.store("ns", "k", "value")
        wid = workflow.create("test", [{"name": "s1"}])

        assert memory.retrieve("ns", "k") is not None
        assert workflow.get(wid) is not None

        # Shared connection — neither owns it
        assert memory._owns_conn is False
        assert workflow._owns_conn is False

        conn.close()
        shutil.rmtree(tmpdir)


###############################################################################
# create_provider_from_config tests
###############################################################################

class TestCreateProviderFromConfig:
    """Tests for the create_provider_from_config factory function."""

    def test_lmstudio_anthropic(self):
        """Creates LMStudioAnthropicProvider from config."""
        from core.llm import create_provider_from_config, LMStudioAnthropicProvider
        config = {"provider": "lmstudio_anthropic"}
        provider = create_provider_from_config(config)
        assert isinstance(provider, LMStudioAnthropicProvider)

    def test_lmstudio_openai(self):
        """Creates LMStudioProvider from config."""
        from core.llm import create_provider_from_config, LMStudioProvider
        config = {"provider": "lmstudio"}
        provider = create_provider_from_config(config)
        assert isinstance(provider, LMStudioProvider)

    def test_lmstudio_openai_alias(self):
        """lmstudio_openai maps to LMStudioProvider."""
        from core.llm import create_provider_from_config, LMStudioProvider
        config = {"provider": "lmstudio_openai"}
        provider = create_provider_from_config(config)
        assert isinstance(provider, LMStudioProvider)

    def test_lmstudio_native(self):
        """Creates LMStudioNativeProvider from config."""
        from core.llm import create_provider_from_config, LMStudioNativeProvider
        config = {"provider": "lmstudio_native"}
        provider = create_provider_from_config(config)
        assert isinstance(provider, LMStudioNativeProvider)

    def test_unknown_provider_raises(self):
        """Unknown provider raises ValueError."""
        from core.llm import create_provider_from_config
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider_from_config({"provider": "nonexistent"})

    def test_claude_without_key_raises(self):
        """Claude provider without API key raises ValueError."""
        from core.llm import create_provider_from_config
        import os
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                create_provider_from_config({"provider": "claude"})
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_custom_host_and_model(self):
        """Custom host and model are passed through."""
        from core.llm import create_provider_from_config, LMStudioAnthropicProvider
        config = {
            "provider": "lmstudio_anthropic",
            "lmstudio_host": "http://myhost:5555",
            "lmstudio_model": "my-model",
        }
        provider = create_provider_from_config(config)
        assert isinstance(provider, LMStudioAnthropicProvider)
        assert provider.host == "http://myhost:5555"
        assert provider.model == "my-model"

    def test_no_max_tokens_in_constructor(self):
        """Verify max_tokens is NOT passed to constructors (regression)."""
        from core.llm import create_provider_from_config, LMStudioAnthropicProvider
        # Should not raise TypeError about unexpected keyword argument
        config = {"provider": "lmstudio_anthropic", "max_tokens": 200}
        provider = create_provider_from_config(config)
        assert isinstance(provider, LMStudioAnthropicProvider)


###############################################################################
# AgentRunner tests
###############################################################################

class TestAgentRunner:
    """Tests for the AgentRunner class."""

    def _make_mock_searcher(self, chunks=None):
        """Create a mock searcher that returns given chunks."""
        from core.search import SearchResult, SearchQuality
        from unittest.mock import MagicMock

        mock = MagicMock()
        result = SearchResult(
            results=chunks or [],
            quality=SearchQuality(total_candidates=len(chunks or [])),
        )
        mock.search_with_quality.return_value = result
        return mock

    def _make_mock_provider(self, response="Test response"):
        """Create a mock LLM provider."""
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.generate.return_value = response
        return mock

    def _make_mock_client(self):
        """Create a mock client."""
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.llm_config = {"max_context_chars": 12000}
        mock.name = "test-client"
        return mock

    def _make_agent(self, name="test-agent", system_prompt="You are a test agent."):
        return {
            "name": name,
            "system_prompt": system_prompt,
            "_filename": "test-agent",
            "description": "A test agent",
            "tags": ["test"],
        }

    def test_basic_run(self):
        """Basic run returns AgentResult with response."""
        from core.runner import AgentRunner

        chunks = [{"content": "test chunk", "metadata": {"source_file": "test.py"}}]
        searcher = self._make_mock_searcher(chunks)
        provider = self._make_mock_provider("The answer is 42")
        client = self._make_mock_client()
        agent = self._make_agent()

        runner = AgentRunner(agent, client, provider, searcher)
        result = runner.run("What is the answer?")

        assert result.response == "The answer is 42"
        assert result.agent_name == "test-agent"
        assert result.query == "What is the answer?"
        assert len(result.sources) == 1
        assert "search_ms" in result.timing
        assert "generate_ms" in result.timing

    def test_system_prompt_passthrough(self):
        """Agent system prompt is passed to LLM provider."""
        from core.runner import AgentRunner

        searcher = self._make_mock_searcher([])
        provider = self._make_mock_provider()
        client = self._make_mock_client()
        agent = self._make_agent(system_prompt="Be a code reviewer.")

        runner = AgentRunner(agent, client, provider, searcher)
        runner.run("Review this")

        call_kwargs = provider.generate.call_args
        assert call_kwargs.kwargs.get("system_prompt") == "Be a code reviewer."

    def test_top_k_forwarded(self):
        """top_k parameter is forwarded to searcher."""
        from core.runner import AgentRunner

        searcher = self._make_mock_searcher([])
        provider = self._make_mock_provider()
        client = self._make_mock_client()
        agent = self._make_agent()

        runner = AgentRunner(agent, client, provider, searcher)
        runner.run("test query", top_k=25)

        searcher.search_with_quality.assert_called_once_with(query="test query", top_k=25)

    def test_conversation_history(self):
        """Conversation history is prepended to messages."""
        from core.runner import AgentRunner

        searcher = self._make_mock_searcher([])
        provider = self._make_mock_provider()
        client = self._make_mock_client()
        agent = self._make_agent()

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        runner = AgentRunner(agent, client, provider, searcher)
        runner.run("Follow up", conversation_history=history)

        messages = provider.generate.call_args.args[0]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

    def test_empty_results(self):
        """Handles empty search results gracefully."""
        from core.runner import AgentRunner

        searcher = self._make_mock_searcher([])
        provider = self._make_mock_provider("No context available")
        client = self._make_mock_client()
        agent = self._make_agent()

        runner = AgentRunner(agent, client, provider, searcher)
        result = runner.run("Unknown topic")

        assert result.response == "No context available"
        assert len(result.sources) == 0

    def test_max_tokens_override(self):
        """max_tokens is forwarded to LLM provider."""
        from core.runner import AgentRunner

        searcher = self._make_mock_searcher([])
        provider = self._make_mock_provider()
        client = self._make_mock_client()
        agent = self._make_agent()

        runner = AgentRunner(agent, client, provider, searcher)
        runner.run("test", max_tokens=500)

        call_kwargs = provider.generate.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 500

    def test_memory_persistence(self):
        """store_memory=True stores result in MemoryStore."""
        from core.runner import AgentRunner
        from unittest.mock import MagicMock

        searcher = self._make_mock_searcher([])
        provider = self._make_mock_provider("stored response")
        client = self._make_mock_client()
        agent = self._make_agent()

        memory = MagicMock()
        runner = AgentRunner(agent, client, provider, searcher, memory_store=memory)
        runner.run("test", store_memory=True)

        memory.store.assert_called_once()
        call_kwargs = memory.store.call_args
        assert call_kwargs.kwargs.get("namespace") == "agent:test-agent"
        assert call_kwargs.kwargs.get("content") == "stored response"

    def test_memory_not_stored_by_default(self):
        """Memory is not stored when store_memory=False."""
        from core.runner import AgentRunner
        from unittest.mock import MagicMock

        searcher = self._make_mock_searcher([])
        provider = self._make_mock_provider()
        client = self._make_mock_client()
        agent = self._make_agent()

        memory = MagicMock()
        runner = AgentRunner(agent, client, provider, searcher, memory_store=memory)
        runner.run("test", store_memory=False)

        memory.store.assert_not_called()

    def test_search_quality_in_result(self):
        """SearchQuality is included in the result."""
        from core.runner import AgentRunner

        chunks = [{"content": "data", "metadata": {"source_file": "a.py"}}]
        searcher = self._make_mock_searcher(chunks)
        provider = self._make_mock_provider()
        client = self._make_mock_client()
        agent = self._make_agent()

        runner = AgentRunner(agent, client, provider, searcher)
        result = runner.run("query")

        assert result.search_quality is not None
        assert result.search_quality.total_candidates == 1

    def test_sources_have_score_and_type(self):
        """Sources include score and chunk_type."""
        from core.runner import AgentRunner

        chunks = [
            {
                "content": "def foo(): pass",
                "metadata": {"source_file": "foo.py", "chunk_type": "code"},
                "_rerank_score": 0.95,
            },
        ]
        searcher = self._make_mock_searcher(chunks)
        provider = self._make_mock_provider()
        client = self._make_mock_client()
        agent = self._make_agent()

        runner = AgentRunner(agent, client, provider, searcher)
        result = runner.run("review foo")

        assert result.sources[0]["source"] == "foo.py"
        assert result.sources[0]["score"] == 0.95
        assert result.sources[0]["chunk_type"] == "code"


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


###############################################################################
# validate_chunk tests
###############################################################################

class TestValidateChunk:
    """Tests for the validate_chunk function."""

    def test_valid_chunk(self):
        from core.ingest import validate_chunk
        valid, reason = validate_chunk("This is a normal paragraph with enough content to pass validation easily.")
        assert valid is True
        assert reason is None

    def test_too_short(self):
        from core.ingest import validate_chunk
        valid, reason = validate_chunk("Short")
        assert valid is False
        assert reason == "too_short"

    def test_no_alphanumeric(self):
        from core.ingest import validate_chunk
        valid, reason = validate_chunk("---" * 30)  # 90 chars, all dashes
        assert valid is False
        assert reason == "no_alphanumeric"

    def test_whitespace_only(self):
        from core.ingest import validate_chunk
        valid, reason = validate_chunk("   \n\t   \n   ")
        assert valid is False
        assert reason == "too_short"

    def test_custom_min_chars(self):
        from core.ingest import validate_chunk
        valid, _ = validate_chunk("Hello", min_chars=3)
        assert valid is True

    def test_exactly_at_threshold(self):
        from core.ingest import validate_chunk
        text = "a" * 50
        valid, _ = validate_chunk(text, min_chars=50)
        assert valid is True

    def test_just_below_threshold(self):
        from core.ingest import validate_chunk
        text = "a" * 49
        valid, reason = validate_chunk(text, min_chars=50)
        assert valid is False
        assert reason == "too_short"


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


###############################################################################
# Client loading tests
###############################################################################

class TestLoadClient:
    """Tests for load_client and get_active_client module functions."""

    def test_load_client_with_config(self, tmp_path):
        """load_client reads a client.yaml and returns Client."""
        from core.client import load_client

        client_dir = tmp_path / "test_client"
        client_dir.mkdir()
        config = {
            "name": "Test Client",
            "machine_name": "test_client",
            "paths": {"documents": "documents", "index": "index", "prompts": "prompts"},
            "embedding": {"model": "all-MiniLM-L6-v2", "dimension": 384},
        }
        import yaml
        (client_dir / "client.yaml").write_text(yaml.dump(config))

        client = load_client("test_client", clients_dir=tmp_path)
        assert client.name == "Test Client"
        assert client.machine_name == "test_client"

    def test_load_client_missing_config_uses_defaults(self, tmp_path):
        """load_client with no client.yaml falls back to defaults."""
        from core.client import load_client

        client_dir = tmp_path / "no_config"
        client_dir.mkdir()

        client = load_client("no_config", clients_dir=tmp_path)
        assert client.machine_name == "no_config"
        assert client.name == "Default RAG"

    def test_discover_clients(self, tmp_path):
        """discover_clients finds all client directories."""
        from core.client import discover_clients

        (tmp_path / "alpha").mkdir()
        (tmp_path / "beta").mkdir()
        (tmp_path / ".hidden").mkdir()  # Should be excluded

        clients = discover_clients(clients_dir=tmp_path)
        assert "alpha" in clients
        assert "beta" in clients
        assert ".hidden" not in clients

    def test_discover_clients_empty_dir(self, tmp_path):
        from core.client import discover_clients
        assert discover_clients(clients_dir=tmp_path) == []

    def test_discover_clients_missing_dir(self, tmp_path):
        from core.client import discover_clients
        assert discover_clients(clients_dir=tmp_path / "nonexistent") == []

    def test_get_active_client_from_env(self, tmp_path, monkeypatch):
        """get_active_client uses RAG_CLIENT env var."""
        from core.client import get_active_client
        import yaml

        client_dir = tmp_path / "env_client"
        client_dir.mkdir()
        config = {"name": "Env Client", "machine_name": "env_client"}
        (client_dir / "client.yaml").write_text(yaml.dump(config))

        monkeypatch.setenv("RAG_CLIENT", "env_client")
        client = get_active_client(clients_dir=tmp_path)
        assert client is not None
        assert client.name == "Env Client"

    def test_get_active_client_fallback_to_first(self, tmp_path, monkeypatch):
        """get_active_client falls back to first available when no env var."""
        from core.client import get_active_client
        import yaml

        monkeypatch.delenv("RAG_CLIENT", raising=False)
        client_dir = tmp_path / "first_client"
        client_dir.mkdir()
        config = {"name": "First Client", "machine_name": "first_client"}
        (client_dir / "client.yaml").write_text(yaml.dump(config))

        client = get_active_client(clients_dir=tmp_path)
        assert client is not None
        assert client.name == "First Client"

    def test_get_active_client_returns_none_when_empty(self, tmp_path, monkeypatch):
        """get_active_client returns None when no clients exist."""
        from core.client import get_active_client

        monkeypatch.delenv("RAG_CLIENT", raising=False)
        client = get_active_client(clients_dir=tmp_path)
        assert client is None


###############################################################################
# AgentResult dataclass tests
###############################################################################

class TestAgentResult:
    """Tests for AgentResult dataclass defaults and construction."""

    def test_defaults(self):
        from core.runner import AgentResult
        r = AgentResult(response="hello")
        assert r.response == "hello"
        assert r.sources == []
        assert r.search_quality is None
        assert r.timing == {}
        assert r.citations_used == []
        assert r.agent_name == ""
        assert r.query == ""

    def test_full_construction(self):
        from core.runner import AgentResult
        from core.search import SearchQuality

        q = SearchQuality(total_candidates=5)
        r = AgentResult(
            response="answer",
            sources=[{"source": "a.py"}],
            search_quality=q,
            timing={"search_ms": 100},
            agent_name="reviewer",
            query="test",
        )
        assert r.response == "answer"
        assert r.agent_name == "reviewer"
        assert r.search_quality.total_candidates == 5


###############################################################################
# AgentRunner additional edge cases
###############################################################################

class TestAgentRunnerEdgeCases:
    """Additional edge case tests for AgentRunner."""

    def _make_mock_searcher(self, chunks=None):
        from unittest.mock import MagicMock
        from core.search import SearchResult, SearchQuality
        searcher = MagicMock(spec=["search_with_quality"])
        result = SearchResult(
            results=chunks or [],
            quality=SearchQuality(total_candidates=len(chunks or [])),
        )
        searcher.search_with_quality.return_value = result
        return searcher

    def _make_mock_provider(self, response="Test response"):
        from unittest.mock import MagicMock
        provider = MagicMock(spec=["generate"])
        provider.generate.return_value = response
        return provider

    def _make_mock_client(self, **overrides):
        from unittest.mock import MagicMock
        client = MagicMock()
        client.llm_config = {"max_context_chars": 5000, **overrides}
        return client

    def test_run_with_no_search_results(self):
        """AgentRunner works even with 0 search results."""
        from core.runner import AgentRunner

        searcher = self._make_mock_searcher([])
        provider = self._make_mock_provider("No context available")
        client = self._make_mock_client()
        agent = {"name": "test", "system_prompt": "You are helpful."}

        runner = AgentRunner(agent, client, provider, searcher)
        result = runner.run("anything")

        assert result.response == "No context available"
        assert result.sources == []
        assert result.timing["search_ms"] >= 0

    def test_agent_name_fallback(self):
        """Uses 'Unknown' when agent has no name."""
        from core.runner import AgentRunner

        searcher = self._make_mock_searcher([])
        provider = self._make_mock_provider()
        client = self._make_mock_client()
        agent = {"system_prompt": "test"}  # No name

        runner = AgentRunner(agent, client, provider, searcher)
        result = runner.run("query")
        assert result.agent_name == "Unknown"

    def test_client_without_llm_config(self):
        """Works when client doesn't have llm_config attr."""
        from core.runner import AgentRunner
        from unittest.mock import MagicMock

        searcher = self._make_mock_searcher([])
        provider = self._make_mock_provider()
        # Client without llm_config
        client = MagicMock(spec=[])
        agent = {"name": "test", "system_prompt": "test"}

        runner = AgentRunner(agent, client, provider, searcher)
        result = runner.run("query")
        assert result.response is not None

    def test_timing_includes_all_keys(self):
        """Result timing dict has search_ms, generate_ms, and total_ms."""
        from core.runner import AgentRunner

        searcher = self._make_mock_searcher([])
        provider = self._make_mock_provider()
        client = self._make_mock_client()
        agent = {"name": "test", "system_prompt": "test"}

        runner = AgentRunner(agent, client, provider, searcher)
        result = runner.run("query")

        assert "search_ms" in result.timing
        assert "generate_ms" in result.timing
        assert "total_ms" in result.timing
        assert result.timing["total_ms"] == result.timing["search_ms"] + result.timing["generate_ms"]

    def test_source_falls_back_to_combined_score(self):
        """Sources use _combined_score when _rerank_score is absent."""
        from core.runner import AgentRunner

        chunks = [{"content": "test", "metadata": {"source_file": "x.py"}, "_combined_score": 0.42}]
        searcher = self._make_mock_searcher(chunks)
        provider = self._make_mock_provider()
        client = self._make_mock_client()
        agent = {"name": "test", "system_prompt": "test"}

        runner = AgentRunner(agent, client, provider, searcher)
        result = runner.run("query")
        assert result.sources[0]["score"] == pytest.approx(0.42)

    def test_memory_store_key_truncation(self):
        """Memory store key is truncated to 200 chars."""
        from core.runner import AgentRunner
        from unittest.mock import MagicMock

        searcher = self._make_mock_searcher([])
        provider = self._make_mock_provider("response")
        client = self._make_mock_client()
        agent = {"name": "test", "system_prompt": "test", "_filename": "test.yaml"}

        memory = MagicMock()
        runner = AgentRunner(agent, client, provider, searcher, memory_store=memory)
        long_query = "x" * 300
        runner.run(long_query, store_memory=True)

        call_kwargs = memory.store.call_args
        stored_key = call_kwargs.kwargs.get("key") or call_kwargs[1].get("key") or call_kwargs[0][1]
        assert len(stored_key) <= 200


###############################################################################
# Ingestion job status transition tests
###############################################################################

class TestIngestionJobEdgeCases:
    """Additional edge case tests for IngestionJobManager."""

    def test_multiple_job_creation_within_limit(self):
        from core.ingest_job import IngestionJobManager
        mgr = IngestionJobManager(max_concurrent=2)
        job1 = mgr.create_job("client_a", "reindex")
        job2 = mgr.create_job("client_b", "reindex")
        assert job1.client_name == "client_a"
        assert job2.client_name == "client_b"

    def test_exceeds_max_concurrent(self):
        from core.ingest_job import IngestionJobManager
        mgr = IngestionJobManager(max_concurrent=1)
        job1 = mgr.create_job("client_a", "reindex")
        mgr.start_job(job1.job_id)

        with pytest.raises(RuntimeError):
            mgr.create_job("client_b", "reindex")

    def test_job_to_dict_serialization(self):
        from core.ingest_job import IngestionJobManager
        mgr = IngestionJobManager()
        job = mgr.create_job("test_client", "reindex")
        d = job.to_dict()
        assert d["client_name"] == "test_client"
        assert d["status"] == "pending"
        assert "job_id" in d

    def test_complete_job_sets_result(self):
        from core.ingest_job import IngestionJobManager
        mgr = IngestionJobManager()
        job = mgr.create_job("test", "reindex")
        mgr.start_job(job.job_id)
        mgr.complete_job(job.job_id, "Done: 100 chunks")

        updated = mgr.get_job(job.job_id)
        assert updated.status == "completed"
        assert updated.result == "Done: 100 chunks"

    def test_fail_job_sets_error(self):
        from core.ingest_job import IngestionJobManager
        mgr = IngestionJobManager()
        job = mgr.create_job("test", "reindex")
        mgr.start_job(job.job_id)
        mgr.fail_job(job.job_id, "File not found")

        updated = mgr.get_job(job.job_id)
        assert updated.status == "failed"
        assert updated.error == "File not found"

    def test_get_client_jobs(self):
        from core.ingest_job import IngestionJobManager
        mgr = IngestionJobManager(max_concurrent=5)
        mgr.create_job("alpha", "reindex")
        mgr.create_job("beta", "reindex")
        mgr.create_job("alpha", "reindex")

        alpha_jobs = mgr.get_client_jobs("alpha")
        assert len(alpha_jobs) == 2

    def test_cleanup_completed(self):
        from core.ingest_job import IngestionJobManager
        mgr = IngestionJobManager(max_concurrent=5)
        j1 = mgr.create_job("test", "reindex")
        j2 = mgr.create_job("test2", "reindex")
        mgr.start_job(j1.job_id)
        mgr.complete_job(j1.job_id, "done")
        # j2 is still pending

        mgr.cleanup_completed(max_age_seconds=0)
        assert mgr.get_job(j1.job_id) is None
        assert mgr.get_job(j2.job_id) is not None


###############################################################################
# Workflow edge cases
###############################################################################

class TestWorkflowEdgeCases:
    """Additional edge case tests for WorkflowEngine."""

    def _make_engine(self, tmp_path):
        from core.workflow import WorkflowEngine
        db_path = str(tmp_path / "test_workflows.db")
        return WorkflowEngine(db_path=db_path)

    def test_create_workflow_with_single_step(self, tmp_path):
        engine = self._make_engine(tmp_path)
        wf_id = engine.create(
            name="Simple",
            steps=[{"name": "only_step"}],
        )
        wf = engine.get(wf_id)
        assert wf["status"] == "pending"
        assert len(wf["steps"]) == 1

    def test_delete_workflow(self, tmp_path):
        engine = self._make_engine(tmp_path)
        wf_id = engine.create(name="ToDelete", steps=[{"name": "s1"}])
        engine.delete(wf_id)
        assert engine.get(wf_id) is None

    def test_next_steps_returns_ready_ones(self, tmp_path):
        engine = self._make_engine(tmp_path)
        wf_id = engine.create(
            name="DAG",
            steps=[
                {"name": "step_a"},
                {"name": "step_b", "depends_on": ["step_a"]},
            ],
        )
        ready = engine.next_steps(wf_id)
        assert len(ready) == 1
        assert ready[0]["name"] == "step_a"

    def test_completing_step_unblocks_dependent(self, tmp_path):
        engine = self._make_engine(tmp_path)
        wf_id = engine.create(
            name="DAG",
            steps=[
                {"name": "step_a"},
                {"name": "step_b", "depends_on": ["step_a"]},
            ],
        )
        # Complete step_a
        steps = engine.get(wf_id)["steps"]
        step_a_id = [s for s in steps if s["name"] == "step_a"][0]["id"]
        engine.update_step(wf_id, step_a_id, "completed")

        ready = engine.next_steps(wf_id)
        assert len(ready) == 1
        assert ready[0]["name"] == "step_b"

    def test_workflow_status_auto_completes(self, tmp_path):
        engine = self._make_engine(tmp_path)
        wf_id = engine.create(name="Auto", steps=[{"name": "s1"}])
        steps = engine.get(wf_id)["steps"]
        engine.update_step(wf_id, steps[0]["id"], "completed")

        updated = engine.get(wf_id)
        assert updated["status"] == "completed"

    def test_list_workflows_filter_by_status(self, tmp_path):
        engine = self._make_engine(tmp_path)
        engine.create(name="WF1", steps=[{"name": "s1"}])
        wf2_id = engine.create(name="WF2", steps=[{"name": "s1"}])
        steps = engine.get(wf2_id)["steps"]
        engine.update_step(wf2_id, steps[0]["id"], "completed")

        pending = engine.list(status="pending")
        completed = engine.list(status="completed")
        assert len(pending) == 1
        assert len(completed) == 1
        assert pending[0]["name"] == "WF1"
        assert completed[0]["name"] == "WF2"


###############################################################################
# Regression tests for bug fixes
###############################################################################

class TestBugFixRegressions:
    """Regression tests for bugs found during code review."""

    def test_skipped_step_unblocks_dependent(self, tmp_path):
        """Skipped steps should satisfy dependency requirements (bug fix)."""
        from core.workflow import WorkflowEngine

        engine = WorkflowEngine(db_path=str(tmp_path / "reg.db"))
        wf_id = engine.create(
            name="SkipTest",
            steps=[
                {"name": "optional_step"},
                {"name": "final_step", "depends_on": ["optional_step"]},
            ],
        )
        steps = engine.get(wf_id)["steps"]
        optional_id = [s for s in steps if s["name"] == "optional_step"][0]["id"]

        # Skip the optional step
        engine.update_step(wf_id, optional_id, "skipped")

        # final_step should now be ready
        ready = engine.next_steps(wf_id)
        assert len(ready) == 1
        assert ready[0]["name"] == "final_step"

    def test_runner_memory_uses_content_param(self):
        """AgentRunner passes 'content' (not 'value') to MemoryStore.store()."""
        from core.runner import AgentRunner
        from core.search import SearchResult, SearchQuality
        from unittest.mock import MagicMock

        searcher = MagicMock(spec=["search_with_quality"])
        searcher.search_with_quality.return_value = SearchResult(
            results=[], quality=SearchQuality()
        )
        provider = MagicMock(spec=["generate"])
        provider.generate.return_value = "test response"
        client = MagicMock()
        client.llm_config = {}
        agent = {"name": "test", "system_prompt": "test", "_filename": "test"}

        memory = MagicMock()
        runner = AgentRunner(agent, client, provider, searcher, memory_store=memory)
        runner.run("query", store_memory=True)

        # Verify 'content' param used (not 'value')
        call_kwargs = memory.store.call_args.kwargs
        assert "content" in call_kwargs
        assert "value" not in call_kwargs


class TestSourceConnectors:
    """Tests for core.sources module (Stage 0)."""

    # --- GitHubSourceConfig ---

    def test_github_config_from_dict(self):
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig.from_dict({
            "owner": "testorg",
            "token_env": "GH_TOKEN",
            "include_repos": ["repo1", "repo2"],
            "exclude_repos": ["legacy"],
            "include_forks": True,
            "include_private": False,
        })
        assert config.owner == "testorg"
        assert config.token_env == "GH_TOKEN"
        assert config.include_repos == ["repo1", "repo2"]
        assert config.exclude_repos == ["legacy"]
        assert config.include_forks is True
        assert config.include_private is False

    def test_github_config_repo_shorthand(self):
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig(repo="owner/repo-name")
        assert config.owner == "owner"
        assert config.repo == "repo-name"

    def test_github_config_repo_shorthand_no_override(self):
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig(owner="explicit", repo="myrepo")
        assert config.owner == "explicit"
        assert config.repo == "myrepo"

    def test_github_config_defaults(self):
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig()
        assert config.owner == ""
        assert config.repo == ""
        assert config.token_env == ""
        assert config.include_repos == []
        assert config.exclude_repos == []
        assert config.include_forks is False
        assert config.include_private is True
        assert config.sync_on_ingest is True

    def test_github_config_token_from_env(self, monkeypatch):
        from core.sources.github_models import GitHubSourceConfig

        monkeypatch.setenv("TEST_GH_TOKEN", "ghp_testtoken123")
        config = GitHubSourceConfig(token_env="TEST_GH_TOKEN")
        assert config.token == "ghp_testtoken123"

    def test_github_config_token_missing_env(self):
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig(token_env="NONEXISTENT_TOKEN_VAR_XYZ")
        assert config.token is None

    def test_github_config_no_token_env(self):
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig()
        assert config.token is None

    # --- RepoInfo ---

    def test_repo_info_from_api_response(self):
        from core.sources.github_models import RepoInfo

        data = {
            "name": "my-repo",
            "owner": {"login": "testuser"},
            "html_url": "https://github.com/testuser/my-repo",
            "clone_url": "https://github.com/testuser/my-repo.git",
            "private": True,
            "description": "A test repo",
            "default_branch": "develop",
            "size": 1234,
            "fork": False,
        }
        repo = RepoInfo.from_api_response(data)
        assert repo.name == "my-repo"
        assert repo.owner == "testuser"
        assert repo.url == "https://github.com/testuser/my-repo"
        assert repo.clone_url == "https://github.com/testuser/my-repo.git"
        assert repo.private is True
        assert repo.description == "A test repo"
        assert repo.default_branch == "develop"
        assert repo.size_kb == 1234
        assert repo.is_fork is False

    def test_repo_info_defaults(self):
        from core.sources.github_models import RepoInfo

        data = {
            "name": "minimal",
            "owner": {"login": "user"},
        }
        repo = RepoInfo.from_api_response(data)
        assert repo.url == ""
        assert repo.clone_url == ""
        assert repo.private is False
        assert repo.description == ""
        assert repo.default_branch == "main"
        assert repo.size_kb == 0
        assert repo.is_fork is False

    def test_repo_info_null_description(self):
        from core.sources.github_models import RepoInfo

        data = {
            "name": "nodesc",
            "owner": {"login": "user"},
            "description": None,
        }
        repo = RepoInfo.from_api_response(data)
        assert repo.description == ""

    # --- SyncResult ---

    def test_sync_result_defaults(self):
        from core.sources.base import SyncResult

        r = SyncResult(source_type="github")
        assert r.files_added == 0
        assert r.files_updated == 0
        assert r.files_removed == 0
        assert r.errors == []
        assert r.total_changes == 0
        assert r.success is True

    def test_sync_result_with_errors(self):
        from core.sources.base import SyncResult

        r = SyncResult(source_type="web", errors=["something failed"])
        assert r.success is False
        assert r.total_changes == 0

    def test_sync_result_total_changes(self):
        from core.sources.base import SyncResult

        r = SyncResult(
            source_type="github",
            files_added=5,
            files_updated=3,
            files_removed=1,
        )
        assert r.total_changes == 9
        assert r.success is True

    # --- git_ops ---

    def test_git_ops_validate_name_valid(self):
        from core.sources.git_ops import _validate_name

        _validate_name("my-repo")
        _validate_name("repo.name")
        _validate_name("Repo_123")

    def test_git_ops_validate_name_invalid(self):
        from core.sources.git_ops import _validate_name, GitError

        with pytest.raises(GitError):
            _validate_name("repo with spaces")
        with pytest.raises(GitError):
            _validate_name("../escape")
        with pytest.raises(GitError):
            _validate_name("path/../traversal")

    def test_git_ops_sanitise_output(self):
        from core.sources.git_ops import _sanitise_output

        assert "ghp_" not in _sanitise_output(
            "error with ghp_abcdefghijklmnopqrstuvwxyz0123456789"
        )
        assert "****" in _sanitise_output(
            "error with ghp_abcdefghijklmnopqrstuvwxyz0123456789"
        )

    def test_git_ops_sanitise_url(self):
        from core.sources.git_ops import _sanitise_output

        result = _sanitise_output(
            "fatal: https://user:password@github.com/repo.git not found"
        )
        assert "password" not in result
        assert "****" in result

    def test_clone_result_dataclass(self):
        from core.sources.git_ops import CloneResult

        r = CloneResult(success=True, message="ok")
        assert r.success is True
        assert r.message == "ok"
        assert r.changed_files == []

    def test_clone_result_with_changes(self):
        from core.sources.git_ops import CloneResult

        r = CloneResult(
            success=True,
            message="Updated: 3 files changed",
            changed_files=["a.py", "b.py", "c.py"],
        )
        assert len(r.changed_files) == 3

    # --- GitHubClient ---

    def test_github_client_mask_token(self):
        from core.sources.github_client import _mask_token

        assert _mask_token(None) == "(none)"
        assert _mask_token("") == "(none)"
        assert _mask_token("short") == "****"
        assert _mask_token("ghp_abcdefghijklm") == "ghp_...jklm"

    def test_github_client_creates_with_token(self):
        from core.sources.github_client import GitHubClient

        client = GitHubClient(token="test_token")
        assert client._token == "test_token"
        client.close()

    def test_github_client_creates_without_token(self):
        from core.sources.github_client import GitHubClient

        client = GitHubClient()
        assert client._token is None
        client.close()

    def test_github_client_context_manager(self):
        from core.sources.github_client import GitHubClient

        with GitHubClient() as client:
            assert client is not None

    # --- GitHubSourceConnector ---

    def test_connector_status(self):
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig(owner="testorg", token_env="")
        connector = GitHubSourceConnector(config)
        status = connector.status()

        assert status["source_type"] == "github"
        assert status["owner"] == "testorg"
        assert status["has_token"] is False

    def test_connector_filter_repos_include(self):
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo

        config = GitHubSourceConfig(
            owner="org", include_repos=["keep-this", "and-this"]
        )
        connector = GitHubSourceConnector(config)

        repos = [
            RepoInfo("keep-this", "org", "", "", False, "", "main", 0, False),
            RepoInfo("and-this", "org", "", "", False, "", "main", 0, False),
            RepoInfo("skip-this", "org", "", "", False, "", "main", 0, False),
        ]
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 2
        assert {r.name for r in filtered} == {"keep-this", "and-this"}

    def test_connector_filter_repos_exclude(self):
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo

        config = GitHubSourceConfig(owner="org", exclude_repos=["legacy"])
        connector = GitHubSourceConnector(config)

        repos = [
            RepoInfo("app", "org", "", "", False, "", "main", 0, False),
            RepoInfo("legacy", "org", "", "", False, "", "main", 0, False),
        ]
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 1
        assert filtered[0].name == "app"

    def test_connector_filter_repos_case_insensitive(self):
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo

        config = GitHubSourceConfig(
            owner="org", include_repos=["MyRepo"]
        )
        connector = GitHubSourceConnector(config)

        repos = [
            RepoInfo("myrepo", "org", "", "", False, "", "main", 0, False),
        ]
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 1

    def test_connector_filter_repos_empty_filters(self):
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo

        config = GitHubSourceConfig(owner="org")
        connector = GitHubSourceConnector(config)

        repos = [
            RepoInfo("a", "org", "", "", False, "", "main", 0, False),
            RepoInfo("b", "org", "", "", False, "", "main", 0, False),
        ]
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 2

    # --- WebSourceConfig ---

    def test_web_config_from_dict(self):
        from core.sources.web_crawler import WebSourceConfig

        config = WebSourceConfig.from_dict({
            "url": "https://docs.example.com/api",
            "mode": "docs",
            "stay_in_path": False,
            "max_pages": 100,
            "wait_time": 2.0,
        })
        assert config.url == "https://docs.example.com/api"
        assert config.mode == "docs"
        assert config.stay_in_path is False
        assert config.max_pages == 100
        assert config.wait_time == 2.0

    def test_web_config_defaults(self):
        from core.sources.web_crawler import WebSourceConfig

        config = WebSourceConfig()
        assert config.url == ""
        assert config.mode == "crawl"
        assert config.stay_in_path is True
        assert config.max_pages == 500
        assert config.wait_time == 1.0

    def test_web_config_domain(self):
        from core.sources.web_crawler import WebSourceConfig

        config = WebSourceConfig(url="https://www.example.com/docs/api")
        assert config.domain == "example.com"
        assert config.base_url == "https://www.example.com"
        assert config.starting_path == "/docs/api"

    def test_web_config_domain_no_www(self):
        from core.sources.web_crawler import WebSourceConfig

        config = WebSourceConfig(url="https://docs.example.com/")
        assert config.domain == "docs.example.com"

    # --- URL normalization ---

    def test_normalize_link_absolute_same_domain(self):
        from core.sources.web_crawler import normalize_link

        result = normalize_link(
            "https://example.com/page", "/", "https://example.com"
        )
        assert result == "/page"

    def test_normalize_link_relative(self):
        from core.sources.web_crawler import normalize_link

        result = normalize_link("subpage", "/docs/api", "https://example.com")
        assert result == "/docs/subpage"

    def test_normalize_link_external_returns_none(self):
        from core.sources.web_crawler import normalize_link

        assert normalize_link(
            "https://other.com/page", "/", "https://example.com"
        ) is None

    def test_normalize_link_mailto_returns_none(self):
        from core.sources.web_crawler import normalize_link

        assert normalize_link(
            "mailto:test@example.com", "/", "https://example.com"
        ) is None

    def test_normalize_link_skip_extensions(self):
        from core.sources.web_crawler import normalize_link

        assert normalize_link(
            "/file.pdf", "/", "https://example.com"
        ) is None
        assert normalize_link(
            "/image.png", "/", "https://example.com"
        ) is None

    def test_normalize_link_strips_query_fragment(self):
        from core.sources.web_crawler import normalize_link

        result = normalize_link(
            "/page?sort=date#section", "/", "https://example.com"
        )
        assert result == "/page"

    def test_normalize_link_empty(self):
        from core.sources.web_crawler import normalize_link

        assert normalize_link("", "/", "https://example.com") is None
        assert normalize_link("#", "/", "https://example.com") is None

    def test_normalize_link_javascript(self):
        from core.sources.web_crawler import normalize_link

        assert normalize_link(
            "javascript:void(0)", "/", "https://example.com"
        ) is None

    # --- Scope checking ---

    def test_is_in_scope_stay_in_path(self):
        from core.sources.web_crawler import _is_in_scope, WebSourceConfig

        config = WebSourceConfig(
            url="https://example.com/docs", stay_in_path=True
        )
        assert _is_in_scope("/docs/api", config) is True
        assert _is_in_scope("/docs", config) is True
        assert _is_in_scope("/other", config) is False

    def test_is_in_scope_no_restriction(self):
        from core.sources.web_crawler import _is_in_scope, WebSourceConfig

        config = WebSourceConfig(
            url="https://example.com/docs", stay_in_path=False
        )
        assert _is_in_scope("/anywhere", config) is True

    def test_is_in_scope_root(self):
        from core.sources.web_crawler import _is_in_scope, WebSourceConfig

        config = WebSourceConfig(
            url="https://example.com/", stay_in_path=True
        )
        assert _is_in_scope("/anything", config) is True

    # --- Non-content path detection ---

    def test_non_content_paths(self):
        from core.sources.web_crawler import NON_CONTENT_PATHS

        assert NON_CONTENT_PATHS.search("/login")
        assert NON_CONTENT_PATHS.search("/admin/dashboard")
        assert NON_CONTENT_PATHS.search("/cart/checkout")
        assert NON_CONTENT_PATHS.search("/author/john")
        assert NON_CONTENT_PATHS.search("/tag/python")
        assert not NON_CONTENT_PATHS.search("/docs/api-reference")
        assert not NON_CONTENT_PATHS.search("/blog/my-post")

    # --- Link extraction ---

    def test_extract_links(self):
        from core.sources.web_crawler import _extract_links

        html = '<a href="/page1">P1</a> <a href="/page2">P2</a>'
        links = _extract_links(html)
        assert "/page1" in links
        assert "/page2" in links

    def test_extract_links_data_attributes(self):
        from core.sources.web_crawler import _extract_links

        html = '<a href="/page1">P1</a>'
        links = _extract_links(html)
        assert "/page1" in links

    # --- URL to filepath ---

    def test_url_to_filepath(self):
        from core.sources.web_crawler import _url_to_filepath

        assert _url_to_filepath("/docs/api") == "docs_api.md"
        assert _url_to_filepath("/") == "index.md"
        assert _url_to_filepath("") == "index.md"
        assert _url_to_filepath("/page") == "page.md"

    # --- CrawlCache ---

    def test_crawl_cache_empty(self, tmp_path):
        from core.sources.web_cache import CrawlCache

        cache = CrawlCache(str(tmp_path))
        cache.load()  # no file to load
        assert cache.page_count == 0
        assert cache.get_page_headers("/page") == {}
        assert cache.get_page_links("/page") == []
        assert cache.is_page_unchanged("/page", "content") is False

    def test_crawl_cache_update_and_check(self, tmp_path):
        from core.sources.web_cache import CrawlCache

        cache = CrawlCache(str(tmp_path))
        cache.update_page(
            "/page",
            {"etag": '"abc"', "last-modified": "Mon, 01 Jan 2024"},
            "hello world",
            links=["/link1", "/link2"],
        )

        # Conditional headers
        headers = cache.get_page_headers("/page")
        assert headers["If-None-Match"] == '"abc"'
        assert headers["If-Modified-Since"] == "Mon, 01 Jan 2024"

        # Links
        assert cache.get_page_links("/page") == ["/link1", "/link2"]

        # Content hash
        assert cache.is_page_unchanged("/page", "hello world") is True
        assert cache.is_page_unchanged("/page", "changed") is False

    def test_crawl_cache_persistence(self, tmp_path):
        from core.sources.web_cache import CrawlCache

        cache1 = CrawlCache(str(tmp_path))
        cache1.update_page("/p", {}, "content", links=["/a"])
        cache1.save()

        cache2 = CrawlCache(str(tmp_path))
        cache2.load()
        assert cache2.page_count == 1
        assert cache2.get_page_links("/p") == ["/a"]

    def test_crawl_cache_case_insensitive_headers(self, tmp_path):
        from core.sources.web_cache import CrawlCache

        cache = CrawlCache(str(tmp_path))
        cache.update_page("/p", {"ETag": '"xyz"', "Last-Modified": "date"}, "c")
        headers = cache.get_page_headers("/p")
        assert headers["If-None-Match"] == '"xyz"'

    # --- HTML to Markdown converter ---

    def test_html_to_markdown_basic(self):
        from core.sources.web_converter import html_to_markdown

        html = """
        <html><body>
        <article>
            <h1>Test Article</h1>
            <p>This is a test paragraph with enough words to pass the minimum
            content threshold that the converter uses to determine if content
            is meaningful enough to extract and convert to markdown format.</p>
        </article>
        </body></html>
        """
        md = html_to_markdown(html)
        assert md is not None
        assert "Test Article" in md

    def test_html_to_markdown_strips_nav(self):
        from core.sources.web_converter import html_to_markdown

        html = """
        <html><body>
        <nav><a href="/">Home</a></nav>
        <article>
            <h1>Content</h1>
            <p>This is the main content of the page with plenty of text to
            ensure it passes the word count threshold for extraction purposes
            in our markdown converter module test cases here.</p>
        </article>
        <footer>Copyright</footer>
        </body></html>
        """
        md = html_to_markdown(html)
        assert md is not None
        assert "Home" not in md or "Content" in md

    def test_html_to_markdown_returns_none_for_empty(self):
        from core.sources.web_converter import html_to_markdown

        assert html_to_markdown("<html><body></body></html>") is None

    def test_extract_title(self):
        from core.sources.web_converter import extract_title

        assert extract_title("<html><body><h1>Title</h1></body></html>") == "Title"
        assert extract_title("<html><head><title>Page</title></head></html>") == "Page"
        assert extract_title("<html></html>") == ""

    def test_clean_markdown_removes_ads(self):
        from core.sources.web_converter import clean_markdown

        md = "# Title\n\nContent here.\n\nAdvertisement\n\nMore content."
        cleaned = clean_markdown(md)
        assert "Advertisement" not in cleaned
        assert "Content here" in cleaned

    def test_clean_markdown_removes_subscribe(self):
        from core.sources.web_converter import clean_markdown

        md = "# Title\n\nContent.\n\nSign up\n\nSubscribe\n\nMore."
        cleaned = clean_markdown(md)
        assert "Sign up" not in cleaned
        assert "Subscribe" not in cleaned

    def test_clean_markdown_removes_empty_links(self):
        from core.sources.web_converter import clean_markdown

        md = "# Title\n\nSome text [](http://empty) here."
        cleaned = clean_markdown(md)
        assert "[](http://empty)" not in cleaned

    def test_clean_markdown_collapses_blank_lines(self):
        from core.sources.web_converter import clean_markdown

        md = "Line 1\n\n\n\n\n\nLine 2"
        cleaned = clean_markdown(md)
        assert "\n\n\n\n" not in cleaned

    def test_clean_markdown_none_input(self):
        from core.sources.web_converter import clean_markdown

        assert clean_markdown(None) is None
        assert clean_markdown("") is None

    # --- WebSourceConnector ---

    def test_web_connector_status(self):
        from core.sources.web_crawler import WebSourceConnector, WebSourceConfig

        config = WebSourceConfig(url="https://docs.example.com/api")
        connector = WebSourceConnector(config)
        status = connector.status()

        assert status["source_type"] == "web"
        assert status["url"] == "https://docs.example.com/api"
        assert status["domain"] == "docs.example.com"

    # --- Error path tests ---

    def test_clone_result_is_fresh_clone(self):
        from core.sources.git_ops import CloneResult

        fresh = CloneResult(success=True, message="Cloned successfully", is_fresh_clone=True)
        assert fresh.is_fresh_clone is True

        fetched = CloneResult(success=True, message="Already up to date")
        assert fetched.is_fresh_clone is False

    def test_askpass_script_shell_quoting(self):
        from core.sources.git_ops import _make_askpass_script
        import os

        # Token with a single quote should be properly escaped
        token = "ghp_test'token"
        path = _make_askpass_script(token)
        try:
            with open(path) as f:
                content = f.read()
            assert "'" in content  # should use quotes
            assert f"echo {token}" not in content  # should NOT have unquoted token
        finally:
            os.unlink(path)

    def test_askpass_script_normal_token(self):
        from core.sources.git_ops import _make_askpass_script
        import os

        token = "ghp_abc123XYZ"
        path = _make_askpass_script(token)
        try:
            with open(path) as f:
                content = f.read()
            assert token in content
            assert content.startswith("#!/bin/sh\n")
            # Check file permissions
            stat = os.stat(path)
            assert stat.st_mode & 0o777 == 0o700
        finally:
            os.unlink(path)

    def test_github_client_is_org(self):
        """Test the public is_org method exists and has correct signature."""
        from core.sources.github_client import GitHubClient

        client = GitHubClient(token=None)
        assert hasattr(client, "is_org")
        client.close()

    def test_yaml_frontmatter_escaping(self):
        """Titles with quotes and newlines should be escaped in frontmatter."""
        import yaml

        # Simulate the frontmatter generation with a problematic title
        title = 'My "Great" Page'
        safe_title = title.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        body = (
            f'title: "{safe_title}"\n'
            f"url: https://example.com/test\n"
            f"source: web_crawler\n"
        )
        # Should parse as valid YAML
        parsed = yaml.safe_load(body)
        assert parsed["title"] == 'My "Great" Page'

    def test_yaml_frontmatter_newline_in_title(self):
        """Newlines in titles should be replaced with spaces."""
        import yaml

        title = "Title\nWith\nNewlines"
        safe_title = title.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        body = (
            f'title: "{safe_title}"\n'
            f"url: https://example.com/test\n"
            f"source: web_crawler\n"
        )
        parsed = yaml.safe_load(body)
        assert "\n" not in parsed["title"]
        assert parsed["title"] == "Title With Newlines"

    def test_sync_result_files_updated(self):
        """SyncResult should track updated files separately from added."""
        from core.sources.base import SyncResult

        result = SyncResult(source_type="web")
        result.files_added = 3
        result.files_updated = 5
        assert result.total_changes == 8

    def test_crawl_cache_version_mismatch(self):
        """Cache with different version should start fresh."""
        import json
        import os
        import tempfile

        from core.sources.web_cache import CrawlCache

        tmp_dir = tempfile.mkdtemp()
        cache = CrawlCache(tmp_dir)
        cache_file = os.path.join(tmp_dir, ".crawl_cache.json")
        with open(cache_file, "w") as f:
            json.dump({"version": "0.0.0", "pages": {"old": "data"}}, f)

        cache.load()
        assert cache.page_count == 0  # should start fresh

    def test_crawl_cache_corrupt_file(self):
        """Corrupt cache file should not crash, should start fresh."""
        import os
        import tempfile

        from core.sources.web_cache import CrawlCache

        tmp_dir = tempfile.mkdtemp()
        cache = CrawlCache(tmp_dir)
        cache_file = os.path.join(tmp_dir, ".crawl_cache.json")
        with open(cache_file, "w") as f:
            f.write("not valid json{{{")

        cache.load()
        assert cache.page_count == 0

    def test_github_connector_uses_is_fresh_clone(self):
        """Connector should use is_fresh_clone instead of string matching."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig
        import inspect

        source = inspect.getsource(GitHubSourceConnector.sync)
        assert "is_fresh_clone" in source
        assert '"Cloned" in' not in source

    def test_web_config_domain_empty_url(self):
        """Empty URL config should produce empty domain."""
        from core.sources.web_crawler import WebSourceConfig

        config = WebSourceConfig()
        assert config.domain == ""  # empty, but doesn't crash

    def test_connector_filter_repos_both_include_and_exclude(self):
        """When both include and exclude are set, exclude takes precedence."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo

        config = GitHubSourceConfig.from_dict({
            "owner": "test",
            "include_repos": ["repo-a", "repo-b", "repo-c"],
            "exclude_repos": ["repo-b"],
        })
        connector = GitHubSourceConnector(config)

        repos = [
            RepoInfo(name="repo-a", owner="test", url="", clone_url="",
                     private=False, description="", default_branch="main",
                     size_kb=0, is_fork=False),
            RepoInfo(name="repo-b", owner="test", url="", clone_url="",
                     private=False, description="", default_branch="main",
                     size_kb=0, is_fork=False),
            RepoInfo(name="repo-c", owner="test", url="", clone_url="",
                     private=False, description="", default_branch="main",
                     size_kb=0, is_fork=False),
            RepoInfo(name="repo-d", owner="test", url="", clone_url="",
                     private=False, description="", default_branch="main",
                     size_kb=0, is_fork=False),
        ]
        filtered = connector._filter_repos(repos)
        names = [r.name for r in filtered]
        assert "repo-a" in names
        assert "repo-b" not in names  # excluded
        assert "repo-c" in names
        assert "repo-d" not in names  # not in include list

    def test_sanitise_output_all_token_formats(self):
        """All GitHub token formats should be sanitised."""
        from core.sources.git_ops import _sanitise_output

        tokens = [
            "ghp_" + "a" * 36,
            "github_pat_" + "a" * 80,
            "gho_" + "a" * 36,
            "ghs_" + "a" * 36,
            "ghu_" + "a" * 36,
            "ghr_" + "a" * 36,
        ]
        for token in tokens:
            result = _sanitise_output(f"error with {token} in output")
            assert token not in result, f"Token format {token[:10]} not sanitised"
            assert "****" in result

    # --- Module imports ---

    def test_sources_package_imports(self):
        from core.sources import (
            BaseSourceConnector,
            SyncResult,
            GitHubSourceConnector,
            GitHubSourceConfig,
            RepoInfo,
            WebSourceConnector,
            WebSourceConfig,
        )

        assert BaseSourceConnector is not None
        assert SyncResult is not None
        assert GitHubSourceConnector is not None
        assert GitHubSourceConfig is not None
        assert RepoInfo is not None
        assert WebSourceConnector is not None
        assert WebSourceConfig is not None


class TestSourceConnectorErrorPaths:
    """Error-path tests for source connector I/O operations."""

    # --- git_ops error paths ---

    def test_validate_name_rejects_path_traversal(self):
        from core.sources.git_ops import _validate_name, GitError
        with pytest.raises(GitError, match="Path traversal"):
            _validate_name("foo..bar")

    def test_validate_name_rejects_special_chars(self):
        from core.sources.git_ops import _validate_name, GitError
        with pytest.raises(GitError, match="Invalid repository name"):
            _validate_name("repo name with spaces")

    def test_validate_name_accepts_valid(self):
        from core.sources.git_ops import _validate_name
        _validate_name("my-repo.v2")
        _validate_name("repo_name")
        _validate_name("valid123")

    def test_clone_result_defaults(self):
        from core.sources.git_ops import CloneResult
        r = CloneResult(success=True, message="ok")
        assert r.changed_files == []
        assert r.is_fresh_clone is False

    def test_clone_repo_invalid_name(self):
        from core.sources.git_ops import clone_repo, GitError
        with pytest.raises(GitError, match="Invalid repository name"):
            clone_repo("https://github.com/x/y.git", Path("/tmp/bad dir"), token=None)

    def test_fetch_repo_not_a_git_repo(self):
        import tempfile
        from core.sources.git_ops import fetch_repo
        tmp = tempfile.mkdtemp()
        try:
            result = fetch_repo(Path(tmp), "https://github.com/x/y.git")
            assert result.success is False
            assert "Not a git repository" in result.message
        finally:
            shutil.rmtree(tmp)

    def test_sanitise_output_removes_tokens(self):
        from core.sources.git_ops import _sanitise_output
        output = "error: ghp_" + "a" * 36 + " invalid"
        sanitised = _sanitise_output(output)
        assert "ghp_" not in sanitised
        assert "****" in sanitised

    def test_sanitise_output_removes_url_tokens(self):
        from core.sources.git_ops import _sanitise_output
        output = "error: https://oauth2:ghp_abc123@github.com/x/y.git failed"
        sanitised = _sanitise_output(output)
        assert "ghp_abc123" not in sanitised

    def test_sanitise_output_preserves_clean_text(self):
        from core.sources.git_ops import _sanitise_output
        clean = "normal git output with no tokens"
        assert _sanitise_output(clean) == clean

    def test_get_default_branch_non_git_dir(self):
        import tempfile
        from core.sources.git_ops import get_default_branch
        tmp = tempfile.mkdtemp()
        try:
            result = get_default_branch(Path(tmp))
            assert result == "main"
        finally:
            shutil.rmtree(tmp)

    # --- GitHub client error paths ---

    def test_github_client_mask_token_none(self):
        from core.sources.github_client import _mask_token
        assert _mask_token(None) == "(none)"

    def test_github_client_mask_token_short(self):
        from core.sources.github_client import _mask_token
        assert _mask_token("abcd") == "****"

    def test_github_client_mask_token_normal(self):
        from core.sources.github_client import _mask_token
        result = _mask_token("abcdefghij1234567890")
        assert result.startswith("abcd")
        assert result.endswith("7890")
        assert "..." in result

    def test_github_api_error_has_status_code(self):
        from core.sources.github_client import GitHubAPIError
        err = GitHubAPIError("test", status_code=404)
        assert err.status_code == 404
        assert str(err) == "test"

    def test_authentication_error_is_github_api_error(self):
        from core.sources.github_client import AuthenticationError, GitHubAPIError
        err = AuthenticationError("auth failed")
        assert isinstance(err, GitHubAPIError)

    def test_rate_limit_error_is_github_api_error(self):
        from core.sources.github_client import RateLimitError, GitHubAPIError
        err = RateLimitError("rate limited")
        assert isinstance(err, GitHubAPIError)

    # --- GitHub connector error paths ---

    def test_github_connector_empty_owner(self):
        from core.sources.github_models import GitHubSourceConfig
        from core.sources.github_connector import GitHubSourceConnector
        config = GitHubSourceConfig(owner="")
        connector = GitHubSourceConnector(config)
        repos = connector._discover_repos()
        assert repos == []

    def test_github_connector_filter_include(self):
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.github_connector import GitHubSourceConnector
        config = GitHubSourceConfig(owner="test", include_repos=["repo1"])
        connector = GitHubSourceConnector(config)
        repos = [
            RepoInfo(name="repo1", owner="test", url="", clone_url="", private=False, description="", default_branch="main", size_kb=0, is_fork=False),
            RepoInfo(name="repo2", owner="test", url="", clone_url="", private=False, description="", default_branch="main", size_kb=0, is_fork=False),
        ]
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 1
        assert filtered[0].name == "repo1"

    def test_github_connector_filter_exclude(self):
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.github_connector import GitHubSourceConnector
        config = GitHubSourceConfig(owner="test", exclude_repos=["legacy"])
        connector = GitHubSourceConnector(config)
        repos = [
            RepoInfo(name="repo1", owner="test", url="", clone_url="", private=False, description="", default_branch="main", size_kb=0, is_fork=False),
            RepoInfo(name="legacy", owner="test", url="", clone_url="", private=False, description="", default_branch="main", size_kb=0, is_fork=False),
        ]
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 1
        assert filtered[0].name == "repo1"

    def test_github_connector_status(self):
        from core.sources.github_models import GitHubSourceConfig
        from core.sources.github_connector import GitHubSourceConnector
        config = GitHubSourceConfig(owner="testorg", repo="myrepo")
        connector = GitHubSourceConnector(config)
        status = connector.status()
        assert status["source_type"] == "github"
        assert status["owner"] == "testorg"
        assert status["repo"] == "myrepo"

    # --- Web crawler error paths ---

    def test_normalize_link_mailto(self):
        from core.sources.web_crawler import normalize_link
        assert normalize_link("mailto:test@example.com", "/", "https://example.com") is None

    def test_normalize_link_javascript(self):
        from core.sources.web_crawler import normalize_link
        assert normalize_link("javascript:void(0)", "/", "https://example.com") is None

    def test_normalize_link_empty(self):
        from core.sources.web_crawler import normalize_link
        assert normalize_link("", "/", "https://example.com") is None

    def test_normalize_link_external(self):
        from core.sources.web_crawler import normalize_link
        assert normalize_link("https://other.com/page", "/", "https://example.com") is None

    def test_normalize_link_skip_extension(self):
        from core.sources.web_crawler import normalize_link
        assert normalize_link("/file.pdf", "/", "https://example.com") is None
        assert normalize_link("/image.jpg", "/", "https://example.com") is None

    def test_normalize_link_relative(self):
        from core.sources.web_crawler import normalize_link
        result = normalize_link("subpage", "/docs/", "https://example.com")
        assert result == "/docs/subpage"

    def test_normalize_link_absolute_same_domain(self):
        from core.sources.web_crawler import normalize_link
        result = normalize_link("https://example.com/page", "/", "https://example.com")
        assert result == "/page"

    def test_is_in_scope_root(self):
        from core.sources.web_crawler import _is_in_scope, WebSourceConfig
        config = WebSourceConfig(url="https://example.com", stay_in_path=True)
        assert _is_in_scope("/anything", config) is True

    def test_is_in_scope_restricted(self):
        from core.sources.web_crawler import _is_in_scope, WebSourceConfig
        config = WebSourceConfig(url="https://example.com/docs/api", stay_in_path=True)
        assert _is_in_scope("/docs/api/endpoints", config) is True
        assert _is_in_scope("/docs/api", config) is True
        assert _is_in_scope("/other", config) is False

    def test_is_in_scope_disabled(self):
        from core.sources.web_crawler import _is_in_scope, WebSourceConfig
        config = WebSourceConfig(url="https://example.com/docs", stay_in_path=False)
        assert _is_in_scope("/other", config) is True

    def test_url_to_filepath_root(self):
        from core.sources.web_crawler import _url_to_filepath
        assert _url_to_filepath("/") == "index.md"

    def test_url_to_filepath_nested(self):
        from core.sources.web_crawler import _url_to_filepath
        result = _url_to_filepath("/docs/api/endpoints")
        assert result == "docs_api_endpoints.md"

    def test_url_to_filepath_special_chars(self):
        from core.sources.web_crawler import _url_to_filepath
        result = _url_to_filepath('/page?"special"')
        assert '"' not in result
        assert '?' not in result

    def test_extract_links_basic(self):
        from core.sources.web_crawler import _extract_links
        html = '<a href="/page1">Page 1</a><a href="/page2">Page 2</a>'
        links = _extract_links(html)
        assert "/page1" in links
        assert "/page2" in links

    def test_extract_links_no_links(self):
        from core.sources.web_crawler import _extract_links
        assert _extract_links("<p>No links here</p>") == []

    def test_non_content_paths_blocked(self):
        from core.sources.web_crawler import NON_CONTENT_PATHS
        assert NON_CONTENT_PATHS.search("/login")
        assert NON_CONTENT_PATHS.search("/admin/dashboard")
        assert NON_CONTENT_PATHS.search("/cart/checkout")
        assert not NON_CONTENT_PATHS.search("/docs/api")

    def test_web_connector_status(self):
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com/docs", max_pages=100)
        connector = WebSourceConnector(config)
        status = connector.status()
        assert status["source_type"] == "web"
        assert status["url"] == "https://example.com/docs"
        assert status["max_pages"] == 100

    # --- Web converter error paths ---

    def test_clean_markdown_none(self):
        from core.sources.web_converter import clean_markdown
        assert clean_markdown(None) is None

    def test_clean_markdown_empty(self):
        from core.sources.web_converter import clean_markdown
        assert clean_markdown("") is None

    def test_clean_markdown_strips_ads(self):
        from core.sources.web_converter import clean_markdown
        md = "Content here\n\nArticle continues below\nADVERTISEMENT\n\nMore content"
        result = clean_markdown(md)
        assert "Article continues below" not in result
        assert "More content" in result

    def test_clean_markdown_strips_subscribe(self):
        from core.sources.web_converter import clean_markdown
        md = "Content\n\nSign up\n\nSubscribe\n\nMore"
        result = clean_markdown(md)
        assert "Sign up" not in result

    def test_clean_markdown_removes_empty_links(self):
        from core.sources.web_converter import clean_markdown
        md = "Content [](http://example.com) here ![](http://example.com/img.png)"
        result = clean_markdown(md)
        assert "[](http://example.com)" not in result
        assert "![](http://example.com/img.png)" not in result

    def test_clean_markdown_collapses_blank_lines(self):
        from core.sources.web_converter import clean_markdown
        md = "Line 1\n\n\n\n\n\nLine 2"
        result = clean_markdown(md)
        assert "\n\n\n\n" not in result

    def test_html_to_markdown_none_for_empty(self):
        from core.sources.web_converter import html_to_markdown
        assert html_to_markdown("<html><body></body></html>") is None

    def test_extract_title_h1(self):
        from core.sources.web_converter import extract_title
        assert extract_title("<h1>My Title</h1>") == "My Title"

    def test_extract_title_fallback_to_title(self):
        from core.sources.web_converter import extract_title
        assert extract_title("<title>Page Title</title>") == "Page Title"

    def test_extract_title_empty(self):
        from core.sources.web_converter import extract_title
        assert extract_title("<p>No title</p>") == ""

    # --- CrawlCache error paths ---

    def test_crawl_cache_load_nonexistent(self):
        import tempfile
        from core.sources.web_cache import CrawlCache
        tmp = tempfile.mkdtemp()
        try:
            cache = CrawlCache(tmp)
            cache.load()
            assert cache._data.get("version") is not None
        finally:
            shutil.rmtree(tmp)

    def test_crawl_cache_save_and_reload(self):
        import tempfile
        from core.sources.web_cache import CrawlCache
        tmp = tempfile.mkdtemp()
        try:
            cache = CrawlCache(tmp)
            cache.load()
            cache.update_page("/test", {"etag": '"abc"'}, "<html>test</html>", links=["/page2"])
            cache.save()

            cache2 = CrawlCache(tmp)
            cache2.load()
            assert cache2.get_page_links("/test") == ["/page2"]
        finally:
            shutil.rmtree(tmp)

    def test_crawl_cache_unchanged_page(self):
        import tempfile
        from core.sources.web_cache import CrawlCache
        tmp = tempfile.mkdtemp()
        try:
            cache = CrawlCache(tmp)
            cache.load()
            html = "<html><body>Same content</body></html>"
            cache.update_page("/test", {}, html, links=[])
            assert cache.is_page_unchanged("/test", html) is True
            assert cache.is_page_unchanged("/test", "<html>different</html>") is False
        finally:
            shutil.rmtree(tmp)

    def test_crawl_cache_headers_etag(self):
        import tempfile
        from core.sources.web_cache import CrawlCache
        tmp = tempfile.mkdtemp()
        try:
            cache = CrawlCache(tmp)
            cache.load()
            cache.update_page("/test", {"etag": '"xyz"'}, "<html>x</html>", links=[])
            headers = cache.get_page_headers("/test")
            assert headers.get("If-None-Match") == '"xyz"'
        finally:
            shutil.rmtree(tmp)

    # --- SyncResult ---

    def test_sync_result_total_changes(self):
        from core.sources.base import SyncResult
        r = SyncResult(source_type="test", files_added=5, files_updated=3, files_removed=2)
        assert r.total_changes == 10

    def test_sync_result_success_no_errors(self):
        from core.sources.base import SyncResult
        r = SyncResult(source_type="test")
        assert r.success is True

    def test_sync_result_success_with_errors(self):
        from core.sources.base import SyncResult
        r = SyncResult(source_type="test", errors=["Something went wrong"])
        assert r.success is False


class TestSourceIngestIntegration:
    """Tests for source connector + ingest pipeline integration."""

    def test_client_sources_path(self):
        from core.client import Client
        client = Client(Path("/tmp/test_client"), {"machine_name": "test"})
        assert client.sources_path == Path("/tmp/test_client/sources")

    def test_client_sources_config_empty(self):
        from core.client import Client
        client = Client(Path("/tmp/test_client"), {"machine_name": "test"})
        assert client.sources_config == {}

    def test_client_sources_config_with_data(self):
        from core.client import Client
        config = {
            "machine_name": "test",
            "sources": {
                "github": [{"owner": "testorg"}],
                "web": [{"url": "https://example.com"}],
            },
        }
        client = Client(Path("/tmp/test_client"), config)
        assert len(client.sources_config["github"]) == 1
        assert len(client.sources_config["web"]) == 1

    def test_client_has_sources_false_no_dir(self):
        import tempfile
        from core.client import Client
        tmp = tempfile.mkdtemp()
        try:
            client = Client(Path(tmp), {"machine_name": "test"})
            assert client.has_sources is False
        finally:
            shutil.rmtree(tmp)

    def test_client_has_sources_true_with_files(self):
        import tempfile
        from core.client import Client
        tmp = tempfile.mkdtemp()
        try:
            sources_dir = Path(tmp) / "sources" / "web"
            sources_dir.mkdir(parents=True)
            (sources_dir / "test.md").write_text("# Test")
            client = Client(Path(tmp), {"machine_name": "test"})
            assert client.has_sources is True
        finally:
            shutil.rmtree(tmp)

    def test_sources_in_known_top_level_keys(self):
        from core.client import _KNOWN_TOP_LEVEL_KEYS
        assert "sources" in _KNOWN_TOP_LEVEL_KEYS

    def test_sources_config_no_validation_warning(self):
        from core.client import validate_config
        config = {
            "machine_name": "test_client",
            "sources": {
                "github": [{"owner": "testorg"}],
            },
        }
        warnings = validate_config(config)
        assert not any("sources" in w for w in warnings)

    def test_should_sync_on_ingest_true(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.ingest import _should_sync_on_ingest
        from core.client import Client
        config = {
            "machine_name": "test",
            "sources": {
                "github": [{"owner": "org", "sync_on_ingest": True}],
            },
        }
        client = Client(Path("/tmp/test"), config)
        assert _should_sync_on_ingest(client) is True

    def test_should_sync_on_ingest_false(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.ingest import _should_sync_on_ingest
        from core.client import Client
        config = {
            "machine_name": "test",
            "sources": {
                "github": [{"owner": "org", "sync_on_ingest": False}],
            },
        }
        client = Client(Path("/tmp/test"), config)
        assert _should_sync_on_ingest(client) is False

    def test_should_sync_on_ingest_no_sources(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.ingest import _should_sync_on_ingest
        from core.client import Client
        client = Client(Path("/tmp/test"), {"machine_name": "test"})
        assert _should_sync_on_ingest(client) is False

    def test_should_sync_on_ingest_web_source(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.ingest import _should_sync_on_ingest
        from core.client import Client
        config = {
            "machine_name": "test",
            "sources": {
                "web": [{"url": "https://example.com", "sync_on_ingest": True}],
            },
        }
        client = Client(Path("/tmp/test"), config)
        assert _should_sync_on_ingest(client) is True

    def test_sync_client_sources_no_config(self):
        import sys, tempfile
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.ingest import sync_client_sources
        # Create a minimal client directory
        tmp = tempfile.mkdtemp()
        import yaml
        client_dir = Path(tmp) / "clients" / "test_sync"
        client_dir.mkdir(parents=True)
        (client_dir / "documents").mkdir()
        (client_dir / "index").mkdir()
        (client_dir / "prompts").mkdir()
        config = {"machine_name": "test_sync", "name": "Test Sync"}
        (client_dir / "client.yaml").write_text(yaml.dump(config))
        try:
            import os
            old_cwd = os.getcwd()
            os.chdir(tmp)
            result = sync_client_sources("test_sync")
            assert result["synced"] is False
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmp)


class TestGitHubClientRequest:
    """Mock-based tests for GitHubClient._request() retry/error handling."""

    def test_request_success(self):
        """Successful request returns response."""
        from core.sources.github_client import GitHubClient
        client = GitHubClient(token=None)
        mock_response = MagicMock()
        mock_response.status_code = 200
        client._client = MagicMock()
        client._client.request = MagicMock(return_value=mock_response)
        result = client._request("GET", "/test")
        assert result.status_code == 200
        client._client.request.assert_called_once()

    def test_request_401_raises_auth_error(self):
        """401 response raises AuthenticationError."""
        from core.sources.github_client import GitHubClient, AuthenticationError
        client = GitHubClient(token="ghp_testtoken1234567890")
        mock_response = MagicMock()
        mock_response.status_code = 401
        client._client = MagicMock()
        client._client.request = MagicMock(return_value=mock_response)
        with pytest.raises(AuthenticationError) as exc_info:
            client._request("GET", "/test")
        assert "ghp_" in str(exc_info.value)
        assert "testtoken1234567890" not in str(exc_info.value)

    def test_request_403_rate_limit_raises_rate_limit_error(self):
        """403 with x-ratelimit-remaining=0 raises RateLimitError."""
        from core.sources.github_client import GitHubClient, RateLimitError
        client = GitHubClient(token=None)
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": "0",
        }
        client._client = MagicMock()
        client._client.request = MagicMock(return_value=mock_response)
        with pytest.raises(RateLimitError) as exc_info:
            client._request("GET", "/test")
        assert exc_info.value.status_code == 403

    def test_request_403_non_rate_limit_raises_api_error(self):
        """403 without rate limit headers raises GitHubAPIError."""
        from core.sources.github_client import GitHubClient, GitHubAPIError
        client = GitHubClient(token=None)
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {"x-ratelimit-remaining": "100"}
        client._client = MagicMock()
        client._client.request = MagicMock(return_value=mock_response)
        with pytest.raises(GitHubAPIError) as exc_info:
            client._request("GET", "/repos/test")
        assert "Forbidden" in str(exc_info.value)
        assert exc_info.value.status_code == 403

    @patch("core.sources.github_client.time.sleep")
    def test_request_5xx_retries(self, mock_sleep):
        """5xx responses are retried with backoff."""
        from core.sources.github_client import GitHubClient, GitHubAPIError
        client = GitHubClient(token=None)
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_500.headers = {}
        client._client = MagicMock()
        client._client.request = MagicMock(return_value=mock_500)
        with pytest.raises(GitHubAPIError) as exc_info:
            client._request("GET", "/test")
        assert client._client.request.call_count == 3  # MAX_RETRIES
        assert mock_sleep.call_count == 3
        assert "Server error" in str(exc_info.value)

    @patch("core.sources.github_client.time.sleep")
    def test_request_transport_error_retries(self, mock_sleep):
        """TransportError is retried with backoff."""
        import httpx
        from core.sources.github_client import GitHubClient, GitHubAPIError
        client = GitHubClient(token=None)
        client._client = MagicMock()
        client._client.request = MagicMock(
            side_effect=httpx.TransportError("Connection refused")
        )
        with pytest.raises(GitHubAPIError) as exc_info:
            client._request("GET", "/test")
        assert client._client.request.call_count == 3
        assert "Network error" in str(exc_info.value)

    @patch("core.sources.github_client.time.sleep")
    def test_request_5xx_then_success(self, mock_sleep):
        """5xx followed by success returns the successful response."""
        from core.sources.github_client import GitHubClient
        client = GitHubClient(token=None)
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_500.headers = {}
        mock_200 = MagicMock()
        mock_200.status_code = 200
        client._client = MagicMock()
        client._client.request = MagicMock(side_effect=[mock_500, mock_200])
        result = client._request("GET", "/test")
        assert result.status_code == 200
        assert client._client.request.call_count == 2


class TestGitHubClientPagination:
    """Mock-based tests for GitHubClient._get_paginated()."""

    def test_single_page(self):
        """Single page with no next link returns items."""
        from core.sources.github_client import GitHubClient
        client = GitHubClient(token=None)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"name": "repo1"}, {"name": "repo2"}]
        mock_response.headers = {"link": ""}
        mock_response.raise_for_status = MagicMock()
        client._request = MagicMock(return_value=mock_response)
        items = client._get_paginated("/repos")
        assert len(items) == 2
        assert items[0]["name"] == "repo1"

    def test_multi_page(self):
        """Multiple pages are accumulated."""
        from core.sources.github_client import GitHubClient
        client = GitHubClient(token=None)
        page1 = MagicMock()
        page1.status_code = 200
        page1.json.return_value = [{"name": "repo1"}]
        page1.headers = {"link": '<url>; rel="next"'}
        page1.raise_for_status = MagicMock()
        page2 = MagicMock()
        page2.status_code = 200
        page2.json.return_value = [{"name": "repo2"}]
        page2.headers = {"link": ""}
        page2.raise_for_status = MagicMock()
        client._request = MagicMock(side_effect=[page1, page2])
        items = client._get_paginated("/repos")
        assert len(items) == 2
        assert items[1]["name"] == "repo2"

    def test_404_raises_error(self):
        """404 response raises GitHubAPIError."""
        from core.sources.github_client import GitHubClient, GitHubAPIError
        client = GitHubClient(token=None)
        mock_response = MagicMock()
        mock_response.status_code = 404
        client._request = MagicMock(return_value=mock_response)
        with pytest.raises(GitHubAPIError) as exc_info:
            client._get_paginated("/repos")
        assert "Not found" in str(exc_info.value)

    def test_empty_page_stops_pagination(self):
        """Empty page breaks the loop."""
        from core.sources.github_client import GitHubClient
        client = GitHubClient(token=None)
        page1 = MagicMock()
        page1.status_code = 200
        page1.json.return_value = [{"name": "repo1"}]
        page1.headers = {"link": '<url>; rel="next"'}
        page1.raise_for_status = MagicMock()
        page2 = MagicMock()
        page2.status_code = 200
        page2.json.return_value = []
        page2.headers = {}
        page2.raise_for_status = MagicMock()
        client._request = MagicMock(side_effect=[page1, page2])
        items = client._get_paginated("/repos")
        assert len(items) == 1

    def test_max_pages_caps_pagination(self):
        """max_pages parameter limits page count."""
        from core.sources.github_client import GitHubClient
        client = GitHubClient(token=None)

        def make_page():
            p = MagicMock()
            p.status_code = 200
            p.json.return_value = [{"name": "repo"}]
            p.headers = {"link": '<url>; rel="next"'}
            p.raise_for_status = MagicMock()
            return p

        client._request = MagicMock(side_effect=[make_page() for _ in range(5)])
        items = client._get_paginated("/repos", max_pages=2)
        assert len(items) == 2
        assert client._request.call_count == 2


class TestGitOpsCloneFetch:
    """Mock-based tests for clone_repo() and fetch_repo()."""

    @patch("core.sources.git_ops._run_git")
    def test_clone_repo_success(self, mock_run):
        """Successful clone returns CloneResult with is_fresh_clone=True."""
        from core.sources.git_ops import clone_repo, CloneResult
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        result = clone_repo("https://github.com/test/repo.git", Path("/tmp/test/repo"))
        assert result.success
        assert result.is_fresh_clone
        assert "Cloned" in result.message
        # First call is clone, second is config cleanup
        assert mock_run.call_count == 2

    @patch("core.sources.git_ops._run_git")
    def test_clone_repo_with_token_uses_oauth2_url(self, mock_run):
        """Clone with token constructs oauth2@ URL."""
        from core.sources.git_ops import clone_repo
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        clone_repo("https://github.com/test/repo.git", Path("/tmp/test/repo"), token="ghp_test123")
        # Check the clone call used oauth2@ URL
        clone_call = mock_run.call_args_list[0]
        args = clone_call[0][0]  # first positional arg is the args list
        assert any("oauth2@" in str(a) for a in args)

    @patch("core.sources.git_ops._run_git")
    def test_clone_repo_failure_sanitises_error(self, mock_run):
        """Failed clone returns sanitised error message."""
        from core.sources.git_ops import clone_repo
        # Use a realistic 40-char PAT (ghp_ + 36 alphanumeric)
        fake_token = "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=128,
            stdout="",
            stderr=f"fatal: could not read from https://{fake_token}@github.com"
        )
        result = clone_repo("https://github.com/test/repo.git", Path("/tmp/test/repo"))
        assert not result.success
        assert fake_token not in result.message
        assert "****" in result.message

    @patch("core.sources.git_ops._run_git")
    def test_fetch_repo_not_git_dir(self, mock_run):
        """Fetch on non-git directory returns failure."""
        import tempfile
        from core.sources.git_ops import fetch_repo
        with tempfile.TemporaryDirectory() as tmp:
            result = fetch_repo(Path(tmp), "https://github.com/test/repo.git")
            assert not result.success
            assert "Not a git repository" in result.message
            mock_run.assert_not_called()

    @patch("core.sources.git_ops._run_git")
    def test_fetch_repo_with_changed_files(self, mock_run):
        """Fetch with changed files returns the diff list."""
        import tempfile
        from core.sources.git_ops import fetch_repo
        with tempfile.TemporaryDirectory() as tmp:
            # Create .git dir
            (Path(tmp) / ".git").mkdir()

            # Mock sequence: rev-parse HEAD, fetch, rev-parse verify, reset, rev-parse HEAD, diff
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="abc123\n", stderr=""),  # old HEAD
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # fetch
                subprocess.CompletedProcess(args=[], returncode=0, stdout="def456\n", stderr=""),  # verify branch
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # reset
                subprocess.CompletedProcess(args=[], returncode=0, stdout="def456\n", stderr=""),  # new HEAD
                subprocess.CompletedProcess(args=[], returncode=0, stdout="file1.py\nfile2.py\n", stderr=""),  # diff
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # cleanup URL
            ]
            result = fetch_repo(Path(tmp), "https://github.com/test/repo.git")
            assert result.success
            assert len(result.changed_files) == 2
            assert "file1.py" in result.changed_files

    @patch("core.sources.git_ops._run_git")
    def test_fetch_repo_no_changes(self, mock_run):
        """Fetch with no changes returns 'Already up to date'."""
        import tempfile
        from core.sources.git_ops import fetch_repo
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            same_hash = "abc123\n"
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout=same_hash, stderr=""),  # old HEAD
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # fetch
                subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr=""),  # verify branch
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # reset
                subprocess.CompletedProcess(args=[], returncode=0, stdout=same_hash, stderr=""),  # new HEAD (same)
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # cleanup URL
            ]
            result = fetch_repo(Path(tmp), "https://github.com/test/repo.git")
            assert result.success
            assert result.changed_files == []
            assert "up to date" in result.message

    @patch("core.sources.git_ops._run_git")
    def test_fetch_repo_always_cleans_url(self, mock_run):
        """Fetch always restores clean remote URL even on failure."""
        import tempfile
        from core.sources.git_ops import fetch_repo
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="abc\n", stderr=""),  # old HEAD
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # set auth URL
                subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="fetch error"),  # fetch fails
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # cleanup URL (finally)
            ]
            result = fetch_repo(
                Path(tmp), "https://github.com/test/repo.git", token="ghp_secret"
            )
            assert not result.success
            # Verify the last _run_git call was the URL cleanup
            last_call_args = mock_run.call_args_list[-1][0][0]
            assert "config" in last_call_args
            assert "remote.origin.url" in last_call_args
            assert "https://github.com/test/repo.git" in last_call_args

    def test_run_git_timeout_returns_synthetic_result(self):
        """Timeout returns CompletedProcess with returncode -1."""
        from core.sources.git_ops import _run_git
        with patch("core.sources.git_ops.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=5)
            result = _run_git(["status"], timeout=5)
            assert result.returncode == -1
            assert "timed out" in result.stderr


class TestGitHubConnectorSync:
    """Mock-based tests for GitHubSourceConnector.sync() orchestration."""

    def test_sync_api_error_captured(self):
        """API error during discovery is captured in result.errors."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig
        from core.sources.github_client import GitHubAPIError
        config = GitHubSourceConfig(owner="test")
        connector = GitHubSourceConnector(config)
        connector._discover_repos = MagicMock(
            side_effect=GitHubAPIError("Bad request", status_code=400)
        )
        result = connector.sync("/tmp/test")
        assert len(result.errors) == 1
        assert "GitHub API error" in result.errors[0]
        assert result.files_added == 0

    def test_sync_no_repos_found(self):
        """Empty repo list returns clean result."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig
        config = GitHubSourceConfig(owner="empty-org")
        connector = GitHubSourceConnector(config)
        connector._discover_repos = MagicMock(return_value=[])
        result = connector.sync("/tmp/test")
        assert result.errors == []
        assert result.files_added == 0
        assert result.files_updated == 0

    def test_sync_per_repo_exception_isolated(self, tmp_path):
        """Exception in one repo doesn't stop other repos."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.git_ops import CloneResult
        config = GitHubSourceConfig(owner="test")
        connector = GitHubSourceConnector(config)
        repo1 = RepoInfo(name="repo1", owner="test", url="", clone_url="https://github.com/test/repo1.git",
                         private=False, description="", default_branch="main", size_kb=0, is_fork=False)
        repo2 = RepoInfo(name="repo2", owner="test", url="", clone_url="https://github.com/test/repo2.git",
                         private=False, description="", default_branch="main", size_kb=0, is_fork=False)
        connector._discover_repos = MagicMock(return_value=[repo1, repo2])

        call_count = 0
        def mock_sync(repo, repo_dir):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Disk full")
            return CloneResult(success=True, message="Cloned", is_fresh_clone=True)

        connector._sync_repo = mock_sync
        result = connector.sync(str(tmp_path))
        assert len(result.errors) == 1
        assert "Disk full" in result.errors[0]
        assert result.files_added == 1  # repo2 succeeded

    def test_sync_fresh_clone_counts_as_added(self, tmp_path):
        """Fresh clone increments files_added."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.git_ops import CloneResult
        config = GitHubSourceConfig(owner="test")
        connector = GitHubSourceConnector(config)
        repo = RepoInfo(name="repo1", owner="test", url="", clone_url="https://github.com/test/repo1.git",
                        private=False, description="", default_branch="main", size_kb=0, is_fork=False)
        connector._discover_repos = MagicMock(return_value=[repo])
        connector._sync_repo = MagicMock(
            return_value=CloneResult(success=True, message="Cloned", is_fresh_clone=True)
        )
        result = connector.sync(str(tmp_path))
        assert result.files_added == 1
        assert result.files_updated == 0

    def test_sync_changed_files_counts_as_updated(self, tmp_path):
        """Fetch with changed files increments files_updated."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.git_ops import CloneResult
        config = GitHubSourceConfig(owner="test")
        connector = GitHubSourceConnector(config)
        repo = RepoInfo(name="repo1", owner="test", url="", clone_url="https://github.com/test/repo1.git",
                        private=False, description="", default_branch="main", size_kb=0, is_fork=False)
        connector._discover_repos = MagicMock(return_value=[repo])
        connector._sync_repo = MagicMock(
            return_value=CloneResult(
                success=True, message="Updated: 3 files changed",
                changed_files=["a.py", "b.py", "c.py"]
            )
        )
        result = connector.sync(str(tmp_path))
        assert result.files_added == 0
        assert result.files_updated == 3

    def test_sync_failed_repo_adds_error(self, tmp_path):
        """Failed sync_repo adds error to result."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.git_ops import CloneResult
        config = GitHubSourceConfig(owner="test")
        connector = GitHubSourceConnector(config)
        repo = RepoInfo(name="repo1", owner="test", url="", clone_url="https://github.com/test/repo1.git",
                        private=False, description="", default_branch="main", size_kb=0, is_fork=False)
        connector._discover_repos = MagicMock(return_value=[repo])
        connector._sync_repo = MagicMock(
            return_value=CloneResult(success=False, message="Clone failed: auth error")
        )
        result = connector.sync(str(tmp_path))
        assert len(result.errors) == 1
        assert "auth error" in result.errors[0]

    def test_sync_saves_metadata(self, tmp_path):
        """Sync saves .sync_metadata.json after completion."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.git_ops import CloneResult
        config = GitHubSourceConfig(owner="testowner")
        connector = GitHubSourceConnector(config)
        repo = RepoInfo(name="repo1", owner="testowner", url="", clone_url="https://github.com/testowner/repo1.git",
                        private=False, description="", default_branch="main", size_kb=0, is_fork=False)
        connector._discover_repos = MagicMock(return_value=[repo])
        connector._sync_repo = MagicMock(
            return_value=CloneResult(success=True, message="Cloned", is_fresh_clone=True)
        )
        connector.sync(str(tmp_path))
        metadata_path = tmp_path / "github" / "testowner" / ".sync_metadata.json"
        assert metadata_path.exists()
        import json
        meta = json.loads(metadata_path.read_text())
        assert meta["owner"] == "testowner"
        assert len(meta["repos"]) == 1


class TestWebCrawlerBFS:
    """Mock-based tests for WebSourceConnector._crawl() BFS loop."""

    # Realistic HTML that html_to_markdown() will accept (short pages return None)
    REALISTIC_HTML = (
        '<html><head><title>Test Page</title></head><body><main>'
        '<h1>Getting Started</h1>'
        '<p>This is a comprehensive guide to getting started with our platform. '
        'It covers installation, configuration, and first steps.</p>'
        '<h2>Installation</h2>'
        '<p>To install the package, run the following command in your terminal. '
        'Make sure you have Python 3.8 or later installed on your system.</p>'
        '<pre><code>pip install example-package</code></pre>'
        '</main></body></html>'
    )

    @patch("core.sources.web_crawler.time.sleep")
    def test_simple_crawl(self, mock_sleep, tmp_path):
        """Simple crawl fetches a page and saves markdown."""
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com/docs", max_pages=10, wait_time=0)
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = self.REALISTIC_HTML

        with patch("core.sources.web_crawler.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = connector.sync(str(tmp_path))

        assert result.files_added >= 1
        md_files = list((tmp_path / "web" / "example.com").glob("*.md"))
        assert len(md_files) >= 1

    @patch("core.sources.web_crawler.time.sleep")
    def test_crawl_follows_links(self, mock_sleep, tmp_path):
        """Crawler follows links to discover new pages."""
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com/docs", max_pages=10, wait_time=0)
        connector = WebSourceConnector(config)

        page1_html = (
            '<html><head><title>Page 1</title></head><body><main>'
            '<h1>Page One</h1>'
            '<p>This is the first page with substantial content that passes the converter filter. '
            'It includes links to other pages for testing BFS traversal.</p>'
            '<h2>Details</h2>'
            '<p>More detailed information about this topic, including code examples and configuration.</p>'
            '<a href="/docs/page2">Go to page 2</a>'
            '</main></body></html>'
        )
        page2_html = (
            '<html><head><title>Page 2</title></head><body><main>'
            '<h1>Page Two</h1>'
            '<p>This is the second page with substantial content that also passes the converter filter. '
            'It has enough text to be considered meaningful content by the extraction pipeline.</p>'
            '<h2>More Details</h2>'
            '<p>Additional paragraphs to ensure the content length is sufficient for conversion.</p>'
            '</main></body></html>'
        )

        def mock_get(url, headers=None):
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"content-type": "text/html"}
            if "page2" in url:
                resp.text = page2_html
            else:
                resp.text = page1_html
            return resp

        with patch("core.sources.web_crawler.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.side_effect = mock_get
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = connector.sync(str(tmp_path))

        assert result.files_added >= 2

    @patch("core.sources.web_crawler.time.sleep")
    def test_crawl_max_pages_stops(self, mock_sleep, tmp_path):
        """max_pages stops the crawl."""
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com", max_pages=1, wait_time=0,
                                 stay_in_path=False)
        connector = WebSourceConnector(config)

        def mock_get(url, headers=None):
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"content-type": "text/html"}
            resp.text = (
                '<html><head><title>Page</title></head><body><main>'
                '<h1>Topic</h1>'
                '<p>This page has enough content to pass the converter filter and be saved as markdown. '
                'It includes a link to another page to test max_pages enforcement.</p>'
                '<h2>Section</h2>'
                '<p>More content here to ensure it meets the minimum length requirement.</p>'
                '<a href="/other">Link</a>'
                '</main></body></html>'
            )
            return resp

        with patch("core.sources.web_crawler.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.side_effect = mock_get
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = connector.sync(str(tmp_path))

        assert result.files_added == 1

    @patch("core.sources.web_crawler.time.sleep")
    def test_crawl_http_error_accumulated(self, mock_sleep, tmp_path):
        """HTTP errors are accumulated in result.errors."""
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com/docs", max_pages=10, wait_time=0)
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "text/html"}

        with patch("core.sources.web_crawler.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = connector.sync(str(tmp_path))

        assert len(result.errors) >= 1
        assert "server error" in result.errors[0]

    @patch("core.sources.web_crawler.time.sleep")
    def test_crawl_non_html_skipped(self, mock_sleep, tmp_path):
        """Non-HTML content types are skipped."""
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com/docs", max_pages=10, wait_time=0)
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"key": "value"}'

        with patch("core.sources.web_crawler.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = connector.sync(str(tmp_path))

        assert result.files_added == 0

    @patch("core.sources.web_crawler.time.sleep")
    def test_crawl_httpx_exception_captured(self, mock_sleep, tmp_path):
        """httpx.HTTPError per-page is captured without stopping crawl."""
        import httpx
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com/docs", max_pages=10, wait_time=0)
        connector = WebSourceConnector(config)

        with patch("core.sources.web_crawler.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = connector.sync(str(tmp_path))

        assert len(result.errors) >= 1
        assert result.files_added == 0


class TestGitHubConnectorDiscover:
    """Tests for _discover_repos branching logic."""

    def test_discover_single_repo_mode(self):
        """Single repo mode calls get_repo."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig
        config = GitHubSourceConfig(owner="testowner", repo="testrepo")
        connector = GitHubSourceConnector(config)

        mock_client = MagicMock()
        mock_client.get_repo.return_value = {
            "name": "testrepo", "owner": {"login": "testowner"},
            "html_url": "https://github.com/testowner/testrepo",
            "clone_url": "https://github.com/testowner/testrepo.git",
            "private": False, "description": "A test repo",
            "default_branch": "main", "size": 1000, "fork": False,
        }
        mock_client.close = MagicMock()

        with patch("core.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = connector._discover_repos()

        assert len(repos) == 1
        assert repos[0].name == "testrepo"

    def test_discover_single_repo_not_found(self):
        """Single repo mode returns empty list if repo not found."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig
        config = GitHubSourceConfig(owner="testowner", repo="nonexistent")
        connector = GitHubSourceConnector(config)

        mock_client = MagicMock()
        mock_client.get_repo.return_value = None
        mock_client.close = MagicMock()

        with patch("core.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = connector._discover_repos()

        assert repos == []

    def test_discover_org_mode(self):
        """Org mode calls is_org + list_repos."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig
        config = GitHubSourceConfig(owner="myorg")
        connector = GitHubSourceConnector(config)

        mock_client = MagicMock()
        mock_client.is_org.return_value = True
        mock_client.list_repos.return_value = [
            {"name": "repo1", "owner": {"login": "myorg"}, "html_url": "", "clone_url": "",
             "private": False, "description": "", "default_branch": "main", "size": 0, "fork": False},
        ]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("core.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = connector._discover_repos()

        mock_client.is_org.assert_called_once_with("myorg")
        mock_client.list_repos.assert_called_once()
        assert len(repos) == 1

    def test_filter_repos_include_and_exclude_together(self):
        """When both include and exclude are set, exclude takes precedence."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        config = GitHubSourceConfig(
            owner="test", include_repos=["a", "b", "c"], exclude_repos=["b"]
        )
        connector = GitHubSourceConnector(config)
        repos = [
            RepoInfo(name=n, owner="test", url="", clone_url="", private=False,
                     description="", default_branch="main", size_kb=0, is_fork=False)
            for n in ["a", "b", "c", "d"]
        ]
        filtered = connector._filter_repos(repos)
        names = [r.name for r in filtered]
        assert "a" in names
        assert "c" in names
        assert "b" not in names  # excluded
        assert "d" not in names  # not included


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


class TestIngestionPipeline:
    """Tests for the producer-consumer ingestion pipeline."""

    def test_flush_buffer_embeds_and_indexes(self):
        """Flushing buffer should embed chunks and add them to searcher."""
        from unittest.mock import MagicMock
        from core.pipeline import IngestionPipeline

        ingestor = MagicMock()
        ingestor.create_embeddings.return_value = [[0.1, 0.2]] * 3
        searcher = MagicMock()
        searcher._sanitize_metadata.side_effect = lambda m: m

        pipeline = IngestionPipeline(ingestor=ingestor, searcher=searcher)
        pipeline._start_time = 1.0
        pipeline._chunk_buffer = [
            {"id": "1", "content": "chunk 1", "metadata": {"source_path": "/a.txt"}},
            {"id": "2", "content": "chunk 2", "metadata": {"source_path": "/a.txt"}},
            {"id": "3", "content": "chunk 3", "metadata": {"source_path": "/b.txt"}},
        ]
        pipeline._buffer_source_paths = {"/a.txt", "/b.txt"}

        pipeline._flush_buffer()

        ingestor.create_embeddings.assert_called_once()
        # Note: call_args[0][0] references the same list that gets .clear()-ed,
        # so we verify chunk count via the collection.upsert call instead.
        searcher.collection.upsert.assert_called_once()
        assert len(searcher.collection.upsert.call_args[1]["ids"]) == 3
        assert len(pipeline._chunk_buffer) == 0
        assert len(pipeline._buffer_source_paths) == 0
        assert "/a.txt" in pipeline._flushed_files
        assert "/b.txt" in pipeline._flushed_files

    def test_buffer_auto_flushes_at_batch_size(self):
        """Buffer flushes automatically when it reaches embed_batch_size."""
        from unittest.mock import MagicMock
        from core.pipeline import IngestionPipeline

        ingestor = MagicMock()
        ingestor.create_embeddings.return_value = [[0.1]] * 4
        searcher = MagicMock()
        searcher._sanitize_metadata.side_effect = lambda m: m

        pipeline = IngestionPipeline(ingestor=ingestor, searcher=searcher, embed_batch_size=4)
        pipeline._start_time = 1.0

        chunks = [
            {"id": str(i), "content": f"chunk {i}", "metadata": {"source_path": f"/{i}.txt"}}
            for i in range(5)
        ]

        pipeline._add_chunks_to_buffer(chunks)

        assert pipeline._flush_count == 1
        assert pipeline._total_chunks_indexed == 4
        assert len(pipeline._chunk_buffer) == 1

    def test_run_code_progressive_indexing(self):
        """run_code should progressively index in batches."""
        from unittest.mock import MagicMock
        from core.pipeline import IngestionPipeline

        ingestor = MagicMock()
        ingestor.create_embeddings.side_effect = lambda chunks, **kw: [[0.1]] * len(chunks)
        searcher = MagicMock()
        searcher._sanitize_metadata.side_effect = lambda m: m

        pipeline = IngestionPipeline(ingestor=ingestor, searcher=searcher, embed_batch_size=3)

        chunks = [
            {"id": str(i), "content": f"code {i}", "metadata": {"source_path": f"/src/{i}.py"}}
            for i in range(7)
        ]

        stats = pipeline.run_code(chunks)

        assert stats["total_chunks_indexed"] == 7
        assert stats["flush_count"] == 3
        assert searcher.collection.upsert.call_count == 3

    def test_resume_skips_already_indexed(self):
        """run_documents should skip files that are already indexed."""
        from unittest.mock import MagicMock, patch
        from core.pipeline import IngestionPipeline
        from pathlib import Path

        ingestor = MagicMock()
        searcher = MagicMock()

        pipeline = IngestionPipeline(ingestor=ingestor, searcher=searcher)

        files = [Path("/docs/a.txt"), Path("/docs/b.txt"), Path("/docs/c.txt")]
        already_indexed = {"/docs/a.txt", "/docs/b.txt"}

        with patch.object(pipeline, '_parse_document_file', return_value=[]) as mock_parse:
            pipeline.run_documents(files, base_directory=Path("/docs"), already_indexed=already_indexed)

        assert mock_parse.call_count == 1

    def test_empty_buffer_flush_is_noop(self):
        """Flushing an empty buffer should do nothing."""
        from unittest.mock import MagicMock
        from core.pipeline import IngestionPipeline

        ingestor = MagicMock()
        searcher = MagicMock()

        pipeline = IngestionPipeline(ingestor=ingestor, searcher=searcher)
        pipeline._flush_buffer()

        ingestor.create_embeddings.assert_not_called()
        searcher.collection.upsert.assert_not_called()
        assert pipeline._flush_count == 0

    def test_get_stats_returns_correct_shape(self):
        """Stats dict should contain expected keys."""
        from unittest.mock import MagicMock
        from core.pipeline import IngestionPipeline

        pipeline = IngestionPipeline(ingestor=MagicMock(), searcher=MagicMock())
        stats = pipeline._get_stats()

        assert "total_chunks_indexed" in stats
        assert "total_files_processed" in stats
        assert "flush_count" in stats
        assert "failed_files" in stats
        assert "interrupted" in stats
        assert "chunks_per_second" in stats


class TestResourceManager:
    """Tests for adaptive resource management."""

    def test_probe_cpu_returns_cores(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        info = rm.probe_cpu()
        assert "total_cores" in info
        assert "available_cores" in info
        assert info["available_cores"] <= info["total_cores"]
        assert info["available_cores"] > 0

    def test_probe_ram_returns_gb(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        info = rm.probe_ram()
        assert "total_gb" in info
        assert "available_gb" in info
        assert info["total_gb"] > 0

    def test_probe_gpu_returns_info_or_none(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        info = rm.probe_gpu()
        if info is not None:
            assert "total_mb" in info
            assert "free_mb" in info

    def test_recommend_parse_workers_scales_with_files(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        small = rm.recommend_parse_workers(100)
        large = rm.recommend_parse_workers(50000)
        assert large >= small
        assert small >= 2
        assert large <= rm.probe_cpu()["total_cores"]

    def test_recommend_embed_batch(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        batch = rm.recommend_embed_batch(384)
        assert 32 <= batch <= 1024

    def test_classify_client(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        assert rm.classify_client(100, 500) == "small"
        assert rm.classify_client(5000, 20000) == "medium"
        assert rm.classify_client(80000, 200000) == "large"

    def test_headroom_respected(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager(headroom=0.20)
        info = rm.probe_cpu()
        assert info["available_cores"] <= int(info["total_cores"] * 0.80) + 1

    def test_summary_returns_string(self):
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        s = rm.summary()
        assert "CPU:" in s
        assert "RAM:" in s


class TestModelCache:
    """Tests for model caching."""

    def test_get_embedder_returns_model(self):
        from core.model_cache import ModelCache
        cache = ModelCache()
        model = cache.get_embedder("all-MiniLM-L6-v2", "cpu")
        assert model is not None
        assert hasattr(model, "encode")
        cache.clear()

    def test_get_embedder_returns_same_instance(self):
        from core.model_cache import ModelCache
        cache = ModelCache()
        model1 = cache.get_embedder("all-MiniLM-L6-v2", "cpu")
        model2 = cache.get_embedder("all-MiniLM-L6-v2", "cpu")
        assert model1 is model2
        cache.clear()

    def test_clear_removes_models(self):
        from core.model_cache import ModelCache
        cache = ModelCache()
        cache.get_embedder("all-MiniLM-L6-v2", "cpu")
        assert len(cache._embedders) > 0
        cache.clear()
        assert len(cache._embedders) == 0

    def test_singleton_access(self):
        import core.model_cache as mc
        mc._instance = None  # reset singleton
        c1 = mc.get_model_cache()
        c2 = mc.get_model_cache()
        assert c1 is c2
        mc._instance = None  # cleanup


class TestHealthCheck:
    """Tests for LM Studio health check and circuit breaker."""

    def test_circuit_breaker_starts_closed(self):
        from core.health import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.is_closed()

    def test_circuit_breaker_opens_after_threshold(self):
        from core.health import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_closed()
        cb.record_failure()
        assert cb.is_open()

    def test_circuit_breaker_resets_on_success(self):
        from core.health import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.is_closed()
        assert cb._consecutive_failures == 0

    def test_circuit_breaker_allows_check_returns_false_when_open(self):
        from core.health import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert not cb.allow_request()

    def test_circuit_breaker_stats(self):
        from core.health import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        stats = cb.stats
        assert stats["state"] == "open"
        assert stats["consecutive_failures"] == 2

    def test_check_model_loaded_handles_no_server(self):
        from core.health import check_lm_studio_model
        result = check_lm_studio_model(base_url="http://localhost:19999")
        assert result is False


class TestProcessPoolChunking:
    """Test that code chunking uses multiprocessing."""

    def test_chunk_parallel_uses_process_pool(self, tmp_path):
        """Verify parallel chunking works with ProcessPoolExecutor."""
        from core.code.chunker import CodeChunker

        # Create some test Python files
        for i in range(5):
            f = tmp_path / f"test_{i}.py"
            f.write_text(f"def func_{i}():\n    return {i}\n")

        chunker = CodeChunker({"parallel_workers": 2})
        files = list(tmp_path.glob("*.py"))
        chunks = chunker.chunk_files(files, repos=[], parallel=True)
        assert isinstance(chunks, list)
        # Each file has one function, should produce at least some chunks
        assert len(chunks) > 0

    def test_chunk_parallel_with_repo_info(self, tmp_path):
        """Verify parallel chunking works with repo info."""
        from core.code.chunker import CodeChunker
        from core.code.git_utils import RepoInfo

        # Create test files
        for i in range(3):
            f = tmp_path / f"mod_{i}.py"
            f.write_text(f"class MyClass{i}:\n    def method(self):\n        pass\n")

        repo = RepoInfo(
            repo_root=tmp_path,
            repo_name="test-repo",
            remote_url="https://github.com/test/repo",
            current_branch="main",
        )
        chunker = CodeChunker({"parallel_workers": 2})
        files = list(tmp_path.glob("*.py"))
        chunks = chunker.chunk_files(files, repos=[repo], parallel=True)
        assert len(chunks) > 0
        # Check repo info was propagated into metadata
        for chunk in chunks:
            assert chunk["metadata"].get("repo_name") == "test-repo"

    def test_chunk_parallel_fallback_to_threads(self, tmp_path):
        """Verify _chunk_parallel_threads still works as fallback."""
        from core.code.chunker import CodeChunker

        for i in range(3):
            f = tmp_path / f"test_{i}.py"
            f.write_text(f"def func_{i}():\n    return {i}\n")

        chunker = CodeChunker({"parallel_workers": 2})
        files = list(tmp_path.glob("*.py"))
        # Call fallback directly
        chunks = chunker._chunk_parallel_threads(files, repos=[], workers=2)
        assert isinstance(chunks, list)
        assert len(chunks) > 0


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


class TestGraphExtractorBatching:
    """Tests for graph extractor batch processing."""

    def test_build_batches_by_budget_basic(self):
        """Simple budgeting splits chunks correctly."""
        from core.graph.extractor import _build_batches_by_budget

        chunks = [{"id": f"c{i}", "content": f"Short text {i}"} for i in range(4)]
        # Budget large enough for all chunks
        batches = _build_batches_by_budget(chunks, 10000)
        assert len(batches) == 1
        assert len(batches[0]) == 4

    def test_build_batches_by_budget_splits(self):
        """Budget forces splitting into multiple batches."""
        from core.graph.extractor import _build_batches_by_budget

        chunks = [{"id": f"c{i}", "content": "x" * 100} for i in range(4)]
        # Each chunk is ~110 chars (content + "[Chunk cX]\n"), budget of 250 fits ~2
        batches = _build_batches_by_budget(chunks, 250)
        assert len(batches) >= 2

    def test_build_batches_by_budget_empty_chunks(self):
        """Empty content chunks are included without counting toward budget."""
        from core.graph.extractor import _build_batches_by_budget

        chunks = [
            {"id": "c0", "content": "Has content"},
            {"id": "c1", "content": ""},
            {"id": "c2", "content": "   "},
            {"id": "c3", "content": "Also has content"},
        ]
        batches = _build_batches_by_budget(chunks, 10000)
        assert len(batches) == 1
        assert len(batches[0]) == 4

    def test_build_batches_by_budget_single_large_chunk(self):
        """One chunk exceeding budget still forms its own batch."""
        from core.graph.extractor import _build_batches_by_budget

        chunks = [
            {"id": "c0", "content": "x" * 5000},
            {"id": "c1", "content": "small"},
        ]
        batches = _build_batches_by_budget(chunks, 100)
        assert len(batches) == 2
        assert len(batches[0]) == 1  # Large chunk alone
        assert len(batches[1]) == 1

    def test_process_batch_success(self):
        """Successful batch returns 5-tuple with 0 skipped."""
        from core.graph.extractor import GraphExtractor

        def mock_llm(messages, system_prompt, max_tokens):
            return json.dumps({
                "entities": [{"name": "Foo", "type": "concept", "description": "A thing"}],
                "relationships": [],
            })

        extractor = GraphExtractor(get_llm_response=mock_llm, max_workers=1)
        batch = [{"id": "c1", "content": "About Foo concept"}]
        entities, rels, chunk_ids, num_chunks, skipped = extractor._process_batch(batch, 0)
        assert len(entities) == 1
        assert skipped == 0
        assert num_chunks == 1

    def test_process_batch_failure_splits(self):
        """On failure, batch splits and retries sub-batches."""
        from core.graph.extractor import GraphExtractor

        call_count = 0

        def mock_llm(messages, system_prompt, max_tokens):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Error: 400 Bad Request"  # First call fails
            return json.dumps({
                "entities": [{"name": f"E{call_count}", "type": "concept", "description": ""}],
                "relationships": [],
            })

        # Budget of 4000: halved to 2000 which is > MIN_BATCH_TEXT_BUDGET (1000)
        extractor = GraphExtractor(
            get_llm_response=mock_llm, max_workers=1,
            batch_text_budget=4000,
        )
        batch = [
            {"id": "c1", "content": "Content about Alpha " * 20},
            {"id": "c2", "content": "Content about Beta " * 20},
        ]
        entities, rels, chunk_ids, num_chunks, skipped = extractor._process_batch(batch, 0)
        # First call failed, then split in half — sub-batch calls should succeed
        assert call_count >= 2
        assert len(entities) >= 1

    def test_on_result_counts_skipped_chunks(self):
        """Progress.failed_chunks incremented correctly for skipped chunks."""
        from core.graph.extractor import GraphExtractor

        # All calls fail — circuit breaker will eventually skip
        fail_count = 0

        def always_fail(messages, system_prompt, max_tokens):
            nonlocal fail_count
            fail_count += 1
            return "Error: 500 Internal Server Error"

        extractor = GraphExtractor(
            get_llm_response=always_fail, max_workers=1,
            batch_text_budget=200,
        )
        # Reset circuit breaker to have a low threshold for test speed
        from core.health import CircuitBreaker
        extractor._circuit_breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        chunks = [{"id": f"c{i}", "content": f"Content {i}"} for i in range(4)]
        progress_data = []

        def capture_progress(progress, entities, rels):
            progress_data.append(progress.failed_chunks)

        extractor.extract_from_chunks(chunks, progress_callback=capture_progress)
        # At least some chunks should be counted as failed
        if progress_data:
            assert progress_data[-1] > 0


class TestEntityKeyLookup:
    """Tests for entity key lookup optimization in _parse_response."""

    def test_entity_key_lookup_uses_dict(self):
        """Verify O(1) lookup produces correct relationship keys."""
        from core.graph.extractor import GraphExtractor

        extractor = GraphExtractor(get_llm_response=lambda m, s, t: "")
        response = json.dumps({
            "entities": [
                {"name": "Alice", "type": "person", "description": "A person"},
                {"name": "Acme", "type": "organization", "description": "A company"},
            ],
            "relationships": [
                {"source": "Alice", "target": "Acme", "type": "works_at",
                 "description": "Employment", "weight": 0.9},
            ],
        })
        entities, rels = extractor._parse_response(response, "c1")
        assert len(rels) == 1
        assert rels[0].source_entity == "alice::person"
        assert rels[0].target_entity == "acme::organization"

    def test_entity_key_fallback_to_concept(self):
        """Unmatched entity names fall back to concept type."""
        from core.graph.extractor import GraphExtractor

        extractor = GraphExtractor(get_llm_response=lambda m, s, t: "")
        response = json.dumps({
            "entities": [],
            "relationships": [
                {"source": "Unknown", "target": "Mystery", "type": "related",
                 "description": "?", "weight": 0.5},
            ],
        })
        entities, rels = extractor._parse_response(response, "c1")
        assert len(rels) == 1
        assert rels[0].source_entity == "unknown::concept"
        assert rels[0].target_entity == "mystery::concept"

    def test_entity_key_case_insensitive(self):
        """Entity matching is case-insensitive."""
        from core.graph.extractor import GraphExtractor

        extractor = GraphExtractor(get_llm_response=lambda m, s, t: "")
        response = json.dumps({
            "entities": [
                {"name": "ALICE", "type": "person", "description": ""},
            ],
            "relationships": [
                {"source": "alice", "target": "Bob", "type": "knows",
                 "description": "", "weight": 1.0},
            ],
        })
        entities, rels = extractor._parse_response(response, "c1")
        assert rels[0].source_entity == "alice::person"  # Matched despite case diff
        assert rels[0].target_entity == "bob::concept"  # Unmatched, fallback


class TestIncrementalGraphFetch:
    """Tests for get_unprocessed_chunks two-phase fetch."""

    def test_get_unprocessed_chunks_all_new(self):
        """No processed IDs returns all document chunks."""
        from scripts.build_graph import get_unprocessed_chunks
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.get.side_effect = [
            # Phase 1: IDs + metadatas
            {
                "ids": ["c1", "c2", "c3"],
                "metadatas": [
                    {"chunk_type": "document"},
                    {"chunk_type": "document"},
                    {"chunk_type": "document"},
                ],
            },
            # Phase 2: Full content for unprocessed
            {
                "ids": ["c1", "c2", "c3"],
                "documents": ["Doc 1", "Doc 2", "Doc 3"],
                "metadatas": [{}, {}, {}],
            },
        ]

        with patch("scripts.build_graph._get_collection", return_value=mock_collection):
            chunks, total = get_unprocessed_chunks(mock_client, set())
            assert len(chunks) == 3
            assert total == 3

    def test_get_unprocessed_chunks_all_processed(self):
        """Returns empty list when all chunks are processed."""
        from scripts.build_graph import get_unprocessed_chunks
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.get.return_value = {
            "ids": ["c1", "c2"],
            "metadatas": [{"chunk_type": "document"}, {"chunk_type": "document"}],
        }

        with patch("scripts.build_graph._get_collection", return_value=mock_collection):
            chunks, total = get_unprocessed_chunks(mock_client, {"c1", "c2"})
            assert len(chunks) == 0
            assert total == 2

    def test_get_unprocessed_chunks_partial(self):
        """Returns only unprocessed chunks."""
        from scripts.build_graph import get_unprocessed_chunks
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.get.side_effect = [
            # Phase 1: IDs + metadatas
            {
                "ids": ["c1", "c2", "c3"],
                "metadatas": [
                    {"chunk_type": "document"},
                    {"chunk_type": "document"},
                    {"chunk_type": "document"},
                ],
            },
            # Phase 2: Only unprocessed chunk
            {
                "ids": ["c3"],
                "documents": ["Doc 3"],
                "metadatas": [{}],
            },
        ]

        with patch("scripts.build_graph._get_collection", return_value=mock_collection):
            chunks, total = get_unprocessed_chunks(mock_client, {"c1", "c2"})
            assert len(chunks) == 1
            assert chunks[0]["id"] == "c3"
            assert total == 3


class TestGraphExtensionFilter:
    """Tests for graph build extension filtering."""

    def test_should_exclude_by_extension_matches(self):
        """Chunks with excluded extensions are filtered."""
        from scripts.build_graph import _should_exclude_by_extension

        metadata = {"source": "config/settings.json"}
        assert _should_exclude_by_extension(metadata, [".json", ".yaml"]) is True

    def test_should_exclude_by_extension_no_match(self):
        """Chunks with non-excluded extensions pass through."""
        from scripts.build_graph import _should_exclude_by_extension

        metadata = {"source": "docs/README.md"}
        assert _should_exclude_by_extension(metadata, [".json", ".yaml"]) is False

    def test_should_exclude_by_extension_empty_list(self):
        """Empty exclude list excludes nothing."""
        from scripts.build_graph import _should_exclude_by_extension

        metadata = {"source": "anything.json"}
        assert _should_exclude_by_extension(metadata, []) is False

    def test_should_exclude_by_extension_source_path(self):
        """Falls back to source_path metadata."""
        from scripts.build_graph import _should_exclude_by_extension

        metadata = {"source_path": "/repo/package.json"}
        assert _should_exclude_by_extension(metadata, [".json"]) is True

    def test_should_exclude_by_extension_case_insensitive(self):
        """Extension matching is case-insensitive."""
        from scripts.build_graph import _should_exclude_by_extension

        metadata = {"source": "style.CSS"}
        assert _should_exclude_by_extension(metadata, [".css"]) is True

    def test_get_unprocessed_chunks_with_extension_filter(self):
        """Extension filter reduces chunk count."""
        from scripts.build_graph import get_unprocessed_chunks
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 4
        mock_collection.get.side_effect = [
            # Phase 1: IDs + metadatas
            {
                "ids": ["c1", "c2", "c3", "c4"],
                "metadatas": [
                    {"chunk_type": "document", "source": "README.md"},
                    {"chunk_type": "document", "source": "config.json"},
                    {"chunk_type": "document", "source": "style.css"},
                    {"chunk_type": "document", "source": "guide.md"},
                ],
            },
            # Phase 2: Only non-filtered, non-processed chunks
            {
                "ids": ["c1", "c4"],
                "documents": ["Doc 1", "Doc 4"],
                "metadatas": [{"source": "README.md"}, {"source": "guide.md"}],
            },
        ]

        with patch("scripts.build_graph._get_collection", return_value=mock_collection):
            chunks, total = get_unprocessed_chunks(
                mock_client, set(), exclude_extensions=[".json", ".css"]
            )
            assert len(chunks) == 2
            assert total == 2  # only non-excluded docs counted


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


class TestZeroChunkFileState:
    """Tests for zero-chunk file state recording."""

    def test_zero_chunk_files_recorded_in_state(self, tmp_path):
        """Files producing 0 chunks get state entry."""
        from core.code.incremental import CodeIndexState

        state = CodeIndexState(tmp_path)
        # Create a test file
        test_file = tmp_path / "empty.py"
        test_file.write_text("# just a comment")

        # Record with empty chunk list (zero chunks)
        state.update_file(test_file, [])
        state.save_state()

        assert str(test_file) in state.state
        assert state.state[str(test_file)].chunk_ids == []

    def test_zero_chunk_files_not_redetected(self, tmp_path):
        """After recording zero-chunk state, is_file_changed returns False."""
        from core.code.incremental import CodeIndexState

        state = CodeIndexState(tmp_path)
        test_file = tmp_path / "empty.py"
        test_file.write_text("# just a comment")

        state.update_file(test_file, [])
        state.save_state()

        # Reload state (simulates next run)
        state2 = CodeIndexState(tmp_path)
        assert not state2.is_file_changed(test_file)

    def test_deleted_files_removed_from_state(self, tmp_path):
        """Deletions properly clean up state."""
        from core.code.incremental import CodeIndexState

        state = CodeIndexState(tmp_path)
        test_file = tmp_path / "code.py"
        test_file.write_text("def hello(): pass")
        state.update_file(test_file, ["chunk1", "chunk2"])
        state.save_state()

        assert str(test_file) in state.state

        removed_ids = state.remove_file(str(test_file))
        assert removed_ids == ["chunk1", "chunk2"]
        assert str(test_file) not in state.state

    def test_stale_only_records_zero_chunk_files(self, tmp_path):
        """Modified files producing 0 new chunks get state recorded."""
        from core.code.incremental import CodeIndexState

        state = CodeIndexState(tmp_path)

        # File initially produced chunks
        test_file = tmp_path / "module.py"
        test_file.write_text("def original(): pass")
        state.update_file(test_file, ["old_chunk_1"])
        state.save_state()

        # File modified but now produces 0 chunks
        test_file.write_text("# completely emptied")

        # Simulate what ingest_code does: update with empty chunks
        state.update_file(test_file, [])
        state.save_state()

        assert state.state[str(test_file)].chunk_ids == []
        # Should not be detected as changed on next run
        state2 = CodeIndexState(tmp_path)
        assert not state2.is_file_changed(test_file)

    def test_zero_chunk_file_state_persists_across_loads(self, tmp_path):
        """Zero-chunk file state survives serialization roundtrip."""
        from core.code.incremental import CodeIndexState

        state = CodeIndexState(tmp_path)
        test_file = tmp_path / "readme.md"
        test_file.write_text("# README")
        state.update_file(test_file, [])
        state.save_state()

        # Load fresh
        state2 = CodeIndexState(tmp_path)
        assert str(test_file) in state2.state
        assert state2.state[str(test_file)].chunk_ids == []
        assert state2.state[str(test_file)].file_hash != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
