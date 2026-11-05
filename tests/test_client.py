"""
Client loading, discovery and configuration.
"""

import pytest
from pathlib import Path
import tempfile
import shutil


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
