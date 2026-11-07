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

        # Create test client directories (a client is a folder with a client.yaml)
        for name in ("client_a", "client_b", ".hidden"):
            (clients_dir / name).mkdir()
            (clients_dir / name / "client.yaml").write_text(f"name: {name}\n")

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

        for name in ("alpha", "beta", ".hidden"):  # .hidden should be excluded
            (tmp_path / name).mkdir()
            (tmp_path / name / "client.yaml").write_text(f"name: {name}\n")

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


class TestUnknownClient:
    """Regression: a mistyped client name used to create folders and fall back to defaults."""

    def test_unknown_client_raises_clear_error(self, clients_dir, make_client):
        from core.client import ClientNotFoundError, load_client

        make_client("alpha")
        with pytest.raises(ClientNotFoundError) as excinfo:
            load_client("alhpa", clients_dir=clients_dir)
        message = str(excinfo.value)
        assert "alhpa" in message
        assert "alpha" in message  # lists the clients that do exist

    def test_unknown_client_creates_nothing(self, clients_dir):
        from core.client import ClientNotFoundError, load_client

        with pytest.raises(ClientNotFoundError):
            load_client("missing", clients_dir=clients_dir)
        assert list(clients_dir.iterdir()) == []

    def test_unknown_client_is_a_file_not_found_error(self, clients_dir):
        """Callers that already catch FileNotFoundError keep working."""
        from core.client import load_client

        with pytest.raises(FileNotFoundError):
            load_client("missing", clients_dir=clients_dir)

    def test_empty_name_is_rejected(self, clients_dir):
        from core.client import ClientNotFoundError, load_client

        with pytest.raises(ClientNotFoundError):
            load_client("", clients_dir=clients_dir)

    def test_get_active_client_with_unknown_env_client(self, clients_dir, monkeypatch):
        from core.client import ClientNotFoundError, get_active_client

        monkeypatch.setenv("RAG_CLIENT", "nope")
        with pytest.raises(ClientNotFoundError):
            get_active_client(clients_dir=clients_dir)
        assert not (clients_dir / "nope").exists()

    def test_known_client_still_gets_its_folders(self, clients_dir, make_client):
        from core.client import load_client

        client = load_client(make_client("alpha"), clients_dir=clients_dir)
        assert client.documents_path.is_dir()
        assert client.index_path.is_dir()


class TestHasIndex:
    """Regression: the "no index" checks used index_path.exists(), which load_client makes true."""

    def test_new_client_has_no_index(self, clients_dir, make_client):
        from core.client import load_client

        client = load_client(make_client("alpha"), clients_dir=clients_dir)
        assert client.index_path.exists()  # created by load_client
        assert client.has_index is False

    @staticmethod
    def _open_collection(client):
        import chromadb
        from chromadb.config import Settings

        db = chromadb.PersistentClient(path=str(client.index_path), settings=Settings(anonymized_telemetry=False))
        return db.get_or_create_collection(client.machine_name)

    def test_empty_database_is_not_an_index(self, clients_dir, make_client):
        """Regression: a first ingest that failed left chroma.sqlite3 behind and the client showed [ready]."""
        from core.client import INDEX_MARKER_FILE, load_client

        client = load_client(make_client("alpha"), clients_dir=clients_dir)
        self._open_collection(client)
        assert (client.index_path / INDEX_MARKER_FILE).is_file()
        assert client.has_index is False

    def test_indexed_chunk_means_indexed(self, clients_dir, make_client):
        from core.client import load_client

        client = load_client(make_client("alpha"), clients_dir=clients_dir)
        self._open_collection(client).add(ids=["c1"], embeddings=[[0.1, 0.2, 0.3]], documents=["hello"])
        assert client.has_index is True

    def test_unreadable_database_counts_as_index(self, clients_dir, make_client):
        """If the file cannot be read as expected, an existing file still counts."""
        from core.client import INDEX_MARKER_FILE, load_client

        client = load_client(make_client("alpha"), clients_dir=clients_dir)
        (client.index_path / INDEX_MARKER_FILE).write_bytes(b"")
        assert client.has_index is True

    def test_relative_index_path(self, clients_dir, make_client, monkeypatch):
        """Regression: a relative path made as_uri() raise, so a client with an index looked broken."""
        from core.client import Client, load_client

        client = load_client(make_client("alpha"), clients_dir=clients_dir)
        self._open_collection(client).add(ids=["c1"], embeddings=[[0.1, 0.2, 0.3]], documents=["hello"])
        monkeypatch.chdir(clients_dir.parent)
        relative = Client(Path("clients") / "alpha", client.config)
        assert not relative.index_path.is_absolute()
        assert relative.has_index is True


class TestClientsDir:
    """TROVE_CLIENTS_DIR and --clients-dir: one place decides where clients live."""

    @pytest.fixture(autouse=True)
    def _reset_override(self):
        from core.client import set_clients_dir

        set_clients_dir(None)
        yield
        set_clients_dir(None)

    def test_default_is_the_repo_clients_folder(self, monkeypatch):
        from core.client import CLIENTS_DIR_ENV, resolve_clients_dir
        from core.paths import default_clients_dir

        monkeypatch.delenv(CLIENTS_DIR_ENV, raising=False)
        assert resolve_clients_dir() == default_clients_dir()

    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        from core.client import CLIENTS_DIR_ENV, resolve_clients_dir

        monkeypatch.setenv(CLIENTS_DIR_ENV, str(tmp_path))
        assert resolve_clients_dir() == tmp_path

    def test_flag_overrides_env_var(self, tmp_path, monkeypatch):
        from core.client import CLIENTS_DIR_ENV, resolve_clients_dir, set_clients_dir

        monkeypatch.setenv(CLIENTS_DIR_ENV, str(tmp_path / "env"))
        set_clients_dir(tmp_path / "flag")
        assert resolve_clients_dir() == tmp_path / "flag"

    def test_explicit_argument_wins(self, tmp_path):
        from core.client import resolve_clients_dir, set_clients_dir

        set_clients_dir(tmp_path / "flag")
        assert resolve_clients_dir(tmp_path / "arg") == tmp_path / "arg"

    def test_load_and_discover_use_env_var(self, tmp_path, monkeypatch):
        from core.client import CLIENTS_DIR_ENV, discover_clients, load_client

        (tmp_path / "proj").mkdir()
        (tmp_path / "proj" / "client.yaml").write_text("name: Proj\nmachine_name: proj\n")
        monkeypatch.setenv(CLIENTS_DIR_ENV, str(tmp_path))
        assert discover_clients() == ["proj"]
        assert load_client("proj").path == tmp_path / "proj"

    def test_discover_skips_underscore_and_folders_without_yaml(self, tmp_path):
        from core.client import discover_clients

        for name in ("real", "_template", ".git"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "client.yaml").write_text("name: x\n")
        (tmp_path / "notes").mkdir()  # no client.yaml
        assert discover_clients(tmp_path) == ["real"]

    def test_absolute_and_symlinked_documents_path(self, tmp_path):
        from core.client import load_client

        outside = tmp_path / "scraped"
        outside.mkdir()
        clients = tmp_path / "clients"
        (clients / "abs").mkdir(parents=True)
        (clients / "abs" / "client.yaml").write_text(f"name: Abs\npaths:\n  documents: {outside}\n")
        assert load_client("abs", clients_dir=clients).documents_path == outside

        (clients / "linked").mkdir()
        (clients / "linked" / "client.yaml").write_text("name: Linked\n")
        (clients / "linked" / "documents").symlink_to(outside, target_is_directory=True)
        client = load_client("linked", clients_dir=clients)
        assert client.documents_path.resolve() == outside.resolve()

    def test_clients_dir_argument_helpers(self, tmp_path):
        import argparse

        from core.client import add_clients_dir_argument, apply_clients_dir_argument, resolve_clients_dir

        parser = argparse.ArgumentParser()
        add_clients_dir_argument(parser)
        apply_clients_dir_argument(parser.parse_args(["--clients-dir", str(tmp_path)]))
        assert resolve_clients_dir() == tmp_path


class TestRelativeClientsDir:
    """Regression: a relative --clients-dir or TROVE_CLIENTS_DIR broke search once a client had an index."""

    @pytest.fixture(autouse=True)
    def _reset_override(self):
        from core.client import set_clients_dir

        set_clients_dir(None)
        yield
        set_clients_dir(None)

    @staticmethod
    def _run(module_name, argv, capsys, monkeypatch):
        import importlib
        import sys

        module = importlib.import_module(f"scripts.{module_name}")
        monkeypatch.setattr(sys, "argv", [module_name, *argv])
        module.main()
        return capsys.readouterr().out

    @pytest.mark.parametrize("given, workdir, clients", [
        ("projects", "work", "work/projects"),
        ("./projects", "work", "work/projects"),
        ("../x/projects", "work", "x/projects"),
        ("env:projects", "work", "work/projects"),
    ])
    def test_ingest_search_and_list(self, given, workdir, clients, tmp_path, capsys, monkeypatch):
        import json

        from core.client import CLIENTS_DIR_ENV, set_clients_dir

        client_dir = tmp_path / clients / "demo"
        (client_dir / "documents").mkdir(parents=True)
        (client_dir / "client.yaml").write_text("name: Demo\nmachine_name: demo\n")
        (client_dir / "documents" / "rivers.md").write_text(
            "# Rivers\n\nA river carves its valley and carries silt down to the estuary over many years.\n")
        (tmp_path / workdir).mkdir(exist_ok=True)
        monkeypatch.chdir(tmp_path / workdir)

        if given.startswith("env:"):
            monkeypatch.setenv(CLIENTS_DIR_ENV, given[4:])
            flag = []
        else:
            monkeypatch.delenv(CLIENTS_DIR_ENV, raising=False)
            flag = ["--clients-dir", given]

        summary = json.loads(self._run("ingest", ["--client", "demo", "--json", "--no-code", *flag], capsys, monkeypatch))
        assert summary["failed"] == 0 and summary["new"] == 1, summary
        set_clients_dir(None)

        out = json.loads(self._run("search", ["--client", "demo", "--json", *flag, "river valley silt"],
                                   capsys, monkeypatch))
        assert any(r["source"] == "rivers.md" for r in out["results"]), out
        set_clients_dir(None)

        listing = self._run("search", ["--list", *flag], capsys, monkeypatch)
        assert "demo" in listing and "error" not in listing, listing
        assert "ready" in listing, listing

    def test_resolved_folder_is_absolute(self, tmp_path, monkeypatch):
        from core.client import CLIENTS_DIR_ENV, resolve_clients_dir, set_clients_dir

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(CLIENTS_DIR_ENV, "env-clients")
        assert resolve_clients_dir() == tmp_path / "env-clients"
        set_clients_dir("../flag-clients")
        assert resolve_clients_dir() == tmp_path.parent / "flag-clients"
        assert resolve_clients_dir("./arg") == tmp_path / "arg"


class TestCreateClientNonInteractive:
    """create_client.py --name ... creates a client without prompts."""

    def test_created_client_loads(self, clients_dir):
        from core.client import discover_clients, load_client
        from scripts.create_client import main

        main(["--name", "news-site", "--description", "Scraped news", "--clients-dir", str(clients_dir)])
        client = load_client("news-site", clients_dir=clients_dir)
        assert client.name == "news-site"
        assert client.config["description"] == "Scraped news"
        assert client.documents_path == clients_dir / "news-site" / "documents"
        assert client.documents_path.is_dir()
        assert discover_clients(clients_dir) == ["news-site"]

    def test_documents_dir_and_embedding_model(self, clients_dir, tmp_path):
        from core.client import load_client
        from scripts.create_client import create_client_from_args

        docs = tmp_path / "scrape-output"
        docs.mkdir()
        create_client_from_args("proj", documents_dir=str(docs), embedding_model="BAAI/bge-base-en-v1.5",
                                clients_dir=clients_dir)
        client = load_client("proj", clients_dir=clients_dir)
        assert client.documents_path == docs.resolve()
        assert client.embedding_config["model"] == "BAAI/bge-base-en-v1.5"
        assert client.embedding_config["dimension"] == 768

    def test_refuses_to_overwrite_without_force(self, clients_dir, capsys):
        from scripts.create_client import main

        main(["--name", "proj", "--clients-dir", str(clients_dir)])
        with pytest.raises(SystemExit) as excinfo:
            main(["--name", "proj", "--description", "second", "--clients-dir", str(clients_dir)])
        assert excinfo.value.code == 1
        assert "already exists" in capsys.readouterr().err
        main(["--name", "proj", "--description", "second", "--force", "--clients-dir", str(clients_dir)])
        assert "second" in (clients_dir / "proj" / "client.yaml").read_text()

    @pytest.mark.parametrize("bad", ["ab", "_template", ".hidden", "has space", "trailing-", "a/b", "x" * 65])
    def test_rejects_bad_names(self, clients_dir, bad, capsys):
        from scripts.create_client import main

        with pytest.raises(SystemExit) as excinfo:
            main(["--name", bad, "--clients-dir", str(clients_dir)])
        assert excinfo.value.code == 1
        assert "Invalid client name" in capsys.readouterr().err
        assert list(clients_dir.iterdir()) == []

    def test_missing_documents_dir_is_rejected(self, clients_dir, tmp_path):
        from scripts.create_client import create_client_from_args

        with pytest.raises(ValueError, match="Documents folder not found"):
            create_client_from_args("proj", documents_dir=str(tmp_path / "nope"), clients_dir=clients_dir)

    def test_no_arguments_runs_the_wizard(self, monkeypatch):
        import scripts.create_client as cli

        called = []
        monkeypatch.setattr(cli, "create_client", lambda: called.append(True))
        cli.main([])
        assert called == [True]


class TestScriptsTakeClientsDir:
    """Every CLI script accepts --clients-dir and its no-argument load_client calls follow it."""

    SCRIPTS = ["build_graph", "run_agent", "search", "evaluate", "benchmark_embeddings",
               "watch_and_reindex", "sync_prompts", "sources", "ingest", "create_client"]

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_help_lists_clients_dir(self, script, monkeypatch, capsys):
        import importlib
        import sys

        module = importlib.import_module(f"scripts.{script}")
        entry = getattr(module, "_main", None) or module.main
        monkeypatch.setattr(sys, "argv", [script, "--help"])
        with pytest.raises(SystemExit) as excinfo:
            entry(["--help"]) if script == "create_client" else entry()
        assert excinfo.value.code == 0
        assert "--clients-dir" in capsys.readouterr().out

    @pytest.mark.parametrize("script", ["search", "ingest"])
    def test_list_uses_clients_dir(self, script, make_client, clients_dir, monkeypatch, capsys):
        import importlib
        import sys

        make_client("outside-repo")
        module = importlib.import_module(f"scripts.{script}")
        monkeypatch.setattr(sys, "argv", [script, "--clients-dir", str(clients_dir), "--list"])
        module.main()
        assert "outside-repo" in capsys.readouterr().out
