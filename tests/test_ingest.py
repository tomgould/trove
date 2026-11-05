"""
Document ingestion, chunking and ingestion jobs.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile


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
