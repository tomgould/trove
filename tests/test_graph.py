"""
Knowledge graph models, extraction, search and export.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile


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
