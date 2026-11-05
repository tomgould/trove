"""
Memory store and workflow engine.
"""

import pytest
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock
import tempfile
import shutil


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
