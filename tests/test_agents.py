"""
Agent templates and the local agent runner.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock
import tempfile
import shutil


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
