"""
LLM providers, prompt building, context budgets and citations.
"""

import pytest
import requests
from unittest.mock import MagicMock


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
