"""
Evaluation datasets and retrieval and RAG metrics.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import tempfile


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
