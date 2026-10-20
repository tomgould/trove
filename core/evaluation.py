"""
Evaluation framework for Trove search quality.

Provides:
- Test dataset loading and validation
- Retrieval metrics (MRR, Hit@k, Precision@k, nDCG, Context Recall)
- RAG quality metrics via LLM-as-judge (Faithfulness, Answer Relevance, Context Precision)
- Result aggregation and reporting
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single evaluation test case."""
    id: str
    query: str
    expected_chunks: list[str]
    ground_truth: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""
    test_case: TestCase
    retrieved_chunks: list[dict]
    metrics: dict[str, float]
    answer: Optional[str] = None
    quality: Optional[object] = None  # SearchQuality


@dataclass
class EvalReport:
    """Full evaluation report."""
    client_name: str
    config_label: str
    results: list[EvalResult]
    aggregate: dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_eval_dataset(client) -> list[TestCase]:
    """
    Load eval_dataset.json from a client's tests/ directory.

    Args:
        client: Client instance with a .path attribute

    Returns:
        List of TestCase instances

    Raises:
        FileNotFoundError: If eval_dataset.json doesn't exist
    """
    path = client.path / "tests" / "eval_dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"No eval dataset found at {path}")

    with open(path) as f:
        data = json.load(f)

    test_cases = []
    for tc in data.get("test_cases", []):
        test_cases.append(TestCase(
            id=tc["id"],
            query=tc["query"],
            expected_chunks=tc.get("expected_chunks", []),
            ground_truth=tc.get("ground_truth"),
            tags=tc.get("tags", []),
        ))

    logger.info(f"Loaded {len(test_cases)} test cases from {path}")
    return test_cases


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def _chunk_matches(chunk: dict, patterns: list[str]) -> bool:
    """Check if a chunk matches any expected pattern (case-insensitive substring)."""
    content = chunk.get("content", "").lower()
    return any(p.lower() in content for p in patterns)


def mrr(retrieved: list[dict], expected: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, chunk in enumerate(retrieved):
        if _chunk_matches(chunk, expected):
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(retrieved: list[dict], expected: list[str], k: int = 10) -> float:
    """Hit@k: 1 if any relevant result in top k, 0 otherwise."""
    for chunk in retrieved[:k]:
        if _chunk_matches(chunk, expected):
            return 1.0
    return 0.0


def precision_at_k(retrieved: list[dict], expected: list[str], k: int = 10) -> float:
    """Precision@k: fraction of top k results that are relevant."""
    if k == 0:
        return 0.0
    relevant = sum(1 for c in retrieved[:k] if _chunk_matches(c, expected))
    return relevant / k


def ndcg(retrieved: list[dict], expected: list[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain."""
    dcg = sum(
        (1.0 if _chunk_matches(c, expected) else 0.0) / math.log2(i + 2)
        for i, c in enumerate(retrieved[:k])
    )
    n_relevant = sum(1 for c in retrieved[:k] if _chunk_matches(c, expected))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def context_recall(retrieved: list[dict], expected: list[str]) -> float:
    """Context Recall: fraction of expected patterns found in retrieved chunks."""
    if not expected:
        return 0.0
    all_content = " ".join(c.get("content", "") for c in retrieved).lower()
    found = sum(1 for p in expected if p.lower() in all_content)
    return found / len(expected)


def compute_retrieval_metrics(
    retrieved: list[dict],
    expected: list[str],
    k: int = 10,
) -> dict[str, float]:
    """Compute all retrieval metrics for a single query."""
    return {
        "mrr": mrr(retrieved, expected),
        "hit@10": hit_at_k(retrieved, expected, k=10),
        "hit@5": hit_at_k(retrieved, expected, k=5),
        "hit@1": hit_at_k(retrieved, expected, k=1),
        "precision@10": precision_at_k(retrieved, expected, k=10),
        "precision@5": precision_at_k(retrieved, expected, k=5),
        "ndcg@10": ndcg(retrieved, expected, k=10),
        "context_recall": context_recall(retrieved, expected),
    }


# ---------------------------------------------------------------------------
# RAG quality metrics (LLM-as-judge)
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are evaluating whether an answer is faithful to the provided context.

Context:
{context}

Question: {question}
Answer: {answer}

Rate the faithfulness of the answer on a scale of 0.0 to 1.0:
- 1.0: Every claim in the answer is supported by the context
- 0.5: Some claims are supported, some are not
- 0.0: The answer contradicts or is not supported by the context

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""

ANSWER_RELEVANCE_PROMPT = """You are evaluating whether an answer addresses the question.

Question: {question}
Answer: {answer}

Rate the relevance on a scale of 0.0 to 1.0:
- 1.0: The answer directly and completely addresses the question
- 0.5: The answer partially addresses the question
- 0.0: The answer does not address the question at all

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""

CONTEXT_PRECISION_PROMPT = """You are evaluating how much of the retrieved context was useful for answering the question.

Question: {question}
Ground truth answer: {ground_truth}

Retrieved context chunks:
{chunks}

For each chunk, determine if it was useful for producing the ground truth answer.
Rate the overall context precision on a scale of 0.0 to 1.0:
- 1.0: All chunks are relevant and useful
- 0.5: About half the chunks are useful
- 0.0: No chunks are useful

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""


def _llm_judge(
    provider,
    prompt: str,
    max_retries: int = 2,
) -> dict:
    """Call LLM and parse JSON response. Returns {"score": float, "reason": str}."""
    for attempt in range(max_retries + 1):
        try:
            response = provider.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            score = float(result.get("score", 0))
            return {"score": max(0.0, min(1.0, score)), "reason": result.get("reason", "")}
        except Exception as e:
            if attempt == max_retries:
                logger.warning(f"LLM judge failed after {max_retries + 1} attempts: {e}")
                return {"score": 0.0, "reason": f"LLM judge error: {e}"}
    return {"score": 0.0, "reason": "unreachable"}


def compute_rag_metrics(
    provider,
    question: str,
    answer: str,
    retrieved_chunks: list[dict],
    ground_truth: Optional[str] = None,
) -> dict[str, float]:
    """Compute RAG quality metrics using LLM-as-judge."""
    context = "\n\n---\n\n".join(c.get("content", "")[:1000] for c in retrieved_chunks[:10])
    metrics = {}

    # Faithfulness
    prompt = FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
    result = _llm_judge(provider, prompt)
    metrics["faithfulness"] = result["score"]

    # Answer Relevance
    prompt = ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
    result = _llm_judge(provider, prompt)
    metrics["answer_relevance"] = result["score"]

    # Context Precision (requires ground truth)
    if ground_truth:
        chunks_text = "\n\n".join(
            f"[Chunk {i+1}]: {c.get('content', '')[:500]}"
            for i, c in enumerate(retrieved_chunks[:10])
        )
        prompt = CONTEXT_PRECISION_PROMPT.format(
            question=question, ground_truth=ground_truth, chunks=chunks_text,
        )
        result = _llm_judge(provider, prompt)
        metrics["context_precision"] = result["score"]

    return metrics


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def generate_answer(
    provider,
    question: str,
    chunks: list[dict],
    max_context_chars: int = 12000,
) -> str:
    """Generate a RAG answer using the LLM provider and retrieved chunks."""
    from core.llm import build_rag_prompt, DEFAULT_SYSTEM_PROMPT
    messages = build_rag_prompt(question, chunks, max_context_chars=max_context_chars)
    return provider.generate(messages, system_prompt=DEFAULT_SYSTEM_PROMPT, max_tokens=1000)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(results: list[EvalResult]) -> dict[str, float]:
    """Average metrics across all results."""
    if not results:
        return {}
    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r.metrics.keys())
    return {
        key: sum(r.metrics.get(key, 0) for r in results) / len(results)
        for key in sorted(all_keys)
    }
