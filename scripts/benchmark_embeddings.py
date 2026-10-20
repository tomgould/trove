#!/usr/bin/env python3
"""
Embedding model benchmark tool.

Benchmarks embedding models against a client's eval dataset by re-embedding
all chunks with each candidate model and measuring retrieval quality.

Usage:
    python scripts/benchmark_embeddings.py --client example --models "all-MiniLM-L6-v2,BAAI/bge-base-en-v1.5"
    python scripts/benchmark_embeddings.py --client example --models "all-MiniLM-L6-v2" --output results.json
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer

from core.client import load_client
from core.evaluation import load_eval_dataset, compute_retrieval_metrics, EvalResult, aggregate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_client_data(client) -> tuple[list[dict], list[dict]]:
    """
    Load chunks and eval dataset from a client.

    Args:
        client: Client instance

    Returns:
        Tuple of (chunks, eval_cases)

    Raises:
        FileNotFoundError: If eval dataset doesn't exist
    """
    eval_cases_raw = load_eval_dataset(client)
    eval_cases = [
        {"query": tc.query, "expected_chunks": tc.expected_chunks, "test_case": tc}
        for tc in eval_cases_raw
    ]

    # Load existing chunks from ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=str(client.index_path),
    )

    try:
        collection = chroma_client.get_collection(client.machine_name)
        result = collection.get(include=["documents", "metadatas"])
        chunks = []
        for i, doc_id in enumerate(result["ids"]):
            chunks.append({
                "id": doc_id,
                "content": result["documents"][i],
                "metadata": result["metadatas"][i] if result["metadatas"] else {},
            })
    except Exception as e:
        logger.warning(f"Could not load chunks from index: {e}")
        chunks = []

    if not chunks:
        raise ValueError(f"No chunks found in index for client {client.machine_name}")

    return chunks, eval_cases


def benchmark_model(
    model_name: str,
    chunks: list[dict],
    eval_cases: list[dict],
) -> dict:
    """
    Benchmark a single embedding model.

    Args:
        model_name: SentenceTransformer model name
        chunks: List of chunk dicts with 'id' and 'content'
        eval_cases: List of dicts with 'query', 'expected_chunks', and 'test_case'

    Returns:
        Dict with model name, dimension, metrics, and timing
    """
    logger.info(f"Benchmarking model: {model_name}")

    # Load model
    model = SentenceTransformer(model_name)
    dimension = model.get_sentence_embedding_dimension()

    # Embed all chunks
    texts = [c["content"] for c in chunks]
    start_time = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embed_time = time.time() - start_time
    embed_time_per_chunk = (embed_time / len(chunks)) * 1000  # ms

    # Create temp ChromaDB collection in a temp directory
    tmp_dir = tempfile.mkdtemp(prefix="trove_bench_")
    chroma_client = chromadb.PersistentClient(path=tmp_dir)
    collection_name = f"bench_{model_name.replace('/', '_')}"
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Index chunks in batches (ChromaDB limit is ~41666 per batch)
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        collection.add(
            ids=[c["id"] for c in chunks[i:batch_end]],
            embeddings=embeddings[i:batch_end].tolist(),
            documents=[c["content"] for c in chunks[i:batch_end]],
        )

    # Run eval queries
    all_results = []
    for case in eval_cases:
        query_embedding = model.encode([case["query"]])[0].tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(10, len(chunks)),
            include=["documents"],
        )

        # Build retrieved chunks format
        retrieved = []
        for i, doc_id in enumerate(results["ids"][0]):
            retrieved.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": {},
            })

        metrics = compute_retrieval_metrics(retrieved, case["expected_chunks"])
        # Wrap in EvalResult for aggregate_metrics
        test_case = case.get("test_case")
        if test_case is None:
            from core.evaluation import TestCase
            test_case = TestCase(id="bench", query=case["query"], expected_chunks=case["expected_chunks"])
        all_results.append(EvalResult(
            test_case=test_case,
            retrieved_chunks=retrieved,
            metrics=metrics,
        ))

    # Aggregate
    agg = aggregate_metrics(all_results) if all_results else {}

    # Cleanup temp directory
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total_time = time.time() - start_time

    return {
        "model": model_name,
        "dimension": dimension,
        "mrr": agg.get("mrr", 0.0),
        "hit_at_5": agg.get("hit@5", 0.0),
        "hit_at_10": agg.get("hit@10", 0.0),
        "ndcg": agg.get("ndcg@10", 0.0),
        "precision_at_5": agg.get("precision@5", 0.0),
        "embed_time_per_chunk_ms": round(embed_time_per_chunk, 2),
        "total_time_s": round(total_time, 1),
        "num_chunks": len(chunks),
        "num_queries": len(eval_cases),
    }


def format_results_table(results: list[dict]) -> str:
    """Format benchmark results as a readable table."""
    if not results:
        return "No results."

    header = (
        f"{'Model':<35} {'Dim':>5} {'MRR':>6} {'Hit@5':>6} {'Hit@10':>7} "
        f"{'nDCG':>6} {'ms/chunk':>9} {'Total':>7}"
    )
    separator = "-" * len(header)

    rows = [header, separator]
    for r in results:
        row = (
            f"{r['model']:<35} {r['dimension']:>5} {r['mrr']:>6.3f} "
            f"{r.get('hit_at_5', 0):>6.3f} {r.get('hit_at_10', 0):>7.3f} "
            f"{r['ndcg']:>6.3f} {r['embed_time_per_chunk_ms']:>9.2f} "
            f"{r['total_time_s']:>6.1f}s"
        )
        rows.append(row)

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Trove Embedding Model Benchmark")
    parser.add_argument("--client", "-c", required=True, help="Client name")
    parser.add_argument(
        "--models", "-m", required=True,
        help="Comma-separated model names (e.g., 'all-MiniLM-L6-v2,BAAI/bge-base-en-v1.5')",
    )
    parser.add_argument("--output", "-o", help="Output JSON file path")
    args = parser.parse_args()

    client = load_client(args.client)
    model_names = [m.strip() for m in args.models.split(",")]

    logger.info(f"Benchmarking {len(model_names)} models against client: {client.name}")

    try:
        chunks, eval_cases = load_client_data(client)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Loaded {len(chunks)} chunks, {len(eval_cases)} eval queries")

    results = []
    for model_name in model_names:
        try:
            result = benchmark_model(model_name, chunks, eval_cases)
            results.append(result)
            logger.info(f"  {model_name}: MRR={result['mrr']:.3f}, nDCG={result['ndcg']:.3f}")
        except Exception as e:
            logger.error(f"  {model_name}: FAILED — {e}")

    # Print table
    print("\n" + format_results_table(results))

    # Save JSON if requested
    if args.output and results:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
