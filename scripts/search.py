#!/usr/bin/env python3
"""
CLI search tool for testing and debugging Trove search quality.

Usage:
    python scripts/search.py --client example "query"
    python scripts/search.py --client example --top-k 10 --verbose "query"
    python scripts/search.py --client example --merge-strategy rrf "query"
    python scripts/search.py --client example --mmr-lambda 0.7 "query"
    python scripts/search.py --client example --json "query"
    python scripts/search.py --client example --no-adaptive "query"
    python scripts/search.py --list
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.client import load_client, discover_clients
from core.search import HybridSearcher

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def list_clients():
    """List available clients."""
    clients = discover_clients()
    if not clients:
        print("No clients found.")
        return

    print("Available clients:")
    for name in clients:
        try:
            client = load_client(name)
            desc = client.config.get("description", "")
            has_index = client.index_path.exists()
            status = "ready" if has_index else "no index"
            print(f"  {name}: {desc} [{status}]")
        except Exception as e:
            print(f"  {name}: error - {e}")


def run_search(args):
    """Run a search with the given arguments."""
    client = load_client(args.client)
    query = " ".join(args.query)

    if not client.index_path.exists():
        print(f"Error: Client '{args.client}' has no index. Run ingestion first.", file=sys.stderr)
        sys.exit(1)

    # Build searcher with CLI overrides
    search_config = client.search_config

    searcher = HybridSearcher(
        index_path=str(client.index_path),
        collection_name=client.machine_name,
        bm25_weight=search_config.get("bm25_weight", 0.5),
        vector_weight=search_config.get("vector_weight", 0.5),
        min_score=search_config.get("min_score"),
        tokenizer_config=client.tokenization_config,
        embedding_config=client.embedding_config,
        # CLI overrides or config defaults
        adaptive_rerank=not args.no_adaptive if args.no_adaptive else search_config.get("adaptive_rerank", True),
        adaptive_rerank_threshold=search_config.get("adaptive_rerank_threshold", 0.9),
        adaptive_rerank_depth=search_config.get("adaptive_rerank_depth", 15),
        merge_strategy=args.merge_strategy or search_config.get("merge_strategy", "weighted"),
        rrf_k=search_config.get("rrf_k", 60),
        mmr_lambda=args.mmr_lambda if args.mmr_lambda is not None else search_config.get("mmr_lambda"),
        code_priority=search_config.get("code_priority", False),
        code_priority_boost=search_config.get("code_priority_boost", 0.5),
        cross_reference_code=search_config.get("cross_reference_code", False),
        cross_reference_top_k=search_config.get("cross_reference_top_k", 3),
    )

    top_k = args.top_k or search_config.get("top_k", 20)

    # Run search with quality
    search_result = searcher.search_with_quality(query=query, top_k=top_k)
    results = search_result.results
    quality = search_result.quality

    if args.json:
        output = {
            "query": query,
            "client": client.machine_name,
            "quality": {
                "avg_rerank_score": quality.avg_rerank_score,
                "top_score": quality.top_score,
                "score_spread": quality.score_spread,
                "unique_sources": quality.unique_sources,
                "total_candidates": quality.total_candidates,
                "merge_strategy": quality.merge_strategy,
                "mmr_applied": quality.mmr_applied,
                "adaptive_rerank_applied": quality.adaptive_rerank_applied,
                "timing": quality.timing,
            },
            "results": [
                {
                    "id": r["id"],
                    "content": r["content"][:500] if not args.full else r["content"],
                    "source": r.get("metadata", {}).get("source_file", "unknown"),
                    "rerank_score": r.get("_rerank_score", 0),
                    "bm25_score": r.get("_bm25_score", 0),
                    "vector_score": r.get("_vector_score", 0),
                    "combined_score": r.get("_combined_score", 0),
                    "chunk_type": r.get("metadata", {}).get("chunk_type", "document"),
                }
                for r in results
            ],
        }
        print(json.dumps(output, indent=2))
        return

    # Text output
    print(f"Search: \"{query}\" in {client.name}")
    print(f"Strategy: {quality.merge_strategy} | MMR: {quality.mmr_applied} | Adaptive: {quality.adaptive_rerank_applied}")
    print(f"Results: {len(results)} (from {quality.total_candidates} candidates, {quality.unique_sources} sources)")
    print(f"Scores: avg={quality.avg_rerank_score:.3f}, top={quality.top_score:.3f}, spread={quality.score_spread:.3f}")

    if args.verbose:
        t = quality.timing
        print(f"Timing: search={t.get('parallel_search', 0):.2f}s, merge={t.get('merge', 0):.3f}s, rerank={t.get('rerank', 0):.2f}s, mmr={t.get('mmr', 0):.2f}s, total={t.get('total', 0):.2f}s")

    print("-" * 60)

    for i, chunk in enumerate(results, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source_file", "unknown")
        rerank = chunk.get("_rerank_score", 0)
        bm25 = chunk.get("_bm25_score", 0)
        vector = chunk.get("_vector_score", 0)
        combined = chunk.get("_combined_score", 0)
        chunk_type = metadata.get("chunk_type", "document")

        print(f"\n[{i}] {source} ({chunk_type})")
        print(f"    rerank={rerank:.3f}  bm25={bm25:.3f}  vector={vector:.3f}  combined={combined:.4f}")

        if args.verbose:
            lang = metadata.get("language", "")
            symbol = metadata.get("symbol_name", "")
            if lang:
                print(f"    language={lang}", end="")
            if symbol:
                print(f"  symbol={metadata.get('symbol_type', '')}.{symbol}", end="")
            if lang or symbol:
                print()

        content = chunk.get("content", "")
        if not args.full and len(content) > 300:
            content = content[:300] + "..."
        # Indent content
        for line in content.split("\n")[:10 if not args.full else 999]:
            print(f"    {line}")
        if not args.full and content.count("\n") > 10:
            print(f"    ... ({content.count(chr(10))} more lines)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Trove CLI search tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -c example "hybrid search"
  %(prog)s -c example --verbose "hybrid search"
  %(prog)s -c example --merge-strategy rrf "hybrid search"
  %(prog)s -c example --mmr-lambda 0.7 "hybrid search"
  %(prog)s -c example --json "hybrid search"
  %(prog)s -c example --no-adaptive "hybrid search"
  %(prog)s --list
        """,
    )

    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument("-c", "--client", help="Client name")
    parser.add_argument("--list", action="store_true", help="List available clients")
    parser.add_argument("--top-k", type=int, help="Number of results")
    parser.add_argument("--merge-strategy", choices=["weighted", "rrf"], help="Merge strategy override")
    parser.add_argument("--mmr-lambda", type=float, help="MMR diversity (0.0=diverse, 1.0=relevant)")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive reranking")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show timing and extra metadata")
    parser.add_argument("--full", action="store_true", help="Show full content (no truncation)")

    args = parser.parse_args()

    if args.list:
        list_clients()
        return

    if not args.client:
        parser.error("--client is required (use --list to see available clients)")

    if not args.query:
        parser.error("query is required")

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    run_search(args)


if __name__ == "__main__":
    main()
