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

from core.client import add_clients_dir_argument, apply_clients_dir_argument
from core.client import ClientNotFoundError, load_client, discover_clients
from core.search import HybridSearcher
from core.search.merge import merge_client_results
from core.search.searcher import MAX_TOP_K, citation

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
            has_index = client.has_index
            status = "ready" if has_index else "no index"
            print(f"  {name}: {desc} [{status}]")
        except Exception as e:
            print(f"  {name}: error - {e}")


def _build_searcher(client, args) -> HybridSearcher:
    """A searcher for ``client`` with the CLI overrides applied."""
    search_config = client.search_config
    return HybridSearcher(
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
        adaptive_rerank_depth=search_config.get("adaptive_rerank_depth", 30),
        merge_strategy=args.merge_strategy or search_config.get("merge_strategy", "weighted"),
        rrf_k=search_config.get("rrf_k", 60),
        mmr_lambda=args.mmr_lambda if args.mmr_lambda is not None else search_config.get("mmr_lambda"),
        code_priority=search_config.get("code_priority", False),
        code_priority_boost=search_config.get("code_priority_boost", 0.5),
        cross_reference_code=search_config.get("cross_reference_code", False),
        cross_reference_top_k=search_config.get("cross_reference_top_k", 3),
        max_chunks_per_source=search_config.get("max_chunks_per_source", 2),
        heading_boost=search_config.get("heading_boost", 1.0),
    )


def _result_json(r: dict, args, client_name: str) -> dict:
    metadata = r.get("metadata", {})
    return {
        "id": r["id"],
        "client": r.get("_source_client", client_name),
        "content": r["content"][:500] if not args.full else r["content"],
        "source": metadata.get("source_file", "unknown"),
        **citation(r),
        "rerank_score": r.get("_rerank_score", 0),
        "bm25_score": r.get("_bm25_score", 0),
        "vector_score": r.get("_vector_score", 0),
        "combined_score": r.get("_combined_score", 0),
        "chunk_type": metadata.get("chunk_type", "document"),
    }


def _print_result(i: int, chunk: dict, args, show_client: bool = False) -> None:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source_file", "unknown")
    rerank = chunk.get("_rerank_score", 0)
    bm25 = chunk.get("_bm25_score", 0)
    vector = chunk.get("_vector_score", 0)
    combined = chunk.get("_combined_score", 0)
    chunk_type = metadata.get("chunk_type", "document")

    label = f"{chunk.get('_source_client')}: " if show_client else ""
    print(f"\n[{i}] {label}{source} ({chunk_type})")
    cite = citation(chunk)
    if cite["title"]:
        print(f"    title: {cite['title']}")
    if cite["url"]:
        print(f"    url: {cite['url']}")
    if cite["section"]:
        print(f"    section: {cite['section']}")
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


def _top_k(requested: int) -> int:
    """``requested`` lowered to MAX_TOP_K, with a note on stderr when it is lowered."""
    if requested > MAX_TOP_K:
        print(f"Note: --top-k is limited to {MAX_TOP_K}; showing at most {MAX_TOP_K} results", file=sys.stderr)
        return MAX_TOP_K
    return requested


def run_search(args):
    """Run a search on one client (--client)."""
    try:
        client = load_client(args.client)
    except ClientNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    query = " ".join(args.query)

    if not client.has_index:
        print(f"Error: Client '{args.client}' has no index. Run ingestion first.", file=sys.stderr)
        sys.exit(1)

    searcher = _build_searcher(client, args)
    top_k = _top_k(args.top_k or client.search_config.get("top_k", 20))

    # Run search with quality
    search_result = searcher.search_with_quality(
        query=query, top_k=top_k, rerank_top_n=client.search_config.get("rerank_top_n", 50),
    )
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
            "results": [_result_json(r, args, client.machine_name) for r in results],
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
        _print_result(i, chunk, args)

    print()


def run_multi_search(args, client_names: list[str]):
    """Search several clients and merge their results by rank (--clients or --all)."""
    query = " ".join(args.query)
    clients = []
    skipped = []
    for name in client_names:
        try:
            client = load_client(name)
        except ClientNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if client.has_index:
            clients.append(client)
        else:
            skipped.append(name)
    for name in skipped:
        print(f"Skipping '{name}': no index", file=sys.stderr)
    if not clients:
        print("Error: none of the clients has an index. Run ingestion first.", file=sys.stderr)
        sys.exit(1)

    top_k = _top_k(args.top_k or 20)
    per_client: dict[str, list[dict]] = {}
    for client in clients:
        searcher = _build_searcher(client, args)
        per_client[client.machine_name] = searcher.search_with_quality(
            query=query, top_k=top_k, rerank_top_n=client.search_config.get("rerank_top_n", 50),
        ).results
        searcher.close()
    results = merge_client_results(per_client, top_k=top_k)

    if args.json:
        output = {
            "query": query,
            "clients": [c.machine_name for c in clients],
            "skipped": skipped,
            "merge": "rrf",
            "results": [_result_json(r, args, r.get("_source_client", "")) for r in results],
        }
        print(json.dumps(output, indent=2))
        return

    print(f"Search: \"{query}\" in {', '.join(c.machine_name for c in clients)}")
    print(f"Results: {len(results)} merged by rank across {len(clients)} clients")
    print("-" * 60)
    for i, chunk in enumerate(results, 1):
        _print_result(i, chunk, args, show_client=True)
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
  %(prog)s --clients docs,blog "hybrid search"
  %(prog)s --all --json "hybrid search"
  %(prog)s --list
        """,
    )

    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument("-c", "--client", help="Client name")
    parser.add_argument("--clients", help="Search several clients, comma-separated, and merge the results")
    parser.add_argument("--all", action="store_true", help="Search every client that has an index")
    parser.add_argument("--list", action="store_true", help="List available clients")
    parser.add_argument("--top-k", type=int, help=f"Number of results (at most {MAX_TOP_K})")
    parser.add_argument("--merge-strategy", choices=["weighted", "rrf"], help="Merge strategy override")
    parser.add_argument("--mmr-lambda", type=float, help="MMR diversity (0.0=diverse, 1.0=relevant)")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive reranking")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show timing and extra metadata")
    parser.add_argument("--full", action="store_true", help="Show full content (no truncation)")

    add_clients_dir_argument(parser)
    args = parser.parse_args()
    apply_clients_dir_argument(args)

    if args.list:
        list_clients()
        return

    chosen = sum(bool(x) for x in (args.client, args.clients, args.all))
    if chosen == 0:
        parser.error("--client, --clients or --all is required (use --list to see available clients)")
    if chosen > 1:
        parser.error("use only one of --client, --clients and --all")

    if not args.query:
        parser.error("query is required")

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    if args.client:
        run_search(args)
    elif args.all:
        run_multi_search(args, discover_clients())
    else:
        run_multi_search(args, [n.strip() for n in args.clients.split(",") if n.strip()])


if __name__ == "__main__":
    main()
