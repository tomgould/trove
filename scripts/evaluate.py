#!/usr/bin/env python3
"""
CLI evaluation tool for measuring Trove search quality.

Usage:
    python scripts/evaluate.py --client example
    python scripts/evaluate.py --client example --tags search
    python scripts/evaluate.py --client example --rag
    python scripts/evaluate.py --client example --compare "weighted,rrf"
    python scripts/evaluate.py --client example --json
    python scripts/evaluate.py --client example --output reports/eval.md
    python scripts/evaluate.py --client example --verbose
    python scripts/evaluate.py --list
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.client import load_client, discover_clients
from core.evaluation import (
    EvalReport,
    EvalResult,
    aggregate_metrics,
    compute_rag_metrics,
    compute_retrieval_metrics,
    generate_answer,
    load_eval_dataset,
)
from core.search import HybridSearcher

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def list_clients_with_datasets():
    """List clients that have eval datasets."""
    clients = discover_clients()
    if not clients:
        print("No clients found.")
        return

    print("Clients with evaluation datasets:")
    found = False
    for name in clients:
        try:
            client = load_client(name)
            dataset_path = client.path / "tests" / "eval_dataset.json"
            if dataset_path.exists():
                with open(dataset_path) as f:
                    data = json.load(f)
                count = len(data.get("test_cases", []))
                print(f"  {name}: {count} test cases")
                found = True
        except Exception as e:
            pass

    if not found:
        print("  (none found)")
    print()
    print("All available clients:")
    for name in clients:
        try:
            client = load_client(name)
            has_dataset = (client.path / "tests" / "eval_dataset.json").exists()
            has_index = client.index_path.exists()
            status_parts = []
            if has_index:
                status_parts.append("indexed")
            if has_dataset:
                status_parts.append("eval dataset")
            status = ", ".join(status_parts) if status_parts else "no index"
            print(f"  {name}: [{status}]")
        except Exception as e:
            print(f"  {name}: error - {e}")


def create_searcher(client, merge_strategy=None):
    """Create a HybridSearcher for a client with optional overrides."""
    search_config = client.search_config
    return HybridSearcher(
        index_path=str(client.index_path),
        collection_name=client.machine_name,
        bm25_weight=search_config.get("bm25_weight", 0.5),
        vector_weight=search_config.get("vector_weight", 0.5),
        min_score=search_config.get("min_score"),
        tokenizer_config=client.tokenization_config,
        embedding_config=client.embedding_config,
        adaptive_rerank=search_config.get("adaptive_rerank", True),
        adaptive_rerank_threshold=search_config.get("adaptive_rerank_threshold", 0.9),
        adaptive_rerank_depth=search_config.get("adaptive_rerank_depth", 15),
        merge_strategy=merge_strategy or search_config.get("merge_strategy", "weighted"),
        rrf_k=search_config.get("rrf_k", 60),
        mmr_lambda=search_config.get("mmr_lambda"),
        code_priority=search_config.get("code_priority", False),
        code_priority_boost=search_config.get("code_priority_boost", 0.5),
        cross_reference_code=search_config.get("cross_reference_code", False),
        cross_reference_top_k=search_config.get("cross_reference_top_k", 3),
    )


def run_evaluation(client, searcher, dataset, include_rag=False, provider=None, top_k=20):
    """Run evaluation on all test cases."""
    results = []
    for i, tc in enumerate(dataset):
        print(f"  [{i+1}/{len(dataset)}] {tc.id}: {tc.query[:60]}...", end="", flush=True)

        search_result = searcher.search_with_quality(query=tc.query, top_k=top_k)
        metrics = compute_retrieval_metrics(search_result.results, tc.expected_chunks)

        answer = None
        if include_rag and provider and tc.ground_truth:
            answer = generate_answer(provider, tc.query, search_result.results)
            rag_metrics = compute_rag_metrics(
                provider, tc.query, answer, search_result.results, tc.ground_truth,
            )
            metrics.update(rag_metrics)

        results.append(EvalResult(
            test_case=tc,
            retrieved_chunks=search_result.results,
            metrics=metrics,
            answer=answer,
            quality=search_result.quality,
        ))

        mrr_val = metrics.get("mrr", 0)
        recall_val = metrics.get("context_recall", 0)
        print(f" MRR={mrr_val:.3f} Recall={recall_val:.3f}")

    return results


def format_aggregate_table(aggregate: dict[str, float]) -> str:
    """Format aggregate metrics as a text table."""
    lines = []
    lines.append(f"{'Metric':<20} {'Score':>8}")
    lines.append("-" * 30)
    for key, val in aggregate.items():
        lines.append(f"{key:<20} {val:>8.3f}")
    return "\n".join(lines)


def format_comparison_table(reports: list[EvalReport]) -> str:
    """Format side-by-side comparison table."""
    all_keys: set[str] = set()
    for r in reports:
        all_keys.update(r.aggregate.keys())

    labels = [r.config_label for r in reports]
    header = f"{'Metric':<20}" + "".join(f" | {l:>10}" for l in labels) + " | Winner"
    sep = "-" * len(header)

    lines = [header, sep]
    for key in sorted(all_keys):
        vals = [r.aggregate.get(key, 0) for r in reports]
        row = f"{key:<20}"
        for v in vals:
            row += f" | {v:>10.3f}"
        # Determine winner
        max_val = max(vals)
        winners = [labels[i] for i, v in enumerate(vals) if abs(v - max_val) < 0.0005]
        winner = "tie" if len(winners) > 1 else winners[0]
        row += f" | {winner}"
        lines.append(row)

    return "\n".join(lines)


def format_markdown_report(report: EvalReport, verbose: bool = False) -> str:
    """Format evaluation results as markdown."""
    lines = []
    lines.append(f"# Evaluation Report: {report.client_name}")
    lines.append(f"Date: {report.timestamp[:10]}")
    lines.append(f"Config: {report.config_label}")
    lines.append(f"Test cases: {len(report.results)}")
    lines.append("")

    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for key, val in report.aggregate.items():
        lines.append(f"| {key} | {val:.3f} |")
    lines.append("")

    if verbose:
        lines.append("## Per-Query Results")
        lines.append("")
        for r in report.results:
            lines.append(f"### {r.test_case.id}: {r.test_case.query}")
            lines.append("")
            for mk, mv in r.metrics.items():
                lines.append(f"- **{mk}**: {mv:.3f}")
            if r.answer:
                lines.append(f"- **Answer**: {r.answer[:200]}...")
            lines.append("")

    return "\n".join(lines)


def format_json_output(reports: list[EvalReport]) -> str:
    """Format results as JSON."""
    output = []
    for report in reports:
        entry = {
            "client": report.client_name,
            "config": report.config_label,
            "timestamp": report.timestamp,
            "aggregate": report.aggregate,
            "results": [
                {
                    "id": r.test_case.id,
                    "query": r.test_case.query,
                    "tags": r.test_case.tags,
                    "metrics": r.metrics,
                    "answer": r.answer,
                    "num_results": len(r.retrieved_chunks),
                }
                for r in report.results
            ],
        }
        output.append(entry)

    if len(output) == 1:
        return json.dumps(output[0], indent=2)
    return json.dumps(output, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Trove evaluation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -c example                         Run retrieval evaluation
  %(prog)s -c example --tags search           Only search-tagged tests
  %(prog)s -c example --rag                   Include LLM-as-judge metrics
  %(prog)s -c example --compare "weighted,rrf"  A/B comparison
  %(prog)s -c example --json                  JSON output
  %(prog)s -c example --output eval.md        Write markdown report
  %(prog)s --list                          List clients with datasets
        """,
    )

    parser.add_argument("-c", "--client", help="Client name")
    parser.add_argument("--list", action="store_true", help="List clients with eval datasets")
    parser.add_argument("--tags", help="Comma-separated tags to filter test cases")
    parser.add_argument("--rag", action="store_true", help="Include RAG quality metrics (requires LLM)")
    parser.add_argument("--compare", help='A/B comparison of merge strategies (e.g. "weighted,rrf")')
    parser.add_argument("--top-k", type=int, help="Override search top_k")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--output", help="Write markdown report to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-query details")

    args = parser.parse_args()

    if args.list:
        list_clients_with_datasets()
        return

    if not args.client:
        parser.error("--client is required (use --list to see available clients)")

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Load client and dataset
    client = load_client(args.client)
    if not client.index_path.exists():
        print(f"Error: Client '{args.client}' has no index. Run ingestion first.", file=sys.stderr)
        sys.exit(1)

    try:
        dataset = load_eval_dataset(client)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"Create an eval dataset at: clients/{args.client}/tests/eval_dataset.json", file=sys.stderr)
        sys.exit(1)

    # Filter by tags
    if args.tags:
        tag_filter = set(args.tags.split(","))
        dataset = [tc for tc in dataset if tag_filter & set(tc.tags)]
        if not dataset:
            print(f"No test cases match tags: {args.tags}", file=sys.stderr)
            sys.exit(1)

    top_k = args.top_k or client.search_config.get("top_k", 20)

    # Set up LLM provider for RAG metrics
    provider = None
    if args.rag:
        from core.llm import get_provider
        llm_config = client.llm_config
        try:
            provider = get_provider(
                llm_config.get("provider", "lmstudio"),
                host=llm_config.get("host", "http://localhost:1234"),
            )
        except Exception as e:
            print(f"Warning: Could not initialize LLM provider: {e}", file=sys.stderr)
            print("RAG metrics will be skipped.", file=sys.stderr)

    # Run evaluation(s)
    reports = []

    if args.compare:
        strategies = [s.strip() for s in args.compare.split(",")]
        print(f"A/B Comparison: {' vs '.join(strategies)} ({len(dataset)} test cases)")
        print("=" * 60)

        for strategy in strategies:
            print(f"\nRunning: {strategy}")
            searcher = create_searcher(client, merge_strategy=strategy)
            results = run_evaluation(
                client, searcher, dataset,
                include_rag=args.rag, provider=provider, top_k=top_k,
            )
            agg = aggregate_metrics(results)
            reports.append(EvalReport(
                client_name=client.machine_name,
                config_label=strategy,
                results=results,
                aggregate=agg,
            ))
            searcher.close()
    else:
        strategy = client.search_config.get("merge_strategy", "weighted")
        print(f"Evaluation: {client.name} ({len(dataset)} test cases, strategy={strategy}, top_k={top_k})")
        print("=" * 60)

        searcher = create_searcher(client)
        results = run_evaluation(
            client, searcher, dataset,
            include_rag=args.rag, provider=provider, top_k=top_k,
        )
        agg = aggregate_metrics(results)
        reports.append(EvalReport(
            client_name=client.machine_name,
            config_label=strategy,
            results=results,
            aggregate=agg,
        ))
        searcher.close()

    # Output results
    if args.json:
        print(format_json_output(reports))
        return

    print()
    if len(reports) > 1:
        print(format_comparison_table(reports))
    else:
        print(format_aggregate_table(reports[0].aggregate))

    if args.verbose and not args.compare:
        report = reports[0]
        print()
        print("Per-Query Details:")
        print("-" * 60)
        for r in report.results:
            print(f"\n  {r.test_case.id}: {r.test_case.query}")
            for mk, mv in r.metrics.items():
                print(f"    {mk:<20} {mv:.3f}")
            if r.answer:
                print(f"    answer: {r.answer[:150]}...")

    # Write markdown report
    if args.output:
        for report in reports:
            md = format_markdown_report(report, verbose=True)
            output_path = Path(args.output)
            if len(reports) > 1:
                output_path = output_path.with_stem(f"{output_path.stem}_{report.config_label}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(md)
            print(f"\nReport written to: {output_path}")

    print()


if __name__ == "__main__":
    main()
