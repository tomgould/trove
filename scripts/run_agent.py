#!/usr/bin/env python3
"""
CLI for running Trove agents locally via LM Studio.

Usage:
    python scripts/run_agent.py --agent code-reviewer --client example "Review the search pipeline"
    python scripts/run_agent.py --agent code-reviewer --client example --interactive
    python scripts/run_agent.py --list-agents
    python scripts/run_agent.py --list-clients
    python scripts/run_agent.py --agent security-reviewer --client myproject --top-k 15 --json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agents import AgentLoader
from core.client import load_client, discover_clients
from core.llm import create_provider_from_config
from core.runner import AgentRunner
from core.search import HybridSearcher

TROVE_ROOT = Path(__file__).parent.parent
AGENTS_DIR = TROVE_ROOT / "agents"
CLIENTS_DIR = TROVE_ROOT / "clients"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_searcher(client):
    """Create a HybridSearcher from client config."""
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
        merge_strategy=search_config.get("merge_strategy", "weighted"),
        rrf_k=search_config.get("rrf_k", 60),
        mmr_lambda=search_config.get("mmr_lambda"),
        code_priority=search_config.get("code_priority", False),
        code_priority_boost=search_config.get("code_priority_boost", 0.5),
        cross_reference_code=search_config.get("cross_reference_code", False),
        cross_reference_top_k=search_config.get("cross_reference_top_k", 3),
        reranker_backend=search_config.get("reranker", "local"),
        reranker_endpoint=search_config.get("reranker_endpoint"),
        bm25_k1=search_config.get("bm25_k1", 1.5),
        bm25_b=search_config.get("bm25_b", 0.75),
        score_normalization=search_config.get("score_normalization", "minmax"),
        hnsw_ef_construction=client.embedding_config.get("hnsw_ef_construction", 200),
        hnsw_max_neighbors=client.embedding_config.get("hnsw_max_neighbors", 16),
        hnsw_ef_search=client.embedding_config.get("hnsw_ef_search", 100),
        cache_ttl=search_config.get("cache_ttl", 300),
        cache_max_size=search_config.get("cache_max_size", 128),
    )


def list_agents():
    """List all available agents."""
    loader = AgentLoader(AGENTS_DIR)
    agents = loader.list()
    if not agents:
        print("No agents found.")
        return

    print(f"\nAvailable Agents ({len(agents)}):\n")
    for a in agents:
        tags = ", ".join(a["tags"]) if a["tags"] else "none"
        print(f"  {a['filename']:<30} {a['name']}")
        print(f"  {'':30} {a['description']}")
        print(f"  {'':30} Tags: {tags}\n")


def list_clients():
    """List all available clients."""
    clients = discover_clients(CLIENTS_DIR)
    if not clients:
        print("No clients found.")
        return

    print(f"\nAvailable Clients ({len(clients)}):\n")
    for name in clients:
        try:
            c = load_client(name, CLIENTS_DIR)
            index_exists = c.index_path.exists()
            status = "indexed" if index_exists else "no index"
            print(f"  {name:<30} {c.name} ({status})")
        except Exception as e:
            print(f"  {name:<30} Error: {e}")


def format_result_json(result):
    """Format AgentResult as JSON-serializable dict."""
    quality_dict = None
    if result.search_quality:
        q = result.search_quality
        quality_dict = {
            "avg_rerank_score": q.avg_rerank_score,
            "top_score": q.top_score,
            "unique_sources": q.unique_sources,
            "total_candidates": q.total_candidates,
            "merge_strategy": q.merge_strategy,
        }

    return {
        "agent": result.agent_name,
        "query": result.query,
        "response": result.response,
        "sources": result.sources,
        "search_quality": quality_dict,
        "timing": result.timing,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Trove agents locally via LM Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", help="Query to run")
    parser.add_argument("--agent", "-a", help="Agent name or filename slug")
    parser.add_argument("--client", "-c", help="Trove client name")
    parser.add_argument("--top-k", type=int, default=10, help="Search results (default: 10)")
    parser.add_argument("--max-tokens", type=int, help="Override max generation tokens")
    parser.add_argument("--provider", help="Override LLM provider (e.g. lmstudio_anthropic)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--list-agents", action="store_true", help="List available agents")
    parser.add_argument("--list-clients", action="store_true", help="List available clients")
    parser.add_argument("--store-memory", action="store_true", help="Store result in memory")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress info logging")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # List modes
    if args.list_agents:
        list_agents()
        return
    if args.list_clients:
        list_clients()
        return

    # Validate required args
    if not args.agent:
        parser.error("--agent is required (or use --list-agents)")
    if not args.client:
        parser.error("--client is required (or use --list-clients)")
    if not args.query and not args.interactive:
        parser.error("query is required (or use --interactive)")

    # Load agent
    loader = AgentLoader(AGENTS_DIR)
    agent = loader.get(args.agent)
    if not agent:
        print(f"Error: Agent '{args.agent}' not found.", file=sys.stderr)
        print("Available agents:", file=sys.stderr)
        for a in loader.list():
            print(f"  {a['filename']}", file=sys.stderr)
        sys.exit(1)

    # Load client
    try:
        client = load_client(args.client, CLIENTS_DIR)
    except Exception as e:
        print(f"Error loading client '{args.client}': {e}", file=sys.stderr)
        sys.exit(1)

    if not client.index_path.exists():
        print(f"Error: Client '{args.client}' has no index. Run ingestion first.", file=sys.stderr)
        sys.exit(1)

    # Create LLM provider
    llm_config = client.llm_config
    if args.provider:
        llm_config = {**llm_config, "provider": args.provider}

    try:
        provider = create_provider_from_config(llm_config)
    except Exception as e:
        print(f"Error creating LLM provider: {e}", file=sys.stderr)
        sys.exit(1)

    # Create searcher
    try:
        searcher = create_searcher(client)
    except Exception as e:
        print(f"Error creating searcher: {e}", file=sys.stderr)
        sys.exit(1)

    # Optional memory store
    memory_store = None
    if args.store_memory:
        from core.memory import MemoryStore
        db_path = client.index_path / "trove_state.db"
        memory_store = MemoryStore(db_path)

    # Create runner
    runner = AgentRunner(
        agent=agent,
        client=client,
        llm_provider=provider,
        searcher=searcher,
        memory_store=memory_store,
    )

    if args.interactive:
        runner.chat(top_k=args.top_k)
    else:
        result = runner.run(
            query=args.query,
            top_k=args.top_k,
            store_memory=args.store_memory,
            max_tokens=args.max_tokens,
        )

        if args.json:
            print(json.dumps(format_result_json(result), indent=2))
        else:
            print(f"\nAgent: {result.agent_name}")
            print(f"Query: {result.query}")
            print(f"Sources: {len(result.sources)}")
            print(f"Timing: {result.timing}")
            print(f"\n{'=' * 60}\n")
            print(result.response)


if __name__ == "__main__":
    main()
