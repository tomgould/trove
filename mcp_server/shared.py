"""Shared state, cache, and helper functions for the Trove MCP server."""

import logging
import os
import re
import threading
from pathlib import Path
from typing import Optional

from core.client import load_client, discover_clients, Client
from core.ingest_job import IngestionJobManager
from core.search import HybridSearcher, SearchResult, SearchQuality
from core.memory import MemoryStore
from core.workflow import WorkflowEngine
from core.agents import AgentLoader
from core.llm import (
    LLMProvider,
    LMStudioProvider,
    LMStudioAnthropicProvider,
    LMStudioNativeProvider,
    ClaudeProvider,
    build_citation_map,
    create_provider_from_config,
)

logger = logging.getLogger("trove-mcp")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TROVE_ROOT = Path(__file__).parent.parent
CLIENTS_DIR = TROVE_ROOT / "clients"

# Client name validation — alphanumeric, hyphens, underscores only
_VALID_CLIENT_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


# ---------------------------------------------------------------------------
# Searcher Cache (replaces single _current_client / _current_searcher globals)
# ---------------------------------------------------------------------------

class SearcherCache:
    """LRU cache for loaded Trove searchers (thread-safe)."""

    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self._cache: dict[str, tuple[Client, HybridSearcher]] = {}
        self._access_order: list[str] = []  # most recent at end
        self._current_name: Optional[str] = None
        self._lock = threading.Lock()

    def get(self, name: str) -> Optional[tuple[Client, HybridSearcher]]:
        """Get a cached searcher, updating access order."""
        with self._lock:
            if name in self._cache:
                self._access_order.remove(name)
                self._access_order.append(name)
                return self._cache[name]
            return None

    def put(self, name: str, client: Client, searcher: HybridSearcher):
        """Add a searcher, evicting LRU if full."""
        with self._lock:
            if name in self._cache:
                old_client, old_searcher = self._cache[name]
                try:
                    old_searcher.close()
                except Exception:
                    pass
                self._access_order.remove(name)
            elif len(self._cache) >= self.max_size:
                evict_name = self._access_order.pop(0)
                evicted = self._cache.pop(evict_name)
                try:
                    evicted[1].close()
                except Exception:
                    pass
                logger.info(f"Evicted searcher: {evict_name}")

            self._cache[name] = (client, searcher)
            self._access_order.append(name)

    @property
    def current(self) -> Optional[tuple[Client, HybridSearcher]]:
        with self._lock:
            if self._current_name and self._current_name in self._cache:
                return self._cache[self._current_name]
            return None

    @property
    def current_name(self) -> Optional[str]:
        return self._current_name

    @current_name.setter
    def current_name(self, name: Optional[str]):
        self._current_name = name

    def close_all(self):
        with self._lock:
            for name, (client, searcher) in self._cache.items():
                try:
                    searcher.close()
                except Exception:
                    pass
            self._cache.clear()
            self._access_order.clear()
            self._current_name = None


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_cache_size = int(os.environ.get("TROVE_CACHE_SIZE", "3"))
_cache = SearcherCache(max_size=_cache_size)
_job_manager = IngestionJobManager(max_concurrent=1)

# Agent orchestration state
_state_stores: dict[str, tuple[MemoryStore, WorkflowEngine]] = {}
_agent_loader = AgentLoader(TROVE_ROOT / "agents")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_state_db(client_name: Optional[str] = None) -> tuple[MemoryStore, WorkflowEngine]:
    """Get or create MemoryStore + WorkflowEngine for a client.

    If client_name is None, uses a global shared database.
    Both stores share a single SQLite connection and lock.
    """
    key = client_name or "__global__"
    if key in _state_stores:
        return _state_stores[key]

    import sqlite3
    import threading

    if client_name:
        client = load_client(client_name, clients_dir=CLIENTS_DIR)
        db_path = client.index_path / "trove_state.db"
    else:
        global_dir = Path.home() / ".trove"
        global_dir.mkdir(parents=True, exist_ok=True)
        db_path = global_dir / "trove_state.db"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    lock = threading.Lock()

    memory = MemoryStore(db_path, connection=conn, lock=lock)
    workflow = WorkflowEngine(db_path, connection=conn, lock=lock)
    _state_stores[key] = (memory, workflow)
    return memory, workflow


def get_current_client() -> Optional[Client]:
    """Get the currently loaded client."""
    pair = _cache.current
    return pair[0] if pair else None


def get_current_searcher() -> Optional[HybridSearcher]:
    """Get the current searcher instance."""
    pair = _cache.current
    return pair[1] if pair else None


def _create_search_llm_provider(client: Client) -> Optional[LLMProvider]:
    """Create an LLM provider for search-time features (HyDE, multi-query).

    Returns None if provider cannot be created.
    """
    try:
        return create_provider_from_config(client.llm_config)
    except Exception as e:
        logger.warning(f"Failed to create LLM provider for search features: {e}")
        return None


def load_trove_client(client_name: str) -> tuple[Client, HybridSearcher]:
    """Load a Trove client and initialize its searcher.

    Uses the searcher cache — if already loaded, returns cached instance.

    Args:
        client_name: Name of the client to load

    Returns:
        Tuple of (Client, HybridSearcher)

    Raises:
        ValueError: If client not found or has no index
    """
    # Check cache first
    cached = _cache.get(client_name)
    if cached:
        _cache.current_name = client_name
        logger.info(f"Using cached searcher for: {client_name}")
        return cached

    # Load the client
    client = load_client(client_name, clients_dir=CLIENTS_DIR)

    if not client.index_path.exists():
        raise ValueError(f"Client '{client_name}' has no index. Run ingestion first.")

    # Initialize searcher
    searcher = HybridSearcher(
        index_path=str(client.index_path),
        collection_name=client.machine_name,
        bm25_weight=client.search_config.get("bm25_weight", 0.5),
        vector_weight=client.search_config.get("vector_weight", 0.5),
        min_score=client.search_config.get("min_score"),
        tokenizer_config=client.tokenization_config,
        embedding_config=client.embedding_config,
        # Adaptive reranking
        adaptive_rerank=client.search_config.get("adaptive_rerank", True),
        adaptive_rerank_threshold=client.search_config.get("adaptive_rerank_threshold", 0.9),
        adaptive_rerank_depth=client.search_config.get("adaptive_rerank_depth", 15),
        # Merge strategy
        merge_strategy=client.search_config.get("merge_strategy", "weighted"),
        rrf_k=client.search_config.get("rrf_k", 60),
        # MMR diversity
        mmr_lambda=client.search_config.get("mmr_lambda"),
        # Code priority settings (code as source of truth)
        code_priority=client.search_config.get("code_priority", False),
        code_priority_boost=client.search_config.get("code_priority_boost", 0.5),
        cross_reference_code=client.search_config.get("cross_reference_code", False),
        cross_reference_top_k=client.search_config.get("cross_reference_top_k", 3),
        # Reranker backend
        reranker_backend=client.search_config.get("reranker", "local"),
        reranker_endpoint=client.search_config.get("reranker_endpoint"),
        # BM25 parameter tuning
        bm25_k1=client.search_config.get("bm25_k1", 1.5),
        bm25_b=client.search_config.get("bm25_b", 0.75),
        # Score normalization
        score_normalization=client.search_config.get("score_normalization", "minmax"),
        # HyDE
        hyde=client.search_config.get("hyde", False),
        llm_provider=_create_search_llm_provider(client) if (
            client.search_config.get("hyde", False) or
            client.search_config.get("multi_query", False)
        ) else None,
        # Multi-query
        multi_query=client.search_config.get("multi_query", False),
        multi_query_count=client.search_config.get("multi_query_count", 3),
        # HNSW index tuning
        hnsw_ef_construction=client.embedding_config.get("hnsw_ef_construction", 200),
        hnsw_max_neighbors=client.embedding_config.get("hnsw_max_neighbors", 16),
        hnsw_ef_search=client.embedding_config.get("hnsw_ef_search", 100),
        # Search result caching
        cache_ttl=client.search_config.get("cache_ttl", 300),
        cache_max_size=client.search_config.get("cache_max_size", 128),
    )

    _cache.put(client_name, client, searcher)
    _cache.current_name = client_name

    logger.info(f"Loaded client: {client.name} ({client.machine_name})")
    return client, searcher


# ---------------------------------------------------------------------------
# Search Mode Presets
# ---------------------------------------------------------------------------

SEARCH_PRESETS = {
    "precise": {
        "bm25_weight": 0.7,
        "vector_weight": 0.3,
        "merge_strategy": "weighted",
        "mmr_lambda": None,
        "_search_params": {"top_k": 10, "min_score": -1.0},
    },
    "broad": {
        "bm25_weight": 0.3,
        "vector_weight": 0.7,
        "merge_strategy": "weighted",
        "mmr_lambda": None,
        "_search_params": {"top_k": 30, "min_score": None},
    },
    "code_first": {
        "code_priority": True,
        "code_priority_boost": 0.5,
        "cross_reference_code": True,
        "_search_params": {"chunk_types": ["code"]},
    },
    "exhaustive": {
        "merge_strategy": "rrf",
        "adaptive_rerank": False,
        "_search_params": {"top_k": 50, "rerank_top_n": 100},
    },
    "diverse": {
        "mmr_lambda": 0.7,
        "_search_params": {},
    },
}


def _apply_preset(searcher: HybridSearcher, mode: str) -> tuple[dict, dict]:
    """Apply a search preset, returning (saved_attrs, search_params).

    Returns saved_attrs dict to pass to _restore_preset() after search.
    Returns search_params dict to pass as **kwargs to search().
    """
    preset = SEARCH_PRESETS.get(mode)
    if not preset:
        return {}, {}

    saved = {}
    search_params = preset.get("_search_params", {}).copy()

    for key, value in preset.items():
        if key == "_search_params":
            continue
        if hasattr(searcher, key):
            saved[key] = getattr(searcher, key)
            setattr(searcher, key, value)

    return saved, search_params


def _restore_preset(searcher: HybridSearcher, saved_attrs: dict):
    """Restore searcher attributes after preset search."""
    for key, value in saved_attrs.items():
        setattr(searcher, key, value)


# ---------------------------------------------------------------------------
# Context Budget Management
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count using character heuristic (1 token ~ 4 chars)."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Multi-Client RRF Merge
# ---------------------------------------------------------------------------

def _cross_client_rrf_merge(
    client_results: dict[str, list[dict]],
    k: int = 60,
    top_k: int = 20,
) -> list[dict]:
    """Merge results from multiple clients using RRF."""
    rrf_scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict] = {}

    for client_name, results in client_results.items():
        for rank, chunk in enumerate(results):
            chunk_id = f"{client_name}:{chunk.get('id', rank)}"
            chunk["_source_client"] = client_name
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            chunks_by_id[chunk_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [chunks_by_id[cid] for cid in sorted_ids[:top_k]]


# ---------------------------------------------------------------------------
# LLM Provider Creation & Summarization
# ---------------------------------------------------------------------------

def _create_llm_provider(client: Client) -> Optional[LLMProvider]:
    """Create an LLM provider from client config. Returns None if unavailable."""
    llm_config = client.llm_config
    provider_type = llm_config.get("provider", "lmstudio")
    host = llm_config.get("lmstudio_host", "http://localhost:1234")
    model = llm_config.get("lmstudio_model", "local-model")

    try:
        if provider_type == "lmstudio_anthropic":
            return LMStudioAnthropicProvider(host=host, model=model)
        elif provider_type == "lmstudio_native":
            return LMStudioNativeProvider(host=host, model=model)
        elif provider_type == "lmstudio":
            return LMStudioProvider(base_url=f"{host}/v1", model=model)
        elif provider_type == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                return ClaudeProvider(api_key=api_key)
        return None
    except Exception as e:
        logger.warning(f"Failed to create LLM provider: {e}")
        return None


def _summarize_chunk(content: str, query: str, provider: Optional[LLMProvider]) -> str:
    """Summarize a chunk. Falls back to truncation if no LLM."""
    if provider:
        try:
            messages = [{"role": "user", "content": (
                f"Summarize the following text in 2-3 sentences, "
                f"focusing on aspects relevant to the query '{query}':\n\n{content[:2000]}"
            )}]
            return provider.generate(messages, max_tokens=150)
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")

    # Fallback: first 200 chars
    if len(content) > 200:
        return content[:200] + "..."
    return content


# ---------------------------------------------------------------------------
# Result Formatting
# ---------------------------------------------------------------------------

def format_search_results(
    results: list[dict],
    include_content: bool = True,
    max_tokens: Optional[int] = None,
) -> str:
    """Format search results for display.

    Args:
        results: List of search result chunks
        include_content: Whether to include full content
        max_tokens: Maximum token budget. Results are trimmed to fit.

    Returns:
        Formatted string
    """
    if not results:
        return "No results found."

    total_results = len(results)
    lines = [f"Found {total_results} results:\n"]
    token_count = estimate_tokens(lines[0])
    shown = 0

    for i, chunk in enumerate(results, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source_file", "unknown")
        score = chunk.get("_rerank_score", chunk.get("_combined_score", 0))
        chunk_type = metadata.get("chunk_type", "document")
        source_client = chunk.get("_source_client")

        result_lines = [f"## Result {i}"]
        if source_client:
            result_lines.append(f"**Client:** {source_client}")
        result_lines.append(f"**Source:** {source}")
        result_lines.append(f"**Score:** {score:.3f}")
        result_lines.append(f"**Type:** {chunk_type}")

        # Add code-specific metadata
        if chunk_type == "code":
            lang = metadata.get("language", "")
            symbol_type = metadata.get("symbol_type", "")
            symbol_name = metadata.get("symbol_name", "")
            line_start = metadata.get("line_start", "")
            repo = metadata.get("repo_name", "")

            if lang:
                result_lines.append(f"**Language:** {lang}")
            if symbol_type and symbol_name:
                result_lines.append(f"**Symbol:** {symbol_type} `{symbol_name}`")
            if line_start:
                result_lines.append(f"**Lines:** {line_start}-{metadata.get('line_end', line_start)}")
            if repo:
                result_lines.append(f"**Repo:** {repo}")

        if include_content:
            content = chunk.get("content", "")
            if len(content) > 2000:
                content = content[:2000] + "...\n[truncated]"
            if chunk_type == "code":
                result_lines.append(f"\n```{metadata.get('language', '')}\n{content}\n```")
            else:
                result_lines.append(f"\n{content}\n")

        result_lines.append("---")
        result_block = "\n".join(result_lines)
        result_tokens = estimate_tokens(result_block)

        if max_tokens and (token_count + result_tokens) > max_tokens:
            break

        lines.append(result_block)
        token_count += result_tokens
        shown += 1

    if max_tokens:
        lines.append(f"\n**Tokens:** ~{token_count}/{max_tokens} (showing {shown}/{total_results} results)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Perform Search (shared logic for trove_search and trove_search_client)
# ---------------------------------------------------------------------------

def _perform_search(
    searcher: HybridSearcher,
    client: Client,
    query: str,
    top_k: int,
    include_content: bool = True,
    mode: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Shared search logic for trove_search and trove_search_client.

    Returns formatted output string.
    """
    saved_attrs = {}
    extra_params = {}

    # Handle graph search modes
    graph_mode = None
    if mode and mode.startswith("graph_"):
        graph_mode = mode.replace("graph_", "")
        mode = None  # Don't apply as a preset

    if mode:
        saved_attrs, extra_params = _apply_preset(searcher, mode)

    try:
        search_top_k = extra_params.pop("top_k", top_k)
        if graph_mode:
            results = searcher.graph_search(query=query, mode=graph_mode, top_k=search_top_k)
            search_result = SearchResult(results=results, quality=SearchQuality())
        else:
            search_result = searcher.search_with_quality(
                query=query, top_k=search_top_k, **extra_params
            )
    finally:
        if saved_attrs:
            _restore_preset(searcher, saved_attrs)

    q = search_result.quality
    output = f"Searching **{client.name}** for: *{query}*"
    if mode:
        output += f" (mode: {mode})"
    output += "\n\n"
    output += (
        f"**Quality:** avg={q.avg_rerank_score:.2f}, top={q.top_score:.2f}, "
        f"spread={q.score_spread:.2f}, sources={q.unique_sources}, "
        f"candidates={q.total_candidates}, strategy={q.merge_strategy}, "
        f"time={q.timing.get('total', 0):.2f}s\n\n"
    )
    output += format_search_results(
        search_result.results,
        include_content=include_content,
        max_tokens=max_tokens,
    )

    # Append citation reference section if citations enabled
    if client.llm_config.get("citations", False) and search_result.results:
        cmap = build_citation_map(search_result.results)
        output += "\n\n**Citation Map:**\n"
        for num, entry in cmap.entries.items():
            page = entry.get("page")
            page_str = f", p.{page}" if page else ""
            output += f"[{num}] {entry['source']}{page_str}\n"

    return output


def _validate_client_name(name: str | None) -> str | None:
    """Validate client_name argument. Returns error message or None if valid."""
    if name is not None and not _VALID_CLIENT_NAME_RE.match(name):
        return f"Error: Invalid client name '{name}'. Only alphanumeric, hyphens, and underscores are allowed."
    return None


def _get_prompts_api_client():
    """Lazy-import and return a PromptsAPIClient instance."""
    from clients.prompts.api_client import PromptsAPIClient
    return PromptsAPIClient()
