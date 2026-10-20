# Trove Changelog

Completed stages from the Trove roadmap. Each stage was independently valuable and testable. See [ROADMAP.md](ROADMAP.md) for planned future work.

---

## Security Hardening & Architecture Refactor — 2026-04-02

### Security Fixes (CRITICAL + HIGH)
- **`core/sources/git_ops.py`**: Replaced `tempfile.mktemp()` with `mkstemp()` to fix TOCTOU race condition in GIT_ASKPASS token handling
- **`core/search/`**: Replaced unsafe deserialization with JSON for BM25 cache — prevents code execution via tampered cache files
- **`mcp_server/`**: Fixed path traversal check (`str.startswith()` → `Path.is_relative_to()`), added client_name regex validation at MCP entry point
- **`core/memory.py`**: Escaped LIKE wildcards (`%`, `_`) in SQL queries
- **`core/workflow.py`**: Added column whitelist assertion for dynamic SQL construction
- **`core/llm.py`**: Added `is_llm_error()` helper and `LLM_ERROR_PREFIX` for machine-detectable error responses

### Architecture Refactoring
- **`core/search.py` → `core/search/` package**: Split 2,344-line monolith into 6 focused modules (types, tokenizer, cache, merge, searcher). All imports backwards-compatible via `__init__.py` re-exports.
- **`mcp_server/server.py` split**: Extracted 3,346-line file into `shared.py` (578 lines) + 9 domain-grouped handler modules under `handlers/` (clients, search, ingest, graph, memory, workflows, agents, prompts, sources). `call_tool` is now a dispatch table lookup. Server reduced from 3,346 to 1,138 lines (66% reduction).

### Thread Safety
- **SearcherCache**: Added `threading.Lock` for concurrent MCP tool call safety
- **Multi-query search**: Replaced instance state mutation with `_skip_multi_query` parameter (thread-safe recursion prevention)
- **Chunk scoring**: All merge/rerank/boost methods now copy chunks before adding score fields (prevents cross-search score leakage)

### OSS Infrastructure
- **`pyproject.toml`**: Package metadata, dependencies, ruff config, pytest config, console entry points
- **`.github/workflows/test.yml`**: CI with pytest on Python 3.10/3.11/3.12 + ruff lint + format check
- **`.pre-commit-config.yaml`**: Ruff lint/format, trailing whitespace, large file detection, secret detection
- **`CONTRIBUTING.md`**: Dev setup, testing, code style, PR process
- **`Dockerfile` + `.dockerignore`**: Python 3.12-slim with system deps for document parsing
- **README.md**: Fixed misleading license claim (now accurately reflects BSL 1.1), added Python 3.10+ requirement

### Gitignore
- Added `clients/*/sources/`, `.omc/`, `.playwright-mcp/` exclusions

**Tests:** 724 passed, 0 failed.

---

## Ingestion Pipeline Optimization — Completed 2026-03-09

**Goal:** Maximize GPU utilization during ingestion by replacing serial per-file embedding with a producer-consumer pipeline. Add Ctrl+C safety with progressive indexing.

### Producer-Consumer Pipeline (`core/pipeline.py`)

New `IngestionPipeline` class decouples file parsing (CPU, threaded) from embedding (GPU, batched) from indexing (ChromaDB, progressive).

- `run_documents(files, ...)` — parallel parse via ThreadPoolExecutor → bounded buffer → GPU embed → ChromaDB index
- `run_code(chunks)` — pre-chunked code → buffer → GPU embed → ChromaDB index
- Buffer bounded at `embed_batch_size` (default 512). Each flush embeds + indexes immediately into ChromaDB.
- SIGINT handler: first Ctrl+C flushes current buffer and stops cleanly, second force-exits
- Resume via ChromaDB query: `get_indexed_source_paths()` returns already-indexed `source_path` values, skipping completed files
- Progress logging every 10 files with buffer/flush stats and chunks/second rate

### Ingestion Integration (`core/ingest.py`)

- `ingest_directory()` gains `pipeline_searcher` param — when provided, uses IngestionPipeline instead of serial per-file loop
- `ingest_code()` gains `pipeline_searcher` param — when provided, uses `pipeline.run_code()` for progressive indexing
- Added `ingest_mbox_chunks_only()` for pipeline-compatible mbox parsing (parse without embed)
- Removed per-file `_clear_gpu_memory()` calls (GPU memory managed by batch lifecycle)
- `DEFAULT_EMBED_BATCH_SIZE` increased from 32 to 512

### Script Updates (`scripts/ingest.py`)

- `ingest_client_documents()` creates HybridSearcher before ingestion, passes as `pipeline_searcher`
- `ingest_client_code_files()` creates HybridSearcher first, passes to `ingest_client_code()`
- `incremental_reindex()` uses IngestionPipeline for new+modified files instead of per-file embedding loop
- BM25 index rebuilt once at end of ingestion (not per-file)
- Removed legacy progress display and checkpoint logic

### Code File Collection Speedup

Replaced slow recursive Python directory walking with fast alternatives for the pre-pipeline phase (file discovery + change detection).

- **`git ls-files` fast path** (`core/code/git_utils.py`): New `list_repo_files()` uses git index for near-instant file listing in git repos. `walk_code_files()` tries git fast path first for each detected repo, falls back to directory walk for non-git dirs.
- **`os.scandir()` fallback** (`core/code/git_utils.py`): Replaced `sorted(Path.iterdir())` with `os.scandir()` for non-git directories. Uses cached `DirEntry.stat()` and avoids `Path` object creation overhead.
- **Parallel change detection** (`core/code/incremental.py`): `get_changes()` uses `ThreadPoolExecutor` (8 workers) for parallel stat/hash checks on large file sets.
- **Repo-aware collection** (`core/code/chunker.py`, `core/ingest.py`): Repos detected before file collection so `collect_files()` can use git fast path. Works for all clients with git repos in code paths.

### Tests

6 new tests in `TestIngestionPipeline`:
- `test_flush_buffer_embeds_and_indexes`
- `test_buffer_auto_flushes_at_batch_size`
- `test_run_code_progressive_indexing`
- `test_resume_skips_already_indexed`
- `test_empty_buffer_flush_is_noop`
- `test_get_stats_returns_correct_shape`

---

## Stage 0: Source Connectors — Completed 2026-03-09

**Goal:** Automated document acquisition from GitHub repositories and websites. Trove discovers, fetches, converts, and keeps documents up to date from external sources.

### 0.1 GitHub Connector (`core/sources/github_connector.py`)

Discover repos from GitHub accounts/orgs, clone them, keep them synced.

- `GitHubClient` — HTTP client with rate limiting, pagination, retries, auth
- `GitHubSourceConnector` — discover → clone → fetch → filter workflow
- `GitHubSourceConfig` / `RepoInfo` — configuration and metadata models
- `git_ops.py` — clone/fetch operations with GIT_ASKPASS token authentication (shell-quoted)
- Token via env vars only (`token_env: GITHUB_TOKEN`)
- Include/exclude repo filtering, fork/private toggles

### 0.2 Web Crawler (`core/sources/web_crawler.py`)

Crawl websites and convert pages to Markdown for Trove indexing.

- `WebSourceConnector` — BFS crawl with scope control (`stay_in_path`, `max_pages`)
- `web_converter.py` — HTML → clean Markdown (markdownify + trafilatura fallback)
- `web_cache.py` — incremental crawling cache (ETag, Last-Modified, content hash)
- Junk content removal (ads, newsletters, sidebars, navigation, related articles)
- Non-content path skipping (login, cart, admin, etc.)
- YAML frontmatter on crawled pages (title, url, source)

### 0.3 Shared Infrastructure

- `BaseSourceConnector` ABC with `sync(target_dir) -> SyncResult` and `status() -> dict`
- `SyncResult` dataclass with files_added/updated/removed/errors
- Source files at `clients/{name}/sources/{github|web}/`
- `core/sources/__init__.py` — clean re-exports

### 0.4 MCP Integration

4 new MCP tools in `mcp_server/server.py`:
- `trove_connect_github` — connect a GitHub source and clone repos
- `trove_crawl_web` — crawl a website and save as Markdown
- `trove_sync_sources` — sync all configured sources for a client
- `trove_list_sources` — list configured sources with status

### 0.5 Ingest Pipeline Integration

- `client.py`: `sources_path`, `sources_config`, `has_sources` properties; `"sources"` in known config keys
- `scripts/ingest.py`: `sync_client_sources()` function, `_should_sync_on_ingest()` helper
- Auto-scans `sources/` directory alongside `documents/` during ingestion
- File manifest includes both directories for incremental reindexing
- `sync_on_ingest` config flag for automatic sync before reindexing
- `--sync-sources` CLI flag for explicit sync

### 0.6 CLI (`scripts/sources.py`)

- `connect github` — connect GitHub org/user/repo with filtering
- `crawl` — crawl a website with mode/scope/page-limit control
- `sync` — sync all configured sources for a client (or all clients)
- `list` — show configured sources and on-disk state

### 0.7 Tests

75+ tests across two test classes:
- `TestSourceConnectors` — connector unit tests, config, models, conversion
- `TestSourceConnectorErrorPaths` — error handling, edge cases, validation
- `TestSourceIngestIntegration` — pipeline integration tests

---

## Bug Fixes — 2026-03-08

### Knowledge graph incremental build fix

The incremental graph builder determined "already processed" chunks by scanning `source_chunk_ids` from entities. Chunks that were processed but yielded zero entities were never tracked, causing them to be reprocessed on every run. This caused large clients (e.g. a large client) to re-process thousands of chunks unnecessarily.

- **`core/graph/store.py`**: Added `_processed_chunk_ids` set to `KnowledgeGraph`, with `processed_chunk_ids` property, `mark_chunks_processed()` method. Persisted in `save()`, restored in `load()` with backward-compatible backfill from entity/relationship sources for old-format graphs. Included in `clear()` and `stats`.
- **`core/graph/extractor.py`**: Fixed callback signature inconsistency — empty batches now pass 3 params (`progress, entities, relationships`) matching normal batches.
- **`scripts/build_graph.py`**: Uses `graph.processed_chunk_ids` instead of scanning `_entities`. Calls `graph.mark_chunks_processed()` per batch so zero-entity chunks are tracked.
- **`mcp_server/server.py`**: Same fix for MCP `trove_build_graph` handler.
- **Tests**: 6 new tests for processed chunk tracking (roundtrip, backfill, clear, stats).

---

## Stage 11: Agent Orchestration — Completed 2026-03-02

**Goal:** Add persistent memory, workflow DAGs, and agent templates inspired by Ruflo (claude-flow) concepts, adapted to Trove's architecture.

### 11.1 Persistent Memory (`core/memory.py`)

SQLite-backed memory store for context that survives across Claude Code sessions.

- `MemoryStore` class with namespace/key-based storage, CRUD operations
- Optional vector search via SentenceTransformer embeddings, LIKE-based fallback
- TTL support for ephemeral context with automatic cleanup
- Per-client scoping (`clients/{name}/index/trove_state.db`) or global (`~/.trove/trove_state.db`)
- Thread-safe with threading lock for concurrent MCP access

### 11.2 Workflow DAGs (`core/workflow.py`)

Dependency-ordered task pipeline tracking for multi-step operations.

- `WorkflowEngine` class with DAG validation via Kahn's topological sort
- Circular dependency detection on creation
- Step status tracking: pending, running, completed, failed, skipped
- `next_steps()` returns steps where all dependencies are completed
- Auto-cascading workflow status (running when any step running, completed when all done)
- Shares SQLite database with MemoryStore (separate tables)

### 11.3 Agent Templates (`core/agents.py` + `agents/`)

YAML-defined reusable prompt configurations for specialized roles.

- `AgentLoader` class for listing, getting, and searching agent templates
- 6 starter agents: security-reviewer, code-reviewer, performance-analyzer, documentation-writer, test-generator, architecture-analyst
- Case-insensitive matching on name field and filename slug
- Substring search across name, description, and tags with relevance scoring

### 11.4 MCP Tools (10 new)

- `trove_memory_store` — store a memory entry
- `trove_memory_search` — semantic/keyword search over memories
- `trove_memory_list` — list stored memories
- `trove_memory_delete` — delete a memory
- `trove_workflow_create` — create a workflow with DAG dependencies
- `trove_workflow_status` — get workflow progress
- `trove_workflow_update` — update step status
- `trove_workflow_list` — list workflows
- `trove_agents_list` — list agent templates
- `trove_agents_get` — get full agent config

### 11.5 Local Agent Runner (`core/runner.py` + `scripts/run_agent.py`)

Run Trove agents end-to-end using local LLM (LM Studio) — offload low-level tasks from cloud credits.

- `AgentRunner` class: search Trove for context, inject agent system prompt, generate response via LLM
- `AgentResult` dataclass: response, sources, search_quality, timing, agent_name, query
- Conversation history support for multi-turn interactions
- Optional memory persistence (stores results in MemoryStore under `agent:{filename}` namespace)
- Interactive REPL via `AgentRunner.chat()`
- New `scripts/run_agent.py` CLI: `--agent`, `--client`, `--interactive`, `--json`, `--list-agents`, `--list-clients`
- `create_provider_from_config()` factory in `core/llm.py` — shared across MCP server, CLI, and build_graph.py
- `scripts/build_graph.py` refactored to use shared factory (33 lines → 4 lines)

### 11.6 MCP Tool: `trove_run_agent`

- Delegates tasks to local LM Studio via AgentRunner
- Parameters: agent_name (required), query (required), client_name (optional), top_k, store_memory
- Returns formatted markdown with response, sources, and timing

### Bug Fixes

- **`core/search.py`**: Fixed `get_response()` → `generate()` calls in HyDE and multi-query (was `AttributeError` at runtime)
- **`mcp_server/server.py`**: Fixed `_create_search_llm_provider()` passing invalid `max_tokens` to provider constructors
- **`core/runner.py`**: Fixed `value=` → `content=` parameter in `MemoryStore.store()` call (was `TypeError` at runtime)
- **`core/workflow.py`**: Fixed `next_steps()` to treat skipped steps as terminal (was blocking dependents)
- **`core/llm.py`**: Added empty response guards for LMStudioProvider and ClaudeProvider (was `IndexError` risk)

### Cross-cutting

- 96 new tests (447 total)
- No new external dependencies (sqlite3 stdlib, existing yaml and numpy)

---

## Stage 10: Scale & Performance — Completed 2026-03-01

**Goal:** Make Trove performant at scale by tuning HNSW index parameters and adding in-memory search result caching.

### 10.1 HNSW Index Tuning

Configurable HNSW parameters for ChromaDB collections.

- Three new config keys: `embedding.hnsw_ef_construction` (build-time, default 200), `embedding.hnsw_max_neighbors` (build-time M, default 16), `embedding.hnsw_ef_search` (query-time, default 100)
- Build-time params applied to ChromaDB collection metadata on creation
- Query-time `ef_search` applied per-query without reindexing
- Defaults match ChromaDB defaults — no breaking changes for existing clients

### 10.2 Search Result Caching

In-memory LRU cache with TTL for full search pipeline results.

- New `SearchCache` class with `OrderedDict`-based LRU eviction and TTL expiry
- Caches full pipeline output (after merge + rerank + MMR)
- Cache key: `(query, top_k, min_score, search_config_hash)`
- Configurable: `search.cache_ttl: 300` (seconds, 0 = disabled), `search.cache_max_size: 128`
- Automatic invalidation on BM25 index rebuild, ingest, and close
- `SearchQuality.cache_hit` and `SearchQuality.cache_size` fields for monitoring

### Cross-cutting

- 23 new tests (292 total)
- Binary quantization and Matryoshka embeddings deferred (ChromaDB doesn't support natively, marginal gain at current scale)

---

## Stage 9: Embedding & Indexing Quality — Completed 2026-03-01

**Goal:** Improve the quality of indexed data. Requires reindexing but produces permanently better search results.

### 9.1 Contextual Embeddings

Prepend document context to chunks before embedding so each chunk retains awareness of its source document.

- New `build_context_prefix()` function in `core/ingest.py`
- `create_embeddings()` gains `context_strings` parameter — prepends context to text before encoding
- `ingest_document()` gains `contextualize` parameter — builds context from metadata and passes to embedder
- ChromaDB stores raw content, BM25 indexes raw content — only the embedding vector is contextualized
- Chunk metadata gains `"contextualized": true` flag
- Configurable: `embedding.contextualize: true` in `client.yaml` (default: `false`)

### 9.2 Recursive Chunking

Add recursive character splitting as a third chunking strategy alongside title and semantic.

- New `chunk_by_recursive()` function in `core/chunking.py`
- Tries separators in priority order: `\n\n` → `\n` → `. ` → ` ` (configurable)
- Falls back to next separator when pieces still exceed max_chars
- Overlap applied between consecutive chunks
- Configurable: `chunking.strategy: "recursive"`, `chunking.separators: [...]` in `client.yaml`
- `ChunkingConfig` dataclass updated with `separators` field

### 9.3 Embedding Model Benchmark

Standalone CLI to benchmark embedding models against eval datasets.

- New `scripts/benchmark_embeddings.py` CLI
- Benchmarks: MRR, Hit@5, Hit@10, nDCG, Precision@5, embed time, total time
- `_KNOWN_EMBEDDING_DIMS` updated with `BAAI/bge-m3` (1024d), `nomic-ai/nomic-embed-text-v1.5` (768d)
- Console table output + optional JSON export

### Cross-cutting

- 28 new tests (269 total)
- `embedding.contextualize` config key added to defaults
- `chunking.separators` config key added to defaults

---

## Stage 8: Advanced Query Processing — Completed 2026-02-28

**Goal:** Improve retrieval quality at query time — no reindexing required. All features opt-in via `client.yaml`.

### 8.1 HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer via LLM, embed it alongside the raw query for vector search. Improves recall for vague or short queries.

- New `_generate_hypothetical_document()` method in `HybridSearcher`
- Augment-not-replace: embeds BOTH raw query AND hypothetical doc, merges results keeping higher score per chunk
- BM25 still uses raw query (keyword matching doesn't benefit from hypothetical answers)
- Configurable: `search.hyde: true` in `client.yaml` (default: `false`)
- Graceful degradation: if LLM unavailable or fails, falls back to raw query silently
- LLM provider auto-created in MCP server when HyDE or multi-query is enabled
- `SearchQuality.hyde_applied` tracks whether HyDE was used

### 8.2 Multi-Query / RAG-Fusion

Generate query variants via LLM, search each through the full pipeline, merge via RRF.

- New `generate_query_variants()` in `core/refinement.py` — upfront diverse reformulations
- New `_multi_query_search()` method: runs each variant through full BM25+vector+merge+rerank pipeline
- RRF merge across all variant results using configurable `rrf_k`
- Recursion prevention: `multi_query` temporarily disabled during inner search calls
- Composes with HyDE (each variant uses HyDE if enabled)
- Configurable: `search.multi_query: true`, `search.multi_query_count: 3` (range 2-7)
- `SearchQuality.multi_query_applied` and `multi_query_variants` track usage

### 8.3 BM25 Parameter Tuning

Configurable BM25 `k1` and `b` parameters per client.

- `search.bm25_k1` (default: 1.5) — term frequency saturation
- `search.bm25_b` (default: 0.75) — document length normalization
- Passed to `BM25Okapi` constructor (previously hardcoded defaults)
- BM25 params included in corpus hash fingerprint — changing k1/b auto-invalidates cache
- Validation: `k1 > 0`, `0 <= b <= 1`

### 8.4 Score Normalization Strategies

Configurable normalization for the weighted merge strategy.

- `search.score_normalization: "minmax"` (default) — divide by max, 0-1 range (current behavior)
- `"zscore"` — z-score normalization, shifted so min=0; falls back to minmax if std=0 or < 2 results
- `"raw"` — no normalization, pass scores through unchanged
- `SearchQuality.score_normalization` tracks which strategy was used
- Applied only to weighted merge (RRF is rank-based and doesn't need normalization)

### Cross-cutting

- `SearchQuality` dataclass extended with `score_normalization`, `hyde_applied`, `multi_query_applied`, `multi_query_variants`
- MCP server `load_trove_client()` passes all new params to `HybridSearcher`
- New `_create_search_llm_provider()` helper creates LLM provider when HyDE or multi-query enabled
- Config validation for all new keys in `validate_config()`
- 38 new tests covering all sub-features

---

## Stage 1: Search Quality & Robustness — Completed 2026-02-15

**Goal:** Make search results measurably better and more diverse. Fix the foundation so everything built on top is solid.

**Why first:** Search quality is Trove's core value proposition. Every other improvement (MCP, evaluation, knowledge graph) depends on good retrieval. This stage also adds a CLI search tool needed to test all future changes.

### 1.1 Reciprocal Rank Fusion (RRF)

Added RRF as an alternative to weighted score merging. RRF uses rank position instead of raw scores, making it robust to score distribution differences between search methods.

- New `_rrf_merge()` method in `HybridSearcher`
- Configurable via `search.merge_strategy: "rrf"` in `client.yaml` (default: `"weighted"`)
- RRF constant configurable via `search.rrf_k` (default: 60)
- RRF scores (~0.003-0.033) naturally don't trigger adaptive reranking — correct by design

### 1.2 Maximal Marginal Relevance (MMR)

Added MMR as a post-reranking diversity step. Penalizes similarity to already-selected results, producing more diverse context for the LLM.

- New `_mmr_diversify()` method using numpy cosine similarity on embeddings
- Configurable via `search.mmr_lambda` in `client.yaml` (None=disabled, 0.0=max diversity, 1.0=pure relevance)
- Applied after cross-encoder reranking, before min_score filtering

### 1.3 Search Quality Indicators

Search results now include quality metadata via `SearchQuality` and `SearchResult` dataclasses.

- Metrics: `avg_rerank_score`, `top_score`, `score_spread`, `unique_sources`, `total_candidates`
- Strategy tracking: `merge_strategy`, `mmr_applied`, `adaptive_rerank_applied`
- Timing breakdown: `parallel_search`, `merge`, `rerank`, `mmr`, `total`
- New `search_with_quality()` method (does not break `search()` return type)
- MCP `trove_search` now shows quality summary line before results

### 1.4 Configurable Adaptive Reranking

Made adaptive reranking configurable instead of hardcoded.

- `search.adaptive_rerank` (default: true) — enable/disable
- `search.adaptive_rerank_threshold` (default: 0.9) — combined score threshold
- `search.adaptive_rerank_depth` (default: 15) — reduced rerank depth when triggered

### 1.5 CLI Search Tool

New `scripts/search.py` for testing search quality from the command line.

- Supports all search options: `--merge-strategy`, `--mmr-lambda`, `--no-adaptive`, `--top-k`
- Output modes: text (default), `--json`, `--verbose` (timing/metadata), `--full` (no truncation)
- `--list` to show available clients

### 1.6 Critical Error Handling Fixes

- `core/search.py`: BM25/vector futures wrapped in try/except — one failing returns empty instead of crashing; both failing returns `[]` early
- `core/search.py`: `normalize()` guards against all-zero scores explicitly
- `core/llm.py`: Logs warning when context is truncated (shows included/total chunks and chars)
- Note: `advanced_tokenize()` retained — used by `conversation_docs.py`

**Files changed:** `core/search.py`, `core/client.py`, `core/llm.py`, `mcp_server/server.py`, `tests/test_core.py`
**Files added:** `scripts/search.py`
**Tests:** 52 total (20 new), all passing

---

## Stage 2: MCP Enhancement — Completed 2026-02-15

**Goal:** Make Trove's MCP interface smarter so Claude can use it more effectively.

**Why second:** Once search quality is solid, we enhance the interface. These changes make Claude a better consumer of Trove results.

### 2.1 Searcher Cache

Replaced single-client globals with an LRU cache (`SearcherCache`) holding multiple loaded searchers. Switching between clients is now instant if the client was recently used.

- `SearcherCache` class with LRU eviction, configurable size via `TRAG_CACHE_SIZE` env var (default: 3)
- Backward compatible: `get_current_client()` / `get_current_searcher()` still work unchanged
- Cached searchers are properly closed on eviction and cleanup

### 2.2 Multi-Client Search

New MCP tool: `trove_search_multi(query, clients=[], top_k=20, mode=..., max_tokens=...)`

- When `clients` is empty, searches all clients with indices
- Cross-client RRF merge: each client's result list is a separate rank list
- Each result tagged with `_source_client` for source attribution
- Per-client quality summary in output

### 2.3 Search Mode Presets

Added `mode` parameter to `trove_search`, `trove_search_client`, and `trove_search_multi`.

- `"precise"` — High BM25 weight (0.7), strict min_score (-1.0), top 10
- `"broad"` — High vector weight (0.7), no min_score, top 30
- `"code_first"` — Boost code chunks, enable cross-reference
- `"exhaustive"` — RRF merge, no adaptive rerank, top 50, rerank depth 100
- `"diverse"` — Enable MMR diversity (lambda=0.7)
- Presets temporarily override searcher attributes; restored after each search

### 2.4 Context Budget Management

Added `max_tokens` parameter to `trove_search`, `trove_search_client`, and `trove_search_multi`.

- Token estimation via character heuristic (1 token ≈ 4 chars)
- Results are trimmed to fit within budget
- Output includes token summary: `Tokens: ~used/budget (showing N/total results)`

### 2.5 Result Summarization Tool

New MCP tool: `trove_summarize(query, top_k=5, client_name=...)`

- Searches and returns 2-3 sentence summaries per result
- Uses LLM from client config (LM Studio or Claude) for intelligent summarization
- Falls back to first 200 chars if no LLM available
- Useful for getting an overview before diving deep with full search

**Files changed:** `mcp_server/server.py`, `tests/test_core.py`, `ROADMAP.md`, `CLAUDE.md`
**Tests:** 81 total (29 new), all passing

---

## Stage 3: Evaluation Framework — Completed 2026-02-15

**Goal:** Measure search quality so improvements can be validated objectively.

**Why third:** With better search (Stage 1) and better MCP tools (Stage 2) in place, we need to measure whether changes actually help. This enables data-driven decisions for all future stages.

### 3.1 Test Dataset Format

Per-client test dataset format for evaluation.

- Format: `clients/{name}/tests/eval_dataset.json`
- Fields: `id`, `query`, `expected_chunks` (substring patterns), `ground_truth` (reference answer), `tags` (for filtering)
- Seed dataset: 5 test cases for the `trag` client
- Supports retrieval-only tests (do we find the right chunks?) and end-to-end tests (does the answer match?)

### 3.2 Retrieval Metrics

Standard information retrieval metrics in `core/evaluation.py`.

- Metrics: MRR (Mean Reciprocal Rank), Hit@k (k=1,5,10), Precision@k (k=5,10), nDCG@10, Context Recall
- Chunk matching: case-insensitive substring matching against `expected_chunks` patterns
- `compute_retrieval_metrics()` computes all metrics in one call

### 3.3 RAG Quality Metrics (LLM-as-Judge)

RAGAS-style quality metrics using LLM-as-judge in `core/evaluation.py`.

- **Faithfulness** — Is the answer grounded in retrieved context?
- **Answer Relevance** — Does the answer address the question?
- **Context Precision** — How much retrieved context was useful? (requires ground truth)
- Structured prompts with 0-1 scoring scale
- JSON response parsing with markdown code block handling
- Retry logic with graceful fallback on LLM errors
- All LLM calls go through `core/llm.py` providers

### 3.4 Evaluation CLI

`scripts/evaluate.py` — command-line evaluation tool.

- `--client` — Client to evaluate (required)
- `--tags` — Filter test cases by tag
- `--rag` — Enable LLM-as-judge RAG metrics
- `--compare "weighted,rrf"` — A/B comparison of merge strategies
- `--json` — Structured JSON output
- `--output` — Write markdown report to file
- `--verbose` — Show per-query details
- `--list` — List clients with eval datasets

**Files added:** `core/evaluation.py`, `scripts/evaluate.py`, `clients/NAME/tests/eval_dataset.json`
**Files changed:** `tests/test_core.py`, `ROADMAP.md`, `CLAUDE.md`, `README.md`
**Tests:** 98 total (17 new), all passing

---

## Stage 4: Codebase Health — Completed 2026-02-15

**Goal:** Pay down technical debt and improve reliability. Not glamorous, but prevents future features from being built on shaky ground.

**Why fourth:** By now we've added significant new functionality. Before adding more, clean up the foundation.

### 4.1 Input Validation & Security

- Query sanitization: empty/whitespace rejection, control char stripping, 10,000 char length limit
- Applied to `search()`, `fast_search()`, and `search_with_quality()` via `_sanitize_query()`
- Deferred: path traversal validation, API key detection (not needed for local-first architecture)

### 4.2 Error Handling Cleanup

- Narrowed `except Exception` to specific types in `core/conversation.py` (7 catches), `core/conversation_docs.py` (5 catches), `core/audit_trail.py` (5 catches)
- Fixed 3 silent `except: pass` failures — now log warnings with error details
- Added `_retry_with_backoff()` to `core/llm.py` with exponential backoff (max 2 retries, 1s/2s delays)
- Retry applied to all 4 LLM providers for transient connection/timeout errors
- Kept broad exceptions in `core/llm.py` and `core/search.py` where failure modes are unpredictable (HTTP/LLM/ChromaDB)

### 4.3 Logging & Observability — Deferred

Already adequate: all modules use `logging.getLogger(__name__)`, no `print()` calls, timing is in `SearchQuality`. Not worth the churn.

### 4.4 BM25 Cache Improvements — Deferred

Already adequate: cache includes tokenizer version hash, corruption detection exists. LRU eviction and `--clear-cache` are nice-to-haves for later.

### 4.5 Configuration Validation

- `validate_config()` in `core/client.py` checks: required fields, type checks, value ranges, unknown key detection, embedding dimension matching
- Called from `load_client()` — warnings are logged but don't block loading (backwards compatible)
- Catches typos like `bm25_weght` that would silently fall through to defaults

### 4.6 Test Coverage

- 14 new tests: config validation (6), input validation (4), LLM retry (3), integration (1)
- Integration test loads real `trag` client, runs a search, verifies result structure
- Total: 112 tests, all passing

**Files changed:** `core/client.py`, `core/search.py`, `core/llm.py`, `core/conversation.py`, `core/conversation_docs.py`, `core/audit_trail.py`, `tests/test_core.py`, `ROADMAP.md`, `CLAUDE.md`
**Tests:** 112 total (14 new), all passing

---

## Stage 5: Response Quality — Completed 2026-02-16

**Goal:** Make LLM answers more trustworthy, accurate, and well-sourced. All query-time improvements — no reindexing required.

**Why fifth:** Search quality is solid (Stage 1), MCP is smart (Stage 2), we can measure quality (Stage 3), and the codebase is clean (Stage 4). Before improving the data pipeline, make the most of what we already retrieve.

**Informed by:** Citation systems in `notebook-lm-clone` and `open-notebook`, context evaluation in `context-engineering-workflow`, reranker flexibility in LightRAG, token budgeting in LightRAG's `QueryParam` — all from patchy631 repos.

### 5.1 Inline Citations & Source Attribution

Added numbered citations `[1]`, `[2]` in LLM responses that reference specific retrieved chunks.

- `CitationMap` dataclass and `build_citation_map()` in `core/llm.py`
- `parse_citations_from_response()` extracts used citations from LLM output
- `CITATION_SYSTEM_PROMPT_SUFFIX` instructs LLM to use `[N]` format
- `build_rag_prompt_with_citations()` convenience function
- Citation reference footer rendered in UI after response
- Citation map appended in MCP `_perform_search()` output
- Configurable per client: `llm.citations: true` in `client.yaml`

### 5.2 Context Relevance Pre-Filtering

Filter out marginally relevant chunks before sending context to the LLM, reducing noise.

- After reference resolution, chunks below rerank score threshold are removed
- Filtered count logged in search_log and processing_log
- `context_filtered_count` field in `QueryAuditEntry`
- Configurable: `llm.context_filter: true`, `llm.context_filter_threshold: 0.1` in `client.yaml`

### 5.3 Configurable Reranker Backend

Support API-based rerankers alongside the local cross-encoder.

- New `core/reranker.py`: `Reranker` protocol, `LocalReranker`, `APIReranker`, `FallbackReranker`
- `create_reranker()` factory function
- `HybridSearcher` uses `create_reranker()` instead of direct `CrossEncoder`
- API reranker uses Cohere-compatible `POST /v1/rerank` endpoint
- `FallbackReranker` auto-falls back to local on API failure
- Configurable in `client.yaml`:
  ```yaml
  search:
    reranker: "local"              # "local" or "api"
    reranker_model: "BAAI/bge-reranker-v2-m3"
    reranker_endpoint: "http://localhost:8000"
  ```

### 5.4 Fine-Grained Token Budgeting

Allocate separate character budgets for different context source types.

- `categorize_chunks()` — splits chunks into doc/code/reference categories
- `allocate_budgets()` — proportional allocation with redistribution for empty categories
- `build_rag_prompt()` extended with `doc_budget_ratio`, `code_budget_ratio`, `reference_budget_ratio` params
- Configurable in `client.yaml`:
  ```yaml
  llm:
    doc_budget_ratio: 0.6          # 60% for documents
    code_budget_ratio: 0.3         # 30% for code
    reference_budget_ratio: 0.1    # 10% for resolved references
  ```

**Files added:** `core/reranker.py`
**Files changed:** `core/llm.py`, `core/search.py`, `core/client.py`, `core/audit_schema.py`, `mcp_server/server.py`, `tests/test_core.py`
**Tests:** 140 total (28 new), all passing

---

## Stage 6: Ingestion Improvements — Completed 2026-02-16

**Goal:** Make document ingestion more robust, resumable, and accessible via MCP. Add new chunking and parsing strategies.

**Why sixth:** Response quality is polished (Stage 5). Now improve the data pipeline that feeds search and retrieval.

### 6.1 MCP Ingestion Tools

Allow Claude to trigger document ingestion during a session.

- New MCP tools:
  - `trove_ingest_file(client, file_path)` — Index a single file (synchronous)
  - `trove_ingest_url(client, url)` — Fetch and index a URL (synchronous, 30s timeout, 50MB max)
  - `trove_reindex_client(client)` — Trigger full rebuild (background thread, returns job ID)
  - `trove_ingest_status(job_id, client_name)` — Check ingestion progress/status
- `IngestionJobManager` enforces max 1 concurrent job
- SearcherCache invalidated on successful ingestion
- Changes: `mcp_server/server.py` (4 new tools), new `core/ingest_job.py`

### 6.2 Chunk Quality Validation & Position Metadata

Filter out low-quality chunks during ingestion and extract Unstructured metadata.

- `validate_chunk(content, min_chars=50)` — rejects too-short and non-alphanumeric chunks
- Content-hash deduplication via MD5 in `chunk_elements()`
- Skip stats logged: `too_short`, `no_alphanumeric`, `duplicate`
- Extracts Unstructured metadata: `page_number`, `parent_id`, `is_continuation`, `content_hash`
- Changes: `core/ingest.py` (`validate_chunk()`, updated `chunk_elements()`)

### 6.3 Progress Tracking & Resumable Ingestion

- `IngestionProgress` dataclass with percent, ETA, status tracking
- `IngestionCheckpoint` with save/load for resumable ingestion
- `ingest_directory()` now accepts `progress_callback` and `resume_from` parameters
- `scripts/ingest.py --resume` flag to resume from last checkpoint
- Progress display with ETA during CLI ingestion
- Checkpoint auto-removed on successful completion
- Changes: new `core/ingest_progress.py`, `core/ingest.py`, `scripts/ingest.py`

### 6.4 Alternative Chunking Strategies

Offer semantic chunking alongside the existing title-based chunking.

- **Title-based** (default): Unstructured's `chunk_by_title`, splits by section headings
- **Semantic**: Embedding-based boundary detection — cosine similarity between consecutive sentences, splits where similarity drops below threshold, merges short segments
- `chunk_elements()` accepts `chunking_config` dict to select strategy
- Client config property: `client.chunking_config`
- Configurable per client in `client.yaml`:
  ```yaml
  chunking:
    strategy: "title"         # "title" (default) or "semantic"
    semantic_threshold: 0.5   # similarity threshold for semantic chunking
    max_chunk_size: 1500
    min_chunk_size: 100
  ```
- Changes: new `core/chunking.py`, `core/ingest.py`, `core/client.py`

### 6.5 Better Document Parsing

Improve parsing for HTML and email.

- Replaced regex-based `_strip_html()` with BeautifulSoup (`html.parser`)
- Added `_detect_charset()` with declared -> chardet -> utf-8 fallback chain
- Email body extraction now uses `_detect_charset()` for all charset resolution
- Docling evaluation deferred (large dependency ~500MB+, uncertain benefit)
- Changes: `core/ingest.py` (`_strip_html()` rewrite, `_detect_charset()`, `_extract_email_body()` updated)

**Files added:** `core/ingest_progress.py`, `core/ingest_job.py`, `core/chunking.py`
**Files changed:** `core/ingest.py`, `core/client.py`, `mcp_server/server.py`, `scripts/ingest.py`, `tests/test_core.py`
**Tests:** 170 total (30 new), all passing

---

## Stage 7: Knowledge Graph — Completed 2026-02-16

**Goal:** Add a knowledge graph layer for multi-hop reasoning and relationship-aware search. Informed by LightRAG and Graphiti patterns from evaluated codebases.

**Why last:** This is the most ambitious feature. It requires solid search, evaluation, and ingestion foundations. It's also optional — many use cases work fine without it.

### 7.1 Entity Extraction

Extract entities from chunks using the local LLM.

- Structured prompt: "Extract entities (people, organizations, concepts, locations) from this text. Return as JSON."
- Store entities with: name, type, description, source_chunk_ids, aliases
- Opt-in per client: `graph.enabled: true` in `client.yaml`
- Entity type filtering (only configured types kept)
- New `core/graph/extractor.py`

### 7.2 Relationship Extraction

Extract relationships between entities (combined with 7.1 in a single LLM call).

- Store relationships: source_entity, target_entity, relationship_type, description, weight, source_chunk_ids
- Batch extraction: multiple chunks per LLM call for efficiency
- Merge/dedup: duplicate entities and relationships are merged automatically
- `core/graph/extractor.py`

### 7.3 Graph Storage

Store and query the knowledge graph.

- NetworkX for in-memory graph (DiGraph), JSON persistence to `clients/{name}/index/knowledge_graph.json`
- Entity and Relationship dataclasses with key normalization, merge, serialization
- BFS traversal for neighbor queries (configurable depth)
- Source chunk ID tracking for provenance
- New `core/graph/models.py`, `core/graph/store.py`

### 7.4 Graph-Augmented Search & Query Modes

Five search modes leveraging the knowledge graph, inspired by LightRAG.

- **`"naive"`**: Standard hybrid search only (current Trove behavior, no graph)
- **`"local"`**: Entity-focused — find entities matching query, retrieve their source chunks
- **`"global"`**: Relationship-focused — find entities, BFS traverse relationships, retrieve connected chunks
- **`"hybrid"`**: Local + global combined with deduplication
- **`"mix"`**: Knowledge graph + standard hybrid search merged via RRF
- New `graph_search()` method on `HybridSearcher` with automatic fallback to naive
- New MCP tools: `trove_search_entities`, `trove_search_relationships`
- Graph modes available as search presets: `graph_local`, `graph_global`, `graph_hybrid`, `graph_mix`
- New `core/graph/search.py`, changes to `core/search.py`, `mcp_server/server.py`

### 7.5 Graph Visualization & Export

Export knowledge graph for visualization and stats.

- `graph_stats()`: entity/relationship counts, type distributions, top entities by connections, density
- `export_for_d3()`: D3.js-compatible JSON with nodes, links, type groups, max_nodes limit
- New MCP tools: `trove_graph_stats`, `trove_graph_export`
- CLI: `scripts/build_graph.py` for building graphs with `--rebuild`, `--sample`, `--dry-run`
- New `core/graph/export.py`

**Files added:** `core/graph/__init__.py`, `core/graph/models.py`, `core/graph/store.py`, `core/graph/extractor.py`, `core/graph/progress.py`, `core/graph/search.py`, `core/graph/export.py`, `scripts/build_graph.py`
**Files changed:** `core/client.py`, `core/search.py`, `mcp_server/server.py`, `requirements.txt`, `tests/test_core.py`
**Tests:** 205 total (35 new), all passing
