# CLAUDE.md - Trove Project Guide

## Project Philosophy

Trove uses proven libraries for document parsing, chunking, and search rather than custom implementations.

### Core Principles

1. **Use proven libraries** - Don't reinvent document parsing, chunking, or search
2. **Hybrid search** - BM25 keyword + vector semantic search combined
3. **Re-rank before LLM** - Cross-encoder filters bad results
4. **Agentic when needed** - Recursive search, reference resolution, query refinement
5. **Full audit trail** - Every search step logged for debugging and compliance
6. **Test with diverse data** - Must work for books, legal docs, real estate, not just compliance
7. **Local-first LLM** - All LLM processing uses LM Studio locally; no cloud API dependency for core features
8. **Incremental improvement** - Follow the staged roadmap in `ROADMAP.md`; each stage is independently valuable

---

## Roadmap

See **[ROADMAP.md](ROADMAP.md)** for planned future work. See **[CHANGELOG.md](CHANGELOG.md)** for completed stages.

**Completed (Stages 0-11 + Pipeline):** Source connectors (GitHub repo sync, web crawling, incremental caching, MCP tools, CLI), search quality (RRF, MMR, quality indicators), MCP enhancement (cache, multi-client, presets), evaluation framework, codebase health, response quality (citations, reranker, token budgeting), ingestion improvements (semantic chunking, progress tracking, MCP tools), knowledge graph (entity extraction, graph search modes, D3 export), advanced query processing (HyDE, multi-query/RAG-Fusion, BM25 tuning, score normalization), embedding & indexing quality (contextual embeddings, recursive chunking, model benchmarking), scale & performance (HNSW tuning, search result caching), agent orchestration (persistent memory, workflow DAGs, agent templates), ingestion pipeline optimization (producer-consumer GPU pipeline, parallel file parsing, progressive ChromaDB indexing, Ctrl+C resume, batch size 512, git ls-files fast collection, parallel change detection).

**Planned (Stages 12-21):** Plugin system (extension points, registry, discovery), distribution packaging (build/export, PyInstaller, unified CLI — local setup untouched), event system (pub/sub bus, webhooks), REST API (FastAPI, WebSockets, MCP adapter), Flutter Desktop UI (foundation, search, client management, graph explorer, agent chat). See ROADMAP.md for details.

---

## Architecture

```
trove/
├── core/
│   ├── ingest.py             # Document processing with Unstructured
│   ├── pipeline.py           # Producer-consumer ingestion pipeline (parallel parse → GPU embed → index)
│   ├── search/               # Hybrid search package
│   │   ├── types.py          # SearchQuality, SearchResult dataclasses
│   │   ├── tokenizer.py      # EnhancedTokenizer, stemming, stop words
│   │   ├── cache.py          # Search result LRU+TTL cache
│   │   ├── merge.py          # merge_search_results() for multi-index merging
│   │   └── searcher.py       # HybridSearcher (BM25 + vector + re-ranking + recursive)
│   ├── llm.py                # LLM integration (LM Studio, Claude)
│   ├── client.py             # Client loading and configuration
│   ├── refinement.py         # Query refinement and contextualization
│   ├── reference_resolver.py # Detect and fetch referenced appendices/sections
│   ├── reranker.py           # Reranker backends (local CrossEncoder, API, fallback)
│   ├── evaluation.py         # Retrieval metrics + LLM-as-judge RAG quality
│   ├── chunking.py           # Chunking strategies (title, semantic, recursive)
│   ├── ingest_progress.py    # Progress tracking and resumable ingestion
│   ├── ingest_job.py         # Ingestion job management for MCP tools
│   ├── graph/                # Knowledge graph subsystem
│   │   ├── models.py         # Entity, Relationship dataclasses
│   │   ├── store.py          # KnowledgeGraph (NetworkX + JSON persistence)
│   │   ├── extractor.py      # LLM-based entity/relationship extraction
│   │   ├── search.py         # Graph search modes (local/global/hybrid/mix)
│   │   └── export.py         # Stats and D3.js export
│   ├── memory.py             # Persistent memory store (SQLite + vector search)
│   ├── workflow.py            # Workflow DAG engine (task pipelines)
│   ├── agents.py              # Agent template loader (YAML configs)
│   ├── runner.py              # Local agent runner (search → RAG → LLM generate)
│   ├── plugins/             # Plugin system
│   │   ├── interfaces.py    # ABCs: BaseEmbedder, BaseChunker, BaseParser, BaseReranker, BaseLLMProvider, BaseMergeStrategy
│   │   ├── registry.py      # PluginRegistry: register/lookup by name
│   │   └── loader.py        # discover_plugins(), load_plugins() from plugins/ dir
│   ├── sources/              # Source connectors for automated document acquisition
│   │   ├── base.py           # BaseSourceConnector ABC, SyncResult
│   │   ├── github_connector.py # GitHub repo discovery, clone, sync
│   │   ├── github_client.py  # GitHub API client (rate limiting, pagination, retries)
│   │   ├── github_models.py  # GitHubSourceConfig, RepoInfo
│   │   ├── git_ops.py        # Git clone/fetch operations with token auth
│   │   ├── web_crawler.py    # BFS web crawler with scope control
│   │   ├── web_converter.py  # HTML → clean Markdown conversion
│   │   └── web_cache.py      # Incremental crawling cache (ETag/content hash)
│   └── code/                 # Code indexing subsystem
│       ├── chunker.py        # Code file chunking orchestration
│       ├── parser.py         # Tree-sitter parser management
│       ├── git_utils.py      # Git repository detection, file walking
│       ├── incremental.py    # Chunk ID generation, deduplication
│       └── extractors/       # Language-specific extractors (Python, TS, JS, PHP, CSS, HTML, YAML, JSON, Gherkin)
├── mcp_server/
│   ├── server.py             # MCP server entry point, tool definitions, dispatcher
│   ├── shared.py             # Shared state, cache, helpers
│   └── handlers/             # Tool handler modules
│       ├── clients.py        # Client discovery, loading, info
│       ├── search.py         # Search, multi-search, code search, symbols
│       ├── ingest.py         # File/URL ingestion, reindexing
│       ├── graph.py          # Knowledge graph operations
│       ├── memory.py         # Persistent memory CRUD
│       ├── workflows.py      # Workflow DAG management
│       ├── agents.py         # Agent discovery and execution
│       ├── prompts.py        # Prompt discovery (local + live API)
│       └── sources.py        # GitHub/web source connectors
├── plugins/                   # User plugin directory (custom extensions)
├── agents/                    # Global agent YAML templates
│   ├── security-reviewer.yaml
│   ├── code-reviewer.yaml
│   ├── performance-analyzer.yaml
│   ├── documentation-writer.yaml
│   ├── test-generator.yaml
│   └── architecture-analyst.yaml
├── clients/{name}/
│   ├── client.yaml           # Client config (including sources: section)
│   ├── prompts/system.txt    # System prompts
│   ├── documents/            # Source documents
│   ├── sources/              # Auto-managed source directories
│   │   ├── github/{owner}/{repo}/  # Cloned git repos
│   │   └── web/{domain}/     # Crawled markdown files
│   ├── tests/                # Evaluation datasets (eval_dataset.json)
│   └── index/                # Vector DB + BM25 index + file_manifest.json + trove_state.db
├── scripts/
│   ├── ingest.py             # CLI for document ingestion
│   ├── search.py             # CLI search tool
│   ├── evaluate.py           # CLI evaluation tool
│   ├── build_graph.py        # CLI for building knowledge graphs
│   ├── watch_and_reindex.py  # Source sync + file watcher
│   ├── create_client.py      # Interactive client creation wizard
│   ├── mcp_server.py         # MCP server launch script
│   ├── run_agent.py          # CLI for running agents locally via LM Studio
│   ├── sync_prompts.py       # Sync prompts from prompts.chat API
│   └── sources.py            # CLI for source connectors (connect, crawl, sync, list)
└── tests/test_core.py        # pytest tests
```

---

## Code Guidelines

### DO
- Use type hints
- Write docstrings
- Keep functions small and focused
- Log with `logging` module
- Write tests for core functionality
- Handle errors gracefully
- Keep business logic in `core/`, MCP interface in `mcp_server/`

### DON'T
- Add workarounds for search issues (fix the search instead)
- Hard-code domain-specific terms (use client config)

---

## Search Pipeline

```
Query → BM25 (top 50) + Vector (top 50) → Merge (weighted or RRF) → Rerank (cross-encoder) → MMR diversity → Results + SearchQuality
```

- **Weighted merge** (default): normalizes scores to 0-1, combines with configurable weights
- **RRF merge**: rank-based fusion, `score = sum(1/(k + rank))`, robust to score distribution differences
- **Score normalization**: configurable `minmax` (default), `zscore`, or `raw` for weighted merge
- **Adaptive reranking**: reduces rerank depth on high-confidence matches
- **MMR diversity**: penalizes similarity to already-selected results (configurable `mmr_lambda`)
- **HyDE**: generates hypothetical answer via LLM, embeds alongside raw query for vector search (`search.hyde: true`)
- **Multi-query / RAG-Fusion**: generates query variants via LLM, searches each, merges via RRF (`search.multi_query: true`)
- **Reference resolution**: auto-fetches content when chunks reference other sections
- **Query contextualization**: rewrites follow-up queries with conversation context
- Use `search_with_quality()` for explicit access to `SearchQuality` metadata (scores, timing, strategy)

---

## Code Indexing

Enabled via `code:` section in `client.yaml`. Uses tree-sitter AST parsing for intelligent chunking.

**Supported languages:** Python, TypeScript/TSX, JavaScript/JSX, PHP, HTML/Vue/Svelte, CSS/SCSS, YAML, JSON, Gherkin

**Code priority** (when docs may be outdated):
```yaml
search:
  code_priority: true           # Boost code chunks above docs
  code_priority_boost: 0.5      # Score boost for code chunks
  cross_reference_code: true    # Auto-fetch related code for doc results
```

---

## Client Configuration

```yaml
# clients/example/client.yaml
name: "Example RAG"
machine_name: example
description: "Optional description"

paths:
  documents: "documents"
  index: "index"
  prompts: "prompts"

embedding:
  model: all-MiniLM-L6-v2      # or bge-base-en-v1.5 (768d)
  dimension: 384                # must match model
  contextualize: false          # Prepend document context to chunks before embedding
  hnsw_ef_construction: 200     # HNSW build-time: neighbor list size (requires reindex)
  hnsw_max_neighbors: 16        # HNSW build-time: max connections per node (requires reindex)
  hnsw_ef_search: 100           # HNSW query-time: neighbor list size (immediate)

search:
  bm25_weight: 0.5
  vector_weight: 0.5
  top_k: 20
  rerank_top_n: 50
  min_score: null               # null=no filter, -2.0=strict
  merge_strategy: "weighted"    # "weighted" or "rrf"
  rrf_k: 60
  mmr_lambda: null              # null=disabled, 0.0-1.0
  adaptive_rerank: true
  reranker: "local"             # "local" or "api"
  # Stage 8: Advanced Query Processing
  bm25_k1: 1.5                  # BM25 term frequency saturation (>0)
  bm25_b: 0.75                  # BM25 document length normalization (0-1)
  score_normalization: "minmax"  # "minmax", "zscore", or "raw"
  hyde: false                   # HyDE: generate hypothetical doc for vector search
  multi_query: false            # Multi-query: generate variants and merge via RRF
  multi_query_count: 3          # Number of query variants (2-7)
  # Stage 10: Scale & Performance
  cache_ttl: 300                # Search cache TTL in seconds (0 = disabled)
  cache_max_size: 128           # Max cached queries per client

llm:
  provider: "lmstudio"                # OpenAI-compatible (correct for LM Studio)
  lmstudio_model: "mistralai/mistral-nemo-instruct-2407"  # must be set explicitly
  max_context_chars: 12000
  max_chunks: 10
  max_tokens: 2000
  citations: false
  context_filter: false
  context_filter_threshold: 0.1
  doc_budget_ratio: 0.6
  code_budget_ratio: 0.3
  reference_budget_ratio: 0.1

chunking:
  strategy: "title"             # "title", "semantic", or "recursive"
  max_chunk_size: 1500
  overlap: 100
  separators: ["\n\n", "\n", ". ", " "]  # for recursive strategy

# Source connectors (auto-acquire documents)
sources:
  github:
    - owner: myorg
      token_env: GITHUB_TOKEN       # env var (never store tokens in yaml)
      include_repos: [api, frontend] # optional whitelist
      exclude_repos: [legacy]        # optional blacklist
      sync_on_ingest: true           # auto-sync before reindexing
  web:
    - url: https://docs.example.com
      mode: crawl                    # crawl, docs, swagger
      stay_in_path: true
      max_pages: 500

# Optional sections: source_sync, graph, audit, code
# See existing client configs for examples
```

---

## Key Commands

```bash
# Ingestion
python scripts/ingest.py --client example              # Incremental (default)
python scripts/ingest.py --client example --rebuild     # Full rebuild
python scripts/ingest.py --client example --rebuild-code # Code only
python scripts/ingest.py --client example --sync-sources # Sync sources before ingest
python scripts/ingest.py --all                          # All clients

# Source connectors
python scripts/sources.py connect github -c myproject --owner myorg  # Connect GitHub org
python scripts/sources.py connect github -c myproject --repo owner/repo  # Connect single repo
python scripts/sources.py crawl -c myproject --url https://docs.example.com  # Crawl website
python scripts/sources.py sync -c myproject             # Sync all configured sources
python scripts/sources.py list -c myproject             # List sources and status

# Search & evaluation
python scripts/search.py --client example "your query"
python scripts/evaluate.py --client example
python scripts/evaluate.py --client example --rag       # + LLM-as-judge

# File watcher & source sync
python scripts/watch_and_reindex.py --client example
python scripts/watch_and_reindex.py --dry-run

# Knowledge graph
python scripts/build_graph.py --client example           # Incremental (new chunks only)
python scripts/build_graph.py --client example --rebuild  # Full rebuild
# Graph saves after every batch — safe to cancel, resumes on next run

# Overnight rebuild (all clients)
./overnight-rebuild.sh              # Incremental (changes only)
./overnight-rebuild.sh --rebuild    # Full rebuild from scratch

# Embedding model benchmark
python scripts/benchmark_embeddings.py --client example --models "all-MiniLM-L6-v2,BAAI/bge-base-en-v1.5"
python scripts/benchmark_embeddings.py --client example --models "all-MiniLM-L6-v2" --output results.json

# Local agent runner
python scripts/run_agent.py --list-agents                                          # List available agents
python scripts/run_agent.py --list-clients                                         # List available clients
python scripts/run_agent.py --agent code-reviewer --client example "Review search"    # One-shot query
python scripts/run_agent.py --agent code-reviewer --client example --interactive      # Interactive REPL
python scripts/run_agent.py --agent code-reviewer --client example --json "query"     # JSON output

# Run tests
pytest tests/
```
