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
7. **Local-first LLM** - LM Studio is the default provider and no core feature needs a cloud API; Claude is an optional provider
8. **Incremental improvement** - Work follows the roadmaps; each item is independently valuable

---

## Roadmap

- **[docs/REVIVAL-PLAN.md](docs/REVIVAL-PLAN.md)**: what Trove does as a building block today, and the planned stability ports, feature ports, package rename and store interfaces.
- **[ROADMAP.md](ROADMAP.md)**: the product stages (plugin system, REST API, web UI, events, distribution).
- **[CHANGELOG.md](CHANGELOG.md)**: what changed in each release.
- **README "Known limits"**: current limits of the code, stated as facts.

Docs describe the current state in the present tense. History goes in CHANGELOG.md only.

---

## Architecture

```
trove/
├── core/
│   ├── paths.py              # Where the clients and agents folders are
│   ├── ingest.py             # Document processing with Unstructured, file manifest, incremental diff
│   ├── markdown.py           # Markdown frontmatter, image stripping, heading splitter
│   ├── pipeline.py           # Producer-consumer ingestion pipeline (parallel parse → GPU embed → index)
│   ├── model_cache.py        # Keeps embedding models and rerankers loaded, shared across clients
│   ├── resource_manager.py   # Probes CPU, RAM and GPU to pick concurrency settings
│   ├── health.py             # Circuit breaker for the LM Studio connection
│   ├── reference_config.py   # Configurable reference patterns per domain
│   ├── logging_config.py     # Audit configuration loading
│   ├── search/               # Hybrid search package
│   │   ├── types.py          # SearchQuality, SearchResult dataclasses
│   │   ├── tokenizer.py      # EnhancedTokenizer, stemming, stop words
│   │   ├── cache.py          # Search result LRU+TTL cache
│   │   ├── merge.py          # Merging results across indexes and across clients (RRF)
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
│   │   ├── store.py          # KnowledgeGraph (NetworkX + SQLite persistence)
│   │   ├── extractor.py      # LLM-based entity/relationship extraction
│   │   ├── progress.py       # Graph build progress tracking
│   │   ├── search.py         # Graph search modes (local/global/hybrid/mix)
│   │   └── export.py         # Stats and D3.js export
│   ├── memory.py             # Persistent memory store (SQLite + vector search)
│   ├── workflow.py            # Workflow DAG engine (task pipelines)
│   ├── agents.py              # Agent template loader (YAML configs)
│   ├── runner.py              # Local agent runner (search → RAG → LLM generate)
│   ├── plugins/             # Plugin system
│   │   ├── interfaces.py    # ABCs: BaseEmbedder, BaseChunker, BaseParser, BaseReranker, BaseLLMProvider, BaseMergeStrategy
│   │   ├── registry.py      # PluginRegistry: register/lookup by name
│   │   └── loader.py        # discover_plugins(), load_plugins() from plugins/ dir (not called at start-up)
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
│       ├── incremental.py    # Incremental code indexing state (file hashes and mtimes)
│       └── extractors/       # Language-specific extractors (Python, TS, JS, PHP, CSS, HTML, YAML, JSON, Gherkin, Dart)
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
├── core/data/agents/           # Agent YAML templates, shipped in the package
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
│   ├── benchmark_embeddings.py # Compare embedding models on an eval dataset
│   ├── sync_prompts.py       # Sync prompts from prompts.chat (needs the optional client, not in this repo)
│   ├── sources.py            # CLI for source connectors (connect, crawl, sync, list)
│   └── list_docs.sh          # Inventory of the office documents and PDFs in a folder
├── overnight-rebuild.sh      # Reindex and rebuild graphs for a list of clients
├── constraints/               # Exact versions from a working install, per Python version
└── tests/                     # One test file per area, shared fixtures in conftest.py, install checks in smoke/
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
Query → BM25 (top rerank_top_n, default 50, at least top_k) + Vector (same) → Merge (weighted or RRF) → Rerank (cross-encoder + heading match bonus) → MMR diversity → Data-first ordering (data_tier) → Page diversity (limit_per_page) → Results + SearchQuality
```

- **Weighted merge** (default): normalizes scores to 0-1, combines with configurable weights
- **RRF merge**: rank-based fusion, `score = sum(1/(k + rank))`, robust to score distribution differences
- **Score normalization**: configurable `minmax` (default), `zscore`, or `raw` for weighted merge
- **Adaptive reranking**: reduces rerank depth (default 30, never below top_k) on high-confidence matches
- **top_k**: honoured up to `MAX_TOP_K` (200) in `core/search/searcher.py`
- **Heading match**: `heading_boost` (default 1.0) times the share of query words in the chunk's title or section path is added to the rerank score
- **Page diversity**: `max_chunks_per_source` (default 2) chunks per page come before other pages; a file whose `canonical_url` is another file's `url` counts as that page
- **MMR diversity**: penalizes similarity to already-selected results (configurable `mmr_lambda`)
- **HyDE**: generates hypothetical answer via LLM, embeds alongside raw query for vector search (`search.hyde: true`)
- **Multi-query / RAG-Fusion**: generates query variants via LLM, searches each, merges via RRF (`search.multi_query: true`)
- **Reference resolution**: auto-fetches content when chunks reference other sections
- **Query contextualization**: rewrites follow-up queries with conversation context
- Use `search_with_quality()` for explicit access to `SearchQuality` metadata (scores, timing, strategy)

---

## Code Indexing

Enabled via `code:` section in `client.yaml`. Uses tree-sitter AST parsing for intelligent chunking.

**Supported languages:** Python, TypeScript/TSX, JavaScript/JSX, PHP, HTML/Vue/Svelte/MDX, CSS/SCSS/Sass/Less, YAML, JSON, Gherkin, Dart. Code indexing needs tree-sitter-languages, which has no wheels for Python 3.13.

---

## Markdown and Ranking

Markdown files (`.md` by default) skip Unstructured and go through `core/markdown.py`: frontmatter becomes chunk metadata, images become their alt text, and the body is split at headings with the heading path stored as `section`. `data_tier()` in `core/search/searcher.py` orders final results by tier, and Markdown section chunks are never put in the data tiers. README "Using Trove with a scraper" and "Known limits" describe the exact rules.

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
  model: all-MiniLM-L6-v2      # or BAAI/bge-base-en-v1.5 (768d)
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
  adaptive_rerank_depth: 30     # never below top_k
  max_chunks_per_source: 2      # chunks per page before other pages (0/null = off)
  heading_boost: 1.0            # rerank bonus for query words in title/section (0 = off)
  reranker: "local"             # "local" or "api"
  # Advanced query processing
  bm25_k1: 1.5                  # BM25 term frequency saturation (>0)
  bm25_b: 0.75                  # BM25 document length normalization (0-1)
  score_normalization: "minmax"  # "minmax", "zscore", or "raw"
  hyde: false                   # HyDE: generate hypothetical doc for vector search
  multi_query: false            # Multi-query: generate variants and merge via RRF
  multi_query_count: 3          # Number of query variants (2-7)
  # Search result caching
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

# Returned by Client.chunking_config but not passed to the chunkers by ingestion
# (README "Known limits"): ingestion always uses 1500 characters with 100 overlap
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

# Optional sections: documents, markdown, references, source_sync, graph, audit, code
# README "Client Configuration" shows each one; clients/example/client.yaml is a working example
```

---

## Key Commands

```bash
# Ingestion
python scripts/ingest.py --client example              # Incremental (default)
python scripts/ingest.py --client example --rebuild     # Full rebuild
python scripts/ingest.py --client example --rebuild-code # Rebuild the code index only, from scratch
python scripts/ingest.py --client example --sync-sources # Sync sources before ingest
python scripts/ingest.py --client example --json         # JSON summary of new/changed/unchanged/deleted/failed files
python scripts/ingest.py --all                          # All clients (incremental unless --rebuild)

# Clients outside the repo: TROVE_CLIENTS_DIR, or --clients-dir on any CLI script
# (the MCP server reads only TROVE_CLIENTS_DIR)
python scripts/create_client.py --name news-site --documents-dir /data/news-site --clients-dir ../projects
python scripts/search.py --clients docs,blog "your query"   # several clients, merged by rank

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
# Graph saves every 5 batches (graph.extract_save_every): safe to cancel, resumes on next run

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
