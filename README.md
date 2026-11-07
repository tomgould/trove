# Trove: Knowledge Engine for AI Development

A local-first knowledge engine with hybrid search, source connectors, plugin extensibility, and MCP integration for Claude Code. Trove discovers, indexes, and searches your documents, code, and web content so AI tools always have the right context.

## Features

### Search & Retrieval
- **Hybrid Search**: BM25 keyword + vector semantic search with cross-encoder re-ranking
- **RRF Merge Strategy**: Reciprocal Rank Fusion as alternative to weighted merging
- **MMR Diversity**: Maximal Marginal Relevance reduces redundant results
- **HyDE**: Hypothetical Document Embeddings for better vector search on vague queries
- **Multi-Query / RAG-Fusion**: Generate query variants and merge results via RRF
- **Search Quality Indicators**: Scores, timing, source diversity, and strategy metadata
- **Configurable Embeddings**: Choose embedding model per client (MiniLM, BGE, E5)
- **Stemming & Synonyms**: Porter/Snowball stemmer and domain-specific synonyms
- **Agentic/Recursive Search**: Automatically refines queries when results are insufficient
- **Reference Resolution**: Detects "see Appendix X" and fetches referenced content
- **Query Contextualization**: Understands follow-up questions in conversation context
- **Search Result Caching**: LRU + TTL cache with filter-aware keys
- **Score Normalization**: MinMax, Z-score, or raw normalization for merge strategies

### Source Connectors
- **GitHub Connector**: Discover repos from GitHub accounts/orgs, clone/fetch, track changed files. API client with rate limiting, pagination, and retries. Tokens via environment variables only.
- **Web Crawler**: BFS crawl with scope control, HTML-to-Markdown conversion (markdownify + trafilatura), incremental cache via ETag/content hash.
- **Auto-sync**: Sources sync before ingestion when `sync_on_ingest` is configured. Manual sync via CLI or MCP.
- **Incremental Updates**: Only changed files are re-crawled and reindexed.

### Plugin System
- **6 Extension Points**: Custom embedding models, chunkers, parsers, rerankers, LLM providers, and merge strategies
- **Plugin Registry**: Register and lookup plugins by name with duplicate detection
- **Auto-Discovery**: Drop a `.py` file or package in `plugins/` and it's loaded on startup
- **Built-in Refactoring**: LLM providers and rerankers already implement plugin interfaces

### Ingestion & Indexing
- **Producer-Consumer Pipeline**: Parallel file parsing (ThreadPool) → GPU batch embedding → progressive ChromaDB indexing
- **Fast File Collection**: `git ls-files` for git repos (near-instant), `os.scandir` fallback, parallel change detection
- **Ctrl+C Safe**: Each GPU batch is indexed immediately; interrupt loses at most one batch, resume skips already-indexed files
- **Code Indexing**: Tree-sitter AST parsing for Python, TypeScript, JavaScript, PHP, HTML, CSS, YAML, JSON, Gherkin, Dart
- **Semantic Chunking**: Split by embedding similarity drops (alternative to title-based)
- **Recursive Chunking**: Priority-ordered separators with fallback
- **Contextual Embeddings**: Prepend document context to chunks before embedding
- **Chunk Quality Validation**: Min-length, alphanumeric, and content-hash deduplication

### Response Generation
- **Inline Citations**: LLM responses cite sources with `[1]`, `[2]` references
- **Configurable Reranker**: Local CrossEncoder or API-based (Cohere-compatible) with automatic fallback
- **Token Budgeting**: Separate context budgets for document, code, and reference chunks
- **Context Pre-Filtering**: Drop low-scoring chunks before LLM context

### Knowledge & Agents
- **Knowledge Graph**: Entity/relationship extraction, graph-augmented search (local/global/hybrid/mix modes)
- **Persistent Memory**: SQLite-backed store with vector search, TTL expiration, per-client or global
- **Workflow DAGs**: Multi-step task pipelines with dependency ordering and auto-status tracking
- **Agent Templates**: YAML-defined personas (code-reviewer, security-reviewer, etc.)
- **Local Agent Runner**: Search for context, inject system prompt, generate via LM Studio
- **Prompt Discovery** (optional): search a local prompt index and the live prompts.chat API; needs a prompts.chat client that is not included, see [Prompt library](#prompt-library-optional)

## Stack

| Layer | Library | License |
|-------|---------|---------|
| Document Processing | Unstructured.io | Apache 2.0 |
| Keyword Search | rank-bm25 | Apache 2.0 |
| Stemming/NLP | NLTK | Apache 2.0 |
| Vector Search | ChromaDB + sentence-transformers | Apache 2.0 |
| Re-ranking | sentence-transformers (cross-encoder) or API | Apache 2.0 |
| Knowledge Graph | NetworkX | BSD |
| GitHub API | httpx | BSD |
| HTML → Markdown | markdownify + trafilatura | MIT / Apache 2.0 |
| LLM (local) | LM Studio via OpenAI-compatible API | - |
| LLM (cloud) | Claude via Anthropic client | - |
| MCP Server | mcp | MIT |

**100% local capable. Zero API costs (with LM Studio).** Licensed under [BSL 1.1](LICENSE) (converts to MIT).

## Quick Start

Trove supports Python 3.11, 3.12 and 3.13. Code indexing needs tree-sitter-languages, which has no wheels for 3.13 yet, so on 3.13 everything works except code indexing.

```bash
# Get the code and install dependencies
git clone https://github.com/tomgould/trove.git
cd trove
python -m venv .venv && source .venv/bin/activate

# Install the CPU build of PyTorch first. Without this, pip pulls the much
# larger CUDA build. Skip it if you want GPU support.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# The full stack: document parsers, LLM clients, MCP server, code indexing
pip install -r requirements.txt

# Create a new client (interactive wizard)
python scripts/create_client.py

# Option A: Add local documents
cp /path/to/docs/* clients/myproject/documents/

# Option B: Connect external sources
python scripts/sources.py connect github --client myproject --owner myorg
python scripts/sources.py crawl --client myproject --url https://docs.example.com

# Ingest and index
python scripts/ingest.py --client myproject

# Use via MCP server (see MCP Server section below)
python scripts/mcp_server.py
```

### Reproducible installs

`requirements.txt` and `pyproject.toml` give tested version ranges. For the exact versions CI tests with, add the constraints file for your Python version:

```bash
pip install -r requirements.txt -c constraints/py312.txt   # or py311.txt, py313.txt
```

`constraints/py311.txt` needs Python 3.11.5 or later, because one pinned package (soupsieve 4.0.1) does. On an earlier 3.11 release, install without the constraints file.

### Installing as a package

Trove also installs as a package, with extras so you only pull what you use:

| Extra | Adds |
|-------|------|
| (none) | Hybrid search, ingestion of text, HTML and Markdown, source connectors, knowledge graph store |
| `docs` | PDF, Word, Excel, PowerPoint and the other formats unstructured can parse |
| `llm` | The OpenAI-compatible (LM Studio) and Anthropic clients, for agents, graph extraction and generated answers |
| `mcp` | The MCP server |
| `code` | Code indexing with tree-sitter (Python 3.11 and 3.12) |
| `dev` | pytest, ruff and build |
| `all` | `docs`, `llm`, `mcp` and `code`: the same stack as `requirements.txt` |

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ".[mcp,docs]"
```

An installed copy provides the commands `trove-server`, `trove-search`, `trove-ingest`, `trove-sources` and `trove-create-client`. They look for clients in a `clients/` folder in the current directory. Run from a checkout, the scripts in `scripts/` use the checkout's own `clients/` folder.

## Creating a Client

The easiest way to create a new client is with the interactive wizard:

```bash
python scripts/create_client.py
```

The wizard will prompt you for:

- **Client name** - Display name (e.g., "Acme Corp RAG")
- **Machine name** - Directory name (auto-generated from display name)
- **Description** - Optional description
- **Embedding model** - MiniLM (fast) or BGE (better quality)
- **LLM settings** - Provider (lmstudio/claude), context limits, max tokens
- **Search settings** - BM25/vector weights, top_k, rerank settings (optional)
- **Tokenization** - Stemmer selection (Porter, Snowball, or none)
- **Reference domain** - Compliance, legal, technical, or academic patterns
- **Audit settings** - Enable/disable logging, log level
- **System prompt** - Custom prompt or use default

After completion, it creates the full directory structure and config files:

```
clients/your_client/
├── client.yaml       # Configuration
├── documents/        # Add your source files here
├── index/            # Search index (created on ingest)
└── prompts/          # System prompt
    └── system.txt
```

### Without the wizard

Scripts and pipelines can create a client in one command:

```bash
python scripts/create_client.py --name news-site --description "Scraped news" \
    --documents-dir /data/scrapes/news-site --embedding-model all-MiniLM-L6-v2
```

Only `--name` is required. A name is 3 to 64 letters, digits, `_` or `-`, starting and ending with a letter or digit. The command writes a minimal `client.yaml` (everything else uses the defaults), creates `index/` and `prompts/` (and `documents/` unless `--documents-dir` is given), and refuses to overwrite an existing `client.yaml` unless you pass `--force`. With no arguments it starts the wizard.

### Manual Setup

Alternatively, create a client manually:

```bash
mkdir -p clients/myproject/{documents,index,prompts,data}
```

Then create `clients/myproject/client.yaml` (see Client Configuration below).

## How It Works

### 1. Document Ingestion

```
PDF/DOCX/XLSX/CSV/MBOX/MD → Unstructured → Chunks with metadata → Vector DB + BM25 Index
```

Supported file types:
- **PDF** - Text extraction with tables, lists preserved
- **Word** - DOCX, DOC
- **Spreadsheets** - XLSX, XLS, CSV
- **Email archives** - MBOX (Gmail exports, etc.)
- **Markdown** - Ideal for scraped API docs
- **HTML** - Web pages
- **Plain text** - TXT files

Features:
- Document structure preserved (headings, sections)
- Automatic chunking by section
- Email metadata preserved (subject, from, to, date)
- Recursive directory scanning (all subdirectories)

### 2. Search (Hybrid + Agentic)

```
Query → Contextualize (if follow-up)
      → BM25 (keyword) + Vector (semantic) in parallel
      → Merge (weighted or RRF)
      → Cross-encoder re-rank (adaptive depth)
      → MMR diversity (if enabled)
      → Reference resolution (fetch appendices/sections)
      → [If recursive: analyze → retry with better query]
      → Final results + quality indicators
```

- BM25 finds exact matches ("Appendix C" → actual Appendix C)
- Vector finds semantic matches ("equivalent jurisdictions" → related content)
- Merge via weighted scores (default) or RRF rank fusion
- Cross-encoder picks the best results (adaptive depth reduces cost on high-confidence matches)
- MMR reduces redundancy by penalizing similarity to already-selected chunks
- Reference resolver fetches content when chunks say "see Appendix X"

### 3. Response Generation

```
Top chunks → Context filter (drop low-scoring)
           → Token budgeting (doc/code/reference allocations)
           → LLM prompt with citation instructions
           → Answer with inline [1] [2] citations
           → Citation footer + Follow-up suggestions
```

## Project Structure

```
Trove/
├── core/
│   ├── data/agents/          # Agent YAML templates (shipped in the package)
│   ├── paths.py              # Where the clients and agents folders are
│   ├── ingest.py             # Document processing with Unstructured
│   ├── pipeline.py           # Producer-consumer ingestion pipeline
│   ├── search/               # Hybrid search package
│   │   ├── types.py          # SearchQuality, SearchResult dataclasses
│   │   ├── tokenizer.py      # EnhancedTokenizer, stemming, stop words
│   │   ├── cache.py          # Search result LRU+TTL cache
│   │   ├── merge.py          # merge_search_results() for multi-index merging
│   │   └── searcher.py       # HybridSearcher (BM25 + vector + re-ranking + recursive)
│   ├── llm.py                # LLM providers (LM Studio, Claude)
│   ├── client.py             # Client configuration + validation
│   ├── refinement.py         # Query refinement & contextualization
│   ├── reference_resolver.py # Appendix/section reference resolution
│   ├── reference_config.py   # Configurable reference patterns per domain
│   ├── logging_config.py     # Audit configuration loading
│   ├── reranker.py           # Reranker backends (local, API, fallback)
│   ├── evaluation.py         # Retrieval metrics + LLM-as-judge RAG quality
│   ├── chunking.py           # Chunking strategies (title, semantic, recursive)
│   ├── ingest_progress.py    # Progress tracking and resumable ingestion
│   ├── ingest_job.py         # Ingestion job management for MCP tools
│   ├── memory.py             # Persistent memory store (SQLite + vector search)
│   ├── workflow.py           # Workflow DAG engine (task pipelines)
│   ├── agents.py             # Agent template loader (YAML configs)
│   ├── runner.py             # Local agent runner (search → RAG → LLM generate)
│   ├── plugins/              # Plugin system
│   │   ├── interfaces.py     # ABCs: BaseEmbedder, BaseChunker, BaseParser, etc.
│   │   ├── registry.py       # Plugin registry (register/lookup by name)
│   │   └── loader.py         # Auto-discovery from plugins/ directory
│   ├── sources/              # Source connectors
│   │   ├── base.py           # BaseSourceConnector ABC, SyncResult
│   │   ├── github_connector.py # GitHub repo discovery, clone, sync
│   │   ├── github_client.py  # GitHub API client (rate limiting, pagination)
│   │   ├── github_models.py  # GitHubSourceConfig, RepoInfo
│   │   ├── git_ops.py        # Git clone/fetch with token auth
│   │   ├── web_crawler.py    # BFS web crawler with scope control
│   │   ├── web_converter.py  # HTML → clean Markdown conversion
│   │   └── web_cache.py      # Incremental crawling cache (ETag/content hash)
│   ├── code/                 # Code indexing subsystem
│   │   ├── chunker.py        # Code file chunking orchestration
│   │   ├── parser.py         # Tree-sitter parser management
│   │   ├── git_utils.py      # Git repository detection
│   │   ├── incremental.py    # Incremental indexing state
│   │   └── extractors/       # Language-specific extractors
│   └── graph/                # Knowledge graph subsystem
│       ├── models.py         # Entity, Relationship dataclasses
│       ├── store.py          # KnowledgeGraph (NetworkX + JSON)
│       ├── extractor.py      # LLM-based entity/relationship extraction
│       ├── search.py         # Graph search modes
│       └── export.py         # Stats and D3.js export
├── plugins/                   # User plugin directory (drop .py files here)
├── mcp_server/
│   ├── server.py             # MCP server entry point, tool definitions, dispatcher
│   ├── shared.py             # Shared state, cache, helpers
│   └── handlers/             # Tool handler modules (clients, search, ingest, graph, memory, workflows, agents, prompts, sources)
├── clients/
│   ├── {name}/
│   │   ├── client.yaml       # Config (including sources: section)
│   │   ├── documents/        # Local source files
│   │   ├── sources/          # Auto-managed source directories
│   │   │   ├── github/       # Cloned git repos
│   │   │   └── web/          # Crawled markdown files
│   │   ├── index/            # Search indices
│   │   ├── prompts/          # System prompts
│   │   └── tests/            # Evaluation datasets
├── scripts/
│   ├── create_client.py      # Interactive client creation wizard
│   ├── ingest.py             # CLI for document ingestion
│   ├── search.py             # CLI search tool
│   ├── evaluate.py           # CLI evaluation tool
│   ├── build_graph.py        # CLI for building knowledge graphs
│   ├── sources.py            # CLI for source connectors (GitHub, web)
│   ├── mcp_server.py         # MCP server launch script
│   ├── run_agent.py          # CLI for running agents locally
│   ├── watch_and_reindex.py  # Source sync + file watcher
│   └── sync_prompts.py       # Sync prompts from prompts.chat API
├── constraints/              # Exact versions from a working install, per Python version
└── tests/
    ├── conftest.py           # Shared fixtures
    ├── test_*.py             # One file per area: client, ingest, search, ranking, graph, ...
    └── smoke/                # Install checks for a built wheel
```

## Client Configuration

```yaml
# clients/example/client.yaml
name: "My RAG System"
machine_name: example

paths:
  documents: "documents"
  index: "index"
  prompts: "prompts"

# Embedding model settings
embedding:
  model: "all-MiniLM-L6-v2"  # or "BAAI/bge-base-en-v1.5" for better quality
  dimension: 384             # 384 for MiniLM, 768 for BGE
  query_prefix: ""           # Use "query: " for E5 models
  passage_prefix: ""         # Use "passage: " for E5 models
  contextualize: false       # Prepend doc context to chunks before embedding (requires reindex)
  hnsw_ef_construction: 200  # HNSW build-time neighbor list (requires reindex)
  hnsw_max_neighbors: 16     # HNSW max connections per node (requires reindex)
  hnsw_ef_search: 100        # HNSW query-time neighbor list (no reindex needed)

search:
  bm25_weight: 0.5
  vector_weight: 0.5
  top_k: 20
  rerank_top_n: 50
  # Merge strategy: "weighted" (default) or "rrf" (rank fusion)
  merge_strategy: "weighted"
  rrf_k: 60                         # RRF constant (only used with merge_strategy: "rrf")
  # MMR diversity: null (disabled) or 0.0-1.0 (0.0=max diversity, 1.0=pure relevance)
  mmr_lambda: null
  # Adaptive reranking: reduce rerank depth on high-confidence matches
  adaptive_rerank: true
  adaptive_rerank_threshold: 0.9     # Combined score threshold to trigger
  adaptive_rerank_depth: 15          # Reduced depth when triggered
  # BM25 tuning
  bm25_k1: 1.5                        # Term frequency saturation (>0)
  bm25_b: 0.75                        # Document length normalization (0-1)
  score_normalization: "minmax"        # "minmax", "zscore", or "raw"
  # Advanced query processing (requires LLM provider)
  hyde: false                          # Generate hypothetical doc for vector search
  multi_query: false                   # Generate query variants and merge via RRF
  multi_query_count: 3                 # Number of query variants (2-7)
  # Search result caching
  cache_ttl: 0                         # TTL in seconds (0 = disabled)
  cache_max_size: 128                  # Max cached queries per client
  # Reranker backend: "local" (CrossEncoder) or "api" (Cohere-compatible)
  reranker: "local"
  reranker_model: null               # Custom reranker model (null=default)
  reranker_endpoint: null            # API endpoint for "api" reranker
  # Tokenization settings for BM25
  tokenization:
    stemmer: "porter"  # "porter", "snowball", or null
    synonyms:          # Domain-specific synonyms
      aml: ["anti-money laundering", "financial crime"]
      kyc: ["know your customer"]

# Reference resolution settings
references:
  enabled: true
  mode: "extend"        # "extend" adds to domain defaults, "replace" uses only custom
  domain: "compliance"  # "compliance", "legal", "technical", or "academic"
  # Optional: custom patterns
  patterns:
    custom_ref:
      priority: high
      patterns:
        - '(?:see\s+)?REF-(\d+)'
      query_templates:
        - "Reference {id}"

llm:
  provider: "lmstudio"                                    # OpenAI-compatible (recommended for LM Studio)
  lmstudio_model: "mistralai/mistral-nemo-instruct-2407"  # must be set explicitly
  max_chunks: 10
  max_context_chars: 12000
  max_tokens: 2000
  citations: false                # Enable inline [N] citations in responses
  context_filter: false           # Drop low-scoring chunks before LLM context
  context_filter_threshold: 0.1   # Minimum rerank score for context inclusion
  doc_budget_ratio: 0.6           # Token budget share for document chunks
  code_budget_ratio: 0.3          # Token budget share for code chunks
  reference_budget_ratio: 0.1     # Token budget share for reference chunks

# Optional: Knowledge graph (entity/relationship extraction)
graph:
  enabled: false                # Opt-in per client
  entity_types:                 # Entity types to extract
    - person
    - organization
    - concept
    - location
  extract_batch_size: 10        # Chunks per LLM call
  max_description_chars: 200    # Truncate descriptions
  search_mode: "naive"          # Default search: naive, local, global, hybrid, mix

# Optional: Chunking strategy
chunking:
  strategy: "title"             # "title", "semantic", or "recursive"
  max_chunk_size: 1500
  min_chunk_size: 100
  overlap: 100                  # Overlap between chunks (recursive strategy)
  semantic_threshold: 0.5       # Cosine similarity for semantic split points
  separators: ["\n\n", "\n", ". ", " "]  # Priority-ordered separators (recursive strategy)

# Optional: Source connectors (auto-fetch from GitHub/web)
sources:
  github:
    - owner: myorg                  # GitHub user or org
      token_env: GITHUB_TOKEN       # Env var name (never store tokens in YAML)
      include_repos: [api, frontend] # Optional whitelist
      exclude_repos: [legacy-app]    # Optional blacklist
      sync_on_ingest: true           # Auto-sync before reindexing
    - repo: owner/repo              # Single repo shorthand
      token_env: GITHUB_TOKEN
  web:
    - url: https://docs.example.com
      stay_in_path: true            # Don't leave starting URL path
      max_pages: 500                # Safety limit
      sync_on_ingest: true

# Optional: Code indexing (tree-sitter AST parsing)
code:
  enabled: true
  paths: ["src/"]                   # Directories to index (relative to client root)
  extensions: [".py", ".ts", ".js"] # File extensions to include
  exclude_patterns: ["**/test_*", "**/__pycache__/**"]

# Optional: Code priority in search results (when docs may be outdated)
# Add to search: section
#   code_priority: true             # Boost code chunks above docs
#   code_priority_boost: 0.5        # Score boost for code chunks
#   cross_reference_code: true      # Auto-fetch related code for doc results

# Optional: Audit settings
audit:
  enabled: true
  log_level: "full"  # "full", "minimal", or "none"
  include_chunks: true
  include_scores: true
  include_timing: true
```

### Embedding Models

| Model | Dimensions | Speed | Quality | Notes |
|-------|------------|-------|---------|-------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Default, good for most use cases |
| `BAAI/bge-base-en-v1.5` | 768 | Medium | Better | Better retrieval, recommended for production |
| `intfloat/e5-base-v2` | 768 | Medium | Better | Requires query/passage prefixes |

### Reference Domains

| Domain | Patterns Detected |
|--------|-------------------|
| `compliance` | Appendix, Annex, Schedule, Section, Table |
| `legal` | Exhibit, Clause, Article, Recital, Appendix, Schedule |
| `technical` | RFC, Issue #, PR #, Section |
| `academic` | Equation, Theorem, Lemma, Definition, Figure, Table, Chapter |

## CLI Commands

### Ingestion

```bash
# Ingest documents for a client
python scripts/ingest.py --client myproject

# Rebuild from client's documents directory
python scripts/ingest.py --client myproject --rebuild

# Rebuild ALL clients at once
python scripts/ingest.py --all

# Rebuild all clients, documents only (skip code)
python scripts/ingest.py --all --no-code

# Resume an interrupted ingestion (skips already-processed files)
python scripts/ingest.py --client myproject --rebuild --resume

# Force BM25 cache rebuild (use after changing tokenization settings)
python scripts/ingest.py --client myproject --rebuild --force-bm25-rebuild

# List available clients
python scripts/ingest.py --list

# Print a JSON summary of what changed (for scripts and schedulers)
python scripts/ingest.py --client myproject --json
```

The default run is incremental. Unchanged files are not parsed or embedded again, including files a scraper rewrote with identical content, because the manifest keeps a hash of each file. A file that fails is reported and retried on the next run. `--json` prints:

```json
{"client": "myproject", "mode": "incremental", "new": 2, "changed": 1, "unchanged": 40,
 "deleted": 0, "failed": 0, "failed_files": [], "chunks_added": 9, "chunks_removed": 3}
```

`mode` is `rebuild` on the first run (or with `--rebuild`), when every file counts as new. With `--all` the summaries are listed under `"clients"`.

### Search

```bash
# Basic search
python scripts/search.py --client myproject "your search query"

# Verbose output with timing and metadata
python scripts/search.py --client myproject --verbose "your query"

# Use RRF merge instead of weighted
python scripts/search.py --client myproject --merge-strategy rrf "your query"

# Enable MMR diversity (0.0=max diversity, 1.0=pure relevance)
python scripts/search.py --client myproject --mmr-lambda 0.7 "your query"

# Disable adaptive reranking
python scripts/search.py --client myproject --no-adaptive "your query"

# JSON output (for scripting/comparison)
python scripts/search.py --client myproject --json "your query"

# Full content (no truncation)
python scripts/search.py --client myproject --full "your query"

# Search several clients, merged by rank, each result labelled with its client
python scripts/search.py --clients docs,blog "your query"
python scripts/search.py --all --json "your query"

# List available clients
python scripts/search.py --list
```

Results show the source file and, when the document has them, its title, URL and section (heading path). `--json` results carry `client`, `source`, `title`, `url` and `section` fields; `title`, `url` and `section` are `null` when unknown.

### Source Connectors

Connect external sources (GitHub repos, websites) to a client. Source files are stored in `clients/{name}/sources/` and indexed alongside `documents/`.

```bash
# Connect a GitHub org's repos to a client
python scripts/sources.py connect github --client myproject --owner myorg

# Connect a single repo
python scripts/sources.py connect github --client myproject --repo owner/repo

# Filter repos
python scripts/sources.py connect github --client myproject --owner myorg \
    --include api frontend --exclude legacy --public-only

# Crawl a website into a client
python scripts/sources.py crawl --client myproject --url https://docs.example.com

# Crawl with options
python scripts/sources.py crawl --client myproject --url https://api.example.com/docs \
    --max-pages 200 --wait-time 2.0 --no-stay-in-path

# Sync all configured sources for a client
python scripts/sources.py sync --client myproject

# Sync all clients at once
python scripts/sources.py sync --all

# List configured sources and on-disk state
python scripts/sources.py list --client myproject
```

After connecting sources, ingest them:

```bash
python scripts/ingest.py --client myproject --rebuild
```

Sources with `sync_on_ingest: true` in `client.yaml` automatically sync before each ingestion.

### Automatic Reindexing (File Watcher)

The watcher compares files on disk against a manifest saved during ingestion. It detects new, modified, and deleted files, and only changed files are reindexed. If a client has `source_sync` configured, files are synced from the source directory before checking for changes.

```bash
# Dry-run to see what changed
python scripts/watch_and_reindex.py --dry-run

# Run for a specific client (sync + reindex)
python scripts/watch_and_reindex.py --client myproject

# Set up as cron (every 15 minutes)
*/15 * * * * cd /path/to/trove && ./venv/bin/python scripts/watch_and_reindex.py
```

#### Source Sync

Clients can auto-sync files from an external directory into their `documents/` folder. Add to `client.yaml`:

```yaml
source_sync:
  enabled: true
  source: ../..                   # Relative to client directory
  include: ["*.py", "*.md", "*.txt", "*.yaml", "*.json"]
  exclude: [clients/, venv/, .git/, __pycache__/]
```

The sync copies new/updated files, removes deleted files, and respects include/exclude patterns. It runs automatically as part of `watch_and_reindex.py`.

### Evaluation

Measure search quality with the evaluation tool. Requires a test dataset at `clients/{name}/tests/eval_dataset.json`.

```bash
# Run retrieval evaluation
python scripts/evaluate.py --client myproject

# Filter by tags
python scripts/evaluate.py --client myproject --tags search

# Include LLM-as-judge RAG quality metrics (requires LM Studio running)
python scripts/evaluate.py --client myproject --rag

# A/B comparison of merge strategies
python scripts/evaluate.py --client myproject --compare "weighted,rrf"

# JSON output for scripting
python scripts/evaluate.py --client myproject --json

# Write markdown report
python scripts/evaluate.py --client myproject --output reports/eval.md

# List clients with eval datasets
python scripts/evaluate.py --list
```

### Embedding Benchmark

Compare embedding models on your own evaluation dataset.

```bash
# Compare two models on a client's eval dataset
python scripts/benchmark_embeddings.py --client myproject \
    --models "all-MiniLM-L6-v2,BAAI/bge-base-en-v1.5"

# Save results to JSON
python scripts/benchmark_embeddings.py --client myproject \
    --models "all-MiniLM-L6-v2" --output results.json
```

### Knowledge Graph

Build a knowledge graph from indexed document chunks using LLM entity/relationship extraction.

```bash
# Build graph for a client (incremental - skips already-processed chunks)
python scripts/build_graph.py --client myproject

# Rebuild from scratch
python scripts/build_graph.py --client myproject --rebuild

# Dry-run (show chunk count and estimated LLM calls)
python scripts/build_graph.py --client myproject --dry-run

# Test with a sample of 50 random chunks
python scripts/build_graph.py --client myproject --sample 50

# Build for all graph-enabled clients
python scripts/build_graph.py --all

# Build even if graph.enabled is false in client.yaml
python scripts/build_graph.py --client myproject --force

# Graph saves after every batch, so it is safe to cancel mid-run, resumes on next incremental run
```

### Local Agent Runner

Run Trove agents end-to-end using a local LLM (LM Studio). Combines agent system prompt + RAG context search + LLM generation.

```bash
# List available agents
python scripts/run_agent.py --list-agents

# List available clients
python scripts/run_agent.py --list-clients

# One-shot query
python scripts/run_agent.py --agent code-reviewer --client example "Review the search pipeline"

# Interactive REPL with conversation history
python scripts/run_agent.py --agent code-reviewer --client example --interactive

# JSON output for scripting
python scripts/run_agent.py --agent security-reviewer --client myproject --top-k 15 --json "Check auth patterns"
```

Available agents: `code-reviewer`, `security-reviewer`, `performance-analyzer`, `documentation-writer`, `test-generator`, `architecture-analyst`

### Prompt Library (optional)

`scripts/sync_prompts.py` and the three `trove_prompts_*` MCP tools work with a prompts.chat API client at `clients/prompts/api_client.py`. That client is not part of this repository. Without it the script exits with "The prompt library is not available" and the MCP tools return the same message instead of failing. `trove_prompts_find` still searches a local `prompts` client if you have indexed one.

If you add your own client module there, the script syncs prompts into `clients/prompts/documents/`:

```bash
python scripts/sync_prompts.py --dry-run   # preview
python scripts/sync_prompts.py --reindex   # sync, then reindex the prompts client
```

### Overnight Rebuild

Reindex all clients and rebuild knowledge graphs in one run. Requires LM Studio running with the configured model loaded.

```bash
# Incremental (changes only, fast on most nights)
./overnight-rebuild.sh

# Full rebuild from scratch
./overnight-rebuild.sh --rebuild
```

Edit the script to uncomment/comment clients in the `REINDEX_CLIENTS` and `GRAPH_CLIENTS` arrays.

### Client Creation

```bash
# Interactive wizard
python scripts/create_client.py

# Without questions
python scripts/create_client.py --name myproject --documents-dir /path/to/docs
```

Every script accepts `--clients-dir PATH`. See [Using Trove with a scraper](#using-trove-with-a-scraper).

## Using Trove with a scraper

Trove works well as the search half of a scrape pipeline: the scraper writes Markdown, Trove indexes it, and answers cite the page they came from.

### Where clients live

By default clients are folders in the repository's `clients/`. To keep them elsewhere, set `TROVE_CLIENTS_DIR` or pass `--clients-dir` to any script. The flag wins over the environment variable, which wins over the default. A relative path is taken from the folder you run the script in. The MCP server reads `TROVE_CLIENTS_DIR` when it starts.

A client is a folder with a `client.yaml`. Folders whose names start with `_` or `.`, and folders without a `client.yaml`, are not clients, so a workspace can keep a `_template/` folder next to its projects:

```
workspace/
└── projects/                  # TROVE_CLIENTS_DIR=workspace/projects
    ├── _template/             # ignored
    ├── news-site/
    │   ├── client.yaml
    │   ├── documents/         # the scraper writes Markdown here
    │   └── index/             # Trove's index, created on ingest
    └── docs-site/
        ├── client.yaml        # paths.documents may be absolute or a symlink
        └── index/
```

```bash
export TROVE_CLIENTS_DIR=$PWD/workspace/projects
python scripts/create_client.py --name news-site
# ... scraper writes workspace/projects/news-site/documents/*.md ...
python scripts/ingest.py --client news-site --json
python scripts/search.py --client news-site --json "what changed in the pricing"
```

### The Markdown contract

Each page is one `.md` (or `.markdown`) file. It may start with YAML frontmatter:

```markdown
---
title: Pricing changes for 2027
url: https://example.com/blog/pricing-2027
date: 2026-10-01
domain: example.com
page_type: article
scraped_at: 2026-11-01T09:30:00Z
content_hash: 4f1c2e
---
# Pricing changes for 2027

## What is changing
...
```

| Field | Type | Stored as | Used for |
|-------|------|-----------|----------|
| `title` | string | `title` (string, at most 300 characters) | Shown with results and citations. Falls back to the first `#` heading |
| `url` | string | `url` (string) | Shown with results and citations |
| `date` | date or string | `date` (ISO string) | Metadata |
| `domain` | string | `domain` | Metadata |
| `page_type` | string | `page_type` | Metadata |
| `scraped_at` | datetime or string | `scraped_at` (ISO string) | Metadata |
| `content_hash` | string | `page_content_hash` | Metadata (`content_hash` is the chunk's own hash) |

All fields are optional and other keys are ignored. A number in `title` or `url` is stored as text (`url: 12345` becomes `"12345"`). A `title` or `url` that is not a plain string or number (a list, a mapping, a date, `.nan`, `true`, or blank) is ignored, so the title falls back to the first `#` heading. A longer title is cut to 300 characters, because it is copied into every chunk. The frontmatter block is removed from the indexed text. If it is not valid YAML, Trove logs a warning and indexes the whole file as text; a file is never dropped because of its frontmatter.

The body is split at its headings (`#` to `######`, and underlined headings). Each chunk records its heading path as `section`, for example `Pricing changes for 2027 > What is changing`, and long sections are split further to the usual chunk size. Headings inside code blocks are ignored; a code block opened with ```` ``` ```` closes only with at least as many backticks (`~~~` likewise), and a block that is never closed runs to the end of the file.

No text is dropped for being short. A section under 80 characters joins the chunk before it if both fit in the chunk size, otherwise the chunk after it, and if neither fits it stays a small chunk of its own. When a chunk holds more than one section, its `section` is the heading path of the section that gives it the most text (the first one on a tie). The other sections' headings stay in the chunk text, so a search for them still finds it.

Images are not indexed: `![alt](url)` becomes its alt text and a line holding only images is dropped. To keep image syntax, set this in `client.yaml`:

```yaml
markdown:
  strip_images: false
```

Other file types (PDF, HTML, Word and so on) are parsed as before and have no `section`.

### Re-running the scraper

Run `ingest.py` after each scrape. Only new and changed files are embedded; identical files are skipped even if the scraper rewrote them. Use `--json` to decide what to do next, for example to alert on `failed`. A page that disappears from `documents/` is reported as `deleted` and the incremental run removes its chunks. Cleaning up chunks left behind in other ways (for example by an interrupted run) is planned work, see [docs/REVIVAL-PLAN.md](docs/REVIVAL-PLAN.md).

## MCP Server (Claude Code Integration)

Trove includes an MCP (Model Context Protocol) server that allows Claude Code to query your indexed documentation directly.

### Use Case

1. Scrape API documentation and convert to markdown
2. Create a Trove client for the API (`python scripts/create_client.py`)
3. Ingest the docs (`python scripts/ingest.py --client api_name`)
4. Configure Claude Code to use the Trove MCP server
5. Claude Code can now search your API docs when helping you code

### Setup

Add to your Claude Code settings (`~/.claude/settings.json`) or project `.mcp.json`:

```json
{
  "mcpServers": {
    "trove": {
      "command": "python",
      "args": ["/path/to/trove/scripts/mcp_server.py"],
      "env": {
        "TROVE_DEFAULT_CLIENT": "your_client_name"
      }
    }
  }
}
```

With Trove installed as a package (`pip install ".[mcp]"`), use `"command": "trove-server"` with no `args`. It reads clients from `clients/` in the folder it is started in.

### Available Tools

| Tool | Description |
|------|-------------|
| `trove_list_clients` | List all available Trove clients/projects |
| `trove_get_client` | Get info about the currently loaded client |
| `trove_load_client` | Load a specific client for searching |
| `trove_search` | Search documents with mode presets and context budgets |
| `trove_search_client` | Load and search a client in one call |
| `trove_search_multi` | Search across multiple clients simultaneously (cross-client RRF merge) |
| `trove_search_code` | Search code only, with language/symbol type filtering |
| `trove_list_repos` | List git repositories in the loaded client's index |
| `trove_search_repo` | Search within a specific repository |
| `trove_find_symbol` | Find a function, class, or method by name |
| `trove_summarize` | Search and return concise summaries instead of full content |
| `trove_ingest_file` | Ingest a single file into a client's index |
| `trove_ingest_url` | Fetch and ingest content from a URL |
| `trove_reindex_client` | Trigger a full background reindex for a client |
| `trove_ingest_status` | Check status/progress of an ingestion job |
| `trove_graph_stats` | Get knowledge graph statistics (entity/relationship counts, top entities) |
| `trove_graph_export` | Export knowledge graph as D3.js-compatible JSON |
| `trove_search_entities` | Search knowledge graph entities by name |
| `trove_search_relationships` | Traverse knowledge graph relationships from an entity |
| `trove_build_graph` | Build/rebuild a knowledge graph (background) |
| `trove_connect_github` | Connect GitHub repos to a client (discover, clone, sync) |
| `trove_crawl_web` | Crawl a website and save pages as Markdown |
| `trove_sync_sources` | Sync all configured sources for a client |
| `trove_list_sources` | List configured sources with sync status |
| `trove_memory_store` | Store a key-value memory entry (with optional TTL and vector embedding) |
| `trove_memory_search` | Search memory by semantic similarity or keyword |
| `trove_memory_list` | List stored memories in a namespace |
| `trove_memory_delete` | Delete a memory entry |
| `trove_workflow_create` | Create a multi-step workflow DAG with dependencies |
| `trove_workflow_status` | Get workflow status and step progress |
| `trove_workflow_update` | Update a workflow step status (running/completed/failed/skipped) |
| `trove_workflow_list` | List all workflows with progress summaries |
| `trove_agents_list` | List available agent templates |
| `trove_agents_get` | Get full agent template details |
| `trove_run_agent` | Run an agent end-to-end: search context, apply system prompt, generate via local LLM |
| `trove_prompts_find` | Combined local index + live prompts.chat API search (deduplicated). Optional, see [Prompt Library](#prompt-library-optional) |
| `trove_prompts_search_live` | Search live prompts.chat API with type/category/tag/sort filters. Optional |
| `trove_prompts_get` | Fetch full prompt by ID with optional variable substitution. Optional |

### Example Usage in Claude Code

```
# List available documentation projects
Use trove_list_clients to see what's available

# Search API documentation
Use trove_search_client with client_name="stripe_api" and query="how to create a payment intent"

# Search currently loaded client
Use trove_search with query="authentication flow"
```

### Creating Skills That Use Trove

You can create custom Claude Code skills in other projects that query Trove for documentation context. See **[docs/CLAUDE_CODE_INTEGRATION.md](docs/CLAUDE_CODE_INTEGRATION.md)** for:

- Setting up global MCP configuration
- Creating project-specific skills
- Example skill templates
- Multi-client workflows

## Development

See `CLAUDE.md` for architecture details, code guidelines, and module reference.

### Running Tests

```bash
pip install -r requirements.txt -c constraints/py312.txt
pytest                         # the whole suite, several minutes
pytest tests/test_search.py    # one area
```

The suite downloads two small models from Hugging Face on first run (all-MiniLM-L6-v2 and ms-marco-MiniLM-L-6-v2). `tests/smoke/` holds checks for a built wheel installed in a clean environment.
