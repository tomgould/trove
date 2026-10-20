# Trove Roadmap

Stages 0-11 are complete — see [CHANGELOG.md](CHANGELOG.md) for details. This roadmap covers the next phase: extensibility, API, web UI and distribution.

Each stage is independently valuable and testable. One stage = one concept.

## Tech Stack

**Backend (Python):** FastAPI, ChromaDB, sentence-transformers, tree-sitter, httpx, Pydantic, uvicorn
**Frontend (Web):** React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui, Zustand, TanStack Query
**Graph Viz:** D3.js / react-force-graph
**Markdown/Code:** react-markdown, Shiki (VS Code syntax grammar)
**Distribution:** Docker, pip, PyInstaller

---

## Stage 0: Source Connectors ✅

**Status:** Complete (2026-03-09)

**Goal:** Automated document acquisition from GitHub repositories and websites. Trove discovers, fetches, converts, and keeps documents up to date from external sources.

**Design:** not published yet.

**Implemented:**
- **GitHub Connector** (`core/sources/github_connector.py`) — discover repos from GitHub accounts/orgs, clone/fetch, track changed files. API client with rate limiting, pagination, retries. Tokens via env vars only.
- **Web Crawler** (`core/sources/web_crawler.py`) — BFS crawl with scope control, HTML→Markdown conversion (markdownify + trafilatura), incremental cache (ETag/content hash).
- **Shared interface** — `BaseSourceConnector` with `sync()` and `status()`. Source files at `clients/{name}/sources/{github|web}/`, scanned alongside `documents/` during ingestion.
- **MCP tools** — `trove_connect_github`, `trove_crawl_web`, `trove_sync_sources`, `trove_list_sources`
- **CLI** — `scripts/sources.py` with `connect github`, `crawl`, `sync`, `list` subcommands
- **Ingest integration** — `sources/` directory auto-scanned during ingestion, `sync_on_ingest` config flag, `--sync-sources` CLI flag
- **Tests** — 75+ tests covering connectors, error paths, and integration

**Depends on:** Nothing (self-contained). Optionally integrates with Stage 1 (source connectors as plugins), Stage 3 (events on sync), Stage 2 (REST endpoints).

---

## Stage 1: Plugin System (In Progress)

**Goal:** Formal extension points so users can add custom embedding models, chunkers, parsers, rerankers, LLM providers, and merge strategies without modifying core code.

**Design:** not published yet.

**Implemented:**
- **Abstract base classes** (`core/plugins/interfaces.py`) for 6 extension points: BaseEmbedder, BaseChunker, BaseParser, BaseReranker, BaseLLMProvider, BaseMergeStrategy
- **Plugin registry** (`core/plugins/registry.py`) — register/lookup by name, duplicate detection
- **Plugin loader** (`core/plugins/loader.py`) — discovers `.py` files and packages in `plugins/`, calls `register()`, graceful error handling
- **Plugin authoring guide** — `plugins/README.md` with examples for each extension point
- **Tests** — 33 tests covering interfaces, registry, and loader

**Remaining:**
- **Refactor built-ins** to implement the same interfaces (no behaviour change, just formalising)
- **Wire loader into startup** — call `load_plugins()` when MCP server starts or ingestion runs

**Depends on:** Nothing (self-contained)

---

## Stage 2: REST API

**Goal:** FastAPI HTTP server as the primary programmatic interface. Serves the web UI as static files. All MCP tool functionality exposed as HTTP endpoints. MCP becomes a thin adapter.

**Design:** not published yet.

- **FastAPI server** at `localhost:8420` — Pydantic request/response models, OpenAPI/Swagger docs auto-generated
- **Static file serving** — built web UI served at `/`, SPA fallback routing for client-side routes
- **Resource groups:** `/clients`, `/search`, `/ingest`, `/graph`, `/memory`, `/workflows`, `/agents`, `/prompts`, `/sources`, `/webhooks`, `/health`, `/settings`, `/auth`
- **Source management endpoints** — CRUD for GitHub and web sources per client, trigger sync, view status
- **WebSocket endpoints** for long-running operations (ingestion progress, graph builds) and a unified event stream
- **Authentication** — session-based auth with configurable password for remote access, disabled by default for local use. Also supports API key for programmatic access.
- **MCP adapter** — MCP server calls REST API when server is running, falls back to direct `core/` imports when not
- **CORS** — configurable allowed origins for remote access (defaults to localhost)

**Architecture:** `Browser / MCP / CLI → REST API (FastAPI) → core/`

**Depends on:** Stage 1 (plugins must be loadable). Event System (Stage 8) can be integrated later for WebSocket event streams.

---

## Stage 3: Web UI Foundation

**Goal:** React + TypeScript project with navigation shell, dashboard, settings, and API integration. The frame that all screens plug into. Served as static files from the FastAPI backend.

**Design:** not published yet.

- **React 19 + TypeScript + Vite** — fast builds, modern tooling
- **Tailwind CSS + shadcn/ui** — utility-first styling with high-quality accessible components
- **API client** — TypeScript HTTP client + WebSocket client for the Trove REST API
- **TypeScript models** — interfaces matching all API response shapes
- **Navigation shell** — sidebar with screen switching (Dashboard, Search, Clients, Graph, Chat, Settings)
- **Dashboard screen** — client list with status, quick search, system status
- **Settings screen** — backend URL, LLM config, theme, license, plugins
- **State management** — Zustand for local state, TanStack Query for server state
- **Dark mode first** — developer tool, dark by default, light mode available

**Depends on:** Stage 2 (REST API must exist for the web UI to talk to)

---

## Stage 4: Search UI

**Goal:** The core search experience — query input, ranked results with quality indicators, source preview.

**Design:** not published yet.

- **Split-panel layout** — query input + filters on the left, ranked results on the right
- **Search mode selector** — precise, broad, code_first, exhaustive, diverse, graph modes
- **Client selector** — dropdown or tabs, multi-client search support
- **Result cards** — score badges, source file, chunk type, expandable full content
- **Syntax highlighting** — Shiki (VS Code grammar) for code chunks
- **Search quality footer** — score distribution, timing, strategy, cache hit
- **Search history** — clickable to re-run
- **Keyboard shortcuts** — Enter to search, arrow keys to navigate, Escape to clear

**Depends on:** Stage 3 (web UI foundation must exist)

---

## Stage 5: Client & Source Management

**Goal:** CRUD interface for clients with document management, source connector management, ingestion progress, and a creation wizard.

**Design:** not published yet.

- **Client list** — cards with name, description, chunk count, status badges
- **Client detail** — form-based config editor (not raw YAML) with validation and tooltips
- **Document management** — drag-and-drop upload, file browser, bulk import
- **Source management** — add/remove GitHub accounts and web crawl URLs, trigger sync, view sync status and history
- **Ingestion progress** — real-time progress bar via WebSocket
- **Create client wizard** — step-by-step flow
- **Quick actions** — reindex, build graph, sync sources, delete (with confirmation)

**Depends on:** Stage 3 (web UI foundation must exist)

---

## Stage 6: Graph Explorer

**Goal:** Interactive knowledge graph visualisation with entity search and relationship traversal.

**Design:** not published yet.

- **D3.js force-directed layout** — nodes = entities (colour-coded by type), edges = labelled relationships
- **Canvas rendering** — WebGL/Canvas for performance with large graphs
- **Zoom, pan, drag** — interactive manipulation
- **Entity search/filter** — find by name or type
- **Entity detail panel** — click a node to see description, relationships, source chunks
- **Cross-navigation** — "Search for this entity" jumps to Search screen
- **Data source:** `GET /api/v1/graph/export` (D3.js-compatible JSON, already exists)

**Depends on:** Stage 3 (web UI foundation must exist)

---

## Stage 7: Agent Chat

**Goal:** Conversational interface for running Trove agents with context-aware responses.

**Design:** not published yet.

- **Agent selector** — list from agents API, with description
- **Client selector** — which knowledge base to search
- **Chat interface** — message bubbles, react-markdown rendering, Shiki code blocks with syntax highlighting
- **Cited sources** — inline collapsible source references in agent responses
- **Conversation history** — persisted in browser localStorage, exportable as markdown
- **Settings per conversation** — top_k, "save to memory" toggle

**Depends on:** Stage 3 (web UI foundation must exist)

---

## Stage 8: Event System

**Goal:** Internal pub/sub event bus in `core/` that fires on key actions. Foundation for real-time WebSocket updates in the web UI and webhook integrations.

**Design:** not published yet.

- **Event bus** (`core/events.py`) — simple publish/subscribe within the process. Core modules emit events; any consumer (REST API, MCP, CLI, webhooks) can subscribe.
- **11 events** covering: client lifecycle, ingestion progress, search completion, graph builds, agent runs
- **Webhook store** — registration and delivery mechanism in `core/webhooks.py`, HMAC-SHA256 signed payloads, stored in SQLite state DB
- **Webhook delivery** — async HTTP callbacks via `httpx`, retry on failure
- **Retrofit** — once built, wire into existing REST API WebSocket endpoints and UI screens for real-time updates
- Events fire regardless of whether the action was triggered via REST, MCP, or CLI

**Depends on:** Nothing (self-contained). Integrates with Stage 2 (REST API WebSocket endpoints) and Stages 4-7 (UI screens) retroactively.

---

## Stage 9: Distribution Packaging

**Goal:** Package Trove for installation without requiring a Python development environment.

**Design:** not published yet.

- **Docker image** — primary distribution for servers (AWS, self-hosted). `docker run -p 8420:8420 trove`
- **pip install** — `pip install trove` for Python users, `trove serve` to start
- **PyInstaller binary** — single-file executable for non-Python users. Bundles Python runtime + all deps + built web UI.
- **Build script** (`build/export.py`) — exports generic codebase to staging dir, builds web UI, compiles binary. Your local repo never modified.
- **Unified CLI** — `trove` binary with subcommands (`ingest`, `search`, `serve`, `create-client`, etc.)

**Depends on:** Stage 1 (plugins must be loadable in compiled binary). Stages 2-7 should be complete for the packaged product to be useful.

---

## Dependency Graph

```
0 (Source Connectors) ✅ — independent, integrates with 1, 2, 5
1 (Plugins) → 2 (REST API) → 3 (Web UI Foundation)
                                    ↙  ↓  ↓  ↘
                                 4    5    6    7
                              (Search) (Clients) (Graph) (Chat)
                                    ↘  ↓  ↓  ↙
                                 8 (Events) — retrofit real-time into screens
                                       ↓
                                 9 (Distribution)
```

Stages 4-7 are independent of each other — build in any order or in parallel.
Stage 8 (Events) can be built at any point, but maximum value comes after the UI screens exist.

---

## Cross-Cutting Concerns

These apply across all stages and should be addressed incrementally:

### Performance

- GPU memory management during ingestion (move models to CPU between batches)
- BM25 corpus hash optimization for large indices (sampling instead of full hash)

### LLM Configuration

- Per-task model selection (e.g. different model for graph extraction vs search generation)
- Support for Claude API, Ollama, and OpenAI-compatible providers (not just LM Studio)
- Currently all LLM tasks share a single `lmstudio_model` per client

### Documentation

- Keep CLAUDE.md in sync with new features as they're added
- Update README.md with new CLI tools and MCP tools

### Backwards Compatibility

- All new features are opt-in via `client.yaml`
- Existing clients work without config changes
- New config keys have sensible defaults matching current behavior

### Rename: TRAG → Trove ✅

- Renamed all code, MCP tools (`trove_` prefix), env vars, state DB paths
- Created public `trove` repo with `sync-to-public.sh` export script
- Private repo retained for client data and development
