# Trove Product Roadmap

This page covers the product stages: extensibility, a REST API, a web UI, events and distribution. The near-term roadmap for Trove as a building block (stability and feature ports, package names, stores) is in [docs/REVIVAL-PLAN.md](docs/REVIVAL-PLAN.md), and [CHANGELOG.md](CHANGELOG.md) records what changed in each release.

Each stage is independently valuable and testable. One stage = one concept.

## Tech Stack

The stages below plan to use:

- Backend (Python): FastAPI, ChromaDB, sentence-transformers, tree-sitter, httpx, Pydantic, uvicorn
- Frontend (web): React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui, Zustand, TanStack Query
- Graph visualisation: D3.js or react-force-graph
- Markdown and code: react-markdown, Shiki (VS Code syntax grammar)
- Distribution: Docker, pip, PyInstaller

---

## Stage 0: Source Connectors (in place)

Trove fetches documents from GitHub repositories and websites, converts them, and keeps them up to date.

- GitHub connector (`core/sources/github_connector.py`): discovers repos from GitHub accounts and orgs, clones and fetches them, and tracks changed files. The API client handles rate limits, pagination and retries. Tokens come from environment variables only.
- Web crawler (`core/sources/web_crawler.py`): breadth-first crawl with scope control, HTML to Markdown conversion (markdownify and trafilatura), and an incremental cache (ETag and content hash).
- Shared interface: `BaseSourceConnector` with `sync()` and `status()`. Source files live in `clients/{name}/sources/{github|web}/` and are scanned alongside `documents/` during ingestion.
- MCP tools: `trove_connect_github`, `trove_crawl_web`, `trove_sync_sources`, `trove_list_sources`.
- CLI: `scripts/sources.py` with the `connect github`, `crawl`, `sync` and `list` subcommands.
- Ingest integration: the `sources/` folder is scanned during ingestion, `sync_on_ingest` in `client.yaml` syncs before each ingest, and `ingest.py --sync-sources` syncs on request.

---

## Stage 1: Plugin System (partly in place)

Formal extension points, so users can add embedding models, chunkers, parsers, rerankers, LLM providers and merge strategies without changing core code.

In place:

- Abstract base classes (`core/plugins/interfaces.py`) for six extension points: `BaseEmbedder`, `BaseChunker`, `BaseParser`, `BaseReranker`, `BaseLLMProvider`, `BaseMergeStrategy`.
- Plugin registry (`core/plugins/registry.py`): register and look up by name, with duplicate detection.
- Plugin loader (`core/plugins/loader.py`): discovers `.py` files and packages in a plugins folder, calls their `register()`, and logs and skips plugins that fail.
- Plugin authoring guide: [plugins/README.md](plugins/README.md).
- The built-in LLM providers (`core/llm.py`) and rerankers (`core/reranker.py`) implement the interfaces.

Planned:

- Call `load_plugins()` when the MCP server starts and when ingestion runs. Nothing calls it today, so plugins are not loaded.
- Make the remaining built-ins (embedders, chunkers, parsers, merge strategies) implement the interfaces, with no change in behaviour.

Depends on: nothing.

---

## Stage 2: REST API (planned)

A FastAPI HTTP server as the main programmatic interface. It serves the web UI as static files and exposes everything the MCP tools do as HTTP endpoints, with MCP as a thin adapter.

- FastAPI server at `localhost:8420`, with Pydantic request and response models and generated OpenAPI docs
- Static file serving: the built web UI at `/`, with SPA fallback routing
- Resource groups: `/clients`, `/search`, `/ingest`, `/graph`, `/memory`, `/workflows`, `/agents`, `/prompts`, `/sources`, `/webhooks`, `/health`, `/settings`, `/auth`
- Source management endpoints: CRUD for GitHub and web sources per client, trigger a sync, view status
- WebSocket endpoints for long-running operations (ingestion progress, graph builds) and a unified event stream
- Authentication: session-based, with a configurable password for remote access, off by default for local use, plus API keys for programmatic access
- MCP adapter: the MCP server calls the REST API when it is running, and imports `core/` directly when it is not
- CORS: configurable allowed origins for remote access (localhost by default)

Architecture: `Browser / MCP / CLI → REST API (FastAPI) → core/`

Depends on: Stage 1 (plugins must be loadable). The event system (Stage 8) can feed the WebSocket event stream later.

---

## Stage 3: Web UI Foundation (planned)

A React and TypeScript project with a navigation shell, dashboard, settings and API integration: the frame every screen plugs into, served as static files by the FastAPI backend.

- React 19, TypeScript and Vite
- Tailwind CSS and shadcn/ui
- API client: a TypeScript HTTP and WebSocket client for the REST API
- TypeScript models matching the API response shapes
- Navigation shell: a sidebar with Dashboard, Search, Clients, Graph, Chat and Settings
- Dashboard: client list with status, quick search, system status
- Settings: backend URL, LLM config, theme, licence, plugins
- State: Zustand for local state, TanStack Query for server state
- Dark mode by default, light mode available

Depends on: Stage 2.

---

## Stage 4: Search UI (planned)

The core search experience: query input, ranked results with quality indicators, source preview.

- Split-panel layout: query and filters on the left, ranked results on the right
- Search mode selector: precise, broad, code_first, exhaustive, diverse and the graph modes
- Client selector, with multi-client search
- Result cards: score badges, source file, chunk type, expandable full content
- Syntax highlighting for code chunks with Shiki
- Search quality footer: score distribution, timing, strategy, cache hit
- Search history, clickable to re-run
- Keyboard shortcuts: Enter to search, arrow keys to move, Escape to clear

Depends on: Stage 3.

---

## Stage 5: Client and Source Management (planned)

- Client list: cards with name, description, chunk count and status
- Client detail: a form-based config editor with validation and tooltips
- Document management: drag-and-drop upload, file browser, bulk import
- Source management: add and remove GitHub accounts and web crawl URLs, trigger a sync, view sync status and history
- Ingestion progress: a live progress bar over WebSocket
- Create-client wizard
- Quick actions: reindex, build graph, sync sources, delete (with confirmation)

Depends on: Stage 3.

---

## Stage 6: Graph Explorer (planned)

Interactive knowledge graph visualisation with entity search and relationship traversal.

- D3.js force-directed layout: nodes are entities, coloured by type; edges are labelled relationships
- Canvas or WebGL rendering for large graphs
- Zoom, pan and drag
- Entity search and filter by name or type
- Entity detail panel: description, relationships, source chunks
- "Search for this entity" jumps to the Search screen
- Data source: the D3.js-compatible graph export that the `trove_graph_export` MCP tool returns, served by the REST API

Depends on: Stage 3.

---

## Stage 7: Agent Chat (planned)

A conversational interface for running Trove agents.

- Agent selector, listing agents with their descriptions
- Client selector
- Chat interface with react-markdown rendering and Shiki code blocks
- Cited sources as collapsible references in agent responses
- Conversation history in browser storage, exportable as Markdown
- Per-conversation settings: `top_k`, "save to memory"

Depends on: Stage 3.

---

## Stage 8: Event System (planned)

An in-process publish/subscribe event bus in `core/` that fires on key actions, as the base for live WebSocket updates and webhooks.

- Event bus (`core/events.py`): core modules emit events; the REST API, MCP, CLI and webhooks can subscribe
- Events for client lifecycle, ingestion progress, search completion, graph builds and agent runs
- Webhook store (`core/webhooks.py`): registration and HMAC-SHA256 signed payloads, stored in the SQLite state database
- Webhook delivery: async HTTP callbacks with `httpx`, retried on failure
- Events fire whether the action came through REST, MCP or the CLI

Depends on: nothing. Integrates with Stage 2 (WebSocket endpoints) and Stages 4 to 7 (UI screens).

---

## Stage 9: Distribution Packaging (planned)

Installing Trove without a Python development environment.

- Docker image as the main server distribution: `docker run -p 8420:8420 trove`
- `pip install` with a `trove serve` command. Trove builds as a wheel today, with the `trove-server`, `trove-search`, `trove-ingest`, `trove-sources` and `trove-create-client` commands; it is not on PyPI.
- PyInstaller binary: a single executable bundling Python, the dependencies and the built web UI
- Build script (`build/export.py`) that exports the codebase to a staging folder, builds the web UI and compiles the binary
- Unified CLI: a `trove` command with subcommands (`ingest`, `search`, `serve`, `create-client` and so on)

Depends on: Stage 1 (plugins must load in a compiled binary). Stages 2 to 7 make the packaged product useful.

---

## Dependency Graph

```
0 (Source Connectors), in place, integrates with 1, 2, 5
1 (Plugins) → 2 (REST API) → 3 (Web UI Foundation)
                                    ↙  ↓  ↓  ↘
                                 4    5    6    7
                              (Search) (Clients) (Graph) (Chat)
                                    ↘  ↓  ↓  ↙
                                 8 (Events), real-time updates for the screens
                                       ↓
                                 9 (Distribution)
```

Stages 4 to 7 are independent of each other and can be built in any order or in parallel. Stage 8 can be built at any point, but gives the most value once the UI screens exist.

---

## Cross-Cutting Concerns

### Performance

- Embedding falls back to smaller batches and then to the CPU when the GPU runs out of memory (`core/ingest.py`). Moving models to the CPU between batches during ingestion is planned.
- The BM25 corpus hash samples content prefixes for indexes of 50,000 chunks or more, instead of hashing every chunk.

### LLM Configuration

- Providers: LM Studio (OpenAI-compatible, native and Anthropic-compatible endpoints) and Claude through the Anthropic client. Ollama and other OpenAI-compatible servers are planned.
- All LLM tasks for a client share one model, `llm.lmstudio_model`. Per-task model selection (for example a different model for graph extraction than for answers) is planned.

### Documentation

- Each change updates README.md, CLAUDE.md and the docs it affects, in the same pull request.

### Compatibility

- New features are opt-in through `client.yaml`.
- Existing clients work without config changes.
- New config keys default to the existing behaviour.
