# Revival plan

Trove is active again. This repository is the single source of truth, and everything is developed here in the open.

The first driver is a research workspace that scrapes public websites with ScrapePy, then makes the pages searchable with Trove, with many separate projects each holding their own index. That use exposed what is missing from Trove as a building block, and the plan below fixes those things first.

A more mature sibling project grew out of Trove and has solved several problems Trove still has. It is used as a read-only reference. Trove does not depend on it, and nothing in this repo points back at it.

## Principles

1. One repo, one history. Nothing private lives here. Commercial plans, machine-specific scripts and client data stay out.
2. Every change ships with tests, and CI must be green before the next phase starts.
3. On-disk formats do not change without a migration note in the CHANGELOG.
4. Features from the sibling project are re-implemented in Trove's structure with their own tests. They are not copied wholesale.
5. The code keeps the word "client". Documentation calls a client a "project" where that reads better.
6. The licence is unchanged for now.

## Phases

### 0. Baseline (done)

Install on a clean environment with the CPU build of PyTorch, run the test suite and record what passes. Note the installed versions of chromadb, sentence-transformers, unstructured, tree-sitter, mcp and torch.

Done when: results and versions are written to `docs/BASELINE.md`.

Done: 723 passed and 1 skipped on Python 3.12 and 3.10, recorded in `docs/BASELINE.md`.

### 1. Make it healthy (done)

- [x] Dependencies: floors replaced by tested ranges, constraints files per Python version, and extras (`docs`, `llm`, `mcp`, `code`, `dev`, `all`) so the core install does not need every heavy package. Python 3.11 to 3.13 are supported. Code indexing is not available on 3.13 because tree-sitter-languages has no wheels for it.
- [x] Fix the known defects, each with a regression test:
  - [x] the `trove-server` entry point points at a `main()` that does not exist
  - [x] the `source_filter` option compares the wrong metadata key and never matches
  - [x] the "no index" checks can never fire
  - [x] `load_client` creates folders for a mistyped client name
  - [x] the README refers to `clients/prompts/api_client.py`, which is not in the repo
- [x] Split the single large test file by area.
- [x] Package so a wheel installs and the entry points run. Pin GitHub Actions by commit.

Done when: CI is green on three Python versions and a built wheel passes a smoke test.

#### Follow-up: rename the top-level packages

The wheel installs three top-level packages with generic names: `core`, `scripts` and `mcp_server`. Any other project that ships a `core` or `scripts` package will overwrite them, or be overwritten, in the same environment, and `import core` says nothing about where it came from. They should move under a single `trove` package (`trove.core`, `trove.cli`, `trove.mcp_server`) before Trove is published to PyPI. It touches every import in the code and tests and the paths in the docs and MCP configuration examples, so it is a change of its own rather than part of Phase 1, and it needs a short deprecation period in which the old module paths still work.

### 2. Use it as a building block

- `TROVE_CLIENTS_DIR` (and `--clients-dir`) so clients can live outside the repo.
- Read YAML frontmatter on Markdown files into chunk metadata, and show title and URL in CLI, `--json` and MCP results so answers can cite their source.
- Create a client non-interactively: `create_client.py --name X --documents-dir PATH`.
- Search several clients from the CLI.
- Skip unchanged files on ingest using a manifest, so a scraper can run on a schedule.

Done when: a workspace outside this repo can create, ingest, search and cite with no symlinks or hand-edited files.

### 3. Stability ports

Re-implement fixes the sibling project has already proven:

- serialise embedder inference to stop concurrent reindex hangs
- the reindex worker-pool hang
- the Chroma settings clash and the client singleton
- pruning orphaned chunks, so deleted pages leave the index
- logging for silent failures
- crawler redirect and address checks

Each gets a regression test written for Trove.

Done when: a reindex under concurrent search completes, and deleted files disappear from results.

### 4. Feature ports

In order of value:

1. BM25 cache on disk in a compact format, with a tokenizer version so stale caches rebuild.
2. Incremental code index state and resumable ingest.
3. Recency and publication-date ranking, for news-style sources.
4. Web sources with a crawl cache, and the web crawler fixes.
5. Incremental graph saves and pruning.
6. More language extractors.

Done when: each item has tests and a CHANGELOG entry.

### 5. Stores

Today vectors, keywords, the graph and memories are hard-wired to ChromaDB, an in-memory BM25 index, NetworkX with SQLite, and SQLite. Design interfaces for each store with the current ones as defaults, so another backend can be added without touching search. Settle the design with real use cases before writing code.

Done when: a short design note is agreed. Implementation follows as its own plan.

## Out of scope for now

The REST API, the desktop UI and the plugin system in `ROADMAP.md` stay where they are.

## Risks

- Dependency drift. Phase 0 showed it: with floors only, Python 3.10 resolved much older unstructured and numpy releases than 3.12. Phase 1 replaced the floors with tested ranges and per-version constraints files; keep them current when dependencies change.
- Ports done by hand can drift from their source. Each port records what behaviour it matches in its test names.
- Old indexes built with older Chroma versions may not open. Rebuilding is the supported path.
