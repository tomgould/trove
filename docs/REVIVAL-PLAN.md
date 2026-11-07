# Roadmap

This page covers Trove as a building block: what it does today and the work planned next. All development happens in this repository, in the open. The longer-term product stages (REST API, web UI, event system, distribution) are in [ROADMAP.md](../ROADMAP.md), and [CHANGELOG.md](../CHANGELOG.md) records what changed in each release.

The main use it serves is a research workspace that scrapes public websites into Markdown and searches them with Trove, with many separate projects, each holding its own index.

## Principles

1. One repo, one history. Nothing private lives here: no commercial plans, machine-specific scripts or client data.
2. Every change ships with tests, and CI is green before the next item starts.
3. On-disk formats do not change without a migration note in the CHANGELOG.
4. Fixes and features ported from a more mature sibling project are re-implemented in Trove's structure with their own tests, not copied wholesale. Trove does not depend on that project.
5. The code says "client". Documentation calls a client a "project" where that reads better.
6. Trove is licensed under the [Business Source License 1.1](../LICENSE).

## What Trove does today

### A healthy base

- Python 3.11, 3.12 and 3.13 are supported. CI runs the full test suite on all three. Code indexing needs tree-sitter-languages, which has no wheels for 3.13, so on 3.13 the `code` extra installs nothing and the code-indexing tests that need a parser are skipped.
- `requirements.txt` and `pyproject.toml` give the tested version ranges, and `constraints/py311.txt`, `py312.txt` and `py313.txt` hold the exact versions from a working install. Extras (`docs`, `llm`, `mcp`, `code`, `dev`, `all`) keep the core install small.
- Trove builds as a wheel. CI builds it, installs it without extras in a clean environment and runs smoke tests for the core install and the MCP server.
- GitHub Actions are pinned by commit.
- Tests are split into one file per area, and each fixed defect has a regression test.

### Use with a scraper

The README's [Using Trove with a scraper](../README.md#using-trove-with-a-scraper) section has the details.

- Clients can live outside the repository: `TROVE_CLIENTS_DIR` or `--clients-dir` points Trove at any folder, and folders starting with `_` or `.` (such as a `_template/`) are not clients.
- `create_client.py --name NAME --documents-dir PATH` creates a client without questions.
- Markdown files are split at their headings. YAML frontmatter (`title`, `url`, `date`, `domain`, `page_type`, `scraped_at`, `content_hash`) becomes chunk metadata, each chunk records its heading path as `section`, and image syntax is stripped. Title, URL and section appear in CLI, `--json` and MCP results, so answers can cite their source.
- Markdown section chunks are never ranked as data, so a listing page cannot jump above the page that answers the question.
- `search.py --clients a,b` and `search.py --all` search several clients and merge the results by rank.
- Ingest is incremental. A manifest with each file's SHA-256 skips unchanged files, including files a scraper rewrote with identical content. Chunks of changed and deleted files are removed, and `ingest.py --json` reports new, changed, unchanged, deleted and failed files.

## Planned

### Stability ports

Fixes the sibling project has already proven, each with a regression test written for Trove:

- Serialised embedder inference, to stop concurrent reindex hangs. The shared model cache has no lock around inference.
- A fix for the reindex worker-pool hang.
- A fix for the Chroma settings clash and the Chroma client singleton.
- Trove does not prune orphan chunks (chunks whose file is not in the manifest, for example after an interrupted run); pruning is planned. The incremental run removes chunks of deleted files.
- Logging for failures that pass silently.
- The web crawler follows redirects without checking where they lead or the target address; redirect and address checks are planned.

Done when: a reindex under concurrent search completes, and orphan chunks disappear from results.

### Feature ports

In order of value:

1. A compact on-disk BM25 cache. Trove keeps the cache as JSON with a tokenizer version, so a stale cache rebuilds.
2. The sibling project's incremental code index state and resumable ingest, in place of Trove's own (`core/code/incremental.py`, `ingest.py --resume`).
3. Recency and publication-date ranking, for news-style sources.
4. The sibling project's web source handling, crawl cache and web crawler fixes.
5. Incremental graph saves and graph pruning.
6. More language extractors.

Done when: each item has tests and a CHANGELOG entry.

### Package names

The wheel installs three top-level packages with generic names: `core`, `scripts` and `mcp_server`. Another project that ships a `core` or `scripts` package overwrites them, or is overwritten, in the same environment, and `import core` says nothing about where it came from. They move under a single `trove` package (`trove.core`, `trove.cli`, `trove.mcp_server`) before Trove is published to PyPI. The change touches every import in the code and tests and the paths in the docs and MCP configuration examples, so it is a change of its own, with a short deprecation period in which the old module paths still work.

### Stores

Vectors, keywords, the graph and memories are hard-wired to ChromaDB, an in-memory BM25 index, NetworkX with SQLite, and SQLite. The plan is an interface for each store, with the current ones as defaults, so another backend can be added without touching search. The design is settled against real use cases before any code is written.

Done when: a short design note is agreed. Implementation follows as its own plan.

## Out of scope

The REST API, the web UI and the rest of the product stages in [ROADMAP.md](../ROADMAP.md) are not part of this roadmap.

## Risks

- Dependency drift. With lower bounds only, different Python versions resolve very different releases of unstructured and numpy. The tested ranges and per-version constraints files guard against it and need updating whenever a dependency changes.
- Ports done by hand can drift from their source. Each port's test names record which behaviour it matches.
- Indexes built with older Chroma versions may not open. Rebuilding is the supported path.
