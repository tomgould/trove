---
description: Search project documentation via the Trove MCP server (supports multiple clients/projects)
user-invocable: true
---

# Trove Documentation Search

<skill-content>
You have access to a local RAG (Retrieval-Augmented Generation) system via the Trove MCP server. This system hosts documentation across multiple project clients — not just one. Always check which clients are available before searching.

## Discovering Available Clients

Before searching, list the available Trove clients to see what documentation is indexed:

```
trove_list_clients()
```

This will return all installed clients (e.g. myproject, infrastructure, platform, etc.). Use the correct `client_name` for your searches.

## How to Search

### Standard Search
```
trove_search_client(
    client_name="<client_name>",
    query="<relevant search terms>",
    top_k=15
)
```

### Search with Mode Presets
Add a `mode` parameter to tune the search strategy for the query type:

```
trove_search_client(
    client_name="<client_name>",
    query="<query>",
    top_k=15,
    mode="precise"
)
```

Available modes:
- **`precise`** — High keyword weight, strict scoring, top 10. Best for exact terms, names, IDs.
- **`broad`** — High semantic weight, no min score, top 30. Best for conceptual or exploratory queries.
- **`code_first`** — Boost code chunks, cross-reference code with docs. Best for implementation questions.
- **`exhaustive`** — RRF merge, deep reranking, top 50. Best when you need thorough coverage.
- **`diverse`** — MMR diversity to reduce redundancy. Best when results seem repetitive.
- **`graph_local`** — Entity-focused graph search. Best for finding chunks about specific entities.
- **`graph_global`** — Relationship traversal. Best for multi-hop reasoning across entities.
- **`graph_hybrid`** — Local + global graph combined. Best for comprehensive graph coverage.
- **`graph_mix`** — Graph + standard hybrid merged via RRF. Best overall graph-augmented search.

Note: Graph modes (`graph_*`) require a knowledge graph to be built for the client. Use `trove_graph_stats` to check, or `trove_build_graph` to build one.

### Search with Token Budget
Use `max_tokens` to limit how much context is returned:

```
trove_search_client(
    client_name="<client_name>",
    query="<query>",
    top_k=20,
    max_tokens=2000
)
```

### Cross-Client Search
Search multiple clients at once and get merged results:

```
trove_search_multi(
    query="<query>",
    clients=["client_a", "client_b"],
    top_k=20
)
```

Leave `clients` empty to search ALL clients with indices. Results are merged using rank fusion and tagged with their source client.

### Quick Overview (Summaries)
Get concise 2-3 sentence summaries instead of full content:

```
trove_summarize(
    query="<query>",
    client_name="<client_name>",
    top_k=5
)
```

Useful for getting an overview before diving deep with a full search.

## Knowledge Graph Tools

If a client has a knowledge graph built, you can explore entities and relationships directly:

### Search Entities
```
trove_search_entities(query="<entity name>", client_name="<client_name>", top_k=10)
```

### Traverse Relationships
```
trove_search_relationships(entity_name="<entity key>", client_name="<client_name>", max_hops=2)
```

### Graph Statistics
```
trove_graph_stats(client_name="<client_name>")
```

### Build/Rebuild Graph
```
trove_build_graph(client_name="<client_name>")
trove_build_graph(client_name="<client_name>", rebuild=true)
```

## Ingestion Tools

You can ingest new content into a client's index:

### Ingest a Single File
```
trove_ingest_file(client_name="<client_name>", file_path="/path/to/file")
```

### Ingest from URL
```
trove_ingest_url(client_name="<client_name>", url="https://example.com/page")
```

### Trigger Full Reindex
```
trove_reindex_client(client_name="<client_name>")
trove_ingest_status(client_name="<client_name>")  # Check progress
```

## Search Strategy Guide

| Scenario | Approach |
|----------|----------|
| Know the exact term | `mode="precise"` |
| Exploring a topic | `mode="broad"` or `trove_summarize` first |
| Need code examples | `mode="code_first"` |
| Question spans multiple projects | `trove_search_multi` |
| Results are repetitive | `mode="diverse"` |
| Need comprehensive coverage | `mode="exhaustive"` |
| Context window is tight | Add `max_tokens=2000` |
| Need entity/relationship info | `mode="graph_mix"` or `trove_search_entities` |
| Multi-hop reasoning needed | `mode="graph_global"` |

## When to Search

Search Trove when the user asks about:

- API endpoints, authentication, parameters, or integration details
- How-to guides, workflows, or implementation patterns
- Code examples or configuration references
- Architecture decisions or project context
- Differences between providers, services, or approaches within a project
- Relationships between concepts, people, or organizations
- Anything that sounds project-specific rather than general knowledge

## Response Format

When answering based on search results:

1. **Cite the source** — Mention which client and document the information comes from
2. **Include code examples** — If the docs contain code, include relevant snippets
3. **Note differences** — If results show varying approaches, highlight the distinctions
4. **Link related concepts** — Connect related topics the user may want to explore next

## Multi-Query Approach

For complex questions, search multiple times:

1. First use `trove_summarize` to get an overview
2. Then do a full `trove_search_client` with `mode="precise"` for specific details
3. Finally search for code examples with `mode="code_first"`
4. For entity exploration, use `trove_search_entities` then `trove_search_relationships`
</skill-content>
