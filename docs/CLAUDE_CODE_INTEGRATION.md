# Trove + Claude Code Integration Guide

This guide explains how to integrate Trove with Claude Code so you can query your indexed documentation from any project.

## Overview

The Trove MCP server exposes your indexed documents to Claude Code. Once configured, you can:

1. Query API documentation while coding against that API
2. Search project-specific knowledge bases
3. Search across multiple projects simultaneously
4. Get quick overviews before diving deep
5. Control search strategy and context budget per query
6. Create custom skills that automatically use Trove for context

## Step 1: Configure the MCP Server

### Global Configuration (all projects)

Add the Trove MCP server to your global Claude Code settings:

**File: `~/.mcp.json`**

```json
{
  "mcpServers": {
    "trove": {
      "command": "python",
      "args": ["/path/to/trove/scripts/mcp_server.py"],
      "env": {
        "TROVE_DEFAULT_CLIENT": "your_default_client",
        "TROVE_CACHE_SIZE": "3"
      }
    }
  }
}
```

See `docs/mcp.json.example` for a template.

### Project-Specific Configuration

For project-specific Trove clients, add an `.mcp.json` to your project root:

**File: `your-project/.mcp.json`**

```json
{
  "mcpServers": {
    "trove": {
      "command": "python",
      "args": ["/path/to/trove/scripts/mcp_server.py"],
      "env": {
        "TROVE_DEFAULT_CLIENT": "your_api_client"
      }
    }
  }
}
```

This sets a default client for that project, so `trove_search` works without specifying the client.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TROVE_DEFAULT_CLIENT` | (none) | Client to auto-load on startup |
| `TROVE_CACHE_SIZE` | `3` | Max number of searchers to keep cached |
| `ANTHROPIC_API_KEY` | (none) | Required if using Claude provider for summarization |

## Step 2: Available MCP Tools

### Core Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `trove_list_clients` | — | List all indexed documentation clients |
| `trove_get_client` | — | Get info about currently loaded client |
| `trove_load_client` | `client_name` | Load a specific client |

### Search Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `trove_search` | `query`, `top_k?`, `include_content?`, `mode?`, `max_tokens?` | Search loaded client |
| `trove_search_client` | `client_name`, `query`, `top_k?`, `mode?`, `max_tokens?` | Load + search in one call |
| `trove_search_multi` | `query`, `clients?`, `top_k?`, `mode?`, `max_tokens?` | Search across multiple clients |
| `trove_summarize` | `query`, `top_k?`, `client_name?` | Search and get concise summaries |

### Code Search Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `trove_search_code` | `query`, `languages?`, `symbol_types?`, `top_k?` | Search code chunks only |
| `trove_search_repo` | `repo_name`, `query`, `chunk_types?`, `top_k?` | Search within a specific repo |
| `trove_find_symbol` | `symbol_name`, `symbol_type?`, `language?` | Find a symbol by exact name |
| `trove_list_repos` | — | List detected repositories |

### Ingestion Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `trove_ingest_file` | `client_name`, `file_path` | Ingest a single file into a client's index |
| `trove_ingest_url` | `client_name`, `url` | Fetch and ingest content from a URL |
| `trove_reindex_client` | `client_name` | Trigger a full background reindex |
| `trove_ingest_status` | `job_id?`, `client_name?` | Check ingestion job progress/status |

### Knowledge Graph Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `trove_graph_stats` | `client_name?` | Get entity/relationship counts, top entities, density |
| `trove_graph_export` | `client_name?`, `format?`, `max_nodes?` | Export graph as D3.js-compatible JSON |
| `trove_search_entities` | `query`, `entity_type?`, `top_k?`, `client_name?` | Search knowledge graph entities by name |
| `trove_search_relationships` | `entity_name`, `relationship_type?`, `max_hops?`, `client_name?` | Traverse relationships from an entity |
| `trove_build_graph` | `client_name`, `rebuild?`, `sample?` | Build/rebuild knowledge graph (background) |

### Prompt Discovery Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `trove_prompts_find` | `query`, `type?`, `top_k?` | Combined local index + live prompts.chat API search (deduplicated) |
| `trove_prompts_search_live` | `query?`, `type?`, `category?`, `tag?`, `sort?`, `limit?` | Search live prompts.chat API with filters |
| `trove_prompts_get` | `prompt_id`, `variables?` | Fetch full prompt by ID with optional `${Var:Default}` substitution |

Use `/prompt-finder <task>` as a skill shortcut for the full discovery workflow.

### Search Mode Presets

The `mode` parameter is available on `trove_search`, `trove_search_client`, and `trove_search_multi`:

| Mode | Strategy | Best For |
|------|----------|----------|
| `precise` | High BM25 weight (0.7), strict scoring, top 10 | Exact terms, names, IDs |
| `broad` | High vector weight (0.7), no min score, top 30 | Conceptual or exploratory queries |
| `code_first` | Boost code chunks, cross-reference code | Implementation questions |
| `exhaustive` | RRF merge, no adaptive rerank, top 50 | Thorough coverage |
| `diverse` | MMR diversity reranking (lambda=0.7) | Reducing redundant results |
| `graph_local` | Entity-focused graph search | Finding entity source chunks |
| `graph_global` | Relationship traversal graph search | Multi-hop reasoning |
| `graph_hybrid` | Local + global graph search combined | Comprehensive graph coverage |
| `graph_mix` | Graph + standard hybrid merged via RRF | Best overall graph-augmented search |

### Context Budget

The `max_tokens` parameter limits how much content is returned. Results are trimmed to fit within the budget, and a token summary line is appended:

```
**Tokens:** ~1500/2000 (showing 8/20 results)
```

## Step 3: Create Skills

Skills are markdown files that tell Claude how to handle specific tasks. Place them in `~/.claude/skills/` for global access or `your-project/.claude/skills/` for project-specific use.

### Generic Trove Skills

Copy the example skills from `docs/skills/` in the Trove repo:

```bash
# Copy generic skills to global location
cp /path/to/trove/docs/skills/trove-*.md ~/.claude/skills/
```

Available example skills:

| Skill | File | Description |
|-------|------|-------------|
| Documentation Search | `trove-docs.md` | Full-featured search with modes, budgets, multi-client |
| Quick Search | `trove-query.md` | Quick `client_name + query` search |
| Overview | `trove-overview.md` | Summarized results for quick scanning |
| Multi-Client | `trove-multi.md` | Cross-client search |
| Prompt Finder | `prompt-finder.md` | Find AI prompt templates from 10k+ indexed + live API |

### Custom Skill Template

Create your own skill for a specific project:

**File: `your-project/.claude/skills/api-docs.md`**

```markdown
---
description: Search API documentation for this project
user-invocable: true
---

# API Documentation Search

<skill-content>
When the user asks about API endpoints, parameters, authentication, or how to use specific API features, use the Trove MCP server to search the indexed documentation.

## Instructions

1. For a quick overview:
   ```
   trove_summarize(query="<user's question>", client_name="your_api", top_k=5)
   ```

2. For detailed results:
   ```
   trove_search_client(
       client_name="your_api",
       query="<user's question>",
       top_k=10,
       mode="precise"
   )
   ```

3. For code examples:
   ```
   trove_search_client(
       client_name="your_api",
       query="<user's question>",
       top_k=10,
       mode="code_first"
   )
   ```

4. Cite the source documents in your response
</skill-content>
```

## Step 4: Using Skills

Once configured, invoke skills from Claude Code:

```
/trove-docs
How do webhooks work?

/trove-query myproject "token authentication"

/trove-overview authentication flow

/trove-multi "error handling patterns"
```

## Complete Example: Setting Up for a New API

### 1. Create Trove Client

```bash
cd /path/to/trove
mkdir -p clients/twilio_api/{documents,prompts}
```

Create `clients/twilio_api/client.yaml`:
```yaml
name: "Twilio API"
machine_name: twilio_api
description: "Twilio API documentation"
```

### 2. Add Documentation

```bash
cp ~/twilio-docs/*.md clients/twilio_api/documents/
```

### 3. Ingest Documents

```bash
python scripts/ingest.py --client twilio_api --rebuild
```

### 4. Create Project Skill

**File: `~/projects/my-twilio-app/.claude/skills/twilio-docs.md`**

```markdown
---
description: Search Twilio API documentation
user-invocable: true
---

# Twilio Documentation

<skill-content>
When answering questions about Twilio APIs, SMS, voice calls, or any Twilio functionality:

1. For a quick overview:
   ```
   trove_summarize(query="<user's question>", client_name="twilio_api", top_k=5)
   ```

2. For detailed answers:
   ```
   trove_search_client(client_name="twilio_api", query="<user's question>", top_k=10)
   ```

3. For code examples:
   ```
   trove_search_client(client_name="twilio_api", query="<query>", mode="code_first")
   ```

4. Always cite which documentation page the information came from.
</skill-content>
```

### 5. Use It

```
cd ~/projects/my-twilio-app
claude

> /twilio-docs
> How do I send an SMS with a callback URL?
```

## Tips

1. **Index Quality Matters**: Better markdown = better search results. Clean up scraped docs before indexing.

2. **Use Search Modes**: `precise` for exact lookups, `broad` for exploration, `code_first` for implementation, `diverse` when results are repetitive.

3. **Control Context Budget**: Use `max_tokens` when working with limited context windows or when you only need a quick reference.

4. **Start with Summaries**: Use `trove_summarize` first to get an overview, then search for specific details.

5. **Cross-Client Search**: Use `trove_search_multi` when a question spans multiple projects instead of searching each client manually.

6. **Cache Benefits**: The searcher cache (default size 3) means switching between recently used clients is instant — no reload penalty.

7. **Update Regularly**: Re-run ingestion when documentation updates:
   ```bash
   python scripts/ingest.py --client your_api --rebuild
   ```

## Troubleshooting

**MCP server not found**: Ensure the path in `.mcp.json` is correct and absolute.

**No results**: Check that:
- The client exists (`trove_list_clients`)
- Documents were ingested (`python scripts/ingest.py --client X --rebuild`)
- Your query matches document content — try `mode="broad"` for wider matching

**Wrong client**: Use `trove_get_client` to see what's currently loaded, or always use `trove_search_client` with explicit client name.

**Slow client switching**: Increase `TROVE_CACHE_SIZE` in your `.mcp.json` env if you frequently switch between more than 3 clients.

**Summarization returns truncated text**: The `trove_summarize` tool needs LM Studio running for intelligent summaries. Without it, it falls back to showing the first 200 characters.
