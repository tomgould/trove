#!/usr/bin/env python3
"""
Trove MCP Server - Model Context Protocol server for Trove.

This server exposes Trove's search functionality to Claude Code,
allowing it to query your indexed documentation and API docs.

Usage:
    # Run directly
    python -m mcp_server.server

    # Or via the CLI script
    python scripts/mcp_server.py

Configure in Claude Code settings or .mcp.json:
    {
        "mcpServers": {
            "trove": {
                "command": "python",
                "args": ["/path/to/trove/scripts/mcp_server.py"],
                "env": {
                    "TROVE_DEFAULT_CLIENT": "claude"
                }
            }
        }
    }
"""

import atexit
import logging
import os
import signal
import sys
import threading
from pathlib import Path

# Add parent directory to path for imports
TROVE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(TROVE_ROOT))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

# Re-export shared state and helpers so existing imports from
# mcp_server.server continue to work (tests, scripts, etc.)
from mcp_server.shared import (  # noqa: F401 — re-exports
    SearcherCache,
    SEARCH_PRESETS,
    _apply_preset,
    _restore_preset,
    _cache,
    _job_manager,
    _state_stores,
    _validate_client_name,
    estimate_tokens,
    format_search_results,
    _cross_client_rrf_merge,
    _summarize_chunk,
    _create_llm_provider,
    _perform_search,
    load_trove_client,
    get_current_client,
    get_current_searcher,
    CLIENTS_DIR,
)

from mcp_server.handlers import (
    clients as _h_clients,
    search as _h_search,
    ingest as _h_ingest,
    graph as _h_graph,
    memory as _h_memory,
    workflows as _h_workflows,
    agents as _h_agents,
    prompts as _h_prompts,
    sources as _h_sources,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("trove-mcp")

# Create MCP server
server = Server("trove")

# Search mode descriptions for tool documentation
_MODE_DESC = (
    "Search mode preset: "
    "'precise' (high BM25 weight, strict scoring, top 10), "
    "'broad' (high vector weight, no min score, top 30), "
    "'code_first' (boost code chunks, cross-reference code), "
    "'exhaustive' (RRF merge, no adaptive rerank, top 50), "
    "'diverse' (MMR diversity reranking), "
    "'graph_local' (entity-focused graph search), "
    "'graph_global' (relationship traversal search), "
    "'graph_hybrid' (local + global graph search), "
    "'graph_mix' (graph + standard hybrid via RRF)"
)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="trove_list_clients",
            description="List all available Trove clients/projects. Each client has its own indexed document collection.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="trove_get_client",
            description="Get information about the currently loaded Trove client, including its name, description, and configuration.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="trove_load_client",
            description="Load a Trove client/project to search its indexed documents. Must be called before searching if no client is loaded.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Name of the client to load (use trove_list_clients to see available clients)",
                    },
                },
                "required": ["client_name"],
            },
        ),
        Tool(
            name="trove_search",
            description="Search the loaded Trove client's indexed documents using hybrid search (BM25 keyword + vector semantic). Returns relevant chunks with source information and relevance scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - can be a question, keywords, or natural language",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 40, max: 50)",
                        "default": 40,
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "Include full chunk content in results (default: true)",
                        "default": True,
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["precise", "broad", "code_first", "exhaustive", "diverse", "graph_local", "graph_global", "graph_hybrid", "graph_mix"],
                        "description": _MODE_DESC,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens in response. Results are trimmed to fit within budget.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="trove_search_client",
            description="Load a specific client and search it in one operation. Convenience method that combines trove_load_client and trove_search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Name of the client to search",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 40)",
                        "default": 40,
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["precise", "broad", "code_first", "exhaustive", "diverse", "graph_local", "graph_global", "graph_hybrid", "graph_mix"],
                        "description": _MODE_DESC,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens in response. Results are trimmed to fit within budget.",
                    },
                },
                "required": ["client_name", "query"],
            },
        ),
        Tool(
            name="trove_search_code",
            description="Search code specifically - functions, classes, methods. Filters results to only code chunks with optional language and symbol type filtering.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for code",
                    },
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by languages (e.g., ['python', 'php']). Empty for all languages.",
                    },
                    "symbol_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by symbol types (e.g., ['function', 'class', 'method']). Empty for all types.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 40)",
                        "default": 40,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="trove_list_repos",
            description="List all git repositories detected in the loaded client's index.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="trove_search_repo",
            description="Search within a specific repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_name": {
                        "type": "string",
                        "description": "Repository name to search in",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "chunk_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by chunk types (['code'], ['document'], or both). Empty for all.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 40)",
                        "default": 40,
                    },
                },
                "required": ["repo_name", "query"],
            },
        ),
        Tool(
            name="trove_find_symbol",
            description="Find a specific symbol (function, class, method) by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Name of the symbol to find",
                    },
                    "symbol_type": {
                        "type": "string",
                        "description": "Type of symbol (function, class, method, etc.). Optional.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language. Optional.",
                    },
                },
                "required": ["symbol_name"],
            },
        ),
        Tool(
            name="trove_search_multi",
            description=(
                "Search across multiple Trove clients simultaneously and merge results. "
                "When clients list is empty, searches all clients with indices."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "clients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Client names to search. Empty = all clients with indices.",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 20,
                        "description": "Results per client before merge",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["precise", "broad", "code_first", "exhaustive", "diverse", "graph_local", "graph_global", "graph_hybrid", "graph_mix"],
                        "description": _MODE_DESC,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens in response",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="trove_summarize",
            description=(
                "Search and return concise summaries of results instead of full content. "
                "Useful for getting an overview before diving deep."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results",
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Client to search (uses current if not specified)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="trove_ingest_file",
            description="Ingest a single file into a client's index. The file must exist under the client's documents directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Name of the client to ingest into",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to ingest",
                    },
                },
                "required": ["client_name", "file_path"],
            },
        ),
        Tool(
            name="trove_ingest_url",
            description="Fetch a URL and ingest its content into a client's index.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Name of the client to ingest into",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to fetch and ingest (http/https only)",
                    },
                },
                "required": ["client_name", "url"],
            },
        ),
        Tool(
            name="trove_reindex_client",
            description="Trigger a full reindex of a client's documents. Runs in the background; use trove_ingest_status to check progress.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Name of the client to reindex",
                    },
                },
                "required": ["client_name"],
            },
        ),
        Tool(
            name="trove_ingest_status",
            description="Check the status of an ingestion job. Returns progress, ETA, and chunk counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to check (from trove_reindex_client)",
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Client name to get latest job for (alternative to job_id)",
                    },
                },
            },
        ),
        Tool(
            name="trove_search_entities",
            description="Search knowledge graph entities by name. Returns matching entity names, types, descriptions, and connection counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Entity name to search for (substring match)",
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "Filter by entity type (e.g. 'person', 'organization', 'concept')",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results (default: 20)",
                        "default": 20,
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Client to search (uses current if not specified)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="trove_search_relationships",
            description="Traverse knowledge graph relationships from an entity. Returns connected entities and relationship descriptions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Entity name to start traversal from",
                    },
                    "relationship_type": {
                        "type": "string",
                        "description": "Filter by relationship type (optional)",
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Max traversal depth (default: 2, max: 4)",
                        "default": 2,
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Client to search (uses current if not specified)",
                    },
                },
                "required": ["entity_name"],
            },
        ),
        Tool(
            name="trove_graph_stats",
            description="Get statistics about a client's knowledge graph: entity/relationship counts, type distributions, top connected entities, density.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Client to get stats for (uses current if not specified)",
                    },
                },
            },
        ),
        Tool(
            name="trove_graph_export",
            description="Export knowledge graph as D3.js-compatible JSON for visualization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Client to export graph for (uses current if not specified)",
                    },
                    "max_nodes": {
                        "type": "integer",
                        "description": "Maximum nodes to include (default: 500)",
                        "default": 500,
                    },
                },
            },
        ),
        Tool(
            name="trove_build_graph",
            description="Build a knowledge graph from the client's indexed document chunks using LLM entity/relationship extraction. Runs in the background.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Client name to build graph for",
                    },
                    "rebuild": {
                        "type": "boolean",
                        "description": "Clear existing graph and rebuild from scratch (default: false, incremental)",
                        "default": False,
                    },
                    "sample": {
                        "type": "integer",
                        "description": "Only process N random chunks (for testing). Omit for all chunks.",
                    },
                    "workers": {
                        "type": "integer",
                        "description": "Number of concurrent LLM workers (default: from client config or 4)",
                    },
                },
                "required": ["client_name"],
            },
        ),
        # --- Agent Orchestration tools ---
        Tool(
            name="trove_memory_store",
            description="Store a memory entry for persistent context across sessions. Supports namespaces, metadata, and optional TTL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Logical grouping (e.g. 'decisions', 'patterns', 'context')",
                    },
                    "key": {
                        "type": "string",
                        "description": "Unique key within the namespace",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to store",
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Client to scope memory to (omit for global)",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata dict",
                    },
                    "ttl": {
                        "type": "integer",
                        "description": "Optional time-to-live in seconds",
                    },
                },
                "required": ["namespace", "key", "content"],
            },
        ),
        Tool(
            name="trove_memory_search",
            description="Search memories by semantic similarity or keyword match. Returns relevant stored context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Client to search (omit for global)",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace filter",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="trove_memory_list",
            description="List all stored memories, optionally filtered by namespace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Client to list from (omit for global)",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace filter",
                    },
                },
            },
        ),
        Tool(
            name="trove_memory_delete",
            description="Delete a specific memory by namespace and key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Client to delete from (omit for global)",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Memory namespace",
                    },
                    "key": {
                        "type": "string",
                        "description": "Memory key",
                    },
                },
                "required": ["namespace", "key"],
            },
        ),
        Tool(
            name="trove_workflow_create",
            description="Create a workflow with named steps and dependency ordering. Steps are executed by Claude, the engine tracks state.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Workflow name",
                    },
                    "steps": {
                        "type": "array",
                        "description": "List of step definitions",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Step name"},
                                "description": {"type": "string", "description": "What this step does"},
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Names of steps that must complete first",
                                },
                            },
                            "required": ["name"],
                        },
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Client to scope workflow to (omit for global)",
                    },
                },
                "required": ["name", "steps"],
            },
        ),
        Tool(
            name="trove_workflow_status",
            description="Get a workflow's current status including all steps and their progress.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "The workflow ID",
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Client the workflow belongs to (omit for global)",
                    },
                },
                "required": ["workflow_id"],
            },
        ),
        Tool(
            name="trove_workflow_update",
            description="Update a workflow step's status (pending, running, completed, failed, skipped).",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "The workflow ID",
                    },
                    "step_id": {
                        "type": "string",
                        "description": "The step ID to update",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "running", "completed", "failed", "skipped"],
                        "description": "New step status",
                    },
                    "result": {
                        "type": "string",
                        "description": "Optional result text",
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Client the workflow belongs to (omit for global)",
                    },
                },
                "required": ["workflow_id", "step_id", "status"],
            },
        ),
        Tool(
            name="trove_workflow_list",
            description="List all workflows, optionally filtered by status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Client to list from (omit for global)",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "running", "completed", "failed"],
                        "description": "Optional status filter",
                    },
                },
            },
        ),
        Tool(
            name="trove_agents_list",
            description="List all available agent templates, optionally filtered by tag.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Optional tag to filter by",
                    },
                },
            },
        ),
        Tool(
            name="trove_agents_get",
            description="Get full agent template config including system prompt by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Agent name or filename slug",
                    },
                },
                "required": ["agent_name"],
            },
        ),
        Tool(
            name="trove_run_agent",
            description="Run a Trove agent locally via LM Studio. Searches for relevant context, injects the agent's system prompt, and generates a response. Use this to delegate analysis tasks (code review, security review, architecture analysis, etc.) to a local LLM instead of using cloud credits.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Agent name or filename slug (e.g. 'code-reviewer', 'security-reviewer')",
                    },
                    "query": {
                        "type": "string",
                        "description": "The question or task for the agent",
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Trove client to search (uses current client if not specified)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of search results for context (default: 10)",
                    },
                    "store_memory": {
                        "type": "boolean",
                        "description": "Store the result in persistent memory (default: false)",
                    },
                },
                "required": ["agent_name", "query"],
            },
        ),
        # --- prompts.chat API tools ---
        Tool(
            name="trove_prompts_search_live",
            description="Search the live prompts.chat API for prompt templates. Supports filtering by type (TEXT/IMAGE/VIDEO/AUDIO/SKILL), category, tag, and sort order. Returns live results from the prompts.chat community.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'coding assistant', 'image generation')",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["TEXT", "IMAGE", "VIDEO", "AUDIO", "SKILL"],
                        "description": "Filter by prompt type",
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category slug or name",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag slug or name",
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["newest", "popular"],
                        "description": "Sort order (default: relevance)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 10, max: 50)",
                        "default": 10,
                    },
                },
            },
        ),
        Tool(
            name="trove_prompts_find",
            description="Find the best prompt template by combining local Trove semantic search (9,970 indexed prompts) with the live prompts.chat API. Returns deduplicated results from both sources. Gracefully degrades if the API is unreachable.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you need a prompt for (e.g. 'react code reviewer', 'midjourney style portrait')",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["TEXT", "IMAGE", "VIDEO", "AUDIO", "SKILL"],
                        "description": "Filter live API results by prompt type",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Total results to return (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="trove_prompts_get",
            description="Fetch a full prompt template by ID from prompts.chat with optional variable substitution. Use after finding a prompt via trove_prompts_find or trove_prompts_search_live.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_id": {
                        "type": "string",
                        "description": "The prompt ID (e.g. 'cmj1zb0yc000ivl0rc9ka596e')",
                    },
                    "variables": {
                        "type": "object",
                        "description": "Variable substitutions for ${Var:Default} placeholders (e.g. {\"Position\": \"Data Engineer\"})",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["prompt_id"],
            },
        ),
        # --- Source Connector Tools ---
        Tool(
            name="trove_connect_github",
            description="Add a GitHub account, org, or repo as a source for a Trove client. Discovers and clones repos to the client's sources directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Name of the client to add the GitHub source to",
                    },
                    "owner": {
                        "type": "string",
                        "description": "GitHub username or organization name (e.g. 'anthropics')",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Specific repo name (optional — if omitted, discovers all repos for owner)",
                    },
                    "token_env": {
                        "type": "string",
                        "description": "Environment variable name containing the GitHub token (default: GITHUB_TOKEN)",
                    },
                    "include_repos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only sync these repos (whitelist)",
                    },
                    "exclude_repos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Skip these repos (blacklist)",
                    },
                },
                "required": ["client_name", "owner"],
            },
        ),
        Tool(
            name="trove_crawl_web",
            description="Crawl a website and save pages as Markdown for a Trove client. Supports incremental crawling via content hashing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Name of the client to crawl for",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to start crawling from (e.g. 'https://docs.example.com')",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum pages to crawl (default: 100)",
                    },
                    "stay_in_path": {
                        "type": "boolean",
                        "description": "Only crawl pages under the starting URL path (default: true)",
                    },
                },
                "required": ["client_name", "url"],
            },
        ),
        Tool(
            name="trove_sync_sources",
            description="Sync all configured sources for a client. Runs git fetch for GitHub sources and re-crawls web sources. Returns sync results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Name of the client to sync sources for",
                    },
                },
                "required": ["client_name"],
            },
        ),
        Tool(
            name="trove_list_sources",
            description="List all configured sources for a client with their sync status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Name of the client to list sources for",
                    },
                },
                "required": ["client_name"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool dispatch table
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    # Client management
    "trove_list_clients": _h_clients.handle_list_clients,
    "trove_get_client": _h_clients.handle_get_client,
    "trove_load_client": _h_clients.handle_load_client,
    # Search
    "trove_search": _h_search.handle_search,
    "trove_search_client": _h_search.handle_search_client,
    "trove_search_code": _h_search.handle_search_code,
    "trove_search_multi": _h_search.handle_search_multi,
    "trove_summarize": _h_search.handle_summarize,
    "trove_list_repos": _h_search.handle_list_repos,
    "trove_search_repo": _h_search.handle_search_repo,
    "trove_find_symbol": _h_search.handle_find_symbol,
    # Ingestion
    "trove_ingest_file": _h_ingest.handle_ingest_file,
    "trove_ingest_url": _h_ingest.handle_ingest_url,
    "trove_reindex_client": _h_ingest.handle_reindex_client,
    "trove_ingest_status": _h_ingest.handle_ingest_status,
    # Knowledge graph
    "trove_graph_stats": _h_graph.handle_graph_stats,
    "trove_graph_export": _h_graph.handle_graph_export,
    "trove_search_entities": _h_graph.handle_search_entities,
    "trove_search_relationships": _h_graph.handle_search_relationships,
    "trove_build_graph": _h_graph.handle_build_graph,
    # Memory
    "trove_memory_store": _h_memory.handle_memory_store,
    "trove_memory_search": _h_memory.handle_memory_search,
    "trove_memory_list": _h_memory.handle_memory_list,
    "trove_memory_delete": _h_memory.handle_memory_delete,
    # Workflows
    "trove_workflow_create": _h_workflows.handle_workflow_create,
    "trove_workflow_status": _h_workflows.handle_workflow_status,
    "trove_workflow_update": _h_workflows.handle_workflow_update,
    "trove_workflow_list": _h_workflows.handle_workflow_list,
    # Agents
    "trove_agents_list": _h_agents.handle_agents_list,
    "trove_agents_get": _h_agents.handle_agents_get,
    "trove_run_agent": _h_agents.handle_run_agent,
    # Prompts
    "trove_prompts_search_live": _h_prompts.handle_prompts_search_live,
    "trove_prompts_find": _h_prompts.handle_prompts_find,
    "trove_prompts_get": _h_prompts.handle_prompts_get,
    # Sources
    "trove_connect_github": _h_sources.handle_connect_github,
    "trove_crawl_web": _h_sources.handle_crawl_web,
    "trove_sync_sources": _h_sources.handle_sync_sources,
    "trove_list_sources": _h_sources.handle_list_sources,
}


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        # Validate client_name if present in arguments
        client_name_arg = arguments.get("client_name")
        if client_name_arg is not None:
            validation_error = _validate_client_name(client_name_arg)
            if validation_error:
                return [TextContent(type="text", text=validation_error)]

        handler = TOOL_HANDLERS.get(name)
        if handler:
            return handler(arguments)

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.exception(f"Tool error: {name}")
        return [TextContent(type="text", text=f"Error: {e}")]


def cleanup():
    """Release GPU models and searcher resources."""
    try:
        logger.info("Cleaning up searcher resources...")
        _cache.close_all()
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
    try:
        for key, (memory, workflow) in _state_stores.items():
            # Both share the same connection; close it under the shared lock
            try:
                with memory._lock:
                    memory._conn.close()
            except Exception:
                pass
        _state_stores.clear()
    except Exception as e:
        logger.warning(f"Error closing state stores: {e}")
    logger.info("Trove MCP server shutdown complete.")


def _signal_handler(signum, frame):
    """Handle termination signals by cleaning up and exiting."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name}, shutting down...")
    cleanup()
    sys.exit(0)


def _start_parent_watchdog():
    """Poll whether the parent process is still alive; exit if orphaned.

    When Claude Code exits, it may not always close stdin cleanly (e.g. if
    killed with SIGKILL). This watchdog checks the parent PID every 2 seconds
    and triggers cleanup if we've been re-parented to init/systemd (PID 1).
    """
    parent_pid = os.getppid()

    def _watch():
        import time
        while True:
            time.sleep(2)
            current_parent = os.getppid()
            if current_parent != parent_pid:
                logger.info(
                    "Parent PID changed (%d -> %d), orphaned — shutting down...",
                    parent_pid, current_parent,
                )
                cleanup()
                os._exit(0)

    t = threading.Thread(target=_watch, daemon=True, name="parent-watchdog")
    t.start()


async def main():
    """Run the MCP server."""
    # Register cleanup for all exit paths
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _signal_handler)

    # Detect if our parent process dies (e.g. Claude Code killed with SIGKILL)
    _start_parent_watchdog()

    # Try to load default client on startup
    default_client = os.environ.get("TROVE_DEFAULT_CLIENT")
    if default_client:
        try:
            load_trove_client(default_client)
            logger.info(f"Loaded default client: {default_client}")
        except Exception as e:
            logger.warning(f"Failed to load default client '{default_client}': {e}")

    logger.info("Starting Trove MCP server (PID: %d, parent: %d)...", os.getpid(), os.getppid())
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        # stdio_server exited (stdin closed or error) — clean up GPU resources
        logger.info("MCP server loop ended, running cleanup...")
        cleanup()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
