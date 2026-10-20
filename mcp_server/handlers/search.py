"""Handler functions for search tools."""

import logging
import os

from mcp.types import TextContent

from core.client import load_client, discover_clients

from mcp_server.shared import (
    CLIENTS_DIR,
    _cache,
    _apply_preset,
    _restore_preset,
    _cross_client_rrf_merge,
    _create_llm_provider,
    _summarize_chunk,
    _perform_search,
    format_search_results,
    get_current_client,
    get_current_searcher,
    load_trove_client,
)

logger = logging.getLogger("trove-mcp")


def handle_search(arguments: dict) -> list[TextContent]:
    query = arguments.get("query")
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    searcher = get_current_searcher()
    client = get_current_client()

    if not searcher or not client:
        default_client = os.environ.get("TROVE_DEFAULT_CLIENT")
        if default_client:
            try:
                client, searcher = load_trove_client(default_client)
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"No client loaded and failed to load default '{default_client}': {e}\n\nUse trove_load_client first.",
                )]
        else:
            return [TextContent(
                type="text",
                text="No client loaded. Use trove_load_client or trove_search_client first.",
            )]

    top_k = min(arguments.get("top_k", 10), 50)
    include_content = arguments.get("include_content", True)
    mode = arguments.get("mode")
    max_tokens = arguments.get("max_tokens")

    try:
        output = _perform_search(
            searcher, client, query, top_k,
            include_content=include_content,
            mode=mode,
            max_tokens=max_tokens,
        )
        return [TextContent(type="text", text=output)]
    except Exception as e:
        logger.exception("Search error")
        return [TextContent(type="text", text=f"Search error: {e}")]


def handle_search_client(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    query = arguments.get("query")

    if not client_name:
        return [TextContent(type="text", text="Error: client_name is required")]
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    try:
        client, searcher = load_trove_client(client_name)
    except Exception as e:
        return [TextContent(type="text", text=f"Error loading client: {e}")]

    top_k = min(arguments.get("top_k", 10), 50)
    mode = arguments.get("mode")
    max_tokens = arguments.get("max_tokens")

    try:
        output = _perform_search(
            searcher, client, query, top_k,
            include_content=True,
            mode=mode,
            max_tokens=max_tokens,
        )
        return [TextContent(type="text", text=output)]
    except Exception as e:
        logger.exception("Search error")
        return [TextContent(type="text", text=f"Search error: {e}")]


def handle_search_code(arguments: dict) -> list[TextContent]:
    query = arguments.get("query")
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    searcher = get_current_searcher()
    client = get_current_client()

    if not searcher or not client:
        default_client = os.environ.get("TROVE_DEFAULT_CLIENT")
        if default_client:
            try:
                client, searcher = load_trove_client(default_client)
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"No client loaded and failed to load default: {e}",
                )]
        else:
            return [TextContent(
                type="text",
                text="No client loaded. Use trove_load_client first.",
            )]

    top_k = min(arguments.get("top_k", 10), 50)
    languages = arguments.get("languages", [])
    symbol_types = arguments.get("symbol_types", [])

    try:
        results = searcher.search(
            query,
            top_k=top_k,
            chunk_types=["code"],
            languages=languages if languages else None,
            symbol_types=symbol_types if symbol_types else None,
        )
        output = f"Searching code in **{client.name}** for: *{query}*\n"
        if languages:
            output += f"Languages: {', '.join(languages)}\n"
        if symbol_types:
            output += f"Symbol types: {', '.join(symbol_types)}\n"
        output += "\n"
        output += format_search_results(results, include_content=True)
        return [TextContent(type="text", text=output)]
    except Exception as e:
        logger.exception("Search code error")
        return [TextContent(type="text", text=f"Search error: {e}")]


def handle_search_multi(arguments: dict) -> list[TextContent]:
    query = arguments.get("query")
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    client_names = arguments.get("clients", [])
    top_k = min(arguments.get("top_k", 20), 50)
    mode = arguments.get("mode")
    max_tokens = arguments.get("max_tokens")

    # Determine client list
    if not client_names:
        all_clients = discover_clients(CLIENTS_DIR)
        client_names = []
        for cn in all_clients:
            try:
                c = load_client(cn, clients_dir=CLIENTS_DIR)
                if c.index_path.exists():
                    client_names.append(cn)
            except Exception:
                pass

    if not client_names:
        return [TextContent(type="text", text="No clients with indices found.")]

    # Search each client and collect results
    client_results: dict[str, list[dict]] = {}
    quality_lines = []
    errors = []

    for cn in client_names:
        try:
            client, searcher = load_trove_client(cn)

            saved_attrs = {}
            extra_params = {}
            if mode:
                saved_attrs, extra_params = _apply_preset(searcher, mode)

            try:
                search_top_k = extra_params.pop("top_k", top_k)
                sr = searcher.search_with_quality(
                    query=query, top_k=search_top_k, **extra_params
                )
            finally:
                if saved_attrs:
                    _restore_preset(searcher, saved_attrs)

            client_results[cn] = sr.results
            q = sr.quality
            quality_lines.append(
                f"  - **{cn}**: {len(sr.results)} results, "
                f"top={q.top_score:.2f}, avg={q.avg_rerank_score:.2f}, "
                f"time={q.timing.get('total', 0):.2f}s"
            )
        except Exception as e:
            errors.append(f"  - **{cn}**: Error - {e}")
            logger.warning(f"Multi-client search failed for {cn}: {e}")

    if not client_results:
        return [TextContent(
            type="text",
            text=f"All clients failed:\n" + "\n".join(errors),
        )]

    # Merge across clients
    merged = _cross_client_rrf_merge(client_results, top_k=top_k)

    # Format output
    output = f"Searching {len(client_results)} clients for: *{query}*"
    if mode:
        output += f" (mode: {mode})"
    output += "\n\n"
    output += "**Per-client quality:**\n"
    output += "\n".join(quality_lines)
    if errors:
        output += "\n" + "\n".join(errors)
    output += "\n\n"
    output += format_search_results(merged, include_content=True, max_tokens=max_tokens)

    return [TextContent(type="text", text=output)]


def handle_summarize(arguments: dict) -> list[TextContent]:
    query = arguments.get("query")
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    top_k = arguments.get("top_k", 5)
    client_name = arguments.get("client_name")

    # Load client
    if client_name:
        try:
            client, searcher = load_trove_client(client_name)
        except Exception as e:
            return [TextContent(type="text", text=f"Error loading client: {e}")]
    else:
        client = get_current_client()
        searcher = get_current_searcher()
        if not client or not searcher:
            default_client = os.environ.get("TROVE_DEFAULT_CLIENT")
            if default_client:
                try:
                    client, searcher = load_trove_client(default_client)
                except Exception as e:
                    return [TextContent(
                        type="text",
                        text=f"No client loaded and failed to load default: {e}",
                    )]
            else:
                return [TextContent(
                    type="text",
                    text="No client loaded. Use trove_load_client first or specify client_name.",
                )]

    try:
        search_result = searcher.search_with_quality(query=query, top_k=top_k)
    except Exception as e:
        logger.exception("Search error in summarize")
        return [TextContent(type="text", text=f"Search error: {e}")]

    # Create LLM provider for summarization
    provider = _create_llm_provider(client)

    lines = [f"Summary of results for: *{query}* (from **{client.name}**)\n"]

    for i, chunk in enumerate(search_result.results, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source_file", "unknown")
        score = chunk.get("_rerank_score", chunk.get("_combined_score", 0))
        content = chunk.get("content", "")

        summary = _summarize_chunk(content, query, provider)

        lines.append(f"**{i}. {source}** (score: {score:.3f})")
        lines.append(f"   {summary}")
        lines.append("")

    if not search_result.results:
        lines.append("No results found.")

    return [TextContent(type="text", text="\n".join(lines))]


def handle_list_repos(arguments: dict) -> list[TextContent]:
    searcher = get_current_searcher()
    client = get_current_client()

    if not searcher or not client:
        return [TextContent(
            type="text",
            text="No client loaded. Use trove_load_client first.",
        )]

    try:
        results = searcher.collection.get(include=["metadatas"])
        repos = {}
        for metadata in results.get("metadatas", []):
            repo_name = metadata.get("repo_name")
            if repo_name:
                if repo_name not in repos:
                    repos[repo_name] = {
                        "count": 0,
                        "remote": metadata.get("repo_remote", ""),
                        "branch": metadata.get("repo_branch", ""),
                    }
                repos[repo_name]["count"] += 1

        if not repos:
            return [TextContent(
                type="text",
                text=f"No repositories detected in **{client.name}**.\n\nThis client may only contain documents, not code.",
            )]

        lines = [f"Repositories in **{client.name}**:\n"]
        for repo_name, info in sorted(repos.items()):
            lines.append(f"- **{repo_name}**: {info['count']} chunks")
            if info['remote']:
                lines.append(f"  - Remote: {info['remote']}")
            if info['branch']:
                lines.append(f"  - Branch: {info['branch']}")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.exception("List repos error")
        return [TextContent(type="text", text=f"Error: {e}")]


def handle_search_repo(arguments: dict) -> list[TextContent]:
    repo_name = arguments.get("repo_name")
    query = arguments.get("query")

    if not repo_name:
        return [TextContent(type="text", text="Error: repo_name is required")]
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    searcher = get_current_searcher()
    client = get_current_client()

    if not searcher or not client:
        return [TextContent(
            type="text",
            text="No client loaded. Use trove_load_client first.",
        )]

    top_k = min(arguments.get("top_k", 10), 50)
    chunk_types = arguments.get("chunk_types", [])

    try:
        results = searcher.search(
            query,
            top_k=top_k,
            chunk_types=chunk_types if chunk_types else None,
            repos=[repo_name],
        )
        output = f"Searching repo **{repo_name}** for: *{query}*\n\n"
        output += format_search_results(results, include_content=True)
        return [TextContent(type="text", text=output)]
    except Exception as e:
        logger.exception("Search repo error")
        return [TextContent(type="text", text=f"Search error: {e}")]


def handle_find_symbol(arguments: dict) -> list[TextContent]:
    symbol_name = arguments.get("symbol_name")
    if not symbol_name:
        return [TextContent(type="text", text="Error: symbol_name is required")]

    searcher = get_current_searcher()
    client = get_current_client()

    if not searcher or not client:
        return [TextContent(
            type="text",
            text="No client loaded. Use trove_load_client first.",
        )]

    symbol_type = arguments.get("symbol_type")
    language = arguments.get("language")

    try:
        results = searcher.search(
            symbol_name,
            top_k=20,
            chunk_types=["code"],
            symbol_types=[symbol_type] if symbol_type else None,
            languages=[language] if language else None,
        )

        exact_matches = [
            r for r in results
            if r.get("metadata", {}).get("symbol_name") == symbol_name
        ]

        if exact_matches:
            output = f"Found symbol **{symbol_name}**:\n\n"
            output += format_search_results(exact_matches, include_content=True)
        elif results:
            output = f"No exact match for **{symbol_name}**, but found similar:\n\n"
            output += format_search_results(results[:5], include_content=True)
        else:
            output = f"Symbol **{symbol_name}** not found."

        return [TextContent(type="text", text=output)]
    except Exception as e:
        logger.exception("Find symbol error")
        return [TextContent(type="text", text=f"Error: {e}")]
