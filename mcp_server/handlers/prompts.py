"""Handler functions for prompts.chat API tools."""

import logging

from mcp.types import TextContent

from mcp_server.shared import (
    _get_prompts_api_client,
    load_trove_client,
)

logger = logging.getLogger("trove-mcp")


def handle_prompts_search_live(arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    prompt_type = arguments.get("type")
    category = arguments.get("category")
    tag = arguments.get("tag")
    sort = arguments.get("sort")
    limit = min(arguments.get("limit", 10), 50)

    try:
        api = _get_prompts_api_client()
        result = api.search_prompts(
            query, type=prompt_type, category=category,
            tag=tag, sort=sort, limit=limit,
        )
    except Exception as e:
        return [TextContent(type="text", text=f"prompts.chat API error: {e}")]

    if not result.prompts:
        return [TextContent(type="text", text=f"No prompts found for: {query or '(browse)'}")]

    lines = [f"**prompts.chat** — {result.total} results"]
    if query:
        lines[0] += f" for *{query}*"
    lines.append("")
    for i, p in enumerate(result.prompts, 1):
        meta = f"{p.type}"
        if p.category:
            meta += f" | {p.category}"
        if p.vote_count:
            meta += f" | {p.vote_count} votes"
        lines.append(f"**{i}. {p.title}** ({meta})")
        lines.append(f"   ID: `{p.id}`")
        preview = (p.content or "")[:200].replace("\n", " ")
        if preview:
            lines.append(f"   {preview}{'...' if len(p.content or '') > 200 else ''}")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


def handle_prompts_find(arguments: dict) -> list[TextContent]:
    query = arguments.get("query")
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    prompt_type = arguments.get("type")
    top_k = min(arguments.get("top_k", 10), 50)
    half_k = max(top_k // 2, 3)

    # Local Trove search
    local_results = []
    try:
        client, searcher = load_trove_client("prompts")
        search_result = searcher.search_with_quality(query=query, top_k=half_k)
        for chunk in search_result.results:
            content = chunk.get("content", chunk.get("text", ""))
            source = chunk.get("metadata", {}).get("source", "")
            title = chunk.get("metadata", {}).get("title", "")
            if not title:
                # Extract title from first markdown heading
                for line in content.split("\n"):
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
            local_results.append({
                "source": "local",
                "title": title or source.split("/")[-1].replace(".md", "").replace("_", " "),
                "content": content,
                "score": chunk.get("rerank_score", chunk.get("combined_score", 0)),
                "file": source,
            })
    except Exception as e:
        logger.warning(f"Local prompts search failed: {e}")

    # Live API search
    api_results = []
    try:
        api = _get_prompts_api_client()
        resp = api.search_prompts(query, type=prompt_type, limit=half_k)
        for p in resp.prompts:
            api_results.append({
                "source": "live",
                "title": p.title,
                "content": p.content,
                "id": p.id,
                "category": p.category,
                "type": p.type,
                "votes": p.vote_count,
                "url": p.url,
            })
    except Exception as e:
        logger.warning(f"prompts.chat API search failed (degrading gracefully): {e}")

    # Deduplicate by title similarity (normalize and compare)
    seen_titles: set[str] = set()
    combined = []
    for r in local_results + api_results:
        norm_title = r["title"].lower().strip()
        if norm_title not in seen_titles:
            seen_titles.add(norm_title)
            combined.append(r)

    if not combined:
        return [TextContent(type="text", text=f"No prompts found for: {query}")]

    lines = [
        f"**Prompt search** for *{query}* — "
        f"{len(local_results)} local, {len(api_results)} live, "
        f"{len(combined)} deduplicated",
        "",
    ]
    for i, r in enumerate(combined[:top_k], 1):
        src_tag = "[LOCAL]" if r["source"] == "local" else "[LIVE]"
        meta_parts = [src_tag]
        if r.get("category"):
            meta_parts.append(r["category"])
        if r.get("type") and r["type"] != "TEXT":
            meta_parts.append(r["type"])
        if r.get("votes"):
            meta_parts.append(f"{r['votes']} votes")

        lines.append(f"**{i}. {r['title']}** ({' | '.join(meta_parts)})")
        if r.get("id"):
            lines.append(f"   ID: `{r['id']}`")
        preview = (r.get("content") or "")[:200].replace("\n", " ")
        if preview:
            lines.append(f"   {preview}{'...' if len(r.get('content', '')) > 200 else ''}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


def handle_prompts_get(arguments: dict) -> list[TextContent]:
    prompt_id = arguments.get("prompt_id")
    if not prompt_id:
        return [TextContent(type="text", text="Error: prompt_id is required")]

    variables = arguments.get("variables")

    try:
        api = _get_prompts_api_client()
        prompt = api.get_prompt(prompt_id)
    except Exception as e:
        return [TextContent(type="text", text=f"Error fetching prompt {prompt_id}: {e}")]

    content = prompt.content
    if variables:
        content = api.substitute_variables(content, variables)

    output = api.format_prompt(prompt)
    if variables:
        output += "\n\n---\n### With substitutions applied:\n\n" + content

    return [TextContent(type="text", text=output)]
