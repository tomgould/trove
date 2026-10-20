"""Handler functions for memory tools."""

from mcp.types import TextContent

from mcp_server.shared import _get_state_db


def handle_memory_store(arguments: dict) -> list[TextContent]:
    namespace = arguments.get("namespace")
    key = arguments.get("key")
    content = arguments.get("content")

    if not namespace:
        return [TextContent(type="text", text="Error: namespace is required")]
    if not key:
        return [TextContent(type="text", text="Error: key is required")]
    if not content:
        return [TextContent(type="text", text="Error: content is required")]

    client_name = arguments.get("client_name")
    metadata = arguments.get("metadata")
    ttl = arguments.get("ttl")

    try:
        memory, _ = _get_state_db(client_name)
        memory_id = memory.store(namespace, key, content, metadata=metadata, ttl=ttl)
        scope = f"client '{client_name}'" if client_name else "global"
        return [TextContent(
            type="text",
            text=f"Stored memory `{namespace}/{key}` ({scope})\nID: `{memory_id}`",
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Error storing memory: {e}")]


def handle_memory_search(arguments: dict) -> list[TextContent]:
    query = arguments.get("query")
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    client_name = arguments.get("client_name")
    namespace = arguments.get("namespace")
    top_k = arguments.get("top_k", 5)

    try:
        memory, _ = _get_state_db(client_name)
        results = memory.search(query, namespace=namespace, top_k=top_k)

        if not results:
            return [TextContent(type="text", text=f"No memories found for: {query}")]

        lines = [f"Found {len(results)} memories for: *{query}*\n"]
        for i, r in enumerate(results, 1):
            sim = f" (similarity: {r['similarity']:.3f})" if "similarity" in r else ""
            lines.append(f"**{i}. {r['namespace']}/{r['key']}**{sim}")
            preview = r["content"][:200].replace("\n", " ")
            lines.append(f"   {preview}{'...' if len(r['content']) > 200 else ''}")
            if r.get("metadata"):
                lines.append(f"   metadata: {r['metadata']}")
            lines.append("")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error searching memories: {e}")]


def handle_memory_list(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    namespace = arguments.get("namespace")

    try:
        memory, _ = _get_state_db(client_name)
        results = memory.list(namespace=namespace)

        if not results:
            scope = f"client '{client_name}'" if client_name else "global"
            ns_filter = f" (namespace: {namespace})" if namespace else ""
            return [TextContent(type="text", text=f"No memories stored ({scope}{ns_filter}).")]

        lines = [f"**Stored memories** ({len(results)} total):\n"]
        lines.append("| Namespace | Key | Content (preview) | Updated |")
        lines.append("|-----------|-----|-------------------|---------|")
        for r in results:
            import datetime
            updated = datetime.datetime.fromtimestamp(r["updated_at"]).strftime("%Y-%m-%d %H:%M")
            preview = r["content"][:50].replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {r['namespace']} | {r['key']} | {preview}{'...' if len(r['content']) > 50 else ''} | {updated} |")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error listing memories: {e}")]


def handle_memory_delete(arguments: dict) -> list[TextContent]:
    namespace = arguments.get("namespace")
    key = arguments.get("key")

    if not namespace:
        return [TextContent(type="text", text="Error: namespace is required")]
    if not key:
        return [TextContent(type="text", text="Error: key is required")]

    client_name = arguments.get("client_name")

    try:
        memory, _ = _get_state_db(client_name)
        deleted = memory.delete(namespace, key)
        if deleted:
            return [TextContent(type="text", text=f"Deleted memory `{namespace}/{key}`.")]
        else:
            return [TextContent(type="text", text=f"Memory `{namespace}/{key}` not found.")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error deleting memory: {e}")]
