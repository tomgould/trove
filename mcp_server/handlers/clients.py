"""Handler functions for client management tools."""

from mcp.types import TextContent

from core.client import load_client, discover_clients

from mcp_server.shared import (
    CLIENTS_DIR,
    get_current_client,
    load_trove_client,
)


def handle_list_clients(arguments: dict) -> list[TextContent]:
    clients = discover_clients(CLIENTS_DIR)
    if not clients:
        return [TextContent(type="text", text="No Trove clients found.")]

    lines = ["Available Trove clients:\n"]
    for client_name in clients:
        try:
            client = load_client(client_name, clients_dir=CLIENTS_DIR)
            desc = client.config.get("description", "No description")
            has_index = client.index_path.exists()
            status = "ready" if has_index else "no index"
            lines.append(f"- **{client_name}**: {desc} [{status}]")
        except Exception as e:
            lines.append(f"- **{client_name}**: Error loading - {e}")

    current = get_current_client()
    if current:
        lines.append(f"\nCurrently loaded: **{current.machine_name}**")

    return [TextContent(type="text", text="\n".join(lines))]


def handle_get_client(arguments: dict) -> list[TextContent]:
    client = get_current_client()
    if not client:
        return [TextContent(
            type="text",
            text="No client loaded. Use trove_load_client or trove_search_client first.",
        )]

    info = [
        f"**Name:** {client.name}",
        f"**Machine Name:** {client.machine_name}",
        f"**Description:** {client.config.get('description', 'N/A')}",
        f"**Index Path:** {client.index_path}",
        f"**Has Index:** {client.index_path.exists()}",
        "",
        "**Search Config:**",
        f"  - BM25 Weight: {client.search_config.get('bm25_weight', 0.5)}",
        f"  - Vector Weight: {client.search_config.get('vector_weight', 0.5)}",
        f"  - Top K: {client.search_config.get('top_k', 20)}",
        "",
        "**Embedding:**",
        f"  - Model: {client.embedding_config.get('model', 'default')}",
    ]
    return [TextContent(type="text", text="\n".join(info))]


def handle_load_client(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    if not client_name:
        return [TextContent(
            type="text",
            text="Error: client_name is required",
        )]

    try:
        client, searcher = load_trove_client(client_name)
        return [TextContent(
            type="text",
            text=f"Loaded client: {client.name} ({client.machine_name})\n\nReady to search.",
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Error loading client: {e}")]
