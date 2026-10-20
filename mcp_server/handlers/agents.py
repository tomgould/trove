"""Handler functions for agent tools."""

import logging

from mcp.types import TextContent

from core.llm import create_provider_from_config

from mcp_server.shared import (
    _agent_loader,
    get_current_client,
    get_current_searcher,
    load_trove_client,
)

logger = logging.getLogger("trove-mcp")


def handle_agents_list(arguments: dict) -> list[TextContent]:
    tag = arguments.get("tag")

    try:
        agents = _agent_loader.list(tag=tag)

        if not agents:
            filter_text = f" with tag '{tag}'" if tag else ""
            return [TextContent(type="text", text=f"No agents found{filter_text}.")]

        lines = [f"**Available Agents** ({len(agents)} total):\n"]
        for a in agents:
            tags_str = ", ".join(a["tags"]) if a["tags"] else "none"
            lines.append(f"- **{a['name']}** — {a['description']}")
            lines.append(f"  Tags: {tags_str}")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


def handle_agents_get(arguments: dict) -> list[TextContent]:
    agent_name = arguments.get("agent_name")
    if not agent_name:
        return [TextContent(type="text", text="Error: agent_name is required")]

    try:
        agent = _agent_loader.get(agent_name)
        if not agent:
            return [TextContent(type="text", text=f"Agent '{agent_name}' not found.")]

        lines = [
            f"**{agent['name']}**",
            f"*{agent['description']}*",
            f"Tags: {', '.join(agent.get('tags', []))}",
            "",
            "**System Prompt:**",
            "```",
            agent.get("system_prompt", "").strip(),
            "```",
        ]
        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


def handle_run_agent(arguments: dict) -> list[TextContent]:
    agent_name = arguments.get("agent_name")
    query = arguments.get("query")
    client_name = arguments.get("client_name")
    top_k = arguments.get("top_k", 10)
    store_memory = arguments.get("store_memory", False)

    if not agent_name:
        return [TextContent(type="text", text="Error: agent_name is required")]
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    try:
        # Load agent
        agent = _agent_loader.get(agent_name)
        if not agent:
            return [TextContent(type="text", text=f"Agent '{agent_name}' not found.")]

        # Load client + searcher
        if client_name:
            client, searcher = load_trove_client(client_name)
        else:
            client = get_current_client()
            searcher = get_current_searcher()
            if not client or not searcher:
                return [TextContent(type="text", text="Error: No client loaded. Specify client_name or load a client first.")]

        # Create LLM provider
        provider = create_provider_from_config(client.llm_config)

        # Optional memory store
        memory_store = None
        if store_memory:
            from core.memory import MemoryStore
            db_path = client.index_path / "trove_state.db"
            memory_store = MemoryStore(db_path)

        # Run agent
        from core.runner import AgentRunner
        runner = AgentRunner(
            agent=agent,
            client=client,
            llm_provider=provider,
            searcher=searcher,
            memory_store=memory_store,
        )
        result = runner.run(
            query=query,
            top_k=top_k,
            store_memory=store_memory,
        )

        # Format response
        lines = [
            f"**Agent: {result.agent_name}**",
            f"*Query: {result.query}*",
            f"Sources: {len(result.sources)} | "
            f"Search: {result.timing.get('search_ms', 0)}ms | "
            f"Generate: {result.timing.get('generate_ms', 0)}ms",
            "",
            result.response,
        ]

        if result.sources:
            lines.append("")
            lines.append("---")
            lines.append("**Sources:**")
            seen = set()
            for s in result.sources[:10]:
                src = s["source"]
                if src not in seen:
                    seen.add(src)
                    lines.append(f"- {src} (score: {s['score']:.3f})")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.exception("trove_run_agent failed")
        return [TextContent(type="text", text=f"Error: {e}")]
