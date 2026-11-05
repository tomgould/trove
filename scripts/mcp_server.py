#!/usr/bin/env python3
"""
Trove MCP Server launcher.

This script starts the Trove MCP server for use with Claude Code.

Usage:
    python scripts/mcp_server.py

Environment Variables:
    TROVE_DEFAULT_CLIENT: Client to load on startup (optional)

Configuration for Claude Code (~/.claude/settings.json or .mcp.json):
    {
        "mcpServers": {
            "trove": {
                "command": "python",
                "args": ["/path/to/trove/scripts/mcp_server.py"],
                "env": {
                    "TROVE_DEFAULT_CLIENT": "your_client_name"
                }
            }
        }
    }
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    """Run the MCP server over stdio. This is the trove-server console script."""
    import asyncio

    try:
        from mcp_server.server import main as serve
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] != "mcp":
            raise
        print(
            "The MCP server needs the mcp package. Install it with: "
            "pip install 'trove-knowledge-engine[mcp]'",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(serve())


if __name__ == "__main__":
    main()
