"""
MCP server: entry point, "no index" checks and the optional prompt library.
"""

import asyncio
import sys
from unittest.mock import patch

import pytest


class TestServerEntryPoint:
    """Regression: trove-server pointed at scripts.mcp_server:main, which did not exist."""

    def test_main_exists_and_is_callable(self):
        import scripts.mcp_server

        assert callable(scripts.mcp_server.main)

    def test_main_runs_the_server(self):
        import scripts.mcp_server

        calls = []

        async def fake_serve():
            calls.append("served")

        with patch("mcp_server.server.main", fake_serve):
            scripts.mcp_server.main()
        assert calls == ["served"]

    def test_server_main_is_a_coroutine_function(self):
        from mcp_server.server import main

        assert asyncio.iscoroutinefunction(main)
