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


@pytest.fixture
def shared_clients_dir(clients_dir, monkeypatch):
    """Point the MCP server's clients folder at a temporary one."""
    import mcp_server.handlers.search as search_handlers
    import mcp_server.shared as shared

    monkeypatch.setattr(shared, "CLIENTS_DIR", clients_dir)
    monkeypatch.setattr(search_handlers, "CLIENTS_DIR", clients_dir)
    return clients_dir


class TestNoIndexChecks:
    """Regression: "no index" checks used index_path.exists(), which is always true."""

    def test_load_trove_client_refuses_client_without_index(self, shared_clients_dir, make_client):
        from mcp_server.shared import load_trove_client

        make_client("empty")
        with pytest.raises(ValueError, match="has no index"):
            load_trove_client("empty")

    def test_load_trove_client_unknown_client(self, shared_clients_dir):
        from core.client import ClientNotFoundError
        from mcp_server.shared import load_trove_client

        with pytest.raises(ClientNotFoundError):
            load_trove_client("nope")
        assert not (shared_clients_dir / "nope").exists()

    def test_search_multi_skips_clients_without_index(self, shared_clients_dir, make_client):
        from mcp_server.handlers.search import handle_search_multi

        make_client("one")
        make_client("two")
        result = handle_search_multi({"query": "anything"})
        assert result[0].text == "No clients with indices found."

    def test_list_clients_reports_no_index(self, shared_clients_dir, make_client, monkeypatch):
        import mcp_server.handlers.clients as client_handlers

        monkeypatch.setattr(client_handlers, "CLIENTS_DIR", shared_clients_dir, raising=False)
        make_client("one")
        text = client_handlers.handle_list_clients({})[0].text
        assert "no index" in text

    def test_cli_search_exits_when_client_has_no_index(self, clients_dir, make_client, capsys):
        import scripts.search as cli
        from core.client import load_client

        make_client("empty")
        args = type("Args", (), {"client": "empty", "query": ["hello"]})()
        with patch.object(cli, "load_client", lambda name: load_client(name, clients_dir=clients_dir)):
            with pytest.raises(SystemExit) as excinfo:
                cli.run_search(args)
        assert excinfo.value.code == 1
        assert "has no index" in capsys.readouterr().err

    def test_cli_search_exits_cleanly_for_unknown_client(self, clients_dir, capsys):
        import scripts.search as cli
        from core.client import load_client

        args = type("Args", (), {"client": "nope", "query": ["hello"]})()
        with patch.object(cli, "load_client", lambda name: load_client(name, clients_dir=clients_dir)):
            with pytest.raises(SystemExit) as excinfo:
                cli.run_search(args)
        assert excinfo.value.code == 1
        assert "not found" in capsys.readouterr().err
