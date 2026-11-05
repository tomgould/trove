"""
MCP server: entry point, "no index" checks and the optional prompt library.
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch

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

    def test_missing_mcp_package_gives_a_clear_message(self, monkeypatch, capsys):
        """A core install without the mcp extra gets one line and exit 1, not a traceback."""
        import scripts.mcp_server

        for name in [m for m in sys.modules if m == "mcp" or m.startswith("mcp.")] + ["mcp"]:
            monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, "mcp_server.server", raising=False)
        with pytest.raises(SystemExit) as excinfo:
            scripts.mcp_server.main()
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "trove-knowledge-engine[mcp]" in err
        assert "Traceback" not in err

    def test_server_main_is_a_coroutine_function(self):
        from mcp_server.server import main

        assert asyncio.iscoroutinefunction(main)


@pytest.fixture
def shared_clients_dir(clients_dir, monkeypatch):
    """Point the MCP server's clients folder at a temporary one."""
    import importlib

    import mcp_server.shared as shared

    monkeypatch.setattr(shared, "CLIENTS_DIR", clients_dir)
    # Handler modules copy the name at import time, so patch each of them too
    for name in ("clients", "graph", "ingest", "search", "sources"):
        module = importlib.import_module(f"mcp_server.handlers.{name}")
        if hasattr(module, "CLIENTS_DIR"):
            monkeypatch.setattr(module, "CLIENTS_DIR", clients_dir)
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


class TestIngestIntoNewClient:
    """Regression: the MCP ingest tools refused a new client because it had no index yet."""

    @pytest.fixture
    def fresh_client(self, shared_clients_dir, make_client):
        import mcp_server.shared as shared
        from core.client import load_client

        name = make_client("fresh")
        yield load_client(name, clients_dir=shared_clients_dir)
        evicted = shared._cache._cache.pop(name, None)
        if evicted is not None:
            evicted[1].close()

    def test_ingest_file_into_new_client(self, fresh_client):
        from mcp_server.handlers.ingest import handle_ingest_file

        doc = fresh_client.documents_path / "a.md"
        doc.write_text("# Notes\n\nA short document about sourdough starters and how to feed them every day.\n")
        text = handle_ingest_file({"client_name": "fresh", "file_path": str(doc)})[0].text
        assert text.startswith("Ingested **a.md**"), text
        assert fresh_client.has_index

    def test_ingest_url_into_new_client(self, fresh_client):
        from mcp_server.handlers.ingest import handle_ingest_url

        page = b"<html><body><h1>Rivers</h1><p>A river carries water from the hills down to the sea and shapes the valley.</p></body></html>"
        response = MagicMock(content=page, headers={"content-type": "text/html", "content-length": str(len(page))})
        response.raise_for_status.return_value = None
        with patch("requests.get", return_value=response):
            text = handle_ingest_url({"client_name": "fresh", "url": "https://example.com/rivers"})[0].text
        assert text.startswith("Ingested URL:"), text
        assert fresh_client.has_index

    def test_load_client_tool_accepts_new_client(self, fresh_client):
        from mcp_server.handlers.clients import handle_load_client

        text = handle_load_client({"client_name": "fresh"})[0].text
        assert text.startswith("Loaded client:"), text
        assert "Nothing is indexed yet" in text

    def test_search_still_refuses_new_client(self, fresh_client):
        """Loading for ingestion must not let search treat the empty client as ready."""
        from mcp_server.handlers.clients import handle_load_client
        from mcp_server.shared import load_trove_client

        handle_load_client({"client_name": "fresh"})
        with pytest.raises(ValueError, match="has no index"):
            load_trove_client("fresh")


@pytest.fixture
def no_prompt_library(monkeypatch):
    """Make the optional prompts.chat client impossible to import."""
    monkeypatch.setitem(sys.modules, "clients.prompts.api_client", None)


class TestPromptLibraryUnavailable:
    """Regression: the prompts tools failed with an ImportError for clients/prompts/api_client.py."""

    def test_get_client_raises_clear_error(self, no_prompt_library):
        from mcp_server.shared import PromptLibraryUnavailable, _get_prompts_api_client

        with pytest.raises(PromptLibraryUnavailable, match="prompt library is not available"):
            _get_prompts_api_client()

    def test_search_live_degrades(self, no_prompt_library):
        from mcp_server.handlers.prompts import handle_prompts_search_live

        text = handle_prompts_search_live({"query": "code review"})[0].text
        assert "prompt library is not available" in text

    def test_get_degrades(self, no_prompt_library):
        from mcp_server.handlers.prompts import handle_prompts_get

        text = handle_prompts_get({"prompt_id": "abc"})[0].text
        assert "prompt library is not available" in text

    def test_find_degrades(self, no_prompt_library, shared_clients_dir):
        from mcp_server.handlers.prompts import handle_prompts_find

        text = handle_prompts_find({"query": "code review"})[0].text
        assert "prompt library is not available" in text

    def test_sync_prompts_script_exits_with_message(self, no_prompt_library, monkeypatch, capsys):
        import importlib

        import scripts.sync_prompts as sync

        sync = importlib.reload(sync)
        monkeypatch.setattr(sys, "argv", ["sync_prompts.py", "--dry-run"])
        with pytest.raises(SystemExit) as excinfo:
            sync.main()
        assert excinfo.value.code == 1
        assert "prompt library is not available" in capsys.readouterr().err
