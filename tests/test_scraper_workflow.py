"""
The scraper workflow: clients outside the repo, ingest summaries, multi-client search.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SMOKE = Path(__file__).parent / "smoke" / "scraper_workflow.py"


class TestMultiClientSearch:
    @pytest.fixture
    def two_clients(self, clients_dir, make_client):
        from core.client import load_client
        from core.ingest import DocumentIngestor, ingest_directory
        from core.search import HybridSearcher

        texts = {
            "alpha": "# Alpha\n\nGlaciers carve wide U-shaped valleys and leave moraines behind as they retreat.",
            "beta": "# Beta\n\nRivers carve narrow V-shaped valleys and carry silt down to the estuary.",
        }
        for name, text in texts.items():
            make_client(name)
            client = load_client(name, clients_dir=clients_dir)
            (client.documents_path / f"{name}.md").write_text(text)
            searcher = HybridSearcher(index_path=str(client.index_path), collection_name=client.machine_name,
                                      skip_bm25_load=True)
            ingest_directory(client.documents_path, DocumentIngestor(), extensions=(".md",), pipeline_searcher=searcher)
            searcher._rebuild_bm25_index()
            searcher.close()
        make_client("empty")
        return clients_dir

    def _search(self, clients_dir, capsys, monkeypatch, *args):
        import scripts.search as cli

        monkeypatch.setattr(sys, "argv", ["search", "--clients-dir", str(clients_dir), *args])
        cli.main()
        return capsys.readouterr()

    def test_clients_json_labels_each_result(self, two_clients, capsys, monkeypatch):
        out = self._search(two_clients, capsys, monkeypatch, "--clients", "alpha,beta", "--json", "valleys")
        data = json.loads(out.out)
        assert data["clients"] == ["alpha", "beta"]
        assert {r["client"] for r in data["results"]} == {"alpha", "beta"}
        assert all(r["source"] == f"{r['client']}.md" for r in data["results"])

    def test_all_skips_clients_without_index(self, two_clients, capsys, monkeypatch):
        out = self._search(two_clients, capsys, monkeypatch, "--all", "--json", "valleys")
        data = json.loads(out.out)
        assert data["clients"] == ["alpha", "beta"]
        assert data["skipped"] == ["empty"]

    def test_plain_output_labels_the_client(self, two_clients, capsys, monkeypatch):
        out = self._search(two_clients, capsys, monkeypatch, "--clients", "alpha,beta", "glaciers moraines")
        assert "alpha: alpha.md" in out.out
        assert "title: Alpha" in out.out

    def test_unknown_client_in_list_is_an_error(self, two_clients, capsys, monkeypatch):
        with pytest.raises(SystemExit) as excinfo:
            self._search(two_clients, capsys, monkeypatch, "--clients", "alpha,nope", "valleys")
        assert excinfo.value.code == 1

    def test_client_options_are_exclusive(self, two_clients, capsys, monkeypatch):
        with pytest.raises(SystemExit) as excinfo:
            self._search(two_clients, capsys, monkeypatch, "--client", "alpha", "--all", "valleys")
        assert excinfo.value.code == 2
