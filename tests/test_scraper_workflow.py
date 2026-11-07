"""
The scraper workflow: clients outside the repo, ingest summaries, multi-client search.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SMOKE = Path(__file__).parent / "smoke" / "scraper_workflow.py"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_end_to_end_scraper_workflow(tmp_path):
    observed = _load(SMOKE).run(tmp_path)
    assert observed["cli_top_result"]["url"] == "https://bakery.example/guides/starter"
    assert observed["chunks_embedded_on_reingest"] == 0
    assert observed["multi_clients"] == ["bakery", "notes"]
    assert observed["mcp_has_citation"]


class TestManifestHash:
    """Regression: a scraper rewriting identical files changed their mtime and forced a re-embed."""

    def test_rewritten_identical_file_is_unchanged(self, tmp_path):
        from core.ingest import build_file_manifest, diff_file_manifest

        page = tmp_path / "a.md"
        page.write_text("# Same\n\nbody")
        before = build_file_manifest(tmp_path, (".md",))
        page.write_text("# Same\n\nbody")
        import os
        os.utime(page, (1, 1))
        after = build_file_manifest(tmp_path, (".md",))
        assert diff_file_manifest(after, before) == ([], [], [])

    def test_changed_content_is_modified(self, tmp_path):
        from core.ingest import build_file_manifest, diff_file_manifest

        page = tmp_path / "a.md"
        page.write_text("# One")
        before = build_file_manifest(tmp_path, (".md",))
        page.write_text("# Two")
        after = build_file_manifest(tmp_path, (".md",))
        assert diff_file_manifest(after, before)[1] == [str(page)]

    def test_old_manifest_without_hash_falls_back_to_mtime(self):
        from core.ingest import diff_file_manifest

        old = {"/x.md": {"mtime": 1.0, "size": 5}}
        assert diff_file_manifest({"/x.md": {"mtime": 1.0, "size": 5, "sha256": "aa"}}, old) == ([], [], [])
        assert diff_file_manifest({"/x.md": {"mtime": 2.0, "size": 5, "sha256": "aa"}}, old)[1] == ["/x.md"]


class TestIngestSummary:
    @pytest.fixture
    def client_with_pages(self, clients_dir, make_client):
        name = make_client("pages")
        docs = clients_dir / name / "documents"
        docs.mkdir()
        (docs / "good.md").write_text("# Good\n\n" + "A page about rivers and the valleys they carve. " * 4)
        (docs / "bad.md").write_text("# Bad\n\n" + "This page will fail to parse in the test. " * 4)
        return name, docs

    def _ingest_json(self, clients_dir, name, capsys, monkeypatch):
        import scripts.ingest as cli

        monkeypatch.setattr(sys, "argv", ["ingest", "--clients-dir", str(clients_dir), "--client", name,
                                          "--json", "--no-code"])
        cli.main()
        return json.loads(capsys.readouterr().out)

    def test_failed_file_is_counted_and_retried(self, client_with_pages, clients_dir, capsys, monkeypatch):
        """Regression: a file that failed to parse was saved in the manifest and never retried."""
        from core.ingest import DocumentIngestor

        name, docs = client_with_pages
        original = DocumentIngestor.chunk_file

        def flaky(self, path):
            if Path(path).name == "bad.md":
                raise ValueError("cannot parse")
            return original(self, path)

        monkeypatch.setattr(DocumentIngestor, "chunk_file", flaky)
        first = self._ingest_json(clients_dir, name, capsys, monkeypatch)
        assert first["new"] == 1 and first["failed"] == 1, first
        assert first["failed_files"] == [str(docs / "bad.md")]

        monkeypatch.setattr(DocumentIngestor, "chunk_file", original)
        second = self._ingest_json(clients_dir, name, capsys, monkeypatch)
        assert second["new"] == 1 and second["unchanged"] == 1 and second["failed"] == 0, second

    def test_deleted_file_is_reported_and_its_chunks_removed(self, client_with_pages, clients_dir, capsys, monkeypatch):
        """Existing behaviour, checked here so the README can rely on it."""
        from core.client import load_client
        from core.search import HybridSearcher

        name, docs = client_with_pages
        self._ingest_json(clients_dir, name, capsys, monkeypatch)
        (docs / "bad.md").unlink()
        summary = self._ingest_json(clients_dir, name, capsys, monkeypatch)
        assert summary["deleted"] == 1 and summary["chunks_removed"] > 0, summary

        client = load_client(name, clients_dir=clients_dir)
        searcher = HybridSearcher(index_path=str(client.index_path), collection_name=client.machine_name)
        sources = {m.get("source_file") for m in searcher.collection.get(include=["metadatas"])["metadatas"]}
        searcher.close()
        assert sources == {"good.md"}

    def test_summary_shape(self, client_with_pages, clients_dir, capsys, monkeypatch):
        name, _ = client_with_pages
        summary = self._ingest_json(clients_dir, name, capsys, monkeypatch)
        assert set(summary) >= {"client", "mode", "new", "changed", "unchanged", "deleted", "failed",
                                "failed_files", "chunks_added", "chunks_removed"}
        assert summary["client"] == name and summary["mode"] == "rebuild"


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


NOISY_PAGES = {
    "parking.md": ("Parking", "Customers can park for free in the car park behind the bakery on Mill Lane. "
                   "There are twelve spaces, two of them wide bays near the back door, and street parking is "
                   "allowed after six in the evening."),
    "hours.md": ("Opening hours", "The bakery opens at seven in the morning from Tuesday to Saturday and at "
                 "eight on Sunday. We close at four in the afternoon, and we are closed all day on Monday."),
    "starter.md": ("Feeding a sourdough starter", "Feed your sourdough starter once a day when it lives on the "
                   "counter: discard half, then add equal weights of flour and water. In the fridge, feed it "
                   "once a week."),
    "delivery.md": ("Delivery", "We deliver bread to the nearby villages of Ashby, Brook and Coldwell on "
                    "Wednesday and Friday mornings. Delivery costs two pounds and orders must arrive by noon "
                    "the day before."),
    "allergens.md": ("Allergens", "Our walnut loaf and the seeded rye contain tree nuts. Every loaf contains "
                     "gluten. Ask at the counter for the full allergen sheet before you buy."),
    "classes.md": ("Bread making classes", "Book a bread making class on the first Saturday of each month. "
                   "The class lasts three hours, costs forty pounds and you take home two loaves you shaped "
                   "yourself."),
}

NOISE_PAGES = {
    "news/_index.md": "# News\n\n" + "\n\n".join(
        f"- [{t}](/news/{i})\n\n  {d}" for i, (t, d) in enumerate([
            ("Saturday classes are back", "Booking opens soon for our bread classes."),
            ("A new delivery van", "Our van now reaches more villages."),
            ("Summer opening hours", "We open a little later in August."),
            ("Parking update", "The council repainted the car park."),
        ])),
    "_index.md": "# Mill Lane Bakery\n\n" + "\n\n".join(
        f"- [{t}](/{t.lower()})" for t in ["Home", "Bread", "Cakes", "Classes", "Delivery", "Parking",
                                          "Hours", "News", "Contact", "Allergens", "About", "Jobs"]),
    "contact.md": "# Contact",
    "rye.md": "# Seeded rye\n\nOur seeded rye is a dense loaf baked with a sourdough starter and sunflower "
              "seeds.\n\nShare\n\nFacebook\n\nTwitter\n\nEmail\n\nRelated\n\n- [Walnut loaf](/walnut)\n\n"
              "- [Spelt loaf](/spelt)\n\n- [Delivery](/delivery)\n\nFollow us\n\nInstagram",
}

NOISY_QUERIES = {
    "where can I park the car": "parking.md",
    "what time do you open on Sunday": "hours.md",
    "how often should I feed my sourdough starter": "starter.md",
    "do you deliver to nearby villages": "delivery.md",
    "which loaves contain nuts": "allergens.md",
    "can I book a bread making class": "classes.md",
}


def test_noisy_pages_do_not_outrank_the_right_page(clients_dir, make_client, capsys, monkeypatch):
    """Regression: listing pages and pages with Share/Related leftovers counted as 'data' and came first."""
    import scripts.ingest as ingest_cli
    import scripts.search as search_cli

    name = make_client("noisy")
    docs = clients_dir / name / "documents"
    for rel, (title, body) in NOISY_PAGES.items():
        (docs / rel).parent.mkdir(parents=True, exist_ok=True)
        (docs / rel).write_text(f"---\ntitle: {title}\nurl: https://bakery.example/{rel[:-3]}\n---\n# {title}\n\n{body}\n")
    for rel, text in NOISE_PAGES.items():
        (docs / rel).parent.mkdir(parents=True, exist_ok=True)
        (docs / rel).write_text(text + "\n")

    monkeypatch.setattr(sys, "argv", ["ingest", "--clients-dir", str(clients_dir), "--client", name,
                                      "--json", "--no-code"])
    ingest_cli.main()
    assert json.loads(capsys.readouterr().out)["failed"] == 0

    monkeypatch.setenv("TROVE_CLIENTS_DIR", str(clients_dir))
    firsts = {}
    for query, expected in NOISY_QUERIES.items():
        monkeypatch.setattr(sys, "argv", ["search", "--client", name, "--json", "--top-k", "3", query])
        search_cli.main()
        results = json.loads(capsys.readouterr().out)["results"]
        firsts[query] = results[0]["source"] if results else None
    assert firsts == NOISY_QUERIES, firsts
