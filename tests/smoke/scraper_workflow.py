"""
End to end: the scraper workflow.

A clients folder outside the repository (TROVE_CLIENTS_DIR), a client created
without the wizard, three scraped Markdown pages with frontmatter and headings,
ingest, search from the CLI (--json, one client and several) and through the
MCP handlers, then re-ingest to show unchanged files cost nothing.

Run it from a checkout:

    python tests/smoke/scraper_workflow.py

tests/test_scraper_workflow.py runs the same steps inside the test suite.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PAGES = {
    "starter.md": """---
title: Feeding a sourdough starter
url: https://bakery.example/guides/starter
date: 2026-10-01
domain: bakery.example
page_type: guide
scraped_at: 2026-11-01T09:30:00
---
# Feeding a sourdough starter

A sourdough starter is a culture of wild yeast and lactic acid bacteria, kept alive with regular feeding.

![A jar of bubbly starter](https://bakery.example/img/jar.jpg)

## Daily routine

Discard half of the starter, then add equal weights of rye flour and water. Keep it warm and stir well.

## Storage

A mature starter can live in the fridge and be fed once a week. Let it warm up before baking again.
""",
    "loaf.md": """---
title: Shaping a loaf
url: https://bakery.example/guides/loaf
date: 2026-10-05
domain: bakery.example
page_type: guide
---
# Shaping a loaf

## Pre-shape

Turn the dough out and fold it into a loose round. Rest it for twenty minutes so the gluten relaxes.

## Final shape

Tighten the round by dragging it towards you on an unfloured bench, then place it seam side up in a basket.
""",
    "oven.md": """---
title: Baking in a Dutch oven
url: https://bakery.example/guides/oven
date: 2026-10-09
domain: bakery.example
page_type: guide
---
# Baking in a Dutch oven

## Preheat

Heat the pot in the oven at 250 C for forty five minutes before the loaf goes in.

## Bake

Bake covered for twenty minutes to trap steam, then uncover and bake until the crust is deep brown.
""",
}


def _run(module_name: str, *argv: str) -> str:
    """Run a CLI script's main() in this process and return what it printed."""
    import importlib

    module = importlib.import_module(f"scripts.{module_name}")
    out = io.StringIO()
    old_argv = sys.argv
    sys.argv = [module_name, *argv]
    try:
        with contextlib.redirect_stdout(out):
            if module_name == "create_client":
                module.main(list(argv))
            else:
                module.main()
    finally:
        sys.argv = old_argv
    return out.getvalue()


def _count_embedding_calls():
    """Patch DocumentIngestor.create_embeddings to count the chunks it embeds."""
    from core.ingest import DocumentIngestor

    counter = {"chunks": 0}
    original = DocumentIngestor.create_embeddings

    def counting(self, chunks, *args, **kwargs):
        counter["chunks"] += len(chunks)
        return original(self, chunks, *args, **kwargs)

    DocumentIngestor.create_embeddings = counting
    return counter, lambda: setattr(DocumentIngestor, "create_embeddings", original)


def run(workdir: Path, verbose: bool = False) -> dict:
    """Run the workflow in ``workdir``; return what was observed. Raises AssertionError on failure."""
    say = print if verbose else (lambda *a, **k: None)
    clients_dir = Path(workdir) / "projects"
    clients_dir.mkdir()
    os.environ["TROVE_CLIENTS_DIR"] = str(clients_dir)
    (clients_dir / "_template").mkdir()  # never listed as a client

    observed: dict = {}
    counter, restore = _count_embedding_calls()
    try:
        # 1. Create a client without the wizard
        _run("create_client", "--name", "bakery", "--description", "Scraped baking guides")
        docs = clients_dir / "bakery" / "documents"
        for name, text in PAGES.items():
            (docs / name).write_text(text)
        say("created client 'bakery' with", len(PAGES), "pages")

        # 2. Ingest
        first = json.loads(_run("ingest", "--client", "bakery", "--json", "--no-code"))
        observed["first_ingest"] = first
        assert first["new"] == 3 and first["failed"] == 0, first
        assert counter["chunks"] > 0
        say("first ingest:", {k: first[k] for k in ("mode", "new", "changed", "unchanged", "failed", "chunks_added")})

        # 3. Search from the CLI
        out = json.loads(_run("search", "--client", "bakery", "--json", "how often should I feed my starter"))
        top = out["results"][0]
        observed["cli_top_result"] = {k: top[k] for k in ("client", "source", "title", "url", "section")}
        assert top["source"] == "starter.md", top
        assert top["title"] == "Feeding a sourdough starter"
        assert top["url"] == "https://bakery.example/guides/starter"
        assert top["section"].startswith("Feeding a sourdough starter")
        assert all(r["title"] and r["url"] and r["section"] for r in out["results"])
        assert "jar.jpg" not in json.dumps(out)
        say("CLI top result:", observed["cli_top_result"])

        # 4. Re-ingest with nothing changed, then with identical files rewritten
        counter["chunks"] = 0
        again = json.loads(_run("ingest", "--client", "bakery", "--json", "--no-code"))
        for name, text in PAGES.items():
            (docs / name).write_text(text)  # same content, new modification time
        rewritten = json.loads(_run("ingest", "--client", "bakery", "--json", "--no-code"))
        observed["unchanged_ingest"] = again
        observed["rewritten_ingest"] = rewritten
        observed["chunks_embedded_on_reingest"] = counter["chunks"]
        for summary in (again, rewritten):
            assert summary["unchanged"] == 3 and summary["new"] == summary["changed"] == 0, summary
            assert summary["chunks_added"] == 0, summary
        assert counter["chunks"] == 0, "an unchanged re-ingest embedded chunks"
        say("re-ingest, unchanged:", {k: again[k] for k in ("new", "changed", "unchanged", "chunks_added")},
            "| rewritten identical:", {k: rewritten[k] for k in ("new", "changed", "unchanged", "chunks_added")},
            "| chunks embedded:", counter["chunks"])

        # 5. A real change is picked up
        (docs / "oven.md").write_text(PAGES["oven.md"].replace("forty five", "sixty"))
        changed = json.loads(_run("ingest", "--client", "bakery", "--json", "--no-code"))
        observed["changed_ingest"] = changed
        assert changed["changed"] == 1 and changed["unchanged"] == 2, changed
        say("after editing one page:", {k: changed[k] for k in ("new", "changed", "unchanged")})

        # 6. Several clients from the CLI
        _run("create_client", "--name", "notes")
        (clients_dir / "notes" / "documents" / "rye.md").write_text(
            "---\ntitle: Rye notes\nurl: https://notes.example/rye\n---\n# Rye notes\n\n"
            "Rye flour ferments faster than wheat flour, so a rye starter needs feeding more often.\n"
        )
        json.loads(_run("ingest", "--client", "notes", "--json", "--no-code"))
        multi = json.loads(_run("search", "--all", "--json", "rye flour starter feeding"))
        observed["multi_clients"] = sorted({r["client"] for r in multi["results"]})
        assert set(multi["clients"]) == {"bakery", "notes"}, multi["clients"]
        assert observed["multi_clients"] == ["bakery", "notes"]
        say("--all searched:", multi["clients"], "results from:", observed["multi_clients"])

        # 7. The MCP handlers (they read TROVE_CLIENTS_DIR when the server starts)
        import importlib

        import mcp_server.shared as shared
        from core.client import resolve_clients_dir

        shared.CLIENTS_DIR = resolve_clients_dir()
        for name in ("clients", "graph", "ingest", "search", "sources"):
            module = importlib.import_module(f"mcp_server.handlers.{name}")
            if hasattr(module, "CLIENTS_DIR"):
                module.CLIENTS_DIR = shared.CLIENTS_DIR
        shared._cache._cache.pop("bakery", None)
        from mcp_server.handlers.search import handle_search_client

        text = handle_search_client({"client_name": "bakery", "query": "how hot should the dutch oven be"})[0].text
        observed["mcp_has_citation"] = all(
            s in text for s in ("**Title:** Baking in a Dutch oven", "**URL:** https://bakery.example/guides/oven",
                                "**Section:** Baking in a Dutch oven > Preheat")
        )
        assert observed["mcp_has_citation"], text
        say("MCP result shows title, URL and section:", observed["mcp_has_citation"])
    finally:
        restore()
        os.environ.pop("TROVE_CLIENTS_DIR", None)
        try:
            import mcp_server.shared as shared

            evicted = shared._cache._cache.pop("bakery", None)
            if evicted is not None:
                evicted[1].close()
        except Exception:
            pass
    return observed


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    with tempfile.TemporaryDirectory() as tmp:
        run(Path(tmp), verbose=True)
    print("scraper workflow passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
