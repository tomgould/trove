"""
Ingest a folder of Markdown files and search it, using only the core install.

Run it directly to check an install:

    python -I tests/smoke/core_smoke.py

``-I`` keeps the source tree off ``sys.path``, so run from another folder it
tests the installed package rather than the checkout. tests/test_smoke.py runs
the same check inside the test suite.
"""

import sys
import tempfile
from pathlib import Path

DOCUMENTS = {
    "solar.md": (
        "# Solar panels\n\n"
        "Photovoltaic solar panels turn sunlight into electricity. Each cell is made of "
        "silicon, and an inverter converts the direct current into alternating current "
        "for the house.\n"
    ),
    "bread.md": (
        "# Sourdough bread\n\n"
        "A sourdough starter is a living culture of wild yeast and lactic acid bacteria. "
        "Feed the starter with flour and water every day, and it will leaven the bread "
        "dough without any packet yeast.\n"
    ),
    "rivers.md": (
        "# Rivers\n\n"
        "A river carries water from its source in the hills down to the sea. Over time "
        "it erodes the valley floor and deposits silt where the current slows near the "
        "estuary.\n"
    ),
}


def run(workdir: Path) -> list[dict]:
    """Create a client in ``workdir``, ingest the documents and search them."""
    from core.client import load_client
    from core.ingest import DocumentIngestor, ingest_directory
    from core.search import HybridSearcher
    from core.search.searcher import chunk_source

    clients_dir = Path(workdir) / "clients"
    client_dir = clients_dir / "smoke"
    docs_dir = client_dir / "documents"
    docs_dir.mkdir(parents=True)
    (client_dir / "client.yaml").write_text("name: Smoke test\nmachine_name: smoke\n")
    for name, text in DOCUMENTS.items():
        (docs_dir / name).write_text(text)

    client = load_client("smoke", clients_dir=clients_dir)
    assert not client.has_index, "a new client should not report an index"

    searcher = HybridSearcher(
        index_path=str(client.index_path),
        collection_name=client.machine_name,
        embedding_config=client.embedding_config,
        tokenizer_config=client.tokenization_config,
        skip_bm25_load=True,
    )
    ingestor = DocumentIngestor(embedding_model=client.embedding_config.get("model", "all-MiniLM-L6-v2"))
    ingest_directory(docs_dir, ingestor, extensions=(".md",), pipeline_searcher=searcher)
    searcher._rebuild_bm25_index()

    assert client.has_index, "the client should report an index after ingest"
    assert searcher.get_stats().get("total_chunks", 0) >= len(DOCUMENTS)

    results = searcher.search("how do I keep a sourdough starter alive", top_k=3)
    assert results, "the search returned nothing"
    assert chunk_source(results[0]) == "bread.md", [chunk_source(r) for r in results]

    filtered = searcher.search(
        "how do I keep a sourdough starter alive", top_k=3, source_filter=["solar.md", "rivers.md"]
    )
    assert all(chunk_source(r) in {"solar.md", "rivers.md"} for r in filtered), [chunk_source(r) for r in filtered]
    return results


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        results = run(Path(tmp))
    print(f"core smoke test passed: top result {results[0]['metadata'].get('source_file')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
