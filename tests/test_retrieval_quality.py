"""
Retrieval quality on small synthetic corpora: result count, per-page diversity
and heading matches. Each corpus is invented to reproduce one failure mode.
"""

import pytest


def _index(searcher, chunks):
    embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
    searcher.index_chunks(chunks, embeddings)


@pytest.fixture
def searcher_factory(tmp_path):
    from core.search import HybridSearcher

    made = []

    def make(**kwargs):
        searcher = HybridSearcher(index_path=str(tmp_path / f"idx{len(made)}"), ephemeral=True,
                                  collection_name=f"quality{len(made)}", cache_ttl=0, **kwargs)
        made.append(searcher)
        return searcher

    yield make
    for searcher in made:
        searcher.close()


ANIMALS = [
    "otter", "badger", "heron", "falcon", "beaver", "marten", "weasel", "plover", "curlew", "lapwing",
    "osprey", "kestrel", "puffin", "gannet", "shrew", "vole", "stoat", "ferret", "hedgehog", "dormouse",
    "squirrel", "rabbit", "hare", "deer", "fox", "wren", "robin", "thrush", "finch", "linnet",
    "swift", "swallow", "martin", "pipit", "wagtail", "dipper", "kingfisher", "egret", "bittern", "crane",
    "stork", "ibis", "spoonbill", "avocet", "dunlin", "knot", "sanderling", "turnstone", "redshank", "godwit",
]


class TestTopK:
    def test_top_k_above_the_adaptive_depth_is_honoured(self, searcher_factory):
        """Regression: with a confident top match, adaptive reranking cut the candidates to its
        depth, so asking for more results than that still returned only the depth."""
        searcher = searcher_factory(adaptive_rerank_depth=15)
        chunks = [
            {"id": f"c{i}", "content": f"Field notes on the {name}: the {name} lives near the river bank.",
             "metadata": {"source_file": f"{name}.md"}}
            for i, name in enumerate(ANIMALS)
        ]
        _index(searcher, chunks)

        results = searcher.search("Field notes on the otter: the otter lives near the river bank.", top_k=40)

        assert searcher._last_search_quality.adaptive_rerank_applied
        assert len(results) == 40

    def test_top_k_above_rerank_top_n_is_honoured(self, searcher_factory):
        """A top_k larger than the candidate pool widens the pool instead of being cut to it."""
        searcher = searcher_factory(adaptive_rerank=False)
        chunks = [
            {"id": f"c{i}", "content": f"The {name} was seen near the river bank this morning.",
             "metadata": {"source_file": f"{name}.md"}}
            for i, name in enumerate(ANIMALS)
        ]
        _index(searcher, chunks)

        results = searcher.search("river bank sightings", top_k=45, rerank_top_n=20)

        assert len(results) == 45

    def test_top_k_is_capped_at_the_maximum(self, searcher_factory):
        from core.search.searcher import MAX_TOP_K

        searcher = searcher_factory(adaptive_rerank=False)
        chunks = [
            {"id": f"c{i}", "content": f"Entry {i}: the {ANIMALS[i % 50]} number {i} by the river bank.",
             "metadata": {"source_file": f"entry{i}.md"}}
            for i in range(MAX_TOP_K + 20)
        ]
        _index(searcher, chunks)

        results = searcher.search("river bank", top_k=MAX_TOP_K + 50)

        assert len(results) == MAX_TOP_K
