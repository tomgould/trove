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


def _page_chunk(chunk_id, url, text, **meta):
    return {"id": chunk_id, "content": text,
            "metadata": {"source_file": url.rsplit("/", 1)[-1] + ".md", "url": url, **meta}}


class TestLimitPerPage:
    def test_extra_chunks_of_a_page_move_behind_other_pages(self):
        from core.search.searcher import limit_per_page

        chunks = [_page_chunk(i, url, "x") for i, url in
                  enumerate(["https://e.org/a", "https://e.org/a", "https://e.org/a", "https://e.org/b", "https://e.org/a"])]
        ordered = limit_per_page(chunks, 2)
        assert [c["id"] for c in ordered] == [0, 1, 3, 2, 4]

    def test_one_page_alone_keeps_all_its_chunks_in_order(self):
        from core.search.searcher import limit_per_page

        chunks = [_page_chunk(i, "https://e.org/a", "x") for i in range(5)]
        assert limit_per_page(chunks, 2) == chunks

    def test_zero_or_none_turns_it_off(self):
        from core.search.searcher import limit_per_page

        chunks = [_page_chunk(i, "https://e.org/a", "x") for i in range(3)] + [_page_chunk(9, "https://e.org/b", "x")]
        assert limit_per_page(chunks, 0) == chunks
        assert limit_per_page(chunks, None) == chunks

    def test_print_copy_counts_as_its_canonical_page(self):
        from core.search.searcher import page_key

        main = _page_chunk(1, "https://e.org/recipe", "x")
        copy = _page_chunk(2, "https://e.org/recipe/print/7", "x", canonical_url="https://e.org/recipe/")
        known = {"https://e.org/recipe", "https://e.org/recipe/print/7"}
        assert page_key(copy, known) == page_key(main, known) == "https://e.org/recipe"

    def test_canonical_url_of_an_unindexed_page_is_ignored(self):
        """A canonical URL that is no other file's URL does not merge unrelated pages."""
        from core.search.searcher import page_key

        chunk = _page_chunk(1, "https://e.org/one", "x", canonical_url="https://e.org/")
        assert page_key(chunk, {"https://e.org/one"}) == "https://e.org/one"

    def test_chunks_without_a_url_group_by_source_file(self):
        from core.search.searcher import page_key

        chunk = {"id": 1, "content": "x", "metadata": {"source_file": "manual.pdf"}}
        assert page_key(chunk) == "manual.pdf"


CHATTY = "https://recipes.example/frozen-grape-pops"
CHATTY_PRINT = "https://recipes.example/frozen-grape-pops/print/3"
GUIDE = "https://health.example/safe-snacks"


class TestPageDiversityInSearch:
    """One chatty page with many matching chunks (and a print copy) must not fill the top results."""

    def _corpus(self):
        chatty = [
            "Frozen grape pops for kids: wash the grapes, freeze the grapes, then serve the grape pops.",
            "Grape pops tips: choose seedless grapes, the kids love grapes, grapes make easy snacks for kids.",
            "Storing grape pops: keep the grapes frozen, the grape pops last a week, kids enjoy grapes cold.",
            "Grape pops variations: mix red grapes and green grapes so kids get colourful grape snacks.",
        ]
        chunks = [_page_chunk(f"c{i}", CHATTY, t) for i, t in enumerate(chatty)]
        chunks += [_page_chunk(f"p{i}", CHATTY_PRINT, t + " Printed copy.", canonical_url=CHATTY + "/")
                   for i, t in enumerate(chatty[:2])]
        chunks.append(_page_chunk("g0", GUIDE, "Choking hazards for young children: cut grapes lengthways into quarters."))
        chunks.append(_page_chunk("x0", "https://other.example/bread", "Bread rolls need flour, yeast and warm water."))
        return chunks

    def test_other_pages_reach_the_top_results(self, searcher_factory):
        searcher = searcher_factory(max_chunks_per_source=2)
        _index(searcher, self._corpus())

        results = searcher.search("grapes for kids", top_k=6)

        top3 = [r["metadata"]["url"] for r in results[:3]]
        assert GUIDE in top3, top3
        chatty_group = [u for u in top3 if u in (CHATTY, CHATTY_PRINT)]
        assert len(chatty_group) <= 2
        assert len(results) == 6  # moved chunks still come back, later in the list

    def test_off_when_set_to_zero(self, searcher_factory):
        searcher = searcher_factory(max_chunks_per_source=0)
        _index(searcher, self._corpus())

        results = searcher.search("grapes for kids", top_k=6)

        top3 = [r["metadata"]["url"] for r in results[:3]]
        assert GUIDE not in top3, top3
