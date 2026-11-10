"""Paged reads of a ChromaDB collection (core/search/chroma_paging.py).

A plain ``collection.get()`` on more than about 32,000 rows fails with "too many SQL variables".
These tests cover the helper on a fake collection and an index past that limit.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.search.chroma_paging import get_all, get_by_ids

# More rows than SQLite's default limit of 32,766 bound variables.
LARGE = 33_000


class FakeCollection:
    """Answers get() the way ChromaDB does: limit and offset, ids, and None for fields not asked for."""

    def __init__(self, count):
        self.rows = [(f"id{i}", f"doc {i}", {"n": i}) for i in range(count)]
        self.calls = []

    def get(self, ids=None, include=None, where=None, limit=None, offset=None):
        self.calls.append({"ids": ids, "limit": limit, "offset": offset, "where": where})
        include = include or []
        rows = self.rows
        if ids is not None:
            wanted = set(ids)
            rows = [r for r in rows if r[0] in wanted]
        if where is not None:
            rows = [r for r in rows if r[2].get("n", -1) % 2 == where["even"]]
        start = offset or 0
        rows = rows[start:] if limit is None else rows[start:start + limit]
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows] if "documents" in include else None,
            "metadatas": [r[2] for r in rows] if "metadatas" in include else None,
            "embeddings": None,
            "uris": None,
            "data": None,
        }


class TestGetAll:
    def test_collects_every_row_in_order_across_pages(self):
        collection = FakeCollection(23)
        result = get_all(collection, include=["documents", "metadatas"], page_size=10)
        assert result["ids"] == [f"id{i}" for i in range(23)]
        assert result["documents"] == [f"doc {i}" for i in range(23)]
        assert [m["n"] for m in result["metadatas"]] == list(range(23))
        assert [c["offset"] for c in collection.calls] == [0, 10, 20]
        assert all(c["limit"] == 10 for c in collection.calls)

    def test_exact_multiple_of_page_size_stops_cleanly(self):
        collection = FakeCollection(20)
        result = get_all(collection, include=["documents"], page_size=10)
        assert len(result["ids"]) == 20
        assert len(collection.calls) == 3  # the third page is empty and ends the loop

    def test_empty_collection_gives_empty_lists_for_requested_fields(self):
        result = get_all(FakeCollection(0), include=["documents", "metadatas"])
        assert result["ids"] == []
        assert result["documents"] == []
        assert result["metadatas"] == []

    def test_fields_not_requested_stay_none(self):
        result = get_all(FakeCollection(5), include=[])
        assert result["ids"] == [f"id{i}" for i in range(5)]
        assert result["documents"] is None
        assert result["metadatas"] is None

    def test_default_include_is_documents_and_metadatas(self):
        result = get_all(FakeCollection(3))
        assert len(result["documents"]) == 3
        assert len(result["metadatas"]) == 3

    def test_where_is_passed_to_every_page(self):
        collection = FakeCollection(12)
        result = get_all(collection, include=[], where={"even": 0}, page_size=4)
        assert result["ids"] == ["id0", "id2", "id4", "id6", "id8", "id10"]
        assert all(c["where"] == {"even": 0} for c in collection.calls)

    def test_bad_page_size_is_refused(self):
        with pytest.raises(ValueError):
            get_all(FakeCollection(1), page_size=0)


class TestGetByIds:
    def test_fetches_in_batches_and_keeps_every_row(self):
        collection = FakeCollection(30)
        ids = [f"id{i}" for i in range(0, 30, 2)]
        result = get_by_ids(collection, ids, include=["documents"], batch_size=4)
        assert result["ids"] == ids
        assert result["documents"] == [f"doc {i}" for i in range(0, 30, 2)]
        assert len(collection.calls) == 4  # 15 ids in batches of 4
        assert all(len(c["ids"]) <= 4 for c in collection.calls)

    def test_no_ids_gives_an_empty_result_without_a_call(self):
        collection = FakeCollection(5)
        result = get_by_ids(collection, [], include=["documents"])
        assert result["ids"] == []
        assert result["documents"] == []
        assert collection.calls == []


@pytest.fixture
def temp_index():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def _fill(searcher, count, batch=4000):
    """Add ``count`` small chunks straight to the collection, with zero vectors."""
    dimension = searcher.embedding_dimension
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        searcher.collection.add(
            ids=[f"chunk-{i}" for i in range(start, stop)],
            documents=[f"recipe chunk number {i} with oats and banana" for i in range(start, stop)],
            metadatas=[{"source_path": f"page-{i // 5}.md", "title": f"Page {i // 5}"} for i in range(start, stop)],
            embeddings=np.zeros((stop - start, dimension), dtype=np.float32),
        )


class TestIndexPastSqliteLimit:
    def test_get_all_reads_every_row_of_a_large_collection(self, temp_index):
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        _fill(searcher, LARGE)
        assert searcher.collection.count() == LARGE

        everything = get_all(searcher.collection, include=["documents", "metadatas"])
        assert len(everything["ids"]) == LARGE
        assert len(set(everything["ids"])) == LARGE
        assert len(everything["documents"]) == LARGE
        assert len(everything["metadatas"]) == LARGE

        ids_only = get_all(searcher.collection, include=[])
        assert len(ids_only["ids"]) == LARGE
        searcher.close()

    def test_opening_a_large_index_without_a_bm25_cache_rebuilds_it(self, temp_index):
        """The failure seen on a 98,000 chunk index: opening it or searching it raised
        InternalError 'too many SQL variables' while rebuilding the keyword index."""
        from core.search import HybridSearcher

        searcher = HybridSearcher(index_path=temp_index)
        _fill(searcher, LARGE)
        searcher.close()
        cache = Path(temp_index) / "bm25_cache.json"
        for leftover in Path(temp_index).glob("bm25_cache*"):
            leftover.unlink()
        assert not cache.exists()

        reopened = HybridSearcher(index_path=temp_index)
        assert reopened.bm25_index is not None
        assert len(reopened.bm25_chunks) == LARGE
        reopened.close()
