"""Read a ChromaDB collection in pages.

ChromaDB's backend turns one ``get()`` into one SQL statement. With no ``limit``, a collection of
more than about 32,000 rows exceeds SQLite's limit on bound variables, and ``get()`` fails with
"too many SQL variables". Reading in pages keeps every statement small, so an index can grow
without a ceiling.

``get_all`` returns the same dict ``collection.get()`` returns. ``get_by_ids`` does the same for a
long list of ids.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

DEFAULT_PAGE_SIZE = 5000

# The list-valued keys of a ``collection.get()`` result, besides "ids".
_LIST_KEYS = ("documents", "metadatas", "embeddings", "uris", "data")


def _empty_result(include: Sequence[str]) -> dict[str, Any]:
    """A result with every requested key as an empty list and the rest None, as ChromaDB returns."""
    result: dict[str, Any] = {"ids": []}
    for key in _LIST_KEYS:
        result[key] = [] if key in include else None
    return result


def _merge(result: dict[str, Any], page: dict[str, Any]) -> None:
    result["ids"].extend(page.get("ids") or [])
    for key in _LIST_KEYS:
        values = page.get(key)
        if result[key] is not None and values is not None:
            result[key].extend(list(values))


def get_all(
    collection: Any,
    include: Optional[Sequence[str]] = None,
    where: Optional[dict] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    """Return every row of ``collection`` (or every row matching ``where``), read in pages.

    ``include`` names the fields to return, as for ``collection.get()``; ``None`` means documents
    and metadatas. Rows come back in the order ChromaDB lists them.
    """
    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    fields = list(include) if include is not None else ["documents", "metadatas"]
    result = _empty_result(fields)
    offset = 0
    while True:
        kwargs: dict[str, Any] = {"include": fields, "limit": page_size, "offset": offset}
        if where is not None:
            kwargs["where"] = where
        page = collection.get(**kwargs)
        count = len(page.get("ids") or [])
        if count == 0:
            break
        _merge(result, page)
        offset += count
        if count < page_size:
            break
    return result


def get_by_ids(
    collection: Any,
    ids: Sequence[str],
    include: Optional[Sequence[str]] = None,
    batch_size: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    """Return the rows for ``ids``, fetched in batches of ``batch_size`` ids."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    fields = list(include) if include is not None else ["documents", "metadatas"]
    result = _empty_result(fields)
    ids = list(ids)
    for start in range(0, len(ids), batch_size):
        page = collection.get(ids=ids[start:start + batch_size], include=fields)
        _merge(result, page)
    return result
