"""Search result caching with LRU eviction and TTL expiry."""

import time
from collections import OrderedDict

# BM25 cache configuration
BM25_CACHE_FILENAME = "bm25_cache.json"


class SearchCache:
    """In-memory LRU cache with TTL for search results.

    Uses OrderedDict for O(1) LRU eviction. Each entry stores
    (value, timestamp) and is checked for TTL on access.
    """

    def __init__(self, max_size: int = 128, ttl: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple] = OrderedDict()

    def get(self, key: str):
        """Get a cached value. Returns None on miss or expiry."""
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]

        # Check TTL
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return value

    def put(self, key: str, value) -> None:
        """Store a value in the cache."""
        if key in self._cache:
            del self._cache[key]

        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Remove all cached entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Number of entries currently in cache."""
        return len(self._cache)
