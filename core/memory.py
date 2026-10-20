"""
Persistent memory store for Trove agent orchestration.

Stores and retrieves context across Claude Code sessions using SQLite.
Supports semantic search via sentence-transformer embeddings and
LIKE-based fallback when no embedding model is available.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MemoryStore:
    """SQLite-backed persistent memory with optional vector search.

    Args:
        db_path: Path to SQLite database file.
        embedding_model: Optional SentenceTransformer model for semantic search.
    """

    def __init__(self, db_path: Path, embedding_model=None,
                 connection: sqlite3.Connection = None,
                 lock: threading.Lock = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self._lock = lock or threading.Lock()
        self._owns_conn = connection is None
        if connection is not None:
            self._conn = connection
        else:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL,
                    UNIQUE(namespace, key)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_namespace
                    ON memories(namespace);
                CREATE INDEX IF NOT EXISTS idx_memories_expires
                    ON memories(expires_at) WHERE expires_at IS NOT NULL;
            """)
            self._conn.commit()

    def store(self, namespace: str, key: str, content: str,
              metadata: dict = None, ttl: int = None) -> str:
        """Upsert a memory entry.

        Args:
            namespace: Logical grouping (e.g. 'decisions', 'patterns').
            key: Unique key within the namespace.
            content: The content to store.
            metadata: Optional dict of extra metadata.
            ttl: Optional time-to-live in seconds.

        Returns:
            The memory ID.
        """
        now = time.time()
        expires_at = now + ttl if ttl else None
        metadata_json = json.dumps(metadata) if metadata else None

        embedding_bytes = None
        if self.embedding_model is not None:
            try:
                vec = self.embedding_model.encode(content)
                embedding_bytes = np.array(vec, dtype=np.float32).tobytes()
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        # Check if exists to preserve ID and created_at
        with self._lock:
            row = self._conn.execute(
                "SELECT id, created_at FROM memories WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()

            if row:
                memory_id, created_at = row
                self._conn.execute(
                    """UPDATE memories SET content = ?, metadata = ?, embedding = ?,
                       updated_at = ?, expires_at = ? WHERE id = ?""",
                    (content, metadata_json, embedding_bytes, now, expires_at, memory_id),
                )
            else:
                memory_id = str(uuid.uuid4())
                created_at = now
                self._conn.execute(
                    """INSERT INTO memories
                       (id, namespace, key, content, metadata, embedding,
                        created_at, updated_at, expires_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (memory_id, namespace, key, content, metadata_json,
                     embedding_bytes, created_at, now, expires_at),
                )
            self._conn.commit()

        return memory_id

    def _not_expired_clause(self) -> str:
        """SQL clause to exclude expired entries."""
        return "(expires_at IS NULL OR expires_at > ?)"

    def retrieve(self, namespace: str, key: str) -> Optional[dict]:
        """Exact lookup by namespace and key.

        Returns:
            Dict with id, namespace, key, content, metadata, timestamps, or None.
            Returns None for expired entries.
        """
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                f"""SELECT id, namespace, key, content, metadata,
                          created_at, updated_at, expires_at
                   FROM memories WHERE namespace = ? AND key = ?
                   AND {self._not_expired_clause()}""",
                (namespace, key, now),
            ).fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "namespace": row[1],
            "key": row[2],
            "content": row[3],
            "metadata": json.loads(row[4]) if row[4] else None,
            "created_at": row[5],
            "updated_at": row[6],
            "expires_at": row[7],
        }

    def search(self, query: str, namespace: str = None,
               top_k: int = 5) -> list[dict]:
        """Search memories by semantic similarity or LIKE fallback.

        Args:
            query: Search query string.
            namespace: Optional namespace filter.
            top_k: Max results to return.

        Returns:
            List of memory dicts with similarity scores, sorted by relevance.
        """
        if self.embedding_model is not None:
            return self._vector_search(query, namespace, top_k)
        return self._like_search(query, namespace, top_k)

    def _vector_search(self, query: str, namespace: Optional[str],
                       top_k: int) -> list[dict]:
        query_vec = self.embedding_model.encode(query)
        query_vec = np.array(query_vec, dtype=np.float32)
        now = time.time()

        where = f"WHERE embedding IS NOT NULL AND {self._not_expired_clause()}"
        params: list = [now]
        if namespace:
            where += " AND namespace = ?"
            params.append(namespace)

        with self._lock:
            rows = self._conn.execute(
                f"""SELECT id, namespace, key, content, metadata,
                           embedding, created_at, updated_at, expires_at
                    FROM memories {where}""",
                params,
            ).fetchall()

        results = []
        for row in rows:
            stored_vec = np.frombuffer(row[5], dtype=np.float32)
            # Cosine similarity
            dot = np.dot(query_vec, stored_vec)
            norm = np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
            similarity = float(dot / norm) if norm > 0 else 0.0

            results.append({
                "id": row[0],
                "namespace": row[1],
                "key": row[2],
                "content": row[3],
                "metadata": json.loads(row[4]) if row[4] else None,
                "created_at": row[6],
                "updated_at": row[7],
                "expires_at": row[8],
                "similarity": similarity,
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _like_search(self, query: str, namespace: Optional[str],
                     top_k: int) -> list[dict]:
        now = time.time()
        escaped_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where = f"WHERE content LIKE ? ESCAPE '\\' AND {self._not_expired_clause()}"
        params: list = [f"%{escaped_query}%", now]
        if namespace:
            where += " AND namespace = ?"
            params.append(namespace)

        with self._lock:
            rows = self._conn.execute(
                f"""SELECT id, namespace, key, content, metadata,
                           created_at, updated_at, expires_at
                    FROM memories {where}
                    ORDER BY updated_at DESC
                    LIMIT ?""",
                params + [top_k],
            ).fetchall()

        return [
            {
                "id": row[0],
                "namespace": row[1],
                "key": row[2],
                "content": row[3],
                "metadata": json.loads(row[4]) if row[4] else None,
                "created_at": row[5],
                "updated_at": row[6],
                "expires_at": row[7],
            }
            for row in rows
        ]

    def list(self, namespace: str = None) -> list[dict]:
        """List all non-expired memories, optionally filtered by namespace.

        Returns:
            List of memory summary dicts.
        """
        now = time.time()
        where = f"WHERE {self._not_expired_clause()}"
        params: list = [now]
        if namespace:
            where += " AND namespace = ?"
            params.append(namespace)

        with self._lock:
            rows = self._conn.execute(
                f"""SELECT id, namespace, key, content, metadata,
                           created_at, updated_at, expires_at
                    FROM memories {where}
                    ORDER BY updated_at DESC""",
                params,
            ).fetchall()

        return [
            {
                "id": row[0],
                "namespace": row[1],
                "key": row[2],
                "content": row[3],
                "metadata": json.loads(row[4]) if row[4] else None,
                "created_at": row[5],
                "updated_at": row[6],
                "expires_at": row[7],
            }
            for row in rows
        ]

    def delete(self, namespace: str, key: str) -> bool:
        """Delete a memory by namespace and key.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def cleanup(self) -> int:
        """Remove expired memories.

        Returns:
            Number of entries removed.
        """
        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            self._conn.commit()
            return cursor.rowcount

    def close(self):
        """Close the SQLite connection (only if owned by this instance)."""
        if self._owns_conn:
            with self._lock:
                self._conn.close()
