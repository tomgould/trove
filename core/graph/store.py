"""
Knowledge graph storage backed by NetworkX + SQLite persistence.

Stores entities and relationships, supports traversal queries,
and persists to clients/{name}/index/knowledge_graph.db.

Incremental saves: only dirty entities/relationships are written,
making periodic saves during graph building fast regardless of graph size.

Backward compatible: auto-migrates old knowledge_graph.json on first load.
"""

import json
import logging
import sqlite3
from collections import Counter
from pathlib import Path

from core.graph.models import Entity, Relationship

logger = logging.getLogger(__name__)

GRAPH_FILENAME = "knowledge_graph.db"
GRAPH_JSON_FILENAME = "knowledge_graph.json"  # Legacy format
GRAPH_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    key TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    description TEXT DEFAULT '',
    source_chunk_ids TEXT DEFAULT '[]',
    aliases TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS relationships (
    key TEXT PRIMARY KEY,
    source_entity TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    description TEXT DEFAULT '',
    weight REAL DEFAULT 1.0,
    source_chunk_ids TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS processed_chunks (
    chunk_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class KnowledgeGraph:
    """In-memory knowledge graph with SQLite persistence."""

    def __init__(self, index_path: Path):
        self._index_path = Path(index_path)
        self._entities: dict[str, Entity] = {}  # key -> Entity
        self._relationships: dict[str, Relationship] = {}  # key -> Relationship
        self._processed_chunk_ids: set[str] = set()  # All chunks ever processed (including zero-entity ones)
        self._graph = None  # Lazy-loaded NetworkX DiGraph
        self._dirty_entities: set[str] = set()  # Keys modified since last save
        self._dirty_relationships: set[str] = set()
        self._dirty_chunks: set[str] = set()  # Processed chunk IDs not yet persisted
        self._db: sqlite3.Connection | None = None

    def _get_db(self) -> sqlite3.Connection:
        """Get or create SQLite connection."""
        if self._db is None:
            db_path = self._index_path / GRAPH_FILENAME
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(db_path), timeout=10)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(_SCHEMA)
            # Store version
            self._db.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('version', ?)",
                (str(GRAPH_VERSION),)
            )
            self._db.commit()
        return self._db

    def _ensure_graph(self):
        """Lazy-import and initialize NetworkX graph."""
        if self._graph is None:
            import networkx as nx
            self._graph = nx.DiGraph()
        return self._graph

    def _rebuild_nx_graph(self):
        """Rebuild NetworkX graph from entities and relationships."""
        import networkx as nx
        self._graph = nx.DiGraph()
        for key, entity in self._entities.items():
            self._graph.add_node(key, **entity.to_dict())
        for key, rel in self._relationships.items():
            if rel.source_entity in self._entities and rel.target_entity in self._entities:
                self._graph.add_edge(
                    rel.source_entity,
                    rel.target_entity,
                    key=key,
                    relationship_type=rel.relationship_type,
                    weight=rel.weight,
                )

    def add_entity(self, entity: Entity) -> None:
        """Add an entity, merging if one with the same key exists."""
        key = entity.key
        if key in self._entities:
            self._entities[key].merge(entity)
        else:
            self._entities[key] = entity
        self._dirty_entities.add(key)
        # Update NetworkX node
        g = self._ensure_graph()
        g.add_node(key, **self._entities[key].to_dict())

    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship, merging if one with the same key exists."""
        key = rel.key
        if key in self._relationships:
            self._relationships[key].merge(rel)
        else:
            self._relationships[key] = rel
        self._dirty_relationships.add(key)
        # Update NetworkX edge
        g = self._ensure_graph()
        stored = self._relationships[key]
        if stored.source_entity in self._entities and stored.target_entity in self._entities:
            g.add_edge(
                stored.source_entity,
                stored.target_entity,
                key=key,
                relationship_type=stored.relationship_type,
                weight=stored.weight,
            )

    def get_entity(self, key: str) -> Entity | None:
        return self._entities.get(key)

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        et = entity_type.lower().strip()
        return [e for e in self._entities.values() if e.entity_type.lower().strip() == et]

    def find_entities_by_name(self, query: str) -> list[Entity]:
        """Find entities by substring match on name or aliases."""
        q = query.lower().strip()
        results = []
        for entity in self._entities.values():
            if q in entity.name.lower():
                results.append(entity)
                continue
            if any(q in alias.lower() for alias in entity.aliases):
                results.append(entity)
        return results

    def get_neighbors(self, entity_key: str, max_depth: int = 2) -> list[tuple[Entity, Relationship]]:
        """BFS traversal from entity up to max_depth hops. Returns (entity, relationship) pairs."""
        g = self._ensure_graph()
        if entity_key not in g:
            return []

        visited = {entity_key}
        results = []
        frontier = [entity_key]

        for _ in range(max_depth):
            next_frontier = []
            for node in frontier:
                # Outgoing edges
                for _, target, data in g.out_edges(node, data=True):
                    if target not in visited:
                        visited.add(target)
                        next_frontier.append(target)
                        rel_key = data.get("key", "")
                        rel = self._relationships.get(rel_key)
                        entity = self._entities.get(target)
                        if entity and rel:
                            results.append((entity, rel))
                # Incoming edges
                for source, _, data in g.in_edges(node, data=True):
                    if source not in visited:
                        visited.add(source)
                        next_frontier.append(source)
                        rel_key = data.get("key", "")
                        rel = self._relationships.get(rel_key)
                        entity = self._entities.get(source)
                        if entity and rel:
                            results.append((entity, rel))
            frontier = next_frontier
            if not frontier:
                break

        return results

    def get_source_chunks(self, entity_keys: list[str]) -> list[str]:
        """Get all unique chunk IDs referenced by the given entities."""
        chunk_ids = set()
        for key in entity_keys:
            entity = self._entities.get(key)
            if entity:
                chunk_ids.update(entity.source_chunk_ids)
            # Also include chunks from relationships involving this entity
            for rel in self._relationships.values():
                if rel.source_entity == key or rel.target_entity == key:
                    chunk_ids.update(rel.source_chunk_ids)
        return list(chunk_ids)

    @property
    def processed_chunk_ids(self) -> set[str]:
        """All chunk IDs that have been processed for graph extraction."""
        return self._processed_chunk_ids

    def mark_chunks_processed(self, chunk_ids: list[str]) -> None:
        """Record chunk IDs as processed (even if they yielded zero entities)."""
        new_ids = set(chunk_ids) - self._processed_chunk_ids
        self._processed_chunk_ids.update(new_ids)
        self._dirty_chunks.update(new_ids)

    def save(self) -> None:
        """Incrementally save dirty entities, relationships, and processed chunks to SQLite."""
        db = self._get_db()

        dirty_ent_count = len(self._dirty_entities)
        dirty_rel_count = len(self._dirty_relationships)
        dirty_chunk_count = len(self._dirty_chunks)

        if dirty_ent_count == 0 and dirty_rel_count == 0 and dirty_chunk_count == 0:
            return  # Nothing to save

        # Write dirty entities
        if self._dirty_entities:
            ent_rows = []
            for key in self._dirty_entities:
                e = self._entities.get(key)
                if e:
                    ent_rows.append((
                        key, e.name, e.entity_type, e.description,
                        json.dumps(e.source_chunk_ids), json.dumps(e.aliases),
                    ))
            db.executemany(
                "INSERT OR REPLACE INTO entities (key, name, entity_type, description, source_chunk_ids, aliases) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ent_rows,
            )
            self._dirty_entities.clear()

        # Write dirty relationships
        if self._dirty_relationships:
            rel_rows = []
            for key in self._dirty_relationships:
                r = self._relationships.get(key)
                if r:
                    rel_rows.append((
                        key, r.source_entity, r.target_entity, r.relationship_type,
                        r.description, r.weight, json.dumps(r.source_chunk_ids),
                    ))
            db.executemany(
                "INSERT OR REPLACE INTO relationships (key, source_entity, target_entity, "
                "relationship_type, description, weight, source_chunk_ids) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rel_rows,
            )
            self._dirty_relationships.clear()

        # Write dirty processed chunks
        if self._dirty_chunks:
            db.executemany(
                "INSERT OR IGNORE INTO processed_chunks (chunk_id) VALUES (?)",
                [(cid,) for cid in self._dirty_chunks],
            )
            self._dirty_chunks.clear()

        db.commit()

        logger.info(
            f"Saved knowledge graph: {len(self._entities)} entities, "
            f"{len(self._relationships)} relationships "
            f"(wrote {dirty_ent_count}e/{dirty_rel_count}r/{dirty_chunk_count}c)"
        )

    def load(self) -> bool:
        """Load graph from SQLite (or migrate from legacy JSON). Returns False if nothing found."""
        db_path = self._index_path / GRAPH_FILENAME
        json_path = self._index_path / GRAPH_JSON_FILENAME

        # Try SQLite first
        if db_path.exists():
            return self._load_from_sqlite()

        # Fall back to legacy JSON migration
        if json_path.exists():
            logger.info(f"Migrating legacy JSON graph to SQLite: {json_path}")
            if self._load_from_json(json_path):
                # Mark everything as dirty so save() writes it all to SQLite
                self._dirty_entities = set(self._entities.keys())
                self._dirty_relationships = set(self._relationships.keys())
                self._dirty_chunks = set(self._processed_chunk_ids)
                self.save()
                logger.info("Migration complete — legacy JSON can be removed")
                return True
            return False

        return False

    def _load_from_sqlite(self) -> bool:
        """Load graph from SQLite database."""
        try:
            db = self._get_db()

            self._entities.clear()
            self._relationships.clear()
            self._processed_chunk_ids.clear()

            # Load entities
            for row in db.execute("SELECT key, name, entity_type, description, source_chunk_ids, aliases FROM entities"):
                entity = Entity(
                    name=row[1],
                    entity_type=row[2],
                    description=row[3] or "",
                    source_chunk_ids=json.loads(row[4]) if row[4] else [],
                    aliases=json.loads(row[5]) if row[5] else [],
                )
                self._entities[row[0]] = entity

            # Load relationships
            for row in db.execute(
                "SELECT key, source_entity, target_entity, relationship_type, "
                "description, weight, source_chunk_ids FROM relationships"
            ):
                rel = Relationship(
                    source_entity=row[1],
                    target_entity=row[2],
                    relationship_type=row[3],
                    description=row[4] or "",
                    weight=row[5] if row[5] is not None else 1.0,
                    source_chunk_ids=json.loads(row[6]) if row[6] else [],
                )
                self._relationships[row[0]] = rel

            # Load processed chunks
            for row in db.execute("SELECT chunk_id FROM processed_chunks"):
                self._processed_chunk_ids.add(row[0])

            self._rebuild_nx_graph()

            logger.info(
                f"Loaded knowledge graph: {len(self._entities)} entities, "
                f"{len(self._relationships)} relationships, "
                f"{len(self._processed_chunk_ids)} processed chunks"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to load knowledge graph from SQLite: {e}")
            return False

    def _load_from_json(self, path: Path) -> bool:
        """Load graph from legacy JSON file (for migration)."""
        try:
            with open(path) as f:
                data = json.load(f)

            self._entities.clear()
            self._relationships.clear()
            self._processed_chunk_ids.clear()

            for ed in data.get("entities", []):
                entity = Entity.from_dict(ed)
                self._entities[entity.key] = entity

            for rd in data.get("relationships", []):
                rel = Relationship.from_dict(rd)
                self._relationships[rel.key] = rel

            # Load processed chunk IDs, backfilling from entities/rels if absent (old format)
            if "processed_chunk_ids" in data:
                self._processed_chunk_ids = set(data["processed_chunk_ids"])
            else:
                for e in self._entities.values():
                    self._processed_chunk_ids.update(e.source_chunk_ids)
                for r in self._relationships.values():
                    self._processed_chunk_ids.update(r.source_chunk_ids)
                logger.info("Backfilled processed_chunk_ids from entity/relationship sources")

            self._rebuild_nx_graph()

            logger.info(
                f"Loaded knowledge graph from JSON: {len(self._entities)} entities, "
                f"{len(self._relationships)} relationships"
            )
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load knowledge graph from JSON: {e}")
            return False

    def clear(self) -> None:
        """Clear all entities, relationships, and processed chunk tracking."""
        self._entities.clear()
        self._relationships.clear()
        self._processed_chunk_ids.clear()
        self._dirty_entities.clear()
        self._dirty_relationships.clear()
        self._dirty_chunks.clear()
        if self._graph is not None:
            self._graph.clear()

        # Clear SQLite tables if DB exists
        db_path = self._index_path / GRAPH_FILENAME
        if db_path.exists():
            db = self._get_db()
            db.execute("DELETE FROM entities")
            db.execute("DELETE FROM relationships")
            db.execute("DELETE FROM processed_chunks")
            db.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def stats(self) -> dict:
        """Summary statistics about the graph."""
        type_counts = Counter(e.entity_type for e in self._entities.values())

        # Top entities by connection count
        connection_counts: dict[str, int] = {}
        for rel in self._relationships.values():
            connection_counts[rel.source_entity] = connection_counts.get(rel.source_entity, 0) + 1
            connection_counts[rel.target_entity] = connection_counts.get(rel.target_entity, 0) + 1

        top_entities = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_entity_names = []
        for key, count in top_entities:
            entity = self._entities.get(key)
            if entity:
                top_entity_names.append({"name": entity.name, "type": entity.entity_type, "connections": count})

        total_chunks = set()
        for e in self._entities.values():
            total_chunks.update(e.source_chunk_ids)
        for r in self._relationships.values():
            total_chunks.update(r.source_chunk_ids)

        return {
            "entity_count": len(self._entities),
            "relationship_count": len(self._relationships),
            "entity_type_distribution": dict(type_counts),
            "top_entities": top_entity_names,
            "source_chunk_count": len(total_chunks),
            "processed_chunk_count": len(self._processed_chunk_ids),
        }
