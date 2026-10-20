"""
Graph-augmented search modes.

Provides local, global, hybrid, and mix search strategies
that leverage the knowledge graph for multi-hop reasoning.
Inspired by LightRAG query modes.
"""

import logging
from typing import TYPE_CHECKING

from core.graph.store import KnowledgeGraph

if TYPE_CHECKING:
    from core.search import HybridSearcher

logger = logging.getLogger(__name__)


def _fetch_chunks_by_ids(
    searcher: "HybridSearcher",
    chunk_ids: list[str],
    tag_metadata: dict | None = None,
) -> list[dict]:
    """Fetch chunks from ChromaDB by their IDs."""
    if not chunk_ids:
        return []

    try:
        # Batch fetch from ChromaDB
        results = searcher.collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        logger.warning(f"Failed to fetch chunks by ID: {e}")
        return []

    chunks = []
    for i in range(len(results["ids"])):
        chunk = {
            "id": results["ids"][i],
            "content": results["documents"][i],
            "metadata": results["metadatas"][i] if results["metadatas"] else {},
        }
        if tag_metadata:
            chunk.update(tag_metadata)
        chunks.append(chunk)

    return chunks


def _rrf_merge_lists(
    *result_lists: list[dict],
    k: int = 60,
    top_k: int = 20,
) -> list[dict]:
    """Merge multiple result lists using Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            chunk_id = chunk.get("id", str(rank))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            chunks_by_id[chunk_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    merged = []
    for cid in sorted_ids[:top_k]:
        chunk = chunks_by_id[cid]
        chunk["_graph_rrf_score"] = rrf_scores[cid]
        merged.append(chunk)
    return merged


def local_search(
    query: str,
    graph: KnowledgeGraph,
    searcher: "HybridSearcher",
    top_k: int = 20,
) -> list[dict]:
    """
    Entity-focused search: find matching entities, retrieve their source chunks.

    1. Find entities by name substring/alias match
    2. Collect source_chunk_ids from matched entities
    3. Fetch those chunks from ChromaDB by ID
    4. Tag results with _graph_entity metadata
    """
    entities = graph.find_entities_by_name(query)
    if not entities:
        logger.info(f"Graph local search: no entities match '{query[:50]}'")
        return []

    logger.info(f"Graph local search: found {len(entities)} matching entities")

    # Collect chunk IDs from matching entities
    entity_keys = [e.key for e in entities]
    chunk_ids = graph.get_source_chunks(entity_keys)

    if not chunk_ids:
        return []

    # Fetch actual chunks
    chunks = _fetch_chunks_by_ids(
        searcher,
        chunk_ids[:top_k * 2],  # Fetch extra, trim later
        tag_metadata={"_graph_entity": True},
    )

    # Tag with entity info
    entity_names = [e.name for e in entities[:5]]
    for chunk in chunks:
        chunk["_graph_matched_entities"] = entity_names

    return chunks[:top_k]


def global_search(
    query: str,
    graph: KnowledgeGraph,
    searcher: "HybridSearcher",
    top_k: int = 20,
    max_hops: int = 2,
) -> list[dict]:
    """
    Relationship-focused search: find entities, traverse relationships, retrieve connected chunks.

    1. Find seed entities matching query
    2. BFS traversal up to max_hops
    3. Collect chunks from all connected entities/relationships
    4. Tag results with _graph_path metadata
    """
    seed_entities = graph.find_entities_by_name(query)
    if not seed_entities:
        logger.info(f"Graph global search: no seed entities for '{query[:50]}'")
        return []

    logger.info(f"Graph global search: {len(seed_entities)} seed entities, max_hops={max_hops}")

    # Traverse from each seed
    all_entity_keys = set()
    for seed in seed_entities[:5]:  # Limit seeds to avoid explosion
        all_entity_keys.add(seed.key)
        neighbors = graph.get_neighbors(seed.key, max_depth=max_hops)
        for entity, rel in neighbors:
            all_entity_keys.add(entity.key)

    # Get chunks for all discovered entities
    chunk_ids = graph.get_source_chunks(list(all_entity_keys))

    if not chunk_ids:
        return []

    chunks = _fetch_chunks_by_ids(
        searcher,
        chunk_ids[:top_k * 2],
        tag_metadata={"_graph_path": True},
    )

    return chunks[:top_k]


def hybrid_graph_search(
    query: str,
    graph: KnowledgeGraph,
    searcher: "HybridSearcher",
    top_k: int = 20,
) -> list[dict]:
    """Local + global merged with deduplication."""
    local_results = local_search(query, graph, searcher, top_k=top_k)
    global_results = global_search(query, graph, searcher, top_k=top_k)

    # Deduplicate by chunk ID
    seen_ids = set()
    merged = []
    for chunk in local_results + global_results:
        cid = chunk.get("id")
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            merged.append(chunk)

    return merged[:top_k]


def mix_search(
    query: str,
    graph: KnowledgeGraph,
    searcher: "HybridSearcher",
    top_k: int = 20,
) -> list[dict]:
    """
    Knowledge graph + standard hybrid search merged via RRF.

    1. Run standard hybrid search (BM25 + vector)
    2. Run hybrid_graph_search
    3. Merge both via RRF
    """
    # Standard hybrid search
    standard_results = searcher.search(query=query, top_k=top_k)

    # Graph hybrid search
    graph_results = hybrid_graph_search(query, graph, searcher, top_k=top_k)

    if not graph_results:
        return standard_results

    # Merge via RRF
    return _rrf_merge_lists(standard_results, graph_results, top_k=top_k)
