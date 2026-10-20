"""
Graph statistics and export utilities.
"""

from core.graph.store import KnowledgeGraph


def graph_stats(graph: KnowledgeGraph) -> dict:
    """
    Detailed statistics about the knowledge graph.

    Returns entity/relationship counts, type distributions,
    top entities by connection count, and graph density.
    """
    stats = graph.stats

    # Calculate density if we have a NetworkX graph
    density = 0.0
    n = stats["entity_count"]
    if n > 1:
        m = stats["relationship_count"]
        # Directed graph density: m / (n * (n - 1))
        density = m / (n * (n - 1))

    stats["density"] = round(density, 6)
    return stats


def export_for_d3(graph: KnowledgeGraph, max_nodes: int = 500) -> dict:
    """
    Export graph as D3.js-compatible JSON.

    Returns:
        {"nodes": [{"id": key, "name": ..., "type": ..., "group": ...}],
         "links": [{"source": ..., "target": ..., "type": ..., "weight": ...}]}
    """
    # Assign group numbers to entity types for coloring
    type_groups: dict[str, int] = {}
    group_counter = 0

    nodes = []
    for key, entity in list(graph._entities.items())[:max_nodes]:
        et = entity.entity_type
        if et not in type_groups:
            type_groups[et] = group_counter
            group_counter += 1

        nodes.append({
            "id": key,
            "name": entity.name,
            "type": entity.entity_type,
            "group": type_groups[et],
            "description": entity.description,
            "chunk_count": len(entity.source_chunk_ids),
        })

    # Only include edges between nodes that are in our node set
    node_keys = {n["id"] for n in nodes}
    links = []
    for rel in graph._relationships.values():
        if rel.source_entity in node_keys and rel.target_entity in node_keys:
            links.append({
                "source": rel.source_entity,
                "target": rel.target_entity,
                "type": rel.relationship_type,
                "weight": rel.weight,
                "description": rel.description,
            })

    return {
        "nodes": nodes,
        "links": links,
        "type_groups": type_groups,
    }
