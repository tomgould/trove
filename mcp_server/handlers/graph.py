"""Handler functions for knowledge graph tools."""

import logging
import threading

from mcp.types import TextContent

from core.client import load_client

from mcp_server.shared import (
    CLIENTS_DIR,
    _create_llm_provider,
    _job_manager,
    get_current_client,
    get_current_searcher,
    load_trove_client,
)

logger = logging.getLogger("trove-mcp")


def handle_graph_stats(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")

    if client_name:
        try:
            client, searcher = load_trove_client(client_name)
        except Exception as e:
            return [TextContent(type="text", text=f"Error loading client: {e}")]
    else:
        searcher = get_current_searcher()
        client = get_current_client()
        if not searcher or not client:
            return [TextContent(type="text", text="No client loaded. Use trove_load_client first.")]

    graph = searcher.load_graph()
    if not graph:
        return [TextContent(type="text", text=f"No knowledge graph found for **{client.machine_name}**. Use trove_build_graph first.")]

    from core.graph.export import graph_stats
    stats = graph_stats(graph)

    lines = [f"Knowledge Graph Stats for **{client.machine_name}**:\n"]
    lines.append(f"**Entities:** {stats['entity_count']}")
    lines.append(f"**Relationships:** {stats['relationship_count']}")
    lines.append(f"**Source Chunks:** {stats['source_chunk_count']}")
    lines.append(f"**Density:** {stats['density']:.6f}")

    if stats.get("entity_type_distribution"):
        lines.append("\n**Entity Types:**")
        for etype, count in sorted(stats["entity_type_distribution"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {etype}: {count}")

    if stats.get("top_entities"):
        lines.append("\n**Most Connected Entities:**")
        for te in stats["top_entities"]:
            lines.append(f"  - **{te['name']}** ({te['type']}): {te['connections']} connections")

    return [TextContent(type="text", text="\n".join(lines))]


def handle_graph_export(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    max_nodes = arguments.get("max_nodes", 500)

    if client_name:
        try:
            client, searcher = load_trove_client(client_name)
        except Exception as e:
            return [TextContent(type="text", text=f"Error loading client: {e}")]
    else:
        searcher = get_current_searcher()
        client = get_current_client()
        if not searcher or not client:
            return [TextContent(type="text", text="No client loaded. Use trove_load_client first.")]

    graph = searcher.load_graph()
    if not graph:
        return [TextContent(type="text", text=f"No knowledge graph found for **{client.machine_name}**. Use trove_build_graph first.")]

    import json as _json
    from core.graph.export import export_for_d3
    d3_data = export_for_d3(graph, max_nodes=max_nodes)

    return [TextContent(
        type="text",
        text=_json.dumps(d3_data, indent=2),
    )]


def handle_search_entities(arguments: dict) -> list[TextContent]:
    query = arguments.get("query")
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    entity_type = arguments.get("entity_type")
    top_k = min(arguments.get("top_k", 20), 100)
    client_name = arguments.get("client_name")

    # Resolve searcher
    if client_name:
        try:
            client, searcher = load_trove_client(client_name)
        except Exception as e:
            return [TextContent(type="text", text=f"Error loading client: {e}")]
    else:
        searcher = get_current_searcher()
        client = get_current_client()
        if not searcher or not client:
            return [TextContent(type="text", text="No client loaded. Use trove_load_client first.")]

    graph = searcher.load_graph()
    if not graph:
        return [TextContent(type="text", text=f"No knowledge graph found for **{client.machine_name}**. Use trove_build_graph first.")]

    entities = graph.find_entities_by_name(query)
    if entity_type:
        entities = [e for e in entities if e.entity_type == entity_type.lower().strip()]

    entities = entities[:top_k]

    if not entities:
        return [TextContent(type="text", text=f"No entities matching '{query}' in **{client.machine_name}** graph.")]

    lines = [f"Found {len(entities)} entities matching '{query}':\n"]
    for e in entities:
        aliases_str = f" (aliases: {', '.join(e.aliases)})" if e.aliases else ""
        chunks_str = f" [{len(e.source_chunk_ids)} chunks]"
        lines.append(f"- **{e.name}** ({e.entity_type}){aliases_str}{chunks_str}")
        if e.description:
            lines.append(f"  {e.description}")

    return [TextContent(type="text", text="\n".join(lines))]


def handle_search_relationships(arguments: dict) -> list[TextContent]:
    entity_name = arguments.get("entity_name")
    if not entity_name:
        return [TextContent(type="text", text="Error: entity_name is required")]

    relationship_type = arguments.get("relationship_type")
    max_hops = min(arguments.get("max_hops", 2), 4)
    client_name = arguments.get("client_name")

    if client_name:
        try:
            client, searcher = load_trove_client(client_name)
        except Exception as e:
            return [TextContent(type="text", text=f"Error loading client: {e}")]
    else:
        searcher = get_current_searcher()
        client = get_current_client()
        if not searcher or not client:
            return [TextContent(type="text", text="No client loaded. Use trove_load_client first.")]

    graph = searcher.load_graph()
    if not graph:
        return [TextContent(type="text", text=f"No knowledge graph found for **{client.machine_name}**. Use trove_build_graph first.")]

    # Find the entity
    entities = graph.find_entities_by_name(entity_name)
    if not entities:
        return [TextContent(type="text", text=f"No entity matching '{entity_name}' found.")]

    seed = entities[0]
    neighbors = graph.get_neighbors(seed.key, max_depth=max_hops)

    if relationship_type:
        rt = relationship_type.lower().strip()
        neighbors = [(e, r) for e, r in neighbors if r.relationship_type.lower().strip() == rt]

    if not neighbors:
        return [TextContent(type="text", text=f"No relationships found for entity '{seed.name}' (hops={max_hops}).")]

    lines = [f"Relationships from **{seed.name}** ({seed.entity_type}, depth={max_hops}):\n"]
    for entity, rel in neighbors:
        direction = "outgoing" if rel.source_entity == seed.key else "incoming"
        lines.append(f"- **{entity.name}** ({entity.entity_type}) [{rel.relationship_type}] ({direction})")
        if rel.description:
            lines.append(f"  {rel.description}")
        lines.append(f"  weight: {rel.weight:.2f}, chunks: {len(rel.source_chunk_ids)}")

    return [TextContent(type="text", text="\n".join(lines))]


def handle_build_graph(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    if not client_name:
        return [TextContent(type="text", text="Error: client_name is required")]

    rebuild = arguments.get("rebuild", False)
    sample = arguments.get("sample")
    workers_override = arguments.get("workers")

    try:
        job = _job_manager.create_job(client_name, "graph_build")
    except RuntimeError as e:
        return [TextContent(type="text", text=f"Error: {e}")]

    def _run_graph_build(job_id: str, cname: str):
        try:
            _job_manager.start_job(job_id)
            import random as _random
            import threading as _threading
            from core.graph import KnowledgeGraph
            from core.graph.extractor import GraphExtractor

            client = load_client(cname, clients_dir=CLIENTS_DIR)
            graph_config = client.graph_config

            # Load graph
            graph = KnowledgeGraph(client.index_path)
            if rebuild:
                graph.clear()
            else:
                graph.load()

            # Get document chunks from ChromaDB
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            chroma = chromadb.PersistentClient(
                path=str(client.index_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            try:
                collection = chroma.get_collection(name=client.machine_name)
            except Exception:
                _job_manager.fail_job(job_id, "No collection found")
                return

            count = collection.count()
            if count == 0:
                _job_manager.complete_job(job_id, "No chunks to process")
                return

            # Fetch chunks — incremental: IDs first, then content for unprocessed only
            processed_ids = set() if rebuild else graph.processed_chunk_ids
            exclude_exts = graph_config.get("exclude_extensions", [])
            unprocessed_ids = []

            for offset in range(0, count, 10000):
                results = collection.get(include=["metadatas"], limit=10000, offset=offset)
                for i, chunk_id in enumerate(results["ids"]):
                    md = results["metadatas"][i] if results["metadatas"] else {}
                    if md.get("chunk_type", "document") != "document":
                        continue
                    # Skip low-value file types for graph extraction
                    if exclude_exts:
                        src = (md.get("source_path") or md.get("source") or "").lower()
                        if any(src.endswith(ext.lower()) for ext in exclude_exts):
                            continue
                    if chunk_id not in processed_ids:
                        unprocessed_ids.append(chunk_id)

            # Fetch full content only for unprocessed chunks
            all_chunks = []
            for i in range(0, len(unprocessed_ids), 5000):
                batch_ids = unprocessed_ids[i:i + 5000]
                results = collection.get(ids=batch_ids, include=["documents", "metadatas"])
                for j in range(len(results["ids"])):
                    all_chunks.append({
                        "id": results["ids"][j],
                        "content": results["documents"][j],
                        "metadata": results["metadatas"][j] if results["metadatas"] else {},
                    })

            if sample and sample < len(all_chunks):
                all_chunks = _random.sample(all_chunks, sample)

            if not all_chunks:
                _job_manager.complete_job(
                    job_id,
                    f"Graph up to date ({graph.stats['entity_count']} entities, "
                    f"{graph.stats['relationship_count']} relationships)"
                )
                return

            # Create LLM provider
            provider = _create_llm_provider(client)
            if not provider:
                _job_manager.fail_job(job_id, "No LLM provider available")
                return

            def get_llm_response(messages, system_prompt, max_tokens):
                return provider.generate(messages, system_prompt=system_prompt, max_tokens=max_tokens)

            num_workers = workers_override or graph_config.get("extract_workers", 4)
            save_every = graph_config.get("extract_save_every", 5)

            extractor = GraphExtractor(
                get_llm_response=get_llm_response,
                entity_types=graph_config.get("entity_types", ["person", "organization", "concept", "location"]),
                max_description_chars=graph_config.get("max_description_chars", 200),
                max_workers=num_workers,
                save_every=save_every,
                batch_text_budget=graph_config.get("extract_text_budget", 6000),
            )

            # Thread-safe callback for periodic graph saves
            _graph_lock = _threading.Lock()
            _last_entity_count = [0]
            _last_rel_count = [0]

            def _on_progress(progress, entities, relationships):
                with _graph_lock:
                    new_entities = entities[_last_entity_count[0]:]
                    new_rels = relationships[_last_rel_count[0]:]
                    _last_entity_count[0] = len(entities)
                    _last_rel_count[0] = len(relationships)

                    for entity in new_entities:
                        graph.add_entity(entity)
                    for rel in new_rels:
                        graph.add_relationship(rel)

                    graph.mark_chunks_processed([c["id"] for c in all_chunks[:progress.processed_chunks]])
                    graph.save()

            entities, relationships = extractor.extract_from_chunks(
                all_chunks, progress_callback=_on_progress
            )

            # Final save
            with _graph_lock:
                new_entities = entities[_last_entity_count[0]:]
                new_rels = relationships[_last_rel_count[0]:]
                for entity in new_entities:
                    graph.add_entity(entity)
                for rel in new_rels:
                    graph.add_relationship(rel)
                graph.mark_chunks_processed([c["id"] for c in all_chunks])
                graph.save()

            stats = graph.stats
            graph.close()
            _job_manager.complete_job(
                job_id,
                f"Graph built: {stats['entity_count']} entities, "
                f"{stats['relationship_count']} relationships from {len(all_chunks)} chunks"
            )

        except Exception as e:
            logger.exception(f"Graph build job {job_id} failed")
            _job_manager.fail_job(job_id, str(e))

    t = threading.Thread(target=_run_graph_build, args=(job.job_id, client_name), daemon=True)
    t.start()

    return [TextContent(
        type="text",
        text=(
            f"Graph build started for **{client_name}**"
            f"{' (rebuild)' if rebuild else ' (incremental)'}"
            f"{f' (sample={sample})' if sample else ''}. "
            f"Job ID: `{job.job_id}`\n\n"
            f"Use `trove_ingest_status` with job_id to check progress."
        ),
    )]
