#!/usr/bin/env python3
"""
CLI for building knowledge graphs from indexed document chunks.

Usage:
    python scripts/build_graph.py --client myproject            # Build graph
    python scripts/build_graph.py --client myproject --rebuild   # Clear + rebuild
    python scripts/build_graph.py --client myproject --dry-run   # Show chunk count
    python scripts/build_graph.py --client myproject --sample 50 # Extract from 50 random chunks
    python scripts/build_graph.py --client myproject --workers 8 # Use 8 concurrent LLM workers
    python scripts/build_graph.py --all                        # All graph-enabled clients
"""

import argparse
import logging
import random
import sys
import threading
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.client import load_client, discover_clients
from core.graph import KnowledgeGraph
from core.graph.extractor import GraphExtractor
from core.graph.progress import GraphBuildProgress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_llm_callable(client):
    """Create an LLM response callable from client config."""
    from core.llm import create_provider_from_config

    provider = create_provider_from_config(client.llm_config)

    def get_response(messages, system_prompt, max_tokens):
        return provider.generate(messages, system_prompt=system_prompt, max_tokens=max_tokens)

    return get_response


def _should_exclude_by_extension(metadata: dict, exclude_extensions: list[str]) -> bool:
    """Check if a chunk should be excluded based on its source file extension."""
    if not exclude_extensions:
        return False
    # Check source or source_path metadata for file extension
    source = metadata.get("source_path") or metadata.get("source") or ""
    if not source:
        return False
    source_lower = source.lower()
    return any(source_lower.endswith(ext.lower()) for ext in exclude_extensions)


def _get_collection(client):
    """Get ChromaDB collection for a client."""
    import chromadb
    from chromadb.config import Settings

    index_path = str(client.index_path)
    chroma_client = chromadb.PersistentClient(
        path=index_path,
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        return chroma_client.get_collection(name=client.machine_name)
    except Exception:
        logger.error(f"No collection found for {client.machine_name}")
        return None


def get_all_chunks(client):
    """Load all document chunks from ChromaDB."""
    collection = _get_collection(client)
    if collection is None:
        return []

    count = collection.count()
    if count == 0:
        return []

    # Fetch in batches (ChromaDB has limits)
    all_chunks = []
    batch_size = 5000
    for offset in range(0, count, batch_size):
        results = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        for i in range(len(results["ids"])):
            all_chunks.append({
                "id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
            })

    return all_chunks


def get_unprocessed_chunks(client, processed_ids: set[str], exclude_extensions: list[str] = None):
    """Load only unprocessed document chunks from ChromaDB.

    Fetches IDs first (cheap), filters against processed set, then fetches
    full content only for unprocessed chunks. For large clients with few
    changes, this avoids loading hundreds of thousands of chunks.
    """
    exclude_extensions = exclude_extensions or []
    collection = _get_collection(client)
    if collection is None:
        return [], 0

    count = collection.count()
    if count == 0:
        return [], 0

    # Phase 1: Fetch only IDs + metadatas (no document content)
    unprocessed_doc_ids = []
    batch_size = 10000
    total_doc_chunks = 0
    skipped_by_ext = 0

    for offset in range(0, count, batch_size):
        results = collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset,
        )
        for i, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            # Only document chunks (skip code chunks)
            if metadata.get("chunk_type", "document") != "document":
                continue
            # Skip low-value file types for graph extraction
            if _should_exclude_by_extension(metadata, exclude_extensions):
                skipped_by_ext += 1
                continue
            total_doc_chunks += 1
            if chunk_id not in processed_ids:
                unprocessed_doc_ids.append(chunk_id)

    if skipped_by_ext > 0:
        logger.info(f"Skipped {skipped_by_ext} chunks by extension filter")

    if not unprocessed_doc_ids:
        return [], total_doc_chunks

    # Phase 2: Fetch full content only for unprocessed chunks
    logger.info(f"Fetching content for {len(unprocessed_doc_ids)} unprocessed chunks "
                f"(skipped {total_doc_chunks - len(unprocessed_doc_ids)} already processed)")
    chunks = []
    # ChromaDB get-by-IDs in batches
    for i in range(0, len(unprocessed_doc_ids), batch_size):
        batch_ids = unprocessed_doc_ids[i:i + batch_size]
        results = collection.get(
            ids=batch_ids,
            include=["documents", "metadatas"],
        )
        for j in range(len(results["ids"])):
            chunks.append({
                "id": results["ids"][j],
                "content": results["documents"][j],
                "metadata": results["metadatas"][j] if results["metadatas"] else {},
            })

    return chunks, total_doc_chunks


def build_graph_for_client(client, args):
    """Build knowledge graph for a single client."""
    graph_config = client.graph_config
    client_name = client.machine_name

    if not graph_config.get("enabled") and not args.force:
        logger.info(f"Skipping {client_name} (graph not enabled, use --force to override)")
        return

    logger.info(f"Building graph for: {client_name}")

    # Load existing graph
    graph = KnowledgeGraph(client.index_path)
    if args.rebuild:
        graph.clear()
        logger.info("Cleared existing graph")
    else:
        graph.load()
        logger.info(f"Loaded existing graph: {graph.stats['entity_count']} entities")

    # Extension filter from graph config
    exclude_extensions = graph_config.get("exclude_extensions", [])
    if exclude_extensions:
        logger.info(f"Graph extension filter: excluding {exclude_extensions}")

    # Get chunks — use fast incremental path when possible
    if args.rebuild:
        # Full rebuild: fetch everything
        chunks = get_all_chunks(client)
        if not chunks:
            logger.warning(f"No chunks found for {client_name}")
            return
        doc_chunks = [
            c for c in chunks
            if c.get("metadata", {}).get("chunk_type", "document") == "document"
            and not _should_exclude_by_extension(c.get("metadata", {}), exclude_extensions)
        ]
        logger.info(f"Found {len(doc_chunks)} document chunks (of {len(chunks)} total)")
    else:
        # Incremental: fetch only IDs first, then content for unprocessed only
        already_processed = graph.processed_chunk_ids
        doc_chunks, total_doc = get_unprocessed_chunks(client, already_processed, exclude_extensions)
        if total_doc == 0:
            logger.warning(f"No chunks found for {client_name}")
            return
        logger.info(f"Incremental: {len(doc_chunks)} new chunks "
                     f"(skipped {total_doc - len(doc_chunks)} already processed)")

    if not doc_chunks:
        logger.info("No new chunks to process")
        print(f"\n{client_name}: Graph already up to date")
        stats = graph.stats
        graph.close()
        print(f"  Entities: {stats['entity_count']}, Relationships: {stats['relationship_count']}")
        return

    # Sample if requested
    if args.sample and args.sample < len(doc_chunks):
        doc_chunks = random.sample(doc_chunks, args.sample)
        logger.info(f"Sampled {args.sample} chunks")

    # Resolve workers and save_every (CLI overrides > config > defaults)
    workers = args.workers or graph_config.get("extract_workers", 4)
    save_every = graph_config.get("extract_save_every", 5)
    from core.graph.extractor import DEFAULT_BATCH_TEXT_BUDGET
    batch_text_budget = graph_config.get("extract_text_budget", DEFAULT_BATCH_TEXT_BUDGET)

    if args.dry_run:
        from core.graph.extractor import _build_batches_by_budget
        batches = _build_batches_by_budget(doc_chunks, batch_text_budget)
        print(f"\n{client_name}: Would process {len(doc_chunks)} chunks")
        print(f"  Text budget: {batch_text_budget:,} chars/batch (adaptive, halves on failure)")
        print(f"  Estimated batches: {len(batches)} (~{len(doc_chunks) // max(len(batches), 1)} chunks/batch)")
        print(f"  Workers: {workers}")
        print(f"  Entity types: {graph_config.get('entity_types', [])}")
        print(f"  Save every: {save_every} batches")
        return

    # Health check — verify LM Studio has a model loaded before starting
    from core.health import check_lm_studio_model
    lm_base_url = client.llm_config.get("base_url", "http://localhost:1234/v1")
    if not check_lm_studio_model(base_url=lm_base_url):
        logger.error(
            f"LM Studio health check failed at {lm_base_url} — "
            f"no model loaded or service down. Skipping graph build for {client_name}."
        )
        return

    # Create LLM callable
    try:
        get_llm_response = create_llm_callable(client)
    except Exception as e:
        logger.error(f"Failed to create LLM provider: {e}")
        return

    # Create extractor with concurrency settings
    extractor = GraphExtractor(
        get_llm_response=get_llm_response,
        entity_types=graph_config.get("entity_types", ["person", "organization", "concept", "location"]),
        max_description_chars=graph_config.get("max_description_chars", 200),
        max_workers=workers,
        save_every=save_every,
        batch_text_budget=batch_text_budget,
    )

    # Thread-safe progress callback
    start_time = time.time()
    graph_lock = threading.Lock()
    _last_entity_count = [0]
    _last_rel_count = [0]

    def on_progress(p: GraphBuildProgress, entities, relationships):
        """Add new entities/rels to graph and save periodically.

        Called by the extractor under its internal lock, so entity/relationship
        lists are stable during this call. We add a graph_lock for graph mutations.
        """
        with graph_lock:
            # Add only the new entities/relationships since last save
            new_entities = entities[_last_entity_count[0]:]
            new_rels = relationships[_last_rel_count[0]:]
            _last_entity_count[0] = len(entities)
            _last_rel_count[0] = len(relationships)

            for entity in new_entities:
                graph.add_entity(entity)
            for rel in new_rels:
                graph.add_relationship(rel)

            # Mark all processed chunks so far
            # With concurrent processing, chunk order isn't sequential,
            # so we track by count of processed chunks
            processed_ids = [c["id"] for c in doc_chunks[:p.processed_chunks]]
            graph.mark_chunks_processed(processed_ids)

            graph.save()

        elapsed = time.time() - start_time
        print(
            f"\r  {p.percent:.0f}% ({p.processed_chunks}/{p.total_chunks} chunks) "
            f"| Entities: {p.entities_found} | Rels: {p.relationships_found} "
            f"| Failed: {p.failed_chunks} | ETA: {p.format_eta()} | {elapsed:.0f}s elapsed"
            f" | {workers}w",
            end="", flush=True,
        )

    # Extract with concurrent processing
    print(f"\nExtracting from {len(doc_chunks)} chunks ({workers} workers, budget={batch_text_budget:,} chars)...")
    entities, relationships = extractor.extract_from_chunks(doc_chunks, progress_callback=on_progress)
    print()  # newline after progress

    elapsed = time.time() - start_time
    logger.info(f"Extraction complete: {len(entities)} entities, {len(relationships)} relationships in {elapsed:.1f}s")

    # Final save — add any entities/rels not yet in graph
    with graph_lock:
        new_entities = entities[_last_entity_count[0]:]
        new_rels = relationships[_last_rel_count[0]:]
        for entity in new_entities:
            graph.add_entity(entity)
        for rel in new_rels:
            graph.add_relationship(rel)
        graph.mark_chunks_processed([c["id"] for c in doc_chunks])
        graph.save()

    stats = graph.stats
    graph.close()
    print(f"\n{client_name}: Graph saved")
    print(f"  Entities: {stats['entity_count']}")
    print(f"  Relationships: {stats['relationship_count']}")
    print(f"  Entity types: {stats['entity_type_distribution']}")
    print(f"  Source chunks: {stats['source_chunk_count']}")
    if stats['top_entities']:
        print(f"  Top entities:")
        for te in stats['top_entities'][:5]:
            print(f"    - {te['name']} ({te['type']}): {te['connections']} connections")


def main():
    parser = argparse.ArgumentParser(description="Build knowledge graphs from indexed chunks")
    parser.add_argument("--client", help="Client name to build graph for")
    parser.add_argument("--all", action="store_true", help="Build for all graph-enabled clients")
    parser.add_argument("--rebuild", action="store_true", help="Clear existing graph and rebuild")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--sample", type=int, help="Process only N random chunks")
    parser.add_argument("--workers", type=int, help="Number of concurrent LLM workers (default: from config or 4)")
    parser.add_argument("--force", action="store_true", help="Build even if graph.enabled is false")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.client and not args.all:
        parser.error("Specify --client NAME or --all")

    if args.all:
        clients = discover_clients()
        for name in clients:
            try:
                client = load_client(name)
                build_graph_for_client(client, args)
            except Exception as e:
                logger.error(f"Failed for {name}: {e}")
    else:
        client = load_client(args.client)
        build_graph_for_client(client, args)


if __name__ == "__main__":
    main()
