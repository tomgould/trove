#!/usr/bin/env python3
"""
CLI script for document and code ingestion.

Usage:
    python scripts/ingest.py --client myproject                    # Incremental (default)
    python scripts/ingest.py --client myproject --rebuild           # Full rebuild
    python scripts/ingest.py --client myproject --incremental       # Explicit incremental
    python scripts/ingest.py --client myproject --path /path/to/docs/
    python scripts/ingest.py --client myproject --code-only
    python scripts/ingest.py --client myproject --rebuild-code
"""

import argparse
import logging
import os
import json
import sys
from pathlib import Path

# Prevent tokenizers fork warnings when using multiprocessing
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.client import add_clients_dir_argument, apply_clients_dir_argument
from core.client import load_client, discover_clients
from core.ingest import (
    DocumentIngestor,
    ingest_directory,
    ingest_client_code,
    build_file_manifest,
    load_file_manifest,
    diff_file_manifest,
    save_file_manifest,
)
from core.search import HybridSearcher
from core.search.chroma_paging import get_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def sync_client_sources(client_name: str) -> dict:
    """
    Sync all configured sources for a client.

    Runs GitHub fetch and web crawl for any configured sources.

    Returns:
        Dict with sync results summary.
    """
    client = load_client(client_name)
    sources_config = client.sources_config
    if not sources_config:
        logger.info(f"No sources configured for {client.name}")
        return {"synced": False}

    sources_dir = str(client.sources_path)
    results = []

    # GitHub sources
    for gh_cfg in sources_config.get("github", []):
        try:
            from core.sources.github_models import GitHubSourceConfig
            from core.sources.github_connector import GitHubSourceConnector
            config = GitHubSourceConfig.from_dict(gh_cfg)
            connector = GitHubSourceConnector(config)
            result = connector.sync(sources_dir)
            results.append(result)
            logger.info(
                "GitHub sync (%s): +%d updated=%d errors=%d",
                config.owner, result.files_added, result.files_updated, len(result.errors),
            )
            for err in result.errors:
                logger.warning("  %s", err)
        except Exception as e:
            logger.error("GitHub source sync failed: %s", e)

    # Web sources
    for web_cfg in sources_config.get("web", []):
        try:
            from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
            config = WebSourceConfig.from_dict(web_cfg)
            connector = WebSourceConnector(config)
            result = connector.sync(sources_dir)
            results.append(result)
            logger.info(
                "Web sync (%s): +%d updated=%d errors=%d",
                config.domain, result.files_added, result.files_updated, len(result.errors),
            )
            for err in result.errors:
                logger.warning("  %s", err)
        except Exception as e:
            logger.error("Web source sync failed: %s", e)

    total_added = sum(r.files_added for r in results)
    total_updated = sum(r.files_updated for r in results)
    total_errors = sum(len(r.errors) for r in results)
    logger.info(
        "Source sync complete: +%d updated=%d errors=%d",
        total_added, total_updated, total_errors,
    )

    return {
        "synced": True,
        "files_added": total_added,
        "files_updated": total_updated,
        "errors": total_errors,
    }


def _summary(client_name: str, mode: str, **counts) -> dict:
    """Ingest summary for --json: file counts a scrape pipeline can act on."""
    summary = {
        "client": client_name,
        "mode": mode,
        "new": 0,
        "changed": 0,
        "unchanged": 0,
        "deleted": 0,
        "failed": 0,
        "failed_files": [],
        "chunks_added": 0,
        "chunks_removed": 0,
    }
    summary.update(counts)
    summary["failed"] = len(summary["failed_files"])
    return summary


def _drop_failed(manifest: dict, failed_files: list[str]) -> dict:
    """Leave failed files out of the manifest so the next run retries them."""
    failed = set(failed_files)
    return {path: entry for path, entry in manifest.items() if path not in failed}


def _should_sync_on_ingest(client) -> bool:
    """Check if any configured source has sync_on_ingest enabled."""
    sources_config = client.sources_config
    if not sources_config:
        return False
    for gh_cfg in sources_config.get("github", []):
        if gh_cfg.get("sync_on_ingest", False):
            return True
    for web_cfg in sources_config.get("web", []):
        if web_cfg.get("sync_on_ingest", False):
            return True
    return False


def get_chunk_ids_for_file(searcher: HybridSearcher, file_path: str) -> list[str]:
    """Get existing chunk IDs for a file from ChromaDB by source_path metadata."""
    try:
        results = searcher.collection.get(
            where={"source_path": file_path},
            include=[],
        )
        return results.get("ids", [])
    except Exception:
        return []


def incremental_reindex(
    client_name: str,
    include_code: bool = True,
) -> dict:
    """
    Incrementally reindex only changed files for a client.

    Compares current files on disk against the saved file manifest to detect
    new, modified, and deleted files. Only reindexes what changed.

    Falls back to full rebuild if no manifest exists or embedding model changed.
    Files rewritten with the same content count as unchanged (the manifest keeps
    a content hash), so a scraper can rewrite its output on every run.

    Args:
        client_name: Name of the client
        include_code: Also run incremental code indexing if configured

    Returns:
        Summary dict (see _summary).
    """
    client = load_client(client_name)
    logger.info(f"Incremental reindex for: {client.name}")

    docs_path = client.documents_path
    if not docs_path.exists():
        logger.error(f"Documents directory not found: {docs_path}")
        return _summary(client_name, "incremental", error=f"Documents directory not found: {docs_path}")

    # Get embedding config
    embedding_config = client.embedding_config
    embedding_model = embedding_config.get("model", "all-MiniLM-L6-v2")
    passage_prefix = embedding_config.get("passage_prefix", "")

    # Get document extensions and exclude patterns
    doc_extensions = client.get_document_extensions()
    doc_exclude_patterns = client.documents_config.get("exclude_patterns", [])

    # Sync sources if configured with sync_on_ingest
    if _should_sync_on_ingest(client):
        logger.info("Running source sync before incremental reindex...")
        sync_client_sources(client_name)

    # Build current manifest from disk (documents/ + sources/)
    current_files = build_file_manifest(docs_path, doc_extensions, exclude_patterns=doc_exclude_patterns)
    sources_path = client.sources_path
    if sources_path.exists():
        sources_manifest = build_file_manifest(sources_path, doc_extensions, exclude_patterns=doc_exclude_patterns)
        current_files.update(sources_manifest)

    # Load saved manifest from last ingestion
    indexed_files = load_file_manifest(client)

    if not indexed_files:
        logger.info("No file manifest found — falling back to full rebuild")
        return rebuild_index(client_name, include_code=include_code)

    # Diff manifests
    new_files, modified_files, deleted_files = diff_file_manifest(current_files, indexed_files)

    unchanged_count = len(current_files) - len(new_files) - len(modified_files)

    if not new_files and not modified_files and not deleted_files:
        logger.info("No document changes detected")
        # Store the new mtimes, so identical rewritten files stay cheap to compare
        save_file_manifest(client, current_files)
        if include_code:
            ingest_client_code_files(client_name, rebuild=False)
        return _summary(client_name, "incremental", unchanged=unchanged_count)

    logger.info(
        f"Changes: {len(new_files)} new, {len(modified_files)} modified, "
        f"{len(deleted_files)} deleted"
    )

    # Create searcher
    searcher = HybridSearcher(
        index_path=str(client.index_path),
        collection_name=client.machine_name,
        bm25_weight=client.search_config.get("bm25_weight", 0.5),
        vector_weight=client.search_config.get("vector_weight", 0.5),
        tokenizer_config=client.tokenization_config,
        embedding_config=embedding_config,
    )

    # Check embedding model match — if mismatch, fall back to full rebuild
    matches, reason = searcher.check_embedding_model_match()
    if not matches:
        logger.warning(f"Embedding model mismatch: {reason}")
        logger.warning("Falling back to full rebuild")
        return rebuild_index(client_name, include_code=include_code)

    # Create ingestor with client's embedding model
    ingestor = DocumentIngestor(
        markdown_config=client.markdown_config,
        embedding_model=embedding_model,
        passage_prefix=passage_prefix,
    )

    total_chunks_added = 0
    total_chunks_removed = 0

    # Remove chunks for deleted files
    for file_path in deleted_files:
        old_ids = get_chunk_ids_for_file(searcher, file_path)
        if old_ids:
            try:
                searcher.collection.delete(ids=old_ids)
                total_chunks_removed += len(old_ids)
                logger.info(f"Removed {len(old_ids)} chunks: {Path(file_path).name}")
            except Exception as e:
                logger.warning(f"Failed to delete chunks for {file_path}: {e}")

    # Remove old chunks for modified files before reindexing
    for file_path in modified_files:
        file_path_str = str(file_path)
        old_ids = get_chunk_ids_for_file(searcher, file_path_str)
        if old_ids:
            try:
                searcher.collection.delete(ids=old_ids)
                total_chunks_removed += len(old_ids)
                logger.info(f"Removed {len(old_ids)} old chunks: {Path(file_path).name}")
            except Exception as e:
                logger.warning(f"Failed to delete old chunks for {Path(file_path).name}: {e}")

    # Reindex new and modified files using pipeline
    from core.pipeline import IngestionPipeline

    files_to_reindex = [Path(f) for f in new_files + modified_files]
    failed_files: list[str] = []

    if files_to_reindex:
        pipeline = IngestionPipeline(ingestor=ingestor, searcher=searcher)
        stats = pipeline.run_documents(files_to_reindex, base_directory=docs_path)
        total_chunks_added = stats["total_chunks_indexed"]
        failed_files = stats["failed_files"]

    # Single BM25 rebuild at the end
    try:
        searcher._rebuild_bm25_index()
    except Exception as e:
        logger.warning(f"Failed to rebuild BM25 index: {e}")

    # Save updated manifest (without failed files, so they are retried next time)
    save_file_manifest(client, _drop_failed(current_files, failed_files))

    stats = searcher.get_stats()
    logger.info(
        f"Incremental reindex complete: +{total_chunks_added} -{total_chunks_removed} chunks | "
        f"Total: {stats.get('total_chunks', 0)} chunks"
    )

    # Code indexing
    if include_code:
        ingest_client_code_files(client_name, rebuild=False)

    failed = set(failed_files)
    return _summary(
        client_name, "incremental",
        new=len([f for f in new_files if f not in failed]),
        changed=len([f for f in modified_files if f not in failed]),
        unchanged=unchanged_count,
        deleted=len(deleted_files),
        failed_files=failed_files,
        chunks_added=total_chunks_added,
        chunks_removed=total_chunks_removed,
    )


def ingest_client_documents(
    client_name: str,
    documents_path: Path = None,
    force_bm25_rebuild: bool = False,
    resume: bool = False,
) -> dict:
    """
    Ingest documents for a client using the progressive pipeline.

    Creates HybridSearcher first, clears index (unless resuming), then
    ingests documents through the pipeline which embeds and indexes
    progressively in batches.

    Args:
        client_name: Name of the client
        documents_path: Optional override for documents directory
        force_bm25_rebuild: Force rebuild of BM25 index (clears cache)
        resume: Resume from last checkpoint if available
    """
    # Load client
    client = load_client(client_name)
    logger.info(f"Loaded client: {client.name}")

    # Use provided path or client's documents path
    docs_path = documents_path or client.documents_path

    if not docs_path.exists():
        logger.error(f"Documents directory not found: {docs_path}")
        return _summary(client_name, "rebuild", error=f"Documents directory not found: {docs_path}")

    # Get embedding config from client
    embedding_config = client.embedding_config
    embedding_model = embedding_config.get("model", "all-MiniLM-L6-v2")
    passage_prefix = embedding_config.get("passage_prefix", "")

    # Create ingestor with client's embedding model
    ingestor = DocumentIngestor(
        markdown_config=client.markdown_config,
        embedding_model=embedding_model,
        passage_prefix=passage_prefix,
    )

    # Get document extensions from client config
    doc_extensions = client.get_document_extensions()
    logger.info(f"Document extensions: {doc_extensions}")

    doc_exclude_patterns = client.documents_config.get("exclude_patterns", [])
    if doc_exclude_patterns:
        logger.info(f"Document exclude patterns: {doc_exclude_patterns}")

    # Create searcher FIRST (pipeline needs it for progressive indexing)
    # Skip BM25 cache load — rebuilt at end, avoids slow deserialize for large clients
    searcher = HybridSearcher(
        index_path=str(client.index_path),
        collection_name=client.machine_name,
        bm25_weight=client.search_config.get("bm25_weight", 0.5),
        vector_weight=client.search_config.get("vector_weight", 0.5),
        tokenizer_config=client.tokenization_config,
        embedding_config=embedding_config,
        skip_bm25_load=True,
    )

    # Clear existing index if not resuming
    if not resume:
        try:
            existing_count = searcher.collection.count()
            if existing_count > 0:
                searcher.chroma_client.delete_collection(client.machine_name)
                searcher.collection = searcher.chroma_client.get_or_create_collection(
                    name=client.machine_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Cleared existing index ({existing_count} chunks)")
        except Exception as e:
            logger.warning(f"Could not clear index: {e}")

    # Ingest documents with pipeline (progressive embed + index)
    logger.info(f"Ingesting documents from: {docs_path}")
    run_stats: dict = {}
    ingest_directory(
        docs_path, ingestor, extensions=doc_extensions,
        exclude_patterns=doc_exclude_patterns,
        pipeline_searcher=searcher,
        contextualize=embedding_config.get("contextualize", False),
        stats_out=run_stats,
    )
    files_found = run_stats.get("files_found", 0)
    failed_files = list(run_stats.get("failed_files", []))
    chunks_added = run_stats.get("total_chunks_indexed", 0)

    # Also ingest documents from sources/ directory if it exists
    sources_path = client.sources_path
    if sources_path.exists() and any(sources_path.iterdir()):
        logger.info(f"Also ingesting source documents from: {sources_path}")
        source_stats: dict = {}
        ingest_directory(
            sources_path, ingestor, extensions=doc_extensions,
            exclude_patterns=doc_exclude_patterns,
            pipeline_searcher=searcher,
            contextualize=embedding_config.get("contextualize", False),
            stats_out=source_stats,
        )
        files_found += source_stats.get("files_found", 0)
        failed_files += source_stats.get("failed_files", [])
        chunks_added += source_stats.get("total_chunks_indexed", 0)

    # BM25 rebuild once at end (or force rebuild if requested)
    if force_bm25_rebuild:
        logger.info("Forcing BM25 cache rebuild...")
        searcher.force_rebuild_bm25()
    else:
        try:
            searcher._rebuild_bm25_index()
        except Exception as e:
            logger.warning(f"Failed to rebuild BM25 index: {e}")

    stats = searcher.get_stats()
    logger.info(f"Indexing complete. Stats: {stats}")

    # Save file manifest for watcher (includes both documents/ and sources/)
    manifest = build_file_manifest(docs_path, doc_extensions, exclude_patterns=doc_exclude_patterns)
    if sources_path.exists():
        sources_manifest = build_file_manifest(sources_path, doc_extensions, exclude_patterns=doc_exclude_patterns)
        manifest.update(sources_manifest)
    save_file_manifest(client, _drop_failed(manifest, failed_files))

    return _summary(
        client_name, "rebuild",
        new=files_found - len(failed_files),
        failed_files=failed_files,
        chunks_added=chunks_added,
    )


def rebuild_index(
    client_name: str,
    force_bm25_rebuild: bool = False,
    include_code: bool = True,
    code_only: bool = False,
    resume: bool = False,
    sync_sources: bool = False,
) -> dict:
    """
    Rebuild index from existing documents and code. Returns the ingest summary.

    Args:
        client_name: Name of the client
        force_bm25_rebuild: Force rebuild of BM25 index (clears cache)
        include_code: Also index code files if configured
        code_only: Only index code files, skip documents
        resume: Resume from last checkpoint if available
        sync_sources: Sync configured sources before rebuilding
    """
    client = load_client(client_name)
    logger.info(f"Rebuilding index for: {client.name}")

    # Sync sources if requested or if sync_on_ingest is configured
    if sync_sources or _should_sync_on_ingest(client):
        sync_client_sources(client_name)

    summary = _summary(client_name, "rebuild")
    if not code_only:
        summary = ingest_client_documents(client_name, client.documents_path, force_bm25_rebuild, resume=resume)

    # Index code if enabled
    if include_code or code_only:
        ingest_client_code_files(client_name, rebuild=True)
    return summary


def ingest_client_code_files(client_name: str, rebuild: bool = False) -> None:
    """
    Ingest code files for a client using the progressive pipeline.

    Creates HybridSearcher first, passes it to ingest_client_code so the
    pipeline can embed and index progressively. BM25 is rebuilt once at end.

    Args:
        client_name: Name of the client
        rebuild: Force full rebuild (ignore incremental state)
    """
    client = load_client(client_name)

    code_config = client.config.get("code", {})
    if not code_config.get("enabled", False):
        logger.info(f"Code indexing not enabled for client {client.name}")
        return

    # Get embedding config
    embedding_config = client.embedding_config
    embedding_model = embedding_config.get("model", "all-MiniLM-L6-v2")
    passage_prefix = embedding_config.get("passage_prefix", "")

    # Create ingestor
    ingestor = DocumentIngestor(
        markdown_config=client.markdown_config,
        embedding_model=embedding_model,
        passage_prefix=passage_prefix,
    )

    # Create searcher FIRST (pipeline needs it for progressive indexing)
    # Skip BM25 cache load — rebuilt at end, avoids slow deserialize for large clients
    searcher = HybridSearcher(
        index_path=str(client.index_path),
        collection_name=client.machine_name,
        bm25_weight=client.search_config.get("bm25_weight", 0.5),
        vector_weight=client.search_config.get("vector_weight", 0.5),
        tokenizer_config=client.tokenization_config,
        embedding_config=embedding_config,
        skip_bm25_load=True,
    )

    # Ingest code with pipeline (progressive embed + index)
    logger.info(f"Ingesting code for client: {client.name}")
    chunks, embeddings = ingest_client_code(client, ingestor, rebuild=rebuild, pipeline_searcher=searcher)

    if not chunks:
        # Pipeline already indexed everything — check collection count
        code_count = searcher.collection.count()
        if code_count > 0:
            logger.info(f"Code indexed progressively ({code_count} total chunks in collection)")
        else:
            logger.info("No code to index")

    # BM25 rebuild once at end
    try:
        searcher._rebuild_bm25_index()
    except Exception as e:
        logger.warning(f"Failed to rebuild BM25 index: {e}")

    stats = searcher.get_stats()
    logger.info(f"Code indexing complete. Stats: {stats}")


def pre_assess_clients(client_names: list[str]) -> dict:
    """Scan clients and classify by size for scheduling."""
    from core.resource_manager import ResourceManager
    rm = ResourceManager()
    assessments = {}
    for name in client_names:
        try:
            client = load_client(name)
            docs_path = client.documents_path
            file_count = sum(1 for f in docs_path.rglob("*") if f.is_file()) if docs_path.exists() else 0
            chunk_count = 0
            try:
                import chromadb
                from chromadb.config import Settings
                cc = chromadb.PersistentClient(
                    path=str(client.index_path),
                    settings=Settings(anonymized_telemetry=False),
                )
                col = cc.get_collection(name=client.machine_name)
                chunk_count = col.count()
            except Exception:
                chunk_count = file_count * 5  # rough estimate
            size = rm.classify_client(file_count, chunk_count)
            assessments[name] = {"files": file_count, "chunks": chunk_count, "size": size}
            logger.info(f"Pre-assess {name}: {file_count} files, {chunk_count} chunks -> {size}")
        except Exception as e:
            logger.warning(f"Pre-assess failed for {name}: {e}")
            assessments[name] = {"files": 0, "chunks": 0, "size": "small"}
    return assessments


def _reindex_single_client(name: str, args) -> dict | None:
    """Reindex a single client (called from parallel or sequential loop)."""
    logger.info(f"--- {name} ---")
    if args.code_only or args.rebuild_code:
        ingest_client_code_files(name, rebuild=args.rebuild_code)
        return None
    elif args.rebuild:
        return rebuild_index(
            name,
            args.force_bm25_rebuild,
            include_code=not args.no_code,
            sync_sources=args.sync_sources,
        )
    else:
        return incremental_reindex(name, include_code=not args.no_code)


def list_clients() -> None:
    """List all available clients."""
    clients = discover_clients()
    if clients:
        print("Available clients:")
        for name in clients:
            print(f"  - {name}")
    else:
        print("No clients found. Create a client directory in clients/")


def _main():
    parser = argparse.ArgumentParser(
        description="Trove Document and Code Ingestion CLI"
    )

    parser.add_argument(
        "--client", "-c",
        help="Client name (directory in clients/)"
    )

    parser.add_argument(
        "--path", "-p",
        type=Path,
        help="Path to documents directory (overrides client config)"
    )

    parser.add_argument(
        "--rebuild", "-r",
        action="store_true",
        help="Full rebuild: delete index and re-ingest everything"
    )

    parser.add_argument(
        "--incremental", "-i",
        action="store_true",
        help="Incremental reindex: only process new/modified/deleted files (default when no flags given)"
    )

    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process all clients"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available clients"
    )

    parser.add_argument(
        "--force-bm25-rebuild",
        action="store_true",
        help="Force rebuild of BM25 index (clears cache)"
    )

    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Only index code files (skip documents)"
    )

    parser.add_argument(
        "--rebuild-code",
        action="store_true",
        help="Rebuild code index from scratch (ignore incremental state)"
    )

    parser.add_argument(
        "--no-code",
        action="store_true",
        help="Skip code indexing (documents only)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume ingestion from last checkpoint"
    )

    parser.add_argument(
        "--sync-sources",
        action="store_true",
        help="Sync configured sources (GitHub/web) before ingestion"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process small clients concurrently (with --all)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON summary (new, changed, unchanged, deleted and failed files) on stdout",
    )

    add_clients_dir_argument(parser)
    args = parser.parse_args()
    apply_clients_dir_argument(args)

    if args.list:
        list_clients()
        return

    if args.all:
        # Process all clients
        client_names = discover_clients()
        if not client_names:
            logger.error("No clients found")
            return

        # Log resource summary
        from core.resource_manager import ResourceManager
        rm = ResourceManager()
        logger.info(f"System: {rm.summary()}")
        logger.info(f"Processing {len(client_names)} clients: {', '.join(client_names)}")
        summaries: list[dict | None] = []

        if args.parallel:
            # Pre-assess clients and process small ones concurrently
            assessments = pre_assess_clients(client_names)
            small_clients = [n for n, a in assessments.items() if a["size"] == "small"]
            other_clients = [n for n, a in assessments.items() if a["size"] != "small"]

            logger.info(
                f"Pre-assessment: {len(small_clients)} small, {len(other_clients)} medium/large"
            )

            # Process small clients concurrently (up to 3 threads)
            if small_clients:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                max_concurrent = min(3, len(small_clients))
                logger.info(
                    f"Processing {len(small_clients)} small clients "
                    f"concurrently ({max_concurrent} threads)..."
                )
                with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    futures = {
                        executor.submit(_reindex_single_client, name, args): name
                        for name in small_clients
                    }
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            summaries.append(future.result())
                            logger.info(f"Done: {name}")
                        except Exception as e:
                            logger.error(f"Failed: {name}: {e}")

            # Process medium/large clients sequentially with full resources
            for name in other_clients:
                try:
                    summaries.append(_reindex_single_client(name, args))
                    logger.info(f"Done: {name}")
                except Exception as e:
                    logger.error(f"Failed: {name}: {e}")
        else:
            # Sequential processing (original behavior)
            for name in client_names:
                try:
                    summaries.append(_reindex_single_client(name, args))
                except Exception as e:
                    logger.error(f"Failed to process {name}: {e}")
                    summaries.append(_summary(name, "error", error=str(e)))
        if args.json:
            print(json.dumps({"clients": [x for x in summaries if x]}, indent=2))
        return

    if not args.client:
        parser.error("--client is required (or use --list or --all)")

    if args.stats:
        show_stats(args.client)
        return

    # Sync sources if explicitly requested
    if args.sync_sources:
        sync_client_sources(args.client)

    summary = None
    if args.code_only or args.rebuild_code:
        ingest_client_code_files(args.client, rebuild=args.rebuild_code)
    elif args.rebuild:
        summary = rebuild_index(
            args.client,
            args.force_bm25_rebuild,
            include_code=not args.no_code,
            resume=args.resume,
            sync_sources=args.sync_sources,
        )
    elif args.path:
        summary = ingest_client_documents(args.client, args.path, args.force_bm25_rebuild, resume=args.resume)
        if not args.no_code:
            ingest_client_code_files(args.client)
    else:
        # Default: incremental reindex (only changed files)
        summary = incremental_reindex(args.client, include_code=not args.no_code)

    if args.json:
        print(json.dumps(summary or _summary(args.client, "code"), indent=2))


def show_stats(client_name: str) -> None:
    """Show index statistics for a client."""
    client = load_client(client_name)

    searcher = HybridSearcher(
        index_path=str(client.index_path),
        collection_name=client.machine_name,
        embedding_config=client.embedding_config,
    )

    stats = searcher.get_stats()
    print(f"\nIndex statistics for {client.name}:")
    print(f"  Total chunks: {stats.get('total_chunks', 0)}")
    print(f"  BM25 documents: {stats.get('bm25_docs', 0)}")

    # Count by chunk type if available
    try:
        results = get_all(searcher.collection, include=["metadatas"])
        if results and results.get("metadatas"):
            chunk_types = {}
            languages = {}
            for metadata in results["metadatas"]:
                ct = metadata.get("chunk_type", "document")
                chunk_types[ct] = chunk_types.get(ct, 0) + 1
                if ct == "code":
                    lang = metadata.get("language", "unknown")
                    languages[lang] = languages.get(lang, 0) + 1

            print("\n  By type:")
            for ct, count in sorted(chunk_types.items()):
                print(f"    {ct}: {count}")

            if languages:
                print("\n  Code by language:")
                for lang, count in sorted(languages.items()):
                    print(f"    {lang}: {count}")
    except Exception as e:
        logger.debug(f"Could not get detailed stats: {e}")


def main():
    from core.client import ClientNotFoundError

    try:
        _main()
    except ClientNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
