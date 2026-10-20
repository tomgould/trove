#!/usr/bin/env python3
"""
Trove File Watcher - Monitors client document changes and reindexes incrementally.

Compares files on disk against the file manifest saved during ingestion.
Detects new, modified, and deleted files without relying on timestamps.

Usage:
    python scripts/watch_and_reindex.py [--verbose] [--dry-run] [--client NAME]

Cron example (every 15 minutes):
    */15 * * * * cd /path/to/trove && ./venv/bin/python scripts/watch_and_reindex.py
"""

import argparse
import fcntl
import fnmatch
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.client import load_client, discover_clients
from core.ingest import (
    DocumentIngestor,
    build_file_manifest,
    load_file_manifest,
    save_file_manifest,
    diff_file_manifest,
)
from core.search import HybridSearcher

# Configuration
LOCK_FILE = "/tmp/trove-watcher.lock"
LOG_FILE = "/tmp/trove-watcher.log"
CUTOFF_SECONDS = 600  # 10 minutes for 15-minute cron interval

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def get_chunk_ids_for_file(searcher: HybridSearcher, file_path: str) -> list[str]:
    """Get existing chunk IDs for a file from ChromaDB."""
    try:
        results = searcher.collection.get(
            where={"source_path": file_path},
            include=[]
        )
        return results.get("ids", [])
    except Exception:
        return []


def reindex_file(client, file_path: Path, ingestor: DocumentIngestor, searcher: HybridSearcher) -> int:
    """Reindex a single file. Returns chunk count."""
    file_path_str = str(file_path)

    # Delete old chunks
    old_ids = get_chunk_ids_for_file(searcher, file_path_str)
    if old_ids:
        try:
            searcher.collection.delete(ids=old_ids)
            logger.debug(f"Deleted {len(old_ids)} old chunks for {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to delete old chunks: {e}")

    # Ingest
    try:
        chunks, embeddings = ingestor.ingest_document(file_path)
        if not chunks:
            return 0

        # Add metadata
        docs_path = client.documents_path
        try:
            rel_path = file_path.relative_to(docs_path)
        except ValueError:
            rel_path = file_path.name

        for chunk in chunks:
            chunk["metadata"]["source_file"] = str(rel_path)
            chunk["metadata"]["source_path"] = file_path_str
            chunk["metadata"]["client"] = client.machine_name

        # Add to index
        searcher.collection.add(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["content"] for c in chunks],
            metadatas=[searcher._sanitize_metadata(c["metadata"]) for c in chunks],
        )

        logger.info(f"Indexed {file_path.name}: {len(chunks)} chunks")
        return len(chunks)
    except Exception as e:
        logger.error(f"Failed to ingest {file_path.name}: {e}")
        return 0


def remove_file_from_index(searcher: HybridSearcher, file_path: str) -> int:
    """Remove a deleted file's chunks from the index. Returns count removed."""
    old_ids = get_chunk_ids_for_file(searcher, file_path)
    if old_ids:
        try:
            searcher.collection.delete(ids=old_ids)
            logger.info(f"Removed {len(old_ids)} chunks for deleted file: {Path(file_path).name}")
            return len(old_ids)
        except Exception as e:
            logger.warning(f"Failed to remove chunks for {file_path}: {e}")
    return 0


def acquire_lock():
    """Acquire exclusive lock."""
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        return fd
    except (OSError, IOError):
        return None


def release_lock(fd: int):
    """Release lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except (OSError, IOError):
        pass


def sync_source_files(client, dry_run: bool = False) -> tuple[int, int, int]:
    """
    Sync source files to client's documents directory based on source_sync config.

    Reads source_sync from client.yaml:
        source_sync:
          enabled: true
          source: ../..         # relative to client dir
          include: ["*.py", "*.md"]
          exclude: ["clients/", "venv/", ".git/"]

    Returns (copied, updated, removed) counts.
    """
    sync_config = client.config.get("source_sync", {})
    if not sync_config.get("enabled", False):
        return 0, 0, 0

    source_rel = sync_config.get("source", "")
    if not source_rel:
        return 0, 0, 0

    source_dir = (client.path / source_rel).resolve()
    dest_dir = client.documents_path

    if not source_dir.exists():
        logger.warning(f"Source sync dir does not exist: {source_dir}")
        return 0, 0, 0

    include_patterns = sync_config.get("include", ["*"])
    exclude_patterns = sync_config.get("exclude", [])

    def should_exclude(rel_path: str) -> bool:
        """Check if a relative path matches any exclude pattern."""
        parts = Path(rel_path).parts
        for pattern in exclude_patterns:
            # Match against full relative path
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            # Match against any path component (for dir patterns like "clients/")
            clean = pattern.rstrip("/")
            if any(fnmatch.fnmatch(part, clean) for part in parts):
                return True
        return False

    def should_include(filename: str) -> bool:
        """Check if filename matches any include pattern."""
        return any(fnmatch.fnmatch(filename, p) for p in include_patterns)

    # Collect source files (follow symlinks with cycle detection)
    source_files: dict[str, Path] = {}
    seen_real = set()
    for root, dirs, filenames in os.walk(source_dir, followlinks=True):
        real_root = os.path.realpath(root)
        if real_root in seen_real:
            dirs.clear()
            continue
        seen_real.add(real_root)
        for fname in filenames:
            path = Path(root) / fname
            rel = str(path.relative_to(source_dir))
            if should_exclude(rel):
                continue
            if not should_include(fname):
                continue
            source_files[rel] = path

    copied = 0
    updated = 0
    removed = 0

    # Copy new and updated files
    for rel, src_path in source_files.items():
        dest_path = dest_dir / rel
        if not dest_path.exists():
            if dry_run:
                logger.info(f"[DRY-RUN] Sync new: {rel}")
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                logger.debug(f"Synced new: {rel}")
            copied += 1
        elif src_path.stat().st_mtime > dest_path.stat().st_mtime:
            if dry_run:
                logger.info(f"[DRY-RUN] Sync updated: {rel}")
            else:
                shutil.copy2(src_path, dest_path)
                logger.debug(f"Synced updated: {rel}")
            updated += 1

    # Remove files in dest that no longer exist in source
    # Only remove files that match include patterns (don't touch unrelated files)
    seen_real_dest = set()
    for root, dirs, filenames in os.walk(dest_dir, followlinks=True):
        real_root = os.path.realpath(root)
        if real_root in seen_real_dest:
            dirs.clear()
            continue
        seen_real_dest.add(real_root)
        for fname in filenames:
            path = Path(root) / fname
            rel = str(path.relative_to(dest_dir))
            if not should_include(fname):
                continue
        if rel not in source_files:
            if dry_run:
                logger.info(f"[DRY-RUN] Sync remove: {rel}")
            else:
                path.unlink()
                logger.debug(f"Synced removed: {rel}")
                # Clean up empty parent dirs
                parent = path.parent
                while parent != dest_dir and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
            removed += 1

    total = copied + updated + removed
    if total > 0:
        logger.info(
            f"Source sync for {client.machine_name}: "
            f"{copied} new, {updated} updated, {removed} removed"
        )

    return copied, updated, removed


def main():
    parser = argparse.ArgumentParser(description="Trove File Watcher")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("--client", "-c", help="Only process specific client")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    lock_fd = acquire_lock()
    if lock_fd is None:
        logger.debug("Another instance running, exiting")
        sys.exit(0)

    try:
        start_time = time.time()
        trove_root = Path(__file__).parent.parent

        # Discover clients
        client_names = discover_clients(trove_root / "clients")
        if args.client:
            if args.client not in client_names:
                logger.error(f"Client not found: {args.client}")
                sys.exit(1)
            client_names = [args.client]

        # Sync source files and compare against manifest for each client
        changes_by_client = {}
        for name in client_names:
            try:
                client = load_client(name)

                # Sync source files if configured
                if client.config.get("source_sync", {}).get("enabled", False):
                    sync_source_files(client, dry_run=args.dry_run)

                doc_extensions = client.get_document_extensions()
                doc_exclude_patterns = client.documents_config.get("exclude_patterns", [])

                # Build current file list from disk
                current_files = build_file_manifest(client.documents_path, doc_extensions, exclude_patterns=doc_exclude_patterns)

                # Load the manifest saved during ingestion
                indexed_files = load_file_manifest(client)
                if not indexed_files and not current_files:
                    continue

                new, modified, deleted = diff_file_manifest(current_files, indexed_files)
                if new or modified or deleted:
                    changes_by_client[name] = {
                        "client": client,
                        "new": new,
                        "modified": modified,
                        "deleted": deleted,
                        "current_files": current_files,
                    }
                    logger.info(
                        f"{name}: {len(new)} new, {len(modified)} modified, {len(deleted)} deleted"
                    )
            except Exception as e:
                logger.warning(f"Failed to check client {name}: {e}")

        if not changes_by_client:
            logger.info("No changes detected")
            return

        total_changes = sum(
            len(c["new"]) + len(c["modified"]) + len(c["deleted"])
            for c in changes_by_client.values()
        )
        logger.info(f"Found {total_changes} changes across {len(changes_by_client)} clients")

        if args.dry_run:
            for name, changes in changes_by_client.items():
                for f in changes["new"]:
                    logger.info(f"[DRY-RUN] New: {name}/{Path(f).name}")
                for f in changes["modified"]:
                    logger.info(f"[DRY-RUN] Modified: {name}/{Path(f).name}")
                for f in changes["deleted"]:
                    logger.info(f"[DRY-RUN] Deleted: {name}/{Path(f).name}")
            return

        total_chunks = 0
        total_files = 0

        for name, changes in changes_by_client.items():
            if time.time() - start_time >= CUTOFF_SECONDS:
                logger.info(f"Hit {CUTOFF_SECONDS}s cutoff, will continue next run")
                break

            client = changes["client"]

            # Create ingestor per-client with the client's embedding config
            embedding_config = client.embedding_config
            ingestor = DocumentIngestor(
                embedding_model=embedding_config.get("model", "all-MiniLM-L6-v2"),
                passage_prefix=embedding_config.get("passage_prefix", ""),
            )

            try:
                searcher = HybridSearcher(
                    index_path=str(client.index_path),
                    collection_name=client.machine_name,
                    embedding_config=embedding_config,
                )
            except Exception as e:
                logger.error(f"Failed to init searcher for {name}: {e}")
                continue

            # Remove deleted files from index
            for f in changes["deleted"]:
                if time.time() - start_time >= CUTOFF_SECONDS:
                    break
                remove_file_from_index(searcher, f)

            # Index new and modified files
            for f in changes["new"] + changes["modified"]:
                if time.time() - start_time >= CUTOFF_SECONDS:
                    break
                chunks = reindex_file(client, Path(f), ingestor, searcher)
                total_chunks += chunks
                total_files += 1

            # Rebuild BM25
            try:
                searcher._rebuild_bm25_index()
            except Exception as e:
                logger.warning(f"Failed to rebuild BM25 for {name}: {e}")

            # Update the manifest for this client
            save_file_manifest(client, changes["current_files"])

        elapsed = time.time() - start_time
        logger.info(f"Done: {total_files} files, {total_chunks} chunks in {elapsed:.1f}s")

    finally:
        release_lock(lock_fd)


if __name__ == "__main__":
    main()
