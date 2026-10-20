"""
Producer-consumer ingestion pipeline.

Decouples file parsing (CPU, threaded) from embedding (GPU, batched) from
indexing (ChromaDB, progressive). Supports Ctrl+C resume.
"""

import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Producer-consumer pipeline: parse (thread pool) -> embed (GPU) -> index (ChromaDB).

    Buffer accumulates chunks until embed_batch_size, then flushes:
    embed on GPU + index in ChromaDB immediately. Ctrl+C loses at most 1 batch.
    """

    def __init__(
        self,
        ingestor,
        searcher,
        parse_workers: int = 4,
        embed_batch_size: int = 512,
        contextualize: bool = False,
        resource_manager=None,
    ):
        # If resource_manager provided, override static defaults with adaptive values
        if resource_manager:
            gpu = resource_manager.probe_gpu()
            if gpu:
                model_dim = getattr(
                    ingestor.embedder, 'get_sentence_embedding_dimension', lambda: 384
                )()
                embed_batch_size = resource_manager.recommend_embed_batch(model_dim)
                logger.info(f"ResourceManager: adaptive embed_batch_size={embed_batch_size}")

        self.ingestor = ingestor
        self.searcher = searcher
        self.parse_workers = parse_workers
        self.embed_batch_size = embed_batch_size
        self.contextualize = contextualize

        # Buffer state
        self._chunk_buffer: list[dict] = []
        self._context_buffer: list[str] = []
        self._buffer_source_paths: set[str] = set()

        # Progress tracking
        self._flushed_files: set[str] = set()
        self._total_chunks_indexed: int = 0
        self._total_files_processed: int = 0
        self._failed_files: list[str] = []
        self._flush_count: int = 0
        self._start_time: float = 0.0

        # Ctrl+C handling
        self._interrupted = False
        self._original_sigint = None

    def _install_signal_handler(self):
        """Install SIGINT handler that flushes buffer before exit.

        Only works in the main thread — silently skips in worker threads
        (e.g. when running inside ThreadPoolExecutor for parallel clients).
        """
        import threading
        if threading.current_thread() is not threading.main_thread():
            return

        self._original_sigint = signal.getsignal(signal.SIGINT)

        def _handler(signum, frame):
            if self._interrupted:
                logger.warning("Force exit requested")
                if self._original_sigint and callable(self._original_sigint):
                    self._original_sigint(signum, frame)
                return
            logger.info("Interrupt received — flushing current buffer before exit...")
            self._interrupted = True

        signal.signal(signal.SIGINT, _handler)

    def _restore_signal_handler(self):
        """Restore original SIGINT handler."""
        import threading
        if threading.current_thread() is not threading.main_thread():
            return
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
            self._original_sigint = None

    def _flush_buffer(self):
        """Embed current buffer on GPU and index into ChromaDB immediately."""
        if not self._chunk_buffer:
            return

        count = len(self._chunk_buffer)
        logger.info(f"Flushing buffer: {count} chunks (batch {self._flush_count + 1})")

        # Deduplicate chunks by ID within this batch (hash collisions possible)
        seen_ids = set()
        deduped_chunks = []
        deduped_contexts = []
        for i, chunk in enumerate(self._chunk_buffer):
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                deduped_chunks.append(chunk)
                if self.contextualize and self._context_buffer:
                    deduped_contexts.append(self._context_buffer[i])
        if len(deduped_chunks) < len(self._chunk_buffer):
            logger.debug(f"Deduplicated {len(self._chunk_buffer) - len(deduped_chunks)} chunks in batch")
        self._chunk_buffer = deduped_chunks
        count = len(self._chunk_buffer)

        context_strings = deduped_contexts if self.contextualize and deduped_contexts else None
        embeddings = self.ingestor.create_embeddings(
            self._chunk_buffer, context_strings=context_strings,
        )

        self.searcher.collection.upsert(
            ids=[c["id"] for c in self._chunk_buffer],
            embeddings=embeddings,
            documents=[c["content"] for c in self._chunk_buffer],
            metadatas=[self.searcher._sanitize_metadata(c["metadata"]) for c in self._chunk_buffer],
        )

        self._total_chunks_indexed += count
        self._flushed_files.update(self._buffer_source_paths)
        self._flush_count += 1

        elapsed = time.time() - self._start_time if self._start_time else 0
        rate = self._total_chunks_indexed / elapsed if elapsed > 0 else 0
        logger.info(
            f"Indexed {count} chunks (total: {self._total_chunks_indexed}, "
            f"files: {len(self._flushed_files)}, rate: {rate:.0f} chunks/s)"
        )

        self._chunk_buffer.clear()
        self._context_buffer.clear()
        self._buffer_source_paths.clear()

    def _add_chunks_to_buffer(self, chunks: list[dict], context_strings: list[str] = None):
        """Add chunks to buffer, flushing when full."""
        for i, chunk in enumerate(chunks):
            self._chunk_buffer.append(chunk)
            if context_strings:
                self._context_buffer.append(context_strings[i])
            source_path = chunk.get("metadata", {}).get("source_path", "")
            if source_path:
                self._buffer_source_paths.add(source_path)

            if len(self._chunk_buffer) >= self.embed_batch_size:
                self._flush_buffer()

    def _parse_document_file(self, path: Path, base_directory: Path) -> list[dict]:
        """Parse a single document file into chunks (runs in thread pool)."""
        try:
            if path.suffix.lower() == ".mbox":
                chunks = self.ingestor.ingest_mbox_chunks_only(path)
            else:
                elements = self.ingestor.parse_document(path)
                chunks = self.ingestor.chunk_elements(elements, source_id=str(path))

            try:
                rel_path = path.relative_to(base_directory)
            except ValueError:
                rel_path = path.name

            for chunk in chunks:
                chunk["metadata"]["source_file"] = str(rel_path)
                chunk["metadata"]["source_path"] = str(path)

            return chunks
        except Exception as e:
            logger.error(f"Failed to parse {path.name}: {e}")
            return []

    def get_indexed_source_paths(self) -> set[str]:
        """Query ChromaDB for all already-indexed source_path values (for resume)."""
        try:
            results = self.searcher.collection.get(include=["metadatas"])
            paths = set()
            for meta in (results.get("metadatas") or []):
                sp = meta.get("source_path", "")
                if sp:
                    paths.add(sp)
            return paths
        except Exception as e:
            logger.warning(f"Could not query indexed paths for resume: {e}")
            return set()

    def run_documents(
        self,
        files: list[Path],
        base_directory: Path,
        already_indexed: set[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Pipeline for document files: parallel parse -> buffer -> embed -> index.

        Args:
            files: List of document file paths
            base_directory: Base directory for relative path computation
            already_indexed: Set of source_path strings to skip (resume)
            progress_callback: Optional callback(stats_dict) after each flush

        Returns:
            Stats dict with counts
        """
        already_indexed = already_indexed or set()
        files_to_process = [f for f in files if str(f) not in already_indexed]
        skipped = len(files) - len(files_to_process)

        if skipped:
            logger.info(f"Resuming: skipping {skipped} already-indexed files")

        logger.info(f"Processing {len(files_to_process)} documents with {self.parse_workers} workers")
        self._start_time = time.time()
        self._install_signal_handler()

        try:
            with ThreadPoolExecutor(max_workers=self.parse_workers) as executor:
                future_to_path = {
                    executor.submit(self._parse_document_file, path, base_directory): path
                    for path in files_to_process
                }

                for future in as_completed(future_to_path):
                    if self._interrupted:
                        logger.info("Interrupted — stopping submission of new files")
                        for f in future_to_path:
                            f.cancel()
                        break

                    path = future_to_path[future]
                    try:
                        chunks = future.result()
                        if chunks:
                            context_strings = None
                            if self.contextualize:
                                from core.ingest import build_context_prefix
                                context_strings = [build_context_prefix(c["metadata"]) for c in chunks]
                                for chunk in chunks:
                                    chunk["metadata"]["contextualized"] = True

                            self._add_chunks_to_buffer(chunks, context_strings)

                        self._total_files_processed += 1

                        if self._total_files_processed % 10 == 0 or self._total_files_processed == len(files_to_process):
                            logger.info(
                                f"Files: {self._total_files_processed}/{len(files_to_process)} "
                                f"| Buffered: {len(self._chunk_buffer)} | Flushed: {self._flush_count}"
                            )
                    except Exception as e:
                        logger.error(f"Failed to process {path.name}: {e}")
                        self._failed_files.append(str(path))
                        self._total_files_processed += 1

                    if progress_callback:
                        progress_callback(self._get_stats())

            # Flush remaining buffer
            self._flush_buffer()

        finally:
            self._restore_signal_handler()

        return self._get_stats()

    def run_code(self, chunks: list[dict]) -> dict:
        """
        Pipeline for pre-chunked code: buffer -> embed -> index.

        Args:
            chunks: Pre-chunked code dicts with 'id', 'content', 'metadata'

        Returns:
            Stats dict with counts
        """
        if not chunks:
            return self._get_stats()

        logger.info(f"Processing {len(chunks)} code chunks through pipeline")
        self._start_time = time.time()
        self._install_signal_handler()

        try:
            self._add_chunks_to_buffer(chunks)
            self._flush_buffer()
        finally:
            self._restore_signal_handler()

        return self._get_stats()

    def _get_stats(self) -> dict:
        """Return current pipeline statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "total_chunks_indexed": self._total_chunks_indexed,
            "total_files_processed": self._total_files_processed,
            "flush_count": self._flush_count,
            "failed_files": list(self._failed_files),
            "flushed_files_count": len(self._flushed_files),
            "buffer_size": len(self._chunk_buffer),
            "elapsed_seconds": round(elapsed, 1),
            "chunks_per_second": round(self._total_chunks_indexed / elapsed, 1) if elapsed > 0 else 0,
            "interrupted": self._interrupted,
        }
