"""
Document ingestion using Unstructured.

This module handles:
- Parsing documents (PDF, DOCX, XLSX, HTML, Markdown, etc.)
- Parsing email archives (mbox format)
- Chunking by section/title
- Creating embeddings
- Indexing in vector DB and BM25
"""

import gc
import hashlib
import html
import json
import logging
import mailbox
import os
import re
from datetime import datetime
from email.header import decode_header
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.ingest_progress import IngestionCheckpoint

import torch
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default embedding model (local, good quality)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunk validation defaults
MIN_CHUNK_CHARS = 50


def validate_chunk(content: str, min_chars: int = MIN_CHUNK_CHARS) -> tuple[bool, str | None]:
    """
    Validate a chunk's content quality.

    Args:
        content: The chunk text content
        min_chars: Minimum character count after stripping whitespace

    Returns:
        Tuple of (is_valid, skip_reason). skip_reason is None when valid.
    """
    stripped = content.strip()
    if len(stripped) < min_chars:
        return False, "too_short"
    if not any(c.isalnum() for c in stripped):
        return False, "no_alphanumeric"
    return True, None

def build_context_prefix(metadata: dict) -> str:
    """
    Build a context prefix string from chunk metadata.

    Used for contextual embeddings: the prefix is prepended to the chunk
    content before embedding, but NOT stored in ChromaDB or BM25.

    Args:
        metadata: Chunk metadata dict (source_file, page_number, element_type, etc.)

    Returns:
        Context prefix string, or empty string if no useful metadata.
    """
    parts = []

    source_file = metadata.get("source_file")
    if source_file:
        parts.append(f"Document: {source_file}.")

    page_number = metadata.get("page_number")
    if page_number is not None:
        parts.append(f"Page {page_number}.")

    if not parts:
        return ""

    return " ".join(parts) + " "


# Batch size for embeddings (smaller = less GPU memory, slower)
# Can be overridden with TROVE_EMBED_BATCH_SIZE env var
# Starts high and auto-settles via OOM retry cascade; settled value is cached globally.
DEFAULT_EMBED_BATCH_SIZE = 512

# Global cache: once the OOM cascade finds the safe max, remember it across all Ingestor instances.
_global_effective_batch_size: int | None = None


def _clear_gpu_memory():
    """Clear GPU memory cache to prevent OOM errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def _decode_email_header(header_value: str) -> str:
    """Decode an email header that may contain encoded words."""
    if not header_value:
        return ""

    decoded_parts = []
    for part, charset in decode_header(header_value):
        if isinstance(part, bytes):
            try:
                decoded_parts.append(part.decode(charset or "utf-8", errors="replace"))
            except (LookupError, UnicodeDecodeError):
                decoded_parts.append(part.decode("utf-8", errors="replace"))
        else:
            decoded_parts.append(part)

    return " ".join(decoded_parts)


def _strip_html(html_content: str) -> str:
    """Convert HTML to plain text using BeautifulSoup."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")
    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Normalize whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _detect_charset(payload: bytes, declared_charset: str | None) -> str:
    """
    Detect the character encoding of a byte payload.

    Tries declared charset first, falls back to chardet detection,
    then utf-8 as final fallback.

    Args:
        payload: Raw byte content
        declared_charset: Charset declared by email headers or HTTP headers

    Returns:
        Best-guess charset string
    """
    if declared_charset:
        try:
            payload.decode(declared_charset)
            return declared_charset
        except (UnicodeDecodeError, LookupError):
            pass

    try:
        import chardet
        detected = chardet.detect(payload)
        if detected and detected.get("encoding"):
            try:
                payload.decode(detected["encoding"])
                return detected["encoding"]
            except (UnicodeDecodeError, LookupError):
                pass
    except ImportError:
        pass

    return "utf-8"


def _extract_email_body(message) -> str:
    """Extract plain text body from an email message."""
    body_parts = []

    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Skip attachments
            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = _detect_charset(payload, part.get_content_charset())
                        body_parts.append(payload.decode(charset, errors="replace"))
                except Exception:
                    pass
            elif content_type == "text/html" and not body_parts:
                # Only use HTML if no plain text found yet
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = _detect_charset(payload, part.get_content_charset())
                        html_text = payload.decode(charset, errors="replace")
                        body_parts.append(_strip_html(html_text))
                except Exception:
                    pass
    else:
        # Single part message
        content_type = message.get_content_type()
        try:
            payload = message.get_payload(decode=True)
            if payload:
                charset = _detect_charset(payload, message.get_content_charset())
                text = payload.decode(charset, errors="replace")
                if content_type == "text/html":
                    text = _strip_html(text)
                body_parts.append(text)
        except Exception:
            pass

    return "\n\n".join(body_parts)


class DocumentIngestor:
    """Handles document parsing, chunking, and indexing."""

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_max_chars: int = 1500,
        chunk_overlap: int = 100,
        passage_prefix: str = "",
    ):
        """
        Initialize the document ingestor.

        Args:
            embedding_model: Sentence transformer model name
            chunk_max_chars: Maximum characters per chunk
            chunk_overlap: Character overlap between chunks
            passage_prefix: Prefix to add to passages before embedding
                            (e.g., "passage: " for E5 models)
        """
        self.chunk_max_chars = chunk_max_chars
        self.chunk_overlap = chunk_overlap
        self.passage_prefix = passage_prefix

        # Use ModelCache to share models across clients (avoids redundant GPU loads)
        from core.model_cache import get_model_cache
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.embedder = get_model_cache().get_embedder(embedding_model, device)
            logger.info(f"Initialized ingestor with model: {embedding_model} on {device}")
        except RuntimeError as e:
            if device == "cuda" and ("out of memory" in str(e).lower() or "cuda" in str(e).lower()):
                logger.warning(f"GPU unavailable ({e}), falling back to CPU")
                self.embedder = get_model_cache().get_embedder(embedding_model, "cpu")
                logger.info(f"Initialized ingestor with model: {embedding_model} on cpu")
            else:
                raise

    def parse_document(self, path: str | Path) -> list[dict]:
        """
        Parse a document into elements using Unstructured.

        Args:
            path: Path to document file

        Returns:
            List of parsed elements with metadata
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        logger.info(f"Parsing document: {path.name}")

        # Unstructured auto-detects file type
        elements = partition(str(path))

        logger.info(f"Parsed {len(elements)} elements from {path.name}")
        return elements

    class _TextChunk:
        """Wraps plain text as a pseudo-chunk with an empty metadata dict."""
        def __init__(self, text):
            self._text = text
            self.metadata = type("M", (), {"to_dict": lambda s: {}})()
        def __str__(self):
            return self._text

    def chunk_elements(
        self, elements: list, source_id: str = "", chunking_config: dict | None = None,
    ) -> list[dict]:
        """
        Chunk parsed elements by title/section or semantic similarity.

        Args:
            elements: Parsed elements from Unstructured
            source_id: Optional source identifier for unique chunk IDs
            chunking_config: Optional chunking strategy config dict

        Returns:
            List of chunk dictionaries with content and metadata
        """
        strategy = (chunking_config or {}).get("strategy", "title")

        if strategy == "semantic":
            from core.chunking import chunk_by_semantic
            semantic_texts = chunk_by_semantic(
                elements,
                embedder=self.embedder,
                threshold=(chunking_config or {}).get("semantic_threshold", 0.5),
                max_chars=self.chunk_max_chars,
                min_chars=(chunking_config or {}).get("min_chunk_size", 100),
            )
            chunks = [self._TextChunk(t) for t in semantic_texts]
        elif strategy == "recursive":
            from core.chunking import chunk_by_recursive
            # Flatten elements to text
            full_text = "\n\n".join(str(e) for e in elements if str(e).strip())
            recursive_texts = chunk_by_recursive(
                full_text,
                max_chars=(chunking_config or {}).get("max_chunk_size", self.chunk_max_chars),
                overlap=(chunking_config or {}).get("overlap", self.chunk_overlap),
                separators=(chunking_config or {}).get("separators"),
            )
            chunks = [self._TextChunk(t) for t in recursive_texts]
        else:
            # Default: chunk by title preserves document structure
            chunks = chunk_by_title(
                elements,
                max_characters=self.chunk_max_chars,
                overlap=self.chunk_overlap,
            )

        result = []
        seen_hashes: set[str] = set()
        skip_stats: dict[str, int] = {"too_short": 0, "no_alphanumeric": 0, "duplicate": 0}

        for i, chunk in enumerate(chunks):
            content = str(chunk)

            # Validate chunk quality
            is_valid, skip_reason = validate_chunk(content)
            if not is_valid:
                skip_stats[skip_reason] = skip_stats.get(skip_reason, 0) + 1
                continue

            # Content-hash deduplication
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in seen_hashes:
                skip_stats["duplicate"] += 1
                continue
            seen_hashes.add(content_hash)

            # Create unique ID from content hash + source + index
            hash_input = f"{source_id}:{i}:{content[:100]}"
            chunk_id = hashlib.md5(hash_input.encode()).hexdigest()[:16]

            # Extract Unstructured metadata
            element_meta = chunk.metadata.to_dict() if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "to_dict") else {}
            metadata = {
                "element_type": type(chunk).__name__,
                "chunk_index": i,
                "page_number": element_meta.get("page_number"),
                "parent_id": element_meta.get("parent_id"),
                "is_continuation": element_meta.get("is_continuation"),
                "content_hash": content_hash,
            }

            result.append({
                "id": chunk_id,
                "content": content,
                "metadata": metadata,
            })

        total_skipped = sum(skip_stats.values())
        if total_skipped:
            logger.info(f"Skipped {total_skipped} chunks: {skip_stats}")
        logger.info(f"Created {len(result)} chunks")
        return result

    def create_embeddings(
        self, chunks: list[dict], batch_size: Optional[int] = None,
        context_strings: Optional[list[str]] = None,
    ) -> list[list[float]]:
        """
        Create embeddings for chunks with memory-efficient batching.

        Args:
            chunks: List of chunk dictionaries
            batch_size: Override batch size (default from env or DEFAULT_EMBED_BATCH_SIZE)
            context_strings: Optional list of context prefixes (one per chunk) to prepend
                before embedding. Used for contextual embeddings — the prefix is NOT
                stored in ChromaDB or BM25, only affects the embedding vector.

        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []

        if context_strings is not None and len(context_strings) != len(chunks):
            raise ValueError(
                f"context_strings length ({len(context_strings)}) must match "
                f"chunks length ({len(chunks)})"
            )

        # Get batch size from env var or use default
        if batch_size is None:
            batch_size = int(os.environ.get("TROVE_EMBED_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE))

        # Build text for embedding: optional context prefix + passage prefix + content
        texts = []
        for i, c in enumerate(chunks):
            text = c["content"]
            if context_strings and context_strings[i]:
                text = context_strings[i] + text
            if self.passage_prefix:
                text = self.passage_prefix + text
            texts.append(text)

        # Try progressively smaller GPU batches before falling back to CPU.
        # Uses a global cache so the settled size persists across Ingestor instances.
        global _global_effective_batch_size

        # Start from globally cached size if available, otherwise from requested size
        current_batch_size = _global_effective_batch_size or batch_size
        while current_batch_size >= 32:
            try:
                _clear_gpu_memory()
                embeddings = self.embedder.encode(
                    texts,
                    show_progress_bar=True,
                    batch_size=current_batch_size,
                )
                # Cache the successful batch size globally for all future calls
                if _global_effective_batch_size != current_batch_size:
                    if current_batch_size != batch_size:
                        logger.info(f"GPU batch size settled at {current_batch_size}")
                    _global_effective_batch_size = current_batch_size
                return embeddings.tolist()
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                    next_size = current_batch_size // 2
                    if next_size >= 32:
                        logger.warning(
                            f"GPU OOM with batch_size={current_batch_size}, "
                            f"retrying with batch_size={next_size}..."
                        )
                        _clear_gpu_memory()
                        current_batch_size = next_size
                    else:
                        break  # Fall through to CPU
                else:
                    raise

        # All GPU batch sizes exhausted — fall back to CPU
        logger.warning(f"GPU OOM at all batch sizes, falling back to CPU (batch_size={batch_size})...")
        _clear_gpu_memory()

        original_device = self.embedder.device
        self.embedder = self.embedder.to("cpu")

        try:
            embeddings = self.embedder.encode(
                texts,
                show_progress_bar=True,
                batch_size=batch_size,
            )
            return embeddings.tolist()
        finally:
            # Try to restore GPU if it was originally on GPU
            if str(original_device) != "cpu" and torch.cuda.is_available():
                _clear_gpu_memory()
                try:
                    self.embedder = self.embedder.to(original_device)
                except RuntimeError:
                    logger.warning("Could not restore model to GPU, keeping on CPU")

    def ingest_document(
        self, path: str | Path, contextualize: bool = False,
    ) -> tuple[list[dict], list[list[float]]]:
        """
        Full ingestion pipeline: parse → chunk → embed.

        Args:
            path: Path to document
            contextualize: If True, prepend document context to chunks before embedding

        Returns:
            Tuple of (chunks, embeddings)
        """
        path = Path(path)
        elements = self.parse_document(path)
        chunks = self.chunk_elements(elements, source_id=str(path))

        # Add source file to metadata (can be overridden by ingest_directory for full path)
        for chunk in chunks:
            chunk["metadata"]["source_file"] = path.name
            chunk["metadata"]["source_path"] = str(path)

        # Build context strings for contextual embeddings
        context_strings = None
        if contextualize:
            context_strings = [build_context_prefix(c["metadata"]) for c in chunks]
            for chunk in chunks:
                chunk["metadata"]["contextualized"] = True

        embeddings = self.create_embeddings(chunks, context_strings=context_strings)

        return chunks, embeddings

    def ingest_mbox(self, path: str | Path) -> tuple[list[dict], list[list[float]]]:
        """
        Ingest an mbox email archive file.

        Each email becomes one or more chunks depending on length.
        Email metadata (subject, from, to, date) is preserved.

        Args:
            path: Path to mbox file

        Returns:
            Tuple of (chunks, embeddings)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Mbox file not found: {path}")

        logger.info(f"Parsing mbox file: {path.name}")

        mbox = mailbox.mbox(str(path))
        chunks = []
        email_count = 0
        skipped_count = 0

        for message in mbox:
            email_count += 1

            # Extract headers
            subject = _decode_email_header(message.get("Subject", ""))
            from_addr = _decode_email_header(message.get("From", ""))
            to_addr = _decode_email_header(message.get("To", ""))
            date_str = message.get("Date", "")
            message_id = message.get("Message-ID", "")

            # Parse date
            date_formatted = ""
            if date_str:
                try:
                    dt = parsedate_to_datetime(date_str)
                    date_formatted = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date_formatted = date_str

            # Extract body
            body = _extract_email_body(message)

            if not body or len(body.strip()) < 10:
                skipped_count += 1
                continue

            # Build email content with headers for context
            header_block = []
            if subject:
                header_block.append(f"Subject: {subject}")
            if from_addr:
                header_block.append(f"From: {from_addr}")
            if to_addr:
                header_block.append(f"To: {to_addr}")
            if date_formatted:
                header_block.append(f"Date: {date_formatted}")

            header_text = "\n".join(header_block)
            full_content = f"{header_text}\n\n{body}" if header_text else body

            # Split into chunks if content exceeds max size
            content_chunks = self._split_email_content(full_content, header_text)

            for i, content in enumerate(content_chunks):
                # Create unique ID
                hash_input = f"{path}:{message_id or email_count}:{i}:{content[:50]}"
                chunk_id = hashlib.md5(hash_input.encode()).hexdigest()[:16]

                chunks.append({
                    "id": chunk_id,
                    "content": content,
                    "metadata": {
                        "element_type": "Email",
                        "source_file": path.name,
                        "source_path": str(path),
                        "email_subject": subject,
                        "email_from": from_addr,
                        "email_to": to_addr,
                        "email_date": date_formatted,
                        "email_message_id": message_id,
                        "chunk_index": i,
                        "is_email": True,
                    }
                })

        mbox.close()

        logger.info(f"Parsed {email_count} emails from {path.name} ({skipped_count} skipped, {len(chunks)} chunks)")

        if not chunks:
            return [], []

        embeddings = self.create_embeddings(chunks)
        return chunks, embeddings

    def ingest_mbox_chunks_only(self, path: str | Path) -> list[dict]:
        """
        Parse an mbox email archive into chunks WITHOUT creating embeddings.

        Uses the same parsing logic as ingest_mbox but returns just the chunks.
        Intended for use with IngestionPipeline which handles embedding in batches.

        Args:
            path: Path to mbox file

        Returns:
            List of chunk dicts with id, content, and metadata
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Mbox file not found: {path}")

        logger.info(f"Parsing mbox file (chunks only): {path.name}")

        mbox = mailbox.mbox(str(path))
        chunks = []
        email_count = 0
        skipped_count = 0

        for message in mbox:
            email_count += 1

            # Extract headers
            subject = _decode_email_header(message.get("Subject", ""))
            from_addr = _decode_email_header(message.get("From", ""))
            to_addr = _decode_email_header(message.get("To", ""))
            date_str = message.get("Date", "")
            message_id = message.get("Message-ID", "")

            # Parse date
            date_formatted = ""
            if date_str:
                try:
                    dt = parsedate_to_datetime(date_str)
                    date_formatted = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date_formatted = date_str

            # Extract body
            body = _extract_email_body(message)

            if not body or len(body.strip()) < 10:
                skipped_count += 1
                continue

            # Build email content with headers for context
            header_block = []
            if subject:
                header_block.append(f"Subject: {subject}")
            if from_addr:
                header_block.append(f"From: {from_addr}")
            if to_addr:
                header_block.append(f"To: {to_addr}")
            if date_formatted:
                header_block.append(f"Date: {date_formatted}")

            header_text = "\n".join(header_block)
            full_content = f"{header_text}\n\n{body}" if header_text else body

            # Split into chunks if content exceeds max size
            content_chunks = self._split_email_content(full_content, header_text)

            for i, content in enumerate(content_chunks):
                # Create unique ID
                hash_input = f"{path}:{message_id or email_count}:{i}:{content[:50]}"
                chunk_id = hashlib.md5(hash_input.encode()).hexdigest()[:16]

                chunks.append({
                    "id": chunk_id,
                    "content": content,
                    "metadata": {
                        "element_type": "Email",
                        "source_file": path.name,
                        "source_path": str(path),
                        "email_subject": subject,
                        "email_from": from_addr,
                        "email_to": to_addr,
                        "email_date": date_formatted,
                        "email_message_id": message_id,
                        "chunk_index": i,
                        "is_email": True,
                    }
                })

        mbox.close()

        logger.info(f"Parsed {email_count} emails from {path.name} ({skipped_count} skipped, {len(chunks)} chunks)")

        return chunks

    def _split_email_content(self, content: str, header_text: str) -> list[str]:
        """
        Split email content into chunks if it exceeds max size.

        Args:
            content: Full email content including headers
            header_text: Just the header block (to prepend to continuation chunks)

        Returns:
            List of content chunks
        """
        if len(content) <= self.chunk_max_chars:
            return [content]

        chunks = []
        # First chunk includes full headers
        chunks.append(content[:self.chunk_max_chars])

        # Subsequent chunks get abbreviated header reference
        remaining = content[self.chunk_max_chars - self.chunk_overlap:]
        continuation_prefix = f"[continued] {header_text.split(chr(10))[0]}\n\n" if header_text else "[continued]\n\n"

        while remaining:
            chunk_size = self.chunk_max_chars - len(continuation_prefix)
            chunk_content = continuation_prefix + remaining[:chunk_size]
            chunks.append(chunk_content)
            remaining = remaining[chunk_size - self.chunk_overlap:] if len(remaining) > chunk_size else ""

        return chunks


def ingest_directory(
    directory: str | Path,
    ingestor: Optional[DocumentIngestor] = None,
    extensions: tuple = (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".html", ".txt", ".md", ".mbox"),
    recursive: bool = True,
    progress_callback: Optional[Callable] = None,
    resume_from: Optional["IngestionCheckpoint"] = None,
    exclude_patterns: Optional[list[str]] = None,
    pipeline_searcher=None,
    parse_workers: int = 4,
    contextualize: bool = False,
) -> tuple[list[dict], list[list[float]]]:
    """
    Ingest all documents in a directory.

    Args:
        directory: Path to directory
        ingestor: DocumentIngestor instance (creates default if None)
        extensions: File extensions to process (includes .mbox for email archives)
        recursive: Search subdirectories recursively (default True)
        progress_callback: Optional callback(IngestionProgress) called after each file
        resume_from: Optional checkpoint to resume from (skips already-processed files)
        exclude_patterns: Optional glob patterns to exclude (e.g., ["**/clients/**"])
        pipeline_searcher: When provided, use IngestionPipeline for parallel parse/embed/index
        parse_workers: Number of parallel parse workers for pipeline mode (default 4)
        contextualize: If True, prepend document context to chunks before embedding (pipeline mode)

    Returns:
        Tuple of (all_chunks, all_embeddings)
    """
    import fnmatch
    from core.ingest_progress import IngestionProgress

    if ingestor is None:
        ingestor = DocumentIngestor()

    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    exclude_patterns = exclude_patterns or []

    def _should_exclude(path: Path) -> bool:
        """Check if path matches any exclude pattern."""
        path_str = str(path)
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
        return False

    # Find all matching files (recursively or not), following symlinks
    if recursive:
        files = []
        seen_real = set()
        for root, dirs, filenames in os.walk(directory, followlinks=True):
            # Cycle detection for symlinks
            real_root = os.path.realpath(root)
            if real_root in seen_real:
                dirs.clear()
                continue
            seen_real.add(real_root)
            for fname in filenames:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in extensions:
                    if _should_exclude(fpath):
                        continue
                    try:
                        fpath.stat()  # Skip broken symlinks
                        files.append(fpath)
                    except (OSError, FileNotFoundError):
                        continue
    else:
        files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in extensions]

    sorted_files = sorted(files)
    logger.info(f"Found {len(sorted_files)} documents to ingest")

    # --- Pipeline mode: parallel parse -> buffer -> embed -> index ---
    if pipeline_searcher is not None:
        from core.pipeline import IngestionPipeline

        pipeline = IngestionPipeline(
            ingestor=ingestor,
            searcher=pipeline_searcher,
            parse_workers=parse_workers,
            contextualize=contextualize,
        )

        # Build resume set: prefer checkpoint processed_files, fall back to ChromaDB query
        if resume_from:
            already_indexed = set(resume_from.processed_files)
        else:
            already_indexed = pipeline.get_indexed_source_paths()

        stats = pipeline.run_documents(
            sorted_files,
            base_directory=directory,
            already_indexed=already_indexed,
            progress_callback=progress_callback,
        )

        logger.info(
            f"Pipeline complete: {stats['total_chunks_indexed']} chunks indexed, "
            f"{stats['total_files_processed']} files processed, "
            f"{stats['flush_count']} batches, {stats['elapsed_seconds']}s "
            f"({stats['chunks_per_second']} chunks/s)"
        )
        if stats["failed_files"]:
            logger.warning(f"Pipeline failed files: {stats['failed_files']}")

        return [], []

    # --- Serial mode (backward compatible) ---
    all_chunks = []
    all_embeddings = []

    # Build set of already-processed files for resume
    processed_set: set[str] = set()
    if resume_from:
        processed_set = set(resume_from.processed_files)
        logger.info(f"Resuming: skipping {len(processed_set)} already-processed files")

    progress = IngestionProgress(total_files=len(sorted_files))

    for i, path in enumerate(sorted_files):
        # Skip already-processed files when resuming
        if str(path) in processed_set:
            progress.skipped_files += 1
            progress.processed_files += 1
            continue

        try:
            # Use specialized handler for mbox files
            if path.suffix.lower() == ".mbox":
                chunks, embeddings = ingestor.ingest_mbox(path)
            else:
                chunks, embeddings = ingestor.ingest_document(path)

            # Store relative path from documents root for better source tracking
            rel_path = path.relative_to(directory)
            for chunk in chunks:
                chunk["metadata"]["source_file"] = str(rel_path)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)

            progress.processed_files += 1
            progress.total_chunks += len(chunks)

            # Log progress
            if (i + 1) % 5 == 0 or (i + 1) == len(sorted_files):
                logger.info(f"Progress: {progress.processed_files}/{progress.total_files} documents processed")

        except Exception as e:
            logger.error(f"Failed to ingest {path.name}: {e}")
            progress.processed_files += 1
            progress.failed_files.append(str(path))

        if progress_callback:
            progress_callback(progress)

    logger.info(f"Ingested {len(all_chunks)} total chunks from {directory}")
    return all_chunks, all_embeddings


def ingest_code(
    code_paths: list[str | Path],
    code_config: dict,
    ingestor: Optional[DocumentIngestor] = None,
    index_path: Optional[Path] = None,
    rebuild: bool = False,
    pipeline_searcher=None,
) -> tuple[list[dict], list[list[float]]]:
    """
    Ingest code files using tree-sitter parsing.

    Args:
        code_paths: List of directories/files to index
        code_config: Code configuration from client.yaml
        ingestor: DocumentIngestor instance for embeddings (creates default if None)
        index_path: Path to index directory for incremental state
        rebuild: If True, ignore incremental state and rebuild
        pipeline_searcher: When provided, use IngestionPipeline for batched embed/index

    Returns:
        Tuple of (all_chunks, all_embeddings)
    """
    from core.code import CodeChunker, CodeIndexState, compute_chunk_id

    if ingestor is None:
        ingestor = DocumentIngestor()

    # Resolve paths
    paths = [Path(p) for p in code_paths]

    # Initialize chunker
    chunker = CodeChunker(code_config)

    # Detect repositories BEFORE collecting files (enables git ls-files fast path)
    logger.info(f"Detecting git repositories in {len(paths)} paths...")
    repos = chunker.detect_repositories(paths)

    # Collect files (uses git ls-files for repos)
    logger.info("Collecting code files...")
    all_files = chunker.collect_files(paths, repos=repos)

    # Handle incremental indexing
    chunks_to_index = []
    stale_chunk_ids = []  # IDs to remove from ChromaDB before adding new chunks
    if code_config.get("incremental", True) and index_path and not rebuild:
        logger.info("Loading incremental index state...")
        state = CodeIndexState(index_path)
        logger.info(f"Loaded state for {len(state.state)} files, detecting changes across {len(all_files)} files...")
        modified_files, unchanged_files, deleted_files = state.get_changes(all_files)

        # Collect stale chunk IDs from deleted and modified files
        for deleted_path in deleted_files:
            old_ids = state.remove_file(deleted_path)
            stale_chunk_ids.extend(old_ids)

        for modified_file in modified_files:
            old_ids = state.get_chunk_ids_for_file(str(modified_file))
            stale_chunk_ids.extend(old_ids)

        # Get chunks only for modified files
        if modified_files:
            chunks_to_index = chunker.chunk_files(modified_files, repos)

        logger.info(
            f"Incremental code indexing: {len(modified_files)} files to process, "
            f"{len(unchanged_files)} unchanged, {len(deleted_files)} deleted"
        )
        if stale_chunk_ids:
            logger.info(f"Will remove {len(stale_chunk_ids)} stale chunks from index")
    else:
        # Full rebuild
        chunks_to_index = chunker.chunk_files(all_files, repos)
        if index_path:
            # Clear state for rebuild
            state = CodeIndexState(index_path)
            state.clear()

    if not chunks_to_index and not stale_chunk_ids:
        logger.info("No code changes to index")
        # Still record zero-chunk files in state so they aren't re-detected next run
        if code_config.get("incremental", True) and index_path and not rebuild:
            for f in modified_files:
                state.update_file(f, [])
            state.save_state()
            logger.info(f"Updated state for {len(modified_files)} zero-chunk files")
        return [], []

    # Generate IDs for chunks
    for chunk in chunks_to_index:
        chunk["id"] = compute_chunk_id(chunk)

    # Remove stale chunks from ChromaDB before adding new ones
    if stale_chunk_ids:
        # Deduplicate IDs (old state files may contain duplicates from previous bugs)
        stale_chunk_ids = list(dict.fromkeys(stale_chunk_ids))
        searcher = pipeline_searcher
        if searcher is not None:
            batch_size = 500
            removed = 0
            for i in range(0, len(stale_chunk_ids), batch_size):
                batch = stale_chunk_ids[i : i + batch_size]
                try:
                    existing = searcher.collection.get(ids=batch, include=[])
                    if existing and existing.get("ids"):
                        searcher.collection.delete(ids=existing["ids"])
                        removed += len(existing["ids"])
                except Exception as e:
                    logger.warning(f"Failed to delete stale chunks batch: {e}")
            if removed:
                logger.info(f"Removed {removed} stale chunks from index")

    if not chunks_to_index:
        logger.info("No new code chunks to index (deletions only)")
        # Save state for deletions + zero-chunk files
        if code_config.get("incremental", True) and index_path:
            for f in modified_files:
                file_str = str(f)
                # Only record files not already in chunks_by_file
                if file_str not in state.state or not state.state[file_str].chunk_ids:
                    state.update_file(f, [])
            state.save_state()
        return [], []

    # --- Pipeline mode: buffer -> embed -> index progressively ---
    if pipeline_searcher is not None:
        from core.pipeline import IngestionPipeline

        pipeline = IngestionPipeline(
            ingestor=ingestor,
            searcher=pipeline_searcher,
        )

        stats = pipeline.run_code(chunks_to_index)

        logger.info(
            f"Code pipeline complete: {stats['total_chunks_indexed']} chunks indexed, "
            f"{stats['flush_count']} batches, {stats['elapsed_seconds']}s "
            f"({stats['chunks_per_second']} chunks/s)"
        )

        # Update incremental state if enabled
        if code_config.get("incremental", True) and index_path:
            state = CodeIndexState(index_path)

            # Group chunks by source file
            chunks_by_file: dict[str, list[str]] = {}
            for chunk in chunks_to_index:
                source = chunk.get("metadata", {}).get("source_file", "")
                if source:
                    if source not in chunks_by_file:
                        chunks_by_file[source] = []
                    chunks_by_file[source].append(chunk["id"])

            # Update state for each file (including zero-chunk files)
            for file_path in all_files:
                file_str = str(file_path)
                matched = False
                for source, chunk_ids in chunks_by_file.items():
                    if file_str.endswith(source) or source in file_str:
                        state.update_file(file_path, chunk_ids)
                        matched = True
                        break
                if not matched:
                    # Record zero-chunk files so they aren't re-detected next run
                    state.update_file(file_path, [])

            state.save_state()

        logger.info(f"Ingested {len(chunks_to_index)} code chunks via pipeline")
        return chunks_to_index, []

    # --- Serial mode (backward compatible) ---

    # Create embeddings
    logger.info(f"Creating embeddings for {len(chunks_to_index)} code chunks")
    embeddings = ingestor.create_embeddings(chunks_to_index)

    # Update incremental state if enabled
    if code_config.get("incremental", True) and index_path:
        state = CodeIndexState(index_path)

        # Group chunks by source file
        chunks_by_file: dict[str, list[str]] = {}
        for chunk in chunks_to_index:
            source = chunk.get("metadata", {}).get("source_file", "")
            if source:
                if source not in chunks_by_file:
                    chunks_by_file[source] = []
                chunks_by_file[source].append(chunk["id"])

        # Update state for each file
        for file_path in all_files:
            file_str = str(file_path)
            # Find relative path used in chunks
            for source, chunk_ids in chunks_by_file.items():
                if file_str.endswith(source) or source in file_str:
                    state.update_file(file_path, chunk_ids)
                    break

        state.save_state()

    logger.info(f"Ingested {len(chunks_to_index)} code chunks")
    return chunks_to_index, embeddings


def ingest_client_code(
    client,
    ingestor: Optional[DocumentIngestor] = None,
    rebuild: bool = False,
    pipeline_searcher=None,
) -> tuple[list[dict], list[list[float]]]:
    """
    Ingest code for a client based on its configuration.

    Args:
        client: Client instance with code configuration
        ingestor: DocumentIngestor instance (creates default if None)
        rebuild: If True, rebuild code index from scratch
        pipeline_searcher: When provided, use IngestionPipeline for batched embed/index

    Returns:
        Tuple of (all_chunks, all_embeddings)
    """
    code_config = client.config.get("code", {})

    if not code_config.get("enabled", False):
        logger.info(f"Code indexing not enabled for client {client.name}")
        return [], []

    # Get code paths (relative to client directory or absolute)
    raw_paths = code_config.get("paths", [])
    code_paths = []

    for path in raw_paths:
        p = Path(path)
        if p.is_absolute():
            code_paths.append(p)
        else:
            # Relative to client directory
            code_paths.append(client.path / path)

    if not code_paths:
        logger.info(f"No code paths configured for client {client.name}")
        return [], []

    logger.info(f"Ingesting code for client {client.name} from {len(code_paths)} paths")

    return ingest_code(
        code_paths=code_paths,
        code_config=code_config,
        ingestor=ingestor,
        index_path=client.index_path,
        rebuild=rebuild,
        pipeline_searcher=pipeline_searcher,
    )


# --- File manifest for incremental reindexing ---

MANIFEST_FILENAME = "file_manifest.json"


def _manifest_path(client) -> Path:
    """Get the manifest file path for a client."""
    return client.index_path / MANIFEST_FILENAME


def build_file_manifest(
    directory: Path,
    extensions: tuple,
    recursive: bool = True,
    exclude_patterns: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Walk a directory and build a manifest of files with their mtime and size.

    Args:
        directory: Directory to walk
        extensions: File extensions to include
        recursive: Whether to recurse into subdirectories
        exclude_patterns: Optional glob patterns to exclude (e.g., ["**/clients/**"])

    Returns:
        Dict of {absolute_path: {"mtime": float, "size": int}}
    """
    import fnmatch
    exclude_patterns = exclude_patterns or []

    def _excluded(path: Path) -> bool:
        path_str = str(path)
        return any(fnmatch.fnmatch(path_str, p) for p in exclude_patterns)

    manifest = {}
    if not directory.exists():
        return manifest

    if recursive:
        seen_real = set()
        for root, dirs, filenames in os.walk(directory, followlinks=True):
            real_root = os.path.realpath(root)
            if real_root in seen_real:
                dirs.clear()
                continue
            seen_real.add(real_root)
            for fname in filenames:
                f = Path(root) / fname
                if f.suffix.lower() in extensions and not f.name.startswith("."):
                    if _excluded(f):
                        continue
                    try:
                        stat = f.stat()
                        manifest[str(f)] = {
                            "mtime": stat.st_mtime,
                            "size": stat.st_size,
                        }
                    except (OSError, FileNotFoundError):
                        continue  # Skip broken symlinks
    else:
        for f in directory.iterdir():
            if f.is_file() and f.suffix.lower() in extensions and not f.name.startswith("."):
                if _excluded(f):
                    continue
                stat = f.stat()
                manifest[str(f)] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }
    return manifest


def save_file_manifest(client, manifest: dict[str, dict]) -> None:
    """
    Save file manifest to the client's index directory.

    Args:
        client: Client instance
        manifest: Dict of {path: {"mtime": float, "size": int}}
    """
    path = _manifest_path(client)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "files": manifest,
        "indexed_at": datetime.now().isoformat(),
        "client": client.machine_name,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved file manifest: {len(manifest)} files for {client.machine_name}")


def load_file_manifest(client) -> dict[str, dict]:
    """
    Load file manifest from the client's index directory.

    Args:
        client: Client instance

    Returns:
        Dict of {path: {"mtime": float, "size": int}}, empty if no manifest exists
    """
    path = _manifest_path(client)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("files", {})
    except Exception as e:
        logger.warning(f"Failed to load manifest for {client.machine_name}: {e}")
        return {}


def diff_file_manifest(
    current_files: dict[str, dict],
    indexed_files: dict[str, dict],
) -> tuple[list[str], list[str], list[str]]:
    """
    Compare current files on disk against the indexed manifest.

    Args:
        current_files: Files currently on disk {path: {"mtime", "size"}}
        indexed_files: Files in the manifest {path: {"mtime", "size"}}

    Returns:
        Tuple of (new_files, modified_files, deleted_files) as lists of paths
    """
    current_set = set(current_files.keys())
    indexed_set = set(indexed_files.keys())

    new_files = sorted(current_set - indexed_set)
    deleted_files = sorted(indexed_set - current_set)

    modified_files = []
    for path in sorted(current_set & indexed_set):
        cur = current_files[path]
        idx = indexed_files[path]
        if cur["mtime"] != idx["mtime"] or cur["size"] != idx["size"]:
            modified_files.append(path)

    return new_files, modified_files, deleted_files
