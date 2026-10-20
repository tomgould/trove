"""
Alternative chunking strategies.

Provides:
- Title-based chunking (default, via Unstructured)
- Semantic chunking (split by embedding similarity drops)
- Recursive chunking (split by separator priority with fallback)
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy."""

    strategy: str = "title"  # "title", "semantic", or "recursive"
    max_chunk_size: int = 1500
    overlap: int = 100
    semantic_threshold: float = 0.5
    min_chunk_size: int = 100
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])


def _split_to_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics."""
    import re
    # Split on sentence-ending punctuation followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings
    return [s.strip() for s in parts if s.strip()]


def chunk_by_semantic(
    elements: list,
    embedder,
    threshold: float = 0.5,
    max_chars: int = 1500,
    min_chars: int = 100,
) -> list[str]:
    """
    Split elements into chunks based on embedding similarity.

    1. Flatten elements to sentences
    2. Embed each sentence
    3. Compute cosine similarity between consecutive sentences
    4. Split where similarity drops below threshold
    5. Merge short splits up to max_chars

    Args:
        elements: Unstructured elements (or list of strings)
        embedder: SentenceTransformer instance
        threshold: Similarity threshold for splitting (0-1)
        max_chars: Maximum characters per chunk
        min_chars: Minimum characters per chunk (merge if shorter)

    Returns:
        List of chunk text strings
    """
    # Flatten elements to sentences
    all_sentences: list[str] = []
    for elem in elements:
        text = str(elem).strip()
        if not text:
            continue
        sentences = _split_to_sentences(text)
        all_sentences.extend(sentences)

    if not all_sentences:
        return []

    if len(all_sentences) == 1:
        return all_sentences

    # Embed all sentences
    embeddings = embedder.encode(all_sentences, show_progress_bar=False)

    # Compute cosine similarity between consecutive sentences
    splits: list[int] = []  # indices where we should split
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
        )
        if sim < threshold:
            splits.append(i + 1)

    # Build raw groups from split points
    groups: list[list[str]] = []
    prev = 0
    for split_idx in splits:
        groups.append(all_sentences[prev:split_idx])
        prev = split_idx
    groups.append(all_sentences[prev:])

    # Merge groups into chunks respecting max_chars and min_chars
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for group in groups:
        group_text = " ".join(group)
        group_len = len(group_text)

        if current_len + group_len + 1 > max_chars and current:
            # Flush current chunk
            chunks.append(" ".join(current))
            current = []
            current_len = 0

        # If single group exceeds max_chars, split it at max_chars boundary
        if group_len > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            # Hard split
            text = group_text
            while len(text) > max_chars:
                chunks.append(text[:max_chars])
                text = text[max_chars:]
            if text:
                current = [text]
                current_len = len(text)
        else:
            current.extend(group)
            current_len += group_len + (1 if current_len > 0 else 0)

    if current:
        chunk_text = " ".join(current)
        # If too short, append to previous chunk if possible
        if len(chunk_text) < min_chars and chunks:
            chunks[-1] += " " + chunk_text
        else:
            chunks.append(chunk_text)

    logger.info(f"Semantic chunking: {len(all_sentences)} sentences -> {len(chunks)} chunks")
    return chunks


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_by_recursive(
    text: str,
    max_chars: int = 1500,
    overlap: int = 100,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursively split text using separator priority.

    Tries separators in order. If splitting on the first separator produces
    pieces that still exceed max_chars, recursively splits those pieces
    using the next separator in the list.

    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        overlap: Character overlap between consecutive chunks
        separators: Ordered list of separators to try (default: paragraph, line, sentence, word)

    Returns:
        List of chunk text strings
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    if separators is None:
        separators = DEFAULT_SEPARATORS

    chunks = _recursive_split(text, max_chars, separators, 0)

    # Apply overlap between consecutive chunks
    if overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap, max_chars)

    logger.info(f"Recursive chunking: {len(text)} chars -> {len(chunks)} chunks")
    return chunks


def _recursive_split(
    text: str,
    max_chars: int,
    separators: list[str],
    sep_index: int,
) -> list[str]:
    """Inner recursive splitting logic."""
    if not text.strip():
        return []

    if len(text) <= max_chars:
        return [text]

    # If we've exhausted all separators, hard-split at max_chars
    if sep_index >= len(separators):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start = end
        return chunks

    sep = separators[sep_index]
    pieces = text.split(sep)

    # If separator doesn't split anything, try next
    if len(pieces) <= 1:
        return _recursive_split(text, max_chars, separators, sep_index + 1)

    # Merge small pieces and recursively split large ones
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        candidate = (current + sep + piece).strip() if current else piece

        if len(candidate) <= max_chars:
            current = candidate
        else:
            # Flush current
            if current:
                chunks.append(current)

            # If this piece alone exceeds max_chars, recurse with next separator
            if len(piece) > max_chars:
                sub_chunks = _recursive_split(piece, max_chars, separators, sep_index + 1)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = piece

    if current:
        chunks.append(current)

    return chunks


def _apply_overlap(chunks: list[str], overlap: int, max_chars: int = 0) -> list[str]:
    """Apply character overlap between consecutive chunks.

    Args:
        chunks: List of chunk strings
        overlap: Number of characters to overlap from previous chunk
        max_chars: If > 0, trim overlapped chunks to this limit
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        overlap_text = prev[-overlap:] if len(prev) > overlap else prev
        merged = overlap_text + " " + chunks[i]
        if max_chars > 0 and len(merged) > max_chars:
            merged = merged[:max_chars]
        result.append(merged)

    return result
