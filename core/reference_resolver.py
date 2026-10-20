"""
Reference Resolution for RAG.

Detects when retrieved chunks reference other sections/appendices and
automatically fetches that content to provide complete answers.

Supports configurable reference patterns for different document domains
(legal, technical, academic, compliance, etc.).
"""

import logging
import re
import time
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.reference_config import ReferenceConfig

logger = logging.getLogger(__name__)

# =============================================================================
# Reference Resolution Cache
# =============================================================================

# Session-level cache for resolved references
# Key: reference key (e.g., "appendix:H"), Value: list of resolved chunks
_reference_cache: dict[str, list[dict]] = {}


def clear_reference_cache() -> None:
    """Clear the reference resolution cache. Call when switching clients or documents."""
    global _reference_cache
    _reference_cache = {}
    logger.info("Reference resolution cache cleared")


def get_cache_stats() -> dict:
    """Get cache statistics for debugging."""
    return {
        "cached_references": len(_reference_cache),
        "total_cached_chunks": sum(len(v) for v in _reference_cache.values()),
    }


# =============================================================================
# Reference Detection Patterns
# =============================================================================

# Patterns that indicate a reference to another section
REFERENCE_PATTERNS = [
    # Appendix references
    r'\b(?:see\s+)?(?:Appendix|Annex|Annexure)\s+([A-Z0-9]+(?:\.[0-9]+)?)\b',
    r'\b(?:refer\s+to\s+)?(?:Appendix|Annex|Annexure)\s+([A-Z0-9]+(?:\.[0-9]+)?)\b',
    r'\bas\s+(?:set\s+out|listed|detailed|described)\s+in\s+(?:Appendix|Annex)\s+([A-Z0-9]+)\b',

    # Section references
    r'\b(?:see\s+)?Section\s+(\d+(?:\.\d+)*)\b',
    r'\b(?:refer\s+to\s+)?Section\s+(\d+(?:\.\d+)*)\b',

    # Schedule references
    r'\b(?:see\s+)?Schedule\s+(\d+|[A-Z])\b',
    r'\b(?:refer\s+to\s+)?Schedule\s+(\d+|[A-Z])\b',

    # Table/Figure references
    r'\b(?:see\s+)?Table\s+(\d+(?:\.\d+)?|[A-Z])\b',
    r'\b(?:see\s+)?Figure\s+(\d+(?:\.\d+)?|[A-Z])\b',

    # Chapter references
    r'\b(?:see\s+)?Chapter\s+(\d+)\b',

    # Part references
    r'\b(?:see\s+)?Part\s+(\d+|[A-Z]+)\b',
]

# Patterns for detecting appendix/section headers in chunk content
HEADER_PATTERNS = [
    r'^Appendix\s+([A-Z0-9]+)',
    r'^Annex\s+([A-Z0-9]+)',
    r'^Schedule\s+(\d+|[A-Z])',
    r'^Section\s+(\d+(?:\.\d+)*)',
    r'^Chapter\s+(\d+)',
    r'^Part\s+(\d+|[A-Z]+)',
    r'Appendix\s+([A-Z0-9]+)\s*[-:–]',
    r'Annex\s+([A-Z0-9]+)\s*[-:–]',
]


def detect_references(
    text: str,
    config: "ReferenceConfig" = None,
) -> list[dict]:
    """
    Detect references to other sections in text.

    Args:
        text: Text to scan for references
        config: Optional ReferenceConfig for custom patterns

    Returns:
        List of detected references with type and identifier
    """
    # Use config if provided, otherwise fall back to built-in patterns
    if config is not None:
        return config.detect_references(text)

    # Legacy detection using built-in patterns
    references = []
    seen = set()

    for pattern in REFERENCE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            ref_id = match.group(1).upper()

            # Determine reference type from pattern
            pattern_lower = pattern.lower()
            if 'appendix' in pattern_lower or 'annex' in pattern_lower:
                ref_type = 'appendix'
            elif 'section' in pattern_lower:
                ref_type = 'section'
            elif 'schedule' in pattern_lower:
                ref_type = 'schedule'
            elif 'table' in pattern_lower:
                ref_type = 'table'
            elif 'figure' in pattern_lower:
                ref_type = 'figure'
            elif 'chapter' in pattern_lower:
                ref_type = 'chapter'
            elif 'part' in pattern_lower:
                ref_type = 'part'
            else:
                ref_type = 'unknown'

            key = f"{ref_type}:{ref_id}"
            if key not in seen:
                seen.add(key)
                references.append({
                    'type': ref_type,
                    'id': ref_id,
                    'match': match.group(0),
                    'key': key,
                })

    return references


def detect_references_in_chunks(
    chunks: list[dict],
    config: "ReferenceConfig" = None,
) -> list[dict]:
    """
    Scan multiple chunks for references.

    Args:
        chunks: List of chunk dicts with 'content' field
        config: Optional ReferenceConfig for custom patterns

    Returns:
        Deduplicated list of all references found, prioritized by importance
    """
    all_refs = []
    seen_keys = set()

    for chunk in chunks:
        content = chunk.get('content', '')
        refs = detect_references(content, config)

        for ref in refs:
            if ref['key'] not in seen_keys:
                seen_keys.add(ref['key'])
                all_refs.append(ref)

    # Prioritize and filter references (config-based refs already have priority)
    if config is None:
        all_refs = _prioritize_references(all_refs)

    return all_refs


def _prioritize_references(references: list[dict]) -> list[dict]:
    """
    Prioritize and filter references.

    - Appendices/Schedules are high priority (contain actual data)
    - Chapters are low priority (usually just narrative sections)
    - If too many references, filter to high priority only
    """
    HIGH_PRIORITY_TYPES = {'appendix', 'schedule', 'table'}
    LOW_PRIORITY_TYPES = {'chapter', 'part'}

    # Separate by priority
    high_priority = [r for r in references if r['type'] in HIGH_PRIORITY_TYPES]
    medium_priority = [r for r in references if r['type'] not in HIGH_PRIORITY_TYPES and r['type'] not in LOW_PRIORITY_TYPES]
    low_priority = [r for r in references if r['type'] in LOW_PRIORITY_TYPES]

    # If we have high priority refs, focus on those
    if high_priority:
        # Include some medium priority but limit low priority
        result = high_priority + medium_priority[:2]
        logger.info(f"Prioritized {len(result)} references (high={len(high_priority)}, filtered {len(low_priority)} low-priority)")
        return result

    # Otherwise include all but limit chapters
    result = medium_priority + low_priority[:3]
    return result


def build_reference_queries(
    references: list[dict],
    context: str = "",
    config: "ReferenceConfig" = None,
) -> list[str]:
    """
    Build search queries to find referenced content.

    Args:
        references: List of reference dicts from detect_references
        context: Additional context from the original text (helps find content
                 when the reference ID isn't in the target document)
        config: Optional ReferenceConfig for custom query templates

    Returns:
        List of search queries
    """
    queries = []

    for ref in references:
        ref_type = ref['type']
        ref_id = ref['id']

        # Use config for query building if available
        if config is not None:
            config_queries = config.build_queries_for_reference(ref)
            queries.extend(config_queries)
        else:
            # Legacy query building
            if ref_type == 'appendix':
                queries.extend([
                    f"Appendix {ref_id}",
                    f"Annex {ref_id}",
                ])
            elif ref_type == 'section':
                queries.append(f"Section {ref_id}")
            elif ref_type == 'schedule':
                queries.append(f"Schedule {ref_id}")
            elif ref_type == 'table':
                queries.append(f"Table {ref_id}")
            elif ref_type == 'chapter':
                queries.append(f"Chapter {ref_id}")
            else:
                queries.append(f"{ref_type.title()} {ref_id}")

    # If context provided, extract semantic queries from it
    # This helps find content when the target document doesn't mention the reference ID
    if context:
        semantic_queries = _extract_semantic_queries(context, config)
        queries.extend(semantic_queries)

    return list(set(queries))  # Deduplicate


def _extract_reference_context(chunks: list[dict], reference: dict) -> str:
    """
    Extract context text around a reference from chunks.

    Finds chunks that mention the reference and extracts surrounding text
    to understand what the reference is about.

    Args:
        chunks: List of chunks that might mention the reference
        reference: Reference dict with 'match' field

    Returns:
        Context text (up to 200 chars around the reference)
    """
    ref_match = reference.get('match', '').lower()
    if not ref_match:
        return ""

    for chunk in chunks:
        content = chunk.get('content', '')
        content_lower = content.lower()

        # Find the reference in the content
        pos = content_lower.find(ref_match)
        if pos >= 0:
            # Extract context: 50 chars before and 150 chars after
            start = max(0, pos - 50)
            end = min(len(content), pos + len(ref_match) + 150)
            context = content[start:end]
            logger.debug(f"Extracted context for {ref_match}: '{context[:60]}...'")
            return context

    return ""


def _extract_semantic_queries(
    context: str,
    config: "ReferenceConfig" = None,
) -> list[str]:
    """
    Extract semantic search queries from context text.

    Looks for meaningful noun phrases that describe the referenced content.
    Uses patterns common in document references like "high risk countries",
    "due diligence requirements", etc.
    """
    if not context or len(context) < 10:
        return []

    queries = []
    context_lower = context.lower()

    # Common adjective-noun patterns that indicate data content
    # These patterns extract ACTUAL phrases from the context
    adjective_patterns = [
        r'\b(high[\s\-]?risk)\b',
        r'\b(low[\s\-]?risk)\b',
        r'\b(higher[\s\-]?risk)\b',
        r'\b(mandatory)\b',
        r'\b(enhanced)\b',
        r'\b(increased)\b',
    ]

    noun_patterns = [
        r'\b(countr(?:y|ies))\b',
        r'\b(jurisdiction[s]?)\b',
        r'\b(territor(?:y|ies))\b',
        r'\b(list[s]?)\b',
        r'\b(requirements?)\b',
    ]

    # Find adjectives and nouns that appear in context
    found_adjectives = []
    found_nouns = []

    for pattern in adjective_patterns:
        matches = re.findall(pattern, context_lower)
        found_adjectives.extend(matches)

    for pattern in noun_patterns:
        matches = re.findall(pattern, context_lower)
        found_nouns.extend(matches)

    # Build queries from combinations found in context
    if found_adjectives and found_nouns:
        # Use the first adjective-noun combination
        adj = found_adjectives[0].replace('-', ' ')
        noun = found_nouns[0]
        queries.append(f"{adj} {noun}")

    # Get compound phrases from config or use defaults
    if config is not None:
        compound_phrases = config.get_semantic_phrases()
    else:
        compound_phrases = [
            'high risk countries',
            'higher risk countries',
            'high risk jurisdictions',
            'due diligence',
            'country risk',
            'risk countries',
            'mandatory list',
        ]

    for phrase in compound_phrases:
        # Check if the key words appear near each other in context
        words = phrase.split()
        if all(w in context_lower for w in words):
            if phrase not in queries:
                queries.append(phrase)

    # Deduplicate
    seen = set()
    result = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            result.append(q)

    logger.debug(f"Extracted semantic queries from context: {result}")
    return result[:3]


def is_reference_content(chunk: dict, reference: dict) -> bool:
    """
    Check if a chunk contains the actual content of a reference.

    This is intentionally lenient to avoid missing relevant content.
    We'd rather include some false positives than miss actual data.

    Args:
        chunk: Chunk dict with 'content' field
        reference: Reference dict with 'type' and 'id'

    Returns:
        True if chunk appears to contain the referenced content
    """
    content = chunk.get('content', '')
    content_lower = content.lower()
    ref_type = reference['type']
    ref_id = reference['id']
    ref_id_lower = ref_id.lower()

    # Check if chunk starts with or contains the reference header
    for pattern in HEADER_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if match.upper() == ref_id:
                logger.debug(f"Matched header pattern for {ref_type} {ref_id}")
                return True

    # Check metadata
    metadata = chunk.get('metadata', {})
    source_file = metadata.get('source_file', '').lower()
    parent_section = metadata.get('parent_section', '').lower()
    section_title = metadata.get('section_title', '').lower()

    # Check if source file name contains the reference
    if ref_type == 'appendix':
        if f'appendix {ref_id_lower}' in source_file or f'appendix_{ref_id_lower}' in source_file:
            logger.debug(f"Matched source file for {ref_type} {ref_id}: {source_file}")
            return True
        if f'appendix {ref_id_lower}' in parent_section or f'appendix {ref_id_lower}' in section_title:
            logger.debug(f"Matched section metadata for {ref_type} {ref_id}")
            return True

    # Check if the reference ID appears prominently in the content
    # (not just "see appendix H" but actually "Appendix H:" or "Appendix H -")
    ref_patterns = [
        f"{ref_type} {ref_id_lower}:",
        f"{ref_type} {ref_id_lower} -",
        f"{ref_type} {ref_id_lower} –",
        f"{ref_type} {ref_id_lower}\n",
    ]
    for pattern in ref_patterns:
        if pattern in content_lower:
            logger.debug(f"Matched content pattern '{pattern}' for {ref_type} {ref_id}")
            return True

    # NOTE: Removed score-based matching here as it caused too many false positives.
    # Semantic queries from context (e.g., "high risk countries") will handle finding
    # related data that doesn't explicitly mention the reference ID.

    return False


def _fetch_adjacent_chunks(
    found_chunks: list[dict],
    searcher,
    existing_ids: set,
    max_adjacent: int = 5,
) -> list[dict]:
    """
    Fetch chunks adjacent to found chunks from the same source file.

    This helps get complete appendix content that spans multiple chunks.
    Uses metadata (source_file, chunk_index) to find adjacent chunks efficiently.

    Args:
        found_chunks: Chunks already found for this reference
        searcher: HybridSearcher instance (must have bm25_chunks populated)
        existing_ids: Set of chunk IDs we already have
        max_adjacent: Maximum adjacent chunks to fetch

    Returns:
        List of additional adjacent chunks
    """
    if not found_chunks or not hasattr(searcher, 'bm25_chunks') or not searcher.bm25_chunks:
        return []

    # Group found chunks by source file
    chunks_by_source = {}
    for chunk in found_chunks:
        source = chunk.get('metadata', {}).get('source_file', '')
        if source:
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)

    # Build index of all chunks by source file
    all_chunks_by_source = {}
    for chunk in searcher.bm25_chunks:
        source = chunk.get('metadata', {}).get('source_file', '')
        if source:
            if source not in all_chunks_by_source:
                all_chunks_by_source[source] = []
            all_chunks_by_source[source].append(chunk)

    # Sort each source's chunks by index
    for source in all_chunks_by_source:
        all_chunks_by_source[source].sort(
            key=lambda c: c.get('metadata', {}).get('chunk_index', 0)
        )

    # Find adjacent chunks
    adjacent = []
    for source, found in chunks_by_source.items():
        if source not in all_chunks_by_source:
            continue

        all_source_chunks = all_chunks_by_source[source]

        # Get indices of found chunks
        found_indices = set()
        for chunk in found:
            idx = chunk.get('metadata', {}).get('chunk_index', -1)
            if idx >= 0:
                found_indices.add(idx)

        if not found_indices:
            continue

        # Find adjacent indices
        min_idx = min(found_indices)
        max_idx = max(found_indices)

        # Get chunks before and after
        for chunk in all_source_chunks:
            idx = chunk.get('metadata', {}).get('chunk_index', -1)
            if idx < 0:
                continue

            # Skip if already found
            if idx in found_indices:
                continue

            # Check if adjacent (within 2 positions)
            if min_idx - 2 <= idx <= max_idx + 2:
                chunk_id = hash(chunk.get('content', '')[:200])
                if chunk_id not in existing_ids:
                    adjacent.append(chunk)

            # Limit total adjacent chunks
            if len(adjacent) >= max_adjacent:
                break

        if len(adjacent) >= max_adjacent:
            break

    return adjacent[:max_adjacent]


def resolve_references(
    chunks: list[dict],
    searcher,
    top_k_per_ref: int = 30,
    max_refs_to_resolve: int = 5,
    scan_top_n_chunks: Optional[int] = 10,
    use_cache: bool = True,
    config: "ReferenceConfig" = None,
) -> tuple[list[dict], list[str]]:
    """
    Detect references in chunks and fetch the referenced content.

    IMPORTANT: For appendices, we fetch ALL matching chunks to ensure
    complete lists (like country lists) are not truncated.

    Args:
        chunks: Original retrieved chunks
        searcher: HybridSearcher instance with search() method
        top_k_per_ref: Number of results to fetch per reference
        max_refs_to_resolve: Maximum references to resolve
        scan_top_n_chunks: Only scan top N chunks for references (reduces noise)
        use_cache: Use session cache to avoid redundant searches
        config: Optional ReferenceConfig for custom patterns

    Returns:
        Tuple of (augmented_chunks, resolution_log)
    """
    global _reference_cache
    start_time = time.time()
    log = []

    # Check if reference resolution is disabled in config
    if config is not None and not config.enabled:
        log.append("Reference resolution disabled by config")
        return chunks, log

    # Only scan top N chunks for references to reduce noise from low-relevance chunks
    chunks_to_scan = chunks[:scan_top_n_chunks] if scan_top_n_chunks else chunks
    references = detect_references_in_chunks(chunks_to_scan, config)

    if scan_top_n_chunks and len(chunks) > scan_top_n_chunks:
        log.append(f"Scanned top {scan_top_n_chunks} chunks for references (of {len(chunks)} total)")

    if not references:
        log.append("No references detected in retrieved chunks")
        return chunks, log

    log.append(f"Detected {len(references)} references: {[r['match'] for r in references]}")

    # Limit number of references to resolve
    references = references[:max_refs_to_resolve]

    # Track which chunks we already have
    existing_ids = set()
    for chunk in chunks:
        chunk_id = hash(chunk.get('content', '')[:200])
        existing_ids.add(chunk_id)

    # Fetch content for each reference
    new_chunks = []
    resolved_refs = set()

    for ref in references:
        ref_start = time.time()
        ref_key = ref['key']

        # Check cache first
        if use_cache and ref_key in _reference_cache:
            cached_chunks = _reference_cache[ref_key]
            # Filter out chunks we already have
            for chunk in cached_chunks:
                chunk_id = hash(chunk.get('content', '')[:200])
                if chunk_id not in existing_ids:
                    new_chunks.append(chunk)
                    existing_ids.add(chunk_id)
            log.append(f"Cache hit for {ref['match']}: {len(cached_chunks)} cached chunks")
            resolved_refs.add(ref_key)
            continue

        # Extract context around the reference from the original chunks
        # This helps find content when target document doesn't mention the reference ID
        ref_context = _extract_reference_context(chunks, ref)

        queries = build_reference_queries([ref], context=ref_context, config=config)
        ref_chunks_found = []
        source_files_found = set()

        # Search with each query variant
        # Use fast_search if available (skips re-ranking for speed)
        search_fn = getattr(searcher, 'fast_search', searcher.search)

        # Separate queries into direct (Appendix H) and semantic (high risk countries)
        # Use word boundary matching to avoid false positives (e.g., "H" matching "high")
        def is_direct_query(query: str, ref_type: str, ref_id: str) -> bool:
            q_lower = query.lower()
            # Check if the reference type appears as a word
            if re.search(rf'\b{re.escape(ref_type)}\b', q_lower):
                return True
            # Check if the reference ID appears as a word (not as part of another word)
            # e.g., "H" should match "Appendix H" but not "high"
            if re.search(rf'\b{re.escape(ref_id.lower())}\b', q_lower):
                return True
            return False

        direct_queries = [q for q in queries if is_direct_query(q, ref['type'], ref['id'])]
        semantic_queries = [q for q in queries if q not in direct_queries]
        logger.debug(f"Queries for {ref['match']}: direct={direct_queries}, semantic={semantic_queries}")

        for query in queries[:4]:  # Allow up to 4 queries (2 direct + 2 semantic)
            is_semantic_query = query in semantic_queries

            try:
                ref_results = search_fn(query=query, top_k=top_k_per_ref)
                logger.debug(f"Query '{query}' returned {len(ref_results)} results")

                if not ref_results:
                    log.append(f"  No results for query: {query}")
                    continue

                matched_count = 0
                for i, chunk in enumerate(ref_results):
                    chunk_id = hash(chunk.get('content', '')[:200])

                    if chunk_id in existing_ids:
                        continue

                    # Check if this chunk contains the referenced content
                    if is_reference_content(chunk, ref):
                        chunk['_resolved_reference'] = ref['key']
                        chunk['_reference_priority'] = True
                        ref_chunks_found.append(chunk)
                        existing_ids.add(chunk_id)
                        source_files_found.add(chunk.get('metadata', {}).get('source_file', ''))
                        matched_count += 1
                        continue

                    # For semantic queries, also include high-scoring results
                    # These likely contain the actual data even if they don't mention the reference
                    if is_semantic_query:
                        combined_score = chunk.get('_combined_score', 0)
                        # Include top results from semantic queries
                        if i < 5 or combined_score > 0.5:
                            chunk['_resolved_reference'] = ref['key']
                            chunk['_semantic_match'] = True
                            ref_chunks_found.append(chunk)
                            existing_ids.add(chunk_id)
                            source_files_found.add(chunk.get('metadata', {}).get('source_file', ''))
                            matched_count += 1
                            continue

                    # Also include chunks that mention the reference prominently
                    content_lower = chunk.get('content', '').lower()
                    ref_mention = f"{ref['type']} {ref['id']}".lower()
                    if content_lower.startswith(ref_mention) or f"\n{ref_mention}" in content_lower:
                        chunk['_resolved_reference'] = ref['key']
                        ref_chunks_found.append(chunk)
                        existing_ids.add(chunk_id)
                        source_files_found.add(chunk.get('metadata', {}).get('source_file', ''))
                        matched_count += 1

                logger.debug(f"Matched {matched_count} chunks for {ref['match']} from query '{query}'")

            except Exception as e:
                log.append(f"Error resolving {ref['match']}: {e}")
                logger.exception(f"Error resolving {ref['match']}")

        # For appendices, fetch adjacent chunks from the same source file
        # This helps get complete lists that span multiple chunks
        if ref['type'] == 'appendix' and ref_chunks_found and source_files_found:
            adjacent_start = time.time()
            adjacent_chunks = _fetch_adjacent_chunks(
                ref_chunks_found, searcher, existing_ids, max_adjacent=5
            )
            if adjacent_chunks:
                for chunk in adjacent_chunks:
                    chunk['_resolved_reference'] = ref['key']
                    chunk['_adjacent_to_reference'] = True
                ref_chunks_found.extend(adjacent_chunks)
                existing_ids.update(hash(c.get('content', '')[:200]) for c in adjacent_chunks)
                logger.debug(f"Added {len(adjacent_chunks)} adjacent chunks in {time.time() - adjacent_start:.2f}s")

        ref_time = time.time() - ref_start
        if ref_chunks_found:
            new_chunks.extend(ref_chunks_found)
            resolved_refs.add(ref_key)
            # Store in cache for future lookups
            if use_cache:
                _reference_cache[ref_key] = ref_chunks_found.copy()
                logger.debug(f"Cached {len(ref_chunks_found)} chunks for {ref_key}")
            log.append(f"Resolved {ref['match']}: found {len(ref_chunks_found)} chunks in {ref_time:.1f}s")
        else:
            log.append(f"Could not resolve {ref['match']} ({ref_time:.1f}s)")

    total_time = time.time() - start_time
    if not new_chunks:
        log.append(f"Could not find content for any references (total: {total_time:.1f}s)")
        return chunks, log

    log.append(f"Added {len(new_chunks)} total chunks from reference resolution (total: {total_time:.1f}s)")

    # Prioritize ALL chunks (original + resolved) by data source type
    def get_chunk_priority(chunk: dict) -> tuple:
        source = chunk.get('metadata', {}).get('source_file', '').lower()
        # Use rerank score if available (original search), otherwise combined score (reference resolution)
        score = chunk.get('_rerank_score', chunk.get('_combined_score', 0))
        chunk_index = chunk.get('metadata', {}).get('chunk_index', 0)

        # Priority tiers (lower = higher priority)
        # Tier 0: Reference data spreadsheets - the actual data we want
        if source.endswith(('.xlsx', '.xls', '.csv')):
            if any(term in source for term in ['reference', 'list', 'countries', 'data']):
                tier = 0  # Reference data spreadsheets - highest priority
            elif any(term in source for term in ['form', 'assessment', 'template']):
                tier = 3  # Forms and assessments - lower priority (not primary data)
            else:
                tier = 1  # Other spreadsheets
        elif any(term in source for term in ['reference', 'list', 'countries']):
            tier = 2  # Non-spreadsheet reference files
        elif chunk.get('_reference_priority') or chunk.get('_semantic_match'):
            tier = 4  # Matched reference content from resolution
        elif source.endswith('.pdf'):
            tier = 5  # PDF documents (regulatory, manuals) - supporting context
        else:
            tier = 6  # Other documents

        return (tier, -score, chunk_index)

    # Combine original and resolved chunks, then sort by tier
    all_chunks = chunks + new_chunks

    # Deduplicate by content hash (keep first occurrence)
    seen_ids = set()
    unique_chunks = []
    for chunk in all_chunks:
        chunk_id = hash(chunk.get('content', '')[:200])
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_chunks.append(chunk)

    # Sort ALL chunks by tier, then by score within tier
    augmented = sorted(unique_chunks, key=get_chunk_priority)

    # Log what data sources were found
    data_sources = set()
    for c in new_chunks[:20]:
        source = c.get('metadata', {}).get('source_file', '')
        if source.endswith(('.xlsx', '.xls', '.csv')):
            data_sources.add(source)
    if data_sources:
        log.append(f"Data sources prioritized: {list(data_sources)[:3]}")

    return augmented, log


def resolve_references_in_response(
    response: str,
    searcher,
    top_k_per_ref: int = 10,
) -> tuple[list[dict], list[str]]:
    """
    Detect references in LLM response and fetch that content.

    Use this after initial response to detect if LLM is telling user
    to "see Appendix X" without providing the actual content.

    Args:
        response: LLM response text
        searcher: HybridSearcher instance
        top_k_per_ref: Results per reference

    Returns:
        Tuple of (additional_chunks, resolution_log)
    """
    log = []
    references = detect_references(response)

    if not references:
        return [], log

    log.append(f"Response references: {[r['match'] for r in references]}")

    # Fetch content for each reference
    additional_chunks = []

    for ref in references[:3]:  # Limit to 3 references
        queries = build_reference_queries([ref])

        for query in queries[:1]:
            try:
                ref_chunks = searcher.search(query=query, top_k=top_k_per_ref)

                for chunk in ref_chunks:
                    if is_reference_content(chunk, ref):
                        chunk['_resolved_from_response'] = ref['key']
                        additional_chunks.append(chunk)
                        log.append(f"Found {ref['match']} content")
                        break

            except Exception as e:
                log.append(f"Error: {e}")

    return additional_chunks, log
