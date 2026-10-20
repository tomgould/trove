"""Merge results from main index and conversation index."""


def merge_search_results(
    main_results: list[dict],
    conv_results: list[dict],
    boost_conversation: float = 0.1,
) -> list[dict]:
    """
    Merge results from main index and conversation index.

    Combines results from both indices, deduplicates by ID, and applies
    a slight boost to conversation-specific documents to prioritize
    user-uploaded context.

    Args:
        main_results: Results from main client index
        conv_results: Results from conversation-specific index
        boost_conversation: Score boost for conversation docs (default 0.1)

    Returns:
        Merged and sorted list of chunks
    """
    # Build lookup for deduplication
    seen_ids = {}

    # Add main results
    for chunk in main_results:
        chunk_id = chunk.get("id")
        if chunk_id:
            seen_ids[chunk_id] = chunk

    # Add conversation results (with boost)
    for chunk in conv_results:
        chunk_id = chunk.get("id")
        if chunk_id and chunk_id not in seen_ids:
            # Apply boost to rerank score
            if "_rerank_score" in chunk:
                chunk["_rerank_score"] = chunk["_rerank_score"] + boost_conversation
            if "_combined_score" in chunk:
                chunk["_combined_score"] = chunk["_combined_score"] + boost_conversation

            # Mark as from conversation upload
            chunk["_from_conversation"] = True
            seen_ids[chunk_id] = chunk

    # Combine and sort by score
    merged = list(seen_ids.values())

    def get_score(c):
        return c.get("_rerank_score", c.get("_combined_score", 0))

    merged.sort(key=get_score, reverse=True)

    return merged
