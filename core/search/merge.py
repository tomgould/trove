"""Merge search results: main index with a conversation index, or several clients."""


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


def merge_client_results(
    client_results: dict[str, list[dict]],
    k: int = 60,
    top_k: int = 20,
) -> list[dict]:
    """Merge ranked results from several clients with Reciprocal Rank Fusion.

    Each chunk is labelled with ``_source_client``. Ranks are compared, not raw
    scores, because scores from different indexes are not on the same scale.
    """
    rrf_scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict] = {}

    for client_name, results in client_results.items():
        for rank, chunk in enumerate(results):
            chunk_id = f"{client_name}:{chunk.get('id', rank)}"
            chunk["_source_client"] = client_name
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            chunks_by_id[chunk_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    for cid in sorted_ids:
        chunks_by_id[cid]["_rrf_score"] = rrf_scores[cid]
    return [chunks_by_id[cid] for cid in sorted_ids[:top_k]]
