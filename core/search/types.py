"""Search result types and constants."""

from dataclasses import dataclass, field

# Default models (all local, open source)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class SearchQuality:
    """Quality indicators for a search result set."""
    avg_rerank_score: float = 0.0
    top_score: float = 0.0
    score_spread: float = 0.0
    unique_sources: int = 0
    total_candidates: int = 0
    merge_strategy: str = "weighted"
    score_normalization: str = "minmax"
    mmr_applied: bool = False
    adaptive_rerank_applied: bool = False
    hyde_applied: bool = False
    multi_query_applied: bool = False
    multi_query_variants: int = 0
    cache_hit: bool = False
    cache_size: int = 0
    timing: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search results with quality metadata."""
    results: list[dict] = field(default_factory=list)
    quality: SearchQuality = field(default_factory=SearchQuality)
