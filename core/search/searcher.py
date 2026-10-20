"""
HybridSearcher: BM25 keyword + vector semantic search with re-ranking.

This module provides:
- BM25 keyword search
- Vector semantic search
- Result merging with configurable weights
- Cross-encoder re-ranking
- Recursive/agentic search with query optimization
- BM25 index persistence with smart cache invalidation
"""

import copy
import hashlib
import json as _json
import logging
import re
import time
import concurrent.futures
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from core.reranker import Reranker, create_reranker
from core.search.types import SearchQuality, SearchResult, DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL
from core.search.tokenizer import EnhancedTokenizer, advanced_tokenize, TOKENIZER_VERSION, STOP_WORDS
from core.search.cache import SearchCache, BM25_CACHE_FILENAME

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Hybrid search combining BM25 keyword and vector semantic search.

    The search process:
    1. BM25 finds exact keyword matches
    2. Vector finds semantically similar content
    3. Results are merged with configurable weights
    4. Cross-encoder re-ranks to pick the best results
    """

    def __init__(
        self,
        index_path: str,
        collection_name: str = "documents",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        min_score: float = None,
        ephemeral: bool = False,
        tokenizer_config: dict = None,
        embedding_config: dict = None,
        # Adaptive reranking
        adaptive_rerank: bool = True,
        adaptive_rerank_threshold: float = 0.9,
        adaptive_rerank_depth: int = 15,
        # Merge strategy
        merge_strategy: str = "weighted",
        rrf_k: int = 60,
        # MMR diversity
        mmr_lambda: float = None,
        # Code priority settings
        code_priority: bool = False,
        code_priority_boost: float = 0.5,
        cross_reference_code: bool = False,
        cross_reference_top_k: int = 3,
        # Reranker backend
        reranker_backend: str = "local",
        reranker_endpoint: str = None,
        # BM25 parameter tuning
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        # Score normalization
        score_normalization: str = "minmax",
        # HyDE
        hyde: bool = False,
        llm_provider=None,
        # Multi-query
        multi_query: bool = False,
        multi_query_count: int = 3,
        # HNSW index tuning
        hnsw_ef_construction: int = 200,
        hnsw_max_neighbors: int = 16,
        hnsw_ef_search: int = 100,
        # Search result caching
        cache_ttl: float = 300.0,
        cache_max_size: int = 128,
        # Skip BM25 load during ingestion (rebuilt at end anyway)
        skip_bm25_load: bool = False,
    ):
        """
        Initialize the hybrid searcher.

        Args:
            index_path: Path to store/load the index (ignored if ephemeral=True)
            collection_name: ChromaDB collection name
            embedding_model: Sentence transformer model for embeddings (deprecated, use embedding_config)
            reranker_model: Cross-encoder model for re-ranking
            bm25_weight: Weight for BM25 scores (0-1)
            vector_weight: Weight for vector scores (0-1)
            min_score: Minimum rerank score threshold. Chunks below this are filtered.
                       Default None means no filtering (return all re-ranked results).
                       Use -2.0 for strict filtering (good for compliance docs).
            ephemeral: If True, use in-memory storage instead of persistent.
                       Useful for temporary indexes (e.g., conversation uploads)
                       to avoid SQLite file locking issues.
            tokenizer_config: Configuration for EnhancedTokenizer
                              {"stemmer": "porter"|"snowball"|None, "synonyms": {...}}
            embedding_config: Configuration for embedding model
                              {"model": str, "dimension": int, "query_prefix": str, "passage_prefix": str}
        """
        self.index_path = index_path
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.default_min_score = min_score  # None = no filtering
        self.ephemeral = ephemeral

        # Adaptive reranking settings
        self.adaptive_rerank = adaptive_rerank
        self.adaptive_rerank_threshold = adaptive_rerank_threshold
        self.adaptive_rerank_depth = adaptive_rerank_depth

        # Merge strategy
        self.merge_strategy = merge_strategy
        self.rrf_k = rrf_k

        # MMR diversity
        self.mmr_lambda = mmr_lambda

        # BM25 parameter tuning
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        # Score normalization strategy
        self.score_normalization = score_normalization

        # HyDE (Hypothetical Document Embeddings)
        self.hyde = hyde
        self.llm_provider = llm_provider

        # Multi-query / RAG-Fusion
        self.multi_query = multi_query
        self.multi_query_count = multi_query_count

        # HNSW index tuning
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_max_neighbors = hnsw_max_neighbors
        self.hnsw_ef_search = hnsw_ef_search

        # Search result cache
        self._cache = SearchCache(max_size=cache_max_size, ttl=cache_ttl)

        # Search quality tracking
        self._last_search_quality = None

        # Knowledge graph (lazy-loaded)
        self._knowledge_graph = None

        # Code priority settings (code as source of truth)
        self.code_priority = code_priority
        self.code_priority_boost = code_priority_boost
        self.cross_reference_code = cross_reference_code
        self.cross_reference_top_k = cross_reference_top_k
        if code_priority:
            logger.info(f"Code priority enabled (boost={code_priority_boost})")
        if cross_reference_code:
            logger.info(f"Cross-reference code enabled (top_k={cross_reference_top_k})")

        # Initialize enhanced tokenizer if config provided
        self.tokenizer: Optional[EnhancedTokenizer] = None
        if tokenizer_config:
            stemmer = tokenizer_config.get("stemmer")
            synonyms = tokenizer_config.get("synonyms", {})
            if stemmer or synonyms:
                self.tokenizer = EnhancedTokenizer(stemmer=stemmer, synonyms=synonyms)
                logger.info(f"Using EnhancedTokenizer (stemmer={stemmer}, synonyms={len(synonyms)})")

        # Configure embedding model from embedding_config or fallback to embedding_model param
        if embedding_config:
            self.embedding_model = embedding_config.get("model", DEFAULT_EMBEDDING_MODEL)
            self.embedding_dimension = embedding_config.get("dimension", 384)
            self.query_prefix = embedding_config.get("query_prefix", "")
            self.passage_prefix = embedding_config.get("passage_prefix", "")
        else:
            self.embedding_model = embedding_model
            self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
            self.query_prefix = ""
            self.passage_prefix = ""

        # Initialize embedding model (try GPU, fall back to CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.embedder = SentenceTransformer(self.embedding_model, device=device)
        except RuntimeError as e:
            if device == "cuda":
                logger.warning(f"GPU unavailable for embedder ({e}), using CPU")
                self.embedder = SentenceTransformer(self.embedding_model, device="cpu")
            else:
                raise
        logger.info(f"Using embedding model: {self.embedding_model} (dim={self.embedding_dimension})")

        # Initialize reranker (local or API-based)
        self.reranker: Reranker = create_reranker(
            backend=reranker_backend,
            model=reranker_model,
            endpoint=reranker_endpoint,
            device=device,
        )

        # Initialize ChromaDB (ephemeral or persistent)
        if ephemeral:
            self.chroma_client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"Initialized ephemeral hybrid searcher for {collection_name}")
        else:
            self.chroma_client = chromadb.PersistentClient(
                path=index_path,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"Initialized hybrid searcher at {index_path}")

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": self.hnsw_ef_construction,
                "hnsw:M": self.hnsw_max_neighbors,
                "hnsw:search_ef": self.hnsw_ef_search,
                "embedding_model": self.embedding_model,
                "embedding_dimension": self.embedding_dimension,
            }
        )

        # BM25 index (in-memory, rebuilt from ChromaDB on load or from cache)
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_chunks: list[dict] = []
        self._bm25_tokenized_corpus: Optional[list] = None
        self._last_corpus_hash: Optional[str] = None

        # Try to load BM25 from cache, rebuild if needed
        if skip_bm25_load:
            logger.info("Skipping BM25 load (will rebuild after ingestion)")
        else:
            count = self.collection.count()
            if count > 0:
                logger.info(f"Loading BM25 index ({count} documents in collection)...")
                if not self._load_bm25_cache():
                    # Cache already tried and failed — skip the cache check inside rebuild
                    self._rebuild_bm25_index(save_cache=True, skip_cache_check=True)

    def _get_bm25_cache_path(self) -> Path:
        """Get the path to the BM25 cache file."""
        return Path(self.index_path) / BM25_CACHE_FILENAME

    # Threshold above which corpus hash uses sampling instead of full scan
    _CORPUS_HASH_SAMPLE_THRESHOLD = 50000

    def _compute_corpus_hash(self) -> str:
        """
        Compute a hash of the corpus to detect changes.

        For small corpora (<50K docs): hashes all IDs and content prefixes.
        For large corpora (>=50K docs): hashes all IDs but only samples
        content prefixes (every Nth doc). The doc count is already checked
        before this method is called, so additions/removals are caught first.
        Sampling catches content modifications probabilistically.

        Also includes tokenizer config so changing stemmer/synonyms invalidates cache.
        """
        count = self.collection.count()
        if count == 0:
            return ""

        h = hashlib.md5()
        h.update(f"count:{count}\n".encode())

        if count < self._CORPUS_HASH_SAMPLE_THRESHOLD:
            # Small corpus: full hash (existing behavior)
            results = self.collection.get(include=["documents"])
            pairs = sorted(zip(results["ids"], results["documents"]))
            for doc_id, doc in pairs:
                content_prefix = (doc or "")[:100]
                h.update(f"{doc_id}:{content_prefix}\n".encode())
        else:
            # Large corpus: hash all IDs + sample content
            # Phase 1: fetch IDs only (cheap — just strings)
            id_results = self.collection.get(include=[])
            sorted_ids = sorted(id_results["ids"])
            for doc_id in sorted_ids:
                h.update(f"{doc_id}\n".encode())

            # Phase 2: sample every Nth doc for content hash
            sample_step = max(1, count // 2000)  # ~2000 samples max
            sample_ids = sorted_ids[::sample_step]
            if sample_ids:
                sample_results = self.collection.get(
                    ids=sample_ids, include=["documents"]
                )
                for doc_id, doc in sorted(zip(sample_results["ids"], sample_results["documents"])):
                    content_prefix = (doc or "")[:100]
                    h.update(f"s:{doc_id}:{content_prefix}\n".encode())

            logger.info(f"Corpus hash: sampled {len(sample_ids)}/{count} docs (step={sample_step})")

        # Include tokenizer config in hash
        if self.tokenizer:
            h.update(f"tokenizer:{self.tokenizer.get_config_hash()}\n".encode())

        # Include BM25 params so changing k1/b invalidates cache
        h.update(f"bm25_params:{self.bm25_k1}:{self.bm25_b}\n".encode())

        result = h.hexdigest()
        self._last_corpus_hash = result
        return result

    def _load_bm25_cache(self) -> bool:
        """
        Try to load BM25 index from cache.

        Returns:
            True if cache was valid and loaded, False otherwise
        """
        if self.ephemeral:
            return False  # No caching for ephemeral indexes

        cache_path = self._get_bm25_cache_path()
        if not cache_path.exists():
            logger.info("No BM25 cache found, will rebuild")
            return False

        try:
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"Loading BM25 cache ({cache_size_mb:.0f} MB)...")
            _t0 = time.monotonic()
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = _json.load(f)
            logger.info(f"BM25 cache deserialized in {time.monotonic() - _t0:.1f}s")

            # Validate cache version
            if cache_data.get("tokenizer_version") != TOKENIZER_VERSION:
                logger.info(f"BM25 cache tokenizer version mismatch (cache={cache_data.get('tokenizer_version')}, current={TOKENIZER_VERSION})")
                return False

            # Fast check: document count changed → skip expensive hash
            current_count = self.collection.count()
            cached_count = cache_data.get("doc_count")
            if cached_count is not None and cached_count != current_count:
                logger.info(f"BM25 cache doc count mismatch (cache={cached_count}, current={current_count}), rebuilding")
                return False

            # Full validation: corpus content hash
            _t1 = time.monotonic()
            current_hash = self._compute_corpus_hash()
            logger.info(f"Corpus hash computed in {time.monotonic() - _t1:.1f}s")
            if cache_data.get("corpus_hash") != current_hash:
                logger.info("BM25 cache corpus hash mismatch, documents changed")
                return False

            # Rebuild BM25 index from cached tokenized corpus (safe JSON, no deserialization risk)
            tokenized_corpus = cache_data["tokenized_corpus"]
            self.bm25_index = BM25Okapi(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)
            self._bm25_tokenized_corpus = tokenized_corpus
            self.bm25_chunks = cache_data["bm25_chunks"]

            logger.info(f"Loaded BM25 index from cache ({len(self.bm25_chunks)} documents)")
            return True

        except Exception as e:
            logger.warning(f"Failed to load BM25 cache: {e}")
            return False

    def _save_bm25_cache(self) -> None:
        """Save BM25 index to cache for fast startup."""
        if self.ephemeral:
            return  # No caching for ephemeral indexes

        if self.bm25_index is None or not self.bm25_chunks:
            return

        cache_path = self._get_bm25_cache_path()

        try:
            cache_data = {
                "tokenizer_version": TOKENIZER_VERSION,
                "corpus_hash": self._last_corpus_hash or self._compute_corpus_hash(),
                "doc_count": len(self.bm25_chunks),
                "tokenized_corpus": getattr(self, "_bm25_tokenized_corpus", None),
                "bm25_chunks": self.bm25_chunks,
            }

            if cache_data["tokenized_corpus"] is None:
                logger.warning("No tokenized corpus available, skipping BM25 cache save")
                return

            with open(cache_path, "w", encoding="utf-8") as f:
                _json.dump(cache_data, f)

            logger.info(f"Saved BM25 cache ({len(self.bm25_chunks)} documents)")

        except Exception as e:
            logger.warning(f"Failed to save BM25 cache: {e}")

    def clear_bm25_cache(self) -> bool:
        """
        Clear the BM25 cache file.

        Returns:
            True if cache was cleared or didn't exist, False on error
        """
        if self.ephemeral:
            return True

        cache_path = self._get_bm25_cache_path()
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.info("Cleared BM25 cache")
                return True
            except Exception as e:
                logger.warning(f"Failed to clear BM25 cache: {e}")
                return False
        return True

    def force_rebuild_bm25(self) -> None:
        """Force rebuild of BM25 index, clearing any existing cache."""
        self.clear_bm25_cache()
        self._rebuild_bm25_index(save_cache=True)

    def check_embedding_model_match(self) -> tuple[bool, str]:
        """
        Check if the current embedding model matches the indexed model.

        Returns:
            Tuple of (matches, reason)
            - matches: True if models match or index is empty
            - reason: Description of mismatch or "OK"
        """
        if self.collection.count() == 0:
            return True, "Index is empty"

        metadata = self.collection.metadata or {}
        indexed_model = metadata.get("embedding_model")
        indexed_dimension = metadata.get("embedding_dimension")

        if indexed_model is None:
            # Legacy index without model metadata - assume it's compatible
            return True, "Legacy index (no model metadata)"

        if indexed_model != self.embedding_model:
            return False, f"Model mismatch: index has '{indexed_model}', config wants '{self.embedding_model}'"

        if indexed_dimension and indexed_dimension != self.embedding_dimension:
            return False, f"Dimension mismatch: index has {indexed_dimension}, config wants {self.embedding_dimension}"

        return True, "OK"

    def close(self) -> None:
        """
        Close ChromaDB connection and release resources.

        Call this before deleting or modifying the index directory
        to prevent stale connection issues.
        """
        try:
            # Release GPU models to free VRAM
            if hasattr(self, 'embedder') and self.embedder is not None:
                del self.embedder
                self.embedder = None
            if hasattr(self, 'reranker') and self.reranker is not None:
                self.reranker.close()
                self.reranker = None

            # Force CUDA memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Close knowledge graph SQLite connection
            if self._knowledge_graph is not None:
                self._knowledge_graph.close()
                self._knowledge_graph = None

            # ChromaDB PersistentClient doesn't have an explicit close method,
            # but we can delete our references to allow garbage collection
            self._cache.clear()
            self.collection = None
            self.chroma_client = None
            self.bm25_index = None
            self.bm25_chunks = []
            logger.info(f"Closed hybrid searcher at {self.index_path}")
        except Exception as e:
            logger.warning(f"Error closing hybrid searcher: {e}")

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """
        Sanitize metadata for ChromaDB compatibility.

        ChromaDB only accepts str, int, float, bool values.
        Lists and dicts are converted to JSON strings.
        None values are skipped.
        """
        import json
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                # Skip None values - ChromaDB doesn't accept them
                continue
            elif isinstance(value, bool):
                # Handle bool before int (bool is subclass of int)
                sanitized[key] = value
            elif isinstance(value, (str, int, float)):
                sanitized[key] = value
            elif isinstance(value, (list, dict)):
                # Convert lists and dicts to JSON strings
                sanitized[key] = json.dumps(value)
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        return sanitized

    def index_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """
        Add chunks to both vector and BM25 indices.

        Args:
            chunks: List of chunk dictionaries with 'id', 'content', 'metadata'
            embeddings: Corresponding embedding vectors
        """
        # Deduplicate chunks by ID (keep first occurrence)
        seen_ids = set()
        deduped_chunks = []
        deduped_embeddings = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk["id"]
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                deduped_chunks.append(chunk)
                deduped_embeddings.append(embedding)

        if len(deduped_chunks) < len(chunks):
            logger.warning(f"Removed {len(chunks) - len(deduped_chunks)} duplicate chunks")

        # ChromaDB has a max batch size limit (typically 5461)
        # Batch the additions to avoid errors
        batch_size = 5000
        total = len(deduped_chunks)

        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_chunks = deduped_chunks[i:batch_end]
            batch_embeddings = deduped_embeddings[i:batch_end]

            self.collection.upsert(
                ids=[c["id"] for c in batch_chunks],
                embeddings=batch_embeddings,
                documents=[c["content"] for c in batch_chunks],
                metadatas=[self._sanitize_metadata(c["metadata"]) for c in batch_chunks],
            )
            logger.info(f"Indexed batch {i//batch_size + 1}: {len(batch_chunks)} chunks ({batch_end}/{total})")

        # Update BM25 index
        self._rebuild_bm25_index()

        logger.info(f"Indexed {len(chunks)} chunks total")

    def _rebuild_bm25_index(self, save_cache: bool = True, skip_cache_check: bool = False) -> None:
        """
        Rebuild BM25 index from ChromaDB contents.

        If a valid cache exists and the corpus hasn't changed, skip the rebuild.

        Args:
            save_cache: If True, save the index to disk cache after building
            skip_cache_check: If True, skip the cache validity check (used when
                caller already tried _load_bm25_cache and it failed)
        """
        # Fast path: if cache is valid, skip the expensive rebuild
        if save_cache and not skip_cache_check and self._load_bm25_cache():
            return

        # Invalidate search cache — index contents are changing
        self._cache.clear()
        self._last_corpus_hash = None

        # Get all documents from ChromaDB
        results = self.collection.get(include=["documents", "metadatas"])

        if not results["documents"]:
            logger.warning("No documents in collection for BM25 index")
            return

        self.bm25_chunks = [
            {
                "id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
            }
            for i in range(len(results["ids"]))
        ]

        # Tokenize for BM25 using enhanced tokenizer (with stemming) or basic tokenizer
        if self.tokenizer:
            tokenized = [self.tokenizer.tokenize(doc) for doc in results["documents"]]
            tokenizer_info = f"EnhancedTokenizer (stemmer={self.tokenizer.stemmer_type})"
        else:
            tokenized = [advanced_tokenize(doc) for doc in results["documents"]]
            tokenizer_info = "advanced tokenization"

        self.bm25_index = BM25Okapi(tokenized, k1=self.bm25_k1, b=self.bm25_b)
        self._bm25_tokenized_corpus = tokenized  # retained for JSON cache serialization

        logger.info(f"Built BM25 index with {len(self.bm25_chunks)} documents ({tokenizer_info}, k1={self.bm25_k1}, b={self.bm25_b})")

        # Save to cache for faster startup next time
        if save_cache:
            self._save_bm25_cache()

    def _bm25_search(self, query: str, top_n: int = 50) -> list[tuple[dict, float]]:
        """
        BM25 keyword search.

        Args:
            query: Search query
            top_n: Number of results

        Returns:
            List of (chunk, score) tuples
        """
        if self.bm25_index is None or not self.bm25_chunks:
            self._rebuild_bm25_index()

        if self.bm25_index is None:
            return []

        # Use enhanced tokenizer for queries (with synonym expansion) if available
        if self.tokenizer:
            tokenized_query = self.tokenizer.tokenize_query(query)
        else:
            tokenized_query = advanced_tokenize(query)

        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top N indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                results.append((self.bm25_chunks[idx], scores[idx]))

        return results

    def _generate_hypothetical_document(self, query: str) -> Optional[str]:
        """Generate a hypothetical document that would answer the query.

        Used by HyDE to create an additional embedding for vector search.

        Args:
            query: The search query

        Returns:
            Hypothetical document text, or None on failure
        """
        if not self.llm_provider:
            return None

        try:
            system = (
                "Write a short paragraph (3-5 sentences) that would answer this question. "
                "Write as if you are a reference document providing factual information. "
                "Do not mention that you are generating a hypothetical answer."
            )
            messages = [{"role": "user", "content": query}]
            response = self.llm_provider.generate(messages, system_prompt=system)
            if response and len(response.strip()) > 20:
                logger.debug(f"HyDE generated hypothetical doc ({len(response)} chars)")
                return response.strip()
        except Exception as e:
            logger.warning(f"HyDE hypothetical document generation failed: {e}")

        return None

    def _vector_search(self, query: str, top_n: int = 50, extra_embedding: list[float] = None) -> list[tuple[dict, float]]:
        """
        Vector semantic search.

        Args:
            query: Search query
            top_n: Number of results
            extra_embedding: Additional embedding (e.g., from HyDE) to search alongside the query.
                When provided, runs two queries and merges results keeping higher scores.

        Returns:
            List of (chunk, score) tuples
        """
        # Apply query prefix if configured (e.g., "query: " for E5 models)
        query_text = f"{self.query_prefix}{query}" if self.query_prefix else query

        # Get query embedding
        query_embedding = self.embedder.encode(query_text).tolist()

        # Search ChromaDB with primary query
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            include=["documents", "metadatas", "distances"]
        )

        if not results["ids"][0]:
            chunks_by_id = {}
        else:
            chunks_by_id = {}
            for i in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][i]
                chunk = {
                    "id": chunk_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                }
                score = 1 - results["distances"][0][i]
                chunks_by_id[chunk_id] = (chunk, score)

        # If extra embedding provided (HyDE), run second query and merge
        if extra_embedding is not None:
            try:
                hyde_results = self.collection.query(
                    query_embeddings=[extra_embedding],
                    n_results=top_n,
                    include=["documents", "metadatas", "distances"]
                )
                if hyde_results["ids"][0]:
                    for i in range(len(hyde_results["ids"][0])):
                        chunk_id = hyde_results["ids"][0][i]
                        score = 1 - hyde_results["distances"][0][i]
                        if chunk_id not in chunks_by_id or score > chunks_by_id[chunk_id][1]:
                            chunk = {
                                "id": chunk_id,
                                "content": hyde_results["documents"][0][i],
                                "metadata": hyde_results["metadatas"][0][i] if hyde_results["metadatas"] else {},
                            }
                            chunks_by_id[chunk_id] = (chunk, score)
            except Exception as e:
                logger.warning(f"HyDE vector search failed, using raw query results: {e}")

        # Sort by score descending and return
        all_results = list(chunks_by_id.values())
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_n]

    def _normalize_scores(self, results: list[tuple[dict, float]]) -> dict[str, float]:
        """Normalize scores using the configured strategy.

        Strategies:
            minmax: Divide by max score -> 0-1 range (default, current behavior)
            zscore: Z-score normalization, shifted so min=0. Falls back to minmax
                    if std=0 or fewer than 2 results.
            raw: No normalization, pass scores through unchanged.

        Args:
            results: List of (chunk, score) tuples

        Returns:
            Dict mapping chunk ID to normalized score
        """
        if not results:
            return {}

        strategy = self.score_normalization

        if strategy == "raw":
            return {r[0]["id"]: r[1] for r in results}

        if strategy == "zscore" and len(results) >= 2:
            scores = [r[1] for r in results]
            mean = sum(scores) / len(scores)
            std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
            if std > 0:
                normalized = {r[0]["id"]: (r[1] - mean) / std for r in results}
                # Shift so min=0
                min_val = min(normalized.values())
                return {k: v - min_val for k, v in normalized.items()}
            # std=0 means all scores identical, fall through to minmax

        # minmax (default / fallback)
        max_score = max(r[1] for r in results)
        if max_score <= 0:
            return {r[0]["id"]: 0.0 for r in results}
        return {r[0]["id"]: r[1] / max_score for r in results}

    def _merge_results(
        self,
        bm25_results: list[tuple[dict, float]],
        vector_results: list[tuple[dict, float]],
    ) -> list[dict]:
        """
        Merge BM25 and vector results with weighted scoring.

        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search

        Returns:
            Merged and deduplicated list of chunks with combined scores
        """
        bm25_scores = self._normalize_scores(bm25_results)
        vector_scores = self._normalize_scores(vector_results)

        # Combine all unique chunk IDs
        all_ids = set(bm25_scores.keys()) | set(vector_scores.keys())

        # Build merged results
        chunks_by_id = {}
        for chunk, _ in bm25_results + vector_results:
            chunks_by_id[chunk["id"]] = chunk

        merged = []
        for chunk_id in all_ids:
            chunk = chunks_by_id[chunk_id].copy()
            bm25_score = bm25_scores.get(chunk_id, 0)
            vector_score = vector_scores.get(chunk_id, 0)

            combined_score = (
                self.bm25_weight * bm25_score +
                self.vector_weight * vector_score
            )

            chunk["_bm25_score"] = bm25_score
            chunk["_vector_score"] = vector_score
            chunk["_combined_score"] = combined_score
            merged.append(chunk)

        # Sort by combined score
        merged.sort(key=lambda x: x["_combined_score"], reverse=True)

        return merged

    def _rrf_merge(
        self,
        bm25_results: list[tuple[dict, float]],
        vector_results: list[tuple[dict, float]],
    ) -> list[dict]:
        """
        Merge BM25 and vector results using Reciprocal Rank Fusion.

        RRF scores are based on rank position rather than raw scores,
        making it more robust when score distributions differ between
        retrieval methods.

        Score formula: sum(1 / (k + rank)) across retrieval methods.

        Note: RRF scores are typically small (e.g., 0.003-0.033 for k=60),
        which means adaptive reranking (threshold 0.9) naturally won't
        trigger with RRF. This is correct behavior since RRF already
        handles score normalization via rank positions.

        Args:
            bm25_results: Results from BM25 search (sorted by score desc)
            vector_results: Results from vector search (sorted by score desc)

        Returns:
            Merged list of chunks with RRF scores
        """
        k = self.rrf_k

        # Build rank-based scores
        rrf_scores: dict[str, float] = {}
        bm25_raw_scores: dict[str, float] = {}
        vector_raw_scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict] = {}

        for rank, (chunk, score) in enumerate(bm25_results):
            chunk_id = chunk["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            bm25_raw_scores[chunk_id] = score
            chunks_by_id[chunk_id] = chunk

        for rank, (chunk, score) in enumerate(vector_results):
            chunk_id = chunk["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            vector_raw_scores[chunk_id] = score
            chunks_by_id[chunk_id] = chunk

        # Build merged results
        merged = []
        for chunk_id, rrf_score in rrf_scores.items():
            chunk = chunks_by_id[chunk_id].copy()
            chunk["_bm25_score"] = bm25_raw_scores.get(chunk_id, 0)
            chunk["_vector_score"] = vector_raw_scores.get(chunk_id, 0)
            chunk["_combined_score"] = rrf_score
            merged.append(chunk)

        merged.sort(key=lambda x: x["_combined_score"], reverse=True)
        return merged

    def _mmr_diversify(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
        lambda_param: float,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance (MMR) for result diversity.

        Iteratively selects chunks that balance relevance with novelty
        relative to already-selected chunks.

        MMR = lambda * relevance - (1 - lambda) * max_similarity_to_selected

        Args:
            query: Search query (for embedding)
            chunks: Reranked chunks with _rerank_score
            top_k: Number of results to select
            lambda_param: Balance between relevance (1.0) and diversity (0.0)

        Returns:
            Diversified list of chunks
        """
        if len(chunks) <= 1:
            return chunks

        # Get embeddings for all chunks
        query_text = f"{self.query_prefix}{query}" if self.query_prefix else query
        texts = [query_text] + [c["content"] for c in chunks]
        try:
            embeddings = self.embedder.encode(texts)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # CUDA OOM — clear cache and retry on CPU
            logger.warning("CUDA OOM during MMR embeddings, falling back to CPU")
            torch.cuda.empty_cache()
            try:
                embeddings = self.embedder.encode(texts, device="cpu")
            except Exception as e:
                logger.warning(f"MMR CPU fallback also failed: {e}, skipping MMR")
                return chunks

        query_emb = embeddings[0]
        chunk_embs = embeddings[1:]

        # Normalize rerank scores to 0-1 for MMR calculation
        scores = np.array([c.get("_rerank_score", 0) for c in chunks])
        score_min = scores.min()
        score_max = scores.max()
        if score_max > score_min:
            norm_scores = (scores - score_min) / (score_max - score_min)
        else:
            norm_scores = np.ones_like(scores)

        # Precompute normalized embeddings for cosine similarity
        norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed_embs = chunk_embs / norms

        selected_indices = []
        remaining = set(range(len(chunks)))

        for _ in range(min(top_k, len(chunks))):
            best_idx = -1
            best_mmr = -float("inf")

            for idx in remaining:
                relevance = norm_scores[idx]

                if selected_indices:
                    # Max similarity to any already-selected chunk
                    selected_embs = normed_embs[selected_indices]
                    similarities = selected_embs @ normed_embs[idx]
                    max_sim = similarities.max()
                else:
                    max_sim = 0.0

                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx == -1:
                break

            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return [chunks[i] for i in selected_indices]

    def _sanitize_query(self, query: str) -> Optional[str]:
        """Sanitize and validate search query. Returns None if invalid."""
        if not query or not query.strip():
            logger.warning("Empty search query")
            return None
        # Strip control characters (null bytes, etc.) but keep newlines/tabs
        query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', query)
        if len(query) > 10_000:
            logger.warning(f"Query truncated from {len(query)} to 10000 chars")
            query = query[:10_000]
        return query.strip()

    def fast_search(
        self,
        query: str,
        top_k: int = 20,
        source_filter: list[str] = None,
    ) -> list[dict]:
        """
        Fast hybrid search WITHOUT re-ranking.

        Use this for reference resolution where speed matters more than
        perfect ranking. Returns results sorted by combined BM25+vector score.

        Args:
            query: Search query
            top_k: Number of results to return
            source_filter: Optional list of allowed source files (None = all)

        Returns:
            List of chunks sorted by combined score (no re-ranking)
        """

        query = self._sanitize_query(query)
        if query is None:
            return []

        t0 = time.time()
        logger.debug(f"Fast search for: {query[:50]}...")

        # Run BM25 and vector search in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self._bm25_search, query, top_k * 2)
            vector_future = executor.submit(self._vector_search, query, top_k * 2)

            try:
                bm25_results = bm25_future.result()
            except Exception as e:
                logger.warning(f"BM25 search failed in fast_search: {e}")
                bm25_results = []

            try:
                vector_results = vector_future.result()
            except Exception as e:
                logger.warning(f"Vector search failed in fast_search: {e}")
                vector_results = []

        if not bm25_results and not vector_results:
            return []

        # Merge without re-ranking
        if self.merge_strategy == "rrf":
            merged = self._rrf_merge(bm25_results, vector_results)
        else:
            merged = self._merge_results(bm25_results, vector_results)

        # Apply source filter if provided
        if source_filter is not None:
            merged = [c for c in merged if c.get("metadata", {}).get("source", "") in source_filter]

        logger.debug(f"Fast search completed in {time.time() - t0:.2f}s ({len(merged)} results)")

        return merged[:top_k]

    def _rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        """
        Re-rank chunks using cross-encoder.

        Args:
            query: Original query
            chunks: Chunks to re-rank
            top_k: Number of results to return

        Returns:
            Top K chunks after re-ranking
        """
        if not chunks:
            return []

        # Prepare query-document pairs for cross-encoder
        pairs = [(query, chunk["content"]) for chunk in chunks]

        # Get re-ranking scores
        scores = self.reranker.predict(pairs)

        # Add scores to chunk copies (avoid mutating originals)
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk = chunk.copy()
            chunk["_rerank_score"] = float(score)
            scored_chunks.append(chunk)

        # Sort by re-rank score and return top K
        scored_chunks.sort(key=lambda x: x["_rerank_score"], reverse=True)

        return scored_chunks[:top_k]

    def _multi_query_search(self, query: str, top_k: int = 10, **kwargs) -> list[dict]:
        """Execute multi-query / RAG-Fusion search.

        Generates query variants via LLM, runs each through the full search pipeline,
        then merges all results via RRF.

        Args:
            query: Original search query
            top_k: Number of final results
            **kwargs: Additional args passed to self.search() for each variant

        Returns:
            Merged list of top_k results
        """
        from core.refinement import generate_query_variants

        search_start = time.time()

        # Generate variants
        def _get_llm_response(messages, system):
            return self.llm_provider.generate(messages, system_prompt=system)

        variants = generate_query_variants(
            query, count=self.multi_query_count, get_llm_response=_get_llm_response
        )
        logger.info(f"Multi-query: generated {len(variants)} variants: {[v[:40] for v in variants]}")

        # Search each variant with multi_query disabled to prevent recursion
        all_variant_results = []
        for i, variant in enumerate(variants):
            try:
                results = self.search(variant, top_k=top_k, _skip_multi_query=True, **kwargs)
                all_variant_results.append(results)
                logger.debug(f"Multi-query variant {i+1}/{len(variants)}: {len(results)} results")
            except Exception as e:
                logger.warning(f"Multi-query variant {i+1} failed: {e}")

        if not all_variant_results:
            return []

        # RRF merge across all variants
        rrf_scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict] = {}
        k = self.rrf_k

        for variant_results in all_variant_results:
            for rank, chunk in enumerate(variant_results):
                chunk_id = chunk["id"]
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
                # Keep the chunk with higher rerank score
                if chunk_id not in chunks_by_id:
                    chunks_by_id[chunk_id] = chunk
                else:
                    existing_score = chunks_by_id[chunk_id].get("_rerank_score", 0)
                    new_score = chunk.get("_rerank_score", 0)
                    if new_score > existing_score:
                        chunks_by_id[chunk_id] = chunk

        # Build final results sorted by RRF score
        merged = []
        for chunk_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            chunk = chunks_by_id[chunk_id]
            chunk["_multi_query_rrf_score"] = rrf_score
            merged.append(chunk)

        result = merged[:top_k]

        total_time = time.time() - search_start
        logger.info(f"Multi-query search completed in {total_time:.2f}s: {len(variants)} variants -> {len(merged)} unique -> {len(result)} final")

        # Store quality metrics
        timing = {"multi_query_total": round(total_time, 3), "variants": len(variants)}
        self._last_search_quality = self._compute_quality(
            result, len(merged), timing, False,
            multi_query_applied=True,
            multi_query_variants=len(variants),
        )

        return result

    def _build_cache_key(
        self, query: str, top_k: int, min_score: float | None,
        source_filter: list[str] | None = None,
        chunk_types: list[str] | None = None,
        languages: list[str] | None = None,
        symbol_types: list[str] | None = None,
        repos: list[str] | None = None,
        rerank_top_n: int | None = None,
    ) -> str:
        """Build a deterministic cache key from search parameters."""
        config_parts = [
            self.merge_strategy,
            str(self.bm25_weight),
            str(self.vector_weight),
            str(self.mmr_lambda),
            str(self.code_priority),
            str(self.hyde),
            str(self.multi_query),
            str(sorted(source_filter) if source_filter else None),
            str(sorted(chunk_types) if chunk_types else None),
            str(sorted(languages) if languages else None),
            str(sorted(symbol_types) if symbol_types else None),
            str(sorted(repos) if repos else None),
            str(rerank_top_n),
        ]
        config_hash = hashlib.md5("|".join(config_parts).encode()).hexdigest()[:8]
        return f"{query}|{top_k}|{min_score}|{config_hash}"

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_n: int = 50,
        prioritize_data: bool = True,
        min_score: float = None,
        source_filter: list[str] = None,
        # Code-aware filtering
        chunk_types: list[str] = None,
        languages: list[str] = None,
        symbol_types: list[str] = None,
        repos: list[str] = None,
        _skip_multi_query: bool = False,
    ) -> list[dict]:
        """
        Hybrid search with re-ranking.

        Args:
            query: Search query
            top_k: Number of final results
            rerank_top_n: Number of candidates to re-rank
            prioritize_data: Reorder to prioritize actual data over references
            min_score: Minimum rerank score to include (default: -2.0)
                       Chunks with lower scores are filtered out as irrelevant.
            source_filter: Optional list of allowed source files (None = all)
            chunk_types: Filter by chunk type ("code", "document") - None for all
            languages: Filter by programming language ("python", "php", etc.) - None for all
            symbol_types: Filter by symbol type ("function", "class", "method") - None for all
            repos: Filter by repository name - None for all

        Returns:
            List of top K chunks after hybrid search and re-ranking
        """

        query = self._sanitize_query(query)
        if query is None:
            return []

        # Multi-query dispatch: generate variants, search each, merge via RRF.
        # Bypasses top-level cache intentionally — LLM-generated variants may differ
        # across calls, but individual variant searches ARE cached below.
        if self.multi_query and self.llm_provider and not _skip_multi_query:
            return self._multi_query_search(
                query, top_k=top_k, rerank_top_n=rerank_top_n,
                prioritize_data=prioritize_data, min_score=min_score,
                source_filter=source_filter, chunk_types=chunk_types,
                languages=languages, symbol_types=symbol_types, repos=repos,
            )

        if min_score is None:
            min_score = self.default_min_score  # Use instance default (can be set per-client)

        # Check search cache
        cache_key = self._build_cache_key(
            query, top_k, min_score,
            source_filter=source_filter, chunk_types=chunk_types,
            languages=languages, symbol_types=symbol_types,
            repos=repos, rerank_top_n=rerank_top_n,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            cached_results, cached_quality = cached
            quality = copy.copy(cached_quality)
            quality.cache_hit = True
            quality.cache_size = self._cache.size
            self._last_search_quality = quality
            logger.info(f"Cache hit for: {query[:50]}...")
            return list(cached_results)

        search_start = time.time()
        logger.info(f"Searching for: {query[:50]}...")

        # HyDE: generate hypothetical document embedding for vector search
        hyde_applied = False
        hyde_embedding = None
        hyde_time = 0.0
        if self.hyde and self.llm_provider:
            t0 = time.time()
            hypo_doc = self._generate_hypothetical_document(query)
            if hypo_doc:
                query_text = f"{self.passage_prefix}{hypo_doc}" if self.passage_prefix else hypo_doc
                hyde_embedding = self.embedder.encode(query_text).tolist()
                hyde_applied = True
                logger.info(f"HyDE: generated hypothetical doc ({len(hypo_doc)} chars)")
            hyde_time = time.time() - t0

        # Step 1 & 2: Run BM25 and Vector search in parallel
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self._bm25_search, query, rerank_top_n)
            vector_future = executor.submit(self._vector_search, query, rerank_top_n, hyde_embedding)

            try:
                bm25_results = bm25_future.result()
            except Exception as e:
                logger.warning(f"BM25 search failed, using vector-only: {e}")
                bm25_results = []

            try:
                vector_results = vector_future.result()
            except Exception as e:
                logger.warning(f"Vector search failed, using BM25-only: {e}")
                vector_results = []

        if not bm25_results and not vector_results:
            logger.warning("Both BM25 and vector search failed, returning empty results")
            return []

        parallel_time = time.time() - t0
        logger.debug(f"Parallel search: BM25={len(bm25_results)}, Vector={len(vector_results)} in {parallel_time:.2f}s")

        # Step 3: Merge results
        t0 = time.time()
        if self.merge_strategy == "rrf":
            merged = self._rrf_merge(bm25_results, vector_results)
        else:
            merged = self._merge_results(bm25_results, vector_results)
        merge_time = time.time() - t0
        logger.debug(f"Merged to {len(merged)} unique results ({self.merge_strategy}) in {merge_time:.3f}s")

        # Step 4: Re-rank with cross-encoder (this is typically the slowest step)
        # Adaptive reranking: if top combined score is very high, reduce rerank depth
        effective_rerank_n = rerank_top_n
        adaptive_applied = False
        if self.adaptive_rerank and merged and merged[0].get("_combined_score", 0) > self.adaptive_rerank_threshold:
            # High-confidence match - both BM25 and vector agree strongly
            effective_rerank_n = min(self.adaptive_rerank_depth, rerank_top_n)
            adaptive_applied = True
            logger.info(f"High-confidence match (score={merged[0]['_combined_score']:.2f}), reducing rerank depth to {effective_rerank_n}")

        t0 = time.time()
        reranked = self._rerank(query, merged[:effective_rerank_n], top_k * 3)  # Get extra for filtering
        rerank_time = time.time() - t0
        logger.debug(f"Re-ranked {min(len(merged), effective_rerank_n)} chunks in {rerank_time:.2f}s")

        # Step 4b: MMR diversity (if enabled)
        mmr_time = 0.0
        if self.mmr_lambda is not None and reranked:
            # Free reranker GPU memory before MMR embedding
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            t0 = time.time()
            reranked = self._mmr_diversify(query, reranked, top_k * 3, self.mmr_lambda)
            mmr_time = time.time() - t0
            logger.debug(f"MMR diversification (lambda={self.mmr_lambda}) in {mmr_time:.2f}s")

        # Step 5: Filter by minimum score threshold (if set)
        if min_score is not None:
            before_filter = len(reranked)
            reranked = [c for c in reranked if c.get("_rerank_score", 0) >= min_score]
            filtered_count = before_filter - len(reranked)
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} chunks below score threshold {min_score}")

        # Step 5b: Filter by source files (if source_filter provided)
        if source_filter is not None:
            before_filter = len(reranked)
            reranked = [c for c in reranked if c.get("metadata", {}).get("source", "") in source_filter]
            filtered_count = before_filter - len(reranked)
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} chunks by source selection ({len(reranked)} remaining)")
            if len(reranked) == 0 and before_filter > 0:
                # Log some debug info about why nothing matched
                sample_sources = set()
                for c in merged[:5]:
                    src = c.get("metadata", {}).get("source_file", "")
                    if src:
                        sample_sources.add(src)
                logger.warning(f"Source filter removed ALL results. Filter has {len(source_filter)} files. Sample sources in results: {sample_sources}")

        # Step 5c: Apply code-aware filters
        reranked = self._apply_code_filters(
            reranked, chunk_types, languages, symbol_types, repos
        )

        # Step 5d: Cross-reference docs with code (if enabled)
        # This helps verify doc accuracy against actual code
        if self.cross_reference_code and reranked:
            reranked = self.cross_reference_with_code(reranked, self.cross_reference_top_k)

        # Step 5e: Prioritize code chunks (if enabled)
        # Code is the source of truth - docs may be outdated
        if self.code_priority and reranked:
            reranked = self.prioritize_code_chunks(reranked, self.code_priority_boost)

        total_search_time = time.time() - search_start
        logger.info(f"Search completed in {total_search_time:.2f}s (parallel_search={parallel_time:.2f}s, merge={merge_time:.3f}s, rerank={rerank_time:.2f}s, mmr={mmr_time:.2f}s)")

        # Step 6: Prioritize data chunks over references (within remaining chunks)
        if prioritize_data and reranked:
            reranked = self.prioritize_data_chunks(reranked)

        # Return top K
        result = reranked[:top_k]

        # Compute and store quality indicators
        timing = {
            "parallel_search": round(parallel_time, 3),
            "merge": round(merge_time, 3),
            "rerank": round(rerank_time, 3),
            "mmr": round(mmr_time, 3),
            "hyde": round(hyde_time, 3),
            "total": round(total_search_time, 3),
        }
        self._last_search_quality = self._compute_quality(
            result, len(merged), timing, adaptive_applied,
            hyde_applied=hyde_applied,
        )

        # Log quality metrics
        if result:
            q = self._last_search_quality
            logger.info(
                f"Returning {len(result)} results "
                f"(avg={q.avg_rerank_score:.2f}, top={q.top_score:.2f}, "
                f"sources={q.unique_sources})"
            )
        else:
            logger.warning("No results passed quality threshold")

        # Store in cache for future lookups
        self._cache.put(cache_key, (result, self._last_search_quality))

        return result

    def _compute_quality(
        self,
        results: list[dict],
        total_candidates: int,
        timing: dict,
        adaptive_applied: bool,
        hyde_applied: bool = False,
        multi_query_applied: bool = False,
        multi_query_variants: int = 0,
    ) -> SearchQuality:
        """Compute quality indicators for search results."""
        common = dict(
            total_candidates=total_candidates,
            merge_strategy=self.merge_strategy,
            score_normalization=self.score_normalization,
            mmr_applied=self.mmr_lambda is not None,
            adaptive_rerank_applied=adaptive_applied,
            hyde_applied=hyde_applied,
            multi_query_applied=multi_query_applied,
            multi_query_variants=multi_query_variants,
            timing=timing,
            cache_size=self._cache.size,
        )

        if not results:
            return SearchQuality(**common)

        scores = [c.get("_rerank_score", 0) for c in results]
        sources = {
            c.get("metadata", {}).get("source_file", "unknown")
            for c in results
        }

        return SearchQuality(
            avg_rerank_score=round(sum(scores) / len(scores), 4),
            top_score=round(max(scores), 4),
            score_spread=round(max(scores) - min(scores), 4),
            unique_sources=len(sources),
            **common,
        )

    def search_with_quality(self, **kwargs) -> SearchResult:
        """
        Run search() and return results with quality metadata.

        Accepts the same arguments as search(). Does NOT break
        the search() return type — this is a separate method.

        Returns:
            SearchResult with results list and SearchQuality
        """
        results = self.search(**kwargs)
        return SearchResult(
            results=results,
            quality=self._last_search_quality or SearchQuality(),
        )

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_chunks": self.collection.count(),
            "bm25_indexed": len(self.bm25_chunks) if self.bm25_chunks else 0,
        }

    def prioritize_code_chunks(
        self,
        chunks: list[dict],
        boost: float = 0.5,
    ) -> list[dict]:
        """
        Prioritize code chunks over document chunks.

        Code is the source of truth - documentation may be outdated.
        This boosts code chunk scores so they appear before docs in results.

        Args:
            chunks: List of chunks to reorder
            boost: Score boost for code chunks (added to rerank score)

        Returns:
            Reordered list with code chunks boosted
        """
        boosted = []
        for chunk in chunks:
            chunk = chunk.copy()
            metadata = chunk.get("metadata", {})
            chunk_type = metadata.get("chunk_type", "document")

            if chunk_type == "code":
                # Boost the score for code chunks
                if "_rerank_score" in chunk:
                    chunk["_rerank_score"] = chunk["_rerank_score"] + boost
                    chunk["_code_boosted"] = True
                elif "_combined_score" in chunk:
                    chunk["_combined_score"] = chunk["_combined_score"] + boost
                    chunk["_code_boosted"] = True

            boosted.append(chunk)
        chunks = boosted

        # Re-sort by score after boosting
        def get_score(c):
            return c.get("_rerank_score", c.get("_combined_score", 0))

        chunks.sort(key=get_score, reverse=True)
        return chunks

    def cross_reference_with_code(
        self,
        chunks: list[dict],
        top_k_per_doc: int = 3,
    ) -> list[dict]:
        """
        Cross-reference document chunks with related code.

        For each document chunk, searches for related code to verify/supplement
        the documentation. Useful when docs may be outdated but code is current.

        Args:
            chunks: List of document chunks
            top_k_per_doc: Number of related code chunks to fetch per document

        Returns:
            Original chunks + related code chunks (deduplicated)
        """
        if not chunks:
            return chunks

        # Separate code and doc chunks
        doc_chunks = []
        code_chunks = []
        seen_ids = set()

        for chunk in chunks:
            seen_ids.add(chunk["id"])
            metadata = chunk.get("metadata", {})
            if metadata.get("chunk_type") == "code":
                code_chunks.append(chunk)
            else:
                doc_chunks.append(chunk)

        # If no doc chunks or already have plenty of code, return as-is
        if not doc_chunks or len(code_chunks) >= len(doc_chunks):
            return chunks

        logger.info(f"Cross-referencing {len(doc_chunks)} doc chunks with code")

        # For each doc chunk, find related code
        related_code = []
        for doc_chunk in doc_chunks[:5]:  # Limit to avoid too many searches
            content = doc_chunk.get("content", "")

            # Extract key terms from the doc for code search
            # Focus on function names, class names, technical terms
            search_terms = self._extract_code_search_terms(content)
            if not search_terms:
                continue

            query = " ".join(search_terms[:5])

            # Fast search for related code (no re-ranking for speed)
            results = self.fast_search(query, top_k=top_k_per_doc * 2)

            # Filter to only code chunks we haven't seen
            for result in results:
                if result["id"] in seen_ids:
                    continue
                metadata = result.get("metadata", {})
                if metadata.get("chunk_type") == "code":
                    result["_cross_referenced"] = True
                    result["_referenced_from"] = doc_chunk["id"]
                    related_code.append(result)
                    seen_ids.add(result["id"])

                    if len(related_code) >= top_k_per_doc:
                        break

        if related_code:
            logger.info(f"Found {len(related_code)} related code chunks via cross-reference")

        # Return original chunks + related code
        return chunks + related_code

    def _extract_code_search_terms(self, content: str) -> list[str]:
        """
        Extract likely code-related terms from document content.

        Looks for:
        - CamelCase words (likely class names)
        - snake_case words (likely function names)
        - Technical terms in backticks
        - File paths/names
        """
        import re

        terms = []

        # CamelCase (class names)
        camel = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', content)
        terms.extend(camel)

        # snake_case (function names)
        snake = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', content)
        terms.extend(snake)

        # Backtick code references
        backtick = re.findall(r'`([^`]+)`', content)
        terms.extend(backtick)

        # File references (.ts, .tsx, .js, .py, etc.)
        files = re.findall(r'\b[\w/.-]+\.(tsx?|jsx?|py|php|css|scss)\b', content)
        terms.extend(files)

        # Dedupe while preserving order
        seen = set()
        unique = []
        for term in terms:
            if term.lower() not in seen and len(term) > 2:
                seen.add(term.lower())
                unique.append(term)

        return unique[:10]  # Limit to top 10 terms

    def prioritize_data_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Reorder chunks to prioritize those containing actual data over references.

        Tier 0: Reference/list spreadsheets (actual data sources)
        Tier 1: Other spreadsheets
        Tier 2: Content that looks like data/lists
        Tier 3: Everything else (explanatory text)
        Tier 4: Just references to data elsewhere
        """
        def get_tier(chunk: dict) -> int:
            source = chunk.get("metadata", {}).get("source_file", "").lower()
            content = chunk.get("content", "").lower()

            # Spreadsheets - sub-prioritize by type
            if source.endswith((".xlsx", ".xls", ".csv")):
                # Reference/list/data spreadsheets are primary data sources
                if any(term in source for term in ["reference", "list", "countries", "data"]):
                    return 0  # Highest priority - actual reference data
                # Forms and assessments are templates, not primary data
                if any(term in source for term in ["form", "assessment", "template"]):
                    return 2  # Lower than other spreadsheets
                return 1  # Other spreadsheets

            # Non-spreadsheet reference files
            if "reference" in source or "list" in source:
                return 1

            # Content that looks like data/lists
            data_indicators = [
                content.count("\n") > 10,  # Multiple lines (likely a list)
                "high" in content and any(c in content for c in ["afghanistan", "iran", "korea", "myanmar"]),
                content.count(",") > 5,  # Comma-separated values
            ]
            if any(data_indicators):
                return 2

            # Check if this is just a reference to data elsewhere
            reference_patterns = [
                "refer to appendix",
                "see appendix",
                "listed in appendix",
                "detailed in appendix",
                "as per appendix",
            ]
            if any(p in content for p in reference_patterns):
                return 4  # Deprioritize references

            return 3

        # Sort by tier, then by rerank score
        def sort_key(chunk):
            tier = get_tier(chunk)
            score = chunk.get("_rerank_score", chunk.get("_combined_score", 0))
            return (tier, -score)  # Lower tier first, higher score first within tier

        return sorted(chunks, key=sort_key)

    def _apply_code_filters(
        self,
        chunks: list[dict],
        chunk_types: list[str] = None,
        languages: list[str] = None,
        symbol_types: list[str] = None,
        repos: list[str] = None,
    ) -> list[dict]:
        """
        Apply code-aware filters to search results.

        Args:
            chunks: List of chunks to filter
            chunk_types: Filter by chunk type ("code", "document")
            languages: Filter by programming language
            symbol_types: Filter by symbol type ("function", "class", etc.)
            repos: Filter by repository name

        Returns:
            Filtered list of chunks
        """
        if not any([chunk_types, languages, symbol_types, repos]):
            return chunks  # No filters to apply

        filtered = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            # Filter by chunk type
            if chunk_types:
                ct = metadata.get("chunk_type", "document")
                if ct not in chunk_types:
                    continue

            # Filter by language (only applies to code chunks)
            if languages:
                lang = metadata.get("language")
                if lang and lang not in languages:
                    continue
                # If no language metadata and we're filtering by language,
                # only include if we're also allowing documents
                if not lang and "document" not in (chunk_types or []):
                    continue

            # Filter by symbol type (only applies to code chunks)
            if symbol_types:
                st = metadata.get("symbol_type")
                if st and st not in symbol_types:
                    continue
                # If no symbol_type metadata, only include if we're also allowing documents
                if not st and "document" not in (chunk_types or []):
                    continue

            # Filter by repository
            if repos:
                repo = metadata.get("repo_name")
                if repo and repo not in repos:
                    continue
                # If no repo metadata, include it (could be a doc or non-repo code)

            filtered.append(chunk)

        if len(filtered) < len(chunks):
            logger.info(
                f"Code filters: {len(chunks)} -> {len(filtered)} chunks "
                f"(types={chunk_types}, langs={languages}, symbols={symbol_types}, repos={repos})"
            )

        return filtered

    def recursive_search(
        self,
        query: str,
        top_k: int = 20,
        max_iterations: int = 5,
        get_llm_response: Callable = None,
        status_callback: Callable = None,
        min_quality_score: float = 0.3,
        source_filter: list[str] = None,
    ) -> tuple[list[dict], list[str], str]:
        """
        Recursive search with query optimization.

        Iteratively searches, analyzes results, and generates better queries
        until sufficient information is found or max iterations reached.

        Args:
            query: Initial search query
            top_k: Results per search iteration
            max_iterations: Maximum search iterations
            get_llm_response: Function to call LLM for analysis
            status_callback: Function to report progress
            min_quality_score: Minimum top score to consider results sufficient
            source_filter: Optional list of allowed source files (None = all)

        Returns:
            Tuple of (chunks, search_log, reasoning_summary)
        """
        from core.refinement import analyze_search_sufficiency, generate_alternative_queries, get_quality_metrics

        all_chunks = {}  # Dedupe by ID
        search_log = []
        queries_tried = [query]
        reasoning_parts = []
        best_quality = {"top_score": 0, "positive_count": 0}
        total_start = time.time()

        def log_status(msg: str):
            elapsed = time.time() - total_start
            timestamped_msg = f"[{elapsed:.1f}s] {msg}"
            search_log.append(timestamped_msg)
            if status_callback:
                status_callback(timestamped_msg)
            logger.info(msg)

        log_status(f"Starting recursive search: '{query[:50]}...'")

        current_query = query
        for iteration in range(max_iterations):
            iter_start = time.time()
            log_status(f"Iteration {iteration + 1}: Searching '{current_query[:40]}...'")

            # Search (with source filter if provided)
            results = self.search(current_query, top_k=top_k, source_filter=source_filter)
            search_time = time.time() - iter_start

            # Count new chunks and track quality
            new_count = 0
            for chunk in results:
                if chunk["id"] not in all_chunks:
                    all_chunks[chunk["id"]] = chunk
                    new_count += 1

            # Calculate quality metrics for this iteration
            iter_metrics = get_quality_metrics(results)
            log_status(f"  Found {len(results)} results ({new_count} new) in {search_time:.1f}s [top_score={iter_metrics['top_score']:.2f}, positive={iter_metrics['positive_count']}]")

            # Track best quality seen
            if iter_metrics["top_score"] > best_quality["top_score"]:
                best_quality = iter_metrics

            # Quick check: if top score is very low, definitely continue
            if iter_metrics["top_score"] < min_quality_score and iteration < max_iterations - 1:
                log_status(f"  Low quality (top_score={iter_metrics['top_score']:.2f} < {min_quality_score}), will try alternative query")

                # Generate alternatives without waiting for full LLM analysis
                if get_llm_response:
                    # Quick alternative generation
                    new_queries = generate_alternative_queries(
                        query, queries_tried, ["Better matching content"], get_llm_response
                    )
                    if new_queries:
                        current_query = new_queries[0]
                        queries_tried.extend(new_queries)
                        log_status(f"  Generated alternative: {current_query[:40]}...")
                        continue

            # Check if we have enough (without LLM if not provided)
            if not get_llm_response:
                # Use score-based heuristic
                if len(all_chunks) >= top_k and best_quality["top_score"] >= min_quality_score:
                    log_status(f"Sufficient results found (score={best_quality['top_score']:.2f}, no LLM)")
                    break
                elif len(all_chunks) >= top_k * 2:
                    log_status("Max chunks collected, stopping (no LLM)")
                    break
                continue

            # Full sufficiency analysis with LLM
            chunks_list = list(all_chunks.values())
            # Sort by score for analysis
            chunks_list.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)

            llm_start = time.time()
            analysis = analyze_search_sufficiency(
                query, chunks_list, get_llm_response
            )
            llm_time = time.time() - llm_start
            if llm_time > 1.0:
                log_status(f"  LLM analysis took {llm_time:.1f}s")

            quality_info = analysis.get("quality_metrics", {})
            reasoning_parts.append(
                f"Iteration {iteration + 1}: {analysis.get('reason', 'No reason')} "
                f"[top={quality_info.get('top_score', 0):.2f}, pos={quality_info.get('positive_count', 0)}]"
            )

            if analysis.get("sufficient", False):
                log_status(f"  Sufficient: {analysis.get('reason', '')[:60]}")
                break

            # Generate alternative queries
            missing = analysis.get("missing_info", [])
            better_terms = analysis.get("better_search_terms", [])

            if better_terms:
                # Use suggested terms
                current_query = " ".join(better_terms[:3])
                queries_tried.append(current_query)
                log_status(f"  Trying suggested terms: {current_query[:40]}...")
            elif missing:
                # Generate new queries based on missing info
                new_queries = generate_alternative_queries(
                    query, queries_tried, missing, get_llm_response
                )
                if new_queries:
                    current_query = new_queries[0]
                    queries_tried.extend(new_queries)
                    log_status(f"  Generated query: {current_query[:40]}...")
                else:
                    log_status("  No alternative queries generated, stopping")
                    break
            else:
                log_status("  No suggestions for improvement, stopping")
                break

        # Final results
        final_chunks = list(all_chunks.values())
        # Sort by rerank score if available
        final_chunks.sort(
            key=lambda x: x.get("_rerank_score", x.get("_combined_score", 0)),
            reverse=True,
        )

        # Final quality summary
        final_metrics = get_quality_metrics(final_chunks[:top_k])
        log_status(
            f"Search complete: {len(final_chunks)} chunks from {len(queries_tried)} queries "
            f"[final: top={final_metrics['top_score']:.2f}, positive={final_metrics['positive_count']}, high_quality={final_metrics['high_quality_count']}]"
        )

        reasoning_summary = "\n".join(reasoning_parts) if reasoning_parts else ""

        return final_chunks, search_log, reasoning_summary

    # ------------------------------------------------------------------
    # Knowledge Graph integration
    # ------------------------------------------------------------------

    def load_graph(self):
        """Lazy-load knowledge graph from index_path/knowledge_graph.json.

        Returns:
            KnowledgeGraph instance or None if no graph file exists.
        """
        if self._knowledge_graph is not None:
            return self._knowledge_graph

        try:
            from core.graph.store import KnowledgeGraph
            graph = KnowledgeGraph(self.index_path)
            if graph.load():
                self._knowledge_graph = graph
                logger.info(f"Loaded knowledge graph: {graph.stats['entity_count']} entities")
                return graph
        except Exception as e:
            logger.warning(f"Failed to load knowledge graph: {e}")

        return None

    def graph_search(
        self,
        query: str,
        mode: str = "naive",
        top_k: int = 20,
        **kwargs,
    ) -> list[dict]:
        """Dispatch to a graph search mode.

        Falls back to naive (standard) search if no graph is loaded.

        Args:
            query: Search query
            mode: One of "naive", "local", "global", "hybrid", "mix"
            top_k: Number of results
            **kwargs: Extra args passed to the search function

        Returns:
            List of result chunks
        """
        if mode == "naive":
            return self.search(query=query, top_k=top_k, **kwargs)

        graph = self.load_graph()
        if graph is None:
            logger.warning(f"No knowledge graph available, falling back to naive search")
            return self.search(query=query, top_k=top_k, **kwargs)

        from core.graph.search import local_search, global_search, hybrid_graph_search, mix_search

        if mode == "local":
            return local_search(query, graph, self, top_k=top_k)
        elif mode == "global":
            max_hops = kwargs.pop("max_hops", 2)
            return global_search(query, graph, self, top_k=top_k, max_hops=max_hops)
        elif mode == "hybrid":
            return hybrid_graph_search(query, graph, self, top_k=top_k)
        elif mode == "mix":
            return mix_search(query, graph, self, top_k=top_k)
        else:
            logger.warning(f"Unknown graph search mode '{mode}', falling back to naive")
            return self.search(query=query, top_k=top_k, **kwargs)
