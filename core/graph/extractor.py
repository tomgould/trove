"""
LLM-based entity and relationship extraction from document chunks.

Accepts a callable for LLM interaction (decoupled from specific provider).
Supports concurrent batch processing via ThreadPoolExecutor with adaptive
text-budget-based batching (starts large, halves on failure).
"""

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from core.graph.models import Entity, Relationship
from core.graph.progress import GraphBuildProgress

logger = logging.getLogger(__name__)

# Text limits for batch processing
CHUNK_TEXT_LIMIT = 3000        # Max chars per chunk in a batch
DEFAULT_BATCH_TEXT_BUDGET = 24000  # Starting text budget per batch (~6K tokens, fits Mistral Nemo 128K)
MIN_BATCH_TEXT_BUDGET = 1000   # Minimum text budget before giving up on splitting
MAX_OUTPUT_TOKENS = 1024       # Max tokens for LLM response (more entities from larger batches)

EXTRACTION_PROMPT = """Extract entities and relationships from the following text.

Rules:
- Only extract clearly named entities (not vague references)
- Entity types must be one of: {entity_types}
- Relationship descriptions should be concise (one sentence max)
- Weight is confidence from 0.0 to 1.0

Return ONLY valid JSON in this exact format:
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}], "relationships": [{{"source": "...", "target": "...", "type": "...", "description": "...", "weight": 0.9}}]}}

Text:
{text}"""


def _build_batches_by_budget(
    chunks: list[dict], text_budget: int
) -> list[list[dict]]:
    """Split chunks into batches where each batch's combined text fits within the budget."""
    batches = []
    current_batch = []
    current_size = 0
    separator_size = len("\n\n---\n\n")

    for chunk in chunks:
        content = chunk.get("content", "")
        if not content.strip():
            current_batch.append(chunk)
            continue

        chunk_id = chunk.get("id", "unknown")
        # Estimate the text this chunk would add
        chunk_text_size = min(len(content), CHUNK_TEXT_LIMIT) + len(f"[Chunk {chunk_id}]\n")
        entry_size = chunk_text_size + (separator_size if current_size > 0 else 0)

        if current_size + entry_size > text_budget and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0

        current_batch.append(chunk)
        current_size += entry_size

    if current_batch:
        batches.append(current_batch)

    return batches


class GraphExtractor:
    """Extract entities and relationships from chunks using an LLM."""

    def __init__(
        self,
        get_llm_response: Callable,
        entity_types: list[str] | None = None,
        batch_size: int = 512,
        max_description_chars: int = 200,
        max_workers: int = 4,
        save_every: int = 5,
        batch_text_budget: int = DEFAULT_BATCH_TEXT_BUDGET,
    ):
        """
        Args:
            get_llm_response: callable(messages, system_prompt, max_tokens) -> str
            entity_types: Allowed entity types (default: person, organization, concept, location)
            batch_size: Legacy max chunks per batch (kept for config compat, text budget is primary)
            max_description_chars: Truncate descriptions to this length
            max_workers: Number of concurrent LLM requests (default: 4)
            save_every: Call progress_callback every N completed batches (default: 5)
            batch_text_budget: Max chars of combined text per LLM call (default: 60000)
        """
        self.get_llm_response = get_llm_response
        self.entity_types = entity_types or ["person", "organization", "concept", "location"]
        self.batch_size = batch_size
        self.max_description_chars = max_description_chars
        self.max_workers = max_workers
        self.save_every = save_every
        self.batch_text_budget = batch_text_budget
        # Circuit breaker — stops hammering a dead LM Studio
        from core.health import CircuitBreaker
        self._circuit_breaker = CircuitBreaker(failure_threshold=50, cooldown_seconds=60.0)

    def extract_from_chunk(
        self, chunk: dict
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from a single chunk."""
        content = chunk.get("content", "")
        chunk_id = chunk.get("id", "unknown")

        if not content.strip():
            return [], []

        prompt = EXTRACTION_PROMPT.format(
            entity_types=", ".join(self.entity_types),
            text=content[:CHUNK_TEXT_LIMIT],
        )

        try:
            response = self.get_llm_response(
                [{"role": "user", "content": prompt}],
                "You are an entity extraction assistant. Return only valid JSON.",
                MAX_OUTPUT_TOKENS,
            )
            return self._parse_response(response, chunk_id)
        except Exception as e:
            logger.warning(f"Extraction failed for chunk {chunk_id}: {e}")
            return [], []

    def _call_llm_for_batch(
        self, batch: list[dict]
    ) -> tuple[list[Entity], list[Relationship], list[str], bool]:
        """Make a single LLM call for a batch. Returns (entities, rels, chunk_ids, success)."""
        # Circuit breaker check — skip if LM Studio is confirmed down
        if not self._circuit_breaker.allow_request():
            chunk_ids = [c.get("id", "unknown") for c in batch if c.get("content", "").strip()]
            logger.debug(f"Circuit breaker OPEN, skipping batch of {len(batch)} chunks")
            return [], [], chunk_ids, False

        batch_text_parts = []
        batch_chunk_ids = []
        for chunk in batch:
            content = chunk.get("content", "")
            chunk_id = chunk.get("id", "unknown")
            if content.strip():
                batch_text_parts.append(f"[Chunk {chunk_id}]\n{content[:CHUNK_TEXT_LIMIT]}")
                batch_chunk_ids.append(chunk_id)

        if not batch_text_parts:
            return [], [], batch_chunk_ids, True

        combined_text = "\n\n---\n\n".join(batch_text_parts)

        prompt = EXTRACTION_PROMPT.format(
            entity_types=", ".join(self.entity_types),
            text=combined_text,
        )

        try:
            response = self.get_llm_response(
                [{"role": "user", "content": prompt}],
                "You are an entity extraction assistant. Return only valid JSON.",
                MAX_OUTPUT_TOKENS,
            )

            # Detect error responses from LLM provider
            if isinstance(response, str) and response.startswith("Error:"):
                logger.debug(f"LLM returned error: {response[:200]}")
                self._circuit_breaker.record_failure()
                return [], [], batch_chunk_ids, False

            self._circuit_breaker.record_success()

            entities, relationships = self._parse_response(
                response, batch_chunk_ids[0] if batch_chunk_ids else "unknown"
            )
            # Tag all entities/relationships with all batch chunk IDs
            for entity in entities:
                entity.source_chunk_ids = list(set(entity.source_chunk_ids + batch_chunk_ids))
            for rel in relationships:
                rel.source_chunk_ids = list(set(rel.source_chunk_ids + batch_chunk_ids))

            return entities, relationships, batch_chunk_ids, True

        except Exception as e:
            logger.debug(f"LLM call failed: {e}")
            self._circuit_breaker.record_failure()
            return [], [], batch_chunk_ids, False

    def _process_batch(
        self, batch: list[dict], batch_index: int
    ) -> tuple[list[Entity], list[Relationship], list[str], int, int]:
        """Process a batch with per-batch retry on failure.

        On 400 errors, halves the batch locally (doesn't affect other batches).
        Tries up to 2 levels of splitting before giving up on a sub-batch.

        Returns:
            (entities, relationships, chunk_ids, num_chunks_in_batch, skipped_chunks)
        """
        all_chunk_ids = [c.get("id", "unknown") for c in batch if c.get("content", "").strip()]
        content_count = len(all_chunk_ids)

        # Try the full batch
        entities, rels, chunk_ids, success = self._call_llm_for_batch(batch)
        if success:
            return entities, rels, all_chunk_ids, len(batch), 0

        # Failed — split this batch in half and retry each half (local only)
        budget = self.batch_text_budget // 2
        if budget < MIN_BATCH_TEXT_BUDGET:
            logger.warning(f"Batch {batch_index}: failed, skipping {content_count} chunks")
            return [], [], all_chunk_ids, len(batch), content_count

        sub_batches = _build_batches_by_budget(batch, budget)
        all_entities: list[Entity] = []
        all_rels: list[Relationship] = []
        skipped = 0

        for sub in sub_batches:
            ents, rs, ids, ok = self._call_llm_for_batch(sub)
            if ok:
                all_entities.extend(ents)
                all_rels.extend(rs)
            else:
                # One more split attempt
                deeper = budget // 2
                if deeper >= MIN_BATCH_TEXT_BUDGET:
                    for deep_sub in _build_batches_by_budget(sub, deeper):
                        e2, r2, _, ok2 = self._call_llm_for_batch(deep_sub)
                        if ok2:
                            all_entities.extend(e2)
                            all_rels.extend(r2)
                        else:
                            n = len([c for c in deep_sub if c.get("content", "").strip()])
                            skipped += n
                else:
                    n = len([c for c in sub if c.get("content", "").strip()])
                    skipped += n

        if skipped > 0:
            logger.warning(
                f"Batch {batch_index}: {skipped}/{content_count} chunks skipped after splitting"
            )
        return all_entities, all_rels, all_chunk_ids, len(batch), skipped

    def extract_from_chunks(
        self,
        chunks: list[dict],
        progress_callback: Optional[Callable[[GraphBuildProgress], None]] = None,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract from multiple chunks using concurrent batch processing.

        Builds batches by text budget so each LLM call gets a full context
        window. On failure, each batch independently halves and retries (no
        global budget reduction). Sends batches to the LLM in parallel via
        ThreadPoolExecutor.
        """
        all_entities: list[Entity] = []
        all_relationships: list[Relationship] = []

        progress = GraphBuildProgress(total_chunks=len(chunks))
        lock = threading.Lock()

        # Build batches by text budget
        batches = _build_batches_by_budget(chunks, self.batch_text_budget)

        if not batches:
            return all_entities, all_relationships

        completed_since_save = 0

        def _on_result(entities, relationships, chunk_ids, num_chunks,
                       skipped_chunks: int = 0):
            """Merge results under lock and trigger periodic saves."""
            nonlocal completed_since_save

            with lock:
                all_entities.extend(entities)
                all_relationships.extend(relationships)
                progress.entities_found += len(entities)
                progress.relationships_found += len(relationships)
                progress.failed_chunks += skipped_chunks

                progress.processed_chunks += num_chunks
                completed_since_save += 1

                should_save = (
                    completed_since_save >= self.save_every
                    or progress.processed_chunks >= progress.total_chunks
                )

                if should_save and progress_callback:
                    completed_since_save = 0
                    progress_callback(progress, all_entities, all_relationships)

        workers = min(self.max_workers, len(batches))
        avg_batch = len(chunks) / len(batches) if batches else 0
        logger.info(
            f"Processing {len(batches)} batches with {workers} workers "
            f"(text_budget={self.batch_text_budget:,} chars, ~{avg_batch:.0f} chunks/batch, "
            f"save_every={self.save_every})"
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._process_batch, batch, idx): idx
                for idx, batch in enumerate(batches)
            }

            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    entities, relationships, chunk_ids, num_chunks, skipped = future.result()
                    _on_result(entities, relationships, chunk_ids, num_chunks,
                               skipped_chunks=skipped)
                except Exception as e:
                    logger.warning(f"Batch {batch_idx} raised exception: {e}")
                    batch = batches[batch_idx]
                    _on_result([], [], [], len(batch),
                               skipped_chunks=len(batch))

        # Final callback to ensure last results are saved
        if progress_callback and completed_since_save > 0:
            with lock:
                progress_callback(progress, all_entities, all_relationships)

        cb_stats = self._circuit_breaker.stats
        if cb_stats["total_rejected"] > 0:
            logger.warning(
                f"Circuit breaker: {cb_stats['total_rejected']} requests rejected "
                f"(state={cb_stats['state']}, failures={cb_stats['consecutive_failures']})"
            )

        return all_entities, all_relationships

    def _parse_response(
        self, response: str, source_chunk_id: str
    ) -> tuple[list[Entity], list[Relationship]]:
        """Parse LLM response JSON into Entity and Relationship objects."""
        # Strip thinking model tags (e.g. Qwen 3.5 <think>...</think>)
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            # Remove closing fence
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.debug(f"Could not parse JSON from response: {cleaned[:200]}")
                    return [], []
            else:
                logger.debug(f"No JSON found in response: {cleaned[:200]}")
                return [], []

        entities = []
        relationships = []

        for ed in data.get("entities", []):
            name = ed.get("name", "").strip()
            entity_type = ed.get("type", "").strip().lower()
            if not name or not entity_type:
                continue
            # Filter to allowed types
            if entity_type not in self.entity_types:
                continue
            desc = ed.get("description", "")[:self.max_description_chars]
            entities.append(Entity(
                name=name,
                entity_type=entity_type,
                description=desc,
                source_chunk_ids=[source_chunk_id],
            ))

        # Build name-to-key lookup for O(1) relationship resolution
        entity_key_map = {e.name.lower(): e.key for e in entities}

        for rd in data.get("relationships", []):
            source = rd.get("source", "").strip()
            target = rd.get("target", "").strip()
            rel_type = rd.get("type", "").strip()
            if not source or not target or not rel_type:
                continue
            desc = rd.get("description", "")[:self.max_description_chars]
            weight = min(1.0, max(0.0, float(rd.get("weight", 1.0))))

            source_key = entity_key_map.get(source.lower()) or f"{source.lower()}::concept"
            target_key = entity_key_map.get(target.lower()) or f"{target.lower()}::concept"

            relationships.append(Relationship(
                source_entity=source_key,
                target_entity=target_key,
                relationship_type=rel_type,
                description=desc,
                weight=weight,
                source_chunk_ids=[source_chunk_id],
            ))

        return entities, relationships
