"""
Question refinement and optimization.

Uses LLM to generate alternative phrasings and improve search queries.
"""

import json
import logging
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Default refinement prompt
REFINEMENT_SYSTEM_PROMPT = """You are a search query optimizer. Your job is to help users find information in a document knowledge base.

Given a question, generate 5 alternative phrasings that might find better search results.

Consider:
1. Different terminology (synonyms, technical terms)
2. More specific vs more general phrasing
3. List requests vs definition requests
4. Document/section references (e.g., "Appendix", "Section", "Schedule")
5. Regulatory/compliance terminology if applicable

Return JSON:
{
    "topic": "Short topic (2-4 words)",
    "understanding": "What the user is looking for",
    "suggestions": [
        {"text": "Alternative question 1", "angle": "Why this phrasing might help"},
        {"text": "Alternative question 2", "angle": "Why this phrasing might help"},
        {"text": "Alternative question 3", "angle": "Why this phrasing might help"},
        {"text": "Alternative question 4", "angle": "Why this phrasing might help"},
        {"text": "Alternative question 5", "angle": "Why this phrasing might help"}
    ]
}"""


# =============================================================================
# Query Contextualization (for follow-up questions)
# =============================================================================

# Indicators that a question needs context from conversation history
FOLLOWUP_INDICATORS = [
    # Pronouns referring to previous content
    "that", "this", "it", "they", "them", "those", "these",
    # Continuation words
    "more", "else", "also", "another", "other",
    # Explicit follow-up phrases
    "expand", "elaborate", "explain further", "tell me more",
    "what about", "how about", "and the", "any other",
    "continue", "go on", "keep going",
]

# Words that indicate a standalone question (doesn't need context)
STANDALONE_INDICATORS = [
    "what is", "what are", "how do", "how does", "where is",
    "who is", "when is", "why is", "define", "explain",
    "list all", "show me", "tell me about",
]


def needs_contextualization(question: str) -> bool:
    """
    Detect if a question is a follow-up that needs context from conversation history.

    Args:
        question: The user's question

    Returns:
        True if the question appears to be a follow-up needing context
    """
    question_lower = question.lower().strip()
    words = question_lower.split()

    # Very short questions are likely follow-ups
    if len(words) <= 4:
        return True

    # Check for explicit follow-up indicators
    for indicator in FOLLOWUP_INDICATORS:
        if indicator in question_lower:
            return True

    # Check for standalone indicators (less likely to need context)
    for indicator in STANDALONE_INDICATORS:
        if question_lower.startswith(indicator):
            return False

    # Questions starting with pronouns need context
    if words and words[0] in ["that", "this", "it", "they", "those", "these"]:
        return True

    # Default: if question is short and vague, assume it needs context
    return len(words) <= 6


def contextualize_query(
    question: str,
    conversation_history: list[dict],
    get_llm_response: callable = None,
) -> tuple[str, bool]:
    """
    Rewrite a follow-up question to be standalone using conversation context.

    Args:
        question: The user's current question
        conversation_history: Previous messages [{"role": "user/assistant", "content": "..."}]
        get_llm_response: Function to call LLM

    Returns:
        Tuple of (rewritten_question, was_rewritten)
    """
    # Check if contextualization is needed
    if not needs_contextualization(question):
        return question, False

    # Need at least one previous exchange
    if not conversation_history or len(conversation_history) < 2:
        return question, False

    if not get_llm_response:
        logger.warning("No LLM for query contextualization")
        return question, False

    # Get recent conversation (last 2 exchanges = 4 messages)
    recent = conversation_history[-4:]

    # Build context
    context_parts = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Truncate long responses
        if role == "assistant" and len(content) > 500:
            content = content[:500] + "..."
        context_parts.append(f"{role.upper()}: {content}")

    context = "\n".join(context_parts)

    prompt = f"""Previous conversation:
{context}

Current question: "{question}"

The current question appears to be a follow-up that references the previous conversation.
Rewrite it as a standalone question that includes all necessary context.

For example:
- "Can you expand on that?" → "Can you provide more details about electronic CDD requirements and policies?"
- "What about the risks?" → "What are the risks associated with customer due diligence procedures?"

Rewrite the question to be self-contained. Return ONLY the rewritten question, nothing else."""

    try:
        response = get_llm_response(
            [{"role": "user", "content": prompt}],
            "You rewrite follow-up questions to be standalone. Return only the rewritten question.",
        )

        # Clean up response
        rewritten = response.strip()

        # Remove quotes if present
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        if rewritten.startswith("'") and rewritten.endswith("'"):
            rewritten = rewritten[1:-1]

        # Sanity check - rewritten should be longer and different
        if len(rewritten) > len(question) and rewritten.lower() != question.lower():
            logger.info(f"Contextualized query: '{question}' → '{rewritten}'")
            return rewritten, True

    except Exception as e:
        logger.warning(f"Query contextualization failed: {e}")

    return question, False


def refine_question(
    original_question: str,
    conversation_history: list[dict] = None,
    get_llm_response: Callable = None,
) -> dict:
    """
    Generate refined question suggestions using AI.

    Args:
        original_question: The user's original question
        conversation_history: Previous messages for context
        get_llm_response: Function to call LLM (messages, system_prompt) -> str

    Returns:
        Dict with 'topic', 'understanding', 'suggestions'
    """
    if not get_llm_response:
        logger.warning("No LLM function provided for refinement")
        return {
            "topic": "",
            "understanding": "",
            "suggestions": [{"text": original_question, "angle": "Original question"}],
        }

    # Build context from conversation history
    context = ""
    if conversation_history:
        recent = conversation_history[-4:]  # Last 2 exchanges
        context = "\n\nConversation context:\n"
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]
            context += f"{role}: {content}\n"

    prompt = f"""Question to optimize: "{original_question}"{context}

Generate 5 alternative phrasings to improve search results. Return valid JSON only."""

    try:
        response = get_llm_response(
            [{"role": "user", "content": prompt}],
            REFINEMENT_SYSTEM_PROMPT,
        )

        # Parse JSON response
        # Try to extract JSON from response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response[json_start:json_end])
            return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse refinement JSON: {e}")
    except Exception as e:
        logger.error(f"Refinement failed: {e}")

    # Fallback
    return {
        "topic": "",
        "understanding": "",
        "suggestions": [{"text": original_question, "angle": "Original question"}],
    }


def combine_questions_for_search(
    questions: list[str],
    topic: str = "",
    original_question: str = "",
) -> str:
    """
    Combine multiple question phrasings into a structured search query.

    Args:
        questions: List of question phrasings
        topic: Topic heading
        original_question: The original question

    Returns:
        Combined search string
    """
    parts = []

    if topic:
        parts.append(f"## {topic}")

    if original_question:
        parts.append(f"- {original_question}")

    for q in questions:
        if q != original_question:
            parts.append(f"- {q}")

    return "\n".join(parts)


def get_quality_metrics(chunks: list[dict]) -> dict:
    """
    Calculate quality metrics for a set of chunks.

    Returns:
        Dict with metrics: positive_count, avg_score, top_score, score_spread
    """
    if not chunks:
        return {
            "positive_count": 0,
            "avg_score": 0,
            "top_score": 0,
            "score_spread": 0,
            "high_quality_count": 0,  # Chunks with score > 0.5
        }

    scores = [c.get("_rerank_score", c.get("_combined_score", 0)) for c in chunks]
    positive_scores = [s for s in scores if s > 0]

    return {
        "positive_count": len(positive_scores),
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "top_score": max(scores) if scores else 0,
        "score_spread": max(scores) - min(scores) if len(scores) > 1 else 0,
        "high_quality_count": sum(1 for s in scores if s > 0.5),
    }


def analyze_search_sufficiency(
    question: str,
    chunks: list[dict],
    get_llm_response: Callable = None,
) -> dict:
    """
    Analyze if retrieved chunks can answer the question.

    Uses a multi-factor approach:
    1. Score quality - are the top results actually relevant?
    2. Content analysis - does content address the question?
    3. Reference detection - are we finding references instead of data?

    Args:
        question: The user's question
        chunks: Retrieved chunks
        get_llm_response: LLM function

    Returns:
        Dict with 'sufficient', 'reason', 'missing_info', 'better_search_terms', 'quality_metrics'
    """
    if not chunks:
        return {
            "sufficient": False,
            "reason": "No chunks found",
            "missing_info": ["No results to analyze"],
            "better_search_terms": [],
            "quality_metrics": get_quality_metrics([]),
        }

    # Calculate quality metrics
    metrics = get_quality_metrics(chunks[:20])

    # CRITICAL: Check score quality first
    # If top results have poor scores, search hasn't found good matches
    if metrics["top_score"] < 0.3:
        return {
            "sufficient": False,
            "reason": f"Low relevance scores (top={metrics['top_score']:.2f}, need >0.3). Search terms may not match document language.",
            "missing_info": ["Better matching content"],
            "better_search_terms": _suggest_alternative_terms(question),
            "quality_metrics": metrics,
        }

    if metrics["positive_count"] < 2:
        return {
            "sufficient": False,
            "reason": f"Only {metrics['positive_count']} chunk(s) with positive relevance. Need broader search.",
            "missing_info": ["More relevant content"],
            "better_search_terms": _suggest_alternative_terms(question),
            "quality_metrics": metrics,
        }

    # Check for reference-only results (telling user WHERE to find info, not the info itself)
    reference_phrases = [
        "refer to appendix", "see appendix", "listed in appendix",
        "detailed in appendix", "as per appendix", "consult the",
        "refer to section", "see section", "outlined in",
    ]

    top_chunks = chunks[:5]
    reference_only_count = 0
    content_found = False

    for chunk in top_chunks:
        content = chunk.get("content", "").lower()
        source = chunk.get("metadata", {}).get("source_file", "").lower()
        score = chunk.get("_rerank_score", 0)

        # Only check high-scoring chunks
        if score < 0:
            continue

        # Check if this has actual content vs references
        has_reference = any(ref in content for ref in reference_phrases)

        # Check for actual data indicators
        has_data = (
            source.endswith((".xlsx", ".xls", ".csv")) or
            content.count("\n") > 10 or  # Multi-line content
            len(content) > 500  # Substantial content
        )

        if has_reference and not has_data:
            reference_only_count += 1
        elif has_data or score > 0.5:
            content_found = True

    # If all top chunks are just references, continue searching
    if reference_only_count >= 3 and not content_found:
        return {
            "sufficient": False,
            "reason": "Top results are references to other documents, not actual content",
            "missing_info": ["Actual content from referenced sections"],
            "better_search_terms": ["appendix", "schedule", "table", "list"],
            "quality_metrics": metrics,
        }

    # Check if this is a list-type query that needs comprehensive results
    question_lower = question.lower()
    is_list_query = any(term in question_lower for term in [
        "list of", "all the", "every", "complete list", "full list",
        "group by", "categorize", "enumerate",
    ])

    # For list queries, require more high-quality chunks
    if is_list_query and metrics["high_quality_count"] < 3:
        return {
            "sufficient": False,
            "reason": f"List query needs more comprehensive results (only {metrics['high_quality_count']} high-quality chunks found)",
            "missing_info": ["More complete list data"],
            "better_search_terms": ["list", "reference", "complete"],
            "quality_metrics": metrics,
        }

    # Good quality results found
    if metrics["high_quality_count"] >= 2 or (content_found and metrics["top_score"] > 0.5):
        return {
            "sufficient": True,
            "reason": f"Found {metrics['high_quality_count']} high-quality chunks (top_score={metrics['top_score']:.2f})",
            "missing_info": [],
            "better_search_terms": [],
            "quality_metrics": metrics,
        }

    # Borderline case - use LLM if available
    if not get_llm_response:
        # Without LLM, accept if we have at least some positive results
        if metrics["positive_count"] >= 3:
            return {
                "sufficient": True,
                "reason": f"Acceptable quality ({metrics['positive_count']} positive chunks, no LLM for deeper analysis)",
                "missing_info": [],
                "better_search_terms": [],
                "quality_metrics": metrics,
            }
        return {
            "sufficient": False,
            "reason": "Insufficient quality results, no LLM for deeper analysis",
            "missing_info": ["Better matching content"],
            "better_search_terms": _suggest_alternative_terms(question),
            "quality_metrics": metrics,
        }

    # Use LLM for nuanced analysis
    chunk_summaries = []
    for i, chunk in enumerate(chunks[:10], 1):
        source = chunk.get("metadata", {}).get("source_file", "Unknown")
        score = chunk.get("_rerank_score", 0)
        content = chunk.get("content", "")[:300]
        chunk_summaries.append(f"[{i}] (score:{score:.2f}) {source}: {content}...")

    chunks_text = "\n".join(chunk_summaries)

    prompt = f"""Question: "{question}"

Quality metrics: top_score={metrics['top_score']:.2f}, positive_chunks={metrics['positive_count']}, high_quality={metrics['high_quality_count']}

Retrieved content (with relevance scores):
{chunks_text}

Analyze if this content can answer the question. Consider:
1. Do the scores indicate good relevance? (>0.5 is good, <0 is poor)
2. Does the content directly address the question, or just reference other documents?
3. Is there enough specific information to give a complete answer?

Return JSON:
{{
    "sufficient": true/false,
    "reason": "Brief explanation focusing on content quality and relevance",
    "missing_info": ["What specific information is missing, if any"],
    "better_search_terms": ["Alternative terms to try if insufficient"]
}}"""

    system = "Analyze search results for relevance. Focus on whether content directly answers the question. Return valid JSON only."

    try:
        response = get_llm_response([{"role": "user", "content": prompt}], system)

        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response[json_start:json_end])
            result["quality_metrics"] = metrics
            return result

    except Exception as e:
        logger.warning(f"Sufficiency analysis failed: {e}")

    # Fallback based on metrics
    return {
        "sufficient": metrics["positive_count"] >= 3,
        "reason": f"LLM analysis failed, using metrics: {metrics['positive_count']} positive chunks",
        "missing_info": [],
        "better_search_terms": [],
        "quality_metrics": metrics,
    }


def _suggest_alternative_terms(question: str) -> list[str]:
    """Generate basic alternative search terms from question."""
    question_lower = question.lower()

    suggestions = []

    # Common term mappings for compliance/financial domains
    term_alternatives = {
        "cdd": ["customer due diligence", "verification", "identity verification", "KYC"],
        "kyc": ["know your customer", "CDD", "customer verification"],
        "aml": ["anti-money laundering", "financial crime", "money laundering"],
        "pep": ["politically exposed person", "political exposure"],
        "high risk": ["enhanced due diligence", "high-risk countries", "risk rating"],
        "electronic": ["digital", "online", "e-verification", "electronic verification"],
        "rely": ["reliance", "acceptance", "accept", "use"],
    }

    for term, alternatives in term_alternatives.items():
        if term in question_lower:
            suggestions.extend(alternatives[:2])

    # Add some generic expansions
    if not suggestions:
        words = question_lower.split()
        if len(words) > 3:
            # Try key phrases
            suggestions.append(" ".join(words[1:4]))
            suggestions.append(" ".join(words[-3:]))

    return suggestions[:4]


def generate_query_variants(
    query: str,
    count: int = 3,
    get_llm_response: Callable = None,
) -> list[str]:
    """
    Generate diverse query variants for multi-query / RAG-Fusion search.

    Unlike generate_alternative_queries (which uses iteration context and missing info),
    this generates upfront reformulations with different terminology and specificity.

    Args:
        query: Original search query
        count: Number of variants to generate (excludes original)
        get_llm_response: LLM function (messages, system_prompt) -> str

    Returns:
        List starting with the original query, followed by up to `count` variants.
        Falls back to [query] on any failure.
    """
    if not get_llm_response:
        return [query]

    prompt = f"""Generate {count} diverse reformulations of this search query.
Each reformulation should use different terminology, angles, or specificity levels.

Original query: "{query}"

Guidelines:
- Use synonyms and related technical terms
- Vary between specific and general phrasing
- Include different aspects of the topic
- Each variant should find different relevant documents

Return ONLY a JSON array of strings: ["variant1", "variant2", ...]"""

    try:
        response = get_llm_response(
            [{"role": "user", "content": prompt}],
            "Generate search query variants. Return a JSON array of strings only.",
        )

        json_start = response.find("[")
        json_end = response.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            variants = json.loads(response[json_start:json_end])
            if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
                # Always start with original
                return [query] + variants[:count]

    except Exception as e:
        logger.warning(f"Query variant generation failed: {e}")

    return [query]


def generate_alternative_queries(
    question: str,
    previous_queries: list[str],
    missing_info: list[str],
    get_llm_response: Callable = None,
) -> list[str]:
    """
    Generate alternative search queries based on what's missing.

    Args:
        question: Original question
        previous_queries: Queries already tried
        missing_info: Information still needed
        get_llm_response: LLM function

    Returns:
        List of new queries to try
    """
    if not get_llm_response:
        return []

    prompt = f"""Original question: "{question}"

Previous searches tried:
{chr(10).join(f'- {q}' for q in previous_queries)}

Still missing:
{chr(10).join(f'- {m}' for m in missing_info)}

Generate 3 different search queries that might find the missing information.
Focus on specific terms, document references, or alternative phrasings.

Return as JSON array: ["query1", "query2", "query3"]"""

    try:
        response = get_llm_response(
            [{"role": "user", "content": prompt}],
            "Generate search queries. Return JSON array only.",
        )

        json_start = response.find("[")
        json_end = response.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(response[json_start:json_end])

    except Exception as e:
        logger.warning(f"Query generation failed: {e}")

    return []


def create_refinement_log(
    original_question: str,
    final_question: str,
    suggestions: list[dict],
    selected_questions: list[str] = None,
    topic: str = "",
    start_time: datetime = None,
) -> dict:
    """
    Create a log of the refinement session.

    Args:
        original_question: Original user question
        final_question: Question used for search
        suggestions: Suggestions offered
        selected_questions: Questions user selected
        topic: Topic identified
        start_time: When refinement started

    Returns:
        Log entry dict
    """
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() if start_time else 0

    return {
        "timestamp": end_time.isoformat(),
        "original_question": original_question,
        "topic": topic,
        "suggestions_offered": len(suggestions),
        "suggestions": suggestions,
        "selected_questions": selected_questions or [],
        "final_question": final_question,
        "duration_seconds": duration,
    }
