"""
LLM integration for response generation.

Supports:
- LM Studio Native API (v1 REST API with model management)
- LM Studio Anthropic-compatible API
- LM Studio OpenAI-compatible API (legacy)
- Claude (Anthropic API)
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import requests

from openai import OpenAI
from anthropic import Anthropic

from core.plugins.interfaces import BaseLLMProvider

logger = logging.getLogger(__name__)

# Error prefix used by all LLM providers for machine-detectable error responses
LLM_ERROR_PREFIX = "Error: "


def is_llm_error(response: str) -> bool:
    """Check if an LLM response is an error message rather than valid content."""
    return response.startswith(LLM_ERROR_PREFIX)


def _retry_with_backoff(func, max_retries=2, base_delay=1.0, retryable_exceptions=(requests.ConnectionError, requests.Timeout)):
    """Retry a function with exponential backoff for transient errors."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Transient error (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay}s...")
            time.sleep(delay)

# Default settings
DEFAULT_LMSTUDIO_HOST = "http://localhost:1234"
DEFAULT_LMSTUDIO_URL = f"{DEFAULT_LMSTUDIO_HOST}/v1"  # OpenAI-compatible
DEFAULT_LMSTUDIO_MODEL = "local-model"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 2000


class LLMProvider(BaseLLMProvider):
    """Base class for LLM providers.

    Implements BaseLLMProvider from the plugin system. All built-in
    and plugin LLM providers inherit from this.
    """

    def generate(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError


class LMStudioNativeProvider(LLMProvider):
    """
    LM Studio provider using the native v1 REST API.

    Features:
    - Model management (list, load, unload)
    - Context length specification in requests
    - Streaming events for model loading and prompt processing
    """

    name = "lmstudio_native"

    def __init__(
        self,
        host: str = DEFAULT_LMSTUDIO_HOST,
        model: Optional[str] = None,
        context_length: Optional[int] = None,
        api_token: Optional[str] = None,
    ):
        self.host = host.rstrip("/")
        self.api_base = f"{self.host}/api/v1"
        self.model = model
        self.context_length = context_length
        self.headers = {"Content-Type": "application/json"}
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"
        logger.info(f"Initialized LM Studio Native provider at {self.host}")

    def list_models(self) -> list[dict]:
        """List available models in LM Studio."""
        try:
            response = requests.get(f"{self.api_base}/models", headers=self.headers)
            response.raise_for_status()
            # Native API returns {"models": [...]}
            return response.json().get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_loaded_model(self) -> Optional[dict]:
        """Get the currently loaded model, if any."""
        models = self.list_models()
        for model in models:
            # Models with loaded_instances are loaded
            if model.get("loaded_instances"):
                return model
        return None

    def load_model(
        self,
        model_path: str,
        context_length: Optional[int] = None,
        gpu_offload: Optional[float] = None,
    ) -> bool:
        """
        Load a model in LM Studio.

        Args:
            model_path: Path to the model (e.g., "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF")
            context_length: Optional context length override
            gpu_offload: GPU offload ratio (0.0-1.0)

        Returns:
            True if successful
        """
        payload = {"model": model_path}
        if context_length:
            payload["context_length"] = context_length
        if gpu_offload is not None:
            payload["gpu_offload"] = gpu_offload

        try:
            response = requests.post(
                f"{self.api_base}/models/load",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            logger.info(f"Loaded model: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def unload_model(self, model_path: Optional[str] = None) -> bool:
        """Unload a model from LM Studio."""
        payload = {}
        if model_path:
            payload["model"] = model_path

        try:
            response = requests.post(
                f"{self.api_base}/models/unload",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            logger.info(f"Unloaded model: {model_path or 'current'}")
            return True
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return False

    def generate(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        context_length: Optional[int] = None,
    ) -> str:
        """
        Generate response using LM Studio native API.

        The native API uses a different format:
        - Request: {"model": "...", "input": "...", "config": {...}}
        - Response: {"output": [{"type": "message", "content": "..."}]}

        Args:
            messages: Chat messages
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            context_length: Override context length for this request
        """
        # Build input text from messages (native API uses simple input string or structured format)
        # For more complex conversations, use the messages format
        input_parts = []
        if system_prompt:
            input_parts.append(f"System: {system_prompt}\n")

        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            input_parts.append(f"{role}: {content}")

        input_text = "\n".join(input_parts)

        # Determine model to use - prefer loaded model if not specified
        model = self.model
        if not model or model == "local-model":
            loaded = self.get_loaded_model()
            if loaded:
                # Use the loaded instance ID
                instances = loaded.get("loaded_instances", [])
                if instances:
                    model = instances[0].get("id")
                else:
                    model = loaded.get("key")

        if not model:
            return "Error: No model loaded in LM Studio. Please load a model first."

        payload = {
            "model": model,
            "input": input_text,
        }

        # Add config options if specified
        config = {}
        ctx_len = context_length or self.context_length
        if ctx_len:
            config["context_length"] = ctx_len
        if config:
            payload["config"] = config

        try:
            def _do_request():
                resp = requests.post(
                    f"{self.api_base}/chat",
                    json=payload,
                    headers=self.headers,
                    timeout=300,  # 5 minute timeout for long generations
                )
                resp.raise_for_status()
                return resp

            response = _retry_with_backoff(
                _do_request,
                retryable_exceptions=(requests.ConnectionError, requests.Timeout),
            )
            data = response.json()

            # Native API returns: {"output": [{"type": "message", "content": "..."}]}
            output = data.get("output", [])
            for item in output:
                if item.get("type") == "message":
                    return item.get("content", "")

            return ""

        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to LM Studio. Please ensure it's running."
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model may be processing a very long response."
        except Exception as e:
            error_msg = str(e)
            if "context" in error_msg.lower() or "token" in error_msg.lower():
                return f"Error: Context too long. Try reducing max_chunks. ({error_msg})"
            return f"Error: {error_msg}"


class LMStudioAnthropicProvider(LLMProvider):
    """
    LM Studio provider using the Anthropic-compatible API.

    This uses the same message format as Claude, making it easy to
    switch between local and cloud inference.
    """

    name = "lmstudio_anthropic"

    def __init__(
        self,
        host: str = DEFAULT_LMSTUDIO_HOST,
        model: str = DEFAULT_LMSTUDIO_MODEL,
    ):
        self.host = host.rstrip("/")
        # Anthropic-compatible endpoint
        self.api_base = f"{self.host}/v1"
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": "not-needed",  # LM Studio doesn't require a real key
            "anthropic-version": "2023-06-01",
        }
        logger.info(f"Initialized LM Studio Anthropic provider at {self.host}")

    def generate(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Generate response using Anthropic-compatible API."""
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            def _do_request():
                resp = requests.post(
                    f"{self.api_base}/messages",
                    json=payload,
                    headers=self.headers,
                )
                resp.raise_for_status()
                return resp

            response = _retry_with_backoff(
                _do_request,
                retryable_exceptions=(requests.ConnectionError, requests.Timeout),
            )
            data = response.json()

            # Anthropic format: content is a list of content blocks
            content = data.get("content", [])
            if content and isinstance(content, list):
                # Get text from first text block
                for block in content:
                    if block.get("type") == "text":
                        return block.get("text", "")
            return ""

        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to LM Studio. Please ensure it's running."
        except Exception as e:
            error_msg = str(e)
            if "context" in error_msg.lower() or "token" in error_msg.lower():
                return f"Error: Context too long. Try reducing max_chunks. ({error_msg})"
            return f"Error: {error_msg}"


class LMStudioProvider(LLMProvider):
    """LM Studio provider (local LLM via OpenAI-compatible API)."""

    name = "lmstudio"

    def __init__(
        self,
        base_url: str = DEFAULT_LMSTUDIO_URL,
        model: str = DEFAULT_LMSTUDIO_MODEL,
    ):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        logger.info(f"Initialized LM Studio OpenAI-compatible provider at {base_url}")

    def generate(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Generate response using LM Studio."""
        full_messages = []

        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        full_messages.extend(messages)

        try:
            import openai as _openai

            def _do_request():
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )

            response = _retry_with_backoff(
                _do_request,
                retryable_exceptions=(_openai.APIConnectionError, _openai.APITimeoutError),
            )
            if not response.choices:
                return "Error: LM Studio returned empty response"
            return response.choices[0].message.content or ""

        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg:
                return "Error: Cannot connect to LM Studio. Please ensure it's running."
            if "context" in error_msg.lower() or "token" in error_msg.lower():
                return f"Error: Context too long. Try reducing max_chunks. ({error_msg})"
            return f"Error: {error_msg}"


class ClaudeProvider(LLMProvider):
    """Claude provider (Anthropic API)."""

    name = "claude"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_CLAUDE_MODEL,
    ):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = Anthropic(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Claude provider with model {model}")

    def generate(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Generate response using Claude."""
        try:
            import anthropic as _anthropic

            def _do_request():
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt or "",
                    messages=messages,
                )

            response = _retry_with_backoff(
                _do_request,
                retryable_exceptions=(_anthropic.APIConnectionError, _anthropic.APITimeoutError),
            )
            if not response.content:
                return "Error: Claude returned empty response"
            return response.content[0].text

        except Exception as e:
            return f"Error: {str(e)}"


def get_provider(
    provider_name: str = "lmstudio",
    **kwargs,
) -> LLMProvider:
    """
    Get an LLM provider by name.

    Args:
        provider_name: Provider name, one of:
            - "lmstudio" or "lmstudio_openai": OpenAI-compatible API (legacy)
            - "lmstudio_anthropic": Anthropic-compatible API (recommended for local)
            - "lmstudio_native": Native v1 API with model management
            - "claude": Anthropic API (cloud)
        **kwargs: Provider-specific arguments

    Returns:
        LLMProvider instance
    """
    if provider_name in ("lmstudio", "lmstudio_openai"):
        return LMStudioProvider(**kwargs)
    elif provider_name == "lmstudio_anthropic":
        return LMStudioAnthropicProvider(**kwargs)
    elif provider_name == "lmstudio_native":
        return LMStudioNativeProvider(**kwargs)
    elif provider_name == "claude":
        return ClaudeProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def create_provider_from_config(llm_config: dict) -> LLMProvider:
    """Create an LLM provider from a client llm_config dict.

    Extracts provider type and connection settings from the config,
    instantiating the appropriate provider class.

    Args:
        llm_config: Dict from client.llm_config (provider, lmstudio_host, etc.)

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider type is unknown or required config is missing
    """
    provider_type = llm_config.get("provider", "lmstudio")
    host = llm_config.get("lmstudio_host", DEFAULT_LMSTUDIO_HOST)
    model = llm_config.get("lmstudio_model", DEFAULT_LMSTUDIO_MODEL)

    if provider_type == "lmstudio_anthropic":
        return LMStudioAnthropicProvider(host=host, model=model)
    elif provider_type == "lmstudio_native":
        return LMStudioNativeProvider(host=host, model=model)
    elif provider_type in ("lmstudio", "lmstudio_openai"):
        return LMStudioProvider(base_url=f"{host}/v1", model=model)
    elif provider_type == "claude":
        api_key = llm_config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY required for claude provider")
        return ClaudeProvider(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")


# Default system prompt for RAG
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant answering questions based on provided documents.

Instructions:
1. Base your answers on the provided context
2. Cite sources using [Source N] notation
3. If the context doesn't contain the answer, say so clearly
4. Be precise and accurate
5. For lists, include ALL items from the sources - don't summarize

Context will be provided with each question."""


CITATION_SYSTEM_PROMPT_SUFFIX = """

IMPORTANT: When citing sources, use numbered references like [1], [2], etc. that correspond to the source numbers provided in the context. Place citations inline immediately after the relevant claim. You may cite multiple sources for a single claim like [1][3]. Only cite sources that you actually use."""


@dataclass
class CitationMap:
    """Maps citation numbers to source metadata."""

    entries: dict[int, dict] = field(default_factory=dict)

    def get(self, num: int) -> Optional[dict]:
        return self.entries.get(num)

    def __len__(self) -> int:
        return len(self.entries)


def build_citation_map(chunks: list[dict]) -> CitationMap:
    """
    Build a citation map from chunks, numbering them sequentially.

    Each entry contains source file, page, section, score, and content preview.
    """
    cmap = CitationMap()
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        cmap.entries[i] = {
            "source": metadata.get("source_file", "Unknown"),
            "page": metadata.get("page_number"),
            "section": metadata.get("section"),
            "score": chunk.get("_rerank_score", chunk.get("_combined_score", 0)),
            "content_preview": chunk.get("content", "")[:150],
        }
    return cmap


def parse_citations_from_response(
    response: str,
    citation_map: CitationMap,
) -> list[dict]:
    """
    Extract used citations from an LLM response.

    Returns list of citation entries in order of first appearance.
    """
    # Find all [N] patterns
    matches = re.findall(r'\[(\d+)\]', response)

    seen = set()
    used = []
    for match in matches:
        num = int(match)
        if num not in seen and citation_map.get(num) is not None:
            seen.add(num)
            entry = citation_map.get(num).copy()
            entry["citation_number"] = num
            used.append(entry)

    return used


def build_rag_prompt_with_citations(
    question: str,
    chunks: list[dict],
    max_context_chars: int = 12000,
    **kwargs,
) -> tuple[list[dict], CitationMap]:
    """
    Build a RAG prompt with citation-style formatting and return the citation map.

    Returns:
        Tuple of (messages, citation_map)
    """
    citation_map = build_citation_map(chunks)
    messages = build_rag_prompt(
        question=question,
        chunks=chunks,
        max_context_chars=max_context_chars,
        citations=True,
        **kwargs,
    )
    return messages, citation_map


def categorize_chunks(chunks: list[dict]) -> dict[str, list[dict]]:
    """
    Categorize chunks by type for budget allocation.

    Returns:
        Dict with keys "doc", "code", "reference" mapping to chunk lists.
    """
    categories: dict[str, list[dict]] = {"doc": [], "code": [], "reference": []}

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if chunk.get("_resolved_reference") or chunk.get("_cross_referenced"):
            categories["reference"].append(chunk)
        elif metadata.get("chunk_type") == "code":
            categories["code"].append(chunk)
        else:
            categories["doc"].append(chunk)

    return categories


def allocate_budgets(
    total_chars: int,
    doc_ratio: float = 0.6,
    code_ratio: float = 0.3,
    ref_ratio: float = 0.1,
    counts: dict[str, int] = None,
) -> dict[str, int]:
    """
    Allocate character budgets proportionally, redistributing from empty categories.

    Args:
        total_chars: Total character budget
        doc_ratio: Base ratio for document chunks
        code_ratio: Base ratio for code chunks
        ref_ratio: Base ratio for reference chunks
        counts: Dict with "doc", "code", "reference" counts

    Returns:
        Dict with "doc", "code", "reference" character budgets
    """
    counts = counts or {"doc": 1, "code": 1, "reference": 1}
    ratios = {"doc": doc_ratio, "code": code_ratio, "reference": ref_ratio}

    # Zero out ratios for empty categories
    active_ratios = {k: v for k, v in ratios.items() if counts.get(k, 0) > 0}

    if not active_ratios:
        return {"doc": total_chars, "code": 0, "reference": 0}

    # Redistribute proportionally
    total_ratio = sum(active_ratios.values())
    budgets = {}
    for key in ("doc", "code", "reference"):
        if key in active_ratios:
            budgets[key] = int(total_chars * active_ratios[key] / total_ratio)
        else:
            budgets[key] = 0

    return budgets


def _build_context_parts(
    chunks: list[dict],
    max_chars: int,
    start_index: int = 1,
    citations: bool = False,
) -> tuple[list[str], int, int]:
    """
    Build context parts from chunks within a character budget.

    Returns:
        Tuple of (parts, total_chars, next_index)
    """
    parts = []
    total_chars = 0
    idx = start_index

    for chunk in chunks:
        source = chunk.get("metadata", {}).get("source_file", "Unknown")
        content = chunk["content"]

        if citations:
            part = f"[{idx}] {source}\n{content}\n"
        else:
            part = f"[Source {idx}: {source}]\n{content}\n"

        if total_chars + len(part) > max_chars:
            break

        parts.append(part)
        total_chars += len(part)
        idx += 1

    return parts, total_chars, idx


def build_rag_prompt(
    question: str,
    chunks: list[dict],
    max_context_chars: int = 12000,
    best_at_end: bool = False,
    citations: bool = False,
    doc_budget_ratio: float = None,
    code_budget_ratio: float = None,
    reference_budget_ratio: float = None,
) -> list[dict]:
    """
    Build a RAG prompt with context from retrieved chunks.

    Args:
        question: User's question
        chunks: Retrieved chunks (assumed to be sorted by relevance, best first)
        max_context_chars: Maximum characters for context
        best_at_end: If True, reorder chunks so best are at the end
        citations: If True, use [1] format instead of [Source 1: ...]
        doc_budget_ratio: Budget ratio for document chunks (enables categorized budgeting)
        code_budget_ratio: Budget ratio for code chunks
        reference_budget_ratio: Budget ratio for reference chunks

    Returns:
        Messages list for LLM
    """
    if best_at_end and chunks:
        chunks = sorted(
            chunks,
            key=lambda x: x.get("_rerank_score", x.get("_combined_score", 0))
        )

    use_budgets = (
        doc_budget_ratio is not None
        and code_budget_ratio is not None
        and reference_budget_ratio is not None
    )

    if use_budgets:
        categories = categorize_chunks(chunks)
        budgets = allocate_budgets(
            total_chars=max_context_chars,
            doc_ratio=doc_budget_ratio,
            code_ratio=code_budget_ratio,
            ref_ratio=reference_budget_ratio,
            counts={k: len(v) for k, v in categories.items()},
        )

        all_parts = []
        total_chars = 0
        idx = 1

        for cat_key in ("doc", "code", "reference"):
            cat_chunks = categories[cat_key]
            if not cat_chunks:
                continue
            parts, chars, idx = _build_context_parts(
                cat_chunks, budgets[cat_key], start_index=idx, citations=citations
            )
            all_parts.extend(parts)
            total_chars += chars

        context_parts = all_parts
    else:
        context_parts, total_chars, _ = _build_context_parts(
            chunks, max_context_chars, start_index=1, citations=citations
        )

    total_available = len(chunks)
    total_included = len(context_parts)
    if total_included < total_available:
        logger.warning(
            f"Context truncated: included {total_included}/{total_available} chunks "
            f"({total_chars}/{max_context_chars} chars)"
        )

    context = "\n---\n".join(context_parts)

    user_message = f"""Based on the following documents, please answer this question:

**Question:** {question}

---

**Context:**

{context}

---

Please provide a clear answer with citations to the sources."""

    return [{"role": "user", "content": user_message}]
