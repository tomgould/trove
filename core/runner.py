"""
Local agent runner — runs Trove agents end-to-end using local LLM.

Combines:
- Agent template (system prompt from YAML)
- Hybrid search (RAG context from Trove)
- LLM generation (via LM Studio or Claude)
- Optional memory persistence

Usage:
    from core.runner import AgentRunner
    from core.agents import AgentLoader
    from core.llm import create_provider_from_config

    agent = loader.get("code-reviewer")
    runner = AgentRunner(agent, client, llm_provider, searcher)
    result = runner.run("Review the search pipeline")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from core.llm import LLMProvider, build_rag_prompt
from core.search import HybridSearcher, SearchQuality

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent run."""

    response: str
    sources: list[dict] = field(default_factory=list)
    search_quality: Optional[SearchQuality] = None
    timing: dict = field(default_factory=dict)
    citations_used: list[dict] = field(default_factory=list)
    agent_name: str = ""
    query: str = ""


class AgentRunner:
    """Runs a Trove agent: search for context, inject system prompt, generate response.

    Args:
        agent: Agent dict from AgentLoader.get() (must have system_prompt)
        client: Trove Client instance (for config like max_context_chars)
        llm_provider: LLMProvider instance for generation
        searcher: HybridSearcher instance for RAG context
        memory_store: Optional MemoryStore for persisting results
    """

    def __init__(
        self,
        agent: dict,
        client,
        llm_provider: LLMProvider,
        searcher: HybridSearcher,
        memory_store=None,
    ):
        self.agent = agent
        self.client = client
        self.llm_provider = llm_provider
        self.searcher = searcher
        self.memory_store = memory_store

    def run(
        self,
        query: str,
        top_k: int = 10,
        conversation_history: Optional[list[dict]] = None,
        store_memory: bool = False,
        max_tokens: Optional[int] = None,
    ) -> AgentResult:
        """Run the agent on a query.

        1. Search Trove for relevant context
        2. Build RAG prompt with retrieved chunks
        3. Prepend conversation history if provided
        4. Generate response via LLM with agent system prompt
        5. Optionally store result in MemoryStore

        Args:
            query: User question or task
            top_k: Number of search results to retrieve
            conversation_history: Prior messages for multi-turn conversation
            store_memory: If True, store result in MemoryStore
            max_tokens: Override max tokens for generation

        Returns:
            AgentResult with response, sources, quality, and timing
        """
        timings = {}
        agent_name = self.agent.get("name", "Unknown")
        system_prompt = self.agent.get("system_prompt", "")

        # 1. Search for RAG context
        t0 = time.time()
        search_result = self.searcher.search_with_quality(query=query, top_k=top_k)
        timings["search_ms"] = round((time.time() - t0) * 1000)

        chunks = search_result.results
        quality = search_result.quality

        logger.info(
            f"Agent '{agent_name}': retrieved {len(chunks)} chunks for query "
            f"({timings['search_ms']}ms)"
        )

        # 2. Build RAG prompt
        llm_config = self.client.llm_config if hasattr(self.client, "llm_config") else {}
        max_context_chars = llm_config.get("max_context_chars", 12000)

        messages = build_rag_prompt(
            question=query,
            chunks=chunks,
            max_context_chars=max_context_chars,
        )

        # 3. Prepend conversation history
        if conversation_history:
            messages = conversation_history + messages

        # 4. Generate response
        gen_kwargs = {"system_prompt": system_prompt}
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens

        t1 = time.time()
        response = self.llm_provider.generate(messages, **gen_kwargs)
        timings["generate_ms"] = round((time.time() - t1) * 1000)
        timings["total_ms"] = timings["search_ms"] + timings["generate_ms"]

        # Build sources list
        sources = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            sources.append({
                "source": metadata.get("source_file", "Unknown"),
                "score": chunk.get("_rerank_score", chunk.get("_combined_score", 0)),
                "chunk_type": metadata.get("chunk_type", "document"),
            })

        result = AgentResult(
            response=response,
            sources=sources,
            search_quality=quality,
            timing=timings,
            agent_name=agent_name,
            query=query,
        )

        # 5. Optionally store in memory
        if store_memory and self.memory_store:
            namespace = f"agent:{self.agent.get('_filename', agent_name)}"
            self.memory_store.store(
                namespace=namespace,
                key=query[:200],
                content=response[:2000],
                metadata={"agent": agent_name, "timing": timings},
            )
            logger.info(f"Stored result in memory namespace '{namespace}'")

        return result

    def chat(self, top_k: int = 10):
        """Interactive REPL with conversation history.

        Runs an interactive loop, maintaining conversation history across turns.
        Type 'quit' or 'exit' to stop.

        Args:
            top_k: Number of search results per turn
        """
        agent_name = self.agent.get("name", "Unknown")
        print(f"\nAgent: {agent_name}")
        print(f"Client: {self.client.name}")
        print("Type 'quit' or 'exit' to stop.\n")

        history: list[dict] = []

        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit"):
                print("Bye!")
                break

            result = self.run(
                query=query,
                top_k=top_k,
                conversation_history=history if history else None,
            )

            print(f"\n{agent_name}: {result.response}")
            print(
                f"  [{len(result.sources)} sources, "
                f"{result.timing.get('total_ms', 0)}ms]\n"
            )

            # Append to history
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": result.response})
