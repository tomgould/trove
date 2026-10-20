"""
Trove Core Module

Components:
- client: Client configuration and loading
- ingest: Document ingestion with Unstructured
- search: Hybrid search (BM25 + vector) with re-ranking
- llm: LLM provider integration (LM Studio, Claude)
- refinement: Query refinement and contextualization
- reference_resolver: Detect and resolve document references
- reference_config: Reference resolution configuration
- reranker: Cross-encoder reranking backends
- evaluation: Retrieval metrics and LLM-as-judge
- chunking: Chunking strategies (title, semantic, recursive)
- ingest_progress: Progress tracking and resumable ingestion
- ingest_job: Ingestion job management for MCP tools
- logging_config: Audit configuration loading
- memory: Persistent memory store (SQLite + vector search)
- workflow: Workflow DAG engine with topological ordering
- agents: YAML agent template loader
- runner: Local agent runner (search + agent prompt + LLM generation)

Usage:
    from core.client import load_client
    from core.search import HybridSearcher
    from core.llm import get_provider
    from core.ingest import DocumentIngestor
    from core.memory import MemoryStore
    from core.workflow import WorkflowEngine
    from core.agents import AgentLoader
"""
