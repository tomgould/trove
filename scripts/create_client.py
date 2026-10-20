#!/usr/bin/env python3
"""
Interactive CLI script for creating new clients.

Usage:
    python scripts/create_client.py
"""

import re
import sys
from pathlib import Path

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.client import (
    DEFAULT_CONFIG,
    DEFAULT_DOCUMENT_EXTENSIONS,
    _KNOWN_EMBEDDING_DIMS,
    discover_clients,
    validate_config,
)

CLIENTS_DIR = Path(__file__).parent.parent / "clients"


def slugify(text: str) -> str:
    """Convert text to a valid machine name (lowercase, underscores)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "_", text)
    return text


def prompt_input(prompt: str, default: str = None, required: bool = True) -> str:
    """Prompt for input with optional default value."""
    if default:
        display = f"{prompt} [{default}]: "
    else:
        display = f"{prompt}: "

    while True:
        value = input(display).strip()
        if not value and default:
            return default
        if not value and required:
            print("  This field is required. Please enter a value.")
            continue
        return value


def prompt_choice(prompt: str, choices: list[str], default: str = None) -> str:
    """Prompt for a choice from a list."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        marker = " (default)" if choice == default else ""
        print(f"  {i}. {choice}{marker}")

    while True:
        value = input("Enter number or value: ").strip()
        if not value and default:
            return default

        # Check if it's a number
        if value.isdigit():
            idx = int(value) - 1
            if 0 <= idx < len(choices):
                return choices[idx]

        # Check if it's a valid choice
        if value in choices:
            return value

        print(f"  Please enter a number 1-{len(choices)} or one of: {', '.join(choices)}")


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt for yes/no."""
    default_str = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not value:
            return default
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'")


def prompt_float(prompt: str, default: float, min_val: float = 0, max_val: float = 1) -> float:
    """Prompt for a float value."""
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        try:
            f = float(value)
            if min_val <= f <= max_val:
                return f
            print(f"  Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("  Please enter a valid number")


def prompt_int(prompt: str, default: int, min_val: int = 1, max_val: int = 100) -> int:
    """Prompt for an integer value."""
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        try:
            i = int(value)
            if min_val <= i <= max_val:
                return i
            print(f"  Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("  Please enter a valid integer")


def create_client():
    """Interactive client creation wizard."""
    print("\n" + "=" * 50)
    print("  Trove Client Creation Wizard")
    print("=" * 50)

    # Show existing clients
    existing = discover_clients(CLIENTS_DIR)
    if existing:
        print(f"\nExisting clients: {', '.join(existing)}")

    # =========================================================================
    # Basic info
    # =========================================================================
    print("\n--- Basic Information ---\n")

    name = prompt_input("Client display name (e.g., 'Acme Corp RAG')")

    suggested_machine_name = slugify(name)
    machine_name = prompt_input("Machine name (directory name)", default=suggested_machine_name)

    # Check if client already exists
    client_path = CLIENTS_DIR / machine_name
    if client_path.exists():
        print(f"\n  Warning: Client '{machine_name}' already exists at {client_path}")
        if not prompt_yes_no("Overwrite existing client?", default=False):
            print("Aborted.")
            return

    description = prompt_input("Description (optional)", default="", required=False)

    # =========================================================================
    # Embedding settings
    # =========================================================================
    print("\n--- Embedding Model ---\n")

    embedding_choices = [
        f"{model} ({dim}d)" for model, dim in _KNOWN_EMBEDDING_DIMS.items()
        if not model.endswith("large")  # Skip large models for simplicity
    ]
    default_embedding = next(c for c in embedding_choices if "MiniLM-L6" in c)

    embedding_choice = prompt_choice(
        "Select embedding model:",
        embedding_choices,
        default=default_embedding,
    )

    # Parse model name from choice
    embedding_model_name = embedding_choice.split(" (")[0]
    embedding_dimension = _KNOWN_EMBEDDING_DIMS.get(embedding_model_name, 384)

    # Contextual embeddings (Stage 9)
    print("\nContextual embeddings prepend document context to chunks before embedding,")
    print("improving awareness of each chunk's source document.")
    contextualize = prompt_yes_no("Enable contextual embeddings? (requires reindex)", default=False)

    # HNSW tuning (Stage 10)
    configure_hnsw = prompt_yes_no("Configure HNSW index parameters? (advanced)", default=False)

    hnsw_ef_construction = 200
    hnsw_max_neighbors = 16
    hnsw_ef_search = 100

    if configure_hnsw:
        hnsw_ef_construction = prompt_int(
            "HNSW ef_construction (build-time accuracy, higher = better but slower, requires reindex)",
            default=200, min_val=50, max_val=1000,
        )
        hnsw_max_neighbors = prompt_int(
            "HNSW max_neighbors / M (connections per node, requires reindex)",
            default=16, min_val=4, max_val=64,
        )
        hnsw_ef_search = prompt_int(
            "HNSW ef_search (query-time accuracy, higher = better recall, immediate)",
            default=100, min_val=10, max_val=500,
        )

    # =========================================================================
    # Document settings
    # =========================================================================
    print("\n--- Document Types ---\n")

    default_exts = list(DEFAULT_DOCUMENT_EXTENSIONS)
    print(f"Default document extensions: {', '.join(default_exts)}")

    customize_docs = prompt_yes_no("Customize document extensions?", default=False)

    exclude_extensions = []
    if customize_docs:
        print("\nWhich document types should be EXCLUDED?")
        print("  (Enter extensions separated by commas, e.g., .csv,.xlsx)")
        print("  (Press Enter to skip)")
        excl = input("Exclude: ").strip()
        if excl:
            exclude_extensions = [
                ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}"
                for ext in excl.split(",") if ext.strip()
            ]
            if exclude_extensions:
                print(f"  Will exclude: {', '.join(exclude_extensions)}")

    # =========================================================================
    # LLM settings
    # =========================================================================
    print("\n--- LLM Settings ---\n")

    llm_provider = prompt_choice(
        "Select LLM provider:",
        [
            "lmstudio_anthropic (Anthropic-compatible API, recommended)",
            "lmstudio_native (Native API with model management)",
            "lmstudio (OpenAI-compatible API, legacy)",
            "claude (Anthropic cloud API)",
        ],
        default="lmstudio_anthropic (Anthropic-compatible API, recommended)",
    )
    # Extract provider key from choice
    llm_provider = llm_provider.split(" (")[0]

    llm_defaults = DEFAULT_CONFIG["llm"]
    max_context_chars = prompt_int(
        "Max context characters for LLM",
        default=llm_defaults["max_context_chars"],
        min_val=1000,
        max_val=100000,
    )

    max_chunks = prompt_int(
        "Max chunks to send to LLM",
        default=llm_defaults["max_chunks"],
        min_val=1,
        max_val=100,
    )

    max_tokens = prompt_int(
        "Max response tokens",
        default=2000,
        min_val=100,
        max_val=16000,
    )

    # Response quality settings
    print("\n--- Response Quality ---\n")

    citations_enabled = prompt_yes_no(
        "Enable inline citations? (LLM responses cite sources with [1], [2])",
        default=False,
    )

    context_filter_enabled = prompt_yes_no(
        "Enable context pre-filtering? (drop low-scoring chunks before LLM context)",
        default=False,
    )

    context_filter_threshold = 0.1
    if context_filter_enabled:
        context_filter_threshold = prompt_float(
            "Minimum rerank score for context inclusion",
            default=0.1,
            min_val=-5.0,
            max_val=5.0,
        )

    # =========================================================================
    # Search settings
    # =========================================================================
    print("\n--- Search Settings ---\n")

    search_defaults = DEFAULT_CONFIG["search"]

    merge_strategy = prompt_choice(
        "Merge strategy for combining BM25 + vector results:",
        [
            "weighted (normalize scores and combine with weights)",
            "rrf (Reciprocal Rank Fusion — robust to score distribution differences)",
        ],
        default="weighted (normalize scores and combine with weights)",
    )
    merge_strategy = merge_strategy.split(" (")[0]

    bm25_weight = prompt_float("BM25 (keyword) weight", default=search_defaults["bm25_weight"])
    vector_weight = prompt_float("Vector (semantic) weight", default=search_defaults["vector_weight"])
    top_k = prompt_int("Results after re-ranking (top_k)", default=20, min_val=5, max_val=100)
    rerank_top_n = prompt_int("Candidates before re-ranking", default=search_defaults["rerank_top_n"], min_val=10, max_val=200)

    # RRF constant (only relevant for rrf strategy)
    rrf_k = search_defaults["rrf_k"]
    if merge_strategy == "rrf":
        rrf_k = prompt_int(
            "RRF constant (higher = less impact of rank position)",
            default=search_defaults["rrf_k"],
            min_val=1,
            max_val=200,
        )

    # MMR diversity
    print("\n--- Diversity Settings ---\n")
    print("MMR (Maximal Marginal Relevance) reduces redundant results.")
    print("Useful for large document collections with overlapping content.")

    enable_mmr = prompt_yes_no("Enable MMR diversity?", default=False)
    mmr_lambda = None
    if enable_mmr:
        mmr_lambda = prompt_float(
            "MMR lambda (1.0=pure relevance, 0.0=max diversity)",
            default=0.7,
            min_val=0.0,
            max_val=1.0,
        )

    # Adaptive reranking
    configure_adaptive = prompt_yes_no("Configure adaptive reranking? (advanced)", default=False)

    adaptive_rerank = search_defaults["adaptive_rerank"]
    adaptive_rerank_threshold = search_defaults["adaptive_rerank_threshold"]
    adaptive_rerank_depth = search_defaults["adaptive_rerank_depth"]

    if configure_adaptive:
        adaptive_rerank = prompt_yes_no(
            "Enable adaptive reranking? (reduces depth on high-confidence matches)",
            default=adaptive_rerank,
        )
        if adaptive_rerank:
            adaptive_rerank_threshold = prompt_float(
                "Combined score threshold to trigger",
                default=adaptive_rerank_threshold,
                min_val=0.0,
                max_val=1.0,
            )
            adaptive_rerank_depth = prompt_int(
                "Reduced rerank depth when triggered",
                default=adaptive_rerank_depth,
                min_val=5,
                max_val=100,
            )

    # Min score threshold
    min_score = None
    configure_min_score = prompt_yes_no("Set a minimum relevance score threshold? (filters low-quality results)", default=False)
    if configure_min_score:
        min_score = prompt_float(
            "Minimum rerank score (-2.0=strict, -5.0=lenient)",
            default=-2.0,
            min_val=-10.0,
            max_val=0.0,
        )

    # =========================================================================
    # Advanced search settings (Stage 8 + 10)
    # =========================================================================
    print("\n--- Advanced Search (optional) ---\n")

    configure_advanced = prompt_yes_no("Configure advanced search features? (HyDE, multi-query, caching)", default=False)

    # Stage 8 defaults
    bm25_k1 = search_defaults.get("bm25_k1", 1.5)
    bm25_b = search_defaults.get("bm25_b", 0.75)
    score_normalization = search_defaults.get("score_normalization", "minmax")
    hyde = False
    multi_query = False
    multi_query_count = 3
    # Stage 10 defaults
    cache_ttl = search_defaults.get("cache_ttl", 300)
    cache_max_size = search_defaults.get("cache_max_size", 128)

    if configure_advanced:
        # BM25 tuning
        bm25_k1 = prompt_float(
            "BM25 k1 (term frequency saturation, higher = more TF impact)",
            default=1.5, min_val=0.1, max_val=5.0,
        )
        bm25_b = prompt_float(
            "BM25 b (document length normalization, 0=none, 1=full)",
            default=0.75, min_val=0.0, max_val=1.0,
        )

        # Score normalization
        score_normalization = prompt_choice(
            "Score normalization for weighted merge:",
            ["minmax (default, 0-1 range)", "zscore (standard deviations)", "raw (no normalization)"],
            default="minmax (default, 0-1 range)",
        )
        score_normalization = score_normalization.split(" (")[0]

        # HyDE
        print("\nHyDE generates a hypothetical answer via LLM to improve vector search.")
        print("Requires LM Studio running for LLM generation.")
        hyde = prompt_yes_no("Enable HyDE (Hypothetical Document Embeddings)?", default=False)

        # Multi-query
        print("\nMulti-query generates query variants via LLM and merges results.")
        print("Requires LM Studio running for LLM generation.")
        multi_query = prompt_yes_no("Enable multi-query / RAG-Fusion?", default=False)
        if multi_query:
            multi_query_count = prompt_int(
                "Number of query variants to generate",
                default=3, min_val=2, max_val=7,
            )

        # Search caching
        print("\nSearch result caching speeds up repeated queries.")
        cache_ttl = prompt_int(
            "Cache TTL in seconds (0 = disabled)",
            default=300, min_val=0, max_val=3600,
        )
        if cache_ttl > 0:
            cache_max_size = prompt_int(
                "Max cached queries per client",
                default=128, min_val=16, max_val=1024,
            )

    # =========================================================================
    # Tokenization settings
    # =========================================================================
    print("\n--- Tokenization (BM25) ---\n")

    stemmer = prompt_choice(
        "Select stemmer for BM25 search:",
        ["porter (recommended)", "snowball", "none"],
        default="porter (recommended)",
    )

    if "porter" in stemmer:
        stemmer_value = "porter"
    elif "snowball" in stemmer:
        stemmer_value = "snowball"
    else:
        stemmer_value = None

    # =========================================================================
    # Code indexing settings
    # =========================================================================
    print("\n--- Code Indexing ---\n")

    code_enabled = prompt_yes_no("Will this client index source code?", default=False)

    code_extensions = []
    code_paths = ["documents"]
    code_priority = False
    code_priority_boost = 0.5
    cross_reference_code = False
    cross_reference_top_k = 3

    if code_enabled:
        print("\nSelect programming languages to index:")
        print("  1. JavaScript/TypeScript (js, jsx, ts, tsx, mjs, cjs)")
        print("  2. Python (py)")
        print("  3. PHP (php, module, install, theme)")
        print("  4. Web (html, css, scss, json, yaml)")
        print("  5. BDD/Gherkin (feature)")
        print("  6. All of the above")

        lang_choice = prompt_input("Enter numbers separated by commas", default="6")

        if "6" in lang_choice or "all" in lang_choice.lower():
            code_extensions = [
                "js", "jsx", "ts", "tsx", "mjs", "cjs",
                "py",
                "php", "module", "install", "theme",
                "html", "css", "scss", "json", "yaml", "yml", "mdx", "feature",
            ]
        else:
            if "1" in lang_choice:
                code_extensions.extend(["js", "jsx", "ts", "tsx", "mjs", "cjs"])
            if "2" in lang_choice:
                code_extensions.append("py")
            if "3" in lang_choice:
                code_extensions.extend(["php", "module", "install", "theme"])
            if "4" in lang_choice:
                code_extensions.extend(["html", "css", "scss", "json", "yaml", "yml"])
            if "5" in lang_choice:
                code_extensions.append("feature")

        # Code priority settings
        print("\n--- Code Priority (Code as Source of Truth) ---\n")
        print("When documentation may be outdated, code can be prioritized in search results.")

        code_priority = prompt_yes_no("Prioritize code over documentation?", default=True)

        if code_priority:
            code_priority_boost = prompt_float(
                "Code priority boost (score added to code chunks)",
                default=0.5,
                min_val=0.1,
                max_val=2.0,
            )

            cross_reference_code = prompt_yes_no(
                "Cross-reference docs with related code automatically?",
                default=True,
            )

            if cross_reference_code:
                cross_reference_top_k = prompt_int(
                    "Related code chunks to fetch per document",
                    default=3,
                    min_val=1,
                    max_val=10,
                )

    # =========================================================================
    # Source sync settings
    # =========================================================================
    print("\n--- Source Sync ---\n")
    print("Source sync copies files from an external directory into the client's")
    print("documents folder before reindexing. Useful when the client indexes")
    print("files from a project directory (e.g., source code, docs).")

    source_sync_enabled = prompt_yes_no("Enable source sync?", default=False)

    source_sync = None
    if source_sync_enabled:
        source_path = prompt_input(
            "Source directory (relative to client dir, e.g., '../..' for project root)",
            default="../..",
        )

        print("\nFile patterns to include (comma-separated globs):")
        print("  Examples: *.py, *.md, *.yaml")
        include_input = prompt_input(
            "Include patterns",
            default="*.py, *.md, *.txt, *.yaml, *.yml, *.json",
        )
        include_patterns = [p.strip() for p in include_input.split(",") if p.strip()]

        print("\nDirectories/patterns to exclude (comma-separated):")
        print("  Examples: venv/, .git/, node_modules/")
        exclude_input = prompt_input(
            "Exclude patterns",
            default="venv/, .git/, __pycache__/, *.pyc, *.log, .env",
        )
        exclude_patterns = [p.strip() for p in exclude_input.split(",") if p.strip()]

        source_sync = {
            "enabled": True,
            "source": source_path,
            "include": include_patterns,
            "exclude": exclude_patterns,
        }

    # =========================================================================
    # Chunking strategy
    # =========================================================================
    print("\n--- Chunking Strategy ---\n")
    print("Choose how documents are split into chunks for search.")
    print("  title: Split by section headings (default, best for structured docs)")
    print("  semantic: Split by embedding similarity drops (best for long-form prose)")

    chunking_strategy = prompt_choice(
        "Chunking strategy:",
        [
            "title (default, split by headings)",
            "semantic (embedding-based splitting)",
            "recursive (character splitting with fallback separators)",
        ],
        default="title (default, split by headings)",
    )
    if "recursive" in chunking_strategy:
        chunking_strategy_value = "recursive"
    elif "semantic" in chunking_strategy:
        chunking_strategy_value = "semantic"
    else:
        chunking_strategy_value = "title"

    semantic_threshold = 0.5
    if chunking_strategy_value == "semantic":
        semantic_threshold = prompt_float(
            "Semantic similarity threshold (lower = more splits)",
            default=0.5,
            min_val=0.1,
            max_val=0.9,
        )

    # =========================================================================
    # Knowledge graph
    # =========================================================================
    print("\n--- Knowledge Graph ---\n")
    print("A knowledge graph extracts entities (people, organizations, concepts)")
    print("and their relationships from your documents using the LLM.")
    print("This enables graph-augmented search modes for multi-hop reasoning.")
    print("Requires LM Studio running for entity extraction.")

    graph_enabled = prompt_yes_no("Enable knowledge graph?", default=False)

    graph_search_mode = "naive"
    graph_entity_types = ["person", "organization", "concept", "location"]
    if graph_enabled:
        graph_search_mode = prompt_choice(
            "Default graph search mode:",
            [
                "naive (standard search only, no graph)",
                "local (entity-focused — find entities, retrieve their chunks)",
                "global (relationship-focused — traverse entity connections)",
                "hybrid (local + global combined)",
                "mix (graph + standard hybrid merged via RRF, recommended)",
            ],
            default="mix (graph + standard hybrid merged via RRF, recommended)",
        )
        graph_search_mode = graph_search_mode.split(" (")[0]

        print("\nEntity types to extract (comma-separated):")
        print("  Default: person, organization, concept, location")
        entity_input = prompt_input(
            "Entity types",
            default="person, organization, concept, location",
        )
        graph_entity_types = [t.strip() for t in entity_input.split(",") if t.strip()]

    # =========================================================================
    # Reference resolution settings
    # =========================================================================
    print("\n--- Reference Resolution ---\n")

    ref_enabled = prompt_yes_no("Enable reference resolution (auto-fetch appendices/sections)?", default=True)

    if ref_enabled:
        ref_domain = prompt_choice(
            "Select document domain for reference patterns:",
            [
                "compliance (Appendix, Schedule, Section)",
                "legal (Exhibit, Clause, Article)",
                "technical (RFC, Issue, Section)",
                "academic (Theorem, Equation, Figure)",
            ],
            default="compliance (Appendix, Schedule, Section)",
        )

        if "legal" in ref_domain:
            ref_domain_value = "legal"
        elif "technical" in ref_domain:
            ref_domain_value = "technical"
        elif "academic" in ref_domain:
            ref_domain_value = "academic"
        else:
            ref_domain_value = "compliance"
    else:
        ref_domain_value = "compliance"

    # =========================================================================
    # Audit settings
    # =========================================================================
    print("\n--- Audit Settings ---\n")

    audit_enabled = prompt_yes_no("Enable audit logging?", default=True)

    audit_level = "none"
    audit_retention_days = None

    if audit_enabled:
        audit_level = prompt_choice(
            "Audit log level:",
            ["full", "minimal", "none"],
            default="full",
        )

        set_retention = prompt_yes_no("Set audit retention period? (auto-cleanup old audits)", default=False)
        if set_retention:
            audit_retention_days = prompt_int(
                "Retention days (audit data older than this is deleted)",
                default=90,
                min_val=1,
                max_val=3650,
            )

    # =========================================================================
    # System prompt
    # =========================================================================
    print("\n--- System Prompt ---\n")

    create_prompt = prompt_yes_no("Create a custom system prompt?", default=True)

    system_prompt = None
    if create_prompt:
        print("\nEnter your system prompt (press Enter twice to finish):")
        print("(Or press Enter twice now to use a default prompt)\n")

        lines = []
        empty_count = 0
        while empty_count < 1:
            line = input()
            if line == "":
                empty_count += 1
            else:
                empty_count = 0
                lines.append(line)

        if lines:
            system_prompt = "\n".join(lines)
        else:
            # Default prompt
            system_prompt = f"""You are a helpful assistant for {name}.

Answer questions based on the provided context from the document collection.
If the context doesn't contain relevant information, say so clearly.
Always cite the source documents when providing information."""

    # =========================================================================
    # Build config
    # =========================================================================
    config = {
        "name": name,
        "machine_name": machine_name,
    }

    if description:
        config["description"] = description

    config["paths"] = {
        "documents": "documents",
        "index": "index",
        "prompts": "prompts",
    }

    embedding_config = {
        "model": embedding_model_name,
        "dimension": embedding_dimension,
        "query_prefix": "",
        "passage_prefix": "",
    }
    if contextualize:
        embedding_config["contextualize"] = True
    if configure_hnsw:
        embedding_config["hnsw_ef_construction"] = hnsw_ef_construction
        embedding_config["hnsw_max_neighbors"] = hnsw_max_neighbors
        embedding_config["hnsw_ef_search"] = hnsw_ef_search
    config["embedding"] = embedding_config

    # Documents config (only if customized)
    if exclude_extensions:
        config["documents"] = {
            "exclude_extensions": exclude_extensions,
        }

    # Search config
    search_config = {
        "bm25_weight": bm25_weight,
        "vector_weight": vector_weight,
        "top_k": top_k,
        "rerank_top_n": rerank_top_n,
        "min_score": min_score,
        "merge_strategy": merge_strategy,
        "tokenization": {
            "stemmer": stemmer_value,
            "synonyms": {},
        },
    }

    # Only include non-default values to keep config clean
    if merge_strategy == "rrf":
        search_config["rrf_k"] = rrf_k

    if mmr_lambda is not None:
        search_config["mmr_lambda"] = mmr_lambda

    # Adaptive rerank (only if non-default)
    if not adaptive_rerank:
        search_config["adaptive_rerank"] = False
    elif configure_adaptive:
        search_config["adaptive_rerank"] = adaptive_rerank
        search_config["adaptive_rerank_threshold"] = adaptive_rerank_threshold
        search_config["adaptive_rerank_depth"] = adaptive_rerank_depth

    # Code priority (only if code indexing is enabled)
    if code_enabled and code_priority:
        search_config["code_priority"] = code_priority
        search_config["code_priority_boost"] = code_priority_boost
        search_config["cross_reference_code"] = cross_reference_code
        search_config["cross_reference_top_k"] = cross_reference_top_k

    # Advanced search (Stage 8 + 10) — only include non-defaults
    if configure_advanced:
        if bm25_k1 != 1.5:
            search_config["bm25_k1"] = bm25_k1
        if bm25_b != 0.75:
            search_config["bm25_b"] = bm25_b
        if score_normalization != "minmax":
            search_config["score_normalization"] = score_normalization
        if hyde:
            search_config["hyde"] = True
        if multi_query:
            search_config["multi_query"] = True
            search_config["multi_query_count"] = multi_query_count
        if cache_ttl != 300:
            search_config["cache_ttl"] = cache_ttl
        if cache_max_size != 128:
            search_config["cache_max_size"] = cache_max_size

    config["search"] = search_config

    # Code indexing
    if code_enabled:
        config["code"] = {
            "enabled": True,
            "paths": code_paths,
            "extensions": code_extensions,
            "exclude_patterns": [
                "**/node_modules/**",
                "**/.next/**",
                "**/dist/**",
                "**/build/**",
                "**/__pycache__/**",
                "**/.git/**",
                "**/vendor/**",
            ],
        }

    # Source sync
    if source_sync:
        config["source_sync"] = source_sync

    # Chunking
    if chunking_strategy_value != "title":
        chunking_config = {"strategy": chunking_strategy_value}
        if chunking_strategy_value == "semantic":
            chunking_config["semantic_threshold"] = semantic_threshold
        config["chunking"] = chunking_config

    # Knowledge graph
    if graph_enabled:
        config["graph"] = {
            "enabled": True,
            "entity_types": graph_entity_types,
            "search_mode": graph_search_mode,
        }

    # References
    config["references"] = {
        "enabled": ref_enabled,
        "mode": "extend",
        "domain": ref_domain_value,
    }

    # LLM
    llm_config = {
        "provider": llm_provider,
        "max_context_chars": max_context_chars,
        "max_chunks": max_chunks,
        "max_tokens": max_tokens,
    }
    if citations_enabled:
        llm_config["citations"] = True
    if context_filter_enabled:
        llm_config["context_filter"] = True
        llm_config["context_filter_threshold"] = context_filter_threshold
    config["llm"] = llm_config

    # Audit
    audit_config = {
        "enabled": audit_enabled,
        "log_level": audit_level,
        "include_chunks": True,
        "include_scores": True,
        "include_timing": True,
    }
    if audit_retention_days is not None:
        audit_config["retention_days"] = audit_retention_days
    config["audit"] = audit_config

    # =========================================================================
    # Validate config
    # =========================================================================
    warnings = validate_config(config)
    if warnings:
        print("\n  Config validation warnings:")
        for w in warnings:
            print(f"    - {w}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    print(f"\nClient: {name} ({machine_name})")
    print(f"Location: {client_path}")
    print(f"Embedding: {embedding_model_name} ({embedding_dimension}d)")
    print(f"Stemmer: {stemmer_value or 'none'}")
    print(f"Reference domain: {ref_domain_value}")
    print(f"LLM: {llm_provider} (max {max_context_chars} chars, {max_chunks} chunks, {max_tokens} tokens)")
    print(f"Search: strategy={merge_strategy}, BM25={bm25_weight}, Vector={vector_weight}, top_k={top_k}")
    if citations_enabled:
        print("Citations: enabled (inline [1] [2] references)")
    if context_filter_enabled:
        print(f"Context filter: enabled (threshold={context_filter_threshold})")
    if mmr_lambda is not None:
        print(f"MMR diversity: enabled (lambda={mmr_lambda})")
    if min_score is not None:
        print(f"Min score threshold: {min_score}")
    if code_enabled:
        print(f"Code indexing: enabled ({len(code_extensions)} extensions)")
        if code_priority:
            print(f"Code priority: enabled (boost={code_priority_boost}, cross-ref={'yes' if cross_reference_code else 'no'})")
    else:
        print("Code indexing: disabled")
    if source_sync:
        print(f"Source sync: enabled (from {source_sync['source']})")
    if chunking_strategy_value != "title":
        extra = f", threshold={semantic_threshold}" if chunking_strategy_value == "semantic" else ""
        print(f"Chunking: {chunking_strategy_value}{extra}")
    else:
        print("Chunking: title (default)")
    if graph_enabled:
        print(f"Knowledge graph: enabled (mode={graph_search_mode}, types={', '.join(graph_entity_types)})")
    else:
        print("Knowledge graph: disabled")
    if contextualize:
        print("Contextual embeddings: enabled (requires reindex)")
    if configure_hnsw:
        print(f"HNSW: ef_construction={hnsw_ef_construction}, M={hnsw_max_neighbors}, ef_search={hnsw_ef_search}")
    if configure_advanced:
        if hyde:
            print("HyDE: enabled")
        if multi_query:
            print(f"Multi-query: enabled ({multi_query_count} variants)")
        print(f"Search cache: {'disabled' if cache_ttl == 0 else f'TTL={cache_ttl}s, max={cache_max_size}'}")
    if exclude_extensions:
        print(f"Excluded doc types: {', '.join(exclude_extensions)}")
    print(f"Audit: {'enabled' if audit_enabled else 'disabled'} ({audit_level})")
    if audit_retention_days:
        print(f"Audit retention: {audit_retention_days} days")
    print(f"System prompt: {'custom' if system_prompt else 'none'}")

    print()
    if not prompt_yes_no("Create this client?", default=True):
        print("Aborted.")
        return

    # =========================================================================
    # Create files
    # =========================================================================
    print("\nCreating client...")

    client_path.mkdir(parents=True, exist_ok=True)
    (client_path / "documents").mkdir(exist_ok=True)
    (client_path / "index").mkdir(exist_ok=True)
    (client_path / "prompts").mkdir(exist_ok=True)

    # Write config
    config_path = client_path / "client.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Created: {config_path}")

    # Write system prompt
    if system_prompt:
        prompt_path = client_path / "prompts" / "system.txt"
        prompt_path.write_text(system_prompt)
        print(f"  Created: {prompt_path}")

    print(f"\nClient '{machine_name}' created successfully!")
    print("\nNext steps:")
    print(f"  1. Add documents to: {client_path / 'documents'}")
    if code_enabled:
        print(f"     (Code files will also be indexed from this directory)")
        print(f"  2. Run ingestion: python scripts/ingest.py --client {machine_name} --rebuild")
        print("     (Use --rebuild to index both documents and code)")
    elif source_sync:
        print(f"  2. Sync & index: python scripts/watch_and_reindex.py --client {machine_name}")
    else:
        print(f"  2. Run ingestion: python scripts/ingest.py --client {machine_name}")
    if graph_enabled:
        print(f"  3. Build knowledge graph: python scripts/build_graph.py --client {machine_name}")
        print(f"  4. Use via MCP: python scripts/mcp_server.py")
    else:
        print(f"  3. Use via MCP: python scripts/mcp_server.py")


def main():
    try:
        create_client()
    except KeyboardInterrupt:
        print("\n\nAborted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
