# Writing Trove Plugins

Plugins extend Trove with custom embedding models, chunking strategies, file parsers, rerankers, LLM providers, and merge strategies.

## Quick Start

Create a Python file in this directory:

```python
# plugins/my_embedder.py
from core.plugins.interfaces import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    name = "my_embedder"
    dimension = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Your embedding logic here
        ...

def register(registry):
    registry.register_embedder(MyEmbedder)
```

Then reference it in your `client.yaml`:

```yaml
embedding:
  model: my_embedder
```

## Plugin Structure

A plugin is either:

- **A single `.py` file** (e.g., `plugins/ollama.py`)
- **A directory with `__init__.py`** (e.g., `plugins/ollama/__init__.py`)

Every plugin must expose a `register(registry)` function. Trove calls this on startup, passing the global `PluginRegistry`.

## Extension Points

### Embedder (`BaseEmbedder`)

Converts text into vectors for similarity search.

```python
from core.plugins.interfaces import BaseEmbedder

class OllamaEmbedder(BaseEmbedder):
    name = "ollama"
    dimension = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Optional: override if queries need different handling."""
        ...

    def close(self) -> None:
        """Optional: clean up resources."""
        ...
```

### Chunker (`BaseChunker`)

Splits documents into chunks for indexing.

```python
from core.plugins.interfaces import BaseChunker

class ParagraphChunker(BaseChunker):
    name = "paragraph"

    def chunk(self, elements: list, max_chars: int = 1500,
              overlap: int = 100, **kwargs) -> list[str]:
        """Split elements into text chunks."""
        ...
```

### Parser (`BaseParser`)

Parses document files into elements.

```python
from core.plugins.interfaces import BaseParser
from pathlib import Path

class MarkdownParser(BaseParser):
    name = "markdown"
    supported_extensions = [".md", ".mdx"]

    def parse(self, path: Path) -> list:
        """Parse file into elements with text content and metadata."""
        ...

    def can_parse(self, path: Path) -> bool:
        """Optional: override for custom file type detection."""
        ...
```

### Reranker (`BaseReranker`)

Scores query-document pairs for relevance.

```python
from core.plugins.interfaces import BaseReranker

class CohereReranker(BaseReranker):
    name = "cohere"

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, document) pairs. Returns one score per pair."""
        ...

    def close(self) -> None:
        """Optional: clean up resources."""
        ...
```

### LLM Provider (`BaseLLMProvider`)

Generates text from language models.

```python
from core.plugins.interfaces import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):
    name = "ollama"

    def generate(self, messages: list[dict], system_prompt: str = None,
                 max_tokens: int = 2000) -> str:
        """Generate a response. Messages have 'role' and 'content' keys."""
        ...
```

### Merge Strategy (`BaseMergeStrategy`)

Combines BM25 and vector search results.

```python
from core.plugins.interfaces import BaseMergeStrategy

class CustomMerge(BaseMergeStrategy):
    name = "custom"

    def merge(self, bm25_results: list[tuple[dict, float]],
              vector_results: list[tuple[dict, float]],
              config: dict) -> list[dict]:
        """Merge results. Each input is (chunk_dict, score).
        Return chunks with _combined_score field, sorted descending."""
        ...
```

## Configuration

The `TROVE_PLUGINS_DIR` environment variable overrides the default `plugins/` directory:

```bash
export TROVE_PLUGINS_DIR=/path/to/my/plugins
```

## Error Handling

- Plugins that fail to import are logged and skipped
- Plugins without a `register()` function are skipped
- If `register()` raises an exception, the plugin is skipped
- Other plugins continue loading normally
