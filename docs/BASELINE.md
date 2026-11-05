# Baseline

The state of Trove when it was revived, before any Phase 1 change. It is the reference point for later phases: if a change makes the suite slower, noisier or less complete than this, it shows here.

## How it was run

A clean virtual environment per Python version, installed the way CI installed it then:

```bash
uv venv --python <version> .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests/
```

`requirements.txt` held only lower bounds at the time, so each install took the newest versions available.

## Results

| | Python 3.12.15 | Python 3.10.12 |
|---|---|---|
| Tests collected | 724 | 724 |
| Passed | 723 | 723 |
| Skipped | 1 | 1 |
| Failed | 0 | 0 |
| Exit code | 0 | 0 |

All 724 tests were in one file, `tests/test_core.py`. The one skip was `tests/test_core.py:1818`, the integration test that needed an internal client with an index, which a clean checkout does not have. Each run took about 14 minutes.

Each run reported 4 warnings, all `FutureWarning`: three from tree-sitter (`Language(path, name) is deprecated`, raised through tree-sitter-languages) and one from sentence-transformers (`get_sentence_embedding_dimension` has been renamed, raised in `scripts/benchmark_embeddings.py`). Nothing else.

## Resolved versions

| Package | Python 3.12 | Python 3.10 |
|---|---|---|
| chromadb | 1.5.9 | 1.5.9 |
| sentence-transformers | 6.2.0 | 6.2.0 |
| torch (CPU) | 2.15.0 | 2.14.1 |
| torchvision (CPU) | 0.29.2 | 0.29.2 |
| mcp | 1.30.0 | 1.30.0 |
| numpy | 2.5.4 | 2.2.6 |
| unstructured | 0.27.36 | 0.18.32 |
| tree-sitter | 0.21.3 | 0.21.3 |
| tree-sitter-languages | 1.10.2 | 1.10.2 |

On Python 3.10 the resolver fell back to much older releases of unstructured, numpy and several of their dependencies, because the current releases need Python 3.11 or later. tree-sitter 0.21.3 with tree-sitter-languages 1.10.2 works on both, with the FutureWarning above.

## What changed in Phase 1

- Supported Python is 3.11 to 3.13. 3.10 reached end of life in October 2026, and the current unstructured, numpy and networkx releases no longer support it.
- tree-sitter-languages has no wheels for Python 3.13, so on 3.13 the `code` extra installs nothing and the code-indexing tests that need a parser are skipped.
- Dependencies have tested ranges, and `constraints/` holds the exact versions from a working install per Python version.
- The test suite is split into one file per area. The 724 original tests are all still collected, alongside the new regression tests.
