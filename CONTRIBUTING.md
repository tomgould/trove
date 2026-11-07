# Contributing to Trove

Thanks for your interest in contributing to Trove.

## Development Setup

Trove supports Python 3.11, 3.12 and 3.13.

```bash
# Clone the repo
git clone https://github.com/tomgould/trove.git
cd trove

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# The CPU build of PyTorch first, then the full stack at the tested versions
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -c constraints/py312.txt   # match your Python version
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest

# Run one area
pytest tests/test_search.py

# Run a specific test class
pytest tests/test_search.py::TestHybridSearcher -v
```

Tests are split by area (`test_client.py`, `test_ingest.py`, `test_search.py`, `test_ranking.py`, `test_graph.py` and so on). Shared fixtures live in `tests/conftest.py`. A bug fix comes with a regression test in the file for its area.

## Changing Dependencies

`requirements.txt` and `pyproject.toml` must list the same ranges; `tests/test_packaging.py` checks this. After changing a range, rebuild the constraints file for each supported Python from a clean install:

```bash
uv venv --python 3.12 /tmp/trove-lock
uv pip install --python /tmp/trove-lock/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/trove-lock/bin/python -r requirements.txt
uv pip freeze --python /tmp/trove-lock/bin/python | sed -E 's/\+cpu$//' > constraints/py312.txt
```

Keep the three comment lines at the top of the file. Strip the `+cpu` labels, as above, so the file also works with the default package index.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. The pre-commit hooks run ruff on each commit. CI runs the error checks below and fails on them; it also runs `ruff format --check` but only reports the result, because the code is not fully formatted.

```bash
ruff check --select E9,F63,F7,F82 core/ mcp_server/ scripts/ tests/   # the CI check
ruff format core/ mcp_server/ scripts/
```

Key conventions:
- Type hints on all public functions
- Docstrings on classes and public methods
- Keep functions focused and under 50 lines where practical
- Use `logging` module, not `print()`
- Business logic in `core/`, MCP interface in `mcp_server/`

## Architecture Overview

```
core/           # Main library: search, ingestion, LLM, graph, plugins, sources
mcp_server/     # MCP server exposing core functionality to Claude Code
scripts/        # CLI tools for ingestion, search, evaluation, etc.
tests/          # pytest test suite
core/data/agents/  # YAML agent templates, shipped in the package
plugins/        # User plugin directory (drop-in .py files)
clients/        # Per-client config, documents, and indexes
```

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run `pytest`: all tests must pass
4. Run the CI ruff check above: no errors
5. Update the docs the change affects (README.md, CLAUDE.md, docs/) in the present tense, and add a CHANGELOG entry
6. Submit a PR with a clear description of what and why

## What to Work On

[docs/REVIVAL-PLAN.md](docs/REVIVAL-PLAN.md) lists the near-term planned work, [ROADMAP.md](ROADMAP.md) the product stages, and the README's "Known limits" section the current limits of the code.

## License

By contributing, you agree that your contributions will be licensed under the project's [Business Source License 1.1](LICENSE).
