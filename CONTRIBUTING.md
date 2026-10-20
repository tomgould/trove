# Contributing to Trove

Thanks for your interest in contributing to Trove.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/tomgould/trove.git
cd trove

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies + dev tools
pip install -r requirements.txt
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

**Requires Python 3.10+**

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test class
pytest tests/test_core.py::TestHybridSearcher -v
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Pre-commit hooks run automatically, or you can run manually:

```bash
ruff check core/ mcp_server/ scripts/
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
core/           # Main library — search, ingestion, LLM, graph, plugins, sources
mcp_server/     # MCP server exposing core functionality to Claude Code
scripts/        # CLI tools for ingestion, search, evaluation, etc.
tests/          # pytest test suite
agents/         # YAML agent templates
plugins/        # User plugin directory (drop-in .py files)
clients/        # Per-client config, documents, and indexes
```

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

## Pull Request Process

1. Create a feature branch from `master`
2. Make your changes with tests
3. Run `pytest tests/` — all tests must pass
4. Run `ruff check` — no lint errors
5. Submit a PR with a clear description of what and why

## What to Work On

Check the [ROADMAP.md](ROADMAP.md) for planned features. Issues tagged `good first issue` are a great starting point.

## License

By contributing, you agree that your contributions will be licensed under the project's [Business Source License 1.1](LICENSE).
