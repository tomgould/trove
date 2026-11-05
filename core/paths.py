"""
Where Trove finds its folders.

Trove runs either from a source checkout (``python scripts/search.py``) or from
an installed wheel (``trove-search``). In a checkout the clients folder is the
repository's own ``clients/``. An installed copy has no repository, so it uses
``clients/`` in the current working directory instead of writing into
site-packages.

The agent templates ship inside the package (``core/data/agents``), so they are
found the same way in both cases.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLED_AGENTS_DIR = PACKAGE_ROOT / "data" / "agents"


def source_checkout_root() -> Path | None:
    """Return the repository root when running from a source checkout, else None."""
    root = PACKAGE_ROOT.parent
    if (root / "pyproject.toml").is_file() and (root / "clients").is_dir():
        return root
    return None


def default_clients_dir() -> Path:
    """Folder that holds one sub-folder per client."""
    root = source_checkout_root()
    if root is not None:
        return root / "clients"
    return Path.cwd() / "clients"


def default_agents_dir() -> Path:
    """Folder that holds the agent YAML templates."""
    return BUNDLED_AGENTS_DIR
