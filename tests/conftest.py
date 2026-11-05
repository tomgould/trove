"""Shared fixtures for the Trove test suite."""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def clients_dir(tmp_path):
    """An empty clients folder in a temporary directory."""
    path = tmp_path / "clients"
    path.mkdir()
    return path


@pytest.fixture
def make_client(clients_dir):
    """Create a client folder with a client.yaml and return its name.

    Call it as ``make_client("name")``, optionally with extra config keys.
    """

    def _make(name: str = "demo", **config) -> str:
        client_dir = clients_dir / name
        client_dir.mkdir()
        data = {"name": name.title(), "machine_name": name, **config}
        (client_dir / "client.yaml").write_text(yaml.safe_dump(data))
        return name

    return _make
