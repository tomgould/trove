"""
End to end: ingest a folder of Markdown files and search it.

Uses only the core dependencies, so it also runs on a core install with no extras.
"""

import importlib.util
from pathlib import Path

SMOKE_SCRIPT = Path(__file__).parent / "smoke" / "core_smoke.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("core_smoke", SMOKE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ingest_and_search_markdown_folder(tmp_path):
    smoke = _load_smoke_module()
    results = smoke.run(tmp_path)
    assert results[0]["metadata"]["source_file"] == "bread.md"
