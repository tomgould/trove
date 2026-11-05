"""
Check an installed copy of Trove (a wheel in a clean virtual environment).

Run it from a folder outside the source tree, with the venv's Python:

    cd "$(mktemp -d)" && /path/to/venv/bin/python -I /path/to/tests/smoke/installed_smoke.py

It checks that every console script resolves, that the bundled agent templates
are found, and that ``trove-ingest`` and ``trove-search`` index and search a
folder of Markdown files using ``clients/`` in the current folder.
"""

import importlib.util
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
from importlib.metadata import distribution
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = Path(sysconfig.get_path("scripts"))


def check_not_running_from_source() -> None:
    import core

    location = Path(core.__file__).resolve()
    assert "site-packages" in location.parts, f"core imported from {location}, not from the installed wheel"


def check_console_scripts() -> None:
    eps = [ep for ep in distribution("trove-knowledge-engine").entry_points if ep.group == "console_scripts"]
    names = sorted(ep.name for ep in eps)
    assert "trove-server" in names, names
    for ep in eps:
        assert callable(ep.load()), ep.name
        assert (SCRIPTS_DIR / ep.name).exists(), f"{ep.name} is not on the venv's PATH"
    print(f"console scripts resolve: {', '.join(names)}")


def check_agents() -> None:
    from core.agents import AgentLoader
    from core.paths import default_agents_dir

    agents = AgentLoader(default_agents_dir()).list()
    assert agents, f"no agent templates found in {default_agents_dir()}"
    print(f"agent templates found: {len(agents)}")


def check_cli_ingest_and_search() -> None:
    spec = importlib.util.spec_from_file_location("core_smoke", HERE / "core_smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)

    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "clients" / "smoke" / "documents"
        docs.mkdir(parents=True)
        (docs.parent / "client.yaml").write_text("name: Smoke test\nmachine_name: smoke\n")
        for name, text in smoke.DOCUMENTS.items():
            (docs / name).write_text(text)

        env = {**os.environ, "PYTHONPATH": ""}
        run = lambda *cmd: subprocess.run(  # noqa: E731
            [str(SCRIPTS_DIR / cmd[0]), *cmd[1:]], cwd=tmp, env=env, capture_output=True, text=True
        )

        missing = run("trove-search", "--client", "nope", "anything")
        assert missing.returncode == 1 and "not found" in missing.stderr, missing.stderr
        assert not (Path(tmp) / "clients" / "nope").exists()

        ingest = run("trove-ingest", "--client", "smoke", "--rebuild", "--no-code")
        assert ingest.returncode == 0, ingest.stderr[-2000:]

        search = run("trove-search", "--client", "smoke", "--json", "how do I keep a sourdough starter alive")
        assert search.returncode == 0, search.stderr[-2000:]
        payload = json.loads(search.stdout)
        print(f"trove-search returned: {json.dumps(payload)[:200]}")
        assert "bread.md" in search.stdout


def main() -> int:
    check_not_running_from_source()
    check_console_scripts()
    check_agents()
    check_cli_ingest_and_search()
    print("installed smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
