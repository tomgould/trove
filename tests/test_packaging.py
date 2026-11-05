"""
Packaging: dependency files agree, console scripts resolve, bundled data is found.
"""

import importlib
import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
PROJECT = PYPROJECT["project"]


def _requirements_txt() -> list[Requirement]:
    reqs = []
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.split(" #")[0].strip()
        if line and not line.startswith("#"):
            reqs.append(Requirement(line))
    return reqs


def _key(req: Requirement) -> tuple:
    return (req.name.lower(), tuple(sorted(req.extras)), str(req.specifier), str(req.marker or ""))


def _supported_versions() -> list[str]:
    prefix = "Programming Language :: Python :: 3."
    return sorted(
        (c.rsplit(" ", 1)[-1] for c in PROJECT["classifiers"] if c.startswith(prefix) and c[len(prefix):].isdigit()),
        key=lambda v: tuple(int(p) for p in v.split(".")),
    )


class TestDependencies:
    def test_requirements_txt_matches_the_all_extra(self):
        """requirements.txt is the core dependencies plus every extra in "all", plus pytest."""
        extras = PROJECT["optional-dependencies"]
        all_extra = Requirement(extras["all"][0])
        assert all_extra.name == PROJECT["name"]

        expected = [Requirement(r) for r in PROJECT["dependencies"]]
        for name in sorted(all_extra.extras):
            expected += [Requirement(r) for r in extras[name]]
        expected += [Requirement(r) for r in extras["dev"] if r.startswith("pytest")]

        assert sorted(map(_key, _requirements_txt())) == sorted(map(_key, expected))

    def test_every_dependency_has_an_upper_bound(self):
        for req in _requirements_txt():
            assert any(s.operator in ("<", "<=", "==", "~=") for s in req.specifier), f"{req} has no upper bound"

    def test_torch_is_a_core_dependency(self):
        assert "torch" in {Requirement(r).name for r in PROJECT["dependencies"]}

    def test_constraints_file_for_each_supported_python(self):
        for version in _supported_versions():
            path = REPO_ROOT / "constraints" / f"py{version.replace('.', '')}.txt"
            assert path.is_file(), f"missing {path.name}"
            text = path.read_text()
            assert "+cpu" not in text, "local version labels stop the file working with the default index"
            assert re.search(r"(?m)^torch==", text)


class TestPythonVersions:
    def test_requires_python_matches_classifiers(self):
        lowest = _supported_versions()[0]
        assert PROJECT["requires-python"] == f">={lowest}"

    def test_ci_matrix_matches_classifiers(self):
        workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "test.yml").read_text())
        matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]
        assert sorted(matrix) == _supported_versions()

    def test_ruff_target_matches_lowest_python(self):
        lowest = _supported_versions()[0].replace(".", "")
        assert PYPROJECT["tool"]["ruff"]["target-version"] == f"py{lowest}"


class TestConsoleScripts:
    @pytest.mark.parametrize("name", sorted(PROJECT["scripts"]))
    def test_entry_point_resolves(self, name):
        module_name, attr = PROJECT["scripts"][name].split(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attr)), f"{name} -> {module_name}:{attr} is not callable"


class TestBundledData:
    def test_agents_ship_in_the_package(self):
        from core.paths import default_agents_dir

        agents_dir = default_agents_dir()
        assert agents_dir.is_relative_to(REPO_ROOT / "core")
        assert sorted(agents_dir.glob("*.yaml")), "no agent templates in the package"
        globs = PYPROJECT["tool"]["setuptools"]["package-data"]["core"]
        assert "data/agents/*.yaml" in globs

    def test_agent_loader_finds_the_bundled_templates(self):
        from core.agents import AgentLoader
        from core.paths import default_agents_dir

        agents = AgentLoader(default_agents_dir()).list()
        assert len(agents) == len(list(default_agents_dir().glob("*.yaml")))
        assert "Code Reviewer" in {a["name"] for a in agents}

    def test_source_checkout_uses_repo_clients(self):
        from core.paths import default_clients_dir, source_checkout_root

        assert source_checkout_root() == REPO_ROOT
        assert default_clients_dir() == REPO_ROOT / "clients"

    def test_installed_copy_uses_clients_in_working_directory(self, tmp_path, monkeypatch):
        import core.paths as paths

        site = tmp_path / "site-packages"
        (site / "core").mkdir(parents=True)
        monkeypatch.setattr(paths, "PACKAGE_ROOT", site / "core")
        monkeypatch.chdir(tmp_path)
        assert paths.source_checkout_root() is None
        assert paths.default_clients_dir() == tmp_path / "clients"


class TestCreateClientHelp:
    """Regression: trove-create-client --help started the interactive wizard."""

    @pytest.mark.parametrize("flag", ["--help", "-h"])
    def test_help_prints_usage_and_exits(self, flag, capsys, monkeypatch):
        import scripts.create_client as cli

        monkeypatch.setattr(cli, "create_client", lambda: pytest.fail("the wizard started"))
        with pytest.raises(SystemExit) as excinfo:
            cli.main([flag])
        assert excinfo.value.code == 0
        assert "usage: trove-create-client" in capsys.readouterr().out
