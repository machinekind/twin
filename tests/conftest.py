"""Shared test helpers: no config file from the shell, and the test robot as if it were installed."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from lerobot_mp.twin.robots import reload_robots

#: The test robot's package (`examples/robots/testbot`): a robot folder and a backend plugin.
TESTBOT = Path(__file__).resolve().parents[1] / "examples" / "robots" / "testbot"


@pytest.fixture(autouse=True)
def _no_config_from_env(monkeypatch):
    """Testy licza na wartosciach domyslnych - plik z LEROBOT_MP_CONFIG w powloce by je zmienil."""
    # setenv zapamietuje stan sprzed testu, wiec to, co ustawi `lerobot-twin --config`
    # w os.environ, zniknie po tescie (samo delenv nieistniejacej zmiennej niczego nie zapamietuje).
    monkeypatch.setenv("LEROBOT_MP_CONFIG", "")
    monkeypatch.delenv("LEROBOT_MP_CONFIG")


@pytest.fixture
def install_dist(tmp_path, monkeypatch):
    """Put a distribution on `sys.path` as `pip install` would leave it: its dist-info with these
    entry points ({group: {name: value}}), and optional top-level modules ({module: source})."""
    def install(name: str, entry_points: dict[str, dict[str, str]], modules: dict[str, str] | None = None,
                version: str = "0.1.0") -> Path:
        site = tmp_path / f"site-{name}"
        dist = site / f"{name.replace('-', '_')}-{version}.dist-info"
        dist.mkdir(parents=True)
        (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                                       encoding="utf-8")
        groups = [f"[{group}]\n" + "".join(f"{ep} = {value}\n" for ep, value in eps.items())
                  for group, eps in entry_points.items()]
        (dist / "entry_points.txt").write_text("\n".join(groups), encoding="utf-8")
        for module, source in (modules or {}).items():
            (site / f"{module}.py").write_text(source, encoding="utf-8")
        monkeypatch.syspath_prepend(str(site))
        reload_robots()
        return dist

    yield install
    monkeypatch.undo()
    reload_robots()


@pytest.fixture
def testbot(install_dist, monkeypatch):
    """`pip install examples/robots/testbot` without pip: its package and a dist-info on `sys.path`.

    The dist-info carries the entry points of the package's own `pyproject.toml`, so the twin
    finds the robot and its backend the way it finds any installed plugin.
    """
    project = tomllib.loads((TESTBOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    monkeypatch.syspath_prepend(str(TESTBOT))
    install_dist(project["name"], project["entry-points"], version=project["version"])
    return "testbot"
