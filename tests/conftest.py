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
def testbot(tmp_path, monkeypatch):
    """`pip install examples/robots/testbot` without pip: its package and a dist-info on `sys.path`.

    The dist-info carries the entry points of the package's own `pyproject.toml`, so the twin
    finds the robot and its backend the way it finds any installed plugin.
    """
    project = tomllib.loads((TESTBOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dist = tmp_path / f"{project['name'].replace('-', '_')}-{project['version']}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {project['name']}\nVersion: {project['version']}\n", encoding="utf-8")
    groups = [f"[{group}]\n" + "".join(f"{name} = {value}\n" for name, value in eps.items())
              for group, eps in project["entry-points"].items()]
    (dist / "entry_points.txt").write_text("\n".join(groups), encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(TESTBOT))
    reload_robots()
    yield "testbot"
    monkeypatch.undo()
    reload_robots()
