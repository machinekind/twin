"""Shared test helpers: no config file from the shell, and the example packages as if they were installed."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from lerobot_mp.twin.robots import reload_robots

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
#: The test robot's package: a robot folder and a backend plugin.
TESTBOT = EXAMPLES / "robots" / "testbot"
#: The test strategy's package: a strategy with its settings dataclass.
SWING = EXAMPLES / "strategies" / "swing"


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


def install_example(install_dist, monkeypatch, folder: Path) -> None:
    """`pip install <folder>` without pip: its package and a dist-info on `sys.path`.

    The dist-info carries the entry points of the package's own `pyproject.toml`, so the twin
    finds what it registers the way it finds any installed plugin.
    """
    project = tomllib.loads((folder / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    monkeypatch.syspath_prepend(str(folder))
    install_dist(project["name"], project["entry-points"], version=project["version"])


@pytest.fixture
def testbot(install_dist, monkeypatch):
    """The test robot (`examples/robots/testbot`), installed: its robot folder and backend."""
    install_example(install_dist, monkeypatch, TESTBOT)
    return "testbot"


@pytest.fixture
def swing(install_dist, monkeypatch):
    """The test strategy (`examples/strategies/swing`), installed."""
    install_example(install_dist, monkeypatch, SWING)
    return "swing"
