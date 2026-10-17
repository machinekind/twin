"""Shared test helpers: no config file from the shell, and the example packages as if they were installed.

The twin holds no robot of its own, so the test robot (`examples/robots/testbot`) is installed for
the whole run, before collection: test modules build their robot at import (`get_spec(BOT)`).
"""

from __future__ import annotations

import atexit
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

import pytest

from machinekind_twin.robots import reload_robots

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
#: The test robot's package: a robot folder and a backend plugin.
TESTBOT = EXAMPLES / "robots" / "testbot"
#: The test robot's name, as its package registers it.
BOT = "testbot"
#: The test strategy's package: a strategy with its settings dataclass.
SWING = EXAMPLES / "strategies" / "swing"


def write_dist(site: Path, name: str, entry_points: dict[str, dict[str, str]],
               modules: dict[str, str] | None = None, version: str = "0.1.0") -> Path:
    """A distribution in `site` as `pip install` would leave it: its dist-info with these entry
    points ({group: {name: value}}), and optional top-level modules ({module: source})."""
    dist = site / f"{name.replace('-', '_')}-{version}.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                                   encoding="utf-8")
    groups = [f"[{group}]\n" + "".join(f"{ep} = {value}\n" for ep, value in eps.items())
              for group, eps in entry_points.items()]
    (dist / "entry_points.txt").write_text("\n".join(groups), encoding="utf-8")
    for module, source in (modules or {}).items():
        (site / f"{module}.py").write_text(source, encoding="utf-8")
    return dist


def _project(folder: Path) -> dict:
    return tomllib.loads((folder / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _install_testbot_for_the_session() -> None:
    """`pip install examples/robots/testbot` for this test run, ahead of every other distribution."""
    site = Path(tempfile.mkdtemp(prefix="machinekind-twin-testbot-"))
    atexit.register(shutil.rmtree, site, ignore_errors=True)
    project = _project(TESTBOT)
    write_dist(site, project["name"], project["entry-points"], version=project["version"])
    sys.path[:0] = [str(site), str(TESTBOT)]
    reload_robots()


_install_testbot_for_the_session()


@pytest.fixture(autouse=True)
def _no_config_from_env(monkeypatch):
    """Testy licza na wartosciach domyslnych - plik z MACHINEKIND_TWIN_CONFIG w powloce by je zmienil."""
    # setenv zapamietuje stan sprzed testu, wiec to, co ustawi `twin --config`
    # w os.environ, zniknie po tescie (samo delenv nieistniejacej zmiennej niczego nie zapamietuje).
    monkeypatch.setenv("MACHINEKIND_TWIN_CONFIG", "")
    monkeypatch.delenv("MACHINEKIND_TWIN_CONFIG")


@pytest.fixture
def install_dist(tmp_path, monkeypatch):
    """Put a distribution on `sys.path` as `pip install` would leave it: its dist-info with these
    entry points ({group: {name: value}}), and optional top-level modules ({module: source})."""
    def install(name: str, entry_points: dict[str, dict[str, str]], modules: dict[str, str] | None = None,
                version: str = "0.1.0") -> Path:
        site = tmp_path / f"site-{name}"
        dist = write_dist(site, name, entry_points, modules, version)
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
    project = _project(folder)
    monkeypatch.syspath_prepend(str(folder))
    install_dist(project["name"], project["entry-points"], version=project["version"])


@pytest.fixture
def testbot():
    """The test robot (`examples/robots/testbot`, installed for the session): its robot folder and backend."""
    return BOT


@pytest.fixture
def swing(install_dist, monkeypatch):
    """The test strategy (`examples/strategies/swing`), installed."""
    install_example(install_dist, monkeypatch, SWING)
    return "swing"
