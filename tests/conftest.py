"""Shared test helpers: no config file from the shell, and the example packages as if they were installed.

The twin holds no robot of its own, so the test robot (`examples/robots/testbot`) is installed for
the whole run, before collection: test modules build their robot at import (`get_spec(BOT)`).
"""

from __future__ import annotations

import atexit
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tomllib
from machinekind_twin_api.entry_points import DETECTORS

from machinekind_twin.robots import reload_robots

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
#: The test robot's package: a robot folder and a backend plugin.
TESTBOT = EXAMPLES / "robots" / "testbot"
#: The test robot's name, as its package registers it.
BOT = "testbot"
#: The test strategy's package: a strategy with its settings dataclass.
SWING = EXAMPLES / "strategies" / "swing"
#: The example detector's package: a colour cube, settings in a dataclass.
CUBE = EXAMPLES / "detectors" / "cube"
#: The LLM detector's package: frames to the claude or codex CLI.
LLM = Path(__file__).resolve().parents[1] / "packages" / "machinekind_twin_llm"


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


@pytest.fixture(autouse=True)
def _episodes_in_tmp(tmp_path, monkeypatch):
    """Keep explicitly enabled recordings under the test's temporary directory."""
    from machinekind_twin import episode

    monkeypatch.setattr(episode, "data_path", lambda rel: tmp_path / rel)


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


def install_detector(install_dist, dist: str, source: str) -> str:
    """Another installed package: a module named like it, whose `Detector` it registers under that name.

    Every test needs its own `dist`: a module once imported stays in `sys.modules`.
    """
    module = dist.replace("-", "_")
    install_dist(dist, {DETECTORS: {module: f"{module}:Detector"}}, {module: source})
    return module


@pytest.fixture
def testbot():
    """The test robot (`examples/robots/testbot`, installed for the session): its robot folder and backend."""
    return BOT


@pytest.fixture
def swing(install_dist, monkeypatch):
    """The test strategy (`examples/strategies/swing`), installed."""
    install_example(install_dist, monkeypatch, SWING)
    return "swing"


@pytest.fixture
def cube_example(install_dist, monkeypatch):
    """The example detector (`examples/detectors/cube`), installed."""
    install_example(install_dist, monkeypatch, CUBE)
    return "cube"


@pytest.fixture
def llm_package(install_dist, monkeypatch):
    """The LLM detector (`packages/machinekind_twin_llm`), installed."""
    install_example(install_dist, monkeypatch, LLM)
    return "llm"


class CubeScene:
    """A demo-like stand in the twin: two trusted sim cameras, a 3 cm cube free on the table, the arm.

    The cameras' calibrated pose is their true pose (`T_cam2base = sim_pose`), so the perception
    is judged against the physics. Built like so-arm101's `test_lift_from_cameras.py`.
    """

    #: Half the cube's edge [m].
    HALF = 0.015
    #: Camera names, intrinsics and eyes; both look at (0.2, 0, 0) on the table.
    K = ((560.0, 0.0, 322.0), (0.0, 560.0, 236.0), (0.0, 0.0, 1.0))
    EYES = (("a", (0.55, -0.45, 0.45)), ("b", (0.5, 0.5, 0.5)))

    def __init__(self):
        from machinekind_twin import scene as sc
        from machinekind_twin.calib.simulate import look_at
        from machinekind_twin.clock import VirtualClock
        from machinekind_twin.runtime import Twin
        from machinekind_twin.workspace import CameraRecord, Workspace

        self.ws = Workspace(robot=BOT)
        for name, eye in self.EYES:
            T = look_at(eye, (0.2, 0.0, 0.0)).tolist()
            self.ws.add_camera(CameraRecord(name, "sim", 640, 480, K=[list(r) for r in self.K], sim_pose=T,
                                            T_cam2base=T, intrinsics_from="symulacja",
                                            calibration={"trusted": True}))
        self.cameras = tuple(name for name, _ in self.EYES)
        self.twin = Twin(self.ws, clock=VirtualClock(t0=0.0))
        h = self.HALF
        self.twin.configure(objects=[sc.Box("cube", (h, h, h), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0))],
                            grasp_sensors=["cube"])
        self.twin.connect("sim", threaded=False)

    def place(self, x: float, y: float, yaw_deg: float = 0.0) -> None:
        """The cube resting on the table at (x, y) in the base frame, turned by `yaw_deg` about z."""
        import mujoco

        with self.twin.lock:
            s = self.twin.scene
            m, d = s.model, s.data
            b = m.body("cube").id
            a = m.jnt_qposadr[m.body_jntadr[b]]
            T = s.T_base2world
            half = np.radians(yaw_deg) / 2
            q_base = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
            qb, qw = np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(qb, T[:3, :3].ravel())
            mujoco.mju_mulQuat(qw, qb, q_base)
            d.qpos[a:a + 3] = T[:3, :3] @ np.array([x, y, self.HALF]) + T[:3, 3]
            d.qpos[a + 3:a + 7] = qw
            v = m.jnt_dofadr[m.body_jntadr[b]]
            d.qvel[v:v + 6] = 0.0
            mujoco.mj_forward(m, d)

    def truth(self) -> tuple[np.ndarray, np.ndarray]:
        """The cube's centre and rotation in the base frame, from the physics."""
        from machinekind_twin.kinematics import inverse

        with self.twin.lock:
            s = self.twin.scene
            b = s.model.body("cube").id
            Ti = inverse(s.T_base2world)
            return (Ti[:3, :3] @ s.data.xpos[b] + Ti[:3, 3],
                    Ti[:3, :3] @ s.data.xmat[b].reshape(3, 3))

    def frames(self) -> dict[str, np.ndarray]:
        return {c: self.twin.render(c) for c in self.cameras}

    def occluders(self) -> dict[str, np.ndarray]:
        """The arm's pixels in each camera, widened by 5 px, as the panel computes them."""
        from machinekind_twin.ui.watch import arm_mask

        cams = self.cameras
        return self.twin.render_with(lambda s: {c: arm_mask(s, c, dilate=5) for c in cams})

    def close(self) -> None:
        self.twin.close()


@pytest.fixture
def cube_scene():
    """The `CubeScene` stand (render tests only: it needs MuJoCo and OpenGL)."""
    pytest.importorskip("mujoco")
    scene = CubeScene()
    try:
        yield scene
    finally:
        scene.close()
