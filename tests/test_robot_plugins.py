"""The robot contract: robots as folders with `robot.yaml`, backends as plugins found through entry points."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

pytest.importorskip("mujoco")

from conftest import TESTBOT  # noqa: E402

from lerobot_mp.robot import backend_context, backends, create_backend  # noqa: E402
from lerobot_mp.robot.factory import PluginBackend  # noqa: E402
from lerobot_mp.twin.clock import VirtualClock  # noqa: E402
from lerobot_mp.twin.robots import (  # noqa: E402
    ROBOT_FILE, get_spec, load_robot, reload_robots, robot_config, robot_folders,
)
from lerobot_mp.twin.runtime import Twin  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DT = 0.02


def run(twin: Twin, seconds: float) -> None:
    for _ in range(int(round(seconds / DT))):
        twin.step(DT)


# ------------------------------------------------------------ robot folders
@pytest.mark.parametrize("name", ["so101", "a1x"])
def test_robots_still_in_this_repository_load_from_their_folders(name):
    folder = robot_folders()[name]
    assert folder.resolve() == (REPO / "assets" / "robots" / name).resolve()
    spec = get_spec(name)
    assert spec.name == name and spec.root == folder.resolve()
    assert spec.mjcf_path.parent == spec.root and spec.mjcf_path.is_file()
    assert spec.connections and set(spec.joint_labels) == set(spec.joints)


def test_a_robot_folder_from_an_installed_package_is_found_through_its_entry_point(testbot):
    assert robot_folders()[testbot] == TESTBOT / "testbot" / "robot"
    spec = get_spec(testbot)
    assert spec.joints == ("j1", "j2", "gripper") and spec.gripper == "gripper"
    assert spec.mjcf_path == TESTBOT / "testbot" / "robot" / "testbot.xml"
    assert [c.backend for c in spec.connections] == ["testbot"]


def test_two_robots_with_one_name_are_refused(testbot, tmp_path, monkeypatch):
    local = tmp_path / "assets" / "robots" / testbot
    shutil.copytree(TESTBOT / "testbot" / "robot", local)
    monkeypatch.chdir(tmp_path)                       # local robot folders resolve from the working directory first
    reload_robots()
    with pytest.raises(ValueError, match="Dwa ramiona o nazwie 'testbot'"):
        robot_folders()


@pytest.fixture
def folder(tmp_path):
    """A copy of the test robot's folder whose `robot.yaml` a test may break."""
    out = tmp_path / "robot"
    shutil.copytree(TESTBOT / "testbot" / "robot", out)
    return out


def rewrite(folder: Path, **changes) -> None:
    data = yaml.safe_load((folder / ROBOT_FILE).read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            data.pop(key)
        else:
            data[key] = value
    (folder / ROBOT_FILE).write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.mark.parametrize("changes, error", [
    ({"tcp_site": None}, "Brak kluczy.*tcp_site"),
    ({"tcp_frame": "tcp"}, "Nieznane klucze.*tcp_frame"),
    ({"root": "/elsewhere"}, "root"),
    ({"max_vel": {"j1": 90.0, "j2": 90.0}}, "max_vel.*gripper"),
    ({"gripper_range_q": None}, "gripper_range_q"),
    ({"home": {"j1": 0.0, "shoulder_pan": 0.0}}, "home.*shoulder_pan"),
    ({"mjcf": "missing.xml"}, "missing.xml"),
    ({"limits": {"j1": [-90.0], "j2": [-60.0, 60.0], "gripper": [0.0, 100.0]}}, "2 wartosci"),
])
def test_a_broken_description_names_what_is_wrong(folder, changes, error):
    rewrite(folder, **changes)
    with pytest.raises((ValueError, TypeError), match=error):
        load_robot(folder)


def test_the_description_names_the_robot_it_is_registered_as(testbot, tmp_path, monkeypatch):
    local = tmp_path / "assets" / "robots" / "other"
    shutil.copytree(TESTBOT / "testbot" / "robot", local)
    monkeypatch.chdir(tmp_path)
    reload_robots()
    with pytest.raises(ValueError, match="name: testbot.*'other'"):
        get_spec("other")


# ------------------------------------------------------------ config for one robot
def test_joint_tables_of_another_robot_are_skipped_and_this_robots_apply(tmp_path):
    so101_tables = tmp_path / "so101.yaml"
    so101_tables.write_text("joints: {shoulder_pan: {min: -50.0, max: 50.0}}\n"
                            "safety: {home: {shoulder_pan: 10.0, gripper: 20.0}, velocity_scale: 0.5}\n")
    a1x = get_spec("a1x")
    cfg = robot_config(a1x, so101_tables)
    assert cfg.joint("gripper").max == 100.0 and cfg.safety.home == a1x.home
    assert cfg.safety.velocity_scale == 0.5                             # the rest of the file applies

    so101 = get_spec("so101")
    cfg = robot_config(so101, so101_tables)
    assert cfg.joint("shoulder_pan").max == 50.0
    assert cfg.safety.home == {"shoulder_pan": 10.0, "gripper": 20.0}


def test_a_backend_section_reaches_its_plugin_raw(testbot):
    spec = get_spec(testbot)
    cfg = robot_config(spec, overrides={"robot": {"backend": "testbot"}, "testbot": {"start": {"j1": 12.0}},
                                        "feetech": {"baudrate": 9600}})
    ctx = backend_context(cfg, spec)
    assert ctx.settings == {"start": {"j1": 12.0}}
    assert ctx.joints == spec.joints and ctx.limits["j2"] == (-60.0, 60.0)

    arm = create_backend(cfg, spec)
    arm.connect()
    assert arm.read_joints()["j1"] == 12.0

    cfg = robot_config(spec, overrides={"robot": {"backend": "testbot"}, "testbot": {"strat": {}}})
    with pytest.raises(ValueError, match="strat"):
        create_backend(cfg, spec)


def test_backends_of_this_repository_are_registered_like_any_plugin():
    found = backends()
    assert {name: found[name].value for name in ("feetech", "lerobot", "a1x")} == {
        "feetech": "lerobot_mp.robot.feetech:create",
        "lerobot": "lerobot_mp.robot.lerobot_backend:create",
        "a1x": "lerobot_mp.robot.a1x:create",
    }


# ------------------------------------------------------------ a plugin backend in the twin
def test_the_test_robot_package_never_imports_the_twin():
    for path in (TESTBOT / "testbot").rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert not line.lstrip().startswith(("import lerobot_mp", "from lerobot_mp")), f"{path}: {line}"


def test_the_twin_drives_the_test_robot_through_its_plugin(testbot):
    twin = Twin(Workspace(robot=testbot), clock=VirtualClock(t0=0.0))
    try:
        twin.connect("testbot", threaded=False)
        backend = twin._backend
        assert isinstance(backend, PluginBackend)
        assert backend.info.name == "testbot" and not backend.info.simulated
        assert backend.faults() == [] and backend.joint_limits() == {} and backend.gripper_range_q() is None
        arm = backend.plugin
        assert backend.positions is arm.positions                       # anything else reads through
        run(twin, 0.5)
        assert arm.limits == {n: (lo, hi) for n, (lo, hi) in get_spec(testbot).limits.items()}

        twin.set_engaged(True)
        twin.set_target({"j1": 30.0})
        run(twin, 2.0)
        assert arm.positions["j1"] == pytest.approx(30.0, abs=0.5)
        assert twin.status.measured["j1"] == pytest.approx(30.0, abs=0.5)
    finally:
        twin.close()
    assert not arm.is_connected


def test_a_gripper_range_other_than_the_descriptions_is_reported(monkeypatch):
    from test_twin_runtime import FakeArm, connect_fake

    twin = Twin(Workspace(), clock=VirtualClock(t0=0.0))
    try:
        arm = FakeArm()
        arm.grip_range = (-0.0951, 0.70)                    # open 15 servo degrees short of the description
        connect_fake(twin, monkeypatch, arm)
        assert any("gripper_range_q" in w for w in twin.status.warnings)
        twin.disconnect()

        arm.grip_range = (-0.0951, 0.9541)                  # within 20 servo ticks: no warning
        connect_fake(twin, monkeypatch, arm)
        assert not any("gripper_range_q" in w for w in twin.status.warnings)
    finally:
        twin.close()
