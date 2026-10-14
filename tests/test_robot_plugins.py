"""The robot contract: robots as folders with `robot.yaml`, backends as plugins found through entry points."""

from __future__ import annotations

import ast
import shutil
from importlib.metadata import entry_points
from pathlib import Path

import lerobot_mp_api.backend as api_backend
import pytest
import yaml
from lerobot_mp_api.entry_points import BACKENDS, ROBOTS

pytest.importorskip("mujoco")

from conftest import TESTBOT  # noqa: E402

import lerobot_mp.robot.base as twin_backend  # noqa: E402
from lerobot_mp.robot import backend_context, backends, create_backend  # noqa: E402
from lerobot_mp.robot.factory import PluginBackend  # noqa: E402
from lerobot_mp.twin.clock import VirtualClock  # noqa: E402
from lerobot_mp.twin.robots import (  # noqa: E402
    ROBOT_FILE, get_spec, load_robot, robot_config, robot_folders,
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
def test_so101_and_a1x_register_like_any_robot_package(name):
    """They still live here, as packages that leave whole: folder and backends in `lerobot_mp.robot.<x>`."""
    eps = {ep.name: ep.value for ep in entry_points(group="lerobot_mp.robots")}
    assert eps[name] == f"lerobot_mp.robot.{name}:ROBOT_DIR"
    folder = robot_folders()[name]
    assert folder == (REPO / "src" / "lerobot_mp" / "robot" / name / "robot").resolve()
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


def test_the_twin_itself_holds_no_robot_folder():
    package = REPO / "src" / "lerobot_mp"
    found = {p.parent.relative_to(package).parts[:2] for p in package.rglob("robot.yaml")}
    assert found <= {("robot", "so101"), ("robot", "a1x")}      # the two packages that still have to leave
    assert not (REPO / "assets" / "robots").exists()


def test_a_robot_plugin_that_does_not_load_leaves_the_others(install_dist, caplog):
    install_dist("broken-robot", {"lerobot_mp.robots": {"broken": "no_such_module:DIR"}})
    folders = robot_folders()
    assert "broken" not in folders and {"so101", "a1x"} <= set(folders)
    assert "broken" in caplog.text


def a_folder_package(install_dist, tmp_path, dist: str, name: str) -> Path:
    """Another installed package registering a copy of the test robot's folder as robot `name`."""
    folder = tmp_path / f"{dist}-folder"
    shutil.copytree(TESTBOT / "testbot" / "robot", folder)
    module = dist.replace("-", "_")
    install_dist(dist, {"lerobot_mp.robots": {name: f"{module}:DIR"}}, {module: f"DIR = {str(folder)!r}\n"})
    return folder


def test_two_robots_with_one_name_are_refused(testbot, install_dist, tmp_path):
    a_folder_package(install_dist, tmp_path, "second-testbot", testbot)
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


def test_the_description_names_the_robot_it_is_registered_as(install_dist, tmp_path):
    a_folder_package(install_dist, tmp_path, "renamed-robot", "other")
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
    assert cfg.safety.home == {**so101.home, "shoulder_pan": 10.0, "gripper": 20.0}


def test_a_partial_home_in_the_file_keeps_the_rest_of_the_descriptions_home(tmp_path):
    """A table naming only the shared `gripper` applies to the A1X too; its other joints must not go to 0,
    the folded pose at the end stops."""
    grip_only = tmp_path / "grip.yaml"
    grip_only.write_text("safety: {home: {gripper: 35.0}}\n")
    a1x = get_spec("a1x")
    assert robot_config(a1x, grip_only).safety.home == {**a1x.home, "gripper": 35.0}
    station = {n: 1.0 for n in a1x.joints}                              # "Set home = here" names every joint
    assert robot_config(a1x, grip_only, overrides={"safety": {"home": station}}).safety.home == station


def test_the_config_check_at_startup_accepts_joint_tables(tmp_path, monkeypatch):
    from lerobot_mp.twin.cli import _use_config

    good = tmp_path / "local.yaml"
    good.write_text("joints: {shoulder_pan: {min: -90.0}}\nfeetech: {baudrate: 500000}\n")
    _use_config(str(good))
    bad = tmp_path / "bad.yaml"
    bad.write_text("robot: {baudrate: 500000}\n")
    with pytest.raises(SystemExit, match="baudrate"):
        _use_config(str(bad))


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
        "feetech": "lerobot_mp.robot.so101.feetech:create",
        "lerobot": "lerobot_mp.robot.so101.lerobot_backend:create",
        "a1x": "lerobot_mp.robot.a1x.backend:create",
    }


# ------------------------------------------------------------ a plugin backend in the twin
@pytest.mark.parametrize("name", api_backend.__all__)
def test_the_backend_contract_is_lerobot_mp_apis_own_not_a_copy(name):
    """A plugin subclasses `lerobot_mp_api.RobotBackend`; the factory's `isinstance` must recognise it."""
    assert getattr(twin_backend, name) is getattr(api_backend, name)


def test_the_twin_reads_the_entry_point_groups_lerobot_mp_api_names():
    from lerobot_mp.robot import factory
    from lerobot_mp.twin import robots

    assert robots.ENTRY_POINT_GROUP == ROBOTS
    assert factory.ENTRY_POINT_GROUP == BACKENDS


def test_the_test_robot_package_never_imports_the_twin():
    """Only the contracts (`lerobot_mp_api`), anywhere in the package, also inside a function."""
    for path in (TESTBOT / "testbot").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue

            for module in modules:
                assert module.split(".")[0] != "lerobot_mp", f"{path}: {module}"


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


def test_the_wrapper_reports_a_silent_link_the_way_the_backends_here_flag_it(testbot):
    spec = get_spec(testbot)
    backend = create_backend(robot_config(spec, overrides={"robot": {"backend": "testbot"}}), spec)
    assert not backend.link_silent
    backend.plugin._link_silent = True
    assert backend.link_silent


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
