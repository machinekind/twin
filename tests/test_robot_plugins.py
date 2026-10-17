"""The robot contract: robots as folders with `robot.yaml`, backends as plugins found through entry points."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import machinekind_twin_api.backend as api_backend
import pytest
import yaml
from machinekind_twin_api.entry_points import BACKENDS, ROBOTS

pytest.importorskip("mujoco")

from conftest import BOT, TESTBOT  # noqa: E402

import machinekind_twin.backends.base as twin_backend  # noqa: E402
from machinekind_twin.backends import backend_context, create_backend  # noqa: E402
from machinekind_twin.backends.factory import PluginBackend  # noqa: E402
from machinekind_twin.cli import main, parser  # noqa: E402
from machinekind_twin.clock import VirtualClock  # noqa: E402
from machinekind_twin.robots import (  # noqa: E402
    DEMO_FILE, ROBOT_FILE, get_spec, load_robot, robot_config, robot_folders,
)
from machinekind_twin.runtime import Twin  # noqa: E402
from machinekind_twin.workspace import Workspace  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DT = 0.02


def run(twin: Twin, seconds: float) -> None:
    for _ in range(int(round(seconds / DT))):
        twin.step(DT)


# ------------------------------------------------------------ robot folders
def test_a_robot_folder_from_an_installed_package_is_found_through_its_entry_point(testbot):
    assert robot_folders()[testbot] == TESTBOT / "testbot" / "robot"
    spec = get_spec(testbot)
    assert spec.joints == ("j1", "j2", "gripper") and spec.gripper == "gripper"
    assert spec.mjcf_path == TESTBOT / "testbot" / "robot" / "testbot.xml"
    assert [c.backend for c in spec.connections] == ["testbot"]


def test_the_twin_itself_holds_no_robot_folder():
    package = REPO / "src" / "machinekind_twin"
    found = {p.parent.relative_to(package).parts[:2] for p in package.rglob("robot.yaml")}
    assert not found
    assert not (REPO / "assets" / "robots").exists()


def test_a_robot_plugin_that_does_not_load_leaves_the_others(install_dist, caplog):
    install_dist("broken-robot", {"machinekind_twin.robots": {"broken": "no_such_module:DIR"}})
    folders = robot_folders()
    assert "broken" not in folders and BOT in folders
    assert "broken" in caplog.text


def a_folder_package(install_dist, tmp_path, dist: str, name: str) -> Path:
    """Another installed package registering a copy of the test robot's folder as robot `name`."""
    folder = tmp_path / f"{dist}-folder"
    shutil.copytree(TESTBOT / "testbot" / "robot", folder)
    module = dist.replace("-", "_")
    install_dist(dist, {"machinekind_twin.robots": {name: f"{module}:DIR"}},
                 {module: f"DIR = {str(folder)!r}\n"})
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


#: The test robot's wrist camera as its `robot.yaml` declares it.
WRIST = {
    "body": "hand",
    "pose": [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.03], [0.0, 0.0, 0.0, 1.0]],
    "fovy_deg": 60.0,
    "width": 160,
    "height": 120,
}
WRIST_WITHOUT_FOVY = {key: value for key, value in WRIST.items() if key != "fovy_deg"}
SHEARED = [[1.0, 0.5, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


@pytest.mark.parametrize("changes, error", [
    ({"tcp_site": None}, "Brak kluczy.*tcp_site"),
    ({"tcp_frame": "tcp"}, "Nieznane klucze.*tcp_frame"),
    ({"root": "/elsewhere"}, "root"),
    ({"max_vel": {"j1": 90.0, "j2": 90.0}}, "max_vel.*gripper"),
    ({"gripper_range_q": None}, "gripper_range_q"),
    ({"home": {"j1": 0.0, "shoulder_pan": 0.0}}, "home.*shoulder_pan"),
    ({"mjcf": "missing.xml"}, "missing.xml"),
    ({"limits": {"j1": [-90.0], "j2": [-60.0, 60.0], "gripper": [0.0, 100.0]}}, "2 wartosci"),
    ({"cameras": {"wrist": {**WRIST, "lens": "fisheye"}}}, "Nieznane klucze.*cameras.wrist.*lens"),
    ({"cameras": {"wrist": WRIST_WITHOUT_FOVY}}, "Brak kluczy.*cameras.wrist.*fovy_deg"),
    ({"cameras": {"wrist": {**WRIST, "body": "wrist_link"}}}, "cameras.wrist.*nie ma ciala 'wrist_link'"),
    ({"cameras": {"wrist": {**WRIST, "pose": SHEARED}}}, "cameras.wrist.*przeksztalceniem sztywnym"),
    ({"cameras": {"wrist": {**WRIST, "pose": WRIST["pose"][:3]}}}, "4 wartosci"),
    ({"cameras": {"wrist": {**WRIST, "width": 0}}}, "cameras.wrist.*dodatnimi liczbami calkowitymi"),
    ({"cameras": {"wrist": {**WRIST, "height": True}}}, "cameras.wrist.*dodatnimi liczbami calkowitymi"),
])
def test_a_broken_description_names_what_is_wrong(folder, changes, error):
    rewrite(folder, **changes)
    with pytest.raises((ValueError, TypeError), match=error):
        load_robot(folder)


def test_the_description_names_the_robot_it_is_registered_as(install_dist, tmp_path):
    a_folder_package(install_dist, tmp_path, "renamed-robot", "other")
    with pytest.raises(ValueError, match="name: testbot.*'other'"):
        get_spec("other")


def test_the_demo_station_takes_an_installed_robot_and_names_what_to_install_otherwise(tmp_path, capsys):
    from machinekind_twin.robots import default_robot

    station = tmp_path / "twin.json"
    assert main(["demo", "--path", str(station)]) == 0          # the first installed robot
    ws = Workspace.load(station)
    assert ws.robot == default_robot() and ws.spec().name == ws.robot
    assert len(ws.cameras) == 2 and all(c.trusted for c in ws.cameras)
    assert main(["demo", "--path", str(station), "--force", "--robot", BOT]) == 0
    assert Workspace.load(station).robot == BOT

    assert main(["demo", "--path", str(tmp_path / "other.json"), "--robot", "no_such_robot"]) == 1
    assert "uv sync" in capsys.readouterr().err
    assert not (tmp_path / "other.json").exists()


def test_the_first_twin_ui_writes_the_example_workstation_and_keeps_an_existing_one(tmp_path, monkeypatch,
                                                                                    capsys):
    pytest.importorskip("viser")
    from machinekind_twin.robots import default_robot
    from machinekind_twin.ui import app

    started = []
    monkeypatch.setattr(app, "main", lambda argv: started.append(argv) or 0)
    station = tmp_path / "twin.json"

    assert main(["ui", "--workspace", str(station)]) == 0
    ws = Workspace.load(station)
    assert ws.robot == default_robot() and len(ws.cameras) == 2
    assert "przykladowe stanowisko" in capsys.readouterr().out

    station.write_text(json.dumps({"robot": BOT}), encoding="utf-8")
    assert main(["ui", "--workspace", str(station)]) == 0
    assert json.loads(station.read_text(encoding="utf-8")) == {"robot": BOT}
    assert started == [["--host", "127.0.0.1", "--port", "8080", "--workspace", str(station)]] * 2


def test_a_new_workspace_names_the_first_installed_robot():
    from machinekind_twin.robots import default_robot

    assert Workspace().robot == default_robot() == next(iter(robot_folders()))


def test_demo_copies_the_example_workspace_its_robot_folder_ships(install_dist, tmp_path):
    example = {"robot": "demobot", "backend": "sim"}
    folder = a_folder_package(install_dist, tmp_path, "demo-robot", "demobot")
    rewrite(folder, name="demobot")
    (folder / DEMO_FILE).write_text(json.dumps(example), encoding="utf-8")
    out = tmp_path / "twin.json"

    code = main(["demo", "--robot", "demobot", "--path", str(out)])

    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8")) == example


def test_calib_sim_takes_any_installed_robot(testbot):
    args = parser().parse_args(["calib-sim", "--robot", testbot])

    assert args.robot == testbot


# ------------------------------------------------------------ config for one robot
def test_joint_tables_of_another_robot_are_skipped_and_this_robots_apply(tmp_path):
    so101_tables = tmp_path / "so101.yaml"
    so101_tables.write_text("joints: {shoulder_pan: {min: -50.0, max: 50.0}}\n"
                            "safety: {home: {shoulder_pan: 10.0, gripper: 20.0}, velocity_scale: 0.5}\n")
    bot = get_spec(BOT)
    cfg = robot_config(bot, so101_tables)
    assert cfg.joint("gripper").max == 100.0 and cfg.safety.home == bot.home
    assert cfg.safety.velocity_scale == 0.5                             # the rest of the file applies

    bot_tables = tmp_path / "testbot.yaml"
    bot_tables.write_text("joints: {j1: {min: -50.0, max: 50.0}}\n"
                          "safety: {home: {j1: 10.0, gripper: 20.0}, velocity_scale: 0.5}\n")
    cfg = robot_config(bot, bot_tables)
    assert cfg.joint("j1").max == 50.0
    assert cfg.safety.home == {**bot.home, "j1": 10.0, "gripper": 20.0}


def test_a_partial_home_in_the_file_keeps_the_rest_of_the_descriptions_home(testbot, tmp_path):
    """A table naming only the shared `gripper` applies to any robot with one; its other joints must not
    go to 0, which on an arm like the A1X is the folded pose at the end stops."""
    grip_only = tmp_path / "grip.yaml"
    grip_only.write_text("safety: {home: {gripper: 35.0}}\n")
    spec = get_spec(testbot)
    assert robot_config(spec, grip_only).safety.home == {**spec.home, "gripper": 35.0}
    station = {n: 1.0 for n in spec.joints}                             # "Set home = here" names every joint
    assert robot_config(spec, grip_only, overrides={"safety": {"home": station}}).safety.home == station


def test_the_config_check_at_startup_accepts_joint_tables(tmp_path, monkeypatch):
    from machinekind_twin.cli import _use_config

    good = tmp_path / "local.yaml"
    good.write_text("joints: {j1: {min: -90.0}}\ntestbot: {start: {j1: 5.0}}\n")
    _use_config(str(good))
    bad = tmp_path / "bad.yaml"
    bad.write_text("robot: {baudrate: 500000}\n")
    with pytest.raises(SystemExit, match="baudrate"):
        _use_config(str(bad))


def test_a_backend_section_reaches_its_plugin_raw(testbot):
    spec = get_spec(testbot)
    cfg = robot_config(spec, overrides={"robot": {"backend": "testbot"}, "testbot": {"start": {"j1": 12.0}},
                                        "other_backend": {"baudrate": 9600}})
    ctx = backend_context(cfg, spec)
    assert ctx.settings == {"start": {"j1": 12.0}}
    assert ctx.joints == spec.joints and ctx.limits["j2"] == (-60.0, 60.0)

    arm = create_backend(cfg, spec)
    arm.connect()
    assert arm.read_joints()["j1"] == 12.0

    cfg = robot_config(spec, overrides={"robot": {"backend": "testbot"}, "testbot": {"strat": {}}})
    with pytest.raises(ValueError, match="strat"):
        create_backend(cfg, spec)


# ------------------------------------------------------------ a plugin backend in the twin
@pytest.mark.parametrize("name", api_backend.__all__)
def test_the_backend_contract_is_machinekind_twin_apis_own_not_a_copy(name):
    """A plugin subclasses `machinekind_twin_api.RobotBackend`; the factory's `isinstance` must
    recognise it."""
    assert getattr(twin_backend, name) is getattr(api_backend, name)


def test_the_twin_reads_the_entry_point_groups_machinekind_twin_api_names():
    from machinekind_twin import robots
    from machinekind_twin.backends import factory

    assert robots.ENTRY_POINT_GROUP == ROBOTS
    assert factory.ENTRY_POINT_GROUP == BACKENDS


def test_the_test_robot_package_never_imports_the_twin():
    """Only the contracts (`machinekind_twin_api`), anywhere in the package, also inside a function."""
    for path in (TESTBOT / "testbot").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue

            for module in modules:
                assert module.split(".")[0] != "machinekind_twin", f"{path}: {module}"


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

    twin = Twin(Workspace(robot=BOT), clock=VirtualClock(t0=0.0))
    try:
        arm = FakeArm()
        arm.grip_range = (0.0, 0.015)                       # opens 5 mm short of the description's 20 mm
        connect_fake(twin, monkeypatch, arm)
        assert any("gripper_range_q" in w for w in twin.status.warnings)
        twin.disconnect()

        arm.grip_range = (0.0, 0.0198)                      # within 3 % of the travel: no warning
        connect_fake(twin, monkeypatch, arm)
        assert not any("gripper_range_q" in w for w in twin.status.warnings)
    finally:
        twin.close()
