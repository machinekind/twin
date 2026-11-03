"""Other robots of the station: in the scene, kinematic, read only, kept clear of, masked out.

The test robot stands as the second arm 19 cm to the driven arm's left (+y), as on the stand.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco
from conftest import BOT

from machinekind_twin import scene as sc
from machinekind_twin.backends import open_read_only
from machinekind_twin.clock import VirtualClock
from machinekind_twin.collision import CollisionChecker
from machinekind_twin.robots import get_spec, robot_config, robot_folders
from machinekind_twin.runtime import Twin
from machinekind_twin.workspace import RobotRecord, Workspace

LEFT = [0.0, 0.19, 0.0]
#: The driven arm turned towards the other one, reaching through its upright link.
INTO_IT = {"j1": 75.0, "j2": 0.0, "gripper": 0.0}
DT = 0.02


def station(robot: str = BOT, **kw) -> Workspace:
    ws = Workspace(robot=robot)
    ws.other_robots.append(RobotRecord("lewe", robot, xyz=list(LEFT), **kw))
    return ws


def body_pos(scene: sc.Scene, name: str) -> np.ndarray:
    return scene.data.xpos[scene.model.body(name).id].copy()


# ------------------------------------------------------------------ workspace
def test_a_workspace_without_other_robots_saves_as_before(tmp_path):
    ws = Workspace(robot=BOT)
    assert "other_robots" not in ws.to_dict()
    path = ws.save(tmp_path / "twin.json")
    assert "other_robots" not in json.loads(path.read_text(encoding="utf-8"))
    assert Workspace.load(path).other_robots == []
    assert [r.name for r in ws.robots()] == ["robot"]


def test_other_robots_round_trip_and_unknown_keys_fail(tmp_path):
    ws = station(backend="testbot", port="/dev/ttyACM1", joints={"j1": 10.0})
    path = ws.save(tmp_path / "twin.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["other_robots"] == [{"name": "lewe", "robot": BOT, "xyz": LEFT, "yaw_deg": 0.0,
                                     "backend": "testbot", "port": "/dev/ttyACM1", "joints": {"j1": 10.0}}]
    assert Workspace.load(path).other_robots == ws.other_robots
    assert [r.name for r in Workspace.load(path).robots()] == ["robot", "lewe"]

    data["other_robots"][0]["speed"] = 1.0
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="speed"):
        Workspace.load(path)


# ---------------------------------------------------------------------- scene
def test_the_scene_takes_a_full_copy_of_the_other_robot_at_its_pose():
    ws = station()
    alone = sc.build(Workspace(robot=BOT).scene_config())
    both = sc.build(ws.scene_config())
    m, m0 = both.model, alone.model
    robot = mujoco.MjModel.from_xml_path(str(get_spec(BOT).mjcf_path))

    # Unique names; the driven arm and everything else keep their names and ids.
    names = [m.body(b).name for b in range(m.nbody)]
    assert len(set(names)) == len(names)
    assert names[:m0.nbody] == [m0.body(b).name for b in range(m0.nbody)]
    assert [m.actuator(a).name for a in range(m.nu)] == [m0.actuator(a).name for a in range(m0.nu)]
    assert (m.njnt, m.nq, m.nsensor, m.ncam) == (m0.njnt, m0.nq, m0.nsensor, m0.ncam)
    # Every body of the robot with its mass, and every geom (visual and collision).
    for b in range(1, robot.nbody):
        copy = m.body("lewe/" + robot.body(b).name).id
        assert m.body_mass[copy] == pytest.approx(robot.body_mass[b])
    passive = both.passive_geom
    assert passive.sum() == robot.ngeom
    assert (m.geom_margin[passive & (m.geom_contype > 0)] == sc.PASSIVE_CLEARANCE).all()
    # Its base 19 cm to the left of the driven arm's, same orientation.
    T = both.T_base2world @ ws.other_robots[0].T_base()
    assert np.allclose(body_pos(both, "lewe/base"), T[:3, 3])
    assert np.allclose(T[:3, 3] - both.T_base2world[:3, 3], [0.0, 0.19, 0.0])


def test_the_other_robot_stands_in_its_joints_and_follows_new_ones():
    ws = station(joints={"j1": 90.0})
    scene = sc.build(ws.scene_config())
    turned = body_pos(scene, "lewe/hand") - body_pos(scene, "lewe/base")
    assert abs(turned[0]) < 1e-6 and turned[1] > 0.1           # j1 = 90: the link points to +y

    scene.set_passive_joints("lewe", {"j1": 0.0})
    ahead = body_pos(scene, "lewe/hand") - body_pos(scene, "lewe/base")
    assert ahead[0] > 0.1 and abs(ahead[1]) < 1e-6


def test_the_other_robot_does_not_move_under_gravity_or_when_hit():
    scene = sc.build(station().scene_config())
    bodies = [b for b in range(scene.model.nbody) if scene.model.body(b).name.startswith("lewe/")]
    before = scene.data.xpos[bodies].copy()
    scene.set_joints({"j1": 0.0, "j2": 0.0, "gripper": 0.0})
    scene.command(INTO_IT)
    scene.step(2.0)

    assert np.array_equal(scene.data.xpos[bodies], before)
    # The driven arm was stopped by it, short of where it was sent.
    assert scene.joints()["j1"] < INTO_IT["j1"] - 5.0
    hit = {scene.model.body(scene.model.geom_bodyid[g]).name
           for c in scene.data.contact[:scene.data.ncon] for g in (c.geom1, c.geom2)}
    assert any(n.startswith("lewe/") for n in hit) and any(n.startswith("robot/") for n in hit)


def test_the_other_robot_exports_where_it_stands(tmp_path):
    ws = station(joints={"j1": 45.0})
    path = sc.export(ws.scene_config(), tmp_path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    live = sc.build(ws.scene_config())
    assert np.allclose(data.xpos[model.body("lewe/hand").id], body_pos(live, "lewe/hand"), atol=1e-5)


# ------------------------------------------------------------------ collisions
def test_a_pose_into_the_other_robot_is_a_collision():
    scene = sc.build(station().scene_config())
    checker = CollisionChecker(scene)
    home = get_spec(BOT).home

    assert checker.config_clear(home) is None and checker.passive_gap(home) is None
    assert "lewe/" in (checker.config_clear(INTO_IT) or "")
    assert "lewe/" in (checker.path_clear(home, INTO_IT) or "")
    gap, what = checker.passive_gap(INTO_IT)
    assert gap < sc.PASSIVE_CLEARANCE and "lewe/" in what
    # Closer is refused, backing away is not.
    assert checker.closer(home, INTO_IT)
    assert checker.closer(INTO_IT, home) is None


def test_a_clearance_does_not_add_to_the_other_robots_gap():
    """The wave's `clearance` grows the scene's geoms. The other robot's margin keeps
    PASSIVE_CLEARANCE already, so the larger of the two counts, not their sum."""
    scene = sc.build(station().scene_config())
    plain = CollisionChecker(scene)
    same = CollisionChecker(scene, clearance=sc.PASSIVE_CLEARANCE)
    wider = CollisionChecker(scene, clearance=2 * sc.PASSIVE_CLEARANCE)
    sweep = [{**INTO_IT, "j1": float(a)} for a in np.arange(0.0, INTO_IT["j1"], 0.5)]

    for pose in sweep:
        a, b = plain.passive_gap(pose), same.passive_gap(pose)
        assert (a is None) == (b is None)
        if a is not None and b is not None:
            assert b[0] == pytest.approx(a[0], abs=1e-6) and b[1] == a[1]

    def first(checker: CollisionChecker) -> int:
        return next(i for i, pose in enumerate(sweep) if checker.passive_gap(pose) is not None)

    assert first(wider) < first(plain)
    near = wider.passive_gap(sweep[first(wider)])
    assert near is not None and near[0] > sc.PASSIVE_CLEARANCE and "lewe/" in near[1]


def test_the_checker_sees_a_live_move_of_the_other_robot():
    scene = sc.build(station(joints={"j1": 90.0}).scene_config())     # its link points away
    checker = CollisionChecker(scene)
    reach = {"j1": 60.0, "j2": 0.0, "gripper": 0.0}
    assert checker.passive_gap(reach) is None
    scene.set_passive_joints("lewe", {"j1": -90.0})                  # its link swings towards ours
    assert checker.passive_gap(reach) is not None


def test_the_loop_stops_the_arm_before_the_other_robot():
    twin = Twin(station(), clock=VirtualClock(t0=0.0))
    try:
        twin.connect("sim", threaded=False)
        twin.claim("test")
        twin.set_engaged(True, owner="test")
        twin.set_target(INTO_IT, owner="test")
        checker = CollisionChecker(twin.scene)
        closest = np.inf
        for _ in range(int(3.0 / DT)):
            twin.step(DT)
            gap = checker.passive_gap(twin.joints())
            closest = min(closest, np.inf if gap is None else gap[0])
        assert twin.owner is None                                     # the owner lost the arm
        assert "za blisko innego robota" in twin.status.error
        assert closest > 0.0                                          # never touched it
        assert twin.joints()["j1"] < INTO_IT["j1"] - 10.0

        # Backing away is allowed.
        twin.claim("test")
        twin.set_engaged(True, owner="test")
        twin.set_target({"j1": 0.0}, owner="test")
        for _ in range(int(3.0 / DT)):
            twin.step(DT)
        assert abs(twin.joints()["j1"]) < 2.0
    finally:
        twin.close()


def test_without_other_robots_the_loop_drives_as_before():
    twin = Twin(Workspace(robot=BOT), clock=VirtualClock(t0=0.0))
    try:
        twin.connect("sim", threaded=False)
        twin.claim("test")
        twin.set_engaged(True, owner="test")
        twin.set_target(INTO_IT, owner="test")
        for _ in range(int(3.0 / DT)):
            twin.step(DT)
        assert twin.owner == "test" and twin.joints()["j1"] == pytest.approx(INTO_IT["j1"], abs=1.0)
    finally:
        twin.close()


@pytest.mark.skipif("so101" not in robot_folders(), reason="the soarm package is not installed")
def test_the_stand_so101_on_the_left_is_in_reach():
    ws = station("so101")
    scene = sc.build(ws.scene_config())
    checker = CollisionChecker(scene)
    home = get_spec("so101").home
    assert checker.config_clear(home) is None
    # Turned to the left and stretched up, the driven arm meets the other one.
    assert "lewe/" in (checker.config_clear({**home, "shoulder_pan": -90.0, "shoulder_lift": 0.0,
                                             "elbow_flex": 0.0}) or "")


@pytest.mark.skipif("so101" not in robot_folders(), reason="the soarm package is not installed")
def test_an_exported_station_with_a_second_so101_loads_from_its_own_files(tmp_path):
    """The other arm's meshes go to `assets/lewe/`: an attached model looked for them in its own
    package folder, so the export failed for any robot built from meshes."""
    folder = station("so101").export(tmp_path / "station")
    xml = (folder / "scene.xml").read_text(encoding="utf-8")
    assert 'file="lewe/' in xml and str(tmp_path) not in xml
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    assert model.nmesh == 2 * len(list((folder / "assets" / "lewe").glob("*.stl")))


# --------------------------------------------------------------- read-only bus
FAKE_BUS = '''
from machinekind_twin_api import read_only_capable

#: Every bus created, for the test to look at.
BUSES = []


class Bus:
    """A servo bus that logs every packet it gets: reads and writes apart."""

    def __init__(self, port):
        self.port, self.positions, self.reads, self.writes, self.fail = port, {}, 0, [], None
        BUSES.append(self)

    def read(self):
        if self.fail:
            raise OSError(self.fail)
        self.reads += 1
        return dict(self.positions)

    def write(self, what):
        self.writes.append(what)


class Arm:
    def __init__(self, ctx):
        self.read_only = ctx.read_only
        self.bus = Bus(ctx.port)
        self.bus.positions = {name: 0.0 for name in ctx.joints}
        self.connected = False

    def connect(self):
        if self.bus.port == "/dev/missing":
            raise OSError("no such port: /dev/missing")
        if not self.read_only:
            self.bus.write("torque on")            # what a real backend does on connect
        self.connected = True

    def disconnect(self):
        if not self.read_only:
            self.bus.write("torque off")
        self.connected = False

    @property
    def is_connected(self):
        return self.connected

    def read_joints(self):
        return self.bus.read()

    def send_joints(self, targets):
        self.bus.write(("goal", dict(targets)))
        return dict(targets)


@read_only_capable
def create(ctx):
    return Arm(ctx)


def create_unmarked(ctx):
    raise AssertionError("an unmarked factory must never be called")
'''


@pytest.fixture
def fakebus(install_dist):
    """A backend plugin `fakebus` on a fake bus that logs writes, and `unmarked` without read-only."""
    install_dist("fakebus-dist", {"machinekind_twin.backends": {"fakebus": "fakebus_mod:create",
                                                                "unmarked": "fakebus_mod:create_unmarked"}},
                 {"fakebus_mod": FAKE_BUS})
    import fakebus_mod

    fakebus_mod.BUSES.clear()
    return fakebus_mod


def wait_for(cond, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.02)


def test_a_read_only_bus_moves_the_other_robot_and_is_never_written_to(fakebus):
    ws = station(backend="fakebus", port="/dev/ttyACM1")
    twin = Twin(ws)
    try:
        twin.start_passive()
        wait_for(lambda: bool(fakebus.BUSES) and fakebus.BUSES[-1].reads > 0)
        bus = fakebus.BUSES[-1]
        bus.positions = {"j1": 90.0, "j2": 0.0, "gripper": 0.0}
        wait_for(lambda: ws.other_robot("lewe").joints.get("j1") == 90.0)
        with twin.lock:
            link = body_pos(twin.scene, "lewe/hand") - body_pos(twin.scene, "lewe/base")
        assert link[1] > 0.1                                          # turned to +y, as read
        assert twin.passive_status()["lewe"] == "odczyt na zywo"
        assert bus.port == "/dev/ttyACM1"
    finally:
        twin.close()
    assert fakebus.BUSES and all(b.writes == [] for b in fakebus.BUSES)


def test_the_twin_holds_a_read_only_backend_with_nothing_to_send(fakebus):
    spec = get_spec(BOT)
    arm = open_read_only(robot_config(spec), spec, "fakebus", "/dev/ttyACM1")
    arm.connect()
    assert arm.read_joints() == {"j1": 0.0, "j2": 0.0, "gripper": 0.0}
    arm.disconnect()
    for name in ("send_joints", "set_joint_limits", "step", "plugin"):
        assert not hasattr(arm, name)
    assert fakebus.BUSES[-1].writes == []


def test_a_backend_without_a_read_only_mode_is_never_opened(fakebus):
    spec = get_spec(BOT)
    with pytest.raises(ValueError, match="read_only"):
        open_read_only(robot_config(spec), spec, "unmarked", "/dev/ttyACM1")
    assert fakebus.BUSES == []


def test_a_missing_or_failing_bus_keeps_the_last_pose_and_says_so(fakebus):
    ws = station(backend="fakebus", port="/dev/missing", joints={"j1": 30.0})
    twin = Twin(ws)
    try:
        before = body_pos(twin.scene, "lewe/hand")
        twin.start_passive()
        wait_for(lambda: "zostaje ostatnia poza" in twin.passive_status()["lewe"])
        assert "/dev/missing" in twin.passive_status()["lewe"]
        assert np.allclose(body_pos(twin.scene, "lewe/hand"), before)
        assert ws.other_robot("lewe").joints == {"j1": 30.0}

        # A bus that worked and then fails.
        ws.other_robot("lewe").port = "/dev/ttyACM1"
        twin.start_passive()
        wait_for(lambda: twin.passive_status()["lewe"] == "odczyt na zywo")
        fakebus.BUSES[-1].fail = "device reports readiness to read but returned no data"
        wait_for(lambda: "zostaje ostatnia poza" in twin.passive_status()["lewe"])
        assert ws.other_robot("lewe").joints["j1"] == 0.0             # the last pose read
    finally:
        twin.close()
    assert all(b.writes == [] for b in fakebus.BUSES)


def test_a_bus_already_in_use_is_not_read_twice(fakebus):
    ws = station(backend="fakebus", port="/dev/ttyACM0")
    ws.backend, ws.port = "feetech", "/dev/ttyACM0"                   # the driven arm's bus
    twin = Twin(ws)
    try:
        twin.start_passive()
        assert "jest juz uzywany" in twin.passive_status()["lewe"]
        assert fakebus.BUSES == []
        with pytest.raises(ValueError, match="inny robot"):
            twin.connect("testbot", port="/dev/ttyACM0", threaded=False)
    finally:
        twin.close()


# ---------------------------------------------------------------------- render
@pytest.mark.render
def test_a_simulated_camera_sees_the_other_robot_and_its_mask_covers_it():
    from machinekind_twin.calib.simulate import look_at
    from machinekind_twin.ui.watch import arm_mask

    def frame_and_mask(ws):
        cfg = ws.scene_config()
        view = sc.CameraView.from_fov("cam", 320, 240, 50.0, look_at((0.3, 0.45, 0.4), (0.0, 0.19, 0.05)))
        cfg.cameras = [view]
        with sc.build(cfg) as scene:
            return scene.render("cam"), arm_mask(scene, "cam", dilate=0)

    alone, mask_alone = frame_and_mask(Workspace(robot=BOT))
    both, mask_both = frame_and_mask(station())
    differs = np.abs(both.astype(int) - alone.astype(int)).sum(axis=2) > 30
    added = mask_both & ~mask_alone
    # The camera sees it, and the mask covers it (beyond it the frame changes only by its shadow).
    assert added.sum() > 200
    assert (added & differs).sum() > 0.8 * added.sum()
