"""A robot that moves its base (a walking robot, a cart) reports where it is; its copy in the scene follows.

The test walker's fixed frame stands on the floor 40 cm to the driven arm's left, turned 90 degrees.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from conftest import BOT
from machinekind_twin_api.entry_points import BACKENDS
from test_twin_other_robots import FAKE_BUS, wait_for

from machinekind_twin import scene as sc
from machinekind_twin.backends import ReadOnlyArm
from machinekind_twin.collision import CollisionChecker
from machinekind_twin.kinematics import inverse, pose
from machinekind_twin.passive import PassiveReader
from machinekind_twin.robots import get_spec
from machinekind_twin.runtime import Twin
from machinekind_twin.workspace import RobotRecord, Workspace

ON_THE_FLOOR = [0.1, 0.4, -0.75]
#: Height of the trunk over the floor with the leg straight [m].
STANDING = 0.27
#: `odometry(0.2, 0.0, 30.0)` in the driven arm's base frame: 20 cm ahead of the frame turned to +y.
AHEAD = [0.1, 0.6, -0.48]

WALKER_BUS = FAKE_BUS + '''

class Walker(Arm):
    def base_pose(self):
        return getattr(self.bus, "base", None)


@read_only_capable
def create_walker(ctx):
    return Walker(ctx)
'''


@pytest.fixture
def walkerbus(install_dist):
    """The fake bus as a backend `walkerbus` that reports the base pose the test puts on its bus (`base`)."""
    install_dist("walkerbus-dist", {BACKENDS: {"walkerbus": "walkerbus_mod:create_walker"}},
                 {"walkerbus_mod": WALKER_BUS})
    import walkerbus_mod

    walkerbus_mod.BUSES.clear()
    return walkerbus_mod


def station(walker: str, **kw) -> Workspace:
    ws = Workspace(robot=BOT)
    ws.other_robots.append(RobotRecord("walker", walker, xyz=list(ON_THE_FLOOR), yaw_deg=90.0, **kw))
    return ws


def odometry(x: float, y: float, yaw_deg: float) -> np.ndarray:
    """The trunk standing at `x`, `y` [m], turned `yaw_deg`, in the walker's fixed frame."""
    c = np.cos(np.radians(yaw_deg))
    s = np.sin(np.radians(yaw_deg))
    return pose(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([x, y, STANDING]))


def in_arm_frame(scene: sc.Scene, body: str) -> np.ndarray:
    b = scene.model.body(body).id
    return inverse(scene.T_base2world) @ pose(scene.data.xmat[b].reshape(3, 3), scene.data.xpos[b])


def test_the_read_only_wrapper_hands_the_base_pose_through_and_none_for_a_plugin_without_one():
    reported = (odometry(0.2, 0.0, 30.0), 12.5)
    plugin = SimpleNamespace(connect=None, disconnect=None, read_joints=None, base_pose=lambda: reported)
    without = SimpleNamespace(connect=None, disconnect=None, read_joints=None)

    assert ReadOnlyArm(plugin, "walkerbus").base_pose() is reported
    assert ReadOnlyArm(without, "testbot").base_pose() is None


def test_the_compiled_trunk_hangs_from_the_world_in_world_coordinates(walker):
    scene = sc.build(station(walker).scene_config())
    b = scene.model.body("walker/trunk").id

    assert scene.model.body_parentid[b] == 0
    assert np.allclose(scene.model.body_pos[b], scene.data.xpos[b])
    assert np.allclose(scene.model.body_quat[b], scene.data.xquat[b])


def test_the_scene_puts_the_trunk_at_the_record_times_the_reported_pose(walker):
    scene = sc.build(station(walker).scene_config())
    scene.set_passive_base("walker", odometry(0.2, 0.0, 30.0))
    trunk = in_arm_frame(scene, "walker/trunk")

    assert np.allclose(trunk[:3, 3], AHEAD)
    assert np.degrees(np.arctan2(trunk[1, 0], trunk[0, 0])) == pytest.approx(120.0)


def test_the_leg_follows_the_trunk(walker):
    scene = sc.build(station(walker, joints={"hip": 45.0}).scene_config())
    before = inverse(in_arm_frame(scene, "walker/trunk")) @ in_arm_frame(scene, "walker/shin")
    scene.set_passive_base("walker", odometry(0.2, 0.0, 30.0))
    after = inverse(in_arm_frame(scene, "walker/trunk")) @ in_arm_frame(scene, "walker/shin")

    assert np.allclose(after, before)


def test_a_base_body_below_the_top_of_the_model_is_refused(walker):
    cfg = Workspace(robot=BOT).scene_config()
    spec = dataclasses.replace(get_spec(walker), base_body="thigh")
    cfg.passive = [sc.PassiveRobot("walker", spec, np.eye(4))]
    scene = sc.build(cfg)

    with pytest.raises(ValueError, match="'thigh' nie jest cialem najwyzszego poziomu modelu"):
        scene.set_passive_base("walker", np.eye(4))


def test_the_copy_follows_the_reported_base_and_the_twin_gives_it_in_the_arm_frame(walker, walkerbus):
    twin = Twin(station(walker, backend="walkerbus"))
    try:
        twin.start_passive()
        wait_for(lambda: bool(walkerbus.BUSES))
        walkerbus.BUSES[-1].base = (odometry(0.2, 0.0, 30.0), 12.5)
        wait_for(lambda: twin.other_base_pose("walker") is not None)
        latest = twin.other_base_pose("walker")
        with twin.lock:
            trunk = in_arm_frame(twin.scene, "walker/trunk")
    finally:
        twin.close()

    assert np.allclose(trunk[:3, 3], AHEAD)
    assert latest is not None
    assert np.allclose(latest[0], trunk)
    assert latest[1] == 12.5


def test_a_move_under_a_millimetre_and_a_fifth_of_a_degree_is_not_applied(walker):
    twin = Twin(station(walker))
    try:
        twin.set_passive_joints("walker", {}, (odometry(0.2, 0.0, 30.0), 1.0))
        twin.set_passive_joints("walker", {}, (odometry(0.2009, 0.0, 30.19), 2.0))
        with twin.lock:
            trunk = in_arm_frame(twin.scene, "walker/trunk")
        latest = twin.other_base_pose("walker")
    finally:
        twin.close()

    assert np.allclose(trunk[:3, 3], AHEAD)
    assert latest is not None
    assert latest[1] == 2.0                                       # the latest is kept all the same


def test_the_reader_polls_faster_once_a_base_pose_came():
    plugin = SimpleNamespace(connect=lambda: None, disconnect=lambda: None, read_joints=dict,
                             base_pose=lambda: None)
    reader = PassiveReader("walker", lambda: ReadOnlyArm(plugin, "walkerbus"), lambda joints, base: None)
    slow = reader.read_once()
    plugin.base_pose = lambda: (np.eye(4), 1.0)
    fast = reader.read_once()

    assert slow == pytest.approx(0.1)
    assert fast == pytest.approx(1 / 30)


def test_the_checker_sees_the_moved_base(walker):
    ws = Workspace(robot=BOT)
    ws.other_robots.append(RobotRecord("walker", walker, xyz=[0.0, 0.6, 0.0]))    # on the table, 60 cm left
    scene = sc.build(ws.scene_config())
    checker = CollisionChecker(scene)
    home = get_spec(BOT).home
    assert checker.passive_gap(home) is None

    scene.set_passive_base("walker", odometry(0.12, -0.6, 0.0))                    # its shin through our hand

    assert checker.passive_gap(home) is not None


def test_an_other_robot_without_a_base_pose_stays_where_its_model_puts_it(walker):
    twin = Twin(station(walker))
    try:
        with twin.lock:
            before = in_arm_frame(twin.scene, "walker/trunk")
        twin.set_passive_joints("walker", {"hip": 30.0, "knee": 0.0})
        with twin.lock:
            after = in_arm_frame(twin.scene, "walker/trunk")
        latest = twin.other_base_pose("walker")
    finally:
        twin.close()

    assert np.allclose(after, before)
    assert latest is None
