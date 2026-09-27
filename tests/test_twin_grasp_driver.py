"""`GraspDriver` on the MuJoCo twin (threaded loop), with scripted policies and the simulated wrist camera.

One twin for the whole file, configured once like the sim evaluation (training's table, a 3 cm
cube with jaw contact sensors, the hand camera). Each test parks the arm at its hover first, so
pre-positioning is short, and places the cube where it wants it.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields, replace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")
pytest.importorskip("cv2")

import mujoco  # noqa: E402

from lerobot_mp.control.safety import SafetyState  # noqa: E402
from lerobot_mp.robot.base import GripperState  # noqa: E402
from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin.grasp import driver as driver_mod  # noqa: E402
from lerobot_mp.twin.grasp import motion  # noqa: E402
from lerobot_mp.twin.grasp.camera import Frame, SimWristSource, WristCamera  # noqa: E402
from lerobot_mp.twin.grasp.contract import (  # noqa: E402
    FLOOR_MARGIN,
    FOVY_DEG,
    LIFT,
    OBJECT_HEIGHT_DEFAULT,
    OWNER,
    SHADOW_CLEAR,
    T_GRIPPER_CAMERA,
    TABLE_END_BELOW,
    TOUCH_EXPECTED_Z,
    grasp_R,
)
from lerobot_mp.twin.grasp.driver import GraspDriver, TickRecord  # noqa: E402
from lerobot_mp.twin.grasp.hover import Hover, from_scene  # noqa: E402
from lerobot_mp.twin.grasp.motion import JointHistory, jaw_hold_value  # noqa: E402
from lerobot_mp.twin.grasp.policy import GraspStudentPolicy  # noqa: E402
from lerobot_mp.twin.grasp.rig import Rig  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics  # noqa: E402
from lerobot_mp.twin.runtime import TickSample  # noqa: E402
from lerobot_mp.twin.scene import Box, HandCamera, Table  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

ARM = tuple(f"arm_joint{i}" for i in range(1, 7))
CUBE_AWAY = (-0.30, -0.55)                     # on the table, out of the arm's way
FAST = 60.0                                    # deg/s, pre-positioning in tests that do not test it


# ------------------------------------------------------------------ scripted policy
class Scripted(GraspStudentPolicy):
    """`act` from a script of the tick number (and the observation); `inputs` is the real one."""

    def __init__(self, script: Callable[[int, np.ndarray], list[float]]) -> None:
        self.script = script
        self.n = 0

    def act(self, image: np.ndarray, obs: np.ndarray, uv: np.ndarray) -> np.ndarray:
        assert image.shape == (120, 160, 3) and image.dtype == np.uint8
        a = np.asarray(self.script(self.n, obs), np.float32)
        self.n += 1
        return a


def segments(*parts: tuple[int, list[float]]) -> Callable[[int, np.ndarray], list[float]]:
    """A script of (ticks, action) segments; the last action repeats."""
    def script(n: int, obs: np.ndarray) -> list[float]:
        for ticks, action in parts:
            if n < ticks:
                return action
            n -= ticks
        return parts[-1][1]
    return script


STILL = [0.0, 0.0, 0.0, 0.0, 1.0]
DOWN = [0.0, 0.0, -1.0, 0.0, 1.0]


# ------------------------------------------------------------------ the twin
def sim_extras() -> dict:
    return dict(table=Table(size=(1.2, 1.2), base_xy=(0.0, -0.20), base_yaw=0.0),
                objects=[Box("cube", size=(0.015, 0.015, 0.015), xy=CUBE_AWAY)], grasp_sensors=["cube"],
                hand_cameras=[HandCamera("wrist", T_GRIPPER_CAMERA, FOVY_DEG, 160, 120)])


@pytest.fixture(scope="module")
def twin() -> Iterator[rt.Twin]:
    tw = rt.Twin(Workspace(robot="a1x"))
    tw.connect("sim", threaded=True)
    tw.configure(**sim_extras())
    yield tw
    tw.close()


@pytest.fixture(scope="module")
def kin(twin: rt.Twin) -> RobotKinematics:
    return RobotKinematics(twin.spec)


@pytest.fixture(scope="module")
def frames(twin: rt.Twin) -> SimWristSource:
    src = SimWristSource(twin)
    src.latest()                               # the first render creates the renderer
    return src


_HOVERS: dict[tuple[float, float, float], Hover] = {}


def hover_at(twin: rt.Twin, kin: RobotKinematics, xy: tuple[float, float], yaw: float = 0.0) -> Hover:
    """Solved once per place (two full IK solves each); the twin's limits do not change in this file."""
    key = (float(xy[0]), float(xy[1]), float(yaw))
    if key not in _HOVERS:
        _HOVERS[key] = from_scene(kin, np.array(xy), yaw, touch_z=TOUCH_EXPECTED_Z,
                                  limits=twin.joint_limits())
    return _HOVERS[key]


def place_cube(twin: rt.Twin, xy: tuple[float, float], yaw: float = 0.0, settle_s: float = 0.2) -> float:
    """Writes the cube's free joint (the scene is never rebuilt); returns its settled world z."""
    with twin.lock:
        s = twin.scene
        m, d = s.model, s.data
        body = m.body("cube").id
        adr, dof = m.jnt_qposadr[m.body_jntadr[body]], m.jnt_dofadr[m.body_jntadr[body]]
        p = s.T_base2world @ np.array([xy[0], xy[1], 0.015, 1.0])
        d.qpos[adr:adr + 7] = [p[0], p[1], p[2], math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(m, d)
    time.sleep(settle_s)
    return cube_z(twin)


def cube_z(twin: rt.Twin) -> float:
    with twin.lock:
        return float(twin.scene.data.xpos[twin.scene.model.body("cube").id][2])


def cube_rests_at(twin: rt.Twin, xy: tuple[float, float]) -> bool:
    """The cube lies still on the table within 2 mm of `xy` (base frame)."""
    with twin.lock:
        s = twin.scene
        body = s.model.body("cube").id
        p = np.linalg.inv(s.T_base2world) @ np.append(s.data.xpos[body], 1.0)
        speed = float(np.linalg.norm(s.data.cvel[body]))
    return bool(np.hypot(p[0] - xy[0], p[1] - xy[1]) < 0.002 and abs(p[2] - 0.015) < 0.001 and speed < 1e-3)


def scene_success(twin: rt.Twin, z0: float) -> Callable[[], bool]:
    """Training's success: both jaws on the cube and the cube more than LIFT above its start."""
    def ok() -> bool:
        with twin.lock:
            d = twin.scene.data
            jaws = [float(d.sensor(f"cube_jaw{k}").data[0]) for k in (0, 1)]
        return all(j > 0.0 for j in jaws) and cube_z(twin) > z0 + LIFT
    return ok


def ready(twin: rt.Twin, hover: Hover, cube: tuple[float, float] = CUBE_AWAY) -> float:
    """Park the arm at the hover with the jaw open (unguarded, test owner), then place the cube."""
    if twin.safety_state == SafetyState.ESTOP:
        twin.clear_estop()
    twin.set_speed(None)
    m = twin.status.measured
    if max(abs(m[j] - hover.joints[j]) for j in ARM) > 0.5 or m["gripper"] < 95.0:
        for attempt in range(2):
            try:
                twin.move({**hover.joints, "gripper": 100.0}, 0.8, settle=0.1, owner="test")
                break
            except RuntimeError:
                if attempt:
                    raise
            finally:
                twin.release("test")
    return cube_z(twin) if cube_rests_at(twin, cube) else place_cube(twin, cube)


SPIED = ("set_target", "set_speed", "hold_measured")


@contextmanager
def spying(twin: rt.Twin) -> Iterator[dict[str, list]]:
    """Records every `set_target`, `set_speed` and `hold_measured` of the twin; `order` has their names."""
    calls: dict[str, list] = {name: [] for name in (*SPIED, "order")}
    originals = {name: getattr(twin, name) for name in SPIED}

    def wrap(name: str):
        def spy(*args, **kwargs):
            calls[name].append((args, kwargs))
            calls["order"].append(name)
            return originals[name](*args, **kwargs)
        return spy

    for name in SPIED:
        setattr(twin, name, wrap(name))
    try:
        yield calls
    finally:
        for name in SPIED:
            delattr(twin, name)


def targets(calls: dict[str, list]) -> list[dict[str, float]]:
    return [dict(args[0]) for args, _ in calls["set_target"]]


def make(twin: rt.Twin, script, *, frames=None, mode: str = "sim", **kw) -> GraspDriver:
    kw.setdefault("gap", "scene")
    frames = frames if frames is not None else SimWristSource(twin)
    return GraspDriver(twin, Scripted(script), frames, WristCamera(), Rig.for_sim(), mode=mode,
                       real_ok=mode in ("shadow", "real"), **kw)


def start(drv: GraspDriver, hover: Hover, speed: float = FAST) -> None:
    assert drv.check() is None, drv.detail
    assert drv.take() is None, drv.detail
    assert drv.preposition(hover, speed) is None, drv.detail


def tcp_of(kin: RobotKinematics, joints: dict[str, float]) -> np.ndarray:
    return kin.tcp({j: joints[j] for j in ARM})[:3, 3]


def assert_seven_finite(tg: list[dict[str, float]], spec_joints) -> None:
    for t in tg:
        assert set(t) == set(spec_joints), t
        assert all(math.isfinite(v) for v in t.values()), t


# ------------------------------------------------------------------ pieces without a twin
def test_load_workspace_gives_an_in_memory_a1x_without_a_file_and_refuses_another_robot(tmp_path,
                                                                                         monkeypatch):
    monkeypatch.setattr(motion, "default_path", lambda: tmp_path / "none" / "twin.json")
    ws = motion.load_workspace(None)
    assert ws is not None and ws.robot == "a1x" and ws.path is None
    assert not (tmp_path / "none").exists()
    Workspace(robot="so101").save(tmp_path / "so101.json")
    Workspace(robot="a1x").save(tmp_path / "a1x.json")
    assert motion.load_workspace(tmp_path / "so101.json") is None
    assert motion.load_workspace(tmp_path / "missing.json") is None      # an empty workspace is an SO-101
    assert motion.load_workspace(tmp_path / "a1x.json").robot == "a1x"


def test_enable_refusal_only_when_connecting_would_send_the_enable_sequence(tmp_path, monkeypatch):
    ws = Workspace(robot="a1x")
    assert motion.enable_refusal(ws, "sim", None) is None
    assert motion.enable_refusal(ws, "a1x", "can0") is None
    cfg = tmp_path / "config.yaml"
    cfg.write_text("a1x:\n  enable_on_connect: true\n", encoding="utf-8")
    monkeypatch.setenv("LEROBOT_MP_CONFIG", str(cfg))
    assert motion.enable_refusal(ws, "a1x", "can0") == "enable on connect"
    assert motion.enable_refusal(ws, "sim", None) is None


def test_jaw_hold_value_keeps_the_grip_after_a_lift_and_stops_a_moving_jaw_otherwise():
    gs = GripperState(commanded=0.0, target=0.0, streamed=63.1, position=None, frozen=False, effort=None,
                      t=0.0, age_s=0.0, rate=76.9, force_threshold=1.2, measured=False)
    assert jaw_hold_value(None, keep_grip=False) is None
    assert jaw_hold_value(replace(gs, measured=True, position=40.0), keep_grip=True) is None
    assert jaw_hold_value(gs, keep_grip=True) == 0.0
    assert jaw_hold_value(gs, keep_grip=False) == 63.1
    assert jaw_hold_value(replace(gs, streamed=None), keep_grip=False) == 0.0


def test_joint_history_keeps_fresh_samples_and_pairs_by_the_nearest_time():
    def sample(t: float, j1: float, fresh: bool = True) -> TickSample:
        return TickSample(t, {}, {"arm_joint1": j1, "gripper": 50.0}, t, fresh)

    h = JointHistory(keep_s=1.0)
    assert h.newest() is None and h.nearest(0.0) is None and h.speed_deg_s() == {}
    for t, j1, fresh in ((0.0, 0.0, True), (0.02, 9.0, False), (0.04, 2.0, True), (0.08, 4.0, True)):
        h.add(sample(t, j1, fresh))
    assert h.newest().t == 0.08
    assert h.nearest(0.05).t == 0.04 and h.nearest(-1.0).t == 0.0
    assert h.speed_deg_s()["arm_joint1"] == pytest.approx(50.0)
    h.add(sample(1.5, 5.0))
    assert h.nearest(0.0).t == 1.5


# ------------------------------------------------------------------ endings
DESCEND_CLOSE_LIFT = segments((10, DOWN), (16, [0.0, 0.0, -1.0, 0.0, -1.0]), (20, [0.0, 0.0, 1.0, 0.0, -1.0]))


def test_scripted_descend_close_lift_ends_success_with_the_cube_lifted_and_one_owner_throughout(
        twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    z0 = ready(twin, hover, cube=(0.0, 0.33))
    owners: list[str | None] = []
    refused: list[bool] = []

    def on_tick(rec: TickRecord) -> None:
        owners.append(twin.owner)
        try:
            twin.claim("panel")
            refused.append(False)
        except RuntimeError:
            refused.append(True)

    with spying(twin) as calls:
        drv = make(twin, DESCEND_CLOSE_LIFT, frames=frames, success=scene_success(twin, z0), on_tick=on_tick)
        try:
            start(drv, hover)
            assert twin.owner == OWNER
            with pytest.raises(RuntimeError):
                twin.claim("panel")
            rep = drv.episode(hover)
            assert twin.owner == OWNER
        finally:
            drv.close()
    assert rep.ending == "success", rep
    assert cube_z(twin) > z0 + LIFT
    assert owners and set(owners) == {OWNER} and all(refused)
    assert twin.owner is None
    assert_seven_finite(targets(calls), twin.spec.joints)


def test_the_same_script_without_a_success_callable_ends_lifted(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover, cube=(0.0, 0.33))
    drv = make(twin, DESCEND_CLOSE_LIFT, frames=frames)
    try:
        start(drv, hover)
        rep = drv.episode(hover)
    finally:
        drv.close()
    assert rep.ending == "lifted", rep
    assert rep.attempts == 1


def test_a_nan_action_ends_non_finite_and_nothing_non_finite_reaches_set_target(twin, kin, frames, tmp_path):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    script = segments((4, DOWN), (1, [math.nan, 0.0, 0.0, 0.0, 1.0]), (10, STILL))
    with spying(twin) as calls:
        drv = make(twin, script, frames=frames, log_dir=tmp_path / "log")
        try:
            start(drv, hover)
            rep = drv.episode(hover)
        finally:
            drv.close()
    assert rep.ending == "non-finite", rep
    assert rep.ticks == 4                              # the NaN tick ended before any bookkeeping
    assert_seven_finite(targets(calls), twin.spec.joints)
    # The episode log: numbers only, one JSON line per tick, the ending's tick included.
    assert rep.log == tmp_path / "log" / "episode_1.jsonl"
    lines = rep.log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    names = [f.name for f in fields(TickRecord)]
    recs = [json.loads(line, parse_constant=lambda c: pytest.fail(f"{c} in the log")) for line in lines]
    for k, rec in enumerate(recs):
        assert list(rec) == names and rec["tick"] == k
    assert [rec["kind"] for rec in recs] == ["policy"] * 4 + ["skipped"]
    assert recs[-1]["skip"] == "non-finite" and recs[-1]["action"][0] is None
    assert recs[-1]["p_cmd"] == recs[-2]["p_cmd"]


class StaleFrames:
    """Frames stamped 0.2 s ago: old enough to skip, not old enough to end at once."""

    def __init__(self, twin: rt.Twin) -> None:
        self.src = SimWristSource(twin)

    def latest(self) -> Frame:
        f = self.src.latest()
        return Frame(f.image, time.monotonic() - 0.2, f.seq)

    def close(self) -> None:
        pass


def test_stale_frames_skip_then_end_camera(twin, kin):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    recs: list[TickRecord] = []
    drv = make(twin, segments((10, STILL)), frames=StaleFrames(twin), on_tick=recs.append)
    try:
        start(drv, hover)
        rep = drv.episode(hover)
    finally:
        drv.close()
    assert rep.ending == "camera", rep
    assert rep.skipped == 5 and rep.ticks == 5
    assert [r.skip for r in recs] == ["camera"] * 5
    assert all(r.frame_age_s is not None and 0.12 < r.frame_age_s < 0.5 for r in recs)


def test_the_tick_cap_ends_the_episode_and_the_hold_is_the_twins_braking_pose_not_the_last_command(
        twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    j6 = "arm_joint6"
    # J6 = J1 + 90 deg - yaw: turn away from the nearer J6 limit, at full speed.
    turn = -1.0 if hover.joints[j6] < 0.0 else 1.0
    samples: deque[tuple[float, float]] = deque()
    handle = twin.add_tick_listener(lambda s: samples.append((s.t, s.measured[j6])) if s.fresh else None)
    with spying(twin) as calls:
        drv = make(twin, segments((20, [0.0, 0.0, 0.0, turn, 1.0])), frames=frames, max_ticks=8,
                   on_tick=lambda rec: calls["order"].append("record"))
        try:
            start(drv, hover)
            rep = drv.episode(hover)
            t_end = time.monotonic()
            m_end = samples[-1][1]
            hold_target = dict(twin._target)
            time.sleep(0.5)
        finally:
            drv.close()
            twin.remove_tick_listener(handle)
    assert rep.ending == "tick cap", rep
    assert rep.ticks == 8
    moving = -turn                                     # sign of J6's motion
    after = [v for t, v in samples if t >= t_end]
    assert after
    assert min(moving * (v - m_end) for v in after) > -0.5
    # One hold, with no pose: the twin's braking pose, between the reading and the last command.
    assert len(calls["hold_measured"]) == 1
    args, kwargs = calls["hold_measured"][0]
    assert args == () and kwargs.get("measured") is None
    last_cmd = targets(calls)[-1]
    assert moving * (last_cmd[j6] - m_end) > 1.0       # the command was well ahead of the joint
    assert moving * (hold_target[j6] - m_end) > -0.5
    assert moving * (hold_target[j6] - last_cmd[j6]) < 1e-6
    # The hold comes before the last record is written and handed to `on_tick`.
    order = calls["order"]
    k_hold = order.index("hold_measured")
    assert order.count("record") == 8 and order[k_hold + 1:].count("record") == 1
    assert "set_target" not in order[k_hold:]


def test_stop_mid_episode_ends_stop_and_the_arm_stands(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    ticks = threading.Event()
    n = [0]

    def on_tick(rec: TickRecord) -> None:
        n[0] += 1
        if n[0] == 8:
            ticks.set()

    script = segments((6, [1.0, 0.0, -1.0, 0.0, 1.0]), (40, [-1.0, 0.0, 0.0, 0.0, 1.0]))
    drv = make(twin, script, frames=frames, on_tick=on_tick)
    out: dict = {}
    try:
        start(drv, hover)
        worker = threading.Thread(target=lambda: out.setdefault("rep", drv.episode(hover)))
        worker.start()
        assert ticks.wait(5.0)
        motion.stop_now(twin, "test: STOP")
        worker.join(5.0)
        assert not worker.is_alive()
        time.sleep(0.5)                                # braking
        a = dict(twin.status.measured)
        time.sleep(0.5)
        b = dict(twin.status.measured)
    finally:
        drv.close()
        twin.clear_estop()
    assert out["rep"].ending == "stop", out["rep"]
    assert max(abs(a[j] - b[j]) for j in ARM) < 0.5


def test_after_an_impact_ending_the_target_is_the_measured_pose(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    drv = make(twin, segments((30, [1.0, 1.0, -1.0, 0.0, 1.0])), frames=frames)
    try:
        start(drv, hover)
        # Any motion of joint 1 now counts as an impact; unloaded at rest, it keeps the effort rule.
        drv.arm._effort_limits = lambda: {"arm_joint1": 0.05}
        rep = drv.episode(hover)
        target = dict(twin._target)
        measured = dict(twin.status.measured)
    finally:
        drv.close()
    assert rep.ending == "impact effort arm_joint1", rep
    assert max(abs(target[j] - measured[j]) for j in ARM) < 2.0


def test_a_stop_that_comes_while_a_tick_computes_sends_no_target_from_that_tick(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    at_stop: dict[str, int] = {}

    def script(n: int, obs: np.ndarray) -> list[float]:
        if n == 3:
            drv.stop("stop during inference")
            at_stop["targets"] = len(calls["set_target"])
        return [1.0, 1.0, -1.0, 1.0, 1.0]

    with spying(twin) as calls:
        drv = make(twin, script, frames=frames)
        try:
            start(drv, hover)
            rep = drv.episode(hover)
        finally:
            drv.close()
    assert rep.ending == "stop during inference", rep
    assert len(calls["set_target"]) == at_stop["targets"]
    assert len(calls["hold_measured"]) == 1


def test_an_episode_whose_set_up_fails_ends_error_and_holds(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    drv = make(twin, segments((1, STILL)), frames=frames)
    try:
        start(drv, hover)
        drv._floor_of = lambda *a, **k: 1.0            # above the ceiling: the bookkeeping refuses it
        with spying(twin) as calls:
            rep = drv.episode(hover)
        assert rep.ending == "error" and "ValueError" in rep.detail, rep
        assert calls["set_target"] == [] and len(calls["hold_measured"]) == 1
        assert twin._speed_cap == 20.0 and twin.owner == OWNER
    finally:
        drv.close()


def test_close_during_an_episode_on_another_thread_lets_it_hold_once_then_releases(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    ticks = threading.Event()
    n = [0]

    def on_tick(rec: TickRecord) -> None:
        n[0] += 1
        if n[0] == 5:
            ticks.set()

    out: dict = {}
    with spying(twin) as calls:
        drv = make(twin, segments((200, [0.3, 0.3, -0.3, 0.3, 1.0])), frames=frames, on_tick=on_tick)
        start(drv, hover)
        worker = threading.Thread(target=lambda: out.setdefault("rep", drv.episode(hover)))
        worker.start()
        try:
            assert ticks.wait(5.0)
            holds = len(calls["hold_measured"])
            drv.close()
        finally:
            worker.join(5.0)
    assert not worker.is_alive()
    assert out["rep"].ending == "stop", out["rep"]
    assert len(calls["hold_measured"]) == holds + 1
    last_hold = max(k for k, name in enumerate(calls["order"]) if name == "hold_measured")
    assert "set_target" not in calls["order"][last_hold:]
    assert twin.owner is None


def test_close_during_pre_positioning_on_another_thread_lets_it_hold_once_then_releases(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover_at(twin, kin, (0.15, 0.30)))    # far enough for a ramp of a few seconds at 20 deg/s
    out: dict = {}
    with spying(twin) as calls:
        drv = make(twin, segments((1, STILL)), frames=frames)
        assert drv.check() is None and drv.take() is None
        worker = threading.Thread(target=lambda: out.setdefault("reason", drv.preposition(hover, 20.0)))
        worker.start()
        try:
            t_end = time.monotonic() + 5.0
            while len(calls["set_target"]) < 10 and time.monotonic() < t_end:
                time.sleep(0.01)
            assert len(calls["set_target"]) >= 10 and calls["hold_measured"] == []
            drv.close()
        finally:
            worker.join(5.0)
    assert not worker.is_alive()
    assert out["reason"] == "stop", drv.detail
    # The ramp's own hold, while it still owned the arm (the sim jaw is measured: no jaw-only hold).
    assert len(calls["hold_measured"]) == 1
    last_hold = max(k for k, name in enumerate(calls["order"]) if name == "hold_measured")
    assert "set_target" not in calls["order"][last_hold:]
    assert twin.owner is None and twin._speed_cap is None


def test_a_pre_positioning_that_ended_after_its_first_target_counts_as_moved(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover_at(twin, kin, (0.15, 0.30)))
    drv = make(twin, segments((1, STILL)), frames=frames)
    arm = drv.arm
    send = arm.command

    def command_once(target) -> None:
        send(target)
        arm.interrupt("test stop")

    try:
        assert drv.check() is None and drv.take() is None
        assert not arm.moved
        arm.command = command_once
        assert drv.preposition(hover, FAST) == "test stop"
        assert arm.moved
    finally:
        drv.close()


# ------------------------------------------------------------------ floor and collision
def finger_geoms(model: mujoco.MjModel, spec) -> set[int]:
    jaws = {model.body("robot/" + b).id for b in spec.jaw_bodies}
    return {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in jaws}


def lowest_tip_z(twin: rt.Twin) -> float:
    """Lowest corner of the two fingertip boxes, base frame (z = 0 on the table)."""
    s = twin.scene
    m, d = s.model, s.data
    top = float(s.T_base2world[2, 3])
    zs = []
    for name in twin.spec.fingertips:
        g = m.geom("robot/" + name).id
        c, R, h = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        zs += [float((c + R @ (np.array(sg) * h))[2]) - top for sg in itertools.product((-1, 1), repeat=3)]
    return min(zs)


class ContactWatch:
    """Loop-thread listener: every contact between a finger and `other` geoms, and the lowest fingertip."""

    def __init__(self, twin: rt.Twin, other: str) -> None:
        self.twin = twin
        m = twin.scene.model
        self.fingers = finger_geoms(m, twin.spec)
        if other == "table":
            self.other = {m.geom("table").id}
        else:
            body = m.body("robot/" + other).id
            self.other = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body}
        self.contacts = 0
        self.seen: list[str] = []
        self.lowest = math.inf
        self.handle = twin.add_tick_listener(self)

    def __call__(self, sample) -> None:
        with self.twin.lock:
            m, d = self.twin.scene.model, self.twin.scene.data
            for c in d.contact[: d.ncon]:
                pair = {int(c.geom1), int(c.geom2)}
                if pair & self.fingers and pair & self.other:
                    self.contacts += 1
                    finger = min(pair & self.fingers)
                    low = lowest_tip_z(self.twin) * 1000
                    self.seen.append(f"{m.body(m.geom_bodyid[finger]).name}/{finger} at depth "
                                     f"{-c.dist * 1000:.2f} mm, lowest tip {low:.2f} mm")
            self.lowest = min(self.lowest, lowest_tip_z(self.twin))

    def close(self) -> None:
        self.twin.remove_tick_listener(self.handle)


# Down to the floor at the centre, then along the box edges through all four corners, pressing down all along.
CORNERS = segments((10, DOWN), (11, [1.0, 0.0, -1.0, 0.0, 1.0]), (11, [0.0, 1.0, -1.0, 0.0, 1.0]),
                   (21, [0.0, -1.0, -1.0, 0.0, 1.0]), (21, [-1.0, 0.0, -1.0, 0.0, 1.0]),
                   (21, [0.0, 1.0, -1.0, 0.0, 1.0]), (5, DOWN))


@pytest.mark.parametrize("xy", [(0.0, 0.33), (0.15, 0.30)])
def test_walking_to_the_four_box_corners_at_the_bench_floor_never_touches_the_table(twin, kin, frames, xy):
    hover = hover_at(twin, kin, xy)
    ready(twin, hover)
    recs: list[TickRecord] = []
    drv = make(twin, CORNERS, frames=frames, on_tick=recs.append)
    watch = ContactWatch(twin, "table")
    try:
        start(drv, hover)
        rep = drv.episode(hover)
    finally:
        watch.close()
        drv.close()
    assert rep.ending == "tick cap", rep
    assert rep.box_hits > 0 and rep.floor_hits > 0
    corners = {(round(r.p_cmd[0] - hover.p[0], 3), round(r.p_cmd[1] - hover.p[1], 3)) for r in recs}
    assert {(0.08, 0.08), (0.08, -0.08), (-0.08, -0.08), (-0.08, 0.08)} <= corners
    assert watch.contacts == 0, f"{watch.contacts} finger-table contacts: {watch.seen[:4]}"
    assert watch.lowest > 0.0, f"lowest fingertip {watch.lowest * 1000:.1f} mm"
    touch = Rig.for_sim().touch_tcp_z()
    assert min(r.p_cmd[2] for r in recs) >= touch + FLOOR_MARGIN - 1e-9


def test_a_scripted_press_below_the_touch_height_ends_table(twin, kin, frames, monkeypatch):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    recs: list[TickRecord] = []
    drv = make(twin, segments((60, DOWN)), frames=frames, on_tick=recs.append)
    touch = Rig.for_sim().touch_tcp_z()
    # Both floors missed: the command side 15 mm under the taught touch, the measured side never raising.
    drv._floor_of = lambda *a, **k: touch - 0.015
    monkeypatch.setattr(driver_mod, "MEASURED_FLOOR_START", -1.0)
    watch = ContactWatch(twin, "table")
    try:
        start(drv, hover)
        rep = drv.episode(hover)
    finally:
        watch.close()
        drv.close()
    assert rep.ending == "table", rep
    assert recs[-1].kind == "skipped" and recs[-1].skip == "table"      # the ending's tick is logged
    # In sim the touch height is TIP_AHEAD: the measured tool came within 0.5 mm of the table.
    assert watch.lowest < TABLE_END_BELOW + 0.0005, watch.lowest


def test_the_measured_floor_sees_a_tilted_open_finger_and_rises_by_the_deficit_from_the_floor(twin, kin):
    hover = hover_at(twin, kin, (0.0, 0.33))
    drv = make(twin, segments((1, STILL)))
    touch = Rig.for_sim().touch_tcp_z()
    try:
        level = dict(hover.joints, gripper=100.0)
        z = float(kin.tcp(level)[2, 3])
        assert drv._tool_low(level, z) == pytest.approx(z, abs=1e-4)      # pointing down: the TCP itself
        rolled = dict(level, arm_joint5=level["arm_joint5"] + 5.0)
        z = float(kin.tcp(rolled)[2, 3])
        assert drv._tool_low(rolled, z) < z - 0.004                        # an open finger rolled 5 deg
        assert drv._tool_low(rolled, z, 0.0) > z - 0.001                   # a closed one hardly moves
        # After a dive the command is already rising while the tool still overshoots by 1 mm.
        r = drv._new_run(hover, touch, 0.005, 0.0)
        floor0 = r.cmd.floor_z
        r.cmd.state.p_cmd[2] = floor0 + 0.020
        drv._measured_floor(r, np.array([0.0, 0.33, touch + 0.003]), touch + 0.003, dry=False)
        assert r.cmd.floor_z == pytest.approx(floor0 + 0.001, abs=1e-9) and r.floor_raised == 1
    finally:
        drv.close()


def test_shadow_keeps_the_jaw_open_and_the_command_above_the_shadow_floor(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover, cube=(0.0, 0.33))
    recs: list[TickRecord] = []
    touch = Rig.for_sim().touch_tcp_z()
    with spying(twin) as calls:
        drv = make(twin, DESCEND_CLOSE_LIFT, frames=frames, mode="shadow", max_ticks=40, on_tick=recs.append)
        try:
            start(drv, hover)
            n0 = len(calls["set_target"])
            rep = drv.episode(hover)
        finally:
            drv.close()
    assert rep.ending == "tick cap", rep
    assert all(t["gripper"] == 100.0 for t in targets(calls)[n0:])
    assert min(r.g_cmd for r in recs) < 0.4                   # the policy did close, the jaw did not
    assert [r.gap for r in recs if r.kind == "policy"][-1] < 0.02    # the virtual jaw closed
    for r in recs:
        assert r.p_cmd[2] >= touch + OBJECT_HEIGHT_DEFAULT + SHADOW_CLEAR + r.sag - 1e-9, r


def test_a_hover_at_radius_0_22_m_driving_towards_the_base_refuses_by_collision(twin, kin, frames):
    # Bearing 45 deg (joint 1 at the envelope's edge), closing axis at 60 deg: the box corner nearest
    # the base puts an open finger into the base.
    b = math.radians(45.0)
    hover = hover_at(twin, kin, (0.22 * math.cos(b), 0.22 * math.sin(b)), math.radians(60.0))
    ready(twin, hover)
    watch = ContactWatch(twin, "base_link")
    drv = make(twin, segments((40, [-1.0, -1.0, -0.3, 0.0, 1.0])), frames=frames, max_ticks=30)
    try:
        start(drv, hover)
        rep = drv.episode(hover)
    finally:
        watch.close()
        drv.close()
    assert rep.collision_refused > 0, rep
    assert watch.contacts == 0


def test_config_clear_refuses_nothing_at_the_bench_floor_for_hovers_at_radius_0_24_m_and_more(twin, kin):
    drv = make(twin, segments((1, STILL)))
    arm = drv.arm
    touch = Rig.for_sim().touch_tcp_z()
    floor = touch + FLOOR_MARGIN
    refused, checked = [], 0
    try:
        for radius, bearing in itertools.product((0.24, 0.30, 0.37, 0.44), (45.0, 70.0, 90.0, 110.0, 135.0)):
            b = math.radians(bearing)
            xy = (radius * math.cos(b), radius * math.sin(b))
            for yaw in (0.0, math.pi / 4):
                hover = hover_at(twin, kin, xy, yaw)
                R = grasp_R(hover.yaw)
                for cx, cy in ((0, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)):
                    goal = hover.p + np.array([0.08 * cx, 0.08 * cy, floor - hover.p[2]])
                    q = hover.q.copy()
                    # Walk in steps no longer than the policy's, as the driver would, accepting like it does.
                    n = int(math.ceil(np.max(np.abs(goal - hover.p)) / 0.008))
                    for k in range(1, n + 1):
                        p = hover.p + (goal - hover.p) * k / n
                        q_new, step, why = arm.accept(p, R, q)
                        checked += 1
                        if why == "collision":
                            refused.append((xy, yaw, (cx, cy), round(p[2], 4)))
                        if q_new is not None:
                            q = q_new
    finally:
        drv.close()
    assert checked > 1000
    assert refused == [], refused[:10]


# ------------------------------------------------------------------ pre-positioning
def test_after_an_episode_that_ended_at_the_floor_preposition_retreats_straight_up_with_the_jaw_unchanged(
        twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    script = segments((12, [0.3, 0.2, -1.0, 0.0, 1.0]), (20, [0.0, 0.0, -1.0, 0.0, -0.4]))
    drv = make(twin, script, frames=frames, max_ticks=24)
    path: list[np.ndarray] = []
    try:
        start(drv, hover)
        rep = drv.episode(hover)
        assert rep.ending == "tick cap", rep
        handle = twin.add_tick_listener(lambda s: path.append(tcp_of(kin, s.measured)) if s.fresh else None)
        with spying(twin) as calls:
            reason = drv.preposition(hover)
        twin.remove_tick_listener(handle)
    finally:
        drv.close()
    assert reason is None, drv.detail
    p0 = path[0]
    assert p0[2] < hover.p[2] - 0.05                  # it did start low
    risen = next(k for k, p in enumerate(path) if p[2] >= p0[2] + 0.05)
    assert max(np.linalg.norm(p[:2] - p0[:2]) for p in path[:risen + 1]) < 0.005
    grips = [t["gripper"] for t in targets(calls)]
    first_open = grips.index(99.0)
    assert len(set(grips[:first_open])) == 1 and grips[0] < 60.0
    assert set(grips[first_open:]) == {99.0, 100.0} and grips[-1] == 100.0


def test_a_low_tool_pitched_a_few_degrees_rises_straight_up_without_squaring_it_first(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    low = {j: hover.joints[j] for j in ARM}
    for _ in range(15):
        step = kin.ik_step(hover.p - np.array([0.0, 0.0, 0.03]), grasp_R(hover.yaw), low,
                           limits=twin.joint_limits())
        low = {j: step.joints[j] for j in ARM}
    # Pitched 6 deg by joint 4: squaring it in the first 5 mm step would take about 10 deg from one joint.
    low["arm_joint4"] -= 6.0
    try:
        twin.move({**low, "gripper": 100.0}, 0.6, settle=0.2, owner="test")
    finally:
        twin.release("test")
    T = kin.tcp({j: twin.status.measured[j] for j in ARM})
    tilt = math.degrees(math.acos(-float((T[:3, :3] @ np.asarray(twin.spec.tcp_approach, float))[2])))
    assert 2.0 < tilt < 10.0 and T[2, 3] < hover.p[2] - 0.02
    path: list[np.ndarray] = []
    handle = twin.add_tick_listener(lambda s: path.append(tcp_of(kin, s.measured)) if s.fresh else None)
    drv = make(twin, segments((1, STILL)), frames=frames)
    try:
        start(drv, hover)
    finally:
        twin.remove_tick_listener(handle)
        drv.close()
    p0 = path[0]
    risen = next(k for k, p in enumerate(path) if p[2] >= p0[2] + 0.02)
    assert max(np.linalg.norm(p[:2] - p0[:2]) for p in path[:risen + 1]) < 0.005



def test_a_straight_up_retreat_refused_well_above_the_object_hands_over_to_the_staged_plan(
        twin, kin, frames, caplog):
    # At 0.45 m reach with joint 4 pitched 6 deg the warm solver loses the vertical line near
    # z 0.09 m (joint 4 at its limit), under the hover height but well above the object.
    hover = hover_at(twin, kin, (0.0, 0.45))
    ready(twin, hover)
    low = {j: hover.joints[j] for j in ARM}
    for _ in range(15):
        step = kin.ik_step(hover.p - np.array([0.0, 0.0, 0.055]), grasp_R(hover.yaw), low,
                           limits=twin.joint_limits())
        low = {j: step.joints[j] for j in ARM}
    low["arm_joint4"] += 6.0
    try:
        twin.move({**low, "gripper": 100.0}, 0.6, settle=0.2, owner="test")
    finally:
        twin.release("test")
    path: list[np.ndarray] = []
    handle = twin.add_tick_listener(lambda s: path.append(tcp_of(kin, s.measured)) if s.fresh else None)
    drv = make(twin, segments((1, STILL)), frames=frames)
    try:
        with caplog.at_level(logging.WARNING, logger="lerobot_mp.twin.grasp.motion"):
            start(drv, hover)
    finally:
        twin.remove_tick_listener(handle)
        drv.close()
    assert any("the staged plan goes on from" in r.getMessage() for r in caplog.records)
    p0 = path[0]
    assert p0[2] < hover.p[2] - 0.04
    risen = next(k for k, p in enumerate(path) if p[2] >= p0[2] + 0.03)
    assert max(np.linalg.norm(p[:2] - p0[:2]) for p in path[:risen + 1]) < 0.005
    assert np.linalg.norm(tcp_of(kin, twin.status.measured) - hover.p) < 0.01

def test_ctrl_c_during_pre_positioning_holds_and_goes_on_up(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    drv = make(twin, segments((1, STILL)), frames=frames)

    def ctrl_c(*args, **kwargs) -> None:
        raise KeyboardInterrupt

    try:
        assert drv.check() is None and drv.take() is None
        drv.arm._ramp = ctrl_c
        with spying(twin) as calls:
            with pytest.raises(KeyboardInterrupt):
                drv.preposition(hover, FAST)
        assert len(calls["hold_measured"]) == 1 and calls["set_target"] == []
        assert twin._speed_cap == 20.0
    finally:
        drv.close()


def test_arrival_is_not_judged_on_an_old_joint_sample(twin, kin):
    hover = hover_at(twin, kin, (0.0, 0.33))
    offset = [0.0]
    arm = motion.ArmSession(twin, kin, Rig.for_sim(), clock=lambda: time.monotonic() + offset[0])
    arm.listen()
    try:
        time.sleep(0.1)
        assert not any(m.startswith("joint feedback") for m in arm._misses(hover))
        offset[0] = 1.0                                # as if the listener had stopped a second ago
        misses = arm._misses(hover)
        assert len(misses) == 1 and misses[0].startswith("joint feedback"), misses
    finally:
        arm.close()


def test_a_panel_speed_cap_is_cleared_for_the_loop_restored_by_close_and_20_after_every_ending(
        twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    twin.set_speed(35.0)
    caps: list[float | None] = []
    drv = make(twin, segments((10, STILL)), frames=frames, max_ticks=6,
               on_tick=lambda r: caps.append(twin._speed_cap))
    try:
        start(drv, hover, 20.0)
        assert twin._speed_cap == 20.0
        rep = drv.episode(hover)
        assert rep.ending == "tick cap"
        # 90 while the ticks run; the last record reaches `on_tick` after the ending's hold, at 20.
        assert len(caps) == 6 and all(c == 90.0 for c in caps[:-1]) and caps[-1] == 20.0
        assert twin._speed_cap == 20.0
        assert drv.preposition(hover, 20.0) is None
        drv.stop("test ending")
        rep = drv.episode(hover)
        assert rep.ending == "test ending"
        assert twin._speed_cap == 20.0
    finally:
        drv.close()
    assert twin._speed_cap == 35.0
    twin.set_speed(None)


def test_a_supervisor_limit_under_90_refuses_in_take_before_any_target(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    cfg = twin._supervisor.cfg
    with spying(twin) as calls:
        drv = make(twin, segments((1, STILL)), frames=frames)
        cfg.safety.velocity_scale = 0.5
        try:
            assert drv.check() is None
            assert drv.take() == "speed limited"
        finally:
            cfg.safety.velocity_scale = 1.0
            drv.close()
            twin.set_speed(None)
    assert calls["set_target"] == []
    assert twin.owner is None


def test_a_joint_loaded_at_rest_is_listed_in_no_effort_rule_and_the_move_runs(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    drv = make(twin, segments((10, STILL)), frames=frames, max_ticks=4)
    arm = drv.arm
    at_rest = arm._efforts()
    loaded = max(ARM, key=lambda j: abs(at_rest[j]))
    limits = {j: 1000.0 for j in ARM}
    limits[loaded] = 1.5 * abs(at_rest[loaded])
    arm._effort_limits = lambda: dict(limits)
    try:
        start(drv, hover)
        assert arm.no_effort_rule == [loaded]
        rep = drv.episode(hover)
    finally:
        drv.close()
    assert rep.ending == "tick cap", rep
    assert rep.no_effort_rule == [loaded]


# ------------------------------------------------------------------ pairing, IK, dry
class LateFrames:
    """Sim frames delivered and stamped `lag` s after their exposure (like a hub stamping after decode)."""

    def __init__(self, twin: rt.Twin, lag: float) -> None:
        self.src = SimWristSource(twin)
        self.lag = lag
        self.queue: deque[Frame] = deque()
        self.out: Frame | None = None

    def latest(self) -> Frame | None:
        self.queue.append(self.src.latest())
        now = time.monotonic()
        while self.queue and self.queue[0].t + self.lag <= now:
            f = self.queue.popleft()
            self.out = Frame(f.image, f.t + self.lag, f.seq)
        return self.out

    def close(self) -> None:
        pass


def test_pairing_uses_the_joint_sample_nearest_the_exposure_time(twin, kin):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    samples: list[tuple[float, dict[str, float]]] = []
    handle = twin.add_tick_listener(
        lambda s: samples.append((s.measured_t, dict(s.measured))) if s.fresh else None)
    recs: list[TickRecord] = []
    drv = make(twin, segments((8, [1.0, 0.0, -1.0, 0.0, 1.0]), (12, [-1.0, -1.0, 0.0, 0.0, 1.0])),
               frames=LateFrames(twin, 0.08), camera_latency_s=0.08, max_ticks=24, on_tick=recs.append)
    try:
        start(drv, hover)
        rep = drv.episode(hover)
    finally:
        drv.close()
        twin.remove_tick_listener(handle)
    assert rep.ending == "tick cap", rep
    cam = WristCamera()
    times = np.array([t for t, _ in samples])
    scored = [r for r in recs if r.kind == "policy"]
    assert len(scored) >= 15
    differs = 0
    for r in scored:
        assert r.t_exp == pytest.approx(r.frame_t - 0.08, abs=1e-12)
        k = int(np.argmin(np.abs(times - r.t_exp)))
        uv = cam.believed_uv(kin, samples[k][1], hover.believed)
        np.testing.assert_allclose(r.uv, uv, atol=1e-4)
        newest = samples[int(np.searchsorted(times, r.t, side="right")) - 1][1]
        differs += float(np.max(np.abs(cam.believed_uv(kin, newest, hover.believed) - uv))) > 0.5
    assert differs > 0                                   # the arm moved: the stamp alone would pair wrongly


def test_an_ik_refusal_holds_the_command(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    j6 = "arm_joint6"
    # Towards the nearer J6 limit until the step no longer converges.
    turn = 1.0 if hover.joints[j6] < 0.0 else -1.0
    recs: list[TickRecord] = []

    def on_tick(rec: TickRecord) -> None:
        recs.append(rec)
        if sum(r.ik_why == "ik" for r in recs) >= 4:
            drv.stop("enough refusals")

    with spying(twin) as calls:
        drv = make(twin, segments((80, [0.0, 0.0, 0.0, turn, 1.0])), frames=frames, max_ticks=80,
                   on_tick=on_tick)
        try:
            start(drv, hover)
            n0 = len(calls["set_target"])
            rep = drv.episode(hover)
        finally:
            drv.close()
    assert rep.ending == "enough refusals", rep
    assert rep.ik_refused >= 4, rep
    assert rep.jump_refused == sum(r.ik_why == "jump" for r in recs)
    scored = [r for r in recs if r.kind == "policy"]
    for prev, cur in zip(scored, scored[1:]):
        if cur.ik_why:
            assert cur.p_cmd == prev.p_cmd and cur.yaw_cmd == prev.yaw_cmd
    sent = targets(calls)[n0:]
    limit = twin.joint_limits()[j6]
    assert all(limit[0] <= t[j6] <= limit[1] for t in sent)


def test_the_collision_question_assumes_the_widest_jaw_opening_that_can_be_true(twin, kin):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)                                 # the jaw stands open
    drv = make(twin, segments((1, STILL)))
    arm = drv.arm
    try:
        arm._last_grip = 0.0                           # a close was commanded; the fingers are still open
        q, step, why = arm.accept(hover.p, grasp_R(hover.yaw), hover.q)
        assert why == "" and q is not None
        assert step.joints["gripper"] > 95.0
        _, step, _ = arm.accept(hover.p, grasp_R(hover.yaw), hover.q, collision=False)
        assert step.joints["gripper"] == 0.0
    finally:
        drv.close()


def test_dry_mode_sends_nothing_and_dry_tick_returns_the_scored_frame_and_inputs(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    with spying(twin) as calls:
        drv = make(twin, segments((1, DOWN)), frames=frames, mode="dry")
        try:
            assert drv.check() is None
            assert drv.take() == "dry"
            assert drv.preposition(hover) == "dry"
            assert drv.episode(hover).ending == "dry"
            out = []
            for _ in range(6):
                out.append(drv.dry_tick(hover))
                time.sleep(0.05)
            assert twin.owner is None
        finally:
            drv.close()
    assert calls["order"] == []
    rec, inp = out[-1]
    assert rec.kind == "policy", rec
    assert rec.action == pytest.approx(DOWN)
    # Nothing commands the arm in dry mode: no sag is estimated from where it happens to stand.
    touch = Rig.for_sim().touch_tcp_z()
    assert all(r.sag == 0.0 and r.floor_z == pytest.approx(touch + FLOOR_MARGIN, abs=1e-12) for r, _ in out)
    assert inp.frame is not None and inp.frame.image.shape == (120, 160, 3)
    assert inp.inputs is not None and inp.inputs.shape == (4, 120, 160) and inp.inputs.dtype == np.float32
    assert inp.obs is not None and inp.obs.shape == (19,)
    assert set(inp.pair_joints) >= set(ARM)
    heat = inp.inputs[3]
    v, u = np.unravel_index(int(np.argmax(heat)), heat.shape)
    if 0 <= rec.uv[0] < 160 and 0 <= rec.uv[1] < 120:
        assert abs(u - rec.uv[0]) <= 1 and abs(v - rec.uv[1]) <= 1
    assert all(r.tick == k for k, (r, _) in enumerate(out))


def test_tick_time_p99_is_under_40_ms_with_the_fake_policy(twin, kin, frames):
    hover = hover_at(twin, kin, (0.0, 0.33))
    ready(twin, hover)
    wiggle = segments((30, [0.3, 0.3, -0.5, 0.2, 1.0]), (30, [-0.3, -0.3, 0.5, -0.2, 1.0]))
    drv = make(twin, lambda n, obs: wiggle(n % 60, obs), frames=frames, max_ticks=80)
    try:
        start(drv, hover)
        rep = drv.episode(hover)
    finally:
        drv.close()
    assert rep.ending == "tick cap", rep
    print(f"tick ms {rep.tick_ms}, loop gaps {rep.loop_gaps}")
    assert rep.tick_ms["p99"] < 40.0, rep.tick_ms
