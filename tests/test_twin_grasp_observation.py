"""Grasp student's observation path: the 19-vector, the gap estimate and the hover, against training.

The fixtures were recorded from the training environment (`scripts/record_grasp_fixtures.py`).
The vector is compared with the `*_fk` fields, which training's own model gives AT the recorded
joints. Training's recorded `obs` lags its joints by one 2 ms physics substep (positions are
computed before the last substep, `qpos` after it); one test documents that difference.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.robot.base import GripperState  # noqa: E402
from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.collision import CollisionChecker  # noqa: E402
from lerobot_mp.twin.grasp import gripper as gripper_module  # noqa: E402
from lerobot_mp.twin.grasp.command import GraspCommand  # noqa: E402
from lerobot_mp.twin.grasp.contract import (  # noqa: E402
    ARM_BASE_WORLD,
    GAP_CLOSED,
    HOVER,
    HOVER_J5_MAX_DEG,
    HOVER_J6_MARGIN_DEG,
    IK_TOL_POS,
    IK_TOL_ROT,
    JOINT_HI,
    JOINT_LO,
    NOISE_XY,
    NOISE_YAW,
    NOISE_Z,
    STREAM_STROKE_S,
    STREAM_UNITS_S,
    TICK_S,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
    TRAINING_FLOOR_Z,
    grasp_R,
    wrap,
)
from lerobot_mp.twin.grasp.gripper import GripperEstimator  # noqa: E402
from lerobot_mp.twin.grasp.hover import (  # noqa: E402
    HoverError,
    from_rig,
    from_scene,
    solve_hover,
    stage_plan,
    training_envelope,
)
from lerobot_mp.twin.grasp.observer import GraspObserver  # noqa: E402
from lerobot_mp.twin.grasp.rig import Rig  # noqa: E402
from lerobot_mp.twin.kinematics import IKSolution, RobotKinematics  # noqa: E402
from lerobot_mp.twin.robots import A1X  # noqa: E402
from lerobot_mp.twin.runtime import Twin  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

FIX = Path(os.environ.get("GRASP_FIXTURES", Path(__file__).parent / "fixtures" / "grasp"))
RERECORD = "re-record the fixtures (docs/GRASP.md)"
ARM = A1X.arm_joints
LIMITS = A1X.limits
LEFT = np.array([0.0, 0.33, TOUCH_EXPECTED_Z + TIP_CLEAR])     # 0.33 m out on the arm's left, as in training


def fixture(name: str, *keys: str) -> dict[str, np.ndarray]:
    path = FIX / name
    if not path.is_file():
        pytest.fail(f"{path} is missing: {RERECORD}")
    with np.load(path) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            pytest.fail(f"{name} has no {', '.join(missing)}: {RERECORD}")
        return {k: z[k] for k in keys}


def joints_of(q: np.ndarray) -> dict[str, float]:
    return {n: math.degrees(float(v)) for n, v in zip(ARM, q)}


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def rotation_angle(R: np.ndarray) -> float:
    return math.acos(min(1.0, max(-1.0, (float(np.trace(R)) - 1.0) / 2.0)))


def state(*, streamed: float | None = None, position: float | None = None, measured: bool = False,
          commanded: float | None = None) -> GripperState:
    return GripperState(commanded=commanded, target=commanded, streamed=streamed, position=position,
                        frozen=False, effort=None, t=0.0, age_s=0.0, rate=None, force_threshold=None,
                        measured=measured)


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics(A1X)


# ----------------------------------------------------------------------------- the 19-vector

def test_the_vector_rebuilt_from_the_recorded_joints_and_command_equals_trainings(kin):
    z = fixture("shard.npz", "q", "gap", "believed_pos", "believed_yaw", "yaw_cmd", "g_cmd", "prev_action",
                "attempts", "obs", "obs_fk")
    observer = GraspObserver(kin)
    base = np.asarray(ARM_BASE_WORLD)
    got = []
    for i in range(len(z["q"])):
        believed = z["believed_pos"][i] - base
        cmd = GraspCommand(believed + np.array([0.0, 0.0, HOVER]), float(z["believed_yaw"][i]),
                           z["q"][i], floor_z=TRAINING_FLOOR_Z)
        s = cmd.state
        s.yaw_cmd = float(z["yaw_cmd"][i])
        s.g_cmd = float(z["g_cmd"][i])
        s.prev_action = z["prev_action"][i].astype(np.float32)
        s.attempts = int(z["attempts"][i])
        got.append(observer.build(joints_of(z["q"][i]), float(z["gap"][i]), cmd))
    got = np.stack(got)
    assert got.dtype == np.float32 and got.shape == (len(z["q"]), 19)
    err = np.abs(got - z["obs_fk"])
    assert err.max() <= 1e-5, (float(err.max()), np.unravel_index(err.argmax(), err.shape))
    # What the policy was fed in training is within the substep artefact of what the twin builds.
    assert np.abs(got - z["obs"]).max() <= 3e-3


def test_trainings_recorded_vector_lags_its_joints_by_one_substep_in_the_tcp_entries_only():
    z = fixture("shard.npz", "obs", "obs_fk")
    diff = np.abs(z["obs"].astype(float) - z["obs_fk"].astype(float))
    assert diff.max() <= 3e-3
    others = np.r_[0:7, 10:19]
    assert np.array_equal(z["obs"][:, others], z["obs_fk"][:, others])


def test_the_vector_at_the_settled_hover_equals_trainings_first_observation(kin):
    h = fixture("hover.npz", "p_hover", "yaw", "q_hover", "q_measured", "gap0", "obs0", "obs0_fk")
    observer = GraspObserver(kin)
    got = []
    for i in range(len(h["yaw"])):
        cmd = GraspCommand(h["p_hover"][i], float(h["yaw"][i]), h["q_hover"][i], floor_z=TRAINING_FLOOR_Z)
        got.append(observer.build(joints_of(h["q_measured"][i]), float(h["gap0"][i]), cmd))
    got = np.stack(got)
    err = np.abs(got - h["obs0_fk"])
    assert err.max() <= 1e-5, (float(err.max()), np.unravel_index(err.argmax(), err.shape))
    assert np.abs(h["obs0"].astype(float) - h["obs0_fk"].astype(float)).max() <= 3e-3


def test_the_first_tick_tail_is_no_yaw_error_an_open_jaw_and_no_history():
    cmd = GraspCommand(LEFT + np.array([0.0, 0.0, HOVER]), 0.7, np.zeros(6), floor_z=TRAINING_FLOOR_Z)
    want = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert cmd.obs_tail().tolist() == want
    h = fixture("hover.npz", "obs0_fk")
    assert np.array_equal(h["obs0_fk"][:, 10:], np.tile(np.float32(want), (len(h["obs0_fk"]), 1)))


def test_joints_are_normalised_with_the_model_ranges_and_the_vector_is_not_normalised(kin):
    observer = GraspObserver(kin)
    cmd = GraspCommand(LEFT + np.array([0.0, 0.0, HOVER]), 0.0, np.zeros(6), floor_z=TRAINING_FLOOR_Z)
    lo = dict(zip(ARM, np.degrees(JOINT_LO)))
    hi = dict(zip(ARM, np.degrees(JOINT_HI)))
    np.testing.assert_allclose(observer.build(lo, 0.103, cmd)[:6], -1.0, atol=1e-6)
    np.testing.assert_allclose(observer.build(hi, 0.103, cmd)[:6], 1.0, atol=1e-6)

    at_app = {n: LIMITS[n][1] for n in ARM}
    v = observer.build(at_app, 0.103, cmd)
    q = np.radians([at_app[n] for n in ARM])
    want = 2.0 * (q - np.asarray(JOINT_LO)) / (np.asarray(JOINT_HI) - np.asarray(JOINT_LO)) - 1.0
    np.testing.assert_allclose(v[:6], want, atol=1e-6)
    assert np.all(v[:6] < 0.99)                         # the app limit is not the end of the range
    assert v[6] == np.float32(1.03)                     # gap * 10, raw
    tcp = observer.tcp(at_app)
    np.testing.assert_allclose(v[7:10], (cmd.state.believed_pos - tcp) * 10.0, atol=1e-6)
    np.testing.assert_allclose(tcp, kin.tcp(at_app)[:3, 3], atol=1e-12)


def test_a_missing_arm_joint_raises_instead_of_taking_home(kin):
    observer = GraspObserver(kin)
    cmd = GraspCommand(LEFT + np.array([0.0, 0.0, HOVER]), 0.0, np.zeros(6), floor_z=TRAINING_FLOOR_Z)
    joints = {n: 10.0 for n in ARM if n != "arm_joint5"}
    with pytest.raises(KeyError):
        observer.build(joints, 0.1, cmd)
    with pytest.raises(KeyError):
        observer.tcp(joints)


def test_the_observer_refuses_a_model_with_other_joint_ranges():
    other = RobotKinematics(A1X)
    other.lo = other.lo.copy()
    other.lo[0] -= 0.01
    with pytest.raises(ValueError):
        GraspObserver(other)


# ----------------------------------------------------------------------------- gap estimate

def test_the_gap_is_the_opening_mapped_linearly_and_clipped():
    assert GripperEstimator.gap_of(0.0) == GAP_CLOSED
    assert GripperEstimator.gap_of(100.0) == pytest.approx(0.103)
    assert GripperEstimator.gap_of(50.0) == pytest.approx(0.053)
    assert GripperEstimator.gap_of(130.0) == pytest.approx(0.103)
    assert GripperEstimator.gap_of(-5.0) == GAP_CLOSED


def test_the_scene_source_returns_the_measured_position():
    est = GripperEstimator(source="scene")
    est.reset(100.0, 0.0)
    gap = est.update(0.05, 0.0, state(position=37.0, measured=True, streamed=0.0, commanded=0.0))
    assert est.opening == 37.0 and gap == pytest.approx(0.040)
    est.update(0.10, 0.0, state(position=5.0, measured=True, streamed=0.0, commanded=0.0))
    assert est.opening == 5.0                                  # a measurement is not rate limited
    # Without a measurement the scene source falls back to the model.
    est.update(0.15, 0.0, state(streamed=0.0, commanded=0.0))
    assert est.opening == pytest.approx(5.0 - STREAM_UNITS_S * 0.05)


def test_the_model_source_follows_the_streamed_opening_limited_to_the_jaw_speed():
    units = 50.0
    est = GripperEstimator(units_s=units, source="model")
    est.reset(100.0, 0.0)
    for k in range(1, 50):
        t = k * TICK_S
        # The stream is already closed and the command says so; the jaw is slower.
        est.update(t, 0.0, state(streamed=0.0, commanded=0.0))
        assert est.opening == pytest.approx(max(0.0, 100.0 - units * t), abs=1e-9)

    # A stream slower than the jaw is followed exactly.
    est.reset(0.0, 10.0)
    for k in range(1, 40):
        t = 10.0 + k * TICK_S
        streamed = min(100.0, 20.0 * (t - 10.0))
        est.update(t, 100.0, state(streamed=streamed, commanded=100.0))
        assert est.opening == pytest.approx(streamed, abs=1e-9)

    # The command does not lead the jaw: with the stream still open the estimate stays open.
    est.reset(100.0, 20.0)
    est.update(20.05, 0.0, state(streamed=100.0, commanded=0.0))
    assert est.opening == 100.0
    # The model source never reads a position, even a measured one.
    est.update(20.10, 100.0, state(streamed=100.0, position=0.0, measured=True, commanded=100.0))
    assert est.opening == 100.0

    # Skipped ticks still count: the jaw moved all the while. Time going backwards moves nothing.
    est.reset(100.0, 30.0)
    est.update(30.1, 0.0, state(streamed=0.0))
    est.update(30.6, 0.0, state(streamed=0.0))
    assert est.opening == pytest.approx(100.0 - units * 0.6)
    before = est.opening
    est.update(30.5, 0.0, state(streamed=0.0))
    assert est.opening == before


@pytest.mark.parametrize("stroke_s", [STREAM_STROKE_S, 2.0])
def test_the_virtual_jaw_closes_in_the_stroke_time_and_opens_again(stroke_s):
    est = GripperEstimator(units_s=100.0 / stroke_s, source="model")
    est.reset(100.0, 0.0)
    t = 0.0
    while est.opening > 0.0 and t < 5.0:
        t += TICK_S
        gap = est.update(t, 0.0, None)
    assert abs(t - stroke_s) <= TICK_S + 1e-9, t
    assert gap == GAP_CLOSED
    t0 = t
    while est.opening < 100.0 and t < 10.0:
        t += TICK_S
        est.update(t, 100.0, None)
    assert abs((t - t0) - stroke_s) <= TICK_S + 1e-9


def test_a_frozen_stream_stops_the_estimate_where_the_stream_stopped():
    est = GripperEstimator(units_s=100.0 / STREAM_STROKE_S, source="model")
    est.reset(100.0, 0.0)
    t, streamed = 0.0, 100.0
    for _ in range(60):
        t += TICK_S
        streamed = max(63.0, streamed - STREAM_UNITS_S * TICK_S)     # the backend froze the stream at 63
        est.update(t, 0.0, state(streamed=streamed, commanded=0.0))
    assert est.opening == pytest.approx(63.0)


def test_a_jaw_faster_than_the_stream_follows_the_stream():
    est = GripperEstimator(units_s=200.0, source="model")
    est.reset(100.0, 0.0)
    t = 0.0
    while est.opening > 0.0 and t < 5.0:
        t += TICK_S
        est.update(t, 0.0, None)
    assert abs(t - STREAM_STROKE_S) <= TICK_S + 1e-9, t


def test_a_non_finite_command_or_time_raises_and_changes_nothing():
    est = GripperEstimator()
    est.reset(100.0, 0.0)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            est.update(0.05, bad, None)
        with pytest.raises(ValueError):
            est.update(bad, 0.0, None)
        with pytest.raises(ValueError):
            est.reset(bad, 0.0)
        with pytest.raises(ValueError):
            est.reset(100.0, bad)
    assert est.opening == 100.0
    est.update(0.05, 0.0, None)
    assert est.opening == pytest.approx(100.0 - STREAM_UNITS_S * 0.05)


def test_on_the_sim_backend_the_scene_source_gives_the_fingertip_gap_training_measures():
    twin = Twin(Workspace(robot="a1x"))
    twin.connect("sim", threaded=False)
    try:
        m, d = twin.scene.model, twin.scene.data
        f1, f2 = (m.jnt_qposadr[m.joint(sc.PREFIX + n).id] for n in ("gripper", "gripper_finger_joint2"))

        def fingertip_gap() -> float:                   # training's `Arm.gap`
            return 0.003 + abs(float(d.qpos[f1])) + abs(float(d.qpos[f2]))

        twin.set_engaged(True)
        twin.set_target({"gripper": 100.0})
        for _ in range(50):
            twin.step(0.02)
        units = 50.0
        scene, model = GripperEstimator(source="scene"), GripperEstimator(units_s=units, source="model")
        for est in (scene, model):
            est.reset(100.0, twin.clock.now())
        assert scene.update(twin.clock.now(), 100.0, twin.backend.gripper_state()) == pytest.approx(
            fingertip_gap(), abs=1e-6)

        t0 = twin.clock.now()
        twin.set_target({"gripper": 0.0})
        moving = 0.0
        for _ in range(125):
            twin.step(0.02)
            gs = twin.backend.gripper_state()
            gap = scene.update(twin.clock.now(), 0.0, gs)
            model.update(twin.clock.now(), 0.0, gs)
            # The second finger follows the first through a soft equality, so it trails while moving.
            moving = max(moving, abs(gap - fingertip_gap()))
            assert scene.opening == min(max(gs.position, 0.0), 100.0)
            # The scene's jaw command leads the jaw; the model follows it at the jaw speed.
            assert gs.streamed <= model.opening + 1e-9
            assert model.opening == pytest.approx(max(0.0, 100.0 - units * (twin.clock.now() - t0)), abs=1e-9)
        assert moving <= 1e-3, moving
        assert gap == pytest.approx(fingertip_gap(), abs=1e-6) and gap == pytest.approx(GAP_CLOSED, abs=1e-6)
        print(f"scene gap against the fingertip gap: {moving * 1000:.2f} mm while closing")
    finally:
        twin.close()


def test_the_estimator_never_reads_the_twin():
    tree = ast.parse(Path(gripper_module.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
    assert modules <= {"__future__", "logging", "math", "typing", "...robot.base", ".contract"}, modules
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {"status", "twin", "measured_now", "command"}, attributes


# ----------------------------------------------------------------------------- hover

def test_trainings_hovers_solve_from_home_with_j5_at_zero_and_j6_clear_of_its_limits(kin):
    h = fixture("hover.npz", "p_hover", "believed", "yaw")
    lo6, hi6 = LIMITS["arm_joint6"]
    flipped = 0
    t0 = time.perf_counter()
    for i in range(len(h["yaw"])):
        yaw = float(h["yaw"][i])
        hv = solve_hover(kin, h["believed"][i], yaw, limits=LIMITS)
        j = hv.joints
        assert set(j) == set(ARM)
        np.testing.assert_allclose(hv.p, h["believed"][i] + np.array([0.0, 0.0, HOVER]), atol=1e-12)
        np.testing.assert_allclose(hv.p, h["p_hover"][i], atol=1e-9)
        np.testing.assert_allclose(hv.believed, hv.p - np.array([0.0, 0.0, HOVER]), atol=0)
        np.testing.assert_allclose(hv.q, np.radians([j[n] for n in ARM]), atol=0)
        assert abs(j["arm_joint5"]) <= HOVER_J5_MAX_DEG
        margin = min(j["arm_joint6"] - lo6, hi6 - j["arm_joint6"])
        assert margin >= HOVER_J6_MARGIN_DEG and margin == pytest.approx(hv.j6_margin_deg)
        T = kin.tcp(j)
        assert np.linalg.norm(hv.p - T[:3, 3]) < IK_TOL_POS
        assert rotation_angle(grasp_R(hv.yaw) @ T[:3, :3].T) < IK_TOL_ROT
        assert hv.yaw in (wrap(yaw), wrap(yaw + math.pi))
        flipped += hv.yaw != wrap(yaw)
        want_j6 = wrap_deg(j["arm_joint1"] + 90.0 - math.degrees(hv.yaw))
        assert abs(wrap_deg(j["arm_joint6"] - want_j6)) <= 0.05, (i, j, math.degrees(hv.yaw))
        # The other candidate stands 180 deg away on J6; the chosen one is the further from a limit.
        other = wrap_deg(j["arm_joint6"] + 180.0)
        assert hv.j6_margin_deg >= min(other - lo6, hi6 - other) - 0.05
        assert hv.outside_training == [], (i, hv.outside_training)
    print(f"{len(h['yaw'])} hovers, {flipped} solved at yaw + pi, "
          f"{(time.perf_counter() - t0) / len(h['yaw']) * 1000:.0f} ms per hover")


def test_the_taught_yaw_or_its_flip_is_chosen_by_the_j6_margin(kin):
    # J1 is 90 deg on the arm's left; J6 = J1 + 90 - yaw.
    hv = solve_hover(kin, LEFT, math.radians(-60.0), limits=LIMITS)
    assert hv.yaw == pytest.approx(math.radians(120.0))
    assert hv.joints["arm_joint6"] == pytest.approx(60.0, abs=0.5)
    assert hv.j6_margin_deg == pytest.approx(100.0, abs=0.5)

    hv = solve_hover(kin, LEFT, math.radians(150.0), limits=LIMITS)
    assert hv.yaw == pytest.approx(math.radians(150.0))            # the flip would put J6 at -150 deg
    assert hv.joints["arm_joint6"] == pytest.approx(30.0, abs=0.5)
    assert hv.joints["arm_joint1"] == pytest.approx(90.0, abs=0.5)
    assert hv.outside_training == []


def test_the_callers_joint_limits_decide_the_hover_within_the_model_range(kin):
    yaw = math.radians(-60.0)                     # candidates at J6 60 deg and -120 deg
    hv = solve_hover(kin, LEFT, yaw, limits={**LIMITS, "arm_joint6": (-160.0, 40.0)})
    assert hv.yaw == pytest.approx(yaw)
    assert hv.joints["arm_joint6"] == pytest.approx(-120.0, abs=0.5)
    assert hv.j6_margin_deg == pytest.approx(40.0, abs=0.5)

    # A limit wider than the model is cut to the model range.
    hv = solve_hover(kin, LEFT, yaw, limits={**LIMITS, "arm_joint6": (-400.0, 400.0)})
    assert hv.joints["arm_joint6"] == pytest.approx(60.0, abs=0.5)
    assert hv.j6_margin_deg == pytest.approx(math.degrees(JOINT_HI[5]) - 60.0, abs=0.5)

    with pytest.raises(HoverError) as e:
        solve_hover(kin, LEFT, yaw, limits={**LIMITS, "arm_joint6": (-130.0, 70.0)})
    assert e.value.reason == "ik"
    assert e.value.detail.count("zapas J6") == 2, e.value.detail


def test_a_point_in_front_of_the_arm_is_flagged_outside_training(kin):
    hv = solve_hover(kin, np.array([0.33, 0.0, TOUCH_EXPECTED_Z + TIP_CLEAR]), 0.0, limits=LIMITS)
    assert [s for s in hv.outside_training if s.startswith("joint1 ")], hv.outside_training

    assert training_envelope({"arm_joint1": 90.0}, np.array([0.0, 0.33, 0.10])) == []
    assert training_envelope({"arm_joint1": 12.0}, np.array([0.19, 0.0, 0.15])) == [
        "joint1 12.0 st. poza 40..139",
        "promien 0.19 m poza 0.22..0.44",
        "wysokosc zawisu 0.150 m poza 0.05..0.14",
    ]


def test_the_hover_is_judged_by_forward_kinematics_not_by_the_ik_flag(monkeypatch):
    k = RobotKinematics(A1X)

    def lying_ik(position, rotation=None, seed=None, **kw):
        return IKSolution(dict(A1X.home), 0.0, 0.0, True)

    monkeypatch.setattr(k, "ik", lying_ik)
    with pytest.raises(HoverError) as e:
        solve_hover(k, LEFT, 0.3, limits=LIMITS)
    assert e.value.reason == "ik"
    detail = e.value.detail
    assert "yaw 17.2 st." in detail and "yaw -162.8 st." in detail, detail
    assert "pozycja" in detail and "obrot" in detail, detail


def test_the_hover_comes_from_the_taught_point_or_the_scene(kin):
    with pytest.raises(HoverError) as e:
        from_rig(kin, Rig(), limits=LIMITS)
    assert e.value.reason == "grasp point not taught"

    rig = Rig(table={"touch_tcp_z": 0.034}, grasp={"tcp": [0.02, 0.33, 0.035], "yaw": 0.4})
    hv = from_rig(kin, rig, limits=LIMITS)
    np.testing.assert_allclose(hv.believed, [0.02, 0.33, 0.034 + TIP_CLEAR], atol=1e-12)
    assert hv.p[2] == pytest.approx(0.034 + TIP_CLEAR + HOVER)
    assert hv.yaw in (wrap(0.4), wrap(0.4 + math.pi))

    hv = from_scene(kin, np.array([0.0, 0.33]), 0.2, touch_z=TOUCH_EXPECTED_Z, limits=LIMITS)
    np.testing.assert_allclose(hv.believed, LEFT, atol=1e-12)
    assert hv.yaw in (wrap(0.2), wrap(0.2 + math.pi))

    ref = np.random.default_rng(7)
    dxy, dz, dyaw = ref.normal(0.0, NOISE_XY, 2), ref.normal(0.0, NOISE_Z), ref.normal(0.0, NOISE_YAW)
    hv = from_scene(kin, np.array([0.0, 0.33]), 0.2, touch_z=TOUCH_EXPECTED_Z, limits=LIMITS,
                    rng=np.random.default_rng(7))
    np.testing.assert_allclose(hv.believed, LEFT + np.array([dxy[0], dxy[1], dz]), atol=1e-12)
    assert hv.yaw in (wrap(0.2 + dyaw), wrap(0.2 + dyaw + math.pi))


# ----------------------------------------------------------------------------- staged plan

def _pose(**deg: float) -> dict[str, float]:
    return {n: float(deg.get(n, 0.0)) for n in ARM}


def test_the_staged_plan_moves_joint_groups_in_order_then_in_reverse_then_straight():
    start = {**_pose(arm_joint1=0.0, arm_joint2=60.0, arm_joint3=-90.0, arm_joint4=30.0), "gripper": 99.0}
    goal = _pose(arm_joint1=90.0, arm_joint2=120.0, arm_joint3=-80.0, arm_joint4=50.0, arm_joint6=20.0)

    segments = []
    poses, why = stage_plan(A1X, start, goal, lambda a, b: segments.append((dict(a), dict(b))))
    assert why is None
    # (elbow), (wrist), (shoulder); the gripper stage is not an arm stage and the last straight move is empty.
    assert [set(k for k in p if p[k] != a[k]) for a, p in zip([start, *poses], poses)] == [
        {"arm_joint3"}, {"arm_joint4", "arm_joint6"}, {"arm_joint1", "arm_joint2"}]
    assert {k: poses[-1][k] for k in ARM} == goal
    assert all(p["gripper"] == 99.0 for p in poses)
    assert len(segments) == 3

    def no_elbow_first(a, b):
        return "kolizja: test" if a["arm_joint3"] != b["arm_joint3"] and a["arm_joint1"] == 0.0 else None

    poses, why = stage_plan(A1X, start, goal, no_elbow_first)
    assert why is None
    assert {k for k in ARM if poses[0][k] != start[k]} == {"arm_joint1", "arm_joint2"}

    poses, why = stage_plan(A1X, start, goal, lambda a, b: "kolizja: test")
    assert (poses, why) == ([], "kolizja: test")

    straight = dataclasses.replace(A1X, home_stages=())
    poses, why = stage_plan(straight, start, goal, lambda a, b: None)
    assert why is None and len(poses) == 1 and {k: poses[0][k] for k in ARM} == goal


def test_the_staged_plan_from_home_to_a_training_hover_is_clear_on_the_bench_scene(kin):
    checker = CollisionChecker(sc.build(Workspace(robot="a1x").scene_config()))
    hv = solve_hover(kin, LEFT, 0.0, limits=LIMITS)
    start = {n: A1X.home[n] for n in ARM}
    poses, why = stage_plan(A1X, start, hv.joints, checker.path_clear)
    assert why is None, why
    assert poses and {k: poses[-1][k] for k in ARM} == pytest.approx(hv.joints)
