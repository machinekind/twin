"""`RobotKinematics.ik_step`: the grasp training solver on the twin's A1X model.

The recorded fixtures come from the training environment (`scripts/record_grasp_fixtures.py`);
the step has to give the same joints and the same `ok` flag as training's `Arm.ik`, because the
grasp student was trained on exactly those joints.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from lerobot_mp.twin.kinematics import IKStep, RobotKinematics  # noqa: E402
from lerobot_mp.twin.robots import A1X  # noqa: E402

FIX = Path(os.environ.get("GRASP_FIXTURES", Path(__file__).parent / "fixtures" / "grasp"))
RERECORD = "re-record the fixtures (docs/GRASP.md)"

ARM = tuple(f"arm_joint{i}" for i in range(1, 7))
# A top-down hover on the arm's left, where training put every object.
HOVER = {"arm_joint1": 90.0, "arm_joint2": 120.0, "arm_joint3": -77.0, "arm_joint4": 47.0,
         "arm_joint5": 0.0, "arm_joint6": 0.0, "gripper": 100.0}


def fixture(name: str, *keys: str) -> dict[str, np.ndarray]:
    path = FIX / name
    if not path.exists():
        pytest.fail(f"{path} is missing: {RERECORD}")
    with np.load(path) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            pytest.fail(f"{path} lacks {', '.join(missing)}: {RERECORD}")
        return {k: z[k] for k in keys}


def arm_base() -> np.ndarray:
    path = FIX / "meta.json"
    if not path.exists():
        pytest.fail(f"{path} is missing: {RERECORD}")
    meta = json.loads(path.read_text())
    if "arm_base" not in meta:
        pytest.fail(f"{path} lacks arm_base: {RERECORD}")
    return np.asarray(meta["arm_base"], float)


# 70 does not survive the to_q/from_q round trip bit for bit, so it shows whether the seed's value is copied.
def joints_of(q: np.ndarray, gripper: float = 70.0) -> dict[str, float]:
    return {**{name: float(np.degrees(v)) for name, v in zip(ARM, q)}, "gripper": gripper}


def arm_q(step: IKStep) -> np.ndarray:
    return np.radians([step.joints[name] for name in ARM])


def fk_errors(kin: RobotKinematics, joints: dict[str, float], position: np.ndarray,
              rotation: np.ndarray) -> tuple[float, float]:
    """Position and rotation error of the TCP at `joints`, the way training measures them."""
    T = kin.tcp(joints)
    want, cur, err, w = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(want, np.ascontiguousarray(rotation, float).ravel())
    mujoco.mju_mat2Quat(cur, np.ascontiguousarray(T[:3, :3]).ravel())
    mujoco.mju_negQuat(cur, cur)
    mujoco.mju_mulQuat(err, want, cur)
    mujoco.mju_quat2Vel(w, err, 1.0)
    return float(np.linalg.norm(np.asarray(position, float) - T[:3, 3])), float(np.linalg.norm(w))


def angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics(A1X)


@pytest.fixture(scope="module")
def book() -> dict[str, np.ndarray]:
    """The 1000 IK calls training made in the bookkeeping recording, positions in the base frame."""
    b = fixture("bookkeeping.npz", "ik_pos", "ik_R", "ik_q0", "ik_q", "ik_ok")
    b["ik_pos"] = b["ik_pos"] - arm_base()
    return b


def test_the_twin_and_training_models_put_the_tcp_and_its_closing_axis_in_the_same_place(kin):
    fk = fixture("fk.npz", "q", "tcp_base", "tcp_R")
    pos_diff, axis_diff = [], []
    for q, p, R in zip(fk["q"], fk["tcp_base"], fk["tcp_R"]):
        T = kin.tcp(joints_of(q, gripper=0.0))
        pos_diff.append(np.linalg.norm(T[:3, 3] - p))
        axis_diff.append(angle(T[:3, 1], R[:, 1]))
    assert len(pos_diff) == 300
    assert max(pos_diff) < 1e-3
    assert np.degrees(max(axis_diff)) < 0.5


def test_ik_step_gives_training_joints_and_flags_on_the_recorded_steps(kin):
    ik = fixture("ik.npz", "q0", "pos", "R", "q", "ok")
    assert len(ik["ok"]) == 200
    assert ik["ok"].any() and not ik["ok"].all()
    for i, (q0, p, R, q, ok) in enumerate(zip(ik["q0"], ik["pos"], ik["R"], ik["q"], ik["ok"])):
        step = kin.ik_step(p, R, joints_of(q0))
        assert step.converged == bool(ok), i
        assert np.max(np.abs(arm_q(step) - q)) < 1e-9, i


def test_ik_step_gives_training_joints_and_flags_on_the_bookkeeping_calls(kin, book):
    n = len(book["ik_ok"])
    assert n == 1000
    for i in range(n):
        seed = joints_of(book["ik_q0"][i])
        step = kin.ik_step(book["ik_pos"][i], book["ik_R"][i], seed)
        assert step.converged == bool(book["ik_ok"][i]), i
        assert np.max(np.abs(arm_q(step) - book["ik_q"][i])) < 1e-9, i
        jump = max(abs(step.joints[name] - seed[name]) for name in ARM)
        assert step.jump_deg == pytest.approx(jump, abs=1e-9)
        assert step.joints["gripper"] == seed["gripper"]


def test_reported_errors_are_forward_kinematics_at_the_returned_joints(kin, book):
    refused = 0
    for i in range(0, len(book["ik_ok"]), 5):
        p, R, seed = book["ik_pos"][i], book["ik_R"][i], joints_of(book["ik_q0"][i])
        # One update leaves most of these steps unconverged, where the errors must be recomputed.
        for iters in (60, 1):
            step = kin.ik_step(p, R, seed, iters=iters)
            refused += not step.converged
            pos_err, rot_err = fk_errors(kin, step.joints, p, R)
            assert step.pos_err == pytest.approx(pos_err, abs=1e-12)
            assert step.rot_err == pytest.approx(rot_err, abs=1e-12)
    assert refused > 50


def test_a_last_update_that_lands_inside_the_tolerances_is_not_converged(kin):
    T = kin.tcp(HOVER)
    target = T[:3, 3] + np.array([0.003, 0.0, 0.0])
    full = kin.ik_step(target, T[:3, :3], HOVER)
    assert full.converged and full.iters == 1           # one update is enough, the seed is not

    step = kin.ik_step(target, T[:3, :3], HOVER, iters=1)
    assert step.iters == 1
    assert not step.converged
    assert step.pos_err < 1e-3 and step.rot_err < 5e-3
    assert step.joints == full.joints
    assert (step.pos_err, step.rot_err) == pytest.approx(fk_errors(kin, step.joints, target, T[:3, :3]),
                                                         abs=1e-12)


def test_one_update_towards_a_far_target_moves_the_arm_joints_by_the_0_3_rad_cap(kin):
    T = kin.tcp(HOVER)
    far = kin.ik_step(T[:3, 3] + np.array([-0.15, 0.10, 0.10]), T[:3, :3], HOVER, iters=1)
    moved = arm_q(far) - np.radians([HOVER[n] for n in ARM])
    assert np.linalg.norm(moved) == pytest.approx(0.3, abs=1e-9)


def test_a_seed_that_already_meets_the_tolerances_takes_no_update(kin):
    T = kin.tcp(HOVER)
    step = kin.ik_step(T[:3, 3], T[:3, :3], HOVER)
    assert step.converged and step.iters == 0
    assert step.jump_deg == 0.0
    assert step.joints == pytest.approx(HOVER, abs=1e-12)

    none = kin.ik_step(T[:3, 3], T[:3, :3], HOVER, iters=0)
    assert not none.converged and none.iters == 0


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_app_limits_keep_j6_within_160_deg_and_refuse_a_step_that_needs_163(kin, sign):
    seed = dict(HOVER, arm_joint6=sign * 158.0)
    T = kin.tcp(dict(seed, arm_joint6=sign * 163.0))
    free = kin.ik_step(T[:3, 3], T[:3, :3], seed)
    assert free.converged and abs(free.joints["arm_joint6"]) > 162.0

    step = kin.ik_step(T[:3, 3], T[:3, :3], seed, limits=A1X.limits)
    assert not step.converged
    assert abs(step.joints["arm_joint6"]) <= 160.0 + 1e-9
    assert step.joints["gripper"] == seed["gripper"]


def test_limits_wider_than_the_model_leave_the_model_ranges_in_force(kin):
    j6 = A1X.joints.index("arm_joint6")
    seed = dict(HOVER, arm_joint6=163.0)
    T = kin.tcp(dict(seed, arm_joint6=168.0))
    step = kin.ik_step(T[:3, 3], T[:3, :3], seed, limits={"arm_joint6": (-200.0, 200.0)})
    assert not step.converged
    assert step.joints["arm_joint6"] <= np.degrees(kin.hi[j6]) + 1e-9


def test_the_jump_counts_from_the_seed_as_given_before_clipping(kin):
    seed = dict(HOVER, arm_joint6=163.0)
    T = kin.tcp(seed)
    clipped = kin.ik_step(T[:3, 3], T[:3, :3], seed, iters=0, limits=A1X.limits)
    assert clipped.joints["arm_joint6"] == pytest.approx(160.0, abs=1e-9)
    assert clipped.jump_deg == pytest.approx(3.0, abs=1e-9)

    step = kin.ik_step(T[:3, 3], T[:3, :3], seed, limits=A1X.limits)
    assert not step.converged
    assert step.joints["arm_joint6"] <= 160.0 + 1e-9
    assert step.jump_deg == pytest.approx(max(abs(step.joints[n] - seed[n]) for n in ARM), abs=1e-9)
    assert step.jump_deg >= 3.0 - 1e-9


def test_app_limits_hold_on_every_bookkeeping_call(kin, book):
    for i in range(len(book["ik_ok"])):
        step = kin.ik_step(book["ik_pos"][i], book["ik_R"][i], joints_of(book["ik_q0"][i]), limits=A1X.limits)
        for name in ARM:
            lo, hi = A1X.limits[name]
            assert lo - 1e-9 <= step.joints[name] <= hi + 1e-9, (i, name)


def test_a_seed_without_every_arm_joint_raises(kin):
    T = kin.tcp(HOVER)
    seed = {k: v for k, v in HOVER.items() if k != "arm_joint3"}
    with pytest.raises(KeyError, match="arm_joint3"):
        kin.ik_step(T[:3, 3], T[:3, :3], seed)

    without_gripper = {k: v for k, v in HOVER.items() if k != "gripper"}
    assert kin.ik_step(T[:3, 3], T[:3, :3], without_gripper).converged


def test_a_moved_and_turned_base_gives_the_same_step_in_the_base_frame(kin):
    ik = fixture("ik.npz", "q0", "pos", "R")
    # Attached under a prefix to a shifted frame, as the twin's scene does; tilted as well as turned,
    # so a solver that handled only the base yaw would show.
    robot = mujoco.MjSpec.from_file(str(A1X.mjcf_path))
    world = mujoco.MjSpec()
    for name in ("timestep", "integrator", "cone", "impratio"):
        setattr(world.option, name, getattr(robot.option, name))
    mount = world.worldbody.add_frame()
    mount.pos = [0.4, -0.3, 0.75]
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.2, 0.1, 1.0]) / np.linalg.norm([0.2, 0.1, 1.0]),
                              np.radians(35.0))
    mount.quat = quat
    world.attach(robot, prefix="robot/", frame=mount)
    placed = RobotKinematics(A1X, world.compile(), prefix="robot/")
    placed.tcp(HOVER)
    assert np.linalg.norm(placed.data.xpos[placed.base_id]) > 0.5

    for i, (q0, p, R) in enumerate(zip(ik["q0"], ik["pos"], ik["R"])):
        seed = joints_of(q0)
        here, there = kin.ik_step(p, R, seed), placed.ik_step(p, R, seed)
        assert there.converged == here.converged, i
        assert np.max(np.abs(arm_q(there) - arm_q(here))) < 1e-9, i
        assert there.pos_err == pytest.approx(here.pos_err, abs=1e-12)
        assert there.rot_err == pytest.approx(here.rot_err, abs=1e-12)


def test_steps_are_fast_and_converge_in_nearly_all_of_the_200_recorded_steps(kin):
    ik = fixture("ik.npz", "q0", "pos", "R")
    times, converged = [], 0
    for q0, p, R in zip(ik["q0"], ik["pos"], ik["R"]):
        seed = joints_of(q0)
        t0 = time.perf_counter()
        step = kin.ik_step(p, R, seed, limits=A1X.limits)
        times.append(time.perf_counter() - t0)
        converged += step.converged
    assert len(times) == 200
    assert converged > 0.95 * len(times)
    assert np.median(times) < 5e-3
