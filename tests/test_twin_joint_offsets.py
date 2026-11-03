"""Servo zero offsets: the kinematics carries them, step 2 finds them, the workstation keeps them with the poses."""

from __future__ import annotations

import importlib.util
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("mujoco")
cv2 = pytest.importorskip("cv2")

from conftest import BOT

from machinekind_twin.calib.card import Card, pinch_point
from machinekind_twin.calib.handeye import CameraFit, Fit, Observation, judge, solve, solve_offsets
from machinekind_twin.calib.session import offset_joints
from machinekind_twin.calib.simulate import look_at
from machinekind_twin.calib.tags import corners_in_tag
from machinekind_twin.kinematics import RobotKinematics
from machinekind_twin.robots import get_spec
from machinekind_twin.workspace import CameraRecord, Workspace


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


#: Arms with a joint whose zero a wave can tell: the test arm's one arm joint sits right before the card.
ROBOTS = [
    pytest.param("arm6", marks=pytest.mark.skipif(not _installed("arm6"), reason="ramie przykladowe instaluje uv sync")),
    pytest.param("so101", marks=pytest.mark.skipif(not _installed("soarm"), reason="SO-101 przychodzi z pakietu soarm")),
]


def test_a_wave_tells_only_zeros_with_a_turning_joint_after_them():
    assert offset_joints(get_spec(BOT)) == []                     # j2 is the last joint before the card
    if _installed("arm6"):
        assert offset_joints(get_spec("arm6")) == ["shoulder", "elbow", "wrist_pitch", "wrist_yaw"]
    if _installed("soarm"):
        assert offset_joints(get_spec("so101")) == ["shoulder_lift", "elbow_flex", "wrist_flex"]


def test_kinematics_reads_the_servo_plus_its_offset_and_commands_the_servo_minus_it():
    spec = get_spec(BOT)
    name = "j2"
    plain, off = RobotKinematics(spec), RobotKinematics(replace(spec, joint_offsets={name: 2.5}))
    joints = {n: 10.0 for n in spec.arm_joints} | {spec.gripper: 40.0}
    q = off.to_q(joints)
    assert q == pytest.approx(plain.to_q(joints | {name: 12.5}))
    assert off.from_q(q) == pytest.approx(joints)                 # the round trip is the servo's own units
    assert off.tcp(joints) == pytest.approx(plain.tcp(joints | {name: 12.5}))


def test_workspace_spec_carries_the_offsets_and_stays_the_same_object():
    """The arm mask rebuilds its kinematics when the description is another object: once per change."""
    ws = Workspace(robot=BOT)
    assert ws.spec() is get_spec(BOT)
    ws.joint_offsets = {"j2": 1.5}
    a = ws.spec()
    assert a.joint_offsets == {"j2": 1.5} and ws.spec() is a
    ws.joint_offsets = {}
    assert ws.spec() is get_spec(BOT)


def _pose(x: float) -> list[list[float]]:
    return [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1]]


def test_saved_offsets_add_up_travel_with_the_poses_and_distrust_poses_fitted_without_them(tmp_path):
    ws = Workspace(robot=BOT, path=tmp_path / "twin.json")
    for n, x in (("front", 0.1), ("side", 0.3)):
        ws.add_camera(CameraRecord(n, "sim", T_cam2base=_pose(x),
                                   calibration={"trusted": True, "time": "2026-10-27T10:00:00"}))
    ws.joint_offsets = {"j2": 1.0}
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.2]
    fit = Fit({"front": CameraFit(T, 0.2, 0.6, 30, 12.5, trusted=True)}, joint_offsets={"j2": 0.5})
    assert ws.apply_fit(fit) == ["front"]
    assert ws.joint_offsets == {"j2": 1.5}                         # on top of the ones the wave ran with
    assert ws.camera("front").calibration["trusted"]
    assert not ws.camera("side").calibration["trusted"]            # fitted with the old zeros
    preset = ws.save_preset().stem
    ws.save()

    again = Workspace.load(tmp_path / "twin.json")
    assert again.joint_offsets == {"j2": 1.5} and again.spec().joint_offsets == {"j2": 1.5}
    assert again.reset_joint_offsets("wyzerowane") == ["front"]
    assert again.joint_offsets == {} and not again.camera("front").calibration["trusted"]
    again.calib_preset = None
    again.load_preset(preset)                                      # the set brings its zeros back
    assert again.joint_offsets == {"j2": 1.5}


def _observations(spec, true: dict[str, float], rng: np.random.Generator, n: int = 60):
    """Tags on the nominal card as two cameras see them on an arm whose servos read `true` less than it stands."""
    real, model = RobotKinematics(replace(spec, joint_offsets=true)), RobotKinematics(spec)
    card = Card()
    nominal = card.nominal(pinch_point(model), spec)
    mounts = {f"tag{t}": nominal @ T for t, T in card.tag_poses().items()}
    K = np.array([[600.0, 0, 319.5], [0, 600.0, 239.5], [0, 0, 1]])
    centre = np.array([0.2, 0.0, 0.1])
    cams = {"c0": look_at(np.array([0.7, 0.35, 0.55]), centre), "c1": look_at(np.array([0.55, -0.45, 0.65]), centre)}
    X = np.c_[corners_in_tag(card.tag_size), np.ones(4)].T
    # The wave also turns the roll to show the card to a camera: through it the wrist's zeros show.
    roll = {}
    if spec.roll_joint:
        k = spec.joints.index(spec.roll_joint)
        roll[spec.roll_joint] = (max(np.degrees(model.lo[k]), -90.0), min(np.degrees(model.hi[k]), 90.0))
    obs = []
    for pose in range(2000):
        joints = dict(spec.home)
        for name, (lo, hi) in (spec.wave_ranges | roll).items():
            joints[name] = float(rng.uniform(lo, hi))
        if spec.gripper:
            joints[spec.gripper] = 0.0
        for cam, T_cam2base in cams.items():
            for mount, M in mounts.items():
                P = (np.linalg.inv(T_cam2base) @ real.tcp(joints) @ M @ X)[:3]
                if (P[2] < 0.1).any():
                    continue
                px = cv2.projectPoints(P.T, np.zeros(3), np.zeros(3), K, None)[0].reshape(-1, 2)
                if not ((px >= 0) & (px < [640, 480])).all():
                    continue
                obs.append(Observation(model.tcp(joints), np.eye(4), px + rng.normal(0, 0.15, px.shape),
                                       int(mount[3:]), mount, cam, pose, joints=dict(joints)))
        if len(obs) >= n:
            break
    return obs, {c: (K, None) for c in cams}, card.tag_size, mounts, model


@pytest.mark.parametrize("spec_name", ROBOTS)
def test_offsets_the_cameras_can_tell_are_found_and_the_fit_comes_back_to_the_noise(spec_name):
    spec = get_spec(spec_name)
    names = offset_joints(spec)
    true = dict(zip(names, (2.0, -1.5, 2.5, 1.0)))
    obs, cams, tag, mounts, model = _observations(spec, true, np.random.default_rng(0))
    plain = judge(solve(obs, cams, tag, mounts))
    fit = solve_offsets(plain, obs, cams, tag, model.tcp, names)
    assert fit.joint_offsets == pytest.approx(true, abs=0.15), f"{spec_name}: {fit.joint_offsets}"
    assert fit.rms_px < 0.3 < plain.rms_px


@pytest.mark.parametrize("spec_name", ROBOTS)
def test_without_offsets_the_fit_finds_none(spec_name):
    spec = get_spec(spec_name)
    names = offset_joints(spec)
    obs, cams, tag, mounts, model = _observations(spec, {}, np.random.default_rng(1))
    fit = solve_offsets(judge(solve(obs, cams, tag, mounts)), obs, cams, tag, model.tcp, names)
    assert max(abs(v) for v in fit.joint_offsets.values()) < 0.1
