"""The wrist camera on the hand in the twin's simulated scene: pose, render convention, jitter."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.grasp.camera import project  # noqa: E402
from lerobot_mp.twin.grasp.contract import (  # noqa: E402
    CX,
    CY,
    FOCAL_PX,
    FOVY_DEG,
    IMG_H,
    IMG_W,
    T_GRIPPER_CAMERA,
)
from lerobot_mp.twin.grasp import evaluate  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics, pose  # noqa: E402
from lerobot_mp.twin.robots import A1X  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

FIX = Path(os.environ.get("GRASP_FIXTURES", Path(__file__).parent / "fixtures" / "grasp"))

WRIST = sc.PREFIX + "wrist"
#: A second pose besides home: the arm turned and bent, the wrist camera still looking forward and down.
REACH = {"arm_joint1": 60.0, "arm_joint2": 80.0, "arm_joint3": -100.0, "arm_joint4": 40.0,
         "arm_joint5": 10.0, "arm_joint6": -30.0, "gripper": 80.0}


def _wrist(width: int = IMG_W, height: int = IMG_H, T: np.ndarray = T_GRIPPER_CAMERA,
           fovy: float = FOVY_DEG) -> sc.HandCamera:
    return sc.HandCamera("wrist", T, fovy, width, height)


def _world_camera() -> sc.CameraView:
    """In front of the arm, looking back at it and 30 degrees down."""
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    R = np.array([[0.0, s, -c], [1.0, 0.0, 0.0], [0.0, -c, -s]])
    return sc.CameraView.from_fov("front", 320, 240, 50.0, pose(R, np.array([0.60, 0.0, 0.45])))


def _cube() -> sc.Box:
    return sc.Box("cube", size=(0.015, 0.015, 0.015), xy=(0.25, 0.0))


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics(A1X)


def _angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return math.acos(min(1.0, max(-1.0, (np.trace(Ra.T @ Rb) - 1.0) / 2.0)))


def _exp(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    q, R = np.zeros(4), np.zeros(9)
    mujoco.mju_axisAngle2Quat(q, np.asarray(rotvec, float) / angle, angle)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def test_the_hand_camera_sits_on_the_gripper_link_where_the_kinematics_put_it(kin):
    scene = sc.build(sc.SceneConfig(A1X, hand_cameras=[_wrist()]))
    m = scene.model
    c = m.camera(WRIST).id
    assert m.body(int(m.cam_bodyid[c])).name == sc.PREFIX + "gripper_link"
    # MuJoCo takes fovy in degrees whatever the compiler's angle setting (the scene compiles in radians).
    assert m.cam_fovy[c] == pytest.approx(FOVY_DEG, abs=1e-9)

    rng = np.random.default_rng(7)
    for _ in range(20):
        joints = {n: math.degrees(rng.uniform(kin.lo[k], kin.hi[k]))
                  for k, n in enumerate(A1X.joints) if n != A1X.gripper}
        joints[A1X.gripper] = 50.0
        scene.set_joints(joints)
        T_model = scene.camera_pose(WRIST)
        T_kin = kin.body("gripper_link", joints) @ T_GRIPPER_CAMERA
        assert np.abs(T_model[:3, 3] - T_kin[:3, 3]).max() < 1e-6, joints
        assert _angle(T_model[:3, :3], T_kin[:3, :3]) < 1e-6, joints


@pytest.mark.render
def test_after_a_physics_step_the_hand_camera_renders_from_the_measured_joints(kin):
    """`Scene.step` reruns only the body kinematics after `mj_step`; the render must not use the older
    camera pose."""
    scene = sc.build(sc.SceneConfig(A1X, hand_cameras=[_wrist()]))
    try:
        scene.set_joints(A1X.home)
        scene.command(REACH)
        scene.step(0.1)
        joints = scene.joints()
        T_kin = kin.body("gripper_link", joints) @ T_GRIPPER_CAMERA
        assert np.abs(scene.camera_pose(WRIST)[:3, 3] - T_kin[:3, 3]).max() > 1e-4    # the arm is moving
        scene.render_hand("wrist")
        T_model = scene.camera_pose(WRIST)
        assert np.abs(T_model[:3, 3] - T_kin[:3, 3]).max() < 1e-6
        assert _angle(T_model[:3, :3], T_kin[:3, :3]) < 1e-6
    finally:
        scene.close()


@pytest.mark.render
def test_a_dot_renders_where_the_projection_puts_it(kin):
    """A 6 mm box at known camera-frame points: its pixels' centroid against `project` at 640x480.

    No twin and no stepping: the free joint's qpos is written and the kinematics rerun. The small
    table keeps the dot in free space whatever the arm's pose.
    """
    cfg = sc.SceneConfig(A1X, table=sc.Table(size=(0.12, 0.12), base_xy=(0.0, 0.0)),
                         objects=[sc.Box("dot", size=(0.003, 0.003, 0.003), xy=(0.5, 0.5))],
                         hand_cameras=[_wrist(4 * IMG_W, 4 * IMG_H)])
    scene = sc.build(cfg)
    m, d = scene.model, scene.data
    body = m.body("dot").id
    adr = int(m.jnt_qposadr[m.body_jntadr[body]])
    dot = m.geom("dot").id
    f, cx, cy = 4 * FOCAL_PX, 4 * (CX + 0.5) - 0.5, 4 * (CY + 0.5) - 0.5
    assert (cx, cy) == (319.5, 239.5)

    points = [(0.0, 0.0, 0.20), (0.03, -0.03, 0.15), (-0.04, -0.02, 0.25), (0.05, 0.02, 0.30),
              (-0.02, -0.06, 0.22), (0.06, -0.05, 0.28)]
    try:
        for joints in (A1X.home, REACH):
            scene.set_joints(joints)
            T_hand = kin.body("gripper_link", joints)
            T_cam = T_hand @ T_GRIPPER_CAMERA
            for p_cam in points:
                p_base = (T_cam @ np.r_[p_cam, 1.0])[:3]
                p_world = scene.T_base2world @ np.r_[p_base, 1.0]
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (scene.T_base2world[:3, :3] @ T_cam[:3, :3]).ravel())
                d.qpos[adr:adr + 3] = p_world[:3]
                d.qpos[adr + 3:adr + 7] = quat
                mujoco.mj_forward(m, d)

                seg = scene.render_hand("wrist", segmentation=True)
                assert seg.dtype == np.int32 and seg.shape == (4 * IMG_H, 4 * IMG_W)
                assert (seg == -1).any()
                ys, xs = np.nonzero(seg == dot)
                side = f * 0.006 / p_cam[2]
                assert len(xs) > 0.8 * side * side, (p_cam, len(xs))       # whole, not hidden
                assert xs.min() > 0 and ys.min() > 0 and xs.max() < 4 * IMG_W - 1 and ys.max() < 4 * IMG_H - 1
                uv = project(T_hand, p_base, f=f, cx=cx, cy=cy)
                err = np.hypot(xs.mean() - uv[0], ys.mean() - uv[1])
                assert err <= 1.2, (joints, p_cam, err)
    finally:
        scene.close()


@pytest.mark.render
def test_a_position_jitter_moves_the_image_and_the_next_render_is_nominal_again():
    scene = sc.build(sc.SceneConfig(A1X, objects=[_cube()], hand_cameras=[_wrist()]))
    try:
        scene.set_joints(A1X.home)
        c = scene.model.camera(WRIST).id
        m = scene.model
        pos, quat, fovy = m.cam_pos[c].copy(), m.cam_quat[c].copy(), float(m.cam_fovy[c])
        T_before = scene.camera_pose(WRIST)

        first = scene.render_hand("wrist")
        assert first.shape == (IMG_H, IMG_W, 3) and first.dtype == np.uint8
        shift = sc.HandCameraJitter(dpos=np.array([0.010, 0.0, 0.0]), rotvec=np.zeros(3))
        moved = scene.render_hand("wrist", shift)
        again = scene.render_hand("wrist")

        assert (moved != first).any(axis=2).sum() > 200
        assert np.array_equal(again, first)
        assert np.array_equal(m.cam_pos[c], pos)
        assert np.array_equal(m.cam_quat[c], quat)
        assert m.cam_fovy[c] == fovy
        assert np.array_equal(scene.camera_pose(WRIST), T_before)
    finally:
        scene.close()


@pytest.mark.render
def test_a_jitter_renders_like_a_camera_mounted_at_the_jittered_pose():
    """Position in the hand frame, rotation as R_mj @ exp(rotvec) in the camera frame, fovy added."""
    dpos, rotvec, dfovy = np.array([0.003, -0.002, 0.001]), np.radians([0.7, -0.5, 0.3]), 0.3
    T = T_GRIPPER_CAMERA.copy()
    T[:3, 3] += dpos
    T[:3, :3] = T_GRIPPER_CAMERA[:3, :3] @ sc.CV_TO_MJ @ _exp(rotvec) @ sc.CV_TO_MJ
    nominal = sc.build(sc.SceneConfig(A1X, objects=[_cube()], hand_cameras=[_wrist()]))
    mounted = sc.build(sc.SceneConfig(A1X, objects=[_cube()],
                                      hand_cameras=[_wrist(T=T, fovy=FOVY_DEG + dfovy)]))
    try:
        for s in (nominal, mounted):
            s.set_joints(A1X.home)
        jittered = nominal.render_hand("wrist", sc.HandCameraJitter(dpos, rotvec, dfovy))
        expected = mounted.render_hand("wrist")
        assert (jittered != nominal.render_hand("wrist")).any(axis=2).sum() > 200
        differ = (jittered != expected).any(axis=2).sum()
        assert differ <= 0.001 * IMG_W * IMG_H, differ
    finally:
        nominal.close()
        mounted.close()


def test_a_scene_without_hand_cameras_is_unchanged():
    """The hand camera adds a camera and nothing else: no body, geometry, mass or collision."""
    plain = sc.build(sc.SceneConfig(A1X, cameras=[_world_camera()], objects=[_cube()]))
    wrist = sc.build(sc.SceneConfig(A1X, cameras=[_world_camera()], objects=[_cube()],
                                    hand_cameras=[_wrist()]))
    assert sc.SceneConfig(A1X).hand_cameras == []
    a, b = plain.model, wrist.model
    assert (a.ncam, b.ncam) == (1, 2)
    for n in ("nbody", "ngeom", "nq", "nv", "nu", "nsite", "nsensor", "nlight"):
        assert getattr(a, n) == getattr(b, n), n
    for n in ("body_mass", "geom_contype", "geom_conaffinity", "geom_size"):
        assert np.array_equal(getattr(a, n), getattr(b, n)), n
    assert (a.vis.global_.offwidth, a.vis.global_.offheight) == (640, 480)
    assert (b.vis.global_.offwidth, b.vis.global_.offheight) == (640, 480)
    assert np.array_equal(a.cam_pos[0], b.cam_pos[b.camera("front").id])

    big = sc.build(sc.SceneConfig(A1X, hand_cameras=[_wrist(1280, 960)]))
    assert (big.model.vis.global_.offwidth, big.model.vis.global_.offheight) == (1280, 960)


@pytest.mark.render
def test_world_cameras_render_the_same_with_a_hand_camera_in_the_scene():
    plain = sc.build(sc.SceneConfig(A1X, cameras=[_world_camera()], objects=[_cube()]))
    wrist = sc.build(sc.SceneConfig(A1X, cameras=[_world_camera()], objects=[_cube()],
                                    hand_cameras=[_wrist()]))
    try:
        for s in (plain, wrist):
            s.set_joints(A1X.home)
        a, b = plain.render("front"), wrist.render("front")
        # Two scenes built the same way, neither with a hand camera, differ now and then by
        # 9 of 76800 pixels (measured: 22 of 80 pairs), so exact equality would flake.
        differ = (a != b).any(axis=2).sum()
        assert differ <= 0.001 * a.shape[0] * a.shape[1], differ
    finally:
        plain.close()
        wrist.close()


def test_an_unknown_hand_camera_raises():
    scene = sc.build(sc.SceneConfig(A1X, hand_cameras=[_wrist()]))
    assert scene.hand_camera("wrist").fovy_deg == FOVY_DEG
    with pytest.raises(KeyError):
        scene.hand_camera("elbow")
    with pytest.raises(KeyError):
        scene.render_hand("elbow")
    with pytest.raises(KeyError):
        scene.camera("wrist")                     # the fixed-camera lookup does not see it
    bare = sc.build(sc.SceneConfig(A1X))
    with pytest.raises(KeyError):
        bare.render_hand("wrist")


@pytest.mark.render
def test_the_sim_evaluation_scene_looks_like_trainings_at_the_recorded_joints():
    """Training's own wrist images (`shard.npz`) against the evaluation scene at the same joints.

    The shard scenes hold other objects and a dog the twin does not have, so the check is on the
    median pixel, which the table decides. Measured: 5 levels (worst frame 10) with training's
    table colour and lights, 57 with the twin's own look, where the policy's first action
    differed from training's by up to 0.8.
    """
    shard = np.load(FIX / "shard.npz")
    extras = evaluate.sim_extras()
    extras["objects"] = [sc.Box("cube", size=(0.015, 0.015, 0.015), xy=(-0.5, -0.5))]   # out of view
    scene = sc.build(Workspace(robot="a1x").scene_config(**extras))
    try:
        worst = 0.0
        for q, gap, img in zip(shard["q"], shard["gap"], shard["img"]):
            joints = {f"arm_joint{k + 1}": math.degrees(v) for k, v in enumerate(q)}
            joints["gripper"] = (gap - 0.003) / 0.001
            scene.set_joints(joints)
            d = np.abs(scene.render_hand("wrist").astype(int) - img.astype(int)).max(axis=2)
            worst = max(worst, float(np.median(d)))
        assert worst <= 15.0, worst
    finally:
        scene.close()
