"""The grasp contract constants against the training recording (`meta.json`) and hand values."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lerobot_mp.twin.grasp import contract as c

FIX = Path(os.environ.get("GRASP_FIXTURES", Path(__file__).parent / "fixtures" / "grasp"))
RERECORD = "re-record the fixtures (docs/GRASP.md)"


@pytest.fixture(scope="module")
def meta() -> dict:
    path = FIX / "meta.json"
    if not path.is_file():
        pytest.fail(f"{path} missing: {RERECORD}")
    return json.loads(path.read_text(encoding="utf-8"))


def need(meta: dict, key: str):
    if key not in meta:
        pytest.fail(f"meta.json has no {key!r}: {RERECORD}")
    return meta[key]


def test_camera_intrinsics_equal_the_recording(meta):
    assert abs(c.FOCAL_PX - need(meta, "f_px")) < 1e-12
    assert abs(c.CX - need(meta, "cx")) < 1e-12
    assert abs(c.CY - need(meta, "cy")) < 1e-12
    assert abs(c.FOVY_DEG - need(meta, "fovy_deg")) < 1e-12
    assert list(need(meta, "image_hw")) == [c.IMG_H, c.IMG_W]
    assert abs(c.HEAT_SIGMA_FRAC - need(meta, "sigma_frac")) < 1e-12


def test_camera_pose_in_gripper_link_equals_the_recording(meta):
    R = np.array(need(meta, "cam_R_gripper_link_opencv"))
    pos = np.array(need(meta, "cam_pos_gripper_link"))
    np.testing.assert_allclose(c.T_GRIPPER_CAMERA[:3, :3], R, rtol=0, atol=1e-12)
    np.testing.assert_allclose(c.T_GRIPPER_CAMERA[:3, 3], pos, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(c.T_GRIPPER_CAMERA[3], [0.0, 0.0, 0.0, 1.0])


def test_joint_ranges_equal_the_recording(meta):
    np.testing.assert_allclose(c.JOINT_LO, need(meta, "joint_lo"), rtol=0, atol=1e-4)
    np.testing.assert_allclose(c.JOINT_HI, need(meta, "joint_hi"), rtol=0, atol=1e-4)


def test_steps_box_hover_and_arm_base_equal_the_recording(meta):
    assert abs(c.STEP_XYZ - need(meta, "step_xyz")) < 1e-12
    assert abs(c.STEP_YAW - math.radians(need(meta, "step_yaw_deg"))) < 1e-12
    assert abs(c.BOX_XY - need(meta, "box_xy")) < 1e-12
    assert abs(c.HOVER - need(meta, "hover")) < 1e-12
    np.testing.assert_allclose(c.ARM_BASE_WORLD, need(meta, "arm_base"), rtol=0, atol=1e-12)


def test_training_floor_and_ceiling_equal_the_recording(meta):
    # meta.json states them as formulas: "TABLE_TOP + 0.02 (world z 0.72)" and "p0_z + 0.08".
    floor = re.search(r"world z ([0-9.]+)", str(need(meta, "floor")))
    ceiling = re.fullmatch(r"p0_z \+ ([0-9.]+)", str(need(meta, "ceiling")).strip())
    if floor is None or ceiling is None:
        pytest.fail(f"meta.json floor {meta['floor']!r} / ceiling {meta['ceiling']!r} not readable: "
                    f"{RERECORD}")
    assert abs(c.TRAINING_FLOOR_Z - (float(floor.group(1)) - need(meta, "arm_base")[2])) < 1e-12
    assert abs(c.CEILING - float(ceiling.group(1))) < 1e-12

def test_calibration_joint_offsets_equal_the_recording(meta):
    np.testing.assert_allclose(c.CALIBRATION_JOINT_OFFSETS_DEG, need(meta, "calibration_joint_offsets_deg"),
                               rtol=0, atol=1e-12)


def test_loop_ik_and_checkpoint_constants_equal_the_recording(meta):
    assert c.CONTROL_HZ == need(meta, "control_hz")
    assert abs(c.TICK_S - need(meta, "substeps") * need(meta, "timestep")) < 1e-12
    assert c.MAX_TICKS == need(meta, "max_steps")
    assert c.IK_ITERS == need(meta, "ik_iters")
    assert abs(c.IK_DAMPING - need(meta, "ik_damping")) < 1e-12
    assert abs(c.IK_TOL_POS - need(meta, "ik_tol_pos")) < 1e-12
    assert abs(c.IK_TOL_ROT - need(meta, "ik_tol_rot")) < 1e-12
    assert c.CHECKPOINT_SHA256 == need(meta, "checkpoint_sha256")
    assert c.CHECKPOINT_BYTES == need(meta, "checkpoint_bytes")


def test_derived_constants_have_their_documented_values():
    assert c.STEP_YAW == 0.08726646259971647
    assert abs(c.FOCAL_PX - 142.49064161307925) < 1e-12
    assert (c.CX, c.CY) == (79.5, 59.5)
    assert abs(c.STREAM_UNITS_S - 76.92307692307692) < 1e-12
    assert c.TICK_S * c.CONTROL_HZ == 1.0
    assert (c.OBS_DIM, c.ACT_DIM) == (19, 5)
    assert abs(c.TEACH_MAX_AGE_S - 43200.0) < 1e-12


def test_camera_rotation_is_orthonormal_and_right_handed():
    R = c.T_GRIPPER_CAMERA[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_grasp_rotation_points_down_and_closes_along_the_yaw():
    np.testing.assert_allclose(c.grasp_R(0.0), [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(c.grasp_R(math.pi / 2), [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], rtol=0, atol=1e-12)
    for yaw in (-3.0, -1.2, 0.0, 0.4, 2.9):
        R = c.grasp_R(yaw)
        np.testing.assert_allclose(R[:, 0], [0, 0, -1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(R[:, 1], [math.cos(yaw), math.sin(yaw), 0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(R[:, 2], np.cross(R[:, 0], R[:, 1]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_yaw_of_reads_the_closing_axis():
    for yaw in (-3.0, -1.2, 0.0, 0.4, 2.9):
        assert abs(c.yaw_of(c.grasp_R(yaw)) - yaw) < 1e-12
    assert abs(c.yaw_of(c.grasp_R(math.pi / 2 + 2 * math.pi)) - math.pi / 2) < 1e-12
    R = np.zeros((3, 3))
    R[:, 1] = [0.0, -1.0, 0.0]
    assert abs(c.yaw_of(R) + math.pi / 2) < 1e-12


def test_wrap_maps_into_minus_pi_to_pi():
    assert c.wrap(math.pi) == -math.pi
    assert c.wrap(-math.pi) == -math.pi
    assert abs(c.wrap(0.1 + 2 * math.pi) - 0.1) < 1e-12
    assert abs(c.wrap(-0.1 - 4 * math.pi) + 0.1) < 1e-12
    assert abs(c.wrap(1.5 * math.pi) + 0.5 * math.pi) < 1e-12
    assert c.wrap(0.0) == 0.0


def test_wrap_keeps_float32_as_training_does():
    y = c.wrap(np.float32(3.0) + np.float32(0.5))
    assert isinstance(y, np.float32)
    assert y == (np.float32(3.5) + np.float32(math.pi)) % np.float32(2 * math.pi) - np.float32(math.pi)


def test_contract_modules_import_nothing_from_the_twin_torch_or_mujoco():
    code = ("import sys\n"
            "import lerobot_mp.twin.grasp.contract, lerobot_mp.twin.grasp.rig\n"
            "import lerobot_mp.twin.grasp.command, lerobot_mp.twin.grasp.keys\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'mujoco', 'cv2')\n"
            "       or (m.startswith('lerobot_mp.twin.') and not m.startswith('lerobot_mp.twin.grasp'))]\n"
            "print(','.join(sorted(bad)))\n")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
