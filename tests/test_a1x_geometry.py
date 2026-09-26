"""Independent vendor-URDF geometry checks; no claim about hardware calibration."""

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.kinematics import RobotKinematics
from lerobot_mp.twin.robots import A1X


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def urdf_forward(joints, positions):
    transforms = {"base_link": np.eye(4)}
    pending = list(joints)
    while pending:
        before = len(pending)
        for joint in pending.copy():
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in transforms:
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            local = np.eye(4)
            local[:3, :3] = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
            local[:3, 3] = xyz
            move = np.eye(4)
            kind = joint.get("type")
            if kind != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                value = positions[joint.get("name")]
                if kind == "prismatic":
                    move[:3, 3] = axis * value
                else:
                    move[:3, :3] = rotation(axis, value)
            transforms[child] = transforms[parent] @ local @ move
            pending.remove(joint)
        assert len(pending) < before, "URDF links do not form a connected tree"
    return transforms


def test_a1x_link_poses_directions_and_ranges_match_vendor_urdf():
    kin = RobotKinematics(A1X)
    joints = ET.parse(A1X.mjcf_path.with_suffix(".urdf")).getroot().findall("joint")
    rng = np.random.default_rng(57)
    for joint in joints:
        if joint.get("type") == "fixed":
            continue
        name = "gripper" if joint.get("name") == "gripper_finger_joint1" else joint.get("name")
        bounds = [float(joint.find("limit").get(k)) for k in ("lower", "upper")]
        np.testing.assert_allclose(kin.model.joint(name).range, bounds, atol=1e-7)
    for _ in range(100):
        q = rng.uniform(kin.lo, kin.hi)
        app = kin.from_q(q)
        np.testing.assert_allclose(kin.to_q(app), q, atol=1e-12)
        positions = dict(zip(A1X.joints, q))
        positions["gripper_finger_joint1"] = positions.pop("gripper")
        positions["gripper_finger_joint2"] = -positions["gripper_finger_joint1"]
        for link, expected in urdf_forward(joints, positions).items():
            np.testing.assert_allclose(kin.body(link, app), expected, atol=2e-6,
                                       err_msg=f"Vendor/MJCF link mismatch: {link}")


def test_a1x_jaw_units_tcp_and_generator_are_reproducible():
    kin = RobotKinematics(A1X)
    for opening in (0, 50, 100):
        kin.tcp({**A1X.home, "gripper": opening})
        assert kin.data.qpos[kin.qadr[-1]] == pytest.approx(opening * .0005)
        assert kin.data.qpos[kin.mirror[0][1]] == pytest.approx(-opening * .0005)
        midpoint = np.mean([kin.data.geom_xpos[kin.model.geom(name).id] for name in A1X.fingertips], axis=0)
        offset = kin.data.site_xmat[kin.site_id].reshape(3, 3).T @ (midpoint - kin.data.site_xpos[kin.site_id])
        np.testing.assert_allclose(offset, [.02989, 0, 0], atol=1e-7)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("a1x_generator", root / "scripts/import_a1x_model.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert generator.generate(A1X.mjcf_path.with_suffix(".urdf")) == A1X.mjcf_path.read_text()
