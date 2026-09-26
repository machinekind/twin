"""Real MuJoCo geometry checks for common A1X motion and current object poses."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from lerobot_mp.twin.collision import CollisionChecker
from lerobot_mp.twin import scene as sc
from lerobot_mp.twin.robots import A1X
from lerobot_mp.twin.runtime import Twin
from lerobot_mp.twin.workspace import Workspace


REST = dict(arm_joint1=4.0, arm_joint2=3.5, arm_joint3=-3.5, arm_joint4=0.5,
            arm_joint5=-2.0, arm_joint6=-8.0, gripper=60.0)


def test_common_target_guard_rejects_wrist_rotation_into_folded_shoulder():
    twin = Twin(Workspace(robot="a1x"))
    try:
        twin.scene.set_joints(REST)
        twin.connect("sim", threaded=False)
        twin.set_engaged(True)
        with pytest.raises(RuntimeError, match="kolizja"):
            twin.set_target({"arm_joint4": 34.4})
        assert twin._target is None
        with pytest.raises(RuntimeError, match="kolizja"):
            twin.move({"arm_joint4": 34.4}, 1.0, owner="tag")
        assert twin.owner is None
    finally:
        twin.close()


def test_checker_sees_an_object_moved_after_it_was_constructed():
    scene = sc.build(sc.SceneConfig(robot=A1X, objects=[sc.Box("obstacle", (.05, .05, .05), (.5, .4))]))
    try:
        scene.set_joints(A1X.home)
        checker = CollisionChecker(scene)
        assert checker.config_clear(A1X.home) is None
        body = scene.model.body("obstacle").id
        adr = scene.model.jnt_qposadr[scene.model.body_jntadr[body]]
        scene.data.qpos[adr:adr + 3] = scene.data.site_xpos[scene.kin.site_id]
        assert "obstacle" in checker.config_clear(A1X.home)
        # The checker must not alter the simulator's state.
        before = scene.data.qpos.copy()
        checker.path_clear(A1X.home, {**A1X.home, "arm_joint1": 10.0})
        np.testing.assert_array_equal(scene.data.qpos, before)
    finally:
        scene.close()


def test_final_command_guard_catches_an_obstacle_added_after_target_acceptance():
    twin = Twin(Workspace(robot="a1x"))
    try:
        twin.connect("sim", threaded=False)
        twin.set_engaged(True)
        twin.set_target({"arm_joint1": 5.0})
        twin.configure(objects=[sc.Box("obstacle", (.05, .05, .05), (.5, .4))])
        scene = twin.scene
        adr = scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("obstacle").id]]
        scene.data.qpos[adr:adr + 3] = scene.data.site_xpos[scene.kin.site_id]
        before = scene.data.ctrl.copy()
        twin.step(0.02)
        assert not twin.connected
        assert "kolizja" in twin.status.error
        np.testing.assert_array_equal(scene.data.ctrl, before)
    finally:
        twin.close()


@pytest.mark.parametrize("obstacle", [False, True])
def test_grasp_check_uses_measured_opening_without_exempting_the_palm(obstacle):
    twin = Twin(Workspace(robot="a1x"))
    try:
        boxes = [sc.Box("cube", (.015, .015, .015), (.35, 0))]
        if obstacle:
            boxes.append(sc.Box("obstacle", (.015, .015, .015), (.5, .4)))
        twin.configure(objects=boxes)
        scene = twin.scene
        m, d = scene.model, scene.data
        scene.set_joints({**A1X.home, "gripper": 26.0})
        tips = [m.geom("robot/" + name).id for name in A1X.fingertips]
        centre = d.geom_xpos[tips].mean(axis=0)
        rotation = d.site_xmat[scene.kin.site_id].copy()
        adr = m.jnt_qposadr[m.body_jntadr[m.body("cube").id]]
        d.qpos[adr:adr + 3] = centre
        mujoco.mju_mat2Quat(d.qpos[adr + 3:adr + 7], rotation)
        if obstacle:
            adr2 = m.jnt_qposadr[m.body_jntadr[m.body("obstacle").id]]
            d.qpos[adr2:adr2 + 3] = d.xpos[m.body("robot/gripper_link").id]
        mujoco.mj_forward(m, d)
        checker = CollisionChecker(scene, grasp_objects=("cube",))
        assert checker.blocked_gripper_opening() == pytest.approx(26.0)
        # The virtual fully closed pose penetrates the solid cube beyond the pads.
        assert checker.config_clear({**A1X.home, "gripper": 0.0}) is not None
        twin.connect("sim", threaded=False)
        twin.set_engaged(True)
        if obstacle:
            with pytest.raises(RuntimeError, match="obstacle"):
                twin.set_target({"gripper": 0.0})
        else:
            twin.set_target({"gripper": 0.0})
            assert twin._target["gripper"] == 0.0  # Pressure command itself is preserved.
            d.qpos[adr:adr + 3] += rotation.reshape(3, 3)[:, 1] * .02
            assert checker.blocked_gripper_opening() is None  # One fingertip is insufficient.
    finally:
        twin.close()
