from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")
import torch

from lerobot_mp.twin.rl.env import TwinEnv
from lerobot_mp.twin.rl.multi import object_task, shape_warm_start
from lerobot_mp.twin.rl.objects import OBJECT_KINDS, bottom_height, object_spec
from lerobot_mp.twin.rl.policy import Policy, bundled_dir
from lerobot_mp.twin.rl.randomize import Randomization, corrupt_pose
from lerobot_mp.twin.rl.task import make_task, smooth_cost


@pytest.mark.parametrize("kind", OBJECT_KINDS)
def test_objects_compile_remain_finite_and_have_observable_shape(kind):
    task = object_task(make_task("lift"), kind)
    env = TwinEnv(task, randomization=Randomization.none())
    try:
        obs, _ = env.reset(seed=5)
        assert obs.shape == (task.obs_dim,)
        assert obs[33 + OBJECT_KINDS.index(kind)] == 1
        action = np.zeros(6)
        action[-1] = 1.
        for _ in range(12):
            obs, _, _, _, info = env.step(action)
        assert np.isfinite(obs).all()
        assert abs(info["height"]) < .005, info
        assert env.scene.model.body_mass[env.fields.cube_body] == pytest.approx(object_spec(kind).mass)
        assert len(env.fields.cube_geoms) == len(object_spec(kind).parts)
    finally:
        env.close()


def test_warm_start_preserves_actions_for_existing_observation():
    source = Policy.load(bundled_dir() / "lift-v3/policy.pt")
    dest = shape_warm_start(source)
    rng = np.random.default_rng(42)
    obs = rng.normal(size=(20, source.task.obs_dim)).astype(np.float32)
    expanded = np.c_[obs, rng.normal(size=(20, dest.task.obs_dim-source.task.obs_dim))].astype(np.float32)
    np.testing.assert_allclose(source.act(obs), dest.act(expanded), atol=2e-6)
    assert dest.critic_state["0.weight"].shape[1] == dest.task.obs_dim


def test_long_object_touching_table_is_not_lifted_by_its_centre():
    task = object_task(make_task("lift"), "pliers")
    R = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    p = np.array([.2,0.,.063])
    height = bottom_height(np, task, p, R)
    assert height < .01  # centre is >6 cm up, but a handle/head is still at the table
    got = bottom_height(torch, task, torch.tensor(p), torch.tensor(R))
    assert float(got) == pytest.approx(height)


def test_calibration_error_is_rigid_and_cpu_gpu_formula_matches():
    p = np.array([[.2,.03,.08]])
    R = np.eye(3)[None]
    shift, rpy, scale = np.array([[.003,-.004,.002]]), np.array([[.01,-.02,.03]]), np.array([1.02])
    expected = corrupt_pose(np, p, R, shift, rpy, scale)
    actual = corrupt_pose(torch, *map(torch.tensor, (p,R,shift,rpy,scale)))
    for a,b in zip(expected,actual):
        np.testing.assert_allclose(a,b.numpy(),atol=1e-12)
    np.testing.assert_allclose(expected[1] @ expected[1].transpose(0,2,1), R, atol=1e-12)


def test_smoothness_penalizes_jerk_but_not_gripper_closure():
    task = replace(make_task("lift"), smooth_jerk=.1)
    z = np.zeros((1,6))
    abrupt = z.copy(); abrupt[0,0] = .02
    close = z.copy(); close[0,5] = .5
    assert smooth_cost(np,task,abrupt,z,z)[0] > 0
    assert smooth_cost(np,task,close,z,z)[0] == 0
    np.testing.assert_allclose(smooth_cost(np,task,abrupt,z,z),
        smooth_cost(torch,task,*map(torch.tensor,(abrupt,z,z))).numpy())


def test_calibration_bias_is_constant_within_episode():
    env = TwinEnv("lift", randomization=replace(Randomization.none(), calibration_translation=.01,
                                               calibration_rotation_deg=2., joint_bias_deg=.5))
    try:
        env.reset(seed=4)
        bias = env._calibration[0].copy()
        for _ in range(3):
            env.step(np.zeros(6))
            np.testing.assert_array_equal(env._calibration[0], bias)
        env.reset(seed=5)
        assert not np.allclose(env._calibration[0], bias)
    finally:
        env.close()


def test_stale_pose_freezes_arm_and_preserves_gripper_target():
    task = object_task(make_task("lift"), "cube")
    env = TwinEnv(task, randomization=Randomization.none())
    try:
        env.reset(seed=3)
        before = env.q_cmd.copy()
        env.t = 20
        env._last_seen_tick = 0
        _,_,_,_,info = env.step(np.ones(6))
        np.testing.assert_array_equal(env.q_cmd,before)
        assert info["stale_pause"] and not info["finished"]
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_gpu_has_all_shapes_and_finite_contacts():
    pytest.importorskip("mujoco_warp")
    from lerobot_mp.twin.rl.multi import MixedBatchEnv
    task = object_task(make_task("lift", smooth_jerk=.01), "cube")
    rand = replace(Randomization.none(), calibration_translation=.002,
                   calibration_rotation_deg=.5, joint_bias_deg=.1,
                   dropout_probability=.1, dropout_steps=2)
    env = MixedBatchEnv(task, 96, randomization=rand, seed=42)
    obs = env.reset()
    assert obs.shape == (96, task.obs_dim)
    actions = torch.zeros(96,6,device="cuda")
    actions[:,5] = 1
    for _ in range(8):
        obs, reward, done, info = env.step(actions)
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    assert not info["broken"].any()
    assert torch.all(info["height"].abs() < .01)
    for i in range(12):
        assert torch.all(obs[i*8:(i+1)*8,33+i] == 1)
