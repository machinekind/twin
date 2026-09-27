"""GPU-specific mixed-unit noise, overflow, reset and CPU observation/reward checks."""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from lerobot_mp.twin.rl.batch import BatchEnv
from lerobot_mp.twin.rl.env import TwinEnv
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.task import make_task
from lerobot_mp.twin.robots import A1X
from lerobot_mp.twin.workspace import Workspace


def test_gpu_angular_noise_does_not_become_linear_jaw_noise_and_overflow_is_fatal():
    rand = replace(Randomization.none(), obs_noise_deg=.3, obs_noise_m=.0003)
    env = BatchEnv("reach", 256, workspace=Workspace(robot="a1x"), randomization=rand)
    env.reset()
    state = env._state()
    expected = 2 * (state["q"] - env.lo) / (env.hi - env.lo) - 1
    noise = torch.cat([env._obs(state)[:, :7] - expected for _ in range(16)], dim=0)
    measured = noise.double().std(dim=0, unbiased=False).cpu().numpy() * (env.limits.hi - env.limits.lo) / 2
    np.testing.assert_allclose(measured[:6], np.full(6, np.radians(.3)), rtol=.06)
    assert measured[-1] == pytest.approx(.0003, rel=.06)
    # Fault injection in the accumulated GPU diagnostics, not a mocked return value.
    env._overflow[0] = int(env.mjw.OverflowType.NEFC)
    with pytest.raises(RuntimeError, match="capacity overflow"):
        env.step(torch.zeros((256, 7), device=env.device))


def test_cpu_gpu_observations_rewards_and_timeout_reset_agree():
    task = make_task("lift", A1X, episode_steps=8, end_on_success=0, start_noise_deg=0)
    ws, rand = Workspace(robot="a1x"), Randomization.none()
    cpu = TwinEnv(task, workspace=ws, randomization=rand)
    try:
        cpu.reset(seed=4, options={"q": cpu.limits.home, "cube": ([.35, 0, .015], [1, 0, 0, 0])})
        gpu = BatchEnv(task, 1, workspace=ws, randomization=rand)
        gpu.reset()
        gpu.qpos[:] = torch.as_tensor(cpu.scene.data.qpos, dtype=torch.float32, device=gpu.device)
        gpu.qvel.zero_()
        gpu.qacc_ws.zero_()
        gpu.q_cmd[:] = torch.as_tensor(cpu.q_cmd, dtype=torch.float32, device=gpu.device)
        gpu.ctrl[:, gpu.act] = gpu.q_cmd
        gpu.mjw.forward(gpu.m, gpu.d)
        gpu.wp.synchronize_device()
        gpu._perceive(gpu._state())
        np.testing.assert_allclose(gpu._state()["grasp_pos"][0].cpu().numpy(), cpu.state()["grasp_pos"], atol=1e-6)
        action = np.array([.01, -.02, .03, 0, 0, 0, .2], dtype=np.float32)
        for step in range(task.episode_steps):
            c_obs, c_reward, term, trunc, c_info = cpu.step(action)
            g_obs, g_reward, done, g_info = gpu.step(torch.as_tensor(action[None], device=gpu.device))
            assert g_reward.item() == pytest.approx(c_reward, abs=1e-4)
            assert g_info["success"].item() == c_info["success"]
            assert done.item() == (term or trunc)
            if not done.item():
                np.testing.assert_allclose(g_obs[0].cpu().numpy(), c_obs, atol=1e-4)
        assert gpu.t.item() == 0
        assert gpu.streak.item() == 0
        assert not gpu.prev_action.any()
        assert not gpu.qvel.any()
        assert not gpu.physics_diagnostics()["capacity_overflow_worlds"]
        # A folded wrist/shoulder collision must end the episode on both engines.
        folded = dict(arm_joint1=4.0, arm_joint2=3.5, arm_joint3=-3.5, arm_joint4=34.4,
                      arm_joint5=-2.0, arm_joint6=-8.0, gripper=60.0)
        cpu.reset(seed=4)
        cpu.scene.set_joints(folded)
        cpu.q_cmd = cpu.kin.to_q(folded)
        gpu.qpos[:] = torch.as_tensor(cpu.scene.data.qpos, dtype=torch.float32, device=gpu.device)
        gpu.qvel.zero_()
        gpu.qacc_ws.zero_()
        gpu.q_cmd[:] = torch.as_tensor(cpu.q_cmd, dtype=torch.float32, device=gpu.device)
        _, c_reward, term, _, c_info = cpu.step(np.zeros(7))
        _, g_reward, done, g_info = gpu.step(torch.zeros((1, 7), device=gpu.device))
        assert term and done.item()
        assert c_info["unsafe"] and g_info["unsafe"].item()
        assert not c_info["success"] and not g_info["success"].item()
        assert c_reward == g_reward.item() == -10.0
    finally:
        cpu.close()
