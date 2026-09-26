"""A1X action timing and mixed angular/linear observation units."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")

from lerobot_mp.config import A1XConfig
from lerobot_mp.twin.robots import A1X, SO101, robot_config
from lerobot_mp.twin.rl.env import TwinEnv
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl import task as tk
from lerobot_mp.twin.workspace import Workspace


def test_a1x_full_gripper_command_takes_at_least_the_backend_ramp_time():
    env = TwinEnv("reach", workspace=Workspace(robot="a1x"), randomization=Randomization.none())
    try:
        hw = A1XConfig()
        duration = abs(hw.gripper_open_pdes - hw.gripper_closed_pdes) / hw.gripper_rate_pdes_s
        q = env.limits.home[None].copy()
        q[0, -1] = env.limits.hi[-1]
        action = np.zeros_like(q)
        action[0, -1] = -1
        steps = 0
        while q[0, -1] > env.limits.lo[-1] + 1e-9:
            q = tk.apply_action(np, env.task, env.limits, q, action)
            steps += 1
            assert steps <= 100
        assert steps / env.task.control_hz == pytest.approx(duration, abs=1 / env.task.control_hz)
        assert robot_config(A1X).joint("gripper").max_vel <= 100 / duration
    finally:
        env.close()


def test_gripper_rate_tracks_control_frequency_and_so101_is_unchanged():
    assert tk.make_task(spec=A1X, control_hz=40).grip_step * 40 == pytest.approx(
        tk.make_task(spec=A1X).grip_step * 20)
    assert tk.make_task(spec=SO101).grip_step == 0.15


def test_new_a1x_lifts_require_two_seconds_but_saved_overrides_are_preserved():
    for hz in (20, 40):
        task = tk.make_task("lift", A1X, control_hz=hz)
        assert task.end_on_success / hz == 2
    assert tk.make_task("reach", A1X).end_on_success == 0
    assert tk.make_task("lift", SO101).end_on_success == 10
    assert tk.make_task("lift", A1X, end_on_success=7).end_on_success == 7


@pytest.mark.parametrize("finished,expected", [(False, 0.0), (True, 1.0)])
def test_evaluation_does_not_count_a_last_frame_lift_as_a_completed_hold(monkeypatch, finished, expected):
    from types import SimpleNamespace
    import importlib
    evaluation = importlib.import_module("lerobot_mp.twin.rl.evaluate")

    class Episode:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, seed):
            return np.zeros(1), {}

        def step(self, action):
            return np.zeros(1), 1.0, False, True, {
                "success": True, "finished": finished, "height": .07}

        def close(self):
            pass

    monkeypatch.setattr(evaluation, "TwinEnv", Episode)
    policy = SimpleNamespace(meta=SimpleNamespace(robot="a1x"), task=tk.make_task("lift", A1X),
                             act=lambda obs: np.zeros(7))
    result = evaluation.evaluate(policy, episodes=1, workspace=Workspace(robot="a1x"))
    assert result["success"] == expected
    assert result["ever_success"] == result["final_state_success"] == 1.0
    assert result["required_hold_s"] == 2.0


@pytest.mark.parametrize("linear_noise", [0.0, 0.0003])
def test_a1x_cpu_observation_noise_uses_metres_for_the_sliding_joint(linear_noise):
    rand = replace(Randomization.none(), obs_noise_deg=0.3, obs_noise_m=linear_noise)
    env = TwinEnv("reach", workspace=Workspace(robot="a1x"), randomization=rand)
    try:
        env.reset(seed=123)
        state = env.state()
        samples = np.array([env._obs(state)[:7] for _ in range(4096)])
        std_q = samples.astype(np.float64).std(axis=0) * (env.limits.hi - env.limits.lo) / 2
        assert std_q[:6] == pytest.approx(np.full(6, np.radians(0.3)), rel=0.06)
        # Angular noise must not inject 5 mm of fake jaw travel.
        assert std_q[-1] == pytest.approx(linear_noise, rel=0.06, abs=1e-7)
    finally:
        env.close()
