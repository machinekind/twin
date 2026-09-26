"""Demonstrations must exercise the guarded runtime and record its exact inputs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.rl.task import make_task
from lerobot_mp.twin.robots import A1X


@pytest.fixture
def runtime_env(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from a1x_runtime_env import RuntimeLiftEnv
    env = RuntimeLiftEnv(make_task("lift", A1X))
    env.kin.lo, env.kin.hi = env.limits.lo.copy(), env.limits.hi.copy()
    try:
        yield env
    finally:
        env.close()


def test_reset_reproduces_state_and_cannot_select_hardware(runtime_env, monkeypatch):
    env = runtime_env
    connect = env.twin.connect
    calls = []

    def simulation_only(mode, **kwargs):
        assert mode == "sim" and kwargs == {"threaded": False}
        calls.append(mode)
        return connect(mode, **kwargs)

    monkeypatch.setattr(env.twin, "connect", simulation_only)
    first, _ = env.reset(seed=41001)
    before = env.scene.data.qpos.copy()
    action = np.zeros(7)
    action[-1] = .2
    obs, *_ = env.step(action)
    np.testing.assert_array_equal(obs, env.runner.observation(dict(env.twin.status.measured)))
    again, _ = env.reset(seed=41001)
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(before, env.scene.data.qpos)
    assert calls == ["sim", "sim"] and env.twin.status.simulated


def test_invalid_actions_cannot_advance_controller_or_physics(runtime_env):
    env = runtime_env
    env.reset(seed=41001)
    before, command = env.scene.data.qpos.copy(), env.q_cmd.copy()
    for action in (np.zeros(6), np.full(7, np.nan), np.full(7, np.inf)):
        with pytest.raises(ValueError, match="seven finite"):
            env.step(action)
        np.testing.assert_array_equal(before, env.scene.data.qpos)
        np.testing.assert_array_equal(command, env.q_cmd)
        assert env.runner.status.step == 0 and env.t == 0


@pytest.mark.parametrize("seed", [41000, 41001])
def test_scripted_runtime_success_includes_full_physics_hold(runtime_env, seed):
    from collect_a1x_lift_demos import trial
    report, obs, actions = trial(runtime_env, seed, state_triggered=True, synchronized=True,
                                 measured_close=True, lift_rate=.25)
    assert report["success"] and report["max_continuous_hold_s"] >= 2.0 - 1e-8
    assert not report["guard_rejection"] and not report["collision"] and not report["numerical"]
    assert len(obs) == len(actions) == report["steps"]
    assert report["max_coupling_mm"] == runtime_env.monitor.max_coupling_mm
    assert np.isfinite(obs).all() and np.isfinite(actions).all()


def test_rejected_approach_is_a_failed_demonstration(runtime_env):
    from collect_a1x_lift_demos import trial
    report, _, _ = trial(runtime_env, 41000, state_triggered=True)
    assert report["guard_rejection"] and not report["success"]
    assert not report["collision"] and not report["numerical"]
    assert "motion rejected" in report["runtime_stop"]


def test_corrective_labels_are_teacher_commands_not_executed_learner_actions(runtime_env):
    from collect_a1x_lift_demos import trial
    # A learner that never moves the arm cannot reach the first waypoint. Its
    # controls must be recorded for audit but must never become expert labels.
    command = np.r_[np.zeros(6), .2].astype(np.float32)
    learner = SimpleNamespace(act=lambda obs: command.copy())
    executed = []
    report, obs, labels = trial(runtime_env, 42000, state_triggered=True, synchronized=True,
        measured_close=True, lift_rate=.25, learner=learner, learner_probability=1,
        executed_actions=executed)
    assert not report["success"] and not report["scripted"] and report["corrective_teacher_labels"]
    assert report["learner_steps"] == len(executed) == len(labels) == len(obs) > 0
    np.testing.assert_array_equal(executed, np.tile(command, (len(executed), 1)))
    assert np.max(np.abs(np.asarray(labels)[:, :6])) > .9
