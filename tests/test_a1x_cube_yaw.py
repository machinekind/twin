"""Square-cube yaw continuity, checkpoint compatibility and runtime/CPU parity."""

from dataclasses import asdict, replace

import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

from lerobot_mp.twin.rl import task as tk
from lerobot_mp.twin.rl.env import TwinEnv
from lerobot_mp.twin.rl.policy import Policy, PolicyMeta
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.runner import PolicyRunner
from lerobot_mp.twin.robots import A1X
from lerobot_mp.twin.runtime import Twin
from lerobot_mp.twin.workspace import Workspace


def rotation(degrees):
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def metadata(mode="continuous"):
    task = tk.make_task("lift", A1X)
    return PolicyMeta(task=asdict(task), robot="a1x", hidden=[8], obs_dim=36, act_dim=7,
                      randomization={"fold_yaw": True}, cube_yaw_mode=mode)


@pytest.mark.parametrize("angles", [[45.3, 45.22, 44.98, 45.13], [-45.3, -45.22, -44.98, -45.13]])
def test_small_rotation_crossing_boundary_does_not_turn_into_quarter_turn(angles):
    tracker = tk.ContinuousCubeYaw()
    inputs = [rotation(a) for a in angles]
    outputs = [tracker(r) for r in inputs]
    np.testing.assert_array_equal(outputs[0], tk.fold_yaw(np, inputs[0]))
    yaws = np.degrees([np.arctan2(r[1, 0], r[0, 0]) for r in outputs])
    np.testing.assert_allclose(np.diff(yaws), np.diff(angles), atol=1e-12)
    # Different 90-degree detector labels describe the same square cube.
    np.testing.assert_allclose(tracker(rotation(angles[-1] + 90)), outputs[-1], atol=1e-14)
    np.testing.assert_allclose(tracker(inputs[-1]), outputs[-1], atol=1e-14)
    tracker.reset()
    np.testing.assert_array_equal(tracker(inputs[2]), tk.fold_yaw(np, inputs[2]))


@pytest.mark.parametrize("invalid", [np.zeros((2, 2)), np.full((3, 3), np.nan), np.full((3, 3), np.inf)])
def test_invalid_orientation_is_rejected_without_changing_history(invalid):
    tracker = tk.ContinuousCubeYaw()
    tracker(rotation(46))
    before = tracker.previous
    with pytest.raises(ValueError, match="finite 3x3"):
        tracker(invalid)
    assert tracker.previous == before


@pytest.mark.parametrize("changes", [dict(cube_yaw_mode="unknown"), dict(robot="so101"),
    dict(obs_dim=30), dict(act_dim=6),
    dict(randomization={"fold_yaw": False}), dict(task={**metadata().task, "name": "pickplace"}),
    dict(task={**metadata().task, "cube_half": .02})])
def test_unsupported_orientation_contracts_are_rejected(changes):
    with pytest.raises(ValueError, match="cube-lift contract"):
        Policy(replace(metadata(), **changes))


def test_checkpoint_mode_roundtrip_and_legacy_default_do_not_change_weights(tmp_path):
    torch.manual_seed(19)
    old = Policy(metadata("canonical"))
    torch.manual_seed(19)
    new = Policy(metadata())
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], atol=0, rtol=0)
    path = new.save(tmp_path / "policy.pt")
    assert Policy.load(path).meta.cube_yaw_mode == "continuous"
    blob = torch.load(path, weights_only=False)
    del blob["meta"]["cube_yaw_mode"]
    torch.save(blob, path)
    loaded = Policy.load(path)
    assert loaded.meta.cube_yaw_mode == "canonical"
    x = np.random.default_rng(3).normal(size=(5, 36)).astype(np.float32)
    np.testing.assert_array_equal(old.act(x), loaded.act(x))


@pytest.mark.parametrize("mode", ["canonical", "continuous"])
def test_cpu_and_real_runner_observation_paths_agree_and_reset(mode):
    policy = Policy(metadata(mode))
    workspace = Workspace(robot="a1x")
    env = TwinEnv(policy.task, workspace=workspace,
                  randomization=replace(Randomization.none(), fold_yaw=True), cube_yaw_mode=mode)
    twin = Twin(workspace)
    supplied = [np.array([.31, .02, .015]), rotation(45.3), "symulacja"]
    try:
        twin.connect("sim", threaded=False)
        runner = PolicyRunner(twin, policy, cube_provider=lambda: tuple(supplied))
        runner.start(threaded=False)
        env.reset(seed=42)
        # Establish the same first delivered orientation in both paths.
        if env._cube_yaw is not None:
            env._cube_yaw.reset()
        measured = dict(twin.status.measured)
        for angle in [45.3, 45.22, 44.98, 45.13]:
            supplied[1] = rotation(angle)
            before = supplied[1].copy()
            s = env.state()
            s.update(cube_pos=supplied[0], cube_rot=supplied[1])
            env._perceive(s, force=True)
            observed = runner.observation(measured)
            np.testing.assert_array_equal(observed[23:29], env._obs(s)[23:29])
            np.testing.assert_array_equal(supplied[1], before)
            np.testing.assert_array_equal(runner._cube, supplied[0])
            # Repeated reads between controller steps cannot advance the tracker.
            np.testing.assert_array_equal(observed, runner.observation(measured))
        runner.stop()
        supplied[1] = rotation(44.98)
        runner.start(threaded=False)
        expected = tk.fold_yaw(np, supplied[1])[:, :2].T.reshape(-1).astype(np.float32)
        np.testing.assert_array_equal(runner.observation(measured)[23:29], expected)
        if env._cube_yaw is not None:
            env._cube_yaw.previous = 3.0
        reset_obs, _ = env.reset(seed=42)
        true = env.state()["cube_rot"]
        expected = tk.fold_yaw(np, true)[:, :2].T.reshape(-1).astype(np.float32)
        np.testing.assert_array_equal(reset_obs[23:29], expected)
    finally:
        twin.close()
        env.close()


def test_training_cannot_silently_drop_continuous_observation_mode(monkeypatch):
    from lerobot_mp.twin.rl.imitation import Demonstrations, fit
    from lerobot_mp.twin.rl import ppo

    policy = Policy(metadata())
    x, y = np.zeros((4, 36), np.float32), np.zeros((4, 7), np.float32)
    demos = Demonstrations(policy.meta.task, policy.meta.randomization, x, y, x, y, {})
    with pytest.raises(ValueError, match="matching demonstration contract"):
        fit(demos, init=policy, updates=1, seconds=1)
    def must_not_construct_environment(*args, **kwargs):
        pytest.fail("Unsupported observations must fail before GPU initialization")
    monkeypatch.setattr(ppo, "BatchEnv", must_not_construct_environment)
    with pytest.raises(ValueError, match="does not support continuous"):
        ppo.train(policy.task, workspace=Workspace(robot="a1x"), init=policy, device="cpu")
