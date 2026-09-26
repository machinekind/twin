"""Dataset leakage, failed-demonstration exclusion and usable warm-start artifacts."""

from dataclasses import asdict
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lerobot_mp.twin.rl.imitation import Demonstrations, fit, load_demonstrations
from lerobot_mp.twin.rl.policy import Policy
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.task import make_task
from lerobot_mp.twin.robots import A1X


def dataset(path, offset=0):
    task = make_task("lift", A1X)
    (path / "manifest.json").write_text(json.dumps({"task": asdict(task),
        "randomization": asdict(Randomization.none())}))
    reports = []
    for seed in range(offset, offset + 5):
        failed = seed == offset + 4
        reports.append(dict(seed=seed, success=not failed, collision=failed, numerical=False, steps=3))
        if not failed:  # Failed data must be excluded even if its sample file is absent.
            np.savez(path / f"seed-{seed}.npz", obs=np.full((3, task.obs_dim), seed, np.float32),
                     actions=np.zeros((3, 7), np.float32))
    (path / "report.json").write_text(json.dumps({"episodes": reports}))
    return task


def test_whole_episode_split_excludes_failures_and_tracks_exact_files(tmp_path):
    dataset(tmp_path)
    demos = load_demonstrations(tmp_path, seed=8)
    assert set(demos.provenance["train_seeds"]).isdisjoint(demos.provenance["validation_seeds"])
    assert set(demos.train_obs[:, 0]).isdisjoint(demos.valid_obs[:, 0])
    assert demos.provenance["excluded_seeds"] == [4]
    assert len(demos.provenance["file_sha256"]) == 6
    assert demos.provenance == load_demonstrations(tmp_path, seed=8).provenance


def test_aggregation_preserves_each_sources_validation_split_and_hashes(tmp_path):
    paths = [tmp_path / "first", tmp_path / "second"]
    for path, offset in zip(paths, (0, 10)):
        path.mkdir()
        dataset(path, offset)
    individual = [load_demonstrations(p, seed=8) for p in paths]
    merged = load_demonstrations(paths, seed=8)
    assert merged.provenance["sources"] == [d.provenance for d in individual]
    for field in ("train_seeds", "validation_seeds", "excluded_seeds"):
        assert merged.provenance[field] == sum([d.provenance[field] for d in individual], [])
    np.testing.assert_array_equal(merged.train_obs, np.concatenate([d.train_obs for d in individual]))
    np.testing.assert_array_equal(merged.valid_actions, np.concatenate([d.valid_actions for d in individual]))
    assert set(merged.train_obs[:, 0]).isdisjoint(merged.valid_obs[:, 0])
    assert merged.provenance["excluded_seeds"] == [4, 14]


def test_aggregation_rejects_overlapping_seeds_and_mismatched_tasks(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    dataset(a)
    dataset(b, 10)
    with pytest.raises(ValueError, match="seeds overlap"):
        load_demonstrations([a, a])
    path = b / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["task"]["cube_half"] = .016
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identical task"):
        load_demonstrations([a, b])


@pytest.mark.parametrize("problem", ["nan", "out_of_range", "missing_step", "duplicate_seed"])
def test_invalid_successful_episode_is_rejected(tmp_path, problem):
    task = dataset(tmp_path)
    obs, act = np.zeros((3, task.obs_dim), np.float32), np.zeros((3, 7), np.float32)
    if problem == "nan":
        obs[1, 2] = np.nan
    elif problem == "out_of_range":
        act[1, 6] = 1.1
    elif problem == "missing_step":
        act = act[:-1]
    else:
        path = tmp_path / "report.json"
        report = json.loads(path.read_text())
        report["episodes"].append(report["episodes"][0])
        path.write_text(json.dumps(report))
    np.savez(tmp_path / "seed-0.npz", obs=obs, actions=act)
    with pytest.raises(ValueError):
        load_demonstrations(tmp_path)


def test_training_uses_only_training_statistics_and_saves_a_reloadable_actor(tmp_path):
    torch.set_num_threads(2)
    task = make_task("lift", A1X)
    rng = np.random.default_rng(9)
    x = rng.normal(0, .5, (256, task.obs_dim)).astype(np.float32)
    vx = rng.normal(.1, .5, (64, task.obs_dim)).astype(np.float32)
    targets = lambda obs: np.repeat(np.tanh(obs[:, :1]) * .5, 7, axis=1)
    demos = Demonstrations(asdict(task), asdict(Randomization.none()), x, targets(x), vx, targets(vx), {})
    policy, result = fit(demos, updates=300, seconds=20, hidden=(32, 32), lr=.003, report_every=50)
    assert result["selected_validation"]["loss"] < result["initial_validation"]["loss"] * .2
    np.testing.assert_allclose(policy.norm.mean.cpu(), x.mean(0), atol=1e-6)
    assert policy.norm.count.item() == pytest.approx(len(x), abs=.001)
    assert policy.meta.steps == 0 and policy.critic_state is None
    policy.save(tmp_path / "policy.pt")
    restored = Policy.load(tmp_path / "policy.pt")
    np.testing.assert_array_equal(restored.act(vx), policy.act(vx))
    assert restored.task == policy.task


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ppo_critic_warmup_preserves_imitation_actor_and_normalizer():
    pytest.importorskip("mujoco_warp")
    from lerobot_mp.twin.rl.policy import PolicyMeta
    from lerobot_mp.twin.rl.ppo import PPOConfig, train
    from lerobot_mp.twin.workspace import Workspace

    task = make_task("lift", A1X)
    initial = Policy(PolicyMeta(task=asdict(task), hidden=[32], obs_dim=task.obs_dim, act_dim=7,
                               robot="a1x", randomization=asdict(Randomization.none())))
    before = {k: v.clone() for k, v in initial.state_dict().items()}
    warmed = train(task, PPOConfig(num_envs=32, horizon=8, iterations=1, epochs=1,
        minibatches=2, hidden=(32,), critic_warmup=1, init_std=.1),
        init=initial, workspace=Workspace(robot="a1x"), randomization=Randomization.none())
    assert warmed.critic_state is not None
    for name, value in before.items():
        torch.testing.assert_close(warmed.state_dict()[name].cpu(), value, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("frozen", [False, True])
def test_normalizer_freeze_survives_warmup_while_actor_can_learn(frozen, tmp_path):
    pytest.importorskip("mujoco_warp")
    from lerobot_mp.twin.rl.policy import PolicyMeta
    from lerobot_mp.twin.rl.ppo import PPOConfig, train
    from lerobot_mp.twin.workspace import Workspace

    torch.manual_seed(42)
    task = make_task("lift", A1X)
    initial = Policy(PolicyMeta(task=asdict(task), hidden=[32], obs_dim=task.obs_dim, act_dim=7,
                               robot="a1x", randomization=asdict(Randomization.none())))
    initial.norm.update(torch.randn(128, task.obs_dim) * .5)
    before = {k: v.clone() for k, v in initial.state_dict().items()}
    learned = train(task, PPOConfig(num_envs=32, horizon=8, iterations=2, epochs=1,
        minibatches=2, hidden=(32,), critic_warmup=1, init_std=.1, freeze_normalizer=frozen,
        save_every=1, checkpoint_history=True), init=initial, out_dir=tmp_path,
        workspace=Workspace(robot="a1x"), randomization=Randomization.none())
    assert any(not torch.equal(value.cpu(), before[name]) for name, value in learned.state_dict().items()
               if name.startswith("actor."))
    if frozen:
        for name, value in before.items():
            if name.startswith("norm."):
                torch.testing.assert_close(learned.state_dict()[name].cpu(), value, atol=0, rtol=0)
    else:
        assert learned.norm.count.item() > before["norm.count"].item()
    first = Policy.load(tmp_path / "checkpoints/iteration-000001/policy.pt")
    second = Policy.load(tmp_path / "checkpoints/iteration-000002/policy.pt")
    assert first.meta.iterations == 1 and second.meta.iterations == 2
    for name, value in before.items():
        torch.testing.assert_close(first.state_dict()[name].cpu(), value, atol=0, rtol=0)
    for name, value in learned.state_dict().items():
        torch.testing.assert_close(second.state_dict()[name].cpu(), value.cpu(), atol=0, rtol=0)
