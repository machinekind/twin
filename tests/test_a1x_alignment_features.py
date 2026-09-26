from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from lerobot_mp.twin.kinematics import RobotKinematics
from lerobot_mp.twin.rl.features import A1XAlignmentFeatures
from lerobot_mp.twin.rl.imitation import Demonstrations, fit
from lerobot_mp.twin.rl.policy import Policy, PolicyMeta
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.task import Limits, make_task, observe
from lerobot_mp.twin.robots import A1X


def observations(count=20):
    kin = RobotKinematics(A1X)
    limits = Limits.of(kin)
    rng = np.random.default_rng(4)
    q, command = [rng.uniform(limits.lo, limits.hi, (count, 7)) for _ in range(2)]
    tcp = np.array([kin.tcp(kin.from_q(v))[:3, 3] for v in q])
    cube = rng.uniform([.2, -.1, .015], [.4, .1, .1], (count, 3))
    rot = np.tile(np.eye(3), (count, 1, 1))
    return observe(np, make_task('lift', A1X), limits, q, command, tcp,
                   np.zeros((count, 7)), cube_pos=cube, cube_rot=rot).astype(np.float32)


def metadata(features='raw'):
    task = make_task('lift', A1X)
    return PolicyMeta(task=asdict(task), hidden=[32, 32], obs_dim=36, act_dim=7,
                      robot='a1x', randomization=asdict(Randomization.none()), features=features)


def test_derived_tip_positions_match_independent_fk_and_supplied_cube_changes():
    encoder = A1XAlignmentFeatures()
    obs = observations()
    values = encoder(obs)
    kin = RobotKinematics(A1X)
    limits = Limits.of(kin)
    for k, offset in ((0, 0), (7, 3)):
        for i, row in enumerate(obs):
            q = limits.lo + .5*(row[k:k+7].astype(float)+1)*(limits.hi-limits.lo)
            kin.tcp(kin.from_q(q))
            midpoint = np.mean([kin.data.geom_xpos[kin.model.geom(n).id] for n in A1X.fingertips], axis=0)
            np.testing.assert_allclose(values[i, offset:offset+3]*.05 + row[17:20], midpoint, atol=2e-7)
    shifted = obs.copy()
    shifted[:, 17:20] += [.003, -.005, .007]
    changed = encoder(shifted)
    for start in (0, 3):
        np.testing.assert_allclose(changed[:, start:start+3]-values[:, start:start+3],
                                   np.tile([-.06, .1, -.14], (len(obs), 1)), atol=3e-6)
    np.testing.assert_array_equal(values[:, 6:18], changed[:, 6:18])
    np.testing.assert_array_equal(encoder(obs[::-1])[::-1], values)
    np.testing.assert_array_equal(np.concatenate([encoder(row[None]) for row in obs]), values)


def test_initial_feature_extension_preserves_raw_actions_and_rng():
    torch.manual_seed(17)
    raw = Policy(metadata())
    next_raw = torch.rand(6)
    torch.manual_seed(17)
    derived = Policy(metadata(A1XAlignmentFeatures.name))
    torch.testing.assert_close(torch.rand(6), next_raw, atol=0, rtol=0)
    obs = observations()
    np.testing.assert_allclose(raw.act(obs), derived.act(obs), atol=2e-6)
    assert not torch.count_nonzero(derived.actor[0].weight[:, 36:])


def test_orientation_features_follow_supplied_pose_including_roll():
    encoder = A1XAlignmentFeatures()
    obs = observations(1)
    obs[:, 7:14] = obs[:, :7]
    kin = RobotKinematics(A1X)
    limits = Limits.of(kin)
    q = limits.lo + .5*(obs[0, :7].astype(float)+1)*(limits.hi-limits.lo)
    measured = kin.tcp(kin.from_q(q))[:3, :3]
    grasp_rotation = np.column_stack(([0, 0, -1], [0, 1, 0], [1, 0, 0]))
    cube = measured @ grasp_rotation.T
    obs[0, 23:26], obs[0, 26:29] = cube[:, 0], cube[:, 1]
    aligned = encoder(obs)
    np.testing.assert_allclose(aligned[:, 6:18], 0, atol=3e-6)
    c, s = np.cos(.02), np.sin(.02)
    cube = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ cube
    obs[0, 23:26], obs[0, 26:29] = cube[:, 0], cube[:, 1]
    biased = encoder(obs)
    assert np.linalg.norm(biased[:, 6:12]) > .1
    np.testing.assert_array_equal(aligned[:, :6], biased[:, :6])


def test_feature_refinement_cache_reload_and_frozen_baseline(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(11)
    raw = Policy(metadata())
    obs = observations(100)
    raw.norm.update(torch.as_tensor(obs[:80]))
    original = {k: v.clone() for k, v in raw.state_dict().items()}
    targets = raw.act(obs) + .02
    demos = Demonstrations(raw.meta.task, raw.meta.randomization, obs[:80], targets[:80], obs[80:], targets[80:], {})
    candidate, report = fit(demos, init=raw, features=A1XAlignmentFeatures.name,
                            updates=200, seconds=20, lr=.001, report_every=50)
    assert report['initial_validation']['loss'] == pytest.approx(.0008, abs=1e-6)
    assert report['selected_validation']['loss'] < report['initial_validation']['loss']
    assert torch.count_nonzero(candidate.actor[0].weight[:, 36:])
    for key, value in original.items():
        torch.testing.assert_close(raw.state_dict()[key], value, atol=0, rtol=0)
        if key.startswith('norm.'):
            torch.testing.assert_close(candidate.state_dict()[key], value, atol=0, rtol=0)
    x = torch.as_tensor(obs)
    with torch.no_grad():
        torch.testing.assert_close(candidate(x), candidate(x, derived=candidate.derived_features(x)), atol=0, rtol=0)
    candidate.save(tmp_path/'policy.pt')
    reloaded = Policy.load(tmp_path/'policy.pt')
    np.testing.assert_array_equal(reloaded.act(obs), candidate.act(obs))
    with pytest.raises(ValueError, match='cannot remove'):
        fit(demos, init=reloaded, features='raw', updates=1, seconds=1)


def test_arm_history_ablation_is_saved_in_weights_and_preserved_on_refinement(tmp_path):
    torch.set_num_threads(2)
    raw = Policy(metadata())
    obs = observations(50)
    obs[:, 29:36] = np.random.default_rng(9).uniform(-1, 1, (50, 7))
    raw.norm.update(torch.as_tensor(obs[:40]))
    original_obs = obs.copy()
    original_state = {k: v.clone() for k, v in raw.state_dict().items()}
    neutral = obs.copy()
    neutral[:, 29:35] = raw.norm.mean[29:35].numpy()
    targets = raw.act(neutral) + .02
    demos = Demonstrations(raw.meta.task, raw.meta.randomization, obs[:40], targets[:40], obs[40:], targets[40:], {})
    candidate, report = fit(demos, init=raw, ignore_previous_arm=True, updates=100, seconds=10, lr=.001)
    assert report['ignore_previous_arm'] and report['ignored_observation_indices'] == list(range(29, 35))
    assert report['initial_validation']['loss'] == pytest.approx(.0008, abs=1e-6)
    assert report['selected_validation']['loss'] < report['initial_validation']['loss']
    assert not torch.count_nonzero(candidate.actor[0].weight[:, 29:35])
    assert torch.count_nonzero(candidate.actor[0].weight[:, 35])  # Gripper history remains available.
    np.testing.assert_array_equal(obs, original_obs)
    for key, value in original_state.items():
        torch.testing.assert_close(raw.state_dict()[key], value, atol=0, rtol=0)
    candidate.save(tmp_path/'policy.pt')
    restored = Policy.load(tmp_path/'policy.pt')
    changed = obs.copy()
    changed[:, 29:35] = .9
    np.testing.assert_array_equal(restored.act(changed), restored.act(obs))
    again, report = fit(demos, init=restored, updates=1, seconds=5)
    assert report['ignore_previous_arm']
    assert not torch.count_nonzero(again.actor[0].weight[:, 29:35])


@pytest.mark.parametrize('problem', ['unknown', 'robot', 'task', 'geometry'])
def test_feature_contract_fails_closed(problem):
    meta = metadata(A1XAlignmentFeatures.name)
    if problem == 'unknown':
        meta.features = 'unrecognized'
    elif problem == 'robot':
        meta.robot = 'so101'
    elif problem == 'task':
        meta.task['name'] = 'reach'
    else:
        meta.feature_geometry_sha256 = 'different-model'
    with pytest.raises(ValueError, match='feature contract|geometry'):
        Policy(meta)
