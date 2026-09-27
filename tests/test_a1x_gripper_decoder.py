"""Explicit finite gripper actions preserve actor weights, arm outputs and legacy policies."""
from dataclasses import asdict, replace
from types import MethodType

import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

from lerobot_mp.twin.rl.policy import Policy, PolicyMeta
from lerobot_mp.twin.rl.task import make_task
from lerobot_mp.twin.robots import A1X


def metadata(decoder="a1x-templates-v1"):
    task = make_task("lift", A1X)
    return PolicyMeta(task=asdict(task), hidden=[8], obs_dim=36, act_dim=7,
                      robot="a1x", gripper_decoder=decoder)


def test_weak_closure_is_decoded_using_actor_choice_and_measured_jaw_only():
    policy = Policy(metadata())
    observations = np.zeros((6, 36), np.float32)
    observations[:, 6] = [-.36, -.36, -.36, -.98, .8, -.36]
    predictions = np.zeros((6, 7), np.float32)
    predictions[:, :6] = np.arange(6)/10
    predictions[:, 6] = [-.368, .18, -.99, -.999, .78, -.1]
    policy.forward = MethodType(lambda self,x: torch.as_tensor(predictions), policy)
    before = observations.copy()
    actual = policy.act(observations)
    np.testing.assert_array_equal(actual[:, :6], predictions[:, :6])
    expected = np.asarray([float(observations[0,6])-.04, .2, -1., -1.,
                           float(observations[4,6])-.04, .2], np.float32)
    np.testing.assert_array_equal(actual[:, 6], expected)
    np.testing.assert_array_equal(observations, before)
    # No arm pose, history, cube pose or teacher decision enters the decoder.
    observations[:, 17:29] = 7
    np.testing.assert_array_equal(policy.act(observations), actual)


def test_saved_mode_roundtrip_and_legacy_default_preserve_all_tensors(tmp_path):
    torch.manual_seed(31)
    original = Policy(metadata("identity"))
    torch.manual_seed(31)
    decoded = Policy(metadata())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, decoded.state_dict()[key], atol=0, rtol=0)
    x = np.random.default_rng(0).normal(size=(8,36)).astype(np.float32)
    np.testing.assert_array_equal(original.act(x)[:, :6], decoded.act(x)[:, :6])
    path = decoded.save(tmp_path / 'policy.pt')
    loaded = Policy.load(path)
    assert loaded.meta.gripper_decoder == 'a1x-templates-v1'
    np.testing.assert_array_equal(loaded.act(x), decoded.act(x))
    np.testing.assert_allclose(loaded.act(x[0]), loaded.act(x[:1])[0], rtol=1e-6, atol=1e-7)
    blob = torch.load(path, weights_only=False)
    del blob['meta']['gripper_decoder']
    torch.save(blob, path)
    legacy = Policy.load(path)
    assert legacy.meta.gripper_decoder == 'identity'
    np.testing.assert_array_equal(legacy.act(x), original.act(x))


@pytest.mark.parametrize('changes', [dict(gripper_decoder='unknown'), dict(robot='so101'), dict(obs_dim=30),
    dict(act_dim=6), dict(task={**metadata().task, 'cube_half': .02}), dict(gripper_residual='mlp64-v1')])
def test_unsupported_decoder_contract_is_rejected(changes):
    with pytest.raises(ValueError, match='gripper decoder contract'):
        Policy(replace(metadata(), **changes))


@pytest.mark.parametrize('corrupt', ['observation', 'prediction'])
def test_decoding_cannot_hide_nonfinite_inputs_or_predictions(corrupt):
    policy = Policy(metadata())
    x = np.zeros(36, np.float32)
    prediction = torch.zeros((1,7))
    if corrupt == 'observation':
        x[6] = np.nan
    else:
        prediction[0,6] = torch.nan
    policy.forward = MethodType(lambda self,x: prediction, policy)
    with pytest.raises(ValueError, match='finite A1X'):
        policy.act(x)


def test_refinement_cannot_silently_discard_action_decoder(monkeypatch):
    from lerobot_mp.twin.rl.imitation import Demonstrations, fit
    from lerobot_mp.twin.rl import ppo
    from lerobot_mp.twin.workspace import Workspace
    policy = Policy(metadata())
    x, y = np.zeros((4,36), np.float32), np.zeros((4,7), np.float32)
    data = Demonstrations(policy.meta.task, {}, x, y, x, y, {})
    with pytest.raises(ValueError, match='decoded gripper action'):
        fit(data, init=policy, updates=1, seconds=1)
    def no_environment(*args, **kwargs):
        pytest.fail('Unsupported action decoder must fail before creating the environment')
    monkeypatch.setattr(ppo, 'BatchEnv', no_environment)
    with pytest.raises(ValueError, match='decoded gripper action'):
        ppo.train(policy.task, init=policy, workspace=Workspace(robot='a1x'), device='cpu')


def test_decoder_keeps_action_bounds_when_measurement_exceeds_nominal_limits():
    policy = Policy(metadata())
    x = np.zeros((2,36), np.float32)
    x[:,6] = [2, -2]
    prediction = torch.zeros((2,7))
    prediction[:,6] = torch.tensor([1., -1.])
    policy.forward = MethodType(lambda self,x: prediction, policy)
    result = policy.act(x)
    np.testing.assert_array_equal(result[:,6], [1., -1.])
    assert np.max(np.abs(result)) <= 1
