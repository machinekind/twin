"""Independent gripper learning must preserve the original arm and saved contracts."""
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from lerobot_mp.twin.rl.features import A1XAlignmentFeatures
from lerobot_mp.twin.rl.imitation import Demonstrations, fit
from lerobot_mp.twin.rl.policy import Policy, PolicyMeta
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.task import make_task
from lerobot_mp.twin.robots import A1X


def fixture(features='raw', residual='none'):
    task=make_task('lift', A1X)
    meta=PolicyMeta(task=asdict(task), hidden=[32,32], obs_dim=36, act_dim=7,
        robot='a1x', features=features, randomization=asdict(Randomization.none()),
        gripper_residual=residual)
    obs=np.random.default_rng(3).uniform(-.5,.5,(120,36)).astype(np.float32)
    obs[:,17:20]=[.35,0,.015]
    obs[:,23:29]=[1,0,0,0,1,0]
    return meta,obs


@pytest.mark.parametrize('features',['raw',A1XAlignmentFeatures.name])
def test_zero_residual_preserves_baseline_batch_single_cache_reload_and_rng(tmp_path, features):
    meta,obs=fixture(features)
    torch.manual_seed(12)
    baseline=Policy(meta)
    expected_rng=torch.get_rng_state().clone()
    torch.manual_seed(12)
    residual=Policy(fixture(features,'mlp64-v1')[0])
    assert torch.equal(torch.get_rng_state(),expected_rng)
    for name,value in baseline.state_dict().items():
        torch.testing.assert_close(residual.state_dict()[name],value,atol=0,rtol=0)
    np.testing.assert_array_equal(baseline.act(obs),residual.act(obs))
    np.testing.assert_array_equal(baseline.act(obs[0]),residual.act(obs[0]))
    x=torch.as_tensor(obs)
    with torch.no_grad():
        torch.testing.assert_close(residual(x),residual(x,derived=residual.derived_features(x)),atol=0,rtol=0)
    residual.save(tmp_path/'policy.pt')
    restored=Policy.load(tmp_path/'policy.pt')
    assert restored.meta.gripper_residual=='mlp64-v1'
    np.testing.assert_array_equal(restored.act(obs),baseline.act(obs))
    # Old checkpoints without the optional metadata field still load as before.
    baseline.save(tmp_path/'legacy.pt')
    blob=torch.load(tmp_path/'legacy.pt',weights_only=False)
    del blob['meta']['gripper_residual']
    torch.save(blob,tmp_path/'legacy.pt')
    np.testing.assert_array_equal(Policy.load(tmp_path/'legacy.pt').act(obs),baseline.act(obs))


@pytest.mark.parametrize('features',['raw',A1XAlignmentFeatures.name])
def test_residual_learns_nonlinear_gripper_targets_without_changing_original_actor(tmp_path, features):
    torch.set_num_threads(2)
    torch.manual_seed(6)
    meta,obs=fixture(features)
    # A controlled nonlinear response with held-out jaw positions, not a test of
    # generalizing 90 examples across 36 independent nuisance coordinates.
    jaw=np.random.default_rng(7).permutation(np.linspace(-.9,.9,len(obs))).astype(np.float32)
    obs[:]=obs.mean(axis=0)
    obs[:,6]=jaw
    initial=Policy(meta)
    initial.norm.update(torch.as_tensor(obs[:90]))
    before={k:v.clone() for k,v in initial.state_dict().items()}
    with torch.no_grad(): targets=initial(torch.as_tensor(obs)).numpy()
    targets[:,:6]+=.2  # Contradictory arm labels must not change the frozen actor.
    targets[:,-1]+=.3*(obs[:,6]**2-.08)
    demos=Demonstrations(meta.task,meta.randomization,obs[:90],targets[:90],obs[90:],targets[90:],{})
    refined,report=fit(demos,init=initial,train_channels='gripper-residual',gripper_residual='mlp64-v1',
        updates=800,seconds=20,lr=.001,report_every=100)
    assert report['selected_validation']['gripper_mse']<report['initial_validation']['gripper_mse']*.6
    assert report['initial_validation']['arm_mse']==report['selected_validation']['arm_mse']
    for name,value in before.items():
        torch.testing.assert_close(refined.state_dict()[name],value,atol=0,rtol=0)
        torch.testing.assert_close(initial.state_dict()[name],value,atol=0,rtol=0)
    np.testing.assert_array_equal(initial.act(obs)[:,:6],refined.act(obs)[:,:6])
    refined.save(tmp_path/'policy.pt')
    restored=Policy.load(tmp_path/'policy.pt')
    np.testing.assert_array_equal(restored.act(obs),refined.act(obs))
    again,result=fit(demos,init=restored,train_channels='gripper-residual',updates=2,seconds=5)
    assert result['gripper_residual']=='mlp64-v1'
    np.testing.assert_array_equal(again.act(obs)[:,:6],initial.act(obs)[:,:6])
    with pytest.raises(ValueError,match='remove or change'):
        fit(demos,init=restored,gripper_residual='none',updates=1,seconds=1)


@pytest.mark.parametrize('problem',['name','robot','task','history'])
def test_invalid_residual_contract_fails_closed(problem):
    meta,_=fixture(residual='mlp64-v1')
    if problem=='name': meta.gripper_residual='unknown'
    if problem=='robot': meta.robot='so101'
    if problem=='task': meta.task['name']='reach'
    if problem=='history': meta.evals={'imitation':{'ignore_previous_arm':True}}
    with pytest.raises(ValueError,match='gripper residual contract'): Policy(meta)


def test_residual_cannot_be_added_through_the_linear_head_or_change_history():
    meta,obs=fixture()
    initial=Policy(meta)
    targets=initial.act(obs)
    demos=Demonstrations(meta.task,meta.randomization,obs[:90],targets[:90],obs[90:],targets[90:],{})
    for kwargs in ({'train_channels':'gripper'}, {'train_channels':'gripper-residual','ignore_previous_arm':True}):
        with pytest.raises(ValueError,match='requires gripper-residual'):
            fit(demos,init=initial,gripper_residual='mlp64-v1',updates=1,seconds=1,**kwargs)
