"""Persistent calibration transforms, independent jitter, causal delay and true scoring."""
from pathlib import Path

import numpy as np
import pytest

from lerobot_mp.twin.rl.perception import PoseCorruption, PoseCorruptor, rotation_vector_matrix


def test_bias_rotates_position_about_base_origin_and_composes_orientation():
    config = PoseCorruption(translation_m=(.01, -.02, .03), rotation_vector_deg=(0, 0, 90))
    stream = PoseCorruptor(config, seed=7)
    p, r = np.array([.4, 0, .015]), rotation_vector_matrix([90,0,0])
    before = p.copy(), r.copy()
    actual_p, actual_r = stream.sample(0, p, r)
    np.testing.assert_allclose(actual_p, [.01, .38, .045], atol=1e-15)
    np.testing.assert_allclose(actual_r, rotation_vector_matrix([0,0,90]) @ r, atol=1e-15)
    np.testing.assert_array_equal(p, before[0])
    np.testing.assert_array_equal(r, before[1])
    next_p, next_r = stream.sample(.05, p, r)
    np.testing.assert_array_equal(actual_p, next_p)
    np.testing.assert_array_equal(actual_r, next_r)


def test_delay_uses_old_frame_before_bias_and_reports_warmup_and_quantization():
    stream = PoseCorruptor(PoseCorruption(translation_m=(.1,0,0), delay_s=.05), seed=0)
    for time in [0, .04, .1, .14]:
        p, _ = stream.sample(time, np.array([time,0,.015]), np.eye(3))
        expected_source = {0:0, .04:0, .1:.04, .14:.04}[time]
        assert stream.last_log['source_time_s'] == expected_source
        assert stream.last_log['effective_delay_s'] == pytest.approx(time-expected_source)
        assert stream.last_log['delay_warmup'] == (time < .05)
        assert p[0] == pytest.approx(expected_source+.1)


def test_noise_is_reproducible_and_repeated_reads_do_not_resample():
    config = PoseCorruption(position_noise_std_m=.002, rotation_noise_std_deg=.5)
    a, b = PoseCorruptor(config, seed=18), PoseCorruptor(config, seed=18)
    p, r = np.array([.35,.05,.015]), np.eye(3)
    first = a.sample(0,p,r)
    repeated = a.sample(0,p,r)
    repeated[0][:] = 99  # Returned arrays cannot mutate the cached observation.
    for actual, expected in zip(a.sample(0,p,r), first):
        np.testing.assert_array_equal(actual, expected)
    for time in [0,.05,.1]:
        for actual, expected in zip(a.sample(time,p,r), b.sample(time,p,r)):
            np.testing.assert_array_equal(actual, expected)
    c = PoseCorruptor(PoseCorruption(position_noise_std_m=.002), seed=18)
    cp, cr = c.sample(0,p,r)
    np.testing.assert_array_equal(cp, first[0])  # Orientation noise uses a separate RNG stream.
    np.testing.assert_array_equal(cr, r)
    np.testing.assert_allclose(first[1].T @ first[1], np.eye(3), atol=1e-15)
    assert not np.array_equal(first[1], r)


@pytest.mark.parametrize('kwargs', [dict(translation_m=[0,0]), dict(rotation_vector_deg=[0,0,np.nan]),
    dict(delay_s=-.1), dict(delay_s=np.inf), dict(position_noise_std_m=-1), dict(rotation_noise_std_deg=np.nan)])
def test_invalid_corruption_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PoseCorruption(**kwargs)


def test_invalid_or_out_of_order_samples_cannot_advance_history():
    stream = PoseCorruptor(PoseCorruption(), seed=0)
    p, r = np.array([.3,0,.015]), np.eye(3)
    stream.sample(.1,p,r)
    for time, pos, rot in [(.05,p,r), (.1,p+.01,r), (.2,p,np.zeros((3,3))), (.2,p*np.nan,r)]:
        with pytest.raises(ValueError):
            stream.sample(time,pos,rot)
    np.testing.assert_array_equal(stream.sample(.1,p,r)[0], p)
    assert len(stream.frames)==1


def runtime_env(monkeypatch, corruption=None):
    pytest.importorskip('mujoco')
    pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from a1x_runtime_env import RuntimeLiftEnv
    from lerobot_mp.twin.rl.task import make_task
    from lerobot_mp.twin.robots import A1X
    return RuntimeLiftEnv(make_task('lift', A1X), cube_yaw_mode='continuous', pose_corruption=corruption)


def test_zero_corruption_reproduces_actual_runtime_observations_and_physics(monkeypatch):
    plain, wrapped = runtime_env(monkeypatch), runtime_env(monkeypatch, PoseCorruption())
    try:
        a,_ = plain.reset(seed=41001)
        b,_ = wrapped.reset(seed=41001)
        np.testing.assert_array_equal(a,b)
        for _ in range(5):
            action = np.zeros(7, np.float32)
            action[-1] = a[6]
            a,*_,a_info = plain.step(action)
            b,*_,b_info = wrapped.step(action)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(plain.scene.data.qpos, wrapped.scene.data.qpos)
            assert a_info == b_info
    finally:
        plain.close()
        wrapped.close()


def test_bias_is_installed_before_continuous_yaw_initialization(monkeypatch):
    from lerobot_mp.twin.rl.task import fold_yaw
    plain = runtime_env(monkeypatch)
    biased = None
    try:
        plain.reset(seed=41001)
        true = plain.state()
        yaw = np.degrees(np.arctan2(fold_yaw(np,true['cube_rot'])[1,0], fold_yaw(np,true['cube_rot'])[0,0]))
        angle = (46 if yaw >= 0 else -46) - yaw
        config = PoseCorruption(rotation_vector_deg=(0,0,angle))
        biased = runtime_env(monkeypatch, config)
        observed,_ = biased.reset(seed=41001)
        expected = fold_yaw(np, rotation_vector_matrix(config.rotation_vector_deg) @ true['cube_rot'])
        np.testing.assert_array_equal(observed[23:29], expected[:,:2].T.reshape(-1).astype(np.float32))
        np.testing.assert_array_equal(biased.state()['cube_pos'],true['cube_pos'])
    finally:
        plain.close()
        if biased is not None:
            biased.close()


def test_biased_controller_completion_cannot_count_as_true_lift(monkeypatch):
    env = runtime_env(monkeypatch, PoseCorruption(translation_m=(0,0,.2)))
    try:
        obs,_ = env.reset(seed=41001)
        # Deliberately make the estimated-grasp check permissive to exercise a
        # false controller success. Independent true contact/height stays intact.
        monkeypatch.setattr(env.runner,'_jaw_holding',lambda q: True)
        env.runner.grip_creep = float('inf')
        for _ in range(60):
            action = np.zeros(7,np.float32)
            action[-1] = obs[6]
            obs,_,term,trunc,info = env.step(action)
            if term or trunc:
                break
        assert info['runtime_stop']=='zadanie wykonane'
        assert info['max_continuous_hold_s']==0
        assert not info['finished'] and not info['unsafe'] and not info['broken']
        assert env.state()['cube_pos'][2] < .02
    finally:
        env.close()


def test_evaluation_keeps_signed_axes_separate_and_pairs_zero(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from evaluate_a1x_perception import conditions, summary, wilson
    cases = conditions([2], [.5], [1], [.5], [50])
    assert len(cases) == 16 and cases[0]['id'] == 'zero'
    translations = [c for c in cases if c['factor'] == 'translation_mm']
    assert {c['direction'] for c in translations} == {'px','nx','py','ny','pz','nz'}
    assert {tuple(c['corruption']['translation_m']) for c in translations} == {
        (.002,0,0),(-.002,0,0),(0,.002,0),(0,-.002,0),(0,0,.002),(0,0,-.002)}
    assert all(not row['complete'] and row['success_fraction'] is None
               for row in summary([], cases, 20).values())
    assert wilson(20,20)[0] == pytest.approx(.83887484)
    for values in ([0], [-1], [1,1], [float('nan')]):
        with pytest.raises(ValueError):
            conditions(translation_mm=values)


def test_zero_reference_comparison_rejects_changed_true_hold(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from evaluate_a1x_perception import check_reference
    previous = dict(seed=1, success=True, stop_reason='zadanie wykonane', max_continuous_hold_s=2.01,
                    prohibited_contacts=False, numerical_failure=False)
    result = dict(seed=1, success=True, final=dict(runtime_stop='zadanie wykonane',
                  max_continuous_hold_s=1.99, unsafe=False, broken=False))
    with pytest.raises(RuntimeError, match='baseline'):
        check_reference([result], dict(episodes=[previous]))
