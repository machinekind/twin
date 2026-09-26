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


@pytest.mark.parametrize("offset", [-.001, .015, np.nan, np.inf])
def test_invalid_grasp_offsets_are_rejected_before_connecting(runtime_env, offset):
    from collect_a1x_lift_demos import trial
    with pytest.raises(ValueError, match="Grasp offset"):
        trial(runtime_env, 41001, grasp_offset_m=offset)
    assert not runtime_env.twin.connected


@pytest.mark.parametrize("seed,offset", [(41000, 0.0), (41001, 0.0), (41001, .005)])
def test_scripted_runtime_success_includes_full_physics_hold(runtime_env, seed, offset):
    from collect_a1x_lift_demos import trial
    report, obs, actions = trial(runtime_env, seed, state_triggered=True, synchronized=True,
                                 measured_close=True, lift_rate=.25, grasp_offset_m=offset)
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


def test_observable_teacher_has_no_episode_history_or_scene_contact_inputs(runtime_env):
    from a1x_observable_teacher import ObservableLiftTeacher
    teacher = ObservableLiftTeacher()
    first, _ = runtime_env.reset(seed=50007)
    expected = teacher.act(first)
    other, _ = runtime_env.reset(seed=50003)
    teacher.act(other)
    runtime_env.scene.data.sensordata[:] = 100  # Hidden physics truth must have no influence.
    np.testing.assert_array_equal(teacher.act(first), expected)
    np.testing.assert_array_equal(ObservableLiftTeacher().act(first), expected)
    for invalid in (np.zeros(35), np.full(36, np.nan), np.full(36, np.inf)):
        with pytest.raises(ValueError, match='finite A1X lift observation'):
            teacher.act(invalid)


@pytest.mark.parametrize('seed', [50007, 50003])
def test_observation_only_teacher_completes_guarded_grasp_and_full_hold(runtime_env, seed):
    from a1x_observable_teacher import observable_trial
    report, observations, labels = observable_trial(runtime_env, seed)
    assert report['success'] and report['max_continuous_hold_s'] >= 2 - 1e-8
    assert not report['collision'] and not report['numerical'] and not report['guard_rejection']
    assert report['teacher_contract'] == 'a1x-lift-observable-v1'
    assert len(observations) == len(labels) == report['steps']


def test_observable_corrections_reproduce_labels_and_target_close_or_lift(runtime_env):
    from a1x_observable_teacher import ObservableLiftTeacher, observable_trial
    teacher = ObservableLiftTeacher()
    command = np.r_[np.zeros(6), .2].astype(np.float32)
    learner = SimpleNamespace(act=lambda obs: command.copy())
    executed = []
    report, observations, labels = observable_trial(runtime_env, 50007, learner=learner,
        learner_probability=1, executed_actions=executed)
    assert not report['success'] and report['corrective_teacher_labels']
    assert 0 < report['learner_steps'] < len(labels)
    assert report['learner_gripper_steps'] == report['learner_steps']
    for obs, label, applied in zip(observations, labels, executed):
        replayed, decision = teacher.decide(obs)
        np.testing.assert_array_equal(replayed, label)
        np.testing.assert_array_equal(applied, command if decision['stage'] in ('close', 'lift') else label)


def test_collector_rejects_reserved_holdout_before_creating_output(runtime_env, monkeypatch, tmp_path, capsys):
    import collect_a1x_lift_demos
    import sys
    target = tmp_path / 'must-not-exist'
    monkeypatch.setattr(sys, 'argv', ['collect', '--out', str(target), '--seed', '51000'])
    with pytest.raises(SystemExit) as exc:
        collect_a1x_lift_demos.main()
    assert exc.value.code == 2 and 'held-out seeds' in capsys.readouterr().err
    assert not target.exists() and not runtime_env.twin.connected


def test_approach_arm_corrections_preserve_teacher_gripper(runtime_env):
    from a1x_observable_teacher import ObservableLiftTeacher, observable_trial
    teacher = ObservableLiftTeacher()
    command = np.r_[np.zeros(6), -1].astype(np.float32)
    learner = SimpleNamespace(act=lambda obs: command.copy())
    executed = []
    report, observations, labels = observable_trial(runtime_env, 47000, learner=learner,
        learner_probability=1, learner_phases=('approach',), learner_channels='arm',
        executed_actions=executed)
    assert not report['success'] and report['learner_steps'] == len(labels)
    assert report['learner_channels'] == 'arm'
    for obs, label, applied in zip(observations, labels, executed):
        replayed, decision = teacher.decide(obs)
        assert decision['stage'] == 'approach'
        np.testing.assert_array_equal(label, replayed)
        np.testing.assert_array_equal(applied[:6], command[:6])
        assert applied[-1] == label[-1] and applied[-1] != command[-1]


@pytest.mark.parametrize('phases,channels', [((), 'all'), (('close', 'close'), 'all'),
                                           (('hold',), 'all'), (('close',), 'invalid')])
def test_invalid_correction_selection_cannot_connect(runtime_env, phases, channels):
    from a1x_observable_teacher import observable_trial
    with pytest.raises(ValueError, match='phases or channels'):
        observable_trial(runtime_env, 47000, learner_phases=phases, learner_channels=channels)
    assert not runtime_env.twin.connected


def test_uninterrupted_learner_prefix_then_teacher_recovery(runtime_env):
    from a1x_observable_teacher import ObservableLiftTeacher, observable_trial
    teacher = ObservableLiftTeacher()
    command = np.r_[np.zeros(6), .2].astype(np.float32)
    executed = []
    report, observations, labels = observable_trial(runtime_env, 48000,
        learner=SimpleNamespace(act=lambda obs: command.copy()), learner_prefix_steps=20,
        learner_phases=('approach', 'close', 'lift'), executed_actions=executed)
    assert report['success'] and report['max_continuous_hold_s'] >= 2 - 1e-8
    assert report['learner_steps'] == report['learner_prefix_steps'] == 20
    assert report['learner_gripper_steps'] == 20
    assert report['execution_schedule'] == 'learner prefix then teacher'
    assert not report['scripted'] and report['corrective_teacher_labels']
    assert not report['collision'] and not report['guard_rejection']
    for index, (obs, label, applied) in enumerate(zip(observations, labels, executed)):
        np.testing.assert_array_equal(teacher.act(obs), label)
        np.testing.assert_array_equal(applied, command if index < 20 else label)


@pytest.mark.parametrize('prefix,channels,phases', [(-1, 'all', ('approach', 'close', 'lift')),
    (101, 'all', ('approach', 'close', 'lift')), (1.5, 'all', ('approach', 'close', 'lift')),
    (20, 'arm', ('approach', 'close', 'lift')), (20, 'all', ('close', 'lift'))])
def test_invalid_learner_prefix_cannot_connect(runtime_env, prefix, channels, phases):
    from a1x_observable_teacher import observable_trial
    with pytest.raises(ValueError, match='Learner prefix'):
        observable_trial(runtime_env, 48000, learner=SimpleNamespace(act=lambda obs: np.zeros(7)),
                         learner_prefix_steps=prefix, learner_channels=channels, learner_phases=phases)
    assert not runtime_env.twin.connected


@pytest.mark.parametrize('channels,allowed', [('arm', True), ('all', False)])
def test_continuous_learner_cli_requires_teacher_gripper(runtime_env, monkeypatch, tmp_path, capsys, channels, allowed):
    import collect_a1x_lift_demos
    from lerobot_mp.twin.rl.policy import Policy
    import sys
    target = tmp_path/'no-output'
    monkeypatch.setattr(sys, 'argv', ['collect', '--out', str(target), '--runtime', '--teacher', 'observable',
        '--learner', str(tmp_path/'policy.pt'), '--learner-channels', channels,
        '--learner-phases', 'approach', 'close', 'lift', '--learner-probability', '1'])
    def reached_checkpoint(*args, **kwargs):
        raise RuntimeError('accepted probability; stop before loading or connecting')
    monkeypatch.setattr(Policy, 'load', reached_checkpoint)
    if allowed:
        with pytest.raises(RuntimeError, match='accepted probability'):
            collect_a1x_lift_demos.main()
    else:
        with pytest.raises(SystemExit) as exc:
            collect_a1x_lift_demos.main()
        assert exc.value.code == 2 and 'arm-only control permits' in capsys.readouterr().err
    assert not target.exists() and not runtime_env.twin.connected


def test_closing_gripper_burst_records_true_learner_history_then_teacher_recovery(runtime_env):
    from a1x_observable_teacher import ObservableLiftTeacher, observable_trial
    teacher = ObservableLiftTeacher()
    # Keep accurate arms but deliberately hold the gripper open during its burst.
    learner = SimpleNamespace(act=lambda obs: np.r_[teacher.act(obs)[:6], .2].astype(np.float32))
    executed = []
    report, obs, labels = observable_trial(runtime_env, 49100, learner=learner,
        learner_probability=1, learner_channels='arm', learner_phases=('approach', 'close', 'lift'),
        learner_gripper_burst_steps=20, executed_actions=executed)
    assert report['success'] and report['max_continuous_hold_s'] >= 2-1e-8
    assert report['learner_gripper_steps'] == 20
    assert report['learner_steps'] == len(obs)
    start = report['gripper_burst_start_step']
    assert start is not None and teacher.decide(obs[start])[1]['stage'] == 'close'
    for i, (x, target, action) in enumerate(zip(obs, labels, executed)):
        np.testing.assert_array_equal(teacher.act(x), target)
        np.testing.assert_array_equal(action[:6], learner.act(x)[:6])
        assert action[-1] == (learner.act(x)[-1] if start <= i < start+20 else target[-1])
        if i:
            np.testing.assert_array_equal(x[-7:], executed[i-1])
    assert not report['collision'] and not report['guard_rejection'] and not report['numerical']


@pytest.mark.parametrize('burst,probability,channels,prefix', [(-1,1,'arm',0), (51,1,'arm',0),
    (1.5,1,'arm',0), (20,.3,'arm',0), (20,1,'all',0), (20,1,'arm',20)])
def test_invalid_gripper_burst_cannot_connect(runtime_env,burst,probability,channels,prefix):
    from a1x_observable_teacher import observable_trial
    with pytest.raises(ValueError,match='Gripper burst|Learner prefix'):
        observable_trial(runtime_env,49100,learner=SimpleNamespace(act=lambda obs:np.zeros(7)),
            learner_probability=probability,learner_channels=channels,learner_prefix_steps=prefix,
            learner_phases=('approach','close','lift'),learner_gripper_burst_steps=burst)
    assert not runtime_env.twin.connected
