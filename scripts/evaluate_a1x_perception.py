"""Bounded, teacher-free pose-error evaluation through the guarded A1X controller.

Each output directory freezes its protocol before execution and retains partial
traces on interruption. Development results are not held-out qualification.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import traceback
import zipfile

import numpy as np
import torch

from a1x_runtime_env import RuntimeLiftEnv
from lerobot_mp.twin.rl.perception import PoseCorruption
from lerobot_mp.twin.rl.policy import Policy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/a1x_perception_robustness.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def conditions(translation_mm=(), rotation_deg=(), position_noise_mm=(), rotation_noise_deg=(), delay_ms=()):
    result = [dict(id='zero', factor='zero', magnitude=0., direction=None, corruption=asdict(PoseCorruption()))]
    for factor, values, field, scale in (
        ('translation_mm', translation_mm, 'translation_m', .001),
        ('rotation_deg', rotation_deg, 'rotation_vector_deg', 1.),
        ('position_noise_mm', position_noise_mm, 'position_noise_std_m', .001),
        ('rotation_noise_deg', rotation_noise_deg, 'rotation_noise_std_deg', 1.),
        ('delay_ms', delay_ms, 'delay_s', .001)):
        if len(set(values)) != len(values) or any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Magnitudes must be unique, finite and positive')
        for value in values:
            directions = [(i, sign) for i in range(3) for sign in (1, -1)] if factor in (
                'translation_mm', 'rotation_deg') else [(None, 1)]
            for axis, sign in directions:
                setting = value * scale
                direction = None
                if axis is not None:
                    setting = [0., 0., 0.]
                    setting[axis] = sign * value * scale
                    direction = ('p' if sign == 1 else 'n') + 'xyz'[axis]
                result.append(dict(id=f'{factor}-{value:g}' + (f'-{direction}' if direction else ''),
                    factor=factor, magnitude=float(value), direction=direction,
                    corruption=asdict(PoseCorruption(**{field: setting}))))
    return result


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p, z2 = successes / total, z*z
    centre = (p + z2 / (2*total)) / (1 + z2/total)
    half = z * np.sqrt(p*(1-p)/total + z2/(4*total*total)) / (1 + z2/total)
    return [max(0., float(centre-half)), min(1., float(centre+half))]


def summary(trials, condition_list, expected):
    result = {}
    for condition in condition_list:
        rows = [r for r in trials if r['condition'] == condition['id']]
        successes = sum(r['success'] for r in rows)
        result[condition['id']] = dict(episodes=len(rows), expected_episodes=expected, successes=successes,
            complete=len(rows) == expected, success_fraction=successes/len(rows) if rows else None,
            wilson_95=wilson(successes, len(rows)),
            controller_faults=sum(r['final']['guard_rejection'] for r in rows),
            prohibited_contacts=sum(r['final']['unsafe'] for r in rows),
            numerical_failures=sum(r['final']['broken'] for r in rows),
            premature_controller_successes=sum(r['premature_controller_success'] for r in rows),
            failure_categories=dict(Counter(r['failure_category'] for r in rows if not r['success'])))
    return result


def trial(policy, env, seed, condition, path, deadline):
    # The same independent perception RNG seed is paired across conditions.
    perception_seed = int(np.random.SeedSequence([seed, 73129]).generate_state(1)[0])
    obs, _ = env.reset(seed=seed, perception_seed=perception_seed)
    first_bilateral = None
    max_height = 0.
    ages = []
    warmup_frames = 0
    with path.open('x') as trace:
        for step in range(policy.task.episode_steps):
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic wall-time ceiling reached; partial trace retained')
            state = env.state()
            if first_bilateral is None and min(state['jaws']) > .5:
                first_bilateral = step
            action = policy.act(obs)
            perception = env.perception.last_log
            ages.append(perception['effective_delay_s'])
            warmup_frames += int(perception['delay_warmup'])
            row = dict(step=step, perception=perception, observation=obs.tolist(), action=action.tolist(),
                       true_jaws=state['jaws'].tolist())
            obs, _, term, trunc, info = env.step(action)
            row['after'] = info
            trace.write(json.dumps(row, allow_nan=False) + '\n')
            max_height = max(max_height, info['height'])
            if first_bilateral is None and info['grasped']:
                first_bilateral = step + 1
            if term or trunc:
                break
    premature = info['runtime_stop'] == 'zadanie wykonane' and not info['finished']
    category = ('success' if info['finished'] else 'numerical' if info['broken'] else
                'prohibited-contact' if info['unsafe'] else
                'controller-fault-before-grasp' if info['guard_rejection'] and first_bilateral is None else
                'controller-fault-after-grasp' if info['guard_rejection'] else
                'no-bilateral-grasp' if first_bilateral is None else
                'incomplete-true-hold')
    return dict(condition=condition['id'], seed=seed, perception_seed=perception_seed, steps=step+1,
        success=info['finished'], first_bilateral_step=first_bilateral, max_true_height_m=max_height,
        premature_controller_success=premature, failure_category=category, final=info,
        last_measured=dict(env.twin.status.measured), last_command=dict(env.twin.status.command),
        effective_delay_s_range=[min(ages), max(ages)], delay_warmup_frames=warmup_frames,
        contact_penetration_mm=env.monitor.contacts, trace=path.name, trace_sha256=sha(path))


def check_reference(trials, reference):
    def cause(reason):
        # The standalone evaluator catches the exception, whereas the adapter
        # reports the supervisor's stopped status. Preserve the underlying guard
        # cause while ignoring these different exception/status prefixes.
        marker = 'A1X motion rejected:'
        return reason[reason.index(marker):] if marker in reason else reason
    previous = {row['seed']: row for row in reference['episodes']}
    for row in trials:
        old = previous[row['seed']]
        if (row['success'] != old['success'] or cause(row['final']['runtime_stop']) != cause(old['stop_reason'])
                or row['final']['max_continuous_hold_s'] != old['max_continuous_hold_s']
                or row['final']['unsafe'] != old['prohibited_contacts']
                or row['final']['broken'] != old['numerical_failure']
                or row['final']['max_coupling_mm'] != old['max_jaw_coupling_mm']
                or not np.isclose(row['final']['height']*1000, old['final_height_mm'], atol=1e-9, rtol=0)
                or row['last_measured'] != old['last_measured']
                or row['last_command'] != old['last_command']):
            raise RuntimeError(f"Zero-error runtime differs from saved baseline at seed {row['seed']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('policy', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=('development', 'held_out'), default='development')
    parser.add_argument('--seconds', type=int, default=600)
    for flag in ('translation-mm', 'rotation-deg', 'position-noise-mm', 'rotation-noise-deg', 'delay-ms'):
        parser.add_argument('--'+flag, type=float, nargs='+', default=[])
    args = parser.parse_args()
    try:
        batch = conditions(args.translation_mm, args.rotation_deg, args.position_noise_mm,
                           args.rotation_noise_deg, args.delay_ms)
    except ValueError as exc:
        parser.error(str(exc))
    if len(batch) > 16 or not 1 <= args.seconds <= 600:
        parser.error('Diagnostic limit: 16 conditions, 600 wall seconds; use separate bounded comparisons')
    config = json.loads(CONFIG.read_text())
    seeds = config[args.split+'_seeds']
    if len(seeds) != 20 or len(set(seeds)) != 20:
        parser.error('Expected 20 unique predeclared seeds')
    checkpoint_sha = sha(args.policy)
    policy = Policy.load(args.policy)
    if (policy.meta.robot != 'a1x' or policy.task.name != 'lift' or policy.task.lift_height != .06
            or policy.task.end_on_success / policy.task.control_hz < 2):
        parser.error('Expected an A1X lift checkpoint with a 60 mm lift and at least a two-second hold')
    reference = None
    reference_path = ROOT / 'workspace/validation/baselines/lift-imitation-027/development-runtime.json'
    if args.split == 'development' and checkpoint_sha == config['baseline_policy_sha256']:
        reference = json.loads(reference_path.read_text())
        if reference['policy_sha256'] != checkpoint_sha:
            parser.error('Baseline reference policy identity does not match')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source_paths = sorted((ROOT / 'src').rglob('*.py')) + [ROOT / p for p in (
        'scripts/evaluate_a1x_perception.py', 'scripts/evaluate_a1x_runtime.py', 'scripts/a1x_runtime_env.py',
        'configs/a1x_perception_robustness.json', 'assets/robots/a1x/a1x.xml')]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    with zipfile.ZipFile(args.out / 'source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, str(path.relative_to(ROOT)))
    protocol = dict(policy=str(args.policy), policy_sha256=checkpoint_sha, split=args.split, seeds=seeds,
        conditions=batch, configuration=config, teacher_intervention=False, simulation_only=True,
        path='PolicyRunner -> Twin -> SafetySupervisor -> SceneBackend', max_seconds=args.seconds,
        cube_yaw_mode=policy.meta.cube_yaw_mode, gripper_decoder=policy.meta.gripper_decoder,
        cube_height_error_bound_m=policy.meta.cube_height_error_bound_m,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'mujoco', 'numpy')},
        git_head=subprocess.check_output(['git', '-c', f'safe.directory={ROOT}', 'rev-parse', 'HEAD'],
                                         cwd=ROOT, text=True).strip(),
        source_sha256=hashes, source_archive_sha256=sha(args.out / 'source.zip'),
        zero_reference_sha256=sha(reference_path) if reference is not None else None,
        qualification='Development comparison; held-out status must be audited separately before any claim')
    write_json(args.out / 'protocol.json', protocol)
    started = time.monotonic()
    trials, status, error, zero_verified = [], 'running', None, False
    try:
        for condition in batch:
            env = RuntimeLiftEnv(policy.task, cube_yaw_mode=policy.meta.cube_yaw_mode,
                                 pose_corruption=PoseCorruption(**condition['corruption']),
                                 cube_height_error_bound_m=policy.meta.cube_height_error_bound_m)
            try:
                for seed in seeds:
                    path = args.out / f"{condition['id']}-seed-{seed}.jsonl"
                    result = trial(policy, env, seed, condition, path, started+args.seconds)
                    trials.append(result)
                    write_json(args.out / 'progress.json', dict(status='running', trials=trials,
                        summary=summary(trials, batch, len(seeds)), elapsed_s=time.monotonic()-started))
            finally:
                env.close()
            if condition['id'] == 'zero' and reference is not None:
                check_reference(trials, reference)
                zero_verified = True
            print(json.dumps({condition['id']: summary(trials, [condition], len(seeds))[condition['id']]}), flush=True)
        if sha(args.policy) != checkpoint_sha or any(sha(ROOT / p) != h for p, h in hashes.items()):
            raise RuntimeError('Policy or active evaluation source changed during the run')
        status = 'complete'
    except BaseException:
        status, error = 'interrupted-or-failed', traceback.format_exc()
        raise
    finally:
        report = dict(status=status, error=error, protocol=protocol, trials=trials,
            summary=summary(trials, batch, len(seeds)), elapsed_s=time.monotonic()-started,
            zero_matches_saved_baseline=zero_verified,
            protocol_sha256=sha(args.out / 'protocol.json'))
        write_json(args.out / 'report.json', report)


if __name__ == '__main__':
    main()
