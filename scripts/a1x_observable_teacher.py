"""Stateless A1X lift teacher using robot constants and the actor's 36 inputs only.

Simulation demonstration generator, not a deployment controller or a learned policy.
"""
import numpy as np

from lerobot_mp.twin.kinematics import RobotKinematics
from lerobot_mp.twin.rl.task import Limits, make_task
from lerobot_mp.twin.robots import A1X


class ObservableLiftTeacher:
    contract = 'a1x-lift-observable-v1'

    def __init__(self):
        self.task = make_task('lift', A1X)
        self.kin = RobotKinematics(A1X)
        self.limits = Limits.of(self.kin)
        self.kin.lo, self.kin.hi = self.limits.lo.copy(), self.limits.hi.copy()
        home = self.kin.tcp(A1X.home)
        centre = np.mean([self.kin.data.geom_xpos[self.kin.model.geom(n).id]
                          for n in A1X.fingertips], axis=0)
        self.offset = home[:3, :3].T @ (centre - home[:3, 3])

    def decide(self, obs):
        """No scene state, touch forces, elapsed time or waypoint memory is available here."""
        obs = np.asarray(obs, dtype=float)
        if obs.shape != (self.task.obs_dim,) or not np.isfinite(obs).all():
            raise ValueError('Expected one finite A1X lift observation')
        lo, hi = self.limits.lo, self.limits.hi
        q, command = [lo + .5 * (obs[i:i+7] + 1) * (hi-lo) for i in (0, 7)]
        cube = obs[17:20]
        rotation = np.column_stack((obs[23:26], obs[26:29], np.cross(obs[23:26], obs[26:29])))
        desired_R = rotation @ np.column_stack(([0, 0, -1], [0, 1, 0], [1, 0, 0]))
        measured = self.kin.tcp(self.kin.from_q(q))
        tips = measured[:3, 3] + measured[:3, :3] @ self.offset
        commanded_pose = self.kin.tcp(self.kin.from_q(command))
        commanded_tips = commanded_pose[:3, 3] + commanded_pose[:3, :3] @ self.offset
        commanded_lateral = float(np.linalg.norm(commanded_tips[:2] - cube[:2]))
        commanded_vertical = float(commanded_tips[2] - (self.task.cube_half + .005))
        lateral = float(np.linalg.norm(tips[:2] - cube[:2]))
        vertical = float(tips[2] - (cube[2] + .005))
        angle = float(np.arccos(np.clip((np.trace(desired_R.T @ measured[:3, :3])-1)/2, -1, 1)))
        # Nominal model's empty inner-pad gap is about 3 mm. This is a provisional
        # geometric contact proxy, never an independent success/contact detector.
        expected_jaw = (2*self.task.cube_half - .003) / 2
        lag = float(q[-1] - command[-1])
        # Gravity/tracking offsets prevent measured pose from reaching an exact
        # kinematic waypoint. Require command arrival AND a bounded measured gap.
        near = (commanded_lateral < .003 and abs(commanded_vertical) < .003
                and lateral < .010 and abs(vertical) < .010 and angle < .10)
        # Previous action is already an actor input: use it to continue closing
        # without a hidden phase latch or alternating open/close at the boundary.
        closing = obs[-1] < .19 and lateral < .010 and abs(vertical) < .014 and angle < .12
        # A provisional position-only blockage proxy, independent of touch sensors.
        blocked = q[-1] < expected_jaw + .0005 and q[-1] > .003 and lag > .0006
        holding = command[-1] < .006 and q[-1] > .003 and q[-1] < expected_jaw + .002 and lag > .003
        target_tips = cube.copy()
        target_tips[2] = self.task.cube_half + .005
        if holding:
            stage, rate, opening = 'lift', .25, 0.
            # Preserve commanded XY/orientation; chasing the held block's XY can
            # integrate grasp offsets into arm drift. Only observed height drives lift.
            desired_R = commanded_pose[:3, :3]
            target_tips = commanded_pose[:3, 3] + desired_R @ self.offset
            target_tips[2] += np.clip(.2 * (.07 - (cube[2] - self.task.cube_half)), -.0025, .0025)
        elif near or closing:
            stage, rate = 'close', .5
            opening = 0. if blocked else max(lo[-1], q[-1] - .001)
        else:
            stage, rate, opening = 'approach', 1., .03
            target_tips[2] += .075 * np.clip(commanded_lateral / .03, 0, 1)
        target = target_tips - desired_R @ self.offset
        solution = self.kin.ik(target, desired_R, seed=self.kin.from_q(command), restarts=0,
                               tol=.0001 if holding else .001)
        target_q = self.kin.to_q(solution.joints)
        if (not solution.ok or solution.rot_err > .10 or
                (target_q < lo-1e-6).any() or (target_q > hi+1e-6).any()):
            raise RuntimeError(f'Observable teacher IK: {solution.pos_err:.5f} m / {solution.rot_err:.4f} rad')
        delta = (target_q[:6] - command[:6]) / self.task.arm_step
        action = np.r_[delta / max(1., np.abs(delta).max() / rate),
                       2 * (opening-lo[-1])/(hi[-1]-lo[-1]) - 1]
        info = dict(stage=stage, lateral_mm=lateral*1000, vertical_mm=vertical*1000,
                    commanded_lateral_mm=commanded_lateral*1000, commanded_vertical_mm=commanded_vertical*1000,
                    rotation_error_deg=float(np.degrees(angle)), measured_jaw_m=float(q[-1]),
                    commanded_jaw_m=float(command[-1]), blockage_proxy=bool(blocked),
                    holding_proxy=bool(holding), target_tips=target_tips.tolist())
        return action.astype(np.float32), info

    def act(self, obs):
        return self.decide(obs)[0]


def observable_trial(env, seed, *, teacher=None, learner=None, learner_probability=.2,
                     executed_actions=None, learner_phases=('close', 'lift'), learner_channels='all',
                     learner_prefix_steps=0):
    """Correct grasp/lift deviations; evaluate via independent guarded physics.

    Default interventions target close/lift; an explicit phase selection also
    permits approach corrections when diagnostic traces justify them.
    A nonzero prefix executes only the learner for that many steps, then only
    the teacher. It requires all phases/channels and leaves at least half the
    episode for recovery; ordinary guarded termination still applies throughout.
    Labels always come from the observation-only teacher. This is not autonomous
    learned evaluation, even when many executed actions belong to the learner.
    """
    teacher = teacher or ObservableLiftTeacher()
    if env.task != teacher.task:
        raise ValueError('Observable teacher must match the nominal A1X lift task')
    if (not learner_phases or len(set(learner_phases)) != len(learner_phases)
            or not set(learner_phases) <= {'approach', 'close', 'lift'}
            or learner_channels not in ('all', 'arm')):
        raise ValueError('Invalid learner phases or channels')
    if (type(learner_prefix_steps) is not int or not 0 <= learner_prefix_steps <= env.task.episode_steps // 2
            or (learner_prefix_steps and (learner is None or learner_channels != 'all'
                or set(learner_phases) != {'approach', 'close', 'lift'}))):
        raise ValueError('Learner prefix requires all phases/channels, a learner, and at least half an episode for recovery')
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9127]))
    observations, labels = [], []
    report = dict(seed=seed, scripted=learner is None, success=False, collision=False,
                  numerical=False, steps=0, learner_steps=0, phase_counts={},
                  corrective_teacher_labels=learner is not None, teacher_contract=teacher.contract,
                  learner_phases=list(learner_phases), learner_channels=learner_channels,
                  learner_prefix_steps=learner_prefix_steps,
                  execution_schedule='learner prefix then teacher' if learner_prefix_steps else 'interleaved',
                  max_coupling_mm=0.0)
    try:
        for _ in range(env.task.episode_steps):
            label, decision = teacher.decide(obs)
            phase = decision['stage']
            report['phase_counts'][phase] = report['phase_counts'].get(phase, 0) + 1
            applied = label
            use_learner = (len(labels) < learner_prefix_steps if learner_prefix_steps else
                           phase in report['learner_phases'] and rng.random() < learner_probability)
            if learner is not None and use_learner:
                applied = learner.act(obs)
                if learner_channels == 'arm':
                    applied = np.r_[applied[:6], label[-1]].astype(np.float32)
                report['learner_steps'] += 1
            observations.append(obs.copy())
            labels.append(label.copy())
            if executed_actions is not None:
                executed_actions.append(np.asarray(applied).copy())
            obs, _, term, trunc, info = env.step(applied)
            report.update(phase=phase, steps=len(labels), final_height_mm=info['height'] * 1000,
                          collision=info['unsafe'], numerical=info['broken'],
                          max_coupling_mm=info['max_coupling_mm'], runtime_stop=info['runtime_stop'],
                          guard_rejection=info['guard_rejection'], max_continuous_hold_s=info['max_continuous_hold_s'])
            if term or trunc:
                report['success'] = bool(info['finished'] and not term)
                break
    except RuntimeError as exc:
        report['failure'] = str(exc)
    if not report['success'] and 'failure' not in report:
        report['failure'] = 'guard rejection, numerical failure or incomplete hold by time limit'
    return report, observations, labels
