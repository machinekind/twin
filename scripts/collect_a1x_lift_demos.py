"""Collect bounded scripted A1X lift demonstrations for a possible imitation warm start.

These are simulation demonstrations, never learned-policy or hardware successes.
Failed attempts are retained; only successful episodes may train an imitator.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from lerobot_mp.twin.rl.env import TwinEnv
from lerobot_mp.twin.rl.perception import PoseCorruption
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.task import fold_yaw, make_task
from lerobot_mp.twin.robots import A1X
from lerobot_mp.twin.workspace import Workspace


def trial(env, seed, *, state_triggered=False, synchronized=False, measured_close=False, lift_rate=1.0,
          learner=None, learner_probability=.2, executed_actions=None, grasp_offset_m=0.0):
    if not np.isfinite(grasp_offset_m) or not 0 <= grasp_offset_m < env.task.cube_half:
        raise ValueError("Grasp offset must be finite, nonnegative and below the cube half-height")
    obs, _ = env.reset(seed=seed)
    state = env.state()
    cube = state["cube_pos"].copy()
    yaw = fold_yaw(np, state["cube_rot"])
    kin, m, d = env.kin, env.scene.model, env.scene.data
    current = env.scene.joints()
    kin.tcp(current)
    midpoint = np.mean([kin.data.geom_xpos[m.geom(kin.prefix + name).id] for name in A1X.fingertips], axis=0)
    offset = kin.data.site_xmat[kin.site_id].reshape(3, 3).T @ (midpoint - kin.data.site_xpos[kin.site_id])
    report = {"seed": seed, "scripted": learner is None, "success": False, "cube_m": cube.tolist(),
              "collision": False, "numerical": False, "max_coupling_mm": 0.0,
              "state_triggered": state_triggered, "synchronized": synchronized,
              "measured_close": measured_close, "lift_rate": lift_rate, "phase_ends": {},
              "corrective_teacher_labels": learner is not None, "learner_steps": 0,
              "grasp_offset_m": grasp_offset_m}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9127]))
    observations, actions = [], []
    # A square block permits four equivalent jaw orientations. Keep the same
    # physical limits and try another orientation if a wrist limit rejects one.
    attempts = []
    for quarter in range(4):
        angle = quarter * np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        around_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        rotation = yaw @ around_z @ np.column_stack(([0, 0, -1], [0, 1, 0], [1, 0, 0]))
        current, phases = env.scene.joints(), []
        for label, z, opening, count in (("above", .075, 60, 40), ("descend", 0, 60, 30),
                                          ("close", 0, 0, 64 if measured_close else 32),
                                          ("lift", .07, 0, 40), ("hold", .07, 0, 58)):
            current["gripper"] = opening
            # Choose a gripping point above the block centre if requested. All
            # waypoints move by the same amount, so the object lift stays 7 cm.
            solution = kin.ik(cube + [0, 0, z + grasp_offset_m] - rotation @ offset, rotation, seed=current)
            q = kin.to_q(solution.joints)
            if (not solution.ok or solution.rot_err > np.radians(5)
                    or (q < env.limits.lo - 1e-6).any() or (q > env.limits.hi + 1e-6).any()):
                attempts.append(dict(quarter_turn=quarter, phase=label,
                                     position_error_mm=float(solution.pos_err * 1000),
                                     rotation_error_deg=float(np.degrees(solution.rot_err)),
                                     outside_limits=[n for i, n in enumerate(A1X.joints)
                                                     if q[i] < env.limits.lo[i] - 1e-6 or q[i] > env.limits.hi[i] + 1e-6]))
                break
            current = solution.joints
            phases.append((label, q, count))
        if len(phases) == 5:
            report["quarter_turn"] = quarter
            break
    if len(phases) != 5:
        report.update(failure="No full waypoint sequence within IK and configured limits", attempts=attempts)
        return report, observations, actions
    for label, q, count in phases:
        phase_ready = False
        for _ in range(count):
            action = np.clip((q - env.q_cmd) / env.task.arm_step, -1, 1)
            if synchronized:
                # Scale all arm coordinates together. Independent clipping lets
                # some joints arrive early and bends the fingertip approach.
                delta = (q[:6] - env.q_cmd[:6]) / env.task.arm_step
                rate = lift_rate if label == "lift" else 1.0
                action[:6] = delta / max(1.0, float(np.abs(delta).max()) / rate)
            action[-1] = 2 * (q[-1] - env.limits.lo[-1]) / (env.limits.hi[-1] - env.limits.lo[-1]) - 1
            if measured_close and label == "close":
                state = env.state()
                if state["jaws"].min() <= .5:
                    # Approach contact with at most 1 mm of commanded squeeze
                    # beyond the measured opening. Once both pads touch, the
                    # regular squeeze target and runtime guard apply.
                    opening = max(float(q[-1]), float(state["q"][-1]) - .001)
                    action[-1] = 2 * (opening - env.limits.lo[-1]) / (env.limits.hi[-1] - env.limits.lo[-1]) - 1
            observations.append(obs.copy())
            # Labels always come from the teacher at the current observed state.
            # Occasional learner actions expose its own errors to later teacher
            # corrections. Record what was executed separately for replay/audit.
            actions.append(action.copy())
            applied = action
            if learner is not None and rng.random() < learner_probability:
                applied = learner.act(obs)
                report["learner_steps"] += 1
            if executed_actions is not None:
                executed_actions.append(np.asarray(applied).copy())
            obs, _, term, trunc, info = env.step(applied)
            report["max_coupling_mm"] = max(report["max_coupling_mm"],
                abs(float(d.qpos[kin.qadr[-1]] + d.qpos[kin.mirror[0][1]])) * 1000,
                info.get("max_coupling_mm", 0.0))
            report.update(phase=label, steps=len(actions), final_height_mm=info["height"] * 1000,
                          collision=info["unsafe"], numerical=info["broken"])
            report.update({key: info[key] for key in
                           ("runtime_stop", "guard_rejection", "max_continuous_hold_s") if key in info})
            if term or trunc:
                report["success"] = bool(info["finished"] and not term and not any(w.number for w in d.warning))
                if not report["success"]:
                    report["failure"] = "collision, numerical failure or incomplete hold by time limit"
                return report, observations, actions
            if state_triggered:
                command_arrived = np.max(np.abs(env.q_cmd[:6] - q[:6])) < 1e-5
                measured_arrived = np.max(np.abs(env.state()["q"][:6] - q[:6])) < .02
                if label in ("above", "descend", "lift"):
                    phase_ready = command_arrived and measured_arrived
                elif label == "close":
                    phase_ready = command_arrived and measured_arrived and info["grasped"] and env.q_cmd[-1] <= 1e-5
                if phase_ready:
                    break
        report["phase_ends"][label] = len(actions)
        if state_triggered and label != "hold" and not phase_ready:
            report["failure"] = f"State-triggered {label} did not settle or establish contact before its limit"
            return report, observations, actions
    report["failure"] = "expert sequence ended without a completed hold"
    return report, observations, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher", choices=("waypoints", "observable"), default="waypoints",
                        help="Observable teacher uses only actor inputs; corrective moves target close/lift")
    parser.add_argument("--learner-phases", nargs="+", choices=("approach", "close", "lift"),
                        default=["close", "lift"], help="Observable teacher only: stages permitting learner actions")
    parser.add_argument("--learner-channels", choices=("all", "arm"), default="all",
                        help="Observable teacher only: override all actions or only the six arm channels")
    parser.add_argument("--learner-prefix-steps", type=int, default=0,
                        help="Observable teacher only: uninterrupted learner prefix, then teacher recovery; maximum 100")
    parser.add_argument("--learner-gripper-burst-steps", type=int, default=0,
                        help="Continuous learned arms: execute learned gripper after closing starts, then teacher recovery; maximum 50")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--state-triggered", action="store_true",
                        help="Advance on measured waypoint/grasp state instead of fixed waiting periods")
    parser.add_argument("--runtime", action="store_true",
                        help="Collect through PolicyRunner/Twin/SafetySupervisor in simulation")
    parser.add_argument("--synchronized", action="store_true",
                        help="Scale arm increments together to preserve the joint-space waypoint direction")
    parser.add_argument("--measured-close", action="store_true",
                        help="Approach bilateral contact with a 1 mm measured-opening lead")
    parser.add_argument("--lift-rate", type=float, default=1.0,
                        help="Maximum synchronized lift increment as a fraction of the normal arm step")
    parser.add_argument("--grasp-offset-mm", type=float, default=0.0,
                        help="Gripping point above the cube centre; changes teacher waypoints, not task criteria")
    parser.add_argument("--learner", type=Path,
                        help="Optional A1X checkpoint whose occasional actions elicit teacher corrections")
    parser.add_argument("--learner-probability", type=float, default=.2,
                        help="Chance of executing the learner instead of the teacher; labels remain teacher actions")
    parser.add_argument("--cube-yaw-mode", choices=("canonical", "continuous"),
                        help="Runtime only; default inherits the learner, or canonical for pure teaching")
    parser.add_argument("--cube-height-error-bound-mm", type=float,
                        help="Runtime only; default inherits the learner's saved completion bound")
    parser.add_argument("--pose-corruption", type=Path,
                        help="Runtime observable teacher only: JSON PoseCorruption fields, in metres/degrees/seconds")
    args = parser.parse_args()
    if not (1 <= args.episodes <= 200 and 1 <= args.seconds <= 600):
        parser.error("Limits: 200 episodes, 600 seconds")
    root = Path(__file__).resolve().parents[1]
    nominal = json.loads((root / "configs/a1x_lift_evaluation.json").read_text())
    robustness = json.loads((root / "configs/a1x_perception_robustness.json").read_text())
    held_out = set(nominal["held_out"]["seeds"]) | set(robustness["held_out_seeds"])
    requested_seeds = set(range(args.seed, args.seed + args.episodes))
    if requested_seeds & held_out:
        parser.error("Reserved held-out seeds cannot supply demonstrations")
    if requested_seeds & (set(nominal["development"]["seeds"]) | set(robustness["development_seeds"])):
        parser.error("Reserved development seeds cannot supply demonstrations")
    if args.teacher == "observable" and (not args.runtime or args.state_triggered or args.synchronized
            or args.measured_close or args.lift_rate != 1 or args.grasp_offset_mm != 0):
        parser.error("Observable teacher requires --runtime and uses its own recorded controller parameters")
    if (len(set(args.learner_phases)) != len(args.learner_phases) or
            ((args.learner_phases != ["close", "lift"] or args.learner_channels != "all")
             and (args.teacher != "observable" or not args.learner))):
        parser.error("Custom learner phases/channels require an observable teacher and learner, without duplicate phases")
    if not (0 < args.lift_rate <= 1) or (args.lift_rate != 1 and not args.synchronized):
        parser.error("Lift rate must be in (0, 1] and requires --synchronized")
    task = make_task("lift", A1X)
    if (not 0 <= args.learner_prefix_steps <= task.episode_steps // 2 or
            (args.learner_prefix_steps and (args.teacher != "observable" or not args.learner
             or args.learner_channels != "all" or set(args.learner_phases) != {"approach", "close", "lift"}))):
        parser.error("Learner prefix requires an observable teacher, a learner, all phases/channels, and at most 100 steps")
    if (not 0 <= args.learner_gripper_burst_steps <= task.episode_steps // 4 or
            (args.learner_gripper_burst_steps and (args.teacher != 'observable' or not args.learner
             or args.learner_prefix_steps or args.learner_channels != 'arm' or args.learner_probability != 1
             or set(args.learner_phases) != {'approach', 'close', 'lift'}))):
        parser.error("Gripper burst requires continuous learned arms, all phases, no prefix, and at most 50 steps")
    if not 0 <= args.grasp_offset_mm < task.cube_half * 1000:
        parser.error("Grasp offset must be nonnegative and below the cube half-height")
    rand = replace(Randomization.none(), fold_yaw=True)
    learner = None
    if args.learner:
        from lerobot_mp.twin.rl.policy import Policy
        # Continuous learned-arm execution is useful for teaching the gripper.
        # The teacher still supplies every gripper command in this mode.
        probability_limit = 1.0 if args.teacher == "observable" and args.learner_channels == "arm" else .5
        if not (args.runtime and (args.state_triggered or args.teacher == "observable")
                and 0 < args.learner_probability <= probability_limit):
            parser.error("Corrective collection requires --runtime, a state-triggered or observable teacher, "
                         "and learner probability in (0, .5]; observable arm-only control permits up to 1")
        learner = Policy.load(args.learner)
        if (learner.meta.robot != "a1x" or asdict(learner.task) != asdict(task)
                or not learner.meta.randomization.get("fold_yaw", False)):
            parser.error("Learner must match the A1X lift task and folded-yaw observation contract")
    yaw_mode = args.cube_yaw_mode or (learner.meta.cube_yaw_mode if learner else "canonical")
    height_bound = (args.cube_height_error_bound_mm / 1000 if args.cube_height_error_bound_mm is not None
                    else learner.meta.cube_height_error_bound_m if learner else 0.0)
    if (not np.isfinite(height_bound) or height_bound < 0 or
            (learner and (yaw_mode != learner.meta.cube_yaw_mode
                          or height_bound != learner.meta.cube_height_error_bound_m))):
        parser.error("Preserve the learner's yaw and completion contract; height bound must be finite and nonnegative")
    corruption = None
    if args.pose_corruption:
        if not args.runtime or args.teacher != "observable":
            parser.error("Pose corruption requires the runtime and observable teacher")
        corruption = PoseCorruption(**json.loads(args.pose_corruption.read_text()))
    if not args.runtime and (yaw_mode != "canonical" or height_bound):
        parser.error("Saved yaw/completion options require --runtime")
    args.out.mkdir(parents=True, exist_ok=False)
    teacher = None
    if args.teacher == "observable":
        from a1x_observable_teacher import ObservableLiftTeacher, observable_trial
        teacher = ObservableLiftTeacher()
    manifest = dict(purpose="teacher-assisted simulation examples; not autonomous learned or physical results",
                    controller="observation-only feedback" if teacher else
                               "state-triggered waypoints" if args.state_triggered else "timed waypoints",
                    teacher_contract=teacher.contract if teacher else "legacy-waypoints",
                    cube_yaw_mode=yaw_mode, cube_height_error_bound_m=height_bound,
                    pose_corruption=asdict(corruption) if corruption is not None else None,
                    learner_gripper_decoder=learner.meta.gripper_decoder if learner else None,
                    learner_phases=args.learner_phases if teacher else "all phases",
                    learner_channels=args.learner_channels if teacher else "all",
                    learner_prefix_steps=args.learner_prefix_steps,
                    learner_gripper_burst_steps=args.learner_gripper_burst_steps,
                    execution_schedule=("learner prefix then teacher" if args.learner_prefix_steps else
                        "learned arms; closing gripper burst then teacher gripper" if args.learner_gripper_burst_steps else "interleaved"),
                    execution_path="guarded runtime simulation" if args.runtime else "bare CPU physics",
                    synchronized=True if teacher else args.synchronized,
                    measured_close=True if teacher else args.measured_close,
                    lift_rate=.25 if teacher else args.lift_rate,
                    grasp_offset_m=.005 if teacher else args.grasp_offset_mm / 1000,
                    learner=str(args.learner) if learner else None,
                    learner_sha256=hashlib.sha256(args.learner.read_bytes()).hexdigest() if learner else None,
                    learner_probability=None if args.learner_prefix_steps else args.learner_probability if learner else 0.0,
                    action_labels="teacher actions; executed_actions stores the actual mixed controls",
                    task=asdict(task), randomization=asdict(rand), seed=args.seed, max_episodes=args.episodes,
                    max_seconds=args.seconds, git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((root / "src").rglob("*.py"))})
    if args.runtime:
        manifest["helper_sha256"] = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest()
                                    for name in ("a1x_runtime_env.py", "evaluate_a1x_runtime.py")}
        if teacher:
            manifest["helper_sha256"]["a1x_observable_teacher.py"] = hashlib.sha256(
                (root / "scripts/a1x_observable_teacher.py").read_bytes()).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.runtime:
        from a1x_runtime_env import RuntimeLiftEnv
        env = RuntimeLiftEnv(task, cube_yaw_mode=yaw_mode, cube_height_error_bound_m=height_bound,
                             pose_corruption=corruption)
    else:
        env = TwinEnv(task, workspace=Workspace(robot="a1x"), randomization=rand)
    # The generic IK solver otherwise uses wider vendor limits. Search within
    # the controller's configured range; never relax it to make a demo succeed.
    env.kin.lo, env.kin.hi = env.limits.lo.copy(), env.limits.hi.copy()
    started, results = time.monotonic(), []
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            if time.monotonic() - started >= args.seconds:
                break
            executed, perception_records = [], []
            if teacher:
                report, obs, act = observable_trial(env, seed, teacher=teacher, learner=learner,
                    learner_probability=args.learner_probability, executed_actions=executed,
                    learner_phases=args.learner_phases, learner_channels=args.learner_channels,
                    learner_prefix_steps=args.learner_prefix_steps,
                    learner_gripper_burst_steps=args.learner_gripper_burst_steps,
                    perception_seed=int(np.random.SeedSequence([seed, 73129]).generate_state(1)[0])
                                    if corruption is not None else None,
                    perception_records=perception_records if corruption is not None else None)
            else:
                report, obs, act = trial(env, seed, state_triggered=args.state_triggered,
                                        synchronized=args.synchronized, measured_close=args.measured_close,
                                        lift_rate=args.lift_rate, learner=learner,
                                        learner_probability=args.learner_probability, executed_actions=executed,
                                        grasp_offset_m=args.grasp_offset_mm / 1000)
            results.append(report)
            np.savez_compressed(args.out / f"seed-{seed}.npz", obs=np.asarray(obs, np.float32),
                                actions=np.asarray(act, np.float32), executed_actions=np.asarray(executed, np.float32))
            if corruption is not None:
                (args.out / f"seed-{seed}-perception.jsonl").write_text(
                    ''.join(json.dumps(row, allow_nan=False) + '\n' for row in perception_records))
            print(json.dumps(report), flush=True)
    finally:
        env.close()
        (args.out / "report.json").write_text(json.dumps(dict(episodes=results,
            successes=sum(r["success"] for r in results), elapsed_s=time.monotonic()-started), indent=2) + "\n")


if __name__ == "__main__":
    main()
