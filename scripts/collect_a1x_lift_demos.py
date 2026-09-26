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
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.task import fold_yaw, make_task
from lerobot_mp.twin.robots import A1X
from lerobot_mp.twin.workspace import Workspace


def trial(env, seed, *, state_triggered=False):
    obs, _ = env.reset(seed=seed)
    state = env.state()
    cube = state["cube_pos"].copy()
    yaw = fold_yaw(np, state["cube_rot"])
    kin, m, d = env.kin, env.scene.model, env.scene.data
    current = env.scene.joints()
    kin.tcp(current)
    midpoint = np.mean([kin.data.geom_xpos[m.geom(kin.prefix + name).id] for name in A1X.fingertips], axis=0)
    offset = kin.data.site_xmat[kin.site_id].reshape(3, 3).T @ (midpoint - kin.data.site_xpos[kin.site_id])
    report = {"seed": seed, "scripted": True, "success": False, "cube_m": cube.tolist(),
              "collision": False, "numerical": False, "max_coupling_mm": 0.0,
              "state_triggered": state_triggered, "phase_ends": {}}
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
                                          ("close", 0, 0, 32), ("lift", .07, 0, 40), ("hold", .07, 0, 58)):
            current["gripper"] = opening
            solution = kin.ik(cube + [0, 0, z] - rotation @ offset, rotation, seed=current)
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
            action[-1] = 2 * (q[-1] - env.limits.lo[-1]) / (env.limits.hi[-1] - env.limits.lo[-1]) - 1
            observations.append(obs.copy())
            actions.append(action.copy())
            obs, _, term, trunc, info = env.step(action)
            report["max_coupling_mm"] = max(report["max_coupling_mm"],
                abs(float(d.qpos[kin.qadr[-1]] + d.qpos[kin.mirror[0][1]])) * 1000)
            report.update(phase=label, steps=len(actions), final_height_mm=info["height"] * 1000,
                          collision=info["unsafe"], numerical=info["broken"])
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
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--state-triggered", action="store_true",
                        help="Advance on measured waypoint/grasp state instead of fixed waiting periods")
    args = parser.parse_args()
    if not (1 <= args.episodes <= 200 and 1 <= args.seconds <= 600):
        parser.error("Limits: 200 episodes, 600 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    task = make_task("lift", A1X)
    rand = replace(Randomization.none(), fold_yaw=True)
    root = Path(__file__).resolve().parents[1]
    manifest = dict(purpose="scripted simulation demonstrations; not learned or physical results",
                    controller="state-triggered waypoints" if args.state_triggered else "timed waypoints",
                    task=asdict(task), randomization=asdict(rand), seed=args.seed, max_episodes=args.episodes,
                    max_seconds=args.seconds, git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((root / "src").rglob("*.py"))})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = TwinEnv(task, workspace=Workspace(robot="a1x"), randomization=rand)
    # The generic IK solver otherwise uses wider vendor limits. Search within
    # the controller's configured range; never relax it to make a demo succeed.
    env.kin.lo, env.kin.hi = env.limits.lo.copy(), env.limits.hi.copy()
    started, results = time.monotonic(), []
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            if time.monotonic() - started >= args.seconds:
                break
            report, obs, act = trial(env, seed, state_triggered=args.state_triggered)
            results.append(report)
            np.savez_compressed(args.out / f"seed-{seed}.npz", obs=np.asarray(obs, np.float32),
                                actions=np.asarray(act, np.float32))
            print(json.dumps(report), flush=True)
    finally:
        env.close()
        (args.out / "report.json").write_text(json.dumps(dict(episodes=results,
            successes=sum(r["success"] for r in results), elapsed_s=time.monotonic()-started), indent=2) + "\n")


if __name__ == "__main__":
    main()
