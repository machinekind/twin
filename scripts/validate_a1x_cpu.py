"""Repeatable scripted A1X block lifts through the RL action/physics path.

This is a scripted baseline, not a learned policy or physical validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from lerobot_mp.twin.collision import CollisionChecker
from lerobot_mp.twin.rl.env import TwinEnv
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.task import make_task
from lerobot_mp.twin.robots import A1X
from lerobot_mp.twin.workspace import Workspace


def run_trial(seed: int, out: Path, video: bool = False) -> dict:
    task = make_task("lift", A1X, episode_steps=1000, end_on_success=0, start_noise_deg=0)
    env = TwinEnv(task, workspace=Workspace(robot="a1x"), randomization=Randomization.none(),
                  render_mode="rgb_array" if video else None)
    writer = None
    report = {"seed": seed, "scripted": True, "success": False}
    records = []
    try:
        rng = np.random.default_rng(seed)
        cube = np.array([0.35, 0.0, task.cube_half])
        cube[:2] += rng.uniform(-0.01, 0.01, 2)
        env.reset(seed=seed, options={"q": env.limits.home, "cube": (cube, [1, 0, 0, 0])})
        scene, kin = env.scene, env.kin
        model, data = scene.model, scene.data
        # A1X TCP x approaches downwards; TCP y closes the jaws sideways.
        rotation = np.column_stack(([0, 0, -1], [0, 1, 0], [1, 0, 0]))
        # The calibrated TCP lies between the finger plates, behind the tips.
        # Preserve that frame; place the tip midpoint at the tabletop block.
        kin.tcp(A1X.home)
        tip_midpoint = np.mean([kin.data.geom_xpos[model.geom(kin.prefix + name).id]
                                for name in A1X.fingertips], axis=0)
        tcp_R = kin.data.site_xmat[kin.site_id].reshape(3, 3)
        grasp_offset = tcp_R.T @ (tip_midpoint - kin.data.site_xpos[kin.site_id])
        report["tcp_to_tip_midpoint_m"] = grasp_offset.tolist()
        waypoints = []
        current = dict(A1X.home)
        for label, offset, opening, seconds in (("above", .10, 60, 3),
                                                 ("approach", 0, 60, 2), ("close", 0, 0, 1.6),
                                                 ("lift", .07, 0, 2), ("hold", .07, 0, 2)):
            current["gripper"] = opening
            solution = kin.ik(cube + [0, 0, offset] - rotation @ grasp_offset, rotation, seed=current)
            q = kin.to_q(solution.joints)
            if (not solution.ok or solution.rot_err > np.radians(5)
                    or (q < env.limits.lo - 1e-6).any() or (q > env.limits.hi + 1e-6).any()):
                report["failure"] = {"stage": label, "reason": "IK or configured joint limits",
                                     "pos_error_mm": solution.pos_err * 1000,
                                     "rot_error_deg": float(np.degrees(solution.rot_err))}
                return report
            current = solution.joints
            waypoints.append((label, q, round(seconds * task.control_hz)))
        checker = CollisionChecker(scene, grasp_objects=("cube",))
        report["cube_start_m"] = cube.tolist()
        report["trace_columns"] = ["time_s", "height_m", "grasped", "jaw_coupling_error_m",
                                   *[f"q/{n}" for n in A1X.joints],
                                   *[f"action/{n}" for n in A1X.joints],
                                   *[f"command/{n}" for n in A1X.joints], "cube_x", "cube_y", "cube_z"]
        report["prohibited_contacts"] = []
        report["stages"] = []
        bad_pairs = set()
        max_mirror = 0.0
        penetration = 0.0
        peak_contacts = peak_constraints = 0
        hold = []
        for label, q, count in waypoints:
            stage_bad = set()
            for _ in range(count):
                action = np.clip((q - env.q_cmd) / task.arm_step, -1, 1)
                action[-1] = 2 * (q[-1] - env.limits.lo[-1]) / (env.limits.hi[-1] - env.limits.lo[-1]) - 1
                obs, reward, term, trunc, info = env.step(action)
                if not np.isfinite(obs).all() or not np.isfinite(data.qvel).all() or term or trunc:
                    report["failure"] = {"stage": label, "reason": "numerical failure or episode ended"}
                    return report
                mirror = abs(data.qpos[kin.qadr[-1]] + data.qpos[kin.mirror[0][1]])
                max_mirror = max(max_mirror, float(mirror))
                peak_contacts, peak_constraints = max(peak_contacts, data.ncon), max(peak_constraints, data.nefc)
                for contact in data.contact[:data.ncon]:
                    pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                    if checker.robot_geom[pair[0]] or checker.robot_geom[pair[1]]:
                        penetration = max(penetration, -float(contact.dist))
                        if pair not in checker.allowed:
                            bad_pairs.add(checker._describe(pair))
                            stage_bad.add(checker._describe(pair))
                if label == "hold":
                    hold.append(bool(info["success"]))
                records.append([data.time, info["height"], float(info["grasped"]), mirror,
                                *data.qpos[kin.qadr].tolist(), *action.tolist(), *env.q_cmd.tolist(),
                                *env.state()["cube_pos"].tolist()])
                if video and len(records) % 2 == 0:
                    import cv2
                    frame = env.render()
                    if writer is None:
                        writer = cv2.VideoWriter(str(out / f"seed-{seed}.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), task.control_hz / 2,
                                                 (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError("Could not open video output")
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            report["stages"].append({"stage": label, "height_mm": info["height"] * 1000,
                                     "grasped": info["grasped"], "prohibited_contacts": sorted(stage_bad),
                                     "target_tcp_m": kin.tcp(kin.from_q(q))[:3, 3].tolist(),
                                     "actual_tcp_m": env.state()["tcp"].tolist(),
                                     "gripper_q_m": data.qpos[kin.qadr[-1]].item(),
                                     "mirror_q_m": data.qpos[kin.mirror[0][1]].item()})
        report.update(success=bool(hold and all(hold) and not bad_pairs and not any(w.number for w in data.warning)),
                      held_for_2s=bool(hold and all(hold)), prohibited_contacts=sorted(bad_pairs),
                      final_height_mm=records[-1][1] * 1000, max_jaw_coupling_error_mm=max_mirror * 1000,
                      max_contact_penetration_mm=penetration * 1000,
                      peak_contacts=int(peak_contacts), peak_constraint_rows=int(peak_constraints),
                      warnings=[int(w.number) for w in data.warning])
        return report
    finally:
        if writer is not None:
            writer.release()
        env.close()
        np.save(out / f"seed-{seed}.npy", np.asarray(records))
        (out / f"seed-{seed}.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reports = []
    for seed in range(args.seeds):
        report = run_trial(seed, args.out, video=args.video and seed == 0)
        reports.append(report)
        print(json.dumps(report), flush=True)
    (args.out / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    if not all(r["success"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
