"""Evaluate a saved A1X reach policy through the simulation's actual controller path."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from lerobot_mp.twin.collision import CollisionChecker
from lerobot_mp.twin.rl.env import TwinEnv
from lerobot_mp.twin.rl.policy import Policy
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.runner import PolicyRunner
from lerobot_mp.twin.runtime import Twin
from lerobot_mp.twin.workspace import Workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new report file")
    policy = Policy.load(args.policy)
    if policy.meta.robot != "a1x" or policy.task.name != "reach" or policy.task.control_hz != 20:
        parser.error("This validator supports A1X reach at 20 Hz")
    ws = Workspace(robot="a1x")
    cpu = TwinEnv(policy.task, workspace=ws, randomization=Randomization.none())
    twin = Twin(ws)
    results = []
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            cpu.reset(seed=seed)
            twin.disconnect()
            model, data = twin.scene.model, twin.scene.data
            model.opt.timestep = policy.task.timestep
            data.qpos[:] = cpu.scene.data.qpos
            data.qvel[:] = 0
            data.qacc_warmstart[:] = 0
            data.ctrl[:] = cpu.scene.data.ctrl
            mujoco.mj_forward(model, data)
            twin.connect("sim", threaded=False)
            runner = PolicyRunner(twin, policy)
            runner.goal = cpu.goal
            checker = CollisionChecker(twin.scene)
            forbidden = checker.forbidden_pairs()
            unsafe = False
            reason = ""
            try:
                runner.start(threaded=False)
                for step in range(policy.task.episode_steps):
                    if not runner.step_once(step / 20):
                        reason = runner.status.stopped_because
                        break
                    for tick in range(step * 5 + 1, (step + 1) * 5 + 1):
                        if tick % 2 == 0:
                            twin.step(.02)
                            if not twin.connected:
                                raise RuntimeError(twin.status.error)
                            unsafe |= any(c.dist <= 0 and forbidden[int(c.geom1), int(c.geom2)]
                                          for c in data.contact[:data.ncon])
                    if unsafe:
                        reason = "prohibited simulated contact"
                        break
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
            finally:
                runner.stop()
            distance = float(np.linalg.norm(cpu.goal - twin.scene.kin.tcp(twin.scene.joints())[:3, 3]))
            numerical = not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            normal_timeout = reason == "koniec epizodu" and runner.status.step >= policy.task.episode_steps
            fault = bool(reason) and not normal_timeout
            result = {"seed": seed, "success": bool(distance < policy.task.success_dist and not fault and not unsafe and not numerical),
                      "final_distance_mm": distance * 1000, "stop_reason": reason,
                      "completed_episode": bool(normal_timeout),
                      "prohibited_contacts": bool(unsafe), "numerical_failure": bool(numerical)}
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        twin.close()
        cpu.close()
        report = {"policy": str(args.policy), "episodes": results,
                  "successes": sum(r["success"] for r in results),
                  "simulation_only": True, "path": "PolicyRunner -> Twin -> SafetySupervisor -> SceneBackend"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
