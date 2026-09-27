"""Evaluate an A1X reach/lift policy through the simulation's actual controller path."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from lerobot_mp.twin.collision import CollisionChecker
from lerobot_mp.twin.rl.env import TwinEnv, scene_config
from lerobot_mp.twin.rl.evaluate import evaluate
from lerobot_mp.twin.rl.policy import Policy
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.runner import PolicyRunner
from lerobot_mp.twin.runtime import Twin
from lerobot_mp.twin.workspace import Workspace


class PhysicsMonitor:
    """Observe every simulation substep without changing its controls or dynamics."""

    def __init__(self, scene, task):
        self.scene, self.task = scene, task
        self.checker = CollisionChecker(scene, grasp_objects=("cube",) if task.cube else ())
        self.forbidden = self.checker.forbidden_pairs()
        self.contacts = {}
        self.numerical = False
        self.hold_s = self.max_hold_s = 0.0
        self.max_coupling_mm = 0.0
        self.transform = np.linalg.inv(scene.T_base2world)
        self.cube_body = scene.model.body("cube").id if task.cube else None
        self.jaw_adr = [int(scene.model.sensor(f"cube_jaw{i}").adr[0]) for i in range(2)] if task.cube else []
        self.original_step = scene.step

    def cube(self):
        d, R, t = self.scene.data, self.transform[:3, :3], self.transform[:3, 3]
        return R @ d.xpos[self.cube_body] + t, R @ d.xmat[self.cube_body].reshape(3, 3), "symulacja"

    def step(self, seconds):
        m, d, kin = self.scene.model, self.scene.data, self.scene.kin
        for _ in range(max(1, int(round(seconds / m.opt.timestep)))):
            mujoco.mj_step(m, d)
            for c in d.contact[:d.ncon]:
                pair = tuple(sorted((int(c.geom1), int(c.geom2))))
                if c.dist <= 0 and self.forbidden[pair]:
                    name = self.checker._describe(pair)
                    self.contacts[name] = max(self.contacts.get(name, 0.0), -float(c.dist) * 1000)
            self.numerical |= bool(not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
                                   or np.max(np.abs(d.qvel)) > 100)
            self.max_coupling_mm = max(self.max_coupling_mm,
                                      abs(float(d.qpos[kin.qadr[-1]] + d.qpos[kin.mirror[0][1]])) * 1000)
            # Contact sensors and xpos refer to the same pre-integration state.
            if self.task.cube:
                held = (self.cube()[0][2] - self.task.cube_half > self.task.lift_height
                        and all(d.sensordata[a] > .5 for a in self.jaw_adr))
                self.hold_s = self.hold_s + float(m.opt.timestep) if held else 0.0
                self.max_hold_s = max(self.max_hold_s, self.hold_s)
        mujoco.mj_kinematics(m, d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--cpu", action="store_true", help="Also evaluate nominal and default-randomized CPU physics")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new report file")
    policy = Policy.load(args.policy)
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    if policy.meta.robot != "a1x" or policy.task.name not in ("reach", "lift") or policy.task.control_hz != 20:
        parser.error("This validator supports A1X reach/lift at 20 Hz")
    ws = Workspace(robot="a1x")
    cpu = TwinEnv(policy.task, workspace=ws, randomization=Randomization.none())
    twin = Twin(ws)
    cfg = scene_config(policy.task, ws)
    twin.configure(objects=cfg.objects, grasp_sensors=cfg.grasp_sensors)
    results = []
    additional = {}
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            cpu.reset(seed=seed)
            twin.disconnect()
            model, data = twin.scene.model, twin.scene.data
            model.opt.timestep = policy.task.timestep
            mujoco.mj_resetData(model, data)
            data.qpos[:] = cpu.scene.data.qpos
            data.qvel[:] = 0
            data.qacc_warmstart[:] = 0
            data.ctrl[:] = cpu.scene.data.ctrl
            mujoco.mj_forward(model, data)
            twin.connect("sim", threaded=False)
            monitor = PhysicsMonitor(twin.scene, policy.task)
            twin.scene.step = monitor.step
            runner = PolicyRunner(twin, policy, cube_provider=monitor.cube if policy.task.cube else None)
            runner.goal = cpu.goal
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
                            unsafe = bool(monitor.contacts)
                    if unsafe:
                        reason = "prohibited simulated contact"
                        break
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
            finally:
                runner.stop()
                twin.scene.step = monitor.original_step
            unsafe = bool(monitor.contacts)
            distance = float(np.linalg.norm(cpu.goal - twin.scene.kin.tcp(twin.scene.joints())[:3, 3]))
            numerical = monitor.numerical or any(w.number for w in data.warning)
            normal_timeout = reason == "koniec epizodu" and runner.status.step >= policy.task.episode_steps
            task_finished = reason == "zadanie wykonane"
            fault = bool(reason) and not (normal_timeout or task_finished)
            achieved = (monitor.max_hold_s >= max(2.0, policy.task.end_on_success / policy.task.control_hz) - 1e-8
                        if policy.task.cube else distance < policy.task.success_dist)
            result = {"seed": seed, "success": bool(achieved and not fault and not unsafe and not numerical),
                      "stop_reason": reason,
                      "completed_episode": bool(normal_timeout),
                      "controller_finished": bool(task_finished),
                      "contact_penetration_mm": monitor.contacts,
                      "max_jaw_coupling_mm": monitor.max_coupling_mm,
                      "last_measured": dict(twin.status.measured), "last_command": dict(twin.status.command),
                      "prohibited_contacts": bool(unsafe), "numerical_failure": bool(numerical)}
            if policy.task.cube:
                result.update(max_continuous_hold_s=monitor.max_hold_s,
                              final_height_mm=float((monitor.cube()[0][2] - policy.task.cube_half) * 1000),
                              controller_success_streak=runner.status.success_streak,
                              controller_hold_s=runner.status.success_hold_s)
            else:
                result.update(final_distance_mm=distance * 1000, goal_m=cpu.goal.tolist())
            results.append(result)
            print(json.dumps(result), flush=True)
        if args.cpu:
            additional["cpu_nominal"] = evaluate(policy, args.episodes, workspace=ws,
                                                    randomization=Randomization.none(), seed=args.seed)
            additional["cpu_default_randomized"] = evaluate(policy, args.episodes, workspace=ws,
                                                               randomization=Randomization(), seed=args.seed)
    finally:
        twin.close()
        cpu.close()
        report = {"policy": str(args.policy), "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                  "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "task": policy.task.name, "episodes": results, **additional,
                  "successes": sum(r["success"] for r in results),
                  "simulation_only": True, "path": "PolicyRunner -> Twin -> SafetySupervisor -> SceneBackend"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
