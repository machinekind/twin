"""Replay scripted actions through PolicyRunner, Twin and SafetySupervisor in simulation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from lerobot_mp.twin.collision import CollisionChecker
from lerobot_mp.twin.rl.env import TwinEnv, scene_config
from lerobot_mp.twin.rl.policy import PolicyMeta
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.runner import PolicyRunner
from lerobot_mp.twin.rl.task import make_task
from lerobot_mp.twin.robots import A1X
from lerobot_mp.twin.runtime import Twin
from lerobot_mp.twin.workspace import Workspace


def trial(source, trace):
    task = make_task("lift", A1X, episode_steps=1000, end_on_success=0, start_noise_deg=0)
    ws = Workspace(robot="a1x")
    twin = Twin(ws)
    runner = None
    cpu = TwinEnv(task, workspace=ws, randomization=Randomization.none())
    result = {"seed": source["seed"], "scripted": True, "success": False}
    try:
        cpu.reset(seed=source["seed"], options={"q": cpu.limits.home,
                                               "cube": (source["cube_start_m"], [1, 0, 0, 0])})
        cfg = scene_config(task, ws)
        twin.configure(objects=cfg.objects, grasp_sensors=cfg.grasp_sensors)
        model, data = twin.scene.model, twin.scene.data
        model.opt.timestep = task.timestep
        data.qpos[:] = cpu.scene.data.qpos
        data.qvel[:] = 0
        data.ctrl[:] = cpu.scene.data.ctrl
        mujoco.mj_forward(model, data)
        twin.connect("sim", threaded=False)
        actions = iter(trace[:, 11:18].astype(np.float32))
        policy = SimpleNamespace(task=task, meta=PolicyMeta(task=asdict(task), robot="a1x", hidden=[],
                                                           obs_dim=task.obs_dim, act_dim=7),
                                 act=lambda obs: next(actions))
        body = model.body("cube").id
        transform = np.linalg.inv(twin.scene.T_base2world)

        def cube():
            return (transform[:3, :3] @ data.xpos[body] + transform[:3, 3],
                    transform[:3, :3] @ data.xmat[body].reshape(3, 3), "symulacja")

        runner = PolicyRunner(twin, policy, cube_provider=cube)
        checker = CollisionChecker(twin.scene, grasp_objects=("cube",))
        runner.start(threaded=False)
        holds, prohibited, tick_ms = [], set(), []
        for step in range(len(trace)):
            result["step"] = step
            if not runner.step_once(step / task.control_hz):
                raise RuntimeError(runner.status.stopped_because)
            # Independent clocks: 20 Hz policy and 50 Hz runtime, as in the UI.
            for tick in range(step * 5 + 1, (step + 1) * 5 + 1):
                if tick % 2 == 0:
                    started = time.perf_counter()
                    twin.step(.02)
                    tick_ms.append((time.perf_counter() - started) * 1000)
                    if not twin.connected:
                        raise RuntimeError(twin.status.error)
                    for contact in data.contact[:data.ncon]:
                        pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                        if (checker.robot_geom[pair[0]] or checker.robot_geom[pair[1]]) and pair not in checker.allowed:
                            prohibited.add(checker._describe(pair))
            holds.append(bool(runner.status.success))
        result.update(success=bool(all(holds[-40:]) and not prohibited),
                      held_2s=bool(all(holds[-40:])), runner_success_streak=runner.status.success_streak,
                      final_height_mm=float((cube()[0][2] - task.cube_half) * 1000),
                      prohibited_contacts=sorted(prohibited),
                      warnings=[int(w.number) for w in data.warning],
                      runtime_tick_median_ms=float(np.median(tick_ms)),
                      runtime_tick_p99_ms=float(np.percentile(tick_ms, 99)),
                      runtime_tick_max_ms=float(np.max(tick_ms)))
        if any(result["warnings"]):
            result["success"] = False
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        if runner is not None:
            result["measured"] = dict(twin.status.measured)
            result["requested"] = runner.kin.from_q(runner.q_cmd)
            checked = twin._collision_checker
            if checked is not None:
                result["checked_contacts"] = [
                    {"geoms": [checked.model.geom(int(c.geom1)).name, checked.model.geom(int(c.geom2)).name],
                     "distance_mm": float(c.dist * 1000)} for c in checked.data.contact[:checked.data.ncon]]
    finally:
        if runner is not None:
            runner.stop()
        twin.close()
        cpu.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output file")
    reports = json.loads((args.cpu / "report.json").read_text())
    results = []
    for source in reports:
        result = trial(source, np.load(args.cpu / f"seed-{source['seed']}.npy"))
        results.append(result)
        print(json.dumps(result), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    if not all(r["success"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
