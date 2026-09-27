"""Replay recorded scripted CPU lifts in Warp; report drift and capacity usage."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot_mp.twin.rl.batch import BatchEnv
from lerobot_mp.twin.rl.env import TwinEnv
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.task import make_task
from lerobot_mp.twin.robots import A1X
from lerobot_mp.twin.workspace import Workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reports = json.loads((args.cpu / "report.json").read_text())
    if not reports or not all(r["success"] for r in reports):
        raise ValueError("Use a successful scripted CPU baseline")
    traces = np.stack([np.load(args.cpu / f"seed-{r['seed']}.npy") for r in reports])
    task = make_task("lift", A1X, episode_steps=1000, end_on_success=0, start_noise_deg=0)
    ws = Workspace(robot="a1x")
    cpu = TwinEnv(task, workspace=ws, randomization=Randomization.none())
    batch = BatchEnv(task, len(reports), workspace=ws, randomization=Randomization.none())
    batch.reset()
    initial = []
    for r in reports:
        cpu.reset(seed=r["seed"], options={"q": cpu.limits.home,
                                         "cube": (r["cube_start_m"], [1, 0, 0, 0])})
        initial.append(cpu.scene.data.qpos.copy())
    batch.qpos[:] = torch.as_tensor(np.stack(initial), dtype=torch.float32, device=batch.device)
    batch.qvel.zero_()
    batch.qacc_ws.zero_()
    batch.q_cmd[:] = torch.as_tensor(cpu.limits.home, dtype=torch.float32, device=batch.device)
    batch.ctrl[:, batch.act] = batch.q_cmd
    batch.mjw.forward(batch.m, batch.d)
    batch.wp.synchronize_device()
    batch._perceive(batch._state())
    cpu.close()
    errors_q, errors_cube, gpu_holds, jaw_errors = [], [], [], []
    for step in range(traces.shape[1]):
        actions = torch.as_tensor(traces[:, step, 11:18], dtype=torch.float32, device=batch.device)
        obs, reward, done, info = batch.step(actions)
        if done.any() or not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
            raise RuntimeError(f"GPU rollout ended or became non-finite at step {step}")
        state = batch._state()
        errors_q.append(np.abs(state["q"].cpu().numpy() - traces[:, step, 4:11]))
        errors_cube.append(np.linalg.norm(state["cube_pos"].cpu().numpy() - traces[:, step, -3:], axis=1))
        jaw_errors.append((batch.qpos[:, batch.qadr[-1]] + batch.qpos[:, batch.kin.mirror[0][1]]).abs().cpu().numpy())
        gpu_holds.append(info["success"].cpu().numpy())
    q = np.asarray(errors_q)
    holds = np.asarray(gpu_holds)[-40:].all(axis=0)
    result = {"worlds": len(reports), "steps_each": traces.shape[1],
              "max_arm_difference_deg": float(np.degrees(q[:, :, :6]).max()),
              "max_gripper_difference_mm": float(q[:, :, -1].max() * 1000),
              "max_cube_position_difference_mm": float(np.max(errors_cube) * 1000),
              "max_gpu_jaw_coupling_error_mm": float(np.max(jaw_errors) * 1000),
              "held_2s_each": holds.tolist(), "physics": batch.physics_diagnostics(),
              "note": "Measured drift; no claimed universal CPU/GPU equivalence tolerance"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not holds.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
