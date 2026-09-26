"""Bounded local A1X experiment with a frozen manifest and held-out CPU evaluation."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import threading
import time
import traceback

import torch

from lerobot_mp.twin.rl.evaluate import evaluate
from lerobot_mp.twin.rl.policy import Policy
from lerobot_mp.twin.rl.ppo import PPOConfig, train
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.task import make_task
from lerobot_mp.twin.robots import A1X
from lerobot_mp.twin.workspace import Workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", choices=("reach", "lift"), default="reach")
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if not (1 <= args.envs <= 1024 and 1 <= args.iters <= 1000 and 1 <= args.seconds <= 1800):
        parser.error("Limits: 1024 worlds, 1000 iterations, 1800 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    cfg = PPOConfig(num_envs=args.envs, iterations=args.iters, seed=args.seed)
    task = make_task(args.task, A1X)
    rand = Randomization()
    report = {
        "purpose": "simulation candidate, requires independent runtime validation",
        "hypothesis": "Existing PPO can learn A1X reaching with corrected action units and timing",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root / "src").rglob("*.py"))},
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "mujoco", "mujoco-warp", "warp-lang", "numpy")},
        "ppo": asdict(cfg), "task": asdict(task), "randomization": asdict(rand),
        "training_limit_s": args.seconds, "held_out_seeds": [20000, 20019],
        "stop_rules": "Iteration/time limit, numerical failure, or GPU capacity overflow; no automatic retry",
        "status": "running", "device": torch.cuda.get_device_name(),
    }
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    stop, done = threading.Event(), threading.Event()
    timer = threading.Timer(args.seconds, stop.set)
    timer.daemon = True
    used = []

    def monitor():
        while not done.is_set():
            free, total = torch.cuda.mem_get_info()
            used.append(total - free)
            done.wait(.5)

    memory = threading.Thread(target=monitor, daemon=True)
    started = time.monotonic()
    memory.start()
    timer.start()
    try:
        def progress(p):
            if p.numerical_failures:
                raise RuntimeError(f"Numerical-failure stop: {p.numerical_failures} episodes")
            if p.iteration % 10 == 0 or p.iteration == 1 or stop.is_set():
                print(json.dumps({k: getattr(p, k) for k in
                                  ("iteration", "steps", "success", "fps", "elapsed", "physics")}), flush=True)
        ws = Workspace(robot="a1x")
        policy = train(task, cfg, workspace=ws, randomization=rand, out_dir=args.out / "policy",
                       on_progress=progress, stop=stop)
        timer.cancel()
        report["training_elapsed_s"] = time.monotonic() - started
        policy = Policy.load(args.out / "policy" / "policy.pt")
        report["iterations"], report["steps"] = policy.meta.iterations, policy.meta.steps
        report["gpu_training"] = policy.meta.evals["gpu"]
        report["cpu_nominal"] = evaluate(policy, episodes=20, workspace=ws,
                                          randomization=Randomization.none(), seed=20000)
        report["cpu_randomized"] = evaluate(policy, episodes=20, workspace=ws,
                                             randomization=rand, seed=20000)
        report["status"] = "evaluated"  # Evaluation completed; this is not a success gate.
    except BaseException:
        report["status"], report["error"] = "failed", traceback.format_exc()
        raise
    finally:
        timer.cancel()
        done.set()
        memory.join(timeout=2)
        report["elapsed_s"] = time.monotonic() - started
        report["peak_total_gpu_used_mib"] = max(used, default=0) / 2**20
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in ("source_sha256", "versions", "randomization", "ppo", "task")}), flush=True)


if __name__ == "__main__":
    main()
