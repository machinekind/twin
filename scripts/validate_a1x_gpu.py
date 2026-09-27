"""Bounded A1X GPU/PPO plumbing check. This does not certify a learned task.

Run from the repository with its pinned training dependencies installed:
    python scripts/validate_a1x_gpu.py --out workspace/validation/gpu-smoke
"""

from __future__ import annotations

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
from lerobot_mp.twin.workspace import Workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--task", choices=("reach", "lift"), default="reach")
    args = parser.parse_args()
    if not (1 <= args.envs <= 8192 and 1 <= args.iters <= 3):
        parser.error("This smoke check is limited to 8192 worlds and 3 iterations")
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    result = {
        "purpose": "plumbing only, not policy performance",
        "git_head": (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                     if (root / ".git").exists() else None),
        "source_snapshot": json.loads((root / "SOURCE_SNAPSHOT.json").read_text())
                           if (root / "SOURCE_SNAPSHOT.json").exists() else None,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root / "src").rglob("*.py"))},
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "mujoco", "mujoco-warp", "warp-lang", "numpy")},
        "config": asdict(PPOConfig(num_envs=args.envs, iterations=args.iters, seed=0)),
        "task": args.task,
        "status": "running",
    }
    report = args.out / "report.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    stop = threading.Event()
    used = []

    def sample_memory():
        while not stop.is_set():
            free, total = torch.cuda.mem_get_info()
            used.append(total - free)  # Includes Warp and other applications, not just Torch.
            stop.wait(0.5)

    started = time.monotonic()
    monitor = None
    try:
        result["device"] = torch.cuda.get_device_name()
        monitor = threading.Thread(target=sample_memory, daemon=True)
        monitor.start()
        workspace = Workspace(robot="a1x")

        def progress(p):
            result["physics"] = p.physics
            print(json.dumps({k: getattr(p, k) for k in
                              ("iteration", "steps", "fps", "success", "elapsed", "physics")}), flush=True)

        policy = train(args.task, PPOConfig(**result["config"]), workspace=workspace,
                       out_dir=args.out / "policy", on_progress=progress)
        reloaded = Policy.load(args.out / "policy" / "policy.pt")
        assert reloaded.meta.robot == "a1x"
        assert reloaded.meta.act_dim == 7
        assert reloaded.meta.steps == policy.meta.steps
        result["evaluation"] = evaluate(reloaded, episodes=2, workspace=workspace,
                                        randomization=Randomization.none(), seed=10_000)
        result["steps"] = reloaded.meta.steps
        result["status"] = "passed"
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=2)
        result["elapsed_s"] = time.monotonic() - started
        result["peak_total_gpu_used_mib"] = max(used, default=0) / 2**20
        result["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("source_sha256", "source_snapshot")}), flush=True)


if __name__ == "__main__":
    main()
