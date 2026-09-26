"""Bounded A1X experiment with a frozen manifest and held-out CPU evaluation."""

import argparse
from dataclasses import asdict, replace
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
    parser.add_argument("--init", type=Path, help="Fine-tune an existing A1X policy for the same task")
    parser.add_argument("--init-std", type=float, default=.6)
    parser.add_argument("--critic-warmup", type=int, default=0,
                        help="Freeze actor and observation statistics while fitting a missing critic")
    parser.add_argument("--freeze-normalizer", action="store_true",
                        help="Keep pretrained observation statistics fixed after critic warm-up too")
    parser.add_argument("--nominal", action="store_true",
                        help="Train nominal dynamics/perception first, preserving folded-yaw observations")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--desired-kl", type=float, default=.01,
                        help="KL target for adaptive learning rate; zero keeps --lr fixed")
    parser.add_argument("--checkpoint-history", action="store_true",
                        help="Preserve separately reloadable policies every 50 iterations")
    parser.add_argument("--eval-seed", type=int, default=20000)
    args = parser.parse_args()
    if not (1 <= args.envs <= 8192 and 1 <= args.iters <= 1000 and 1 <= args.seconds <= 1800):
        parser.error("Limits: 8192 worlds, 1000 iterations, 1800 seconds; measure memory before scaling")
    if not 0 <= args.critic_warmup < args.iters:
        parser.error("critic-warmup must be nonnegative and less than iters")
    if not 0 < args.init_std <= 1:
        parser.error("init-std must be in (0, 1]")
    if not (0 < args.lr <= .01 and 0 <= args.desired_kl <= .1):
        parser.error("Limits: learning rate in (0, .01], desired KL in [0, .1]")
    init = Policy.load(args.init) if args.init else None
    if init is not None and (init.meta.robot != "a1x" or init.task.name != args.task):
        parser.error("Initial policy must be A1X and match --task")
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    cfg = PPOConfig(num_envs=args.envs, iterations=args.iters, seed=args.seed, init_std=args.init_std,
                    critic_warmup=args.critic_warmup, freeze_normalizer=args.freeze_normalizer,
                    lr=args.lr, desired_kl=args.desired_kl, checkpoint_history=args.checkpoint_history)
    if init is not None:
        cfg.hidden = tuple(init.meta.hidden)
    task = init.task if init is not None else make_task(args.task, A1X)
    rand = replace(Randomization.none(), fold_yaw=True) if args.nominal else Randomization()
    report = {
        "purpose": "simulation candidate, requires independent runtime validation",
        "hypothesis": (f"PPO can improve the existing A1X {args.task} candidate" if init else
                       f"Existing PPO can learn A1X {args.task} with corrected action units and timing"),
        "initial_policy": str(args.init) if init else None,
        "initial_policy_sha256": hashlib.sha256(args.init.read_bytes()).hexdigest() if init else None,
        "prior_steps": init.meta.steps if init else 0,
        "continuation": ("Actor and normalizer restored; " +
                         ("critic restored" if init.critic_state is not None else "critic initialized for warm-up") +
                         "; optimizer and action variance restart") if init else None,
        "git_head": (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                     if (root / ".git").exists() else None),
        "source_snapshot": json.loads((root / "SOURCE_SNAPSHOT.json").read_text())
                           if (root / "SOURCE_SNAPSHOT.json").exists() else None,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root / "src").rglob("*.py"))},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                         ("assets/robots/a1x/a1x.xml", "configs/default.yaml", "constraints.txt")},
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "mujoco", "mujoco-warp", "warp-lang", "numpy")},
        "ppo": asdict(cfg), "task": asdict(task), "randomization": asdict(rand),
        "evaluation_randomization": {"cpu_nominal": asdict(Randomization.none()),
                                     "cpu_randomized": asdict(Randomization())},
        "training_limit_s": args.seconds, "evaluation_seeds": [args.eval_seed, args.eval_seed + 19],
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
                       on_progress=progress, stop=stop, init=init)
        timer.cancel()
        report["training_elapsed_s"] = time.monotonic() - started
        policy = Policy.load(args.out / "policy" / "policy.pt")
        report["iterations"], report["steps"] = policy.meta.iterations, policy.meta.steps
        report["gpu_training"] = policy.meta.evals["gpu"]
        report["cpu_nominal"] = evaluate(policy, episodes=20, workspace=ws,
                                          randomization=Randomization.none(), seed=args.eval_seed)
        report["cpu_randomized"] = evaluate(policy, episodes=20, workspace=ws,
                                             randomization=Randomization(), seed=args.eval_seed)
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
        print(json.dumps({k: v for k, v in report.items() if k not in
                          ("source_sha256", "source_snapshot", "versions", "randomization", "ppo", "task")}), flush=True)


if __name__ == "__main__":
    main()
