"""Bounded A1X demonstration warm start, followed by held-out CPU physics evaluation."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import traceback

import torch

from lerobot_mp.twin.rl.evaluate import evaluate
from lerobot_mp.twin.rl.imitation import fit, load_demonstrations, load_refinement_policy
from lerobot_mp.twin.rl.policy import Policy
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.workspace import Workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demonstrations", type=Path, nargs="+",
                        help="One or more disjoint demonstration sources; existing per-source splits are retained")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", type=Path,
                        help="Refine an imitation checkpoint with its matching report; keep input statistics fixed")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--eval-seed", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--nominal-only", action="store_true",
                        help="Defer randomized physics evaluation until nominal behavior is reliable")
    args = parser.parse_args()
    if not (1 <= args.eval_episodes <= 100 and 1 <= args.updates <= 20000 and 1 <= args.seconds <= 600):
        parser.error("Limits: 100 evaluation episodes, 20,000 updates, 600 training seconds")
    if not 0 < args.lr <= .01:
        parser.error("Learning rate must be in (0, .01]")
    demos = load_demonstrations(args.demonstrations, seed=args.seed)
    init = load_refinement_policy(args.init, demos) if args.init else None
    evaluation_seeds = set(range(args.eval_seed, args.eval_seed + args.eval_episodes))
    recorded_seeds = set(sum([demos.provenance[k] for k in
                             ("train_seeds", "validation_seeds", "excluded_seeds")], []))
    root = Path(__file__).resolve().parents[1]
    held_out = json.loads((root / "configs/a1x_lift_evaluation.json").read_text())["held_out"]["seeds"]
    if (recorded_seeds | evaluation_seeds) & set(held_out):
        parser.error("Reserved held-out seeds cannot enter training or automatic development evaluation")
    if evaluation_seeds & recorded_seeds:
        parser.error("Physics evaluation seeds overlap the demonstration collection")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = dict(purpose="simulation imitation candidate; requires independent runtime validation",
                  arguments={k: ([str(p) for p in v] if k == "demonstrations" else
                                 str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  dataset=demos.provenance, task=demos.task, randomization=demos.randomization,
                  initial_policy_sha256=hashlib.sha256(args.init.read_bytes()).hexdigest() if init else None,
                  initial_report_sha256=hashlib.sha256(args.init.with_name("report.json").read_bytes()).hexdigest()
                                        if init else None,
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                           if (root / ".git").exists() else None,
                  source_snapshot=json.loads((root / "SOURCE_SNAPSHOT.json").read_text())
                                  if (root / "SOURCE_SNAPSHOT.json").exists() else None,
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((root / "src").rglob("*.py"))},
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions={name: importlib.metadata.version(name) for name in ("torch", "mujoco", "numpy")},
                  status="running")
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        policy, report["training"] = fit(demos, device=args.device, updates=args.updates,
            seconds=args.seconds, seed=args.seed, lr=args.lr, init=init,
            on_progress=lambda p: print(json.dumps(p), flush=True))
        policy.save(args.out / "policy.pt")
        policy = Policy.load(args.out / "policy.pt")
        report["policy_sha256"] = hashlib.sha256((args.out / "policy.pt").read_bytes()).hexdigest()
        evaluations = [("cpu_nominal", Randomization.none())]
        if not args.nominal_only:
            evaluations.append(("cpu_randomized", Randomization()))
        for name, rand in evaluations:
            report[name] = evaluate(policy, episodes=args.eval_episodes, workspace=Workspace(robot="a1x"),
                                    randomization=rand, seed=args.eval_seed)
            print(json.dumps({name: report[name]}), flush=True)
        report["status"] = "evaluated"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
