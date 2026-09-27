"""Bounded A1X imitation, followed by separate CPU development evaluation."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import traceback

import torch

from lerobot_mp.twin.rl.evaluate import evaluate
from lerobot_mp.twin.rl.imitation import fit, load_demonstrations, load_refinement_policy, load_sampling_weights
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
    parser.add_argument("--features", choices=("raw", "a1x-alignment-v1"), default=None,
                        help="Optional geometry derived only from existing observations; default preserves the initial policy")
    parser.add_argument("--ignore-previous-arm", action="store_true", default=None,
                        help="Remove previous arm actions from the actor, preserving previous gripper action")
    parser.add_argument("--train-channels", choices=("all", "gripper", "gripper-residual"), default="all",
                        help="Gripper changes the final row; gripper-residual changes only its separate head; both require --init")
    parser.add_argument("--gripper-residual", choices=("none", "mlp64-v1"), default=None,
                        help="Add a zero-initialized nonlinear gripper correction; default preserves the initial policy")
    parser.add_argument("--sampling-weights", type=Path,
                        help="NPZ training priorities bound to the exact demonstration sources and split")
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
    weights = load_sampling_weights(args.sampling_weights, demos) if args.sampling_weights else None
    evaluation_seeds = set(range(args.eval_seed, args.eval_seed + args.eval_episodes))
    recorded_seeds = set(sum([demos.provenance[k] for k in
                             ("train_seeds", "validation_seeds", "excluded_seeds")], []))
    root = Path(__file__).resolve().parents[1]
    nominal = json.loads((root / "configs/a1x_lift_evaluation.json").read_text())
    robustness = json.loads((root / "configs/a1x_perception_robustness.json").read_text())
    held_out = set(nominal["held_out"]["seeds"]) | set(robustness["held_out_seeds"])
    if (recorded_seeds | evaluation_seeds) & held_out:
        parser.error("Reserved held-out seeds cannot enter training or automatic development evaluation")
    if recorded_seeds & (set(nominal["development"]["seeds"]) | set(robustness["development_seeds"])):
        parser.error("Reserved development seeds cannot enter demonstration training or validation")
    if evaluation_seeds & recorded_seeds:
        parser.error("Physics evaluation seeds overlap the demonstration collection")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = dict(purpose="simulation imitation candidate; requires independent runtime validation",
                  arguments={k: ([str(p) for p in v] if k == "demonstrations" else
                                 str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  dataset=demos.provenance, task=demos.task, randomization=demos.randomization,
                  sampling_file_sha256=hashlib.sha256(args.sampling_weights.read_bytes()).hexdigest()
                                       if args.sampling_weights else None,
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
            seconds=args.seconds, seed=args.seed, lr=args.lr, init=init, train_weights=weights, features=args.features,
            ignore_previous_arm=args.ignore_previous_arm,
            train_channels=args.train_channels,
            gripper_residual=args.gripper_residual,
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
