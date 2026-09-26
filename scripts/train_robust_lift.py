"""Reproducible lift experiment with checkpoint selection and untouched test seeds.

Run from the repository: python scripts/train_robust_lift.py
All generated files live in workspace/experiments/lift-robust.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from lerobot_mp.twin.rl.env import TwinEnv
from lerobot_mp.twin.rl.policy import Policy
from lerobot_mp.twin.rl.ppo import PPOConfig, train
from lerobot_mp.twin.rl.randomize import Dynamics, Randomization


def profiles():
    return {
        "nominal": Randomization.nominal(),
        "standard": Randomization(),
        "stress": replace(Randomization.around(Dynamics(), 1.7),
                          max_delay=2, cube_delay=4, cube_period=4,
                          cube_noise=0.004, obs_noise_deg=0.5),
    }


def benchmark(policy, rand, episodes, seed):
    env = TwinEnv(policy.task, randomization=rand)
    rows = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            commands, success_at, grasp_at = [], None, None
            for step in range(policy.task.episode_steps):
                obs, _, term, trunc, info = env.step(policy.act(obs))
                commands.append(env.q_cmd.copy())
                if info["success"] and success_at is None:
                    success_at = step
                if info.get("grasped", False) and grasp_at is None:
                    grasp_at = step
                if term or trunc:
                    break
            q = np.asarray(commands)[:, :5]
            near = np.minimum(abs(q - env.limits.lo[:5]), abs(q - env.limits.hi[:5])) < np.radians(1)
            swing = float(np.degrees(np.ptp(q[success_at:], axis=0)).max()) if success_at is not None else 0.0
            rows.append({"seed": seed + ep, "success": bool(info["success"]),
                         "finished": bool(info["finished"] and info["success"]),
                         "steps": step + 1, "height_mm": float(info["height"] * 1000),
                         "limit_ticks_after_grasp": int(near[grasp_at:].any(1).sum()) if grasp_at is not None else 0,
                         "swing_deg": swing})
    finally:
        env.close()
    successes = sum(r["finished"] for r in rows)
    p = successes / episodes
    z = 1.96
    centre = (p + z*z/(2*episodes)) / (1 + z*z/episodes)
    half = z * np.sqrt(p*(1-p)/episodes + z*z/(4*episodes**2)) / (1+z*z/episodes)
    return {"episodes": episodes, "finished": successes, "success": p,
            "wilson95": [float(centre-half), float(centre+half)],
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "height_mean_mm": float(np.mean([r["height_mm"] for r in rows])),
            "height_max_mm": max(r["height_mm"] for r in rows),
            "limit_ticks_per_episode": float(np.mean([r["limit_ticks_after_grasp"] for r in rows])),
            "max_swing_deg": max(r["swing_deg"] for r in rows), "rows": rows}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def suite(path, out, episodes, seed):
    pol = Policy.load(path)
    result = {}
    for label, rand in profiles().items():
        result[label] = benchmark(pol, rand, episodes, seed)
        print(f"{out.stem}/{label}: {result[label]['finished']}/{episodes}; "
              f"steps={result[label]['mean_steps']:.1f}; "
              f"limits/ep={result[label]['limit_ticks_per_episode']:.2f}", flush=True)
        save(out, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "workspace/experiments/lift-robust")
    p.add_argument("--envs", type=int, default=4096)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--validation-episodes", type=int, default=100)
    p.add_argument("--test-episodes", type=int, default=500)
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--init", type=Path, help="Initialize from this checkpoint; comparison remains lift-v3")
    p.add_argument("--init-std", type=float, default=0.20)
    p.add_argument("--hold-still", type=float, help="Override post-success action penalty")
    p.add_argument("--limit-penalty", type=float)
    p.add_argument("--action-rate", type=float)
    args = p.parse_args()
    if min(args.iters, args.envs, args.validation_episodes, args.test_episodes) < 1:
        p.error("Counts must be positive")
    torch.set_num_threads(4)
    baseline = ROOT / "assets/policies/lift-v3/policy.pt"
    initial_path = args.init.resolve() if args.init else baseline
    initial = Policy.load(initial_path)
    for key in ("hold_still", "limit_penalty", "action_rate"):
        if getattr(args, key) is not None:
            initial.meta.task[key] = getattr(args, key)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = PPOConfig(num_envs=args.envs, iterations=args.iters, seed=260926,
                       init_std=args.init_std, desired_kl=0.005, entropy=0.001, save_every=25)
    save(out / "manifest.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "initial_policy": str(initial_path),
        "initial_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        "training_task": initial.meta.task,
        "python": sys.version, "torch": torch.__version__, "ppo": asdict(config),
        "profiles": {k: asdict(v) for k, v in profiles().items()},
        "validation_seed": 20000, "test_seed": 50000,
        "validation_episodes": args.validation_episodes, "test_episodes": args.test_episodes,
    })
    val_base = suite(baseline, out / "validation-baseline.json", args.validation_episodes, 20000)
    if args.baseline_only:
        return
    training = out / "training"
    checkpoints = []

    # train() writes policy.pt after the progress callback; preserve each previous
    # checkpoint on the following iteration, before any later overwrite.
    def progress(prog):
        if prog.iteration > 1 and (prog.iteration - 1) % config.save_every == 0:
            dest = out / f"checkpoint-{prog.iteration - 1:04d}" / "policy.pt"
            Policy.load(training / "policy.pt").save(dest)
            checkpoints.append(dest)
        if prog.iteration == 1 or prog.iteration % 10 == 0:
            print(f"train {prog.iteration}/{prog.iterations}: success={prog.success:.3%}, "
                  f"{prog.fps:.0f} steps/s, {prog.elapsed:.0f}s", flush=True)

    trained = train("lift", config, randomization=profiles()["stress"],
                    init=initial, out_dir=training, on_progress=progress)
    trained.to("cpu")
    del trained
    gc.collect()
    torch.cuda.empty_cache()
    final = out / f"checkpoint-{args.iters:04d}" / "policy.pt"
    Policy.load(training / "policy.pt").save(final)
    checkpoints.append(final)
    validation = {"baseline": val_base}
    paths = {"baseline": baseline}
    for path in checkpoints:
        name = path.parent.name
        paths[name] = path
        validation[name] = suite(path, out / f"validation-{name}.json", args.validation_episodes, 20000)

    # Eligibility requires no loss of nominal/standard success, plus limits on
    # excessive lift and post-success motion. Select using validation only.
    eligible = ["baseline"]
    for name, v in validation.items():
        if name == "baseline":
            continue
        if all(v[k]["success"] >= val_base[k]["success"] and
               v[k]["height_max_mm"] < 160 and v[k]["height_mean_mm"] < 135 and
               v[k]["max_swing_deg"] < 55 and
               v[k]["limit_ticks_per_episode"] <= max(0.4, val_base[k]["limit_ticks_per_episode"])
               for k in ("nominal", "standard")):
            eligible.append(name)
    winner = max(eligible, key=lambda n: (validation[n]["stress"]["success"],
                                        -validation[n]["stress"]["mean_steps"]))
    save(out / "selection.json", {"eligible": eligible, "selected": winner,
                                  "criterion": "stress success then mean episode length, validation only"})
    test_base = suite(baseline, out / "test-baseline.json", args.test_episodes, 50000)
    test_selected = test_base if winner == "baseline" else suite(
        paths[winner], out / "test-selected.json", args.test_episodes, 50000)
    selected = Policy.load(paths[winner])
    selected.meta.evals.update({f"heldout_{k}": {a:b for a,b in v.items() if a != "rows"}
                               for k,v in test_selected.items()})
    selected.meta.notes += f"; selected by validation seeds 20000+, heldout seeds 50000+, experiment {out}"
    selected.save(out / "selected/policy.pt")
    save(out / "summary.json", {"selected": winner,
        "baseline": {k:{a:b for a,b in v.items() if a != "rows"} for k,v in test_base.items()},
        "selected_test": {k:{a:b for a,b in v.items() if a != "rows"} for k,v in test_selected.items()}})
    print(f"Complete. Validation-selected policy: {out / 'selected/policy.pt'}", flush=True)


if __name__ == "__main__":
    main()
