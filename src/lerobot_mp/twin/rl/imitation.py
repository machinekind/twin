"""Behavior cloning from recorded A1X lift episodes, compatible with the PPO actor.

Validation episodes are kept whole and excluded from observation normalization.
Action prediction scores never count as manipulation success; replay the saved
policy through physics and the normal runtime separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .policy import Policy, PolicyMeta
from .task import TaskConfig


@dataclass
class Demonstrations:
    task: dict
    randomization: dict
    train_obs: np.ndarray
    train_actions: np.ndarray
    valid_obs: np.ndarray
    valid_actions: np.ndarray
    provenance: dict


def load_demonstrations(path: Path | list[Path], *, seed: int = 0,
                        validation_fraction: float = .2) -> Demonstrations:
    """Load successful teacher labels; keep each source's episode split when aggregating.

    Corrective datasets distinguish teacher targets (`actions`) from the actual
    mixed controls (`executed_actions`). Only teacher targets train the actor.
    """
    if isinstance(path, (list, tuple)):
        if not path:
            raise ValueError("At least one demonstration source is required")
        sources = [load_demonstrations(p, seed=seed, validation_fraction=validation_fraction) for p in path]
        if len(sources) == 1:
            return sources[0]
        first, seen = sources[0], set()
        keys = ("train_seeds", "validation_seeds", "excluded_seeds")
        for data in sources:
            if data.task != first.task or data.randomization != first.randomization:
                raise ValueError("Demonstration sources must have identical task and observation contracts")
            recorded = set(sum([data.provenance[k] for k in keys], []))
            if seen & recorded:
                raise ValueError("Episode seeds overlap across demonstration sources")
            seen.update(recorded)
        provenance = dict(sources=[d.provenance for d in sources], split_seed=seed,
                          **{key: sum([d.provenance[key] for d in sources], []) for key in keys})
        return Demonstrations(first.task, first.randomization,
            *[np.concatenate([getattr(d, key) for d in sources]) for key in
              ("train_obs", "train_actions", "valid_obs", "valid_actions")], provenance)
    path = Path(path)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    manifest = json.loads((path / "manifest.json").read_text())
    report = json.loads((path / "report.json").read_text())
    task = TaskConfig(**manifest["task"])
    if task.name != "lift" or task.n_joints != 7:
        raise ValueError("Expected the seven-action A1X lift task")
    seen, good, excluded, hashes = set(), [], [], {}
    for item in report["episodes"]:
        episode = item["seed"]
        if type(episode) is not int or episode in seen:
            raise ValueError("Episode seeds must be unique integers")
        seen.add(episode)
        if item.get("success") is not True or item.get("collision") is not False or item.get("numerical") is not False:
            excluded.append(episode)
            continue
        file = path / f"seed-{episode}.npz"
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        with np.load(file, allow_pickle=False) as data:
            obs, actions = data["obs"].copy(), data["actions"].copy()
        if (obs.ndim != 2 or obs.shape[1] != task.obs_dim or
                actions.shape != (len(obs), task.act_dim) or len(obs) != item.get("steps") or not len(obs)):
            raise ValueError(f"Invalid shapes/length in episode {episode}")
        if not np.isfinite(obs).all() or not np.isfinite(actions).all() or (np.abs(actions) > 1 + 1e-6).any():
            raise ValueError(f"Nonfinite observations or invalid actions in episode {episode}")
        good.append((episode, obs.astype(np.float32), actions.astype(np.float32)))
    if len(good) < 3:
        raise ValueError("Need at least three successful episodes for an episode-level split")
    order = np.random.default_rng(seed).permutation(len(good))
    n_valid = max(1, min(len(good) - 1, round(len(good) * validation_fraction)))
    valid, train = [good[i] for i in order[:n_valid]], [good[i] for i in order[n_valid:]]
    for name in ("manifest.json", "report.json"):
        hashes[name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    provenance = dict(directory=str(path), file_sha256=hashes, split_seed=seed,
                      train_seeds=[x[0] for x in train], validation_seeds=[x[0] for x in valid],
                      excluded_seeds=excluded)
    return Demonstrations(manifest["task"], manifest["randomization"],
                          np.concatenate([x[1] for x in train]), np.concatenate([x[2] for x in train]),
                          np.concatenate([x[1] for x in valid]), np.concatenate([x[2] for x in valid]), provenance)


def fit(demos: Demonstrations, *, device: str = "cpu", updates: int = 3000, seconds: float = 120,
        seed: int = 0, batch_size: int = 256, hidden: tuple[int, ...] = (256, 256, 128),
        lr: float = 3e-4, report_every: int = 100, on_progress=None,
        init: Policy | None = None) -> tuple[Policy, dict]:
    """Bounded supervised fit; an initial policy keeps its observation normalizer fixed.

    Return the best whole-episode validation checkpoint, including the initial
    policy if fitting does not improve it. Never mutate the supplied checkpoint.
    """
    if not (0 < updates <= 20000 and 0 < seconds <= 600 and batch_size > 0 and report_every > 0):
        raise ValueError("Invalid imitation budget; maximum 20,000 updates / 600 seconds")
    if not 0 < lr <= .01:
        raise ValueError("Learning rate must be in (0, .01]")
    if init is not None:
        # JSON canonicalizes tuple/list differences in saved task configurations.
        if (init.meta.robot != "a1x" or json.dumps(init.meta.task, sort_keys=True) !=
                json.dumps(demos.task, sort_keys=True) or
                init.meta.obs_dim != demos.train_obs.shape[1] or init.meta.act_dim != 7 or
                init.meta.randomization.get("fold_yaw", False) != demos.randomization.get("fold_yaw", False)):
            raise ValueError("Initial policy must match the A1X task and observation contract")
        hidden = tuple(init.meta.hidden)
    torch.manual_seed(seed)
    meta = PolicyMeta(task=demos.task, hidden=list(hidden), obs_dim=demos.train_obs.shape[1], act_dim=7,
                      robot="a1x", randomization=demos.randomization,
                      notes="Behavior cloning of scripted simulation lifts; not RL or physical validation")
    policy = Policy(meta).to(device)
    x, y, vx, vy = [torch.as_tensor(a, device=device) for a in
                    (demos.train_obs, demos.train_actions, demos.valid_obs, demos.valid_actions)]
    if init is None:
        policy.norm.update(x)  # Validation observations never enter this estimate.
    else:
        policy.load_state_dict(init.state_dict())
        policy.meta.notes = "Imitation refinement with frozen input statistics; not RL or physical validation"
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=lr)

    def losses(pred, target):
        error = (pred - target).square()
        arm, grip = error[:, :6].mean(), error[:, 6].mean()
        return arm + grip, arm, grip  # Do not dilute the one gripper channel among six arm joints.

    @torch.no_grad()
    def assess():
        values = losses(policy(vx), vy)
        return dict(loss=values[0].item(), arm_mse=values[1].item(), gripper_mse=values[2].item())

    initial, history = assess(), []
    best, best_at, completed = initial["loss"], 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    started = time.monotonic()
    for update in range(1, updates + 1):
        if time.monotonic() - started >= seconds:
            break
        idx = torch.randint(len(x), (min(batch_size, len(x)),), device=device)
        loss, _, _ = losses(policy(x[idx]), y[idx])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite imitation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        completed = update
        if update % report_every == 0 or update == updates:
            metrics = assess()
            if not all(np.isfinite(v) for v in metrics.values()):
                raise RuntimeError("Nonfinite imitation validation score")
            entry = dict(update=update, training_loss=loss.item(), validation=metrics,
                         elapsed_s=time.monotonic() - started)
            history.append(entry)
            if metrics["loss"] < best:
                best, best_at = metrics["loss"], update
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            if on_progress:
                on_progress(entry)
    # Include the final partial interval when the time limit interrupted training.
    metrics = assess()
    if np.isfinite(metrics["loss"]) and metrics["loss"] < best:
        best, best_at = metrics["loss"], completed
        best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    policy.load_state_dict(best_state)
    policy.eval()
    policy.meta.iterations = best_at
    policy.meta.steps = 0  # No new environment transitions were collected by this learner.
    result = dict(method="behavior cloning", initialized=init is not None,
                  normalizer="frozen initial policy" if init is not None else "training observations only",
                  learning_rate=lr, optimizer_updates=completed, selected_update=best_at,
                  train_samples=len(x), validation_samples=len(vx), initial_validation=initial,
                  selected_validation=assess(), history=history, elapsed_s=time.monotonic() - started)
    policy.meta.evals["imitation"] = {k: v for k, v in result.items() if k != "history"}
    return policy, result


def load_refinement_policy(path: Path, demos: Demonstrations) -> Policy:
    """Require the runner's hash-bound record and keep prior train/validation roles."""
    path = Path(path)
    report = json.loads(path.with_name("report.json").read_text())
    if report.get("policy_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Initial policy hash does not match its imitation report")
    previous = report.get("dataset", {})
    if not all(key in previous for key in ("train_seeds", "validation_seeds")):
        raise ValueError("Initial policy report must record its dataset split")
    if any(not set(previous[key]).issubset(demos.provenance[key])
           for key in ("train_seeds", "validation_seeds")):
        raise ValueError("Refinement must retain prior episodes in their training and validation roles")
    return Policy.load(path)
