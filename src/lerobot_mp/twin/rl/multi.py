"""Shape-conditioned mixed GPU environments and conservative warm start."""
from __future__ import annotations

from dataclasses import asdict, replace
import copy

import torch

from .batch import BatchEnv
from .objects import OBJECT_KINDS, object_spec
from .policy import Policy, PolicyMeta


def object_task(task, kind, scale=1.0):
    obj = object_spec(kind, scale)
    return replace(task, object_kind=kind, object_scale=scale, shape_conditioned=True,
                   cube_half=obj.half_size[2], cube_mass=obj.mass)


class MixedBatchEnv:
    """Different compiled collision topologies, one actor and one PPO minibatch."""
    def __init__(self, task, num_envs, *, kinds=OBJECT_KINDS, scale=1.0, **kwargs):
        if num_envs % len(kinds):
            raise ValueError("num_envs must be divisible by the number of object kinds")
        self.task = task
        self.num_envs = num_envs
        self.envs = []
        seed = kwargs.pop("seed", 0)
        rand = kwargs.pop("randomization")
        for i, kind in enumerate(kinds):
            subrand = replace(rand, fold_yaw=kind == "cube")
            self.envs.append(BatchEnv(object_task(task, kind, scale), num_envs // len(kinds),
                                     seed=seed + 1009*i, randomization=subrand, **kwargs))

    def reset(self):
        return torch.cat([env.reset() for env in self.envs])

    def step(self, action):
        chunks = action.chunk(len(self.envs))
        result = [env.step(a) for env, a in zip(self.envs, chunks)]
        return (*[torch.cat([r[i] for r in result]) for i in range(3)],
                {k: torch.cat([r[3][k] for r in result]) for k in result[0][3]})


def shape_warm_start(source: Policy, **task_overrides):
    """Zero-pad extra actor/critic input columns; preserve old cube behaviour.

    No claim is made that the new policy already grasps unfamiliar objects.
    """
    task = object_task(replace(source.task, **task_overrides), "cube")
    meta = PolicyMeta(task=asdict(task), hidden=list(source.meta.hidden),
                      obs_dim=task.obs_dim, act_dim=task.act_dim,
                      notes="Shape-conditioned simulation policy; generic object perception required")
    dest = Policy(meta)
    old, new = source.state_dict(), dest.state_dict()
    for key, value in old.items():
        if new[key].shape == value.shape:
            new[key] = value.clone()
        elif key == "actor.0.weight":
            new[key].zero_()
            new[key][:, :value.shape[1]] = value
        elif key in ("norm.mean", "norm.var"):
            new[key][:len(value)] = value
        else:
            raise ValueError(f"Unexpected warm-start shape mismatch: {key}")
    dest.load_state_dict(new)
    if source.critic_state is not None:
        dest.critic_state = copy.deepcopy(source.critic_state)
        first = dest.critic_state["0.weight"]
        if first.shape[1] != task.obs_dim:
            extra = torch.zeros(first.shape[0], task.obs_dim, dtype=first.dtype)
            extra[:, :first.shape[1]] = first
            dest.critic_state["0.weight"] = extra
    return dest.eval()
