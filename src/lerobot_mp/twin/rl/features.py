"""Deterministic A1X geometry from actor observations; no scene state or teacher."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..kinematics import RobotKinematics
from ..robots import A1X
from .task import Limits


class A1XAlignmentFeatures:
    """24 scaled geometry features, derived only from the existing 36 lift inputs.

    Uses its own FK data; it never reads simulated contacts, task phase, labels,
    object truth or IK solutions. The cube is exactly the policy-supplied pose.
    Fixed scales carry units and do not estimate statistics from validation data.
    """

    name = "a1x-alignment-v1"
    dim = 24

    def __init__(self):
        self.kin = RobotKinematics(A1X)
        self.limits = Limits.of(self.kin)
        self.kin.lo, self.kin.hi = self.limits.lo.copy(), self.limits.hi.copy()
        home = self.kin.tcp(A1X.home)
        midpoint = np.mean([self.kin.data.geom_xpos[self.kin.model.geom(n).id]
                            for n in A1X.fingertips], axis=0)
        self.offset = home[:3, :3].T @ (midpoint - home[:3, 3])
        contract = dict(joints=A1X.joints, home=A1X.home, fingertips=A1X.fingertips,
                        tcp=A1X.tcp_site, base=A1X.base_body,
                        lower=self.limits.lo.tolist(), upper=self.limits.hi.tolist(),
                        offset=self.offset.tolist())
        self.geometry_sha256 = hashlib.sha256(
            A1X.mjcf_path.read_bytes() + json.dumps(contract, sort_keys=True).encode()).hexdigest()

    def __call__(self, observations):
        observations = np.asarray(observations, dtype=np.float32)
        if (observations.ndim != 2 or observations.shape[1] != 36
                or not np.isfinite(observations).all()):
            raise ValueError("Alignment features require finite (N, 36) lift observations")
        result = np.empty((len(observations), self.dim), dtype=np.float32)
        for i, row in enumerate(observations):
            obs = row.astype(float)
            q, command = [self.limits.lo + .5 * (obs[k:k+7]+1) * (self.limits.hi-self.limits.lo)
                          for k in (0, 7)]
            measured = self.kin.tcp(self.kin.from_q(q))
            commanded = self.kin.tcp(self.kin.from_q(command))
            tip = measured[:3, 3] + measured[:3, :3] @ self.offset
            command_tip = commanded[:3, 3] + commanded[:3, :3] @ self.offset
            cube = obs[17:20]
            rotation = np.column_stack((obs[23:26], obs[26:29], np.cross(obs[23:26], obs[26:29])))
            desired = rotation @ np.column_stack(([0, 0, -1], [0, 1, 0], [1, 0, 0]))
            relative, command_relative = tip-cube, command_tip-cube
            result[i] = np.r_[relative/.05, command_relative/.05,
                ((desired.T @ measured[:3, :3]-np.eye(3))[:, :2]/.1).ravel(),
                ((desired.T @ commanded[:3, :3]-np.eye(3))[:, :2]/.1).ravel(),
                np.linalg.norm(relative[:2])/.05, np.linalg.norm(command_relative[:2])/.05,
                q[-1]/.02, command[-1]/.02, (q[-1]-command[-1])/.005,
                (cube[2]-.015)/.05]
        return result
