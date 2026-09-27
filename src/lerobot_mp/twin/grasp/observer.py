"""The student's 19-number vector, built as training's `student_obs` builds it.

    [joints normalised to -1..1 (6), gap * 10, (believed - TCP) * 10 (3), command tail (9)]

The joints are normalised with the MODEL's joint ranges (the MJCF ranges training used),
never with the app limits, and the vector is returned raw: the checkpoint's statistics
are applied only inside the policy. The TCP is forward kinematics of the measured joints
in the base frame; training's positions are world coordinates, but the vector holds only
a difference, so the frames agree.

One `GraspObserver` per thread: it drives its `RobotKinematics`, which mutates its `MjData`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from ..kinematics import RobotKinematics
from .command import GraspCommand
from .contract import JOINT_HI, JOINT_LO, OBS_DIM

logger = logging.getLogger(__name__)

RANGE_TOL = 1e-4


class GraspObserver:
    def __init__(self, kin: RobotKinematics) -> None:
        lo, hi = kin.lo[kin.arm], kin.hi[kin.arm]
        if (lo.shape != (len(JOINT_LO),)
                or np.max(np.abs(lo - np.asarray(JOINT_LO))) > RANGE_TOL
                or np.max(np.abs(hi - np.asarray(JOINT_HI))) > RANGE_TOL):
            raise ValueError(f"arm joint ranges {lo.tolist()} .. {hi.tolist()} are not the ranges "
                             f"the policy was trained with {list(JOINT_LO)} .. {list(JOINT_HI)}")
        self.kin = kin
        self.arm = kin.spec.arm_joints
        self.lo = lo.copy()
        self.hi = hi.copy()

    def _arm(self, joints: Mapping[str, float]) -> dict[str, float]:
        # Every arm joint must be given: `to_q` would fill a missing one with home silently.
        return {n: float(joints[n]) for n in self.arm}

    def tcp(self, joints: Mapping[str, float]) -> np.ndarray:
        """Measured TCP position (3,), base frame, m."""
        return self.kin.tcp(self._arm(joints))[:3, 3].copy()

    def build(self, joints: Mapping[str, float], gap: float, cmd: GraspCommand) -> np.ndarray:
        """(19,) float32, not normalised. `joints` in degrees (app units), `gap` in metres."""
        arm = self._arm(joints)
        q = np.radians([arm[n] for n in self.arm])
        qn = 2.0 * (q - self.lo) / (self.hi - self.lo) - 1.0
        tcp = self.kin.tcp(arm)[:3, 3]
        parts = [qn, [float(gap) * 10.0], (np.asarray(cmd.state.believed_pos, float) - tcp) * 10.0,
                 cmd.obs_tail()]
        obs = np.concatenate([np.asarray(p, np.float32).ravel() for p in parts]).astype(np.float32)
        assert obs.shape == (OBS_DIM,)
        return obs
