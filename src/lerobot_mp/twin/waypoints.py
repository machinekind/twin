"""The twin's test strategy: drives the arm through given poses, one after another, without a network.

It exercises the strategy contract end to end (`strategy.StrategyRunner`) and works on any robot:
the poses name its joints. A pose counts as reached when every joint it names is within
`tolerance` of it for `settle_ticks` ticks in a row; the run ends after the last one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .strategy import Action, Observation, RunInfo


class Waypoints:
    def __init__(self, poses: Sequence[Mapping[str, float]], *, tolerance: float = 1.0, settle_ticks: int = 3,
                 hz: float = 20.0, time_limit_s: float | None = None):
        if not poses:
            raise ValueError("waypoints: podaj co najmniej jedna poze")
        self.poses = [dict(p) for p in poses]
        self.tolerance = float(tolerance)
        self.settle_ticks = int(settle_ticks)
        self.hz = float(hz)
        self.time_limit_s = time_limit_s
        self._i = 0
        self._settled = 0

    def start(self, run: RunInfo, obs: Observation) -> None:
        foreign = {j for p in self.poses for j in p} - set(run.joints)
        if foreign:
            raise ValueError(f"waypoints: stawy spoza ramienia {run.robot}: {sorted(foreign)}")
        self._i, self._settled = 0, 0

    def act(self, obs: Observation) -> Action:
        pose = self.poses[self._i]
        error = max(abs(float(obs.joints[j]) - v) for j, v in pose.items())
        self._settled = self._settled + 1 if error <= self.tolerance else 0
        last = self._i == len(self.poses) - 1
        if self._settled >= self.settle_ticks:
            if last:
                return Action(target=pose, done=True, success=True, error=error,
                              note=f"poza {self._i + 1}/{len(self.poses)}")
            self._i, self._settled = self._i + 1, 0
            pose = self.poses[self._i]
        return Action(target=pose, error=error, note=f"poza {self._i + 1}/{len(self.poses)}")
