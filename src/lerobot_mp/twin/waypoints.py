"""The twin's test strategy: drives the arm through given poses, one after another, without a network.

It exercises the strategy contract end to end (`strategy.StrategyRunner`) and works on any robot:
the poses name its joints. A pose counts as reached when every joint it names stays within
`tolerance` of it for `settle_ticks` ticks in a row. The run ends after the last pose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .strategy import Action, Observation, RunInfo


class Waypoints:
    #: No settings: the poses are given in code.
    settings = None

    def __init__(
        self,
        poses: Sequence[Mapping[str, float]],
        *,
        tolerance: float = 1.0,
        settle_ticks: int = 3,
        hz: float = 20.0,
        time_limit_s: float | None = None,
    ):
        if not poses:
            raise ValueError("waypoints: podaj co najmniej jedna poze")

        self.poses = [dict(pose) for pose in poses]
        self.tolerance = float(tolerance)
        self.settle_ticks = int(settle_ticks)
        self.hz = float(hz)
        self.time_limit_s = time_limit_s

        self._index = 0
        self._settled = 0

    def start(self, run: RunInfo, obs: Observation) -> None:
        named = {joint for pose in self.poses for joint in pose}
        foreign = named - set(run.joints)
        if foreign:
            raise ValueError(f"waypoints: stawy spoza ramienia {run.robot}: {sorted(foreign)}")

        self._index = 0
        self._settled = 0

    def act(self, obs: Observation) -> Action:
        error = self._error(obs)
        if error <= self.tolerance:
            self._settled += 1
        else:
            self._settled = 0

        if self._settled >= self.settle_ticks:
            if self._on_last_pose:
                return Action(target=self._pose, done=True, success=True, error=error, note=self._note)
            self._index += 1
            self._settled = 0

        return Action(target=self._pose, error=error, note=self._note)

    @property
    def _pose(self) -> dict[str, float]:
        return self.poses[self._index]

    @property
    def _on_last_pose(self) -> bool:
        return self._index == len(self.poses) - 1

    @property
    def _note(self) -> str:
        return f"poza {self._index + 1}/{len(self.poses)}"

    def _error(self, obs: Observation) -> float:
        """The largest gap between the measured joints and the current pose [app units]."""
        return max(abs(obs.joints[joint] - value) for joint, value in self._pose.items())
