"""A strategy from outside the twin: swings one joint around where it starts.

The twin finds `Swing` through the entry point in this package's `pyproject.toml`, builds
`SwingSettings` from the config file's `swing:` section and the panel's form, and creates the
strategy as `Swing(settings)`. The package depends on the contracts (`lerobot-mp-api`), never
on the twin: it reads an `Observation` and answers with an `Action`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lerobot_mp_api import Action, Observation, RunInfo, Strategy


@dataclass
class SwingSettings:
    #: The joint to swing, as the robot names it (e.g. shoulder_pan on the SO-101).
    joint: str
    #: How far to each side [degrees; a gripper 0..100].
    amplitude: float = 10.0
    #: One full swing [s].
    period_s: float = 4.0
    cycles: int = 2
    #: End the run only once the joint measures back where it started, not when the swings are over.
    wait_for_return: bool = True


class Swing:
    """Swings one joint around where it starts, then waits for it to come back."""

    hz = 20.0
    settings = SwingSettings

    #: How close to its start [degrees] the joint counts as back.
    tolerance = 1.0
    #: How long the joint may take to come back after the last swing [s].
    return_s = 5.0

    def __init__(self, settings: SwingSettings):
        if settings.period_s <= 0:
            raise ValueError(f"swing: period_s musi byc dodatnie, jest {settings.period_s}")
        if settings.cycles < 1:
            raise ValueError(f"swing: cycles musi byc co najmniej 1, jest {settings.cycles}")

        self.config = settings
        self.time_limit_s = settings.cycles * settings.period_s + self.return_s
        self._centre = 0.0

    def start(self, run: RunInfo, obs: Observation) -> None:
        joint = self.config.joint
        if joint not in run.joints:
            raise ValueError(f"swing: ramie {run.robot} nie ma stawu {joint!r} (ma: {', '.join(run.joints)})")

        self._centre = obs.joints[joint]

    def act(self, obs: Observation) -> Action:
        config = self.config

        if obs.t < config.cycles * config.period_s:
            phase = obs.t / config.period_s
            offset = config.amplitude * math.sin(2.0 * math.pi * phase)
            return Action(target=self._at(offset), note=f"wahniecie {int(phase) + 1}/{config.cycles}")

        error = abs(obs.joints[config.joint] - self._centre)
        back = error <= self.tolerance

        if back or not config.wait_for_return:
            return Action(target=self._at(0.0), done=True, success=back, error=error, note="koniec")

        return Action(target=self._at(0.0), error=error, note="powrot")

    def _at(self, offset: float) -> dict[str, float]:
        return {self.config.joint: self._centre + offset}


# The protocol checks `settings` and `act` for a type checker; at runtime this only binds a name.
_: type[Strategy[SwingSettings]] = Swing
