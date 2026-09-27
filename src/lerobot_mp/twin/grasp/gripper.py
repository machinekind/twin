"""The jaw opening the policy sees, as the fingertip gap in metres.

The A1X has no jaw measurement: its "measured" gripper is the last command, and the
command leads the jaw by up to a second. So the gap is modelled from the opening the
backend really streamed (`GripperState.streamed`), followed at the measured jaw speed.
Without a streamed value (a backend without gripper state, or the virtual jaw of a
shadow episode, which is sent nothing) an internal stream slews towards the command at
the A1X backend's stream rate. On the sim backend the scene's slide can be read as is
(`source="scene"`).

The estimator is given everything it uses; it never reads `twin.status`.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

from ...robot.base import GripperState
from .contract import GAP_CLOSED, GAP_PER_UNIT, STREAM_UNITS_S

logger = logging.getLogger(__name__)

DT_MAX = 0.2


def _clip100(v: float) -> float:
    return min(max(float(v), 0.0), 100.0)


def _toward(value: float, target: float, max_step: float) -> float:
    if target > value:
        return min(target, value + max_step)
    return max(target, value - max_step)


class GripperEstimator:
    """Opening 0..100 (0 closed, 100 open) and the gap `GAP_CLOSED + GAP_PER_UNIT * opening`."""

    def __init__(self, *, units_s: float = STREAM_UNITS_S, source: Literal["scene", "model"] = "model",
                 stream_units_s: float = STREAM_UNITS_S) -> None:
        if source not in ("scene", "model"):
            raise ValueError(f"unknown gap source {source!r}")
        for name, v in (("units_s", units_s), ("stream_units_s", stream_units_s)):
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be a positive number, got {v}")
        self.units_s = float(units_s)
        self.source = source
        self.stream_units_s = float(stream_units_s)
        self.opening = 100.0
        self._stream = 100.0
        self._t: float | None = None

    def reset(self, opening: float, t: float) -> None:
        opening, t = float(opening), float(t)
        if not (math.isfinite(opening) and math.isfinite(t)):
            raise ValueError(f"reset needs a finite opening and time, got {opening}, {t}")
        self.opening = _clip100(opening)
        self._stream = self.opening
        self._t = t

    def update(self, t: float, commanded: float, state: GripperState | None) -> float:
        """Advance the estimate to time `t` (s) and return the gap (m).

        `commanded` is the opening the driver commands (0..100); it drives only the internal
        stream, which is used when `state` has no streamed value.
        """
        commanded, t = float(commanded), float(t)
        if not math.isfinite(commanded):
            raise ValueError(f"commanded opening is not a number: {commanded}")
        # A NaN time would make every later step unbounded: the estimate would jump to the stream.
        if not math.isfinite(t):
            raise ValueError(f"time is not a number: {t}")
        dt = 0.0 if self._t is None else min(max(t - self._t, 0.0), DT_MAX)
        self._t = t

        if (self.source == "scene" and state is not None and state.measured
                and state.position is not None):
            self.opening = _clip100(state.position)
            self._stream = _clip100(state.streamed) if state.streamed is not None else self.opening
            return self.gap_of(self.opening)

        if state is not None and state.streamed is not None:
            self._stream = _clip100(state.streamed)
        else:
            self._stream = _toward(self._stream, _clip100(commanded), self.stream_units_s * dt)
        # A frozen backend stream stops by itself, and the estimate stops with it.
        self.opening = _toward(self.opening, self._stream, self.units_s * dt)
        return self.gap_of(self.opening)

    @staticmethod
    def gap_of(opening: float) -> float:
        return GAP_CLOSED + GAP_PER_UNIT * _clip100(opening)
