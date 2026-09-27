"""Reach box: an axis-aligned box in the robot base frame that the Specs glasses point targets into.

    vol = ReachVolume.from_workspace(ws)               # or ReachVolume.from_spec(get_spec("a1x"))
    vol.contains(p)                                    # p in metres, base frame
    overlay = vol.overlay_object()                     # "volume" object for the glasses overlay
    zone = grab_zone(spec)                             # where `grab` takes objects from (None: no zone)
    zone_overlay = grab_zone_object(zone)              # "volume" object `grab` for the glasses

Single source of truth: the twin sends the box to the glasses as an overlay object, the glasses draw it and
highlight the hand cursor, and the twin still re-checks every target (box, IK, collisions) before moving.

The default boxes are verified by `tests/test_twin_reach.py`: at every point of a 4x4x3 grid (corners
included) IK lands within 3 mm and the pose does not collide with an empty table. The box floor is at most
5 cm above the table - table height is calibrated too, so the fingers must not go lower.

The grab zone is a second box, where a `grab` from the glasses may take an object from the table
(top-down grasp). It reaches below the table top (a detection on the table has its centre a few cm above
it, and depth noise may put it lower) and only as high and as far as the tool can still point straight
down. `tests/test_twin_reach.py` checks that at every point of a 5x5 grid over its footprint IK points the
tool down (tilt < 0.26 rad) without collision, at the lowest grasp height (TCP 0.045 m) and at the hover
heights above it (0.125 and the highest, 0.173 m). The twin sends the zone to the glasses on connect as
the overlay object `grab`: the glasses draw it and highlight the detections inside it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .robots import RobotSpec

MIN_Z = 0.05  # m, lowest box floor above the table
COLOR = (0.2, 0.65, 1.0)
# Hand-picked (lo, hi) in the base frame [m]. A1X: x from 0.18 made the fingers hit the table at the corners
# (0.18, +-0.30, 0.05), so the front starts at 0.20.
DEFAULTS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "a1x": ((0.20, -0.30, 0.05), (0.50, 0.30, 0.30)),
}
# Box derived from `Desk`: the front is pushed away from the shoulder column (SO-101: at 0.13-0.15 m the jaw
# hits the shoulder in the low corners near the base), and the sides use part of the lateral reach.
NEAR_BASE = 0.04  # m
SIDE_FRACTION = 0.6

# Grab zone (lo, hi), base frame [m]. A1X: x up to 0.40, not 0.42 - at (0.42, +-0.25) with the TCP at the
# highest hover (0.173 m) the tool cannot point down (tilt 0.31 rad > 0.26). Other arms: no zone (no grab).
GRAB_ZONES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "a1x": ((0.20, -0.25, -0.03), (0.40, 0.25, 0.18)),
}
GRAB_COLOR = (1.0, 0.55, 0.1)
GRAB_ALPHA = 0.12


@dataclass
class ReachVolume:
    """Axis-aligned box in the base frame [m]."""

    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        self.lo = np.asarray(self.lo, float).reshape(3)
        self.hi = np.asarray(self.hi, float).reshape(3)
        if not np.all(np.isfinite(self.lo)) or not np.all(np.isfinite(self.hi)):
            raise ValueError("pudelko zasiegu: wspolrzedne musza byc liczbami")
        if not np.all(self.lo < self.hi):
            raise ValueError(f"pudelko zasiegu: lo {self.lo.tolist()} musi byc mniejsze od hi {self.hi.tolist()}")

    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) / 2.0

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo

    def contains(self, p: Sequence[float] | np.ndarray, margin: float = 0.0) -> bool:
        """`margin` > 0 shrinks the box (requires clearance from the walls), < 0 grows it."""
        q = np.asarray(p, float).reshape(3)
        if not np.all(np.isfinite(q)):
            return False
        return bool(np.all(q >= self.lo + margin) and np.all(q <= self.hi - margin))

    def clamp(self, p: Sequence[float] | np.ndarray) -> np.ndarray:
        """Nearest point inside the box."""
        return np.clip(np.asarray(p, float).reshape(3), self.lo, self.hi)

    def overlay_object(self, alpha: float = 0.35) -> dict[str, Any]:
        """Overlay object of kind `volume` for the glasses (centre and size in the base frame [m])."""
        return {
            "id": "reach",
            "kind": "volume",
            "pos": [round(float(v), 4) for v in self.center],
            "size": [round(float(v), 4) for v in self.size],
            "color": list(COLOR),
            "alpha": float(alpha),
            "label": "zasieg",
        }

    def to_dict(self) -> dict[str, list[float]]:
        """Shape stored in `Workspace.reach` (twin.json)."""
        return {"lo": [float(v) for v in self.lo], "hi": [float(v) for v in self.hi]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReachVolume:
        try:
            return cls(np.asarray(data["lo"], float), np.asarray(data["hi"], float))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"pudelko zasiegu: oczekiwano {{'lo': [x, y, z], 'hi': [x, y, z]}}, jest {data!r}") from exc

    @classmethod
    def from_spec(cls, spec: RobotSpec) -> ReachVolume:
        """Default box for the arm: from `DEFAULTS`, else fitted inside the `Desk` goal ring.

        From `Desk`: front at `goal_radius[0] + NEAR_BASE`, sides at +-`SIDE_FRACTION` of the ring's lateral
        reach (`goal_radius[1] * sin(cube_bearing)`), back so the corner lies on `goal_radius[1]`; height from
        `goal_height`, floor at least `MIN_Z`.
        """
        if spec.name in DEFAULTS:
            lo, hi = DEFAULTS[spec.name]
            return cls(np.array(lo), np.array(hi))
        d = spec.desk
        r0, r1 = d.goal_radius
        bearing = max(abs(b) for b in d.cube_bearing)
        y = SIDE_FRACTION * r1 * math.sin(bearing)
        x_hi = math.sqrt(max(r1 * r1 - y * y, 0.0))
        z_lo = max(MIN_Z, d.goal_height[0])
        return cls(np.array([r0 + NEAR_BASE, -y, z_lo]), np.array([x_hi, y, d.goal_height[1]]))

    @classmethod
    def from_workspace(cls, ws: Any) -> ReachVolume:
        """Box from `Workspace.reach`, falling back to the arm default."""
        data = getattr(ws, "reach", None)
        return cls.from_dict(data) if data else cls.from_spec(ws.spec())


def grab_zone(spec: RobotSpec) -> ReachVolume | None:
    """Where a `grab` may take an object from (`GRAB_ZONES`); None for an arm without one."""
    if spec.name not in GRAB_ZONES:
        return None
    lo, hi = GRAB_ZONES[spec.name]
    return ReachVolume(np.array(lo), np.array(hi))


def grab_zone_object(zone: ReachVolume, alpha: float = GRAB_ALPHA) -> dict[str, Any]:
    """Overlay object of kind `volume`, id `grab`: the grab zone for the glasses (drawn faintly)."""
    return {
        "id": "grab",
        "kind": "volume",
        "pos": [round(float(v), 4) for v in zone.center],
        "size": [round(float(v), 4) for v in zone.size],
        "color": list(GRAB_COLOR),
        "alpha": float(alpha),
        "label": "grab",
    }
