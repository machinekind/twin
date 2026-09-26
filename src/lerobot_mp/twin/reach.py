"""Pudelko zasiegu: prostopadloscian w ukladzie podstawy, w ktory okulary wskazuja cel.

    vol = ReachVolume.from_workspace(ws)               # albo ReachVolume.from_spec(get_spec("a1x"))
    vol.contains(p)                                    # punkt [m] w ukladzie podstawy
    overlay = vol.overlay_object()                     # obiekt "volume" dla nakladki w okularach

Jedno zrodlo prawdy: blizniak wysyla pudelko do okularow jako obiekt nakladki,
okulary rysuja je i podswietlaja kursor dloni, a blizniak i tak sprawdza kazdy
cel od nowa (pudelko, IK, kolizje) przed ruchem.

Domyslne pudelka sa sprawdzone testem (`tests/test_twin_reach.py`): w kazdym
punkcie siatki 4x4x3 (z osmioma rogami) IK trafia w 3 mm, a poza jest wolna od
kolizji z pustym stolem. Dol pudelka najwyzej 5 cm nad blatem - wysokosc stolu
tez jest kalibrowana, wiec palce nie maja schodzic nizej.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .robots import RobotSpec

#: Najnizszy dol pudelka nad blatem [m].
MIN_Z = 0.05
#: Kolor pudelka w nakladce (RGB 0..1).
COLOR = (0.2, 0.65, 1.0)
#: Pudelka ustalone recznie i sprawdzone testem, [lo, hi] w ukladzie podstawy [m].
#: A1X: plan x 0,18..0,50 dawal kolizje palcow ze stolem w rogach (0,18, +-0,30, 0,05),
#: wiec przod zaczyna sie od 0,20.
DEFAULTS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "a1x": ((0.20, -0.30, 0.05), (0.50, 0.30, 0.30)),
}
#: Pudelko wyliczane z `Desk`: odsuniecie przodu od kolumny barku [m] (SO-101: przy 0,13-0,15
#: szczeka uderza w bark w dolnych rogach blisko podstawy) i czesc bocznego zasiegu.
NEAR_BASE = 0.04
SIDE_FRACTION = 0.6


@dataclass
class ReachVolume:
    """Prostopadloscian osiowy (AABB) w ukladzie podstawy [m]."""

    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        self.lo = np.asarray(self.lo, float).reshape(3)
        self.hi = np.asarray(self.hi, float).reshape(3)
        if not np.all(np.isfinite(self.lo)) or not np.all(np.isfinite(self.hi)):
            raise ValueError("pudelko zasiegu: wspolrzedne musza byc liczbami")
        if not np.all(self.lo < self.hi):
            raise ValueError(f"pudelko zasiegu: lo {self.lo.tolist()} musi byc mniejsze od hi {self.hi.tolist()}")

    # ------------------------------------------------------------ geometria
    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) / 2.0

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo

    def contains(self, p: Sequence[float] | np.ndarray, margin: float = 0.0) -> bool:
        """Punkt w pudelku. `margin` > 0 wymaga zapasu od scian (pudelko zwezone), < 0 - poszerza."""
        q = np.asarray(p, float).reshape(3)
        if not np.all(np.isfinite(q)):
            return False
        return bool(np.all(q >= self.lo + margin) and np.all(q <= self.hi - margin))

    def clamp(self, p: Sequence[float] | np.ndarray) -> np.ndarray:
        """Najblizszy punkt pudelka."""
        return np.clip(np.asarray(p, float).reshape(3), self.lo, self.hi)

    def overlay_object(self, alpha: float = 0.35) -> dict[str, Any]:
        """Obiekt nakladki `volume` dla okularow (srodek i wymiary w ukladzie podstawy [m])."""
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
        """Postac do `Workspace.reach` (twin.json)."""
        return {"lo": [float(v) for v in self.lo], "hi": [float(v) for v in self.hi]}

    # ----------------------------------------------------------- skad wziac
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReachVolume:
        try:
            return cls(np.asarray(data["lo"], float), np.asarray(data["hi"], float))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"pudelko zasiegu: oczekiwano {{'lo': [x, y, z], 'hi': [x, y, z]}}, jest {data!r}") from exc

    @classmethod
    def from_spec(cls, spec: RobotSpec) -> ReachVolume:
        """Domyslne pudelko ramienia: z tabeli `DEFAULTS`, inaczej wpisane w pierscien celow `Desk`.

        Z `Desk`: przod od `goal_radius[0] + NEAR_BASE`, boki +-`SIDE_FRACTION` bocznego zasiegu
        pierscienia (`goal_radius[1] * sin(cube_bearing)`), tyl tak, zeby rog lezal na
        `goal_radius[1]`; wysokosc `goal_height`, dol co najmniej `MIN_Z`.
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
        """Pudelko ze stanowiska (`Workspace.reach`), a bez niego - domyslne ramienia."""
        data = getattr(ws, "reach", None)
        return cls.from_dict(data) if data else cls.from_spec(ws.spec())
