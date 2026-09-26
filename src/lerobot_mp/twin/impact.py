"""Uderzenie w przeszkode: wykrycie z probek petli i poza, do ktorej sie wycofac.

    guard = ImpactGuard(["arm_joint1", ..., "arm_joint6"], effort_abs=20.0)
    hit = guard.feed(s.t, s.sent, s.measured, s.fresh, efforts=arm.efforts(), velocities=arm.velocities())
    if hit:                                            # hit.kind "lag" albo "effort", hit.backoff_pose
        ...

Czysta logika na liczbach (bez blizniaka, bez zegara) - karmiona probkami
`Twin.add_tick_listener` i tym, co zmierzy backend. Dwie reguly:

* **Opoznienie** (glowna na sprzecie): rozkaz jedzie dalej, a staw nie nadaza -
  |rozkaz - pomiar| > `lag_deg` przez `lag_s`, a staw prawie stoi (|predkosc| < `vel_still`).
  Normalna rampa ma rozkaz tuz przed pomiarem i staw w ruchu, wiec nie wyzwala.
  Musi zadzialac dlugo przed wlasnym "gluchym" bledem backendu A1X (3 st. przez 1 s).
* **Wysilek**: |wysilek| stawu powyzej progu bezwzglednego (`effort_abs`, na staw albo
  jeden dla wszystkich) przez `effort_n` kolejnych probek. A1X: surowa skala ramki 0x052
  (prog 20 to zgadywanie); symulacja: 0,8 * forcerange aktuatora.

Poza wycofania: najnowsza "dobra" probka (bez podejrzen) starsza o `backoff_s` od
uderzenia, ktora jest tez co najmniej `min_backoff_deg` od pomiaru na swoim najbardziej
oddalonym stawie; bez takiej - najstarsza dobra probka; bez dobrych probek - pomiar.
Probki z `sent == {}` (nic nie poszlo do serw) sa pomijane w calosci; nieswieze
pomiary (`fresh=False`) tez - nie ma w nich nic nowego o ramieniu.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass
class Impact:
    """Wyzwolenie straznika."""

    t: float
    joint: str
    #: "lag" - staw nie nadaza za rozkazem i stoi; "effort" - wysilek ponad prog.
    kind: str
    #: Opoznienie [st.] albo wysilek (jednostki backendu) w chwili wyzwolenia.
    value: float
    #: Stawy (tylko pilnowane) pozy, do ktorej sie wycofac; None - brak pomiaru tych stawow.
    backoff_pose: dict[str, float] | None


class ImpactGuard:
    """Wykrywa zablokowany staw (opoznienie albo wysilek) i wybiera poze wycofania."""

    def __init__(
        self,
        joints: Sequence[str],
        *,
        lag_deg: float = 6.0,
        lag_s: float = 0.15,
        vel_still: float = 15.0,
        effort_abs: Mapping[str, float] | float | None = None,
        effort_n: int = 3,
        backoff_s: float = 0.4,
        keep_s: float = 3.0,
        min_backoff_deg: float = 3.0,
    ):
        self.joints = list(joints)
        self.lag_deg = float(lag_deg)
        self.lag_s = float(lag_s)
        self.vel_still = float(vel_still)
        if effort_abs is None:
            self.effort_abs: dict[str, float] = {}
        elif isinstance(effort_abs, Mapping):
            self.effort_abs = {j: float(v) for j, v in effort_abs.items() if j in self.joints}
        else:
            self.effort_abs = {j: float(effort_abs) for j in self.joints}
        self.effort_n = max(1, int(effort_n))
        self.backoff_s = float(backoff_s)
        self.keep_s = float(keep_s)
        self.min_backoff_deg = float(min_backoff_deg)
        self.reset()

    def reset(self) -> None:
        """Zapomina wszystko: bufor dobrych probek, liczniki i poprzedni pomiar."""
        self._good: deque[tuple[float, dict[str, float]]] = deque()
        self._lag_since: dict[str, float] = {}
        self._effort_count: dict[str, int] = {}
        self._prev: tuple[float, dict[str, float]] | None = None

    # ------------------------------------------------------------------ probki
    def _velocity(self, t: float, measured: dict[str, float],
                  velocities: Mapping[str, float] | None) -> dict[str, float]:
        """Predkosci [st./s]: z backendu, a brakujace z poprzedniej swiezej probki. Brak = nieznana."""
        vel: dict[str, float] = {}
        if velocities:
            vel.update({j: float(velocities[j]) for j in self.joints if j in velocities})
        prev = self._prev
        if prev is not None and t > prev[0]:
            dt = t - prev[0]
            for j in self.joints:
                if j not in vel and j in measured and j in prev[1]:
                    vel[j] = (measured[j] - prev[1][j]) / dt
        return vel

    def feed(
        self,
        t: float,
        sent: Mapping[str, float],
        measured: Mapping[str, float],
        fresh: bool,
        efforts: Mapping[str, float] | None = None,
        velocities: Mapping[str, float] | None = None,
    ) -> Impact | None:
        """Jedna probka petli. Zwraca `Impact` w takcie wyzwolenia, inaczej None.

        Po wyzwoleniu liczniki startuja od zera (kolejne wyzwolenie wymaga nowego
        `lag_s` / `effort_n`); bufor dobrych probek zostaje, dopoki nie ma `reset`.
        """
        if not sent or not fresh:
            return None
        m = {j: float(measured[j]) for j in self.joints if j in measured}
        if not m:
            return None
        vel = self._velocity(t, m, velocities)
        self._prev = (t, m)

        hit: tuple[str, str, float] | None = None
        suspicious = False
        # --- opoznienie: rozkaz ucieka, a staw stoi
        for j in self.joints:
            if j not in sent or j not in m:
                self._lag_since.pop(j, None)
                continue
            err = abs(float(sent[j]) - m[j])
            v = vel.get(j)
            if err > self.lag_deg and v is not None and abs(v) < self.vel_still:
                suspicious = True
                t0 = self._lag_since.setdefault(j, t)
                if t - t0 >= self.lag_s and hit is None:
                    hit = (j, "lag", err)
            else:
                self._lag_since.pop(j, None)
        # --- wysilek: prog bezwzgledny przez `effort_n` kolejnych probek
        if efforts is not None:
            for j, limit in self.effort_abs.items():
                if j not in efforts:
                    continue
                e = float(efforts[j])
                if abs(e) > limit:
                    suspicious = True
                    n = self._effort_count[j] = self._effort_count.get(j, 0) + 1
                    if n >= self.effort_n and hit is None:
                        hit = (j, "effort", e)
                else:
                    self._effort_count.pop(j, None)

        if hit is not None:
            joint, kind, value = hit
            impact = Impact(t=t, joint=joint, kind=kind, value=value, backoff_pose=self._backoff(t, m))
            self._lag_since.clear()
            self._effort_count.clear()
            return impact
        if not suspicious:
            self._good.append((t, m))
        while self._good and self._good[0][0] < t - self.keep_s:
            self._good.popleft()
        return None

    # ---------------------------------------------------------------- wycofanie
    def _backoff(self, t: float, measured: dict[str, float]) -> dict[str, float] | None:
        """Najnowsza dobra probka >= `backoff_s` przed `t` i >= `min_backoff_deg` od pomiaru."""
        for ts, pose in reversed(self._good):
            if ts > t - self.backoff_s:
                continue
            far = max((abs(pose[j] - measured[j]) for j in pose if j in measured), default=0.0)
            if far >= self.min_backoff_deg:
                return dict(pose)
        if self._good:
            return dict(self._good[0][1])
        return dict(measured) if measured else None

    def good_samples(self) -> list[tuple[float, dict[str, float]]]:
        """Kopia bufora dobrych probek (t, poza) - do podgladu i testow."""
        return [(ts, dict(p)) for ts, p in self._good]
