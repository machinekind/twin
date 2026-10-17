"""Sesja kalibracji: ramie macha karta, wszystkie kamery patrza, solver liczy ich pozy.

Uogolnienie `calib/calibrate.py` z galaxeo-manipulators z jednego ramienia
i jednej kamery na dowolne ramie z `robots` i N kamer naraz. Zostaje to,
co galaxeo zmierzylo i opisalo:

* **Dwie fazy.** Najpierw szukanie - pozy nad stolem z nadgarstkiem obroconym
  gdziekolwiek, az kamera w ogole zobaczy tag. Pierwsza detekcja z nominalna
  karta mowi, gdzie mniej wiecej jest obiektyw. Potem zbieranie - normalna
  karty celuje w te kamere, plus minus rozrzut, czasem pol obrotu na druga strone.
* **Rozrzut obrotow wymuszany, a nie liczony na szczescie.** Kandydaci sa
  sortowani wedlug tego, ile podniosa `rotation_spread_R` celowanej kamery.
* **Bramki na koncu.** Kamera, ktorej fala sie nie udala, wraca jako niezaufana
  z powodem, zamiast z pewna siebie zla poza.
* **Krotkie, wolne ruchy.** Serwa pozycyjne ciagna sie za rampa; ruch ograniczony
  do `step` rad i do szczytowej predkosci `speed` rad/s, jak w galaxeo.

Zmienia sie jedno: galaxeo losuje pozy kartezjansko i rozwiazuje IK, co dla
szesciu stawow A1X dziala. SO-101 ma piec - duza czesc takich poz jest
nieosiagalna. Tu losujemy w przestrzeni STAWOW (kazda poza jest osiagalna
z definicji), a obrot narzedzia dobieramy przeszukaniem jednego stawu tak,
zeby karta stala twarza do kamery. To dziala tak samo dla pieciu i szesciu osi.

Kamery i ramie siedza za dwoma malymi protokolami, wiec ta sama sesja jedzie
na symulacji i na prawdziwym sprzecie.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

import time

import numpy as np

from ..collision import CollisionChecker
from ..kinematics import RobotKinematics
from ..robots import RobotSpec
from .card import Card
from .handeye import Fit, Observation, _median_T, judge, rotation_spread_R, solve
from .tags import detect, tag_pose


class Robot(Protocol):
    def joints(self) -> dict[str, float]: ...                        # zmierzone, jednostki aplikacji
    def move(self, joints: Mapping[str, float], duration: float) -> None: ...   # dojazd i ustalenie


class Cameras(Protocol):
    def grab(self) -> dict[str, np.ndarray]: ...                     # {nazwa: kadr RGB}


#: Kadr do obserwacji musi byc zlapany tyle sekund PO zatrzymaniu ramienia (opoznienie
#: potoku kamery: bufor sterownika, MJPG) - inaczej pokazuje karte jeszcze w ruchu.
FRESH_MARGIN_S = 0.15
FRESH_TIMEOUT_S = 1.5


def fresh_frames(cameras, t_stop: float) -> dict[str, np.ndarray]:
    """Kadry zlapane po `t_stop` (+ margines) z kamer, ktore podaja czas kadru (`grab_t`);
    kamera bez swiezego kadru w `FRESH_TIMEOUT_S` jest w tej pozie pomijana. Kamery bez
    `grab_t` (symulacja) - zwykly `grab`."""
    grab_t = getattr(cameras, "grab_t", None)
    if grab_t is None:
        return cameras.grab()
    need = t_stop + FRESH_MARGIN_S
    deadline = time.monotonic() + FRESH_TIMEOUT_S
    out: dict[str, np.ndarray] = {}
    pending = None
    while True:
        got = grab_t()
        pending = set(got) if pending is None else pending
        for name, (img, t) in got.items():
            if name in pending and img is not None and t >= need:
                out[name] = img
                pending.discard(name)
        if not pending or time.monotonic() >= deadline:
            return out
        time.sleep(0.03)


@dataclass
class WaveConfig:
    #: Minimalna wysokosc TCP i konca karty nad blatem [m].
    min_tcp_height: float = 0.07
    min_tip_height: float = 0.03
    #: Promien TCP od pionowej osi podstawy [m] - karta nad stolem, nie nad ramieniem.
    radius: tuple[float, float] = (0.12, 0.36)
    roll_jitter: float = 0.75          # jak daleko od "twarza do kamery" [rad]
    edge_on: float = 0.70              # odrzuc, gdy |podejscie . do kamery| wieksze - karta bokiem
    candidates: int = 24
    step: float = 1.2                  # najwiekszy ruch jednego stawu w jednym przejezdzie [rad]
    speed: float = 0.8                 # szczytowa predkosc stawu [rad/s]
    duration: tuple[float, float] = (1.2, 3.5)
    min_obs: int = 14
    min_poses: int = 8
    min_spread: float = 8.0
    max_poses: int = 90
    max_px: float = 1.5
    #: Srodek karty ma wpasc w te czesc kadru namierzonej kamery (margines od krawedzi).
    frame_margin: float = 0.12
    #: Faza szukania zamiata staw podstawy sektorami (tam i z powrotem), a nie bladzi
    #: losowo: przy limicie `step` losowy spacer zostawal po jednej stronie stolu.
    sweep_sectors: int = 5
    #: Tyle detekcji, zanim przyblizona poza kamery steruje celowaniem i kadrem: jedna
    #: detekcja malego taga z daleka daje poze PnP zle albo lustrzanie - i kazda
    #: druga poza celowala w miejsce, gdzie kamery nie ma.
    min_estimates: int = 2
    #: Gdzie zwykle stoja kamery - tam celuje faza szukania, zanim ktoras cos zobaczy.
    #: Odleglosc od srodka stolu [m], azymut od osi x podstawy [rad], wysokosc nad blatem [m].
    search_dist: tuple[float, float] = (0.4, 0.9)
    search_bearing: tuple[float, float] = (-1.8, 1.8)
    search_height: tuple[float, float] = (0.05, 0.5)
    search_centre: tuple[float, float, float] = (0.2, 0.0, 0.05)

    @classmethod
    def for_spec(cls, spec: RobotSpec) -> WaveConfig:
        """Nastawy fali w skali stolu tego ramienia (`spec.desk`); dla SO-101 = domyslne."""
        d = spec.desk
        return replace(cls(), radius=d.wave_radius, search_centre=d.search_centre, search_dist=d.search_dist)


@dataclass
class CameraProgress:
    obs: int = 0
    poses: int = 0
    rotations: list[np.ndarray] = field(default_factory=list)
    #: Przyblizone pozy kamery, po jednej z kazdej detekcji (PnP taga + nominalna karta).
    estimates: list[np.ndarray] = field(default_factory=list)

    def spread(self) -> float:
        return rotation_spread_R(self.rotations)

    def ready(self, cfg: WaveConfig) -> bool:
        return self.obs >= cfg.min_obs and self.poses >= cfg.min_poses and self.spread() >= cfg.min_spread

    @property
    def rough_pose(self) -> np.ndarray | None:
        """Mediana przyblizen - z kazda detekcja pewniejsza, odporna na jedna zla."""
        if not self.estimates:
            return None
        return _median_T(self.estimates)

    @property
    def hint(self) -> np.ndarray | None:
        T = self.rough_pose
        return None if T is None else T[:3, 3]


@dataclass
class StepReport:
    index: int
    joints: dict[str, float]
    #: {kamera: lista widzianych tagow} w tym kadrze.
    seen: dict[str, list[int]]
    progress: dict[str, tuple[int, int, float]]    # {kamera: (obserwacje, pozy, rozrzut [st.])}
    done: bool
    note: str = ""
    frames: dict[str, np.ndarray] = field(default_factory=dict)
    corners: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)


def move_time(q0: np.ndarray, q1: np.ndarray, cfg: WaveConfig) -> float:
    """Ile czasu na przejazd, zeby serwa pozycyjne nie zostawaly w tyle (galaxeo)."""
    travel = float(np.max(np.abs(np.asarray(q1, float) - np.asarray(q0, float))))
    return float(np.clip(1.5 * travel / cfg.speed, *cfg.duration))


class Session:
    """Jedna sesja kalibracji. `step()` robi jedna poze - wygodne dla UI w watku."""

    def __init__(
        self,
        robot: Robot,
        cameras: Cameras,
        intrinsics: Mapping[str, tuple[np.ndarray, np.ndarray | None]],
        kin: RobotKinematics,
        card: Card,
        card_nominal: np.ndarray,
        checker: CollisionChecker | None = None,
        cfg: WaveConfig | None = None,
        seed: int = 0,
        keep_frames: bool = False,
    ):
        self.robot, self.cameras, self.kin = robot, cameras, kin
        self.intrinsics = {k: (np.asarray(K, float), None if d is None else np.asarray(d, float))
                           for k, (K, d) in intrinsics.items()}
        self.card, self.card_nominal = card, np.asarray(card_nominal, float)
        self.tags = card.tag_poses()
        self.checker = checker
        self.cfg = cfg or WaveConfig.for_spec(kin.spec)
        self.rng = np.random.default_rng(seed)
        self.keep_frames = keep_frames
        self.obs: list[Observation] = []
        self.progress = {name: CameraProgress() for name in self.intrinsics}
        self.index = 0
        spec = kin.spec
        self._roll_k = spec.joints.index(spec.roll_joint) if spec.roll_joint else None
        self._home = dict(spec.home)
        if spec.gripper:
            self._home[spec.gripper] = 0.0      # karta scisnieta w szczekach przez cala sesje
        # Zamiatanie stawu podstawy: sektory od srodka na jedna strone, z powrotem na druga
        # i znow do srodka - sasiednie sektory miesza sie w jednym przejezdzie (`step`).
        self._pan = spec.joints[0] if spec.joints[0] in spec.wave_ranges else next(iter(spec.wave_ranges), None)
        n = max(1, self.cfg.sweep_sectors)
        lo, hi = spec.wave_ranges.get(self._pan, (0.0, 0.0))
        edges = np.linspace(lo, hi, n + 1)
        self._sectors = [(float(edges[i]), float(edges[i + 1])) for i in range(n)]
        c = n // 2
        self._sweep = list(range(c, n)) + list(range(n - 2, -1, -1)) + list(range(1, c))
        self._sweep_i = 0
        self._skipped = 0                       # przejazdy z rzedu, ktore nie dojechaly
        #: Diagnostyka: kazda poza (numer, stawy, co ktora kamera widziala, uwaga).
        self.log: list[dict] = []

    # --------------------------------------------------------------- stan
    @property
    def done(self) -> bool:
        return all(p.ready(self.cfg) for p in self.progress.values()) or self.index >= self.cfg.max_poses

    def _report(self, joints, seen, note, frames, corners) -> StepReport:
        prog = {n: (p.obs, p.poses, p.spread()) for n, p in self.progress.items()}
        return StepReport(self.index, joints, seen, prog, self.done, note, frames, corners)

    # -------------------------------------------------------------- fala
    def _located(self, p: CameraProgress) -> bool:
        """Kamera namierzona na tyle pewnie, zeby w nia celowac (`min_estimates` detekcji)."""
        return len(p.estimates) >= self.cfg.min_estimates

    def _target_camera(self) -> str | None:
        """Kamera, ktora najbardziej potrzebuje karty - i ktora juz namierzylismy."""
        located = [(n, p) for n, p in self.progress.items() if self._located(p) and not p.ready(self.cfg)]
        if not located:
            return None
        def need(item):
            _, p = item
            return min(p.obs / self.cfg.min_obs, p.poses / self.cfg.min_poses, p.spread() / self.cfg.min_spread)
        return min(located, key=need)[0]

    def _card_centre(self, T_tcp: np.ndarray) -> np.ndarray:
        return (T_tcp @ self.card_nominal)[:3, 3]

    def _search_point(self) -> np.ndarray:
        """Losowe "tu moglaby stac kamera" - cel fazy szukania.

        Celowanie w typowe miejsce kamery zamiast w losowy obrot: karta
        obrocona byle jak jest widziana z ukosa i przy 5 cm tagu z 60 cm
        detektor jej nie czyta - w pierwszej probie pieciu poz szukania
        zadna nie dala detekcji, chociaz karta byla w kadrze w czterech.
        """
        cfg = self.cfg
        b = self.rng.uniform(*cfg.search_bearing)
        dist = self.rng.uniform(*cfg.search_dist)
        c = np.asarray(cfg.search_centre, float)
        return np.array([c[0] + dist * np.cos(b), c[1] + dist * np.sin(b), self.rng.uniform(*cfg.search_height)])

    def _aim_roll(self, joints: dict[str, float], towards: np.ndarray) -> dict[str, float] | None:
        """Dobiera staw obrotu tak, zeby normalna karty patrzyla w `towards`."""
        if self._roll_k is None:
            return joints
        spec, cfg = self.kin.spec, self.cfg
        name = spec.roll_joint
        lo, hi = np.degrees(self.kin.lo[self._roll_k]), np.degrees(self.kin.hi[self._roll_k])
        best, best_score = None, -1.0
        for roll in np.linspace(lo, hi, 48):
            j = dict(joints, **{name: float(roll)})
            T = self.kin.tcp(j)
            d = towards - self._card_centre(T)
            d /= np.linalg.norm(d)
            approach, closing = T[:3, :3] @ self.kin._approach, T[:3, :3] @ self.kin._closing
            if abs(float(approach @ d)) > cfg.edge_on:
                return None                               # kamera wzdluz podejscia: karta bokiem
            score = abs(float(closing @ d))               # ktorakolwiek strona karty
            if score > best_score:
                best, best_score = roll, score
        # Druga strona karty (pol obrotu) i rozrzut wokol "twarza do kamery" -
        # to trzecia os obrotu, bez ktorej hand-eye jest zle uwarunkowany.
        roll = best + (180.0 if self.rng.integers(2) else 0.0)
        roll += np.degrees(self.rng.uniform(-cfg.roll_jitter, cfg.roll_jitter))
        roll = (roll + 180.0) % 360.0 - 180.0             # kat zawija sie co 360, nie co zakres stawu
        return dict(joints, **{name: float(np.clip(roll, lo, hi))})

    def _in_frame(self, cam: str, point: np.ndarray) -> bool:
        """Czy punkt wpada w kadr kamery - wedlug jej przyblizonej pozy i K."""
        p_ = self.progress[cam]
        T = p_.rough_pose if self._located(p_) else None
        if T is None:
            return True
        p = np.linalg.inv(T) @ np.r_[point, 1.0]
        if p[2] < 0.1:
            return False
        K = self.intrinsics[cam][0]
        u, v = (K @ p[:3])[:2] / p[2]
        W, H = 2 * K[0, 2] + 1, 2 * K[1, 2] + 1               # rozmiar kadru z punktu glownego
        m = self.cfg.frame_margin
        return m * W <= u <= (1 - m) * W and m * H <= v <= (1 - m) * H

    def _candidate(self, towards: np.ndarray, cam: str | None,
                   pan: tuple[float, float] | None = None) -> dict[str, float] | None:
        spec, cfg = self.kin.spec, self.cfg
        joints = dict(self._home)
        for name, (lo, hi) in spec.wave_ranges.items():
            joints[name] = float(self.rng.uniform(lo, hi))
        if pan is not None and self._pan is not None:
            joints[self._pan] = float(self.rng.uniform(*pan))
        joints = self._aim_roll(joints, towards)
        if joints is None:
            return None
        T = self.kin.tcp(joints)
        if T[2, 3] < cfg.min_tcp_height:
            return None
        if not cfg.radius[0] <= float(np.hypot(T[0, 3], T[1, 3])) <= cfg.radius[1]:
            return None
        tip = T @ self.card_nominal @ np.array([self.card.out / 2 + 0.01, 0.0, 0.0, 1.0])
        if tip[2] < cfg.min_tip_height:
            return None
        # Namierzona kamera: karta ma wpasc w jej kadr. Bez tego losowanie
        # w przestrzeni stawow chetnie podnosi ramie ponad kadr kamery stojacej
        # nisko przy stole - w pierwszej probie 30 z 40 poz bylo nad kadrem.
        if cam is not None and not self._in_frame(cam, self._card_centre(T)):
            return None
        return joints

    def _candidates(self, current: dict[str, float], target: str | None) -> list[dict[str, float]]:
        """Do `candidates` poz osiagalnych jednym przejazdem: w kolejny sektor zamiatania
        (bez celu) albo w kadr namierzonej kamery `target`."""
        pan = None
        if target is None:
            pan = self._sectors[self._sweep[self._sweep_i % len(self._sweep)]]
            self._sweep_i += 1
        q_now = self.kin.to_q(current)
        cands = []
        for _ in range(self.cfg.candidates * 12):
            if target is None:
                j = self._candidate(self._search_point(), None, pan)
            else:
                j = self._candidate(self.progress[target].hint, target)
            if j is None:
                continue
            if float(np.max(np.abs(self.kin.to_q(j) - q_now))) > self.cfg.step:
                continue
            cands.append(j)
            if len(cands) >= self.cfg.candidates:
                break
        return cands

    def _next_pose(self, current: dict[str, float]) -> dict[str, float] | None:
        target = self._target_camera()
        # Gdy jakas kamera jeszcze nie jest namierzona, co druga poza zamiata dla niej.
        searching = target is None or (any(not self._located(p) for p in self.progress.values())
                                       and self.index % 2 == 1)
        cands = self._candidates(current, None if searching else target)
        if not searching and not cands:
            # Kadru namierzonej kamery nie da sie trafic jednym przejazdem stad -
            # sektor zamiatania zamiast powrotu do domu (ten sam widok, stracony przejazd).
            searching, cands = True, self._candidates(current, None)
        if not searching:
            seen_R = self.progress[target].rotations
            cands.sort(key=lambda j: -rotation_spread_R(seen_R + [self.kin.tcp(j)[:3, :3]]))
        for j in cands:
            if self.checker is None or (self.checker.config_clear(j) is None
                                        and self.checker.path_clear(current, j) is None):
                return j
        return None

    # -------------------------------------------------------------- krok
    def step(self) -> StepReport:
        """Jedna poza: przejazd, kadry ze wszystkich kamer, detekcja."""
        current = self.robot.joints()
        if self.index == 0 and self._far_from(current, self._home):
            self.robot.move(self._home, move_time(self.kin.to_q(current), self.kin.to_q(self._home), self.cfg))
            current = self.robot.joints()
        target = self._next_pose(current)
        note = ""
        if target is None:
            note = "brak czystej pozy stad - wracam do pozycji domowej"
            target = self._home
        try:
            self.robot.move(target, move_time(self.kin.to_q(current), self.kin.to_q(target), self.cfg))
        except RuntimeError as exc:
            # Przejazd, ktory nie dojechal (nadzor wolniejszy niz rampa, staw przy limicie), to jedna
            # stracona poza, nie koniec sesji z dorobkiem - fala z 21 i 11 obserwacjami padala na
            # jednym obrocie nadgarstka. Trzy z rzedu = cos jest naprawde nie tak.
            if "nie dojechal" not in str(exc):
                raise
            self._skipped += 1
            self.index += 1
            if self._skipped >= 3:
                raise RuntimeError(f"trzy przejazdy z rzedu nie dojechaly: {exc}") from exc
            return self._report(self.robot.joints(), {}, f"przejazd pominiety: {exc}", {}, {})
        self._skipped = 0
        self.index += 1
        t_stop = time.monotonic()

        measured = self.robot.joints()
        T_tcp = self.kin.tcp(measured)                   # FK z ZMIERZONYCH katow: blad serw w dopasowaniu
        frames = fresh_frames(self.cameras, t_stop)
        seen, corners_out = {}, {}
        for cam, img in frames.items():
            if cam not in self.intrinsics:
                continue
            found = {t: c for t, c in detect(img).items() if t in self.tags}
            seen[cam] = sorted(found)
            corners_out[cam] = found
            if not found:
                continue
            p = self.progress[cam]
            K, dist = self.intrinsics[cam]
            for tid, corners in found.items():
                # Kazdy tag to osobny "montaz" o wlasnej, swobodnej pozie w chwytaku. Wspolna poza
                # karty zakladala, ze oba tagi maja DOKLADNIE ten sam srodek plecami do siebie; dwa
                # wydruki sklejone recznie z przesunieciem kilku mm dawaly 7-10 px residuum na obu
                # kamerach naraz (kazda widziala inna strone; 2026-09-26).
                self.obs.append(Observation(T_tcp, np.eye(4), corners, tid, f"tag{tid}", cam, self.index))
                p.obs += 1
                # Gdzie jest kamera, wedlug tej jednej detekcji i nominalnej karty.
                # Pierwsza wystarczy, zeby celowac; mediana kolejnych - zeby
                # pilnowac kadru coraz ciasniej.
                p.estimates.append(T_tcp @ self.card_nominal @ self.tags[tid] @ np.linalg.inv(
                    tag_pose(corners, self.card.tag_size, K, dist)))
            p.poses += 1
            p.rotations.append(T_tcp[:3, :3])
        self.log.append({"poza": self.index, "stawy": {k: round(float(v), 2) for k, v in measured.items()},
                         "widzi": {c: list(t) for c, t in seen.items()}, "uwaga": note})
        return self._report(measured, seen, note, frames if self.keep_frames else {}, corners_out)

    def mounts_nominal(self) -> dict[str, np.ndarray]:
        """Nominalna poza kazdego taga w ukladzie TCP: karta nominalna razy tag na karcie."""
        return {f"tag{tid}": self.card_nominal @ T for tid, T in self.tags.items()}

    @staticmethod
    def _far_from(a: Mapping[str, float], b: Mapping[str, float], tol: float = 3.0) -> bool:
        return any(abs(a.get(k, 0.0) - v) > tol for k, v in b.items())

    # ------------------------------------------------------------- wynik
    def solve(self) -> Fit:
        """Dopasowanie wszystkich kamer naraz i werdykt dla kazdej z osobna."""
        cams = sorted({o.camera for o in self.obs})
        if not cams:
            raise RuntimeError("zadna kamera nie zobaczyla karty - nie ma czego dopasowac")
        fit = solve(self.obs, {c: self.intrinsics[c] for c in cams}, self.card.tag_size,
                    self.mounts_nominal(), max_px=self.cfg.max_px, min_keep=self.cfg.min_obs)
        judge(fit, self.cfg.max_px, self.cfg.min_spread, self.cfg.min_obs)
        return fit

    def run(self, on_step: Callable[[StepReport], None] | None = None) -> Fit:
        while not self.done:
            report = self.step()
            if on_step is not None:
                on_step(report)
        return self.solve()
