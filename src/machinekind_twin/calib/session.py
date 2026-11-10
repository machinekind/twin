"""Sesja kalibracji: ramie macha karta, wszystkie kamery patrza, solver liczy ich pozy.

Uogolnienie `calib/calibrate.py` z galaxeo-manipulators z jednego ramienia
i jednej kamery na dowolne ramie z `robots` i N kamer naraz. Zostaje to,
co galaxeo zmierzylo i opisalo:

* **Rozrzut obrotow wymuszany, a nie liczony na szczescie.** Kandydaci sa
  sortowani wedlug tego, ile podniosa `rotation_spread_R` celowanej kamery.
* **Bramki na koncu.** Kamera, ktorej fala sie nie udala, wraca jako niezaufana
  z powodem, zamiast z pewna siebie zla poza.
* **Krotkie, wolne ruchy.** Serwa pozycyjne ciagna sie za rampa; ruch ograniczony
  do `step` rad i do szczytowej predkosci `speed` rad/s, jak w galaxeo.

How the wave finds the cameras (it assumes nothing about where they stand):

* **Explore first.** The arm shows the card all over its workspace - the base's whole turn,
  near and far, low and high - and turns its face to every direction of the half-space above
  the table (`DIRECTIONS`). Per camera it keeps a coarse map of where and facing where it saw
  the card (`summary`). The old search aimed only where cameras "usually stand" and missed a
  camera looking down steeply from 0.9 m on the SO-101 stand.
* **Aim on solid evidence only.** A camera gets aimed poses once `locate_min` sightings from
  different poses agree on its pose (`_locate`: a hypothesis from each sighting's PnP, both
  solutions of the square, kept when it reprojects the others). One or a mirrored PnP no
  longer steers the wave. A pose that later sightings contradict, or that `max_misses` aimed
  poses in a row do not confirm, is dropped and the camera is searched for again.
* **A mount is a rough pose.** A camera on a fixture starts with the fixture's camera site as its
  rough pose (`mounts()` of the camera source); its sightings refit that pose or drop it.
* **Safe moves.** Card corners and the gripper stay `table_margin` above the table, the arm
  stays out of the stand's keep-out boxes and inside its base turn range (`WaveLimits`), at
  every pose and along every move. A `CollisionChecker` adds self-collisions and a clearance
  from everything else in the scene (the table, objects, another robot, a fixture).

Poses are drawn in JOINT space (every one is reachable; the SO-101 has five joints), and the
roll joint is set analytically so the card faces where it should: the same for five and six axes.

Kamery i ramie siedza za dwoma malymi protokolami, wiec ta sama sesja jedzie
na symulacji i na prawdziwym sprzecie.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

import cv2
import numpy as np

from ..collision import CollisionChecker
from ..kinematics import RobotKinematics
from ..robots import RobotSpec
from .card import Card
from .handeye import Fit, Observation, judge, rotation_spread_R, solve, solve_offsets
from .tags import corners_in_tag, detect


class Robot(Protocol):
    def joints(self) -> dict[str, float]: ...                        # zmierzone, jednostki aplikacji
    def move(self, joints: Mapping[str, float], duration: float) -> None: ...   # dojazd i ustalenie


class Cameras(Protocol):
    def grab(self) -> dict[str, np.ndarray]: ...                     # {nazwa: kadr RGB}


def _tags_in(img: np.ndarray, ids) -> dict[int, np.ndarray]:
    return {t: c for t, c in detect(img).items() if t in ids}


def _same(a: dict[int, np.ndarray], b: dict[int, np.ndarray], px: float) -> bool:
    """The same tags in two frames, none of them moved more than `px` (median over its corners)."""
    return set(a) == set(b) and all(float(np.median(np.linalg.norm(a[t] - b[t], axis=1))) <= px for t in a)


def still_frames(cameras, t_stop: float, settle: float, still_px: float, timeout: float,
                 ids) -> tuple[dict[str, np.ndarray], dict[str, dict[int, np.ndarray]], set[str]]:
    """(frames, tags `ids` found in them, cameras whose card did not stand still) for one pose.

    A cheap arm sways after a move: frames grabbed 0.15 s after it stopped gave 8-16 px residuals on the
    SO-101 stand (2026-11-03). So a frame counts only when captured `settle` s after `t_stop`, and a
    camera's tags only when two consecutive new frames show the same tags within `still_px`. A camera
    still moving `timeout` s past the settle gives no tags this pose (`moving`); one without a new frame
    is left out. Cameras without frame times (`grab_t`, the simulation) give one `grab`.
    """
    grab_t = getattr(cameras, "grab_t", None)
    if grab_t is None:
        frames = cameras.grab()
        return frames, {c: _tags_in(img, ids) for c, img in frames.items()}, set()
    need = t_stop + settle
    deadline = max(time.monotonic(), need) + timeout
    last: dict[str, tuple[float, dict[int, np.ndarray]]] = {}
    frames: dict[str, np.ndarray] = {}
    found: dict[str, dict[int, np.ndarray]] = {}
    pending = None
    while True:
        got = grab_t()
        pending = set(got) if pending is None else pending
        for name, (img, t) in got.items():
            if name not in pending or img is None or t < need or (name in last and t <= last[name][0]):
                continue
            tags = _tags_in(img, ids)
            prev = last.get(name)
            last[name], frames[name] = (t, tags), img
            if prev is not None and _same(prev[1], tags, still_px):
                found[name] = tags
                pending.discard(name)
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    moving = {n for n in pending if n in last and last[n][1]}
    found.update({n: {} for n in pending if n in frames})
    return frames, found, moving


# ---------------------------------------------------------------- stand limits
@dataclass(frozen=True)
class KeepOut:
    """A box the wave keeps the arm, its gripper and the card out of: ranges along the base frame's axes [m]."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    name: str = "strefa"

    def __post_init__(self) -> None:
        for axis in "xyz":
            value = getattr(self, axis)
            try:
                lo, hi = (float(v) for v in value)
            except (TypeError, ValueError):
                raise ValueError(f"strefa {self.name!r}: {axis} to [od, do] w metrach, jest {value!r}") from None
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"strefa {self.name!r}: {axis} to [od, do] w metrach, od < do - jest {[lo, hi]}")
            object.__setattr__(self, axis, (lo, hi))

    def hit(self, points: np.ndarray, margin: float = 0.0) -> bool:
        """Whether any of the (N, 3) points lies in the box grown by `margin` on every side."""
        lo = np.array([self.x[0], self.y[0], self.z[0]]) - margin
        hi = np.array([self.x[1], self.y[1], self.z[1]]) + margin
        return bool(np.any(np.all((points >= lo) & (points <= hi), axis=1)))

    def describe(self) -> str:
        return (f"{self.name} (x {self.x[0]:+.2f}..{self.x[1]:+.2f}, y {self.y[0]:+.2f}..{self.y[1]:+.2f}, "
                f"z {self.z[0]:.2f}..{self.z[1]:.2f} m)")


@dataclass(frozen=True)
class WaveLimits:
    """What the stand tells the wave (`twin.json`, section `wave`). Empty: only the arm's own limits."""

    #: Boxes to keep out of, e.g. a second arm next to this one.
    keepout: tuple[KeepOut, ...] = ()
    #: The base's turn range [app units, degrees]; None = the wave's own range.
    base_turn: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "keepout", tuple(self.keepout))
        if self.base_turn is not None:
            try:
                lo, hi = (float(v) for v in self.base_turn)
            except (TypeError, ValueError):
                raise ValueError(f"wave.base_turn to [od, do] w stopniach, jest {self.base_turn!r}") from None
            if not lo < hi:
                raise ValueError(f"wave.base_turn: od < do, jest {[lo, hi]}")
            object.__setattr__(self, "base_turn", (lo, hi))

    def describe(self) -> str:
        """One line for the panel; "" without limits."""
        parts = [b.describe() for b in self.keepout]
        if self.base_turn is not None:
            parts.append(f"obrot podstawy {self.base_turn[0]:+.0f}..{self.base_turn[1]:+.0f} st.")
        return "; ".join(parts)


@dataclass
class WaveConfig:
    #: Minimalna wysokosc TCP nad blatem [m].
    min_tcp_height: float = 0.07
    #: Card corners, TCP and jaw tips stay this far above the table top [m], at every pose and along every move.
    table_margin: float = 0.04
    #: Promien TCP od pionowej osi podstawy [m] - karta nad stolem, nie nad ramieniem.
    radius: tuple[float, float] = (0.12, 0.36)
    roll_jitter: float = 0.75          # jak daleko od "twarza do kamery" [rad]
    edge_on: float = 0.70              # odrzuc, gdy |podejscie . do kamery| wieksze - karta bokiem
    candidates: int = 24
    #: Joint samples drawn when planning one pose.
    samples: int = 160
    step: float = 1.2                  # najwiekszy ruch jednego stawu w jednym przejezdzie [rad]
    speed: float = 0.8                 # szczytowa predkosc stawu [rad/s]
    duration: tuple[float, float] = (1.2, 3.5)
    #: Seconds after a move before a frame counts (the arm sways), the most a tag may move between two
    #: consecutive frames to count [px], and how long to wait for that [s] (`still_frames`).
    settle_s: float = 0.8
    still_px: float = 0.5
    still_timeout: float = 2.5
    #: What the fit needs per camera: observations, poses and rotation spread [deg]. A cheap arm's servo play
    #: averages out over many (the SO-101 stand needed more than 14 / 8 / 8 deg, 2026-11-03).
    min_obs: int = 30
    #: Observations a camera collects beyond `min_obs`, so the solver can leave out frames that disagree.
    spare_obs: int = 6
    min_poses: int = 16
    min_spread: float = 15.0
    max_poses: int = 200
    max_px: float = 1.5
    #: Srodek karty ma wpasc w te czesc kadru namierzonej kamery (margines od krawedzi).
    frame_margin: float = 0.12
    #: Exploration: the base's turn range is the description's `wave_ranges` widened by `turn_extra`
    #: degrees each side (within the joint's limits and `base_turn`), cut into `sectors` for the map.
    turn_extra: float = 30.0
    sectors: int = 5
    #: The card counts as facing a direction when its normal is within acos(min_facing) of it.
    min_facing: float = 0.9
    #: Poses of exploration before any aiming.
    explore_min: int = 12
    #: Highest card centre exploration shows the card at [m]: cameras look at the table, and higher up the
    #: card leaves their frames.
    explore_top: float = 0.35
    #: The camera may look at the card from the wrist's side only this much (approach . to camera): from
    #: further back the jaw tips cover the tag's border.
    jaw_side: float = 0.2
    #: Poses near those a camera not located yet saw the card in, that may miss it in a row before exploring on.
    follow: int = 4
    #: Largest angle between the card's face and an aimed camera [deg].
    aim_angle: float = 50.0
    #: Agreeing sightings from different poses before a camera is aimed at, and how close they must agree:
    #: reprojected corners within `locate_tol` of the tag's side in pixels, plus 6 px.
    locate_min: int = 3
    locate_tol: float = 0.6
    #: How far one sighting's pose may be off and still seed a hypothesis [deg]: the card sits in the jaws
    #: up to about 10 deg off nominal, so one sighting puts the camera that far off.
    loose_deg: float = 15.0
    #: Aimed poses in a row a camera does not see before its rough pose is dropped.
    max_misses: int = 6
    #: Stand limits (`WaveLimits`) and how far outside a keep-out box everything stays [m].
    keepout: tuple[KeepOut, ...] = ()
    keepout_margin: float = 0.04
    base_turn: tuple[float, float] | None = None
    #: Clearance [m] from everything in the scene that is not the arm (`CollisionChecker`); the caller
    #: builds the checker with it.
    clearance: float = 0.02
    #: Fit the zero offsets of the servos the cameras can tell apart (`offset_joints`) with the
    #: cameras. Kept when the residual drops by at least `offset_gain` of itself and 0.3 px; an offset
    #: over `max_offset` [app units] is not a servo's zero and leaves them all out. On the SO-101 stand
    #: (2026-10-27) offsets of 1.8-2.5 deg took one camera from 7.3 to 4.4 px.
    fit_offsets: bool = True
    offset_gain: float = 0.15
    max_offset: float = 4.0

    @classmethod
    def for_spec(cls, spec: RobotSpec, limits: WaveLimits | None = None) -> WaveConfig:
        """Nastawy fali w skali stolu tego ramienia (`spec.desk`) z ograniczeniami stanowiska."""
        limits = limits or WaveLimits()
        return replace(cls(), radius=spec.desk.wave_radius, keepout=limits.keepout, base_turn=limits.base_turn)


def offset_joints(spec: RobotSpec) -> list[str]:
    """Joints whose servo zero a wave can tell, in description (chain) order.

    A zero shows only through a joint after it in the chain that turns during the wave (`wave_ranges`,
    and the roll the wave aims with). So not the last such joint: its zero is the same as the card
    sitting turned in the jaws. And not the base's turn: that is the same as every camera standing
    elsewhere around the arm. On the SO-101: shoulder_lift, elbow_flex, wrist_flex.
    """
    arm = [n for n in spec.joints if n != spec.gripper]
    moving = set(spec.wave_ranges) | {spec.roll_joint}
    return [n for k, n in enumerate(arm) if k > 0 and any(m in moving for m in arm[k + 1:])]


# ----------------------------------------------------------------- directions
def _unit(az_deg: float, el_deg: float) -> np.ndarray:
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


#: Where exploration turns the card's face, in the base frame, up to the sign (a tag on each side):
#: flat every 45 deg with the opposites, 40 deg up every 60 deg, and straight up. No direction of the
#: half-space above the table is more than about 30 deg from one of them.
DIRECTIONS = np.array([_unit(a, 0) for a in (0, 45, 90, 135)] + [_unit(a, 40) for a in range(0, 360, 60)]
                      + [_unit(0, 90)])

_AZIMUTHS = ("przod", "przod-lewo", "lewo", "tyl-lewo", "tyl", "tyl-prawo", "prawo", "przod-prawo")


def direction_name(v: np.ndarray) -> str:
    """Coarse Polish name of a direction in the base frame (x forward, y left, z up), for the map."""
    el = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    if el > 60:
        return "w gore"
    az = float(np.degrees(np.arctan2(v[1], v[0])))
    return f"{_AZIMUTHS[round(az / 45) % 8]} {'poziomo' if el < 25 else 'skosnie w gore'}"


#: Every name `direction_name` gives, for "shown in k of n directions".
DIRECTION_NAMES = 2 * len(_AZIMUTHS) + 1


@dataclass
class Sighting:
    """One tag one camera saw at one pose."""

    pose: int
    T_tcp: np.ndarray
    tag: int
    corners: np.ndarray
    joints: dict[str, float] = field(default_factory=dict)


@dataclass
class CameraProgress:
    obs: int = 0
    poses: int = 0
    rotations: list[np.ndarray] = field(default_factory=list)
    #: Every tag this camera saw, for `Session._locate`.
    sightings: list[Sighting] = field(default_factory=list)
    #: Rough T_cam2base the wave aims with; None = not located. `inliers` index the sightings that agree.
    rough: np.ndarray | None = None
    inliers: list[int] = field(default_factory=list)
    #: Aimed poses in a row this camera did not see; sightings since the last locate that agree / do not.
    misses: int = 0
    #: Poses near its sightings in a row this camera did not see (`WaveConfig.follow`).
    follow_misses: int = 0
    agree: int = 0
    against: int = 0
    #: Rough poses dropped so far, and how many sightings there were at the last drop.
    drops: int = 0
    drop_at: int = 0
    #: Where the card was and which way its seen face looked when this camera saw it: the map.
    seen_dirs: Counter = field(default_factory=Counter)
    seen_places: Counter = field(default_factory=Counter)
    #: More observations and rotation [deg] a check fit asked of this camera (`Session._fit_holds`).
    more_obs: int = 0
    more_spread: float = 0.0

    def spread(self) -> float:
        return rotation_spread_R(self.rotations)

    def ready(self, cfg: WaveConfig) -> bool:
        # A fifth more spread than the fit needs: frames it leaves out take some with them.
        return (self.obs >= cfg.min_obs + cfg.spare_obs + self.more_obs and self.poses >= cfg.min_poses
                and self.spread() >= 1.2 * cfg.min_spread + self.more_spread)

    @property
    def hint(self) -> np.ndarray | None:
        """Rough position of the camera in the base frame; None = not located."""
        return None if self.rough is None else self.rough[:3, 3]


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
    #: "szukanie" (exploring), "celowanie" (aimed at `aim`) or "powrot" (back to the previous pose).
    phase: str = ""
    aim: str | None = None


def move_time(q0: np.ndarray, q1: np.ndarray, cfg: WaveConfig) -> float:
    """Ile czasu na przejazd, zeby serwa pozycyjne nie zostawaly w tyle (galaxeo)."""
    travel = float(np.max(np.abs(np.asarray(q1, float) - np.asarray(q0, float))))
    return float(np.clip(1.5 * travel / cfg.speed, *cfg.duration))


def _tag_poses(corners: np.ndarray, size: float, K: np.ndarray, dist: np.ndarray | None) -> list[np.ndarray]:
    """Both poses of a square tag that fit its corners (T_tag2cam): one of them is the mirror image."""
    dist = np.zeros(5) if dist is None else dist
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(corners_in_tag(size), np.asarray(corners, float), K, dist,
                                             flags=cv2.SOLVEPNP_IPPE_SQUARE)
    out = []
    for r, t in zip(rvecs[:n], tvecs[:n]):
        T = np.eye(4)
        T[:3, :3], _ = cv2.Rodrigues(r)
        T[:3, 3] = np.asarray(t).ravel()
        out.append(T)
    return out


class Session:
    """Jedna sesja kalibracji. `step()` robi jedna poze - wygodne dla UI w watku."""

    #: Path samples: the largest joint change between two checked configurations of a move [rad]. At half a
    #: metre from the base 0.02 rad is 1 cm, so between samples the arm comes at most a few mm nearer.
    PATH_STEP = 0.02
    #: Seconds per pose besides the move and `settle_s`: `Twin.move` settles 0.4 s, two frames and the detector.
    POSE_EXTRA_S = 0.6
    #: Check fits before the wave ends (`_fit_holds`).
    MAX_CHECKS = 3

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
        # Cameras whose source knows where they are mounted (`mounts()`, {camera: T_cam2base}) start with
        # that rough pose; sightings refit it or drop it, as any other.
        mounts = getattr(cameras, "mounts", None)
        for name, T in (mounts() if callable(mounts) else {}).items():
            if name in self.progress:
                self.progress[name].rough = np.asarray(T, float)
        self.index = 0
        spec = kin.spec
        self._home = dict(spec.home)
        if spec.gripper:
            self._home[spec.gripper] = 0.0      # karta scisnieta w szczekach przez cala sesje
        self._skipped = 0                       # przejazdy z rzedu, ktore nie dojechaly
        #: Diagnostyka: kazda poza (numer, stawy, co ktora kamera widziala, faza, uwaga).
        self.log: list[dict] = []
        #: Every move as commanded (from, to), and the seconds it took: `move_time`, `Twin.move`'s settle,
        #: `settle_s` and the frames (`POSE_EXTRA_S`).
        self.moves: list[tuple[dict[str, float], dict[str, float]]] = []
        self.motion_s = 0.0
        self._command: dict[str, float] | None = None
        self._trail: list[dict[str, float]] = []
        self._size: dict[str, tuple[int, int]] = {}
        self._setup_joints()
        self._setup_geometry()
        # The exploration map: poses that faced each direction in each base sector, and what was shown.
        self.visits = np.zeros((len(self._sectors), len(DIRECTIONS)), int)
        self.explored = 0
        self.shown_dirs: Counter = Counter()
        self.shown_places: Counter = Counter()
        self._checks = 0
        self._plain: tuple[int, list[str], Fit] | None = None

    # ------------------------------------------------------------- setup
    def _setup_joints(self) -> None:
        spec, cfg, kin = self.kin.spec, self.cfg, self.kin
        #: App limits within the model's range, per arm joint [app units].
        self._limits: dict[str, tuple[float, float]] = {}
        for k, name in enumerate(spec.joints):
            if name == spec.gripper:
                continue
            lo, hi = spec.limits.get(name, (-180.0, 180.0))
            off = np.degrees(kin.offset[k])
            self._limits[name] = (max(lo, np.degrees(kin.lo[k]) - off), min(hi, np.degrees(kin.hi[k]) - off))
        self._roll = spec.roll_joint
        self._pan = spec.joints[0] if spec.joints[0] in spec.wave_ranges else next(iter(spec.wave_ranges), None)
        self._ranges = {n: (max(lo, self._limits[n][0]), min(hi, self._limits[n][1]))
                        for n, (lo, hi) in spec.wave_ranges.items() if n != self._roll}
        if self._pan is not None:
            lo, hi = self._ranges[self._pan]
            lo, hi = max(lo - cfg.turn_extra, self._limits[self._pan][0]), min(hi + cfg.turn_extra,
                                                                              self._limits[self._pan][1])
            if cfg.base_turn is not None:
                lo, hi = max(lo, cfg.base_turn[0]), min(hi, cfg.base_turn[1])
            if not lo < hi:
                raise ValueError(f"obrot podstawy {cfg.base_turn} nie zostawia fali miejsca "
                                 f"(staw {self._pan}: {self._limits[self._pan]})")
            self._ranges[self._pan] = (lo, hi)
            edges = np.linspace(lo, hi, max(1, cfg.sectors) + 1)
        else:
            edges = np.array([0.0, 1.0])
        self._sectors = [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]

    def _setup_geometry(self) -> None:
        kin, spec, card = self.kin, self.kin.spec, self.card
        m = kin.model
        # The card's corners and centre in its own frame (x out of the jaws, the tag's centre at the origin).
        x0, x1, w = card.out / 2 - card.length, card.out / 2, card.width / 2
        self._card_pts = np.array([[x0, -w, 0, 1], [x0, w, 0, 1], [x1, -w, 0, 1], [x1, w, 0, 1], [0, 0, 0, 1.0]]).T
        self._tips = [m.geom(kin.prefix + g).id for g in spec.fingertips if g]
        # Bodies of the arm (the base's subtree) and the links between them, for keep-out boxes.
        base = kin.base_id
        sub = []
        for b in range(m.nbody):
            p = b
            while p > 0 and p != base:
                p = m.body_parentid[p]
            if p == base and b != base:
                sub.append(b)
        self._links = [(int(m.body_parentid[b]), b) for b in sub]
        self._roll_jid = m.joint(kin.prefix + self._roll).id if self._roll else None
        self._hand = spec.hand_body

    # --------------------------------------------------------------- stan
    @property
    def done(self) -> bool:
        if self.index >= self.cfg.max_poses:
            return True
        if not all(p.ready(self.cfg) or self._given_up(n) for n, p in self.progress.items()):
            return False
        return self._fit_holds()

    def _fit_holds(self) -> bool:
        """Fits once every camera looks ready. A camera the fit finds short of observations or rotation
        (the frames it leaves out as outliers take some with them) is asked for more, `MAX_CHECKS` times."""
        if self._checks >= self.MAX_CHECKS or not self.obs:
            return True
        self._checks += 1
        short = False
        for name, c in self._plain_fit()[1].cameras.items():
            p, need_obs, need_spread = self.progress[name], self.cfg.min_obs - c.n_obs, self.cfg.min_spread - c.spread_deg
            if need_obs > 0 or need_spread > 0:
                p.more_obs += max(0, need_obs) + 2 * (need_obs > 0)
                p.more_spread += 1.5 * max(0.0, need_spread)
                short = True
        return not short

    def _swept(self) -> bool:
        """Every direction shown in every sector of the base's turn, or half as many tries again (some
        directions cannot be shown in some sectors)."""
        return bool((self.visits > 0).all()) or self.explored >= 1.5 * self.visits.size

    def _given_up(self, cam: str) -> bool:
        """A camera that saw nothing in a whole sweep: more of the same will not show it the card."""
        return not self.progress[cam].sightings and self._swept()

    def _report(self, joints, seen, note, frames, corners, phase="", aim=None) -> StepReport:
        prog = {n: (p.obs, p.poses, p.spread()) for n, p in self.progress.items()}
        return StepReport(self.index, joints, seen, prog, self.done, note, frames, corners, phase, aim)

    # ---------------------------------------------------------- geometry
    def _base(self, world: np.ndarray) -> np.ndarray:
        """(N, 3) world points of the kinematics' last pose in the base frame."""
        B = self.kin._base_inv()
        return world @ B[:3, :3].T + B[:3, 3]

    def _points(self, joints: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(card corners and centre, TCP and jaw tips, points along the arm's links), base frame [m]."""
        T = self.kin.tcp(joints)
        card = (T @ self.card_nominal @ self._card_pts)[:3].T
        d = self.kin.data
        grip = np.vstack([T[:3, 3], self._base(d.geom_xpos[self._tips])]) if self._tips else T[:3, 3][None]
        pos = self._base(d.xpos)
        arm = [pos[a] + s * (pos[b] - pos[a]) for a, b in self._links
               for s in np.linspace(0.0, 1.0, 1 + max(1, int(np.ceil(np.linalg.norm(pos[b] - pos[a]) / 0.03))))]
        return card, grip, np.array(arm) if arm else np.zeros((0, 3))

    def _lowest(self, joints: Mapping[str, float]) -> float:
        """Height of the lowest card corner or gripper point above the table [m]."""
        card, grip, _ = self._points(joints)
        return float(min(card[:, 2].min(), grip[:, 2].min()))

    def _problem(self, joints: Mapping[str, float], floor: float | None = None) -> str:
        """Why the arm may not be at `joints` ("" = it may): card or gripper below `floor` (default
        `table_margin`) above the table, or anything of the arm in a keep-out box."""
        cfg = self.cfg
        floor = cfg.table_margin if floor is None else floor
        card, grip, arm = self._points(joints)
        low = float(min(card[:, 2].min(), grip[:, 2].min()))
        if low < floor:
            return f"karta albo chwytak {low * 100:.0f} cm nad stolem"
        for box in cfg.keepout:
            for what, pts in (("karta", card), ("chwytak", grip), ("ramie", arm)):
                if len(pts) and box.hit(pts, cfg.keepout_margin):
                    return f"{what} w strefie {box.name}"
        return ""

    def _path_problem(self, a: Mapping[str, float], b: Mapping[str, float]) -> str:
        """Why the straight move a -> b in joint space is not allowed, checked along the way ("" = it is).

        A move from below the table margin (the rest pose of a short arm) may not go any lower."""
        qa, qb = self.kin.to_q(a), self.kin.to_q(b)
        floor = min(self.cfg.table_margin, self._lowest(a) - 0.005)
        n = max(1, int(np.ceil(float(np.max(np.abs(qb - qa))) / self.PATH_STEP)))
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            why = self._problem(self.kin.from_q(qa + s * (qb - qa)), floor)
            if why:
                return f"{why} w {s:.0%} drogi"
        if self.checker is not None:
            return self.checker.path_clear(a, b, step_rad=self.PATH_STEP) or ""
        return ""

    def _move_problem(self, current: Mapping[str, float], target: Mapping[str, float]) -> str:
        """The move to `target` from where the arm stands and from the last command (the twin ramps from it)."""
        starts = [current]
        if self._command is not None and self._far_from(self._command, current, tol=1.0):
            starts.append(self._command)
        for start in starts:
            why = self._path_problem(start, target)
            if why:
                return why
        return ""

    def _family(self, joints: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """(T_tcp, card normal, roll axis) in the base frame; turning the roll joint by t [rad] turns the
        tool by t about that axis (None without a roll joint)."""
        T = self.kin.tcp(joints)
        n = (T @ self.card_nominal)[:3, 2]
        if self._roll_jid is None:
            return T, n, None
        B = self.kin._base_inv()
        return T, n, B[:3, :3] @ self.kin.data.xaxis[self._roll_jid]

    @staticmethod
    def _facing(n: np.ndarray, axis: np.ndarray | None, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For unit directions d (k, 3): the best |n . d| over the roll and the roll [rad, from now] giving it.

        n . d as a function of the roll t is C + A cos t + B sin t; either side of the card will do."""
        if axis is None:
            return np.abs(d @ n), np.zeros(len(d))
        C = (axis @ n) * (d @ axis)
        A = d @ n - C
        B = d @ np.cross(axis, n)
        R = np.hypot(A, B)
        t = np.arctan2(B, A)
        return np.abs(C) + R, np.where(C >= 0, t, t + np.pi)

    def _roll_spans(self, now: float, centres: list[float], slack: float) -> list[tuple[float, float]]:
        """Roll ranges [app units] within `slack` [rad] of a centre or half a turn from it (relative to `now`),
        inside the joint's limits and one `step` of now."""
        lo, hi = self._limits[self._roll] if self._roll else (now, now)
        step = np.degrees(self.cfg.step)
        out = []
        for c in centres:
            for half in (0.0, np.pi):
                for turn in (-2 * np.pi, 0.0, 2 * np.pi):
                    mid = now + np.degrees(c + half + turn)
                    a = max(mid - np.degrees(slack), lo, now - step)
                    b = min(mid + np.degrees(slack), hi, now + step)
                    if a <= b:
                        out.append((a, b))
        return out

    def _sample(self, current: Mapping[str, float]) -> dict[str, float]:
        """Joints one move from `current`: each wave joint uniform within one `step` of where it is."""
        joints = dict(self._home)
        joints.update({k: v for k, v in current.items() if k != self.kin.spec.gripper})
        step = np.degrees(self.cfg.step)
        for name, (lo, hi) in self._ranges.items():
            c = float(current.get(name, self._home.get(name, 0.0)))
            a, b = max(lo, c - step), min(hi, c + step)
            if a > b:
                a, b = lo, hi
            joints[name] = float(self.rng.uniform(a, b))
        return joints

    def _reach_problem(self, joints: Mapping[str, float]) -> str:
        """Cheap checks of a planned pose before its moves are checked."""
        T = self.kin.tcp(joints)
        if T[2, 3] < self.cfg.min_tcp_height:
            return "TCP za nisko"
        if not self.cfg.radius[0] <= float(np.hypot(T[0, 3], T[1, 3])) <= self.cfg.radius[1]:
            return "TCP poza promieniem fali"
        if self._pan is not None and self.cfg.base_turn is not None:
            lo, hi = self.cfg.base_turn
            if not lo <= joints.get(self._pan, 0.0) <= hi:
                return f"podstawa poza zakresem {lo:+.0f}..{hi:+.0f} st."
        return self._problem(joints)

    def _sector(self, pan: float) -> int:
        lo, hi = self._sectors[0][0], self._sectors[-1][1]
        k = int((pan - lo) / max(hi - lo, 1e-9) * len(self._sectors))
        return min(max(k, 0), len(self._sectors) - 1)

    def _place(self, joints: Mapping[str, float], centre: np.ndarray) -> str:
        """Coarse name of where the card is: the base's sector, near or far, low or high."""
        lo, hi = self._sectors[self._sector(joints.get(self._pan, 0.0) if self._pan else 0.0)]
        far = float(np.hypot(centre[0], centre[1])) >= sum(self.cfg.radius) / 2
        return (f"podstawa {lo:+.0f}..{hi:+.0f} st., {'daleko' if far else 'blisko'}, "
                f"{'wysoko' if centre[2] >= 0.2 else 'nisko'}")

    # -------------------------------------------------------------- fala
    def _target_camera(self) -> str | None:
        """Kamera, ktora najbardziej potrzebuje karty - i ktora juz namierzylismy."""
        located = [(n, p) for n, p in self.progress.items() if p.rough is not None and not p.ready(self.cfg)]
        if not located:
            return None

        def need(item):
            _, p = item
            return min(p.obs / self.cfg.min_obs, p.poses / self.cfg.min_poses, p.spread() / self.cfg.min_spread)
        return min(located, key=need)[0]

    def _explore(self, current: Mapping[str, float]) -> dict[str, float] | None:
        """A pose showing the card in a direction and a place shown least so far, one move from here."""
        cfg = self.cfg
        now = float(current.get(self._roll, 0.0)) if self._roll else 0.0
        q_now = self.kin.to_q(current)
        dir_visits = self.visits.sum(axis=0)
        options = []
        up = DIRECTIONS[:, 2] > 0.1
        for _ in range(cfg.samples):
            j = self._sample(current)
            T, n, axis = self._family(j)
            centre = (T @ self.card_nominal)[:3, 3]
            if centre[2] > cfg.explore_top:
                continue
            q, roll = self._facing(n, axis, DIRECTIONS)
            # A camera up there sees the face turned up; looking at it from behind the jaws it loses the tag.
            q = np.where(up & (DIRECTIONS @ (T[:3, :3] @ self.kin._approach) < -cfg.jaw_side), 0.0, q)
            s = self._sector(j.get(self._pan, 0.0) if self._pan else 0.0)
            place = self._place(j, centre)
            travel = float(np.max(np.abs(self.kin.to_q(j) - q_now))) / cfg.step
            score = (4.0 * self.visits[s] + dir_visits + 0.5 * self.shown_places[place] + 0.5 * travel
                     + 100.0 * (q < cfg.min_facing) + self.rng.uniform(0, 0.3, len(q)))
            k = int(np.argmin(score))
            options.append((float(score[k]), j, float(q[k]), float(roll[k])))
        options.sort(key=lambda o: o[0])
        tried = 0
        for _, j, q, roll in options:
            if self._roll:
                slack = float(np.arccos(min(1.0, cfg.min_facing / q))) if q > cfg.min_facing else 0.0
                spans = self._roll_spans(now, [roll], slack)
                if not spans:
                    continue
                j = dict(j, **{self._roll: min((float(np.clip(now, a, b)) for a, b in spans),
                                               key=lambda r: abs(r - now))})
            if self._reach_problem(j):
                continue
            if self._move_problem(current, j):
                tried += 1
                if tried >= cfg.candidates:
                    break
                continue
            return j
        return None

    def _frame_offset(self, cam: str, point: np.ndarray) -> float | None:
        """How far from the frame's middle the point falls by the camera's rough pose (0 middle, 0.5 edge);
        None outside the frame less `frame_margin`."""
        T = self.progress[cam].rough
        if T is None:
            return 0.0
        p = np.linalg.inv(T) @ np.r_[point, 1.0]
        if p[2] < 0.1:
            return None
        K = self.intrinsics[cam][0]
        u, v = (K @ p[:3])[:2] / p[2]
        W, H = self._size.get(cam, (2 * K[0, 2] + 1, 2 * K[1, 2] + 1))   # bez kadru: rozmiar z punktu glownego
        off = max(abs(u / W - 0.5), abs(v / H - 0.5))
        return None if off > 0.5 - self.cfg.frame_margin else float(off)

    def _aim(self, current: Mapping[str, float], cam: str) -> dict[str, float] | None:
        """A pose with the card facing camera `cam`, in its frame and in its sight, adding the most rotation."""
        cfg, p = self.cfg, self.progress[cam]
        eye = p.hint
        assert eye is not None
        now = float(current.get(self._roll, 0.0)) if self._roll else 0.0
        cos_max = float(np.cos(np.radians(cfg.aim_angle)))
        # The side this camera saw clearly more often (a jaw can hide the other tag's border): mostly that one.
        counts = [*Counter(s.tag for s in p.sightings).most_common(), (None, 0), (None, 0)]
        prefer = counts[0][0] if counts[0][1] >= 3 and counts[0][1] >= 2 * counts[1][1] else None
        cands = []
        for _ in range(cfg.samples):
            j = self._sample(current)
            T, n, axis = self._family(j)
            centre = (T @ self.card_nominal)[:3, 3]
            d = (eye - centre) / np.linalg.norm(eye - centre)
            approach = T[:3, :3] @ self.kin._approach
            if abs(float(approach @ d)) > cfg.edge_on or float(approach @ d) < -cfg.jaw_side:
                continue                                  # edge-on, or seen past the jaws
            q, roll = self._facing(n, axis, d[None])
            if q[0] < cos_max:
                continue
            if self._roll:
                # Either side of the card, and a spread around "facing the camera": the third rotation axis,
                # without which hand-eye is badly conditioned.
                slack = min(cfg.roll_jitter, float(np.arccos(min(1.0, cos_max / q[0]))))
                spans = self._roll_spans(now, [float(roll[0])], slack)
                if not spans:
                    continue
                a, b = spans[int(self.rng.integers(len(spans)))]
                j = dict(j, **{self._roll: float(self.rng.uniform(a, b))})
            if self._reach_problem(j):
                continue
            T = self.kin.tcp(j)
            card = T @ self.card_nominal
            centre = card[:3, 3]
            shown = self.card.ids[0] if card[:3, 2] @ (eye - centre) > 0 else self.card.ids[1]
            if prefer is not None and shown != prefer and self.rng.random() < 0.8:
                continue
            off = self._frame_offset(cam, centre)
            if off is None:
                continue
            if self.checker is not None and self.checker.sight_blocked(j, centre, eye, self._hand):
                continue
            cands.append((j, T[:3, :3], off))
            if len(cands) >= cfg.candidates:
                break
        # The most new rotation, and nearer the middle of the frame (the rough pose is rough).
        cands.sort(key=lambda c: -rotation_spread_R([*p.rotations, c[1]]) + 20.0 * c[2])
        for j, _, _ in cands:
            if not self._move_problem(current, j):
                return j
        return None

    def _follow(self, current: Mapping[str, float], cam: str) -> dict[str, float] | None:
        """A pose near one this camera saw the card in: small joint and roll changes, so the next sightings
        come from different poses that should still show it the card. Of a few, the one adding most rotation."""
        cfg, p = self.cfg, self.progress[cam]
        q_now = self.kin.to_q(current)
        found: list[tuple[dict[str, float], np.ndarray]] = []
        for _ in range(cfg.samples):
            seen = p.sightings[int(self.rng.integers(len(p.sightings)))].joints
            j = {**seen, **({self.kin.spec.gripper: self._home[self.kin.spec.gripper]} if self.kin.spec.gripper else {})}
            for name in [*self._ranges, *([self._roll] if self._roll else [])]:
                lo, hi = self._ranges.get(name) or self._limits[name]
                spread = 25.0 if name == self._roll else 8.0
                j[name] = float(np.clip(seen.get(name, 0.0) + self.rng.normal(0.0, spread), lo, hi))
            if float(np.max(np.abs(self.kin.to_q(j) - q_now))) > cfg.step:
                continue
            if self._reach_problem(j) or self._move_problem(current, j):
                continue
            found.append((j, self.kin.tcp(j)[:3, :3]))
            if len(found) >= 6:
                break
        if not found:
            return None
        return max(found, key=lambda f: rotation_spread_R([*p.rotations, f[1]]))[0]

    def _next_pose(self, current: dict[str, float]) -> tuple[dict[str, float] | None, str, str | None]:
        """(pose, phase, camera): aimed at a located camera, near a sighting of one not located yet, or exploring.

        Exploration comes first (`explore_min`); then aimed poses take turns with the search while a
        camera is not located."""
        cfg = self.cfg
        target = self._target_camera()
        lost = [n for n, p in self.progress.items() if p.rough is None and not self._given_up(n)]
        search = target is None or self.explored < cfg.explore_min or (bool(lost) and self.index % 2 == 1)
        if not search and target is not None:
            # Aimed poses that miss (the arm or the jaws in the way, a rough pose a bit off): near the poses
            # the camera did see the card in, until one hits again.
            j = None
            if self.progress[target].misses >= 2:
                j = self._follow(current, target)
            j = j or self._aim(current, target)
            if j is not None:
                return j, "celowanie", target
        near = [n for n in lost if self.progress[n].sightings and self.progress[n].follow_misses < cfg.follow]
        if near and self.index % 3 != 0:
            cam = min(near, key=lambda n: len(self.progress[n].sightings))
            j = self._follow(current, cam)
            if j is not None:
                return j, "szukanie", cam
        return self._explore(current), "szukanie", None

    # -------------------------------------------------------- evidence
    def _corners_base(self, s: Sighting) -> np.ndarray:
        """(4, 3) corners of the sighted tag in the base frame, on the nominal card."""
        T = s.T_tcp @ self.card_nominal @ self.tags[s.tag]
        return (T @ np.c_[corners_in_tag(self.card.tag_size), np.ones(4)].T)[:3].T

    def _tol(self, s: Sighting) -> float:
        """How far [px] a sighting's corners may sit from where a rough pose puts them and still agree."""
        side = float(np.mean(np.linalg.norm(s.corners - np.roll(s.corners, 1, axis=0), axis=1)))
        return self.cfg.locate_tol * side + 6.0

    @staticmethod
    def _errors(X: np.ndarray, U: np.ndarray, T_cam2base: np.ndarray, K: np.ndarray,
                dist: np.ndarray | None) -> np.ndarray:
        """RMS reprojection error [px] of each sighting's corners X (n, 4, 3) against U (n, 4, 2); inf behind."""
        Ti = np.linalg.inv(T_cam2base)
        P = X.reshape(-1, 3) @ Ti[:3, :3].T + Ti[:3, 3]
        behind = (P[:, 2] < 0.05).reshape(-1, 4).any(axis=1)
        P[:, 2] = np.maximum(P[:, 2], 0.05)
        px, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, np.zeros(5) if dist is None else dist)
        err = np.sqrt(np.mean(np.sum((px.reshape(-1, 4, 2) - U) ** 2, axis=2), axis=1))
        return np.where(behind, np.inf, err)

    @staticmethod
    def _fit_pose(X: np.ndarray, U: np.ndarray, T_cam2base: np.ndarray, K: np.ndarray,
                  dist: np.ndarray | None) -> np.ndarray:
        """The camera pose that best reprojects all these corners with the nominal card, from `T_cam2base`."""
        Ti = np.linalg.inv(T_cam2base)
        rvec, _ = cv2.Rodrigues(Ti[:3, :3])
        tvec = Ti[:3, 3].reshape(3, 1).copy()
        ok, rvec, tvec = cv2.solvePnP(X.reshape(-1, 3), U.reshape(-1, 2).astype(float), K,
                                      np.zeros(5) if dist is None else dist, rvec, tvec, useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return T_cam2base
        T = np.eye(4)
        T[:3, :3], _ = cv2.Rodrigues(rvec)
        T[:3, 3] = tvec.ravel()
        return np.linalg.inv(T)

    def _locate(self, cam: str, fresh: bool = True) -> tuple[np.ndarray, list[int]] | None:
        """A rough camera pose most sightings agree on, or None while the evidence is not solid.

        Every sighting proposes two poses (its PnP and the mirror image). A proposal is kept when it puts
        the other sightings' corners near where they were seen; the best is refitted to all that agree
        and judged again with a tighter tolerance. Solid means `locate_min` agreeing sightings from
        different poses, half of them all, and the card moved between them. After a drop (`fresh`)
        one of them must be new."""
        cfg, p = self.cfg, self.progress[cam]
        S = p.sightings
        if len({s.pose for s in S}) < cfg.locate_min:
            return None
        K, dist = self.intrinsics[cam]
        X = np.stack([self._corners_base(s) for s in S])
        U = np.stack([np.asarray(s.corners, float) for s in S])
        tol = np.array([self._tol(s) for s in S])
        loose = np.maximum(2.0 * tol, K[0, 0] * np.tan(np.radians(cfg.loose_deg)))
        hyps = []
        for s in S:
            for T_tag2cam in _tag_poses(s.corners, self.card.tag_size, K, dist):
                E = s.T_tcp @ self.card_nominal @ self.tags[s.tag] @ np.linalg.inv(T_tag2cam)
                err = self._errors(X, U, E, K, dist)
                hyps.append((int((err < loose).sum()), float(np.median(err)), E))
        hyps.sort(key=lambda h: (-h[0], h[1]))
        best = None
        for _, _, E in hyps[:4]:
            idx = np.flatnonzero(self._errors(X, U, E, K, dist) < loose)
            for _ in range(4):
                if len(idx) < 2:
                    break
                E = self._fit_pose(X[idx], U[idx], E, K, dist)
                err = self._errors(X, U, E, K, dist)
                new = np.flatnonzero(err < tol)
                if set(new.tolist()) == set(idx.tolist()):
                    break
                idx = new
            if not self._solid(cam, idx.tolist(), fresh):
                continue
            rms = float(np.sqrt(np.mean(self._errors(X[idx], U[idx], E, K, dist) ** 2)))
            if best is None or (len(idx), -rms) > (len(best[1]), -best[2]):
                best = (E, idx.tolist(), rms)
        return None if best is None else (best[0], best[1])

    def _solid(self, cam: str, idx: list[int], fresh: bool) -> bool:
        cfg, p = self.cfg, self.progress[cam]
        S = [p.sightings[i] for i in idx]
        if len({s.pose for s in S}) < cfg.locate_min or 2 * len(idx) < len(p.sightings):
            return False
        if fresh and p.drops and max(idx) < p.drop_at:
            return False
        centres = np.array([(s.T_tcp @ self.card_nominal)[:3, 3] for s in S])
        moved = float(np.max(np.linalg.norm(centres[:, None] - centres[None], axis=2)))
        turned = max(float(np.linalg.norm(cv2.Rodrigues(a.T_tcp[:3, :3].T @ b.T_tcp[:3, :3])[0]))
                     for a in S for b in S)
        return moved >= 0.04 or turned >= np.radians(10.0)

    def _refit(self, cam: str) -> None:
        p = self.progress[cam]
        K, dist = self.intrinsics[cam]
        S = [p.sightings[i] for i in p.inliers]
        assert p.rough is not None
        if not S:                                       # a rough pose from the mount, nothing agrees yet
            return
        p.rough = self._fit_pose(np.stack([self._corners_base(s) for s in S]),
                                 np.stack([np.asarray(s.corners, float) for s in S]), p.rough, K, dist)

    def _drop(self, cam: str, why: str) -> str:
        p = self.progress[cam]
        p.rough, p.inliers, p.misses, p.agree, p.against, p.follow_misses = None, [], 0, 0, 0, 0
        p.drops += 1
        p.drop_at = len(p.sightings)
        return f"{cam}: przyblizone polozenie odrzucone ({why}) - szukam jej dalej"

    def _adopt(self, cam: str, found: tuple[np.ndarray, list[int]]) -> None:
        p = self.progress[cam]
        p.rough, p.inliers = found
        p.misses = p.agree = p.against = 0

    def _weigh(self, new: dict[str, list[int]], aimed: str | None, looked: set[str]) -> list[str]:
        """Update the rough poses with this pose's sightings; returns notes for the log."""
        notes = []
        for cam, idx in new.items():
            p = self.progress[cam]
            if p.rough is None:
                found = self._locate(cam)
                if found is not None:
                    self._adopt(cam, found)
                    x, y, z = found[0][:3, 3]
                    notes.append(f"{cam} namierzona: ok. x {x:+.2f}, y {y:+.2f}, z {z:.2f} m "
                                 f"({len(found[1])} zgodnych detekcji)")
                continue
            K, dist = self.intrinsics[cam]
            S = [p.sightings[i] for i in idx]
            err = self._errors(np.stack([self._corners_base(s) for s in S]),
                               np.stack([np.asarray(s.corners, float) for s in S]), p.rough, K, dist)
            for i, e, s in zip(idx, err, S):
                if e <= self._tol(s):
                    p.inliers.append(i)
                    p.agree += 1
                else:
                    p.against += 1
            if p.against >= 3 and p.against > p.agree:
                found = self._locate(cam, fresh=False)
                if found is None:
                    notes.append(self._drop(cam, f"{p.against} nowych detekcji sie z nim nie zgadza"))
                else:
                    self._adopt(cam, found)
            else:
                self._refit(cam)
        for cam in new:
            self.progress[cam].follow_misses = 0
        if aimed is not None and aimed in looked:
            p = self.progress[aimed]
            if p.rough is None:
                p.follow_misses += aimed not in new
            else:
                p.misses = 0 if aimed in new else p.misses + 1
                # Misses are weak evidence: they drop a pose only few sightings stand behind.
                if p.misses >= self.cfg.max_misses and len(p.inliers) < 2 * self.cfg.locate_min:
                    notes.append(self._drop(aimed, f"{p.misses} poz celowanych w nia nie pokazalo jej karty"))
        return notes

    def _record_shown(self, joints: Mapping[str, float], T_tcp: np.ndarray, explored: bool) -> str:
        """Counts what this pose showed: directions per sector, places; returns the place's name."""
        T = T_tcp @ self.card_nominal
        n, centre = T[:3, 2], T[:3, 3]
        s = self._sector(joints.get(self._pan, 0.0) if self._pan else 0.0)
        self.visits[s] += np.abs(DIRECTIONS @ n) >= self.cfg.min_facing
        self.explored += int(explored)
        for v in (n, -n):
            if v[2] > -0.25:                                 # faces the half-space above the table
                self.shown_dirs[direction_name(v)] += 1
        place = self._place(joints, centre)
        self.shown_places[place] += 1
        return place

    # -------------------------------------------------------------- krok
    def _move(self, current: Mapping[str, float], target: dict[str, float]) -> None:
        q0 = self.kin.to_q(self._command if self._command is not None else current)
        duration = move_time(q0, self.kin.to_q(target), self.cfg)
        self.moves.append((dict(self._command if self._command is not None else current), dict(target)))
        self.motion_s += duration + self.cfg.settle_s + self.POSE_EXTRA_S
        self._command = dict(target)
        self.robot.move(target, duration)

    def step(self) -> StepReport:
        """Jedna poza: przejazd, kadry ze wszystkich kamer, detekcja."""
        current = self.robot.joints()
        if self.index == 0:
            why = self._problem(self._home, floor=-np.inf)
            if why:
                raise RuntimeError(f"pozycja domowa: {why} - popraw strefy zakazane albo dom")
            if self._far_from(current, self._home):
                why = self._path_problem(current, self._home)
                if why:
                    raise RuntimeError(f"droga do pozycji domowej: {why} - ustaw ramie blizej domu")
                self._move(current, self._home)
                current = self.robot.joints()
            self._trail = [dict(self._home)]
        target, phase, aim = self._next_pose(current)
        note = ""
        if target is None:
            # Back the way the arm came: that move was checked, so its reverse is clear too.
            target = self._trail[-2] if len(self._trail) >= 2 else dict(self._home)
            phase, aim = "powrot", None
            note = "brak czystej pozy stad - wracam do poprzedniej"
            if len(self._trail) < 2 and self._move_problem(current, target):
                raise RuntimeError("brak czystej pozy i drogi do domu stad - ustaw ramie recznie")
        try:
            self._move(current, target)
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
            return self._report(self.robot.joints(), {}, f"przejazd pominiety: {exc}", {}, {}, phase, aim)
        self._skipped = 0
        self.index += 1
        if phase == "powrot":
            self._trail.pop()
        else:
            self._trail = [*self._trail[-20:], dict(target)]
        t_stop = time.monotonic()

        measured = self.robot.joints()
        T_tcp = self.kin.tcp(measured)                   # FK z ZMIERZONYCH katow: blad serw w dopasowaniu
        place = self._record_shown(measured, T_tcp, phase == "szukanie")
        n = (T_tcp @ self.card_nominal)[:3, 2]
        cfg = self.cfg
        frames, tags, moving = still_frames(self.cameras, t_stop, cfg.settle_s, cfg.still_px, cfg.still_timeout,
                                            self.tags)
        seen, corners_out = {}, {}
        new: dict[str, list[int]] = {}
        for cam, img in frames.items():
            if cam not in self.intrinsics:
                continue
            self._size[cam] = (int(img.shape[1]), int(img.shape[0]))
            found = tags.get(cam, {})
            seen[cam] = sorted(found)
            corners_out[cam] = found
            if not found:
                continue
            p = self.progress[cam]
            for tid, corners in found.items():
                # Kazdy tag to osobny "montaz" o wlasnej, swobodnej pozie w chwytaku. Wspolna poza
                # karty zakladala, ze oba tagi maja DOKLADNIE ten sam srodek plecami do siebie; dwa
                # wydruki sklejone recznie z przesunieciem kilku mm dawaly 7-10 px residuum na obu
                # kamerach naraz (kazda widziala inna strone; 2026-09-26).
                self.obs.append(Observation(T_tcp, np.eye(4), corners, tid, f"tag{tid}", cam, self.index,
                                            joints=dict(measured)))
                p.obs += 1
                p.sightings.append(Sighting(self.index, T_tcp, tid, np.asarray(corners, float), dict(measured)))
                new.setdefault(cam, []).append(len(p.sightings) - 1)
                p.seen_dirs[direction_name(n if tid == self.card.ids[0] else -n)] += 1
            p.poses += 1
            p.rotations.append(T_tcp[:3, :3])
            p.seen_places[place] += 1
        notes = [note] if note else []
        if moving & set(self.intrinsics):
            notes.append(f"karta sie ruszala w kadrze: {', '.join(sorted(moving & set(self.intrinsics)))} - pominiete")
        notes += self._weigh(new, aim, {c for c in frames if c in self.intrinsics and c not in moving})
        note = "; ".join(notes)
        self.log.append({"poza": self.index, "stawy": {k: round(float(v), 2) for k, v in measured.items()},
                         "widzi": {c: list(t) for c, t in seen.items()}, "faza": phase, "cel": aim,
                         "miejsce": place, "karta": direction_name(n if n[2] >= 0 else -n), "uwaga": note})
        return self._report(measured, seen, note, frames if self.keep_frames else {}, corners_out, phase, aim)

    def mounts_nominal(self) -> dict[str, np.ndarray]:
        """Nominalna poza kazdego taga w ukladzie TCP: karta nominalna razy tag na karcie."""
        return {f"tag{tid}": self.card_nominal @ T for tid, T in self.tags.items()}

    @staticmethod
    def _far_from(a: Mapping[str, float], b: Mapping[str, float], tol: float = 3.0) -> bool:
        return any(abs(a.get(k, 0.0) - v) > tol for k, v in b.items())

    # ------------------------------------------------------------ report
    def status(self) -> dict[str, str]:
        """One line per camera for the panel while the wave runs (Polish)."""
        cfg, out = self.cfg, {}
        for cam, p in self.progress.items():
            if p.ready(cfg):
                out[cam] = f"gotowa: {p.obs} obserwacji z {p.poses} poz, rozrzut {p.spread():.0f} st."
            elif p.rough is not None and not p.sightings:
                x, y, z = p.rough[:3, 3]
                out[cam] = f"z mocowania (ok. x {x:+.2f}, y {y:+.2f}, z {z:.2f} m) - jeszcze nie widziala karty"
            elif p.rough is not None:
                x, y, z = p.rough[:3, 3]
                out[cam] = (f"namierzona (ok. x {x:+.2f}, y {y:+.2f}, z {z:.2f} m): {p.obs}/{cfg.min_obs} obserwacji, "
                            f"{p.poses}/{cfg.min_poses} poz, rozrzut {p.spread():.0f}/{cfg.min_spread:.0f} st.")
            elif p.sightings:
                top = ", ".join(k for k, _ in p.seen_dirs.most_common(2))
                out[cam] = (f"szukam - widziala karte {len(p.sightings)} razy (karta zwrocona: {top}), "
                            f"polozenie jeszcze niepewne")
            elif self._given_up(cam):
                out[cam] = "nie widzi karty z zadnego miejsca ani kierunku"
            else:
                out[cam] = "szukam - jeszcze nie widziala karty"
        return out

    def explain(self, cam: str) -> str:
        """Which camera saw the card how often and from where - for a wave that did not solve (Polish)."""
        p = self.progress[cam]
        if not p.sightings:
            return (f"{cam}: nie widziala karty ani razu w {self.index} pozach (karta zwrocona w "
                    f"{len(self.shown_dirs)} z {DIRECTION_NAMES} kierunkow, w {len(self.shown_places)} miejscach)")
        dirs = ", ".join(f"{k} ({v})" for k, v in p.seen_dirs.most_common(3))
        places = ", ".join(f"{k} ({v})" for k, v in p.seen_places.most_common(2))
        if p.rough is not None:
            x, y, z = p.rough[:3, 3]
            where = f"przyblizone polozenie x {x:+.2f}, y {y:+.2f}, z {z:.2f} m"
        elif len({s.pose for s in p.sightings}) >= self.cfg.locate_min:
            where = "polozenie nieustalone - detekcje sie ze soba nie zgadzaly"
        else:
            where = "za malo detekcji, zeby ustalic polozenie"
        dropped = f", {p.drops} razy odrzucone" if p.drops else ""
        return (f"{cam}: widziala karte {len(p.sightings)} razy w {p.poses} z {self.index} poz; karta zwrocona: "
                f"{dirs}; miejsca: {places}; {where}{dropped}")

    def summary(self) -> dict:
        """The wave's map for the record: what was shown and what each camera saw, by direction and place."""
        cams = {}
        for cam, p in self.progress.items():
            cams[cam] = {"widziala": len(p.sightings), "pozy": p.poses, "gotowa": p.ready(self.cfg),
                         "polozenie_m": None if p.rough is None else np.round(p.rough[:3, 3], 3).tolist(),
                         "odrzucone_polozenia": p.drops, "kierunki": dict(p.seen_dirs),
                         "miejsca": dict(p.seen_places), "opis": self.explain(cam)}
        return {"pozy": self.index, "ruch_s": round(self.motion_s, 1),
                "pokazane": {"kierunki": dict(self.shown_dirs), "miejsca": dict(self.shown_places)},
                "kamery": cams}

    # ------------------------------------------------------------- wynik
    def _plain_fit(self) -> tuple[list[str], Fit]:
        """(cameras, fit without servo offsets, judged) of the observations so far; kept until there are more."""
        if self._plain is None or self._plain[0] != len(self.obs):
            cams = sorted({o.camera for o in self.obs})
            fit = solve(self.obs, {c: self.intrinsics[c] for c in cams}, self.card.tag_size,
                        self.mounts_nominal(), max_px=self.cfg.max_px, min_keep=self.cfg.min_obs)
            self._plain = (len(self.obs), cams, judge(fit, self.cfg.max_px, self.cfg.min_spread, self.cfg.min_obs))
        return self._plain[1], self._plain[2]

    def solve(self) -> Fit:
        """Dopasowanie wszystkich kamer naraz i werdykt dla kazdej z osobna."""
        if not self.obs:
            raise RuntimeError(f"zadna kamera nie zobaczyla karty w {self.index} pozach - "
                               + "; ".join(self.explain(c) for c in self.progress))
        cams, fit = self._plain_fit()
        self._plain = None                              # the offsets below change the fit
        return self._with_offsets(fit, cams)

    def _with_offsets(self, fit: Fit, cams: list[str]) -> Fit:
        """`fit` again with servo zero offsets, when they bring the residual down (`WaveConfig.fit_offsets`)."""
        cfg, names = self.cfg, offset_joints(self.kin.spec)
        if not cfg.fit_offsets or not names or fit.n_obs < cfg.min_obs:
            return fit
        intr = {c: self.intrinsics[c] for c in cams}
        try:
            new = solve_offsets(fit, self.obs, intr, self.card.tag_size, self.kin.tcp, names)
            # The offsets can show frames that the fit without them hid among residuals of their size:
            # fit again without those, plain and with offsets.
            bad = self._outliers(new)
            if bad:
                obs = [o for o in self.obs if (o.camera, int(o.pose), int(o.tag_id)) not in bad]
                plain = solve(obs, intr, self.card.tag_size, self.mounts_nominal(), max_px=cfg.max_px,
                              min_keep=cfg.min_obs)
                again = solve_offsets(plain, obs, intr, self.card.tag_size, self.kin.tcp, names)
                if again.rms_px < new.rms_px:
                    for c in again.cameras.values():
                        c.dropped += sum(1 for cam, _, _ in bad if again.cameras.get(cam) is c)
                    judge(plain, cfg.max_px, cfg.min_spread, cfg.min_obs)
                    fit, new = plain, again
        except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
            fit.offsets_note = f"poprawki zer serw: nie policzone ({exc})"
            return fit
        worst = max(names, key=lambda n: abs(new.joint_offsets[n]))
        if abs(new.joint_offsets[worst]) > cfg.max_offset:
            fit.offsets_note = (f"poprawki zer serw pominiete: {worst} {new.joint_offsets[worst]:+.1f} st. to wiecej "
                                f"niz {cfg.max_offset:g} st. - to nie zero serwa, raczej karta albo kamera")
            return fit
        if fit.rms_px - new.rms_px < max(0.3, cfg.offset_gain * fit.rms_px):
            fit.offsets_note = (f"poprawki zer serw pominiete: residuum {fit.rms_px:.2f} -> {new.rms_px:.2f} px, "
                                f"za malo, zeby ruszac zera")
            return fit
        judge(new, cfg.max_px, cfg.min_spread, cfg.min_obs)
        new.offsets_note = f"poprawki zer serw: residuum {fit.rms_px:.2f} -> {new.rms_px:.2f} px"
        return new

    def _outliers(self, fit: Fit) -> set[tuple[str, int, int]]:
        """(camera, pose, tag) of frames far off the rest, by the rule of `handeye._drop_outliers`:
        over 3 x the median and 2 x `max_px`, at most 30 % of a camera's frames, `min_obs` left."""
        out = set()
        for name, cam in fit.cameras.items():
            e = np.asarray(cam.per_obs_px, float)
            room = min(int(0.3 * e.size), e.size - max(6, self.cfg.min_obs))
            if room <= 0:
                continue
            thr = max(3.0 * float(np.median(e)), 2.0 * self.cfg.max_px)
            worst = [k for k in np.argsort(-e)[:room] if e[k] > thr]
            out |= {(name, *cam.obs_ids[k]) for k in worst}
        return out

    def run(self, on_step: Callable[[StepReport], None] | None = None) -> Fit:
        while not self.done:
            report = self.step()
            if on_step is not None:
                on_step(report)
        return self.solve()
