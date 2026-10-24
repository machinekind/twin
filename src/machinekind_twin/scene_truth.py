"""The scene's own account of one object on the table: where it lies, what touches it, and two rules
that judge a run by the physics, never by what the strategy says.

    obj = ObjectSpec(label="cube", size_m=(0.03, 0.03, 0.03))
    configure_object(twin, obj)                          # before `connect`, once per twin
    twin.connect("sim", threaded=False)
    place_object(twin, obj, draw_placement(Area.from_desk(spec.desk), seed + i))
    start = sample(twin)                                  # the pose and both jaws' contacts
    ...
    verdict = judge("touch", start, clash, watch, end, settled)

The object is a free box named `OBJECT_BODY`, whatever its label. The scene gets a contact sensor
between it and each of `RobotSpec.jaw_bodies` (`<body>_jaw0`, `<body>_jaw1`), which count contacts.
Episode i draws its pose from `np.random.default_rng(seed + i)` alone, so it does not depend on how
many episodes a campaign has. A strategy sees the true pose as `ObjectPose(source="symulacja")`.

Positions are metres in the robot's base frame, whose z = 0 is the table top.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
from machinekind_twin_api.strategy import ObjectPose

from . import scene as sc
from .kinematics import inverse
from .robots import Desk, RobotSpec

__all__ = [
    "CONTACT_MIN", "LIFT_RISE_MIN_M", "OBJECT_BODY", "RULES", "SETTLE_S", "SHAPES", "SIM_SOURCE",
    "TOUCH_PUSH_MAX_M", "TOUCH_TILT_MAX_DEG", "Area", "ObjectSpec", "Placement", "SceneSample",
    "SimulationPoses", "Verdict", "Watch", "check_jaws", "check_on_table", "configure_object",
    "draw_placement", "judge", "place_object", "robot_contacts", "rule_constants", "sample",
]

#: The object's body and geom in the scene: no robot part, `floor` or `table` takes this name.
OBJECT_BODY = "eval_object"
#: `ObjectPose.source` of the true pose, the label strategies already accept for the simulation.
SIM_SOURCE = "symulacja"
SHAPES = ("box",)
RULES = ("touch", "lift")

#: A jaw's contact sensor above this counts as touching (the sensor counts contacts).
CONTACT_MIN = 0.5
#: How long the scene runs on after the run before its second look [s].
SETTLE_S = 0.5
#: `touch`: the object may move this far on the table and tilt this much, both exclusive.
TOUCH_PUSH_MAX_M = 0.01
TOUCH_TILT_MAX_DEG = 20.0
#: `lift`: how far the object's centre must rise, at the end and after settling.
LIFT_RISE_MIN_M = 0.04


# --------------------------------------------------------------------------- the object and its area


@dataclass(frozen=True)
class ObjectSpec:
    """The object every episode places: its label in `Observation.objects`, shape, edges and mass."""

    label: str = "cube"
    shape: str = "box"
    #: Full edge lengths [m].
    size_m: tuple[float, float, float] = (0.03, 0.03, 0.03)
    mass_kg: float = 0.03

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("obiekt potrzebuje etykiety")
        if self.shape not in SHAPES:
            raise ValueError(f"ksztalt {self.shape!r}: znam tylko {', '.join(SHAPES)}")
        if len(self.size_m) != 3 or not all(_positive(value) for value in self.size_m):
            raise ValueError(f"wymiary obiektu musza byc trzema dodatnimi liczbami, sa {self.size_m}")
        if not _positive(self.mass_kg):
            raise ValueError(f"masa obiektu musi byc dodatnia, jest {self.mass_kg}")

    @property
    def half_m(self) -> tuple[float, float, float]:
        x, y, z = self.size_m
        return x / 2.0, y / 2.0, z / 2.0

    def box(self, xy: tuple[float, float]) -> sc.Box:
        """The scene's box resting at `xy` in the base frame, in `Box`'s default red."""
        return sc.Box(OBJECT_BODY, self.half_m, (float(xy[0]), float(xy[1])), mass=float(self.mass_kg))

    def to_json(self) -> dict[str, Any]:
        return {"label": self.label, "shape": self.shape, "size_m": [float(value) for value in self.size_m],
                "mass_kg": float(self.mass_kg)}


@dataclass(frozen=True)
class Area:
    """Where episodes place the object: a ring sector around the base and a yaw range [m, rad]."""

    radius_m: tuple[float, float]
    bearing_rad: tuple[float, float]
    yaw_rad: tuple[float, float] = (-math.pi, math.pi)

    def __post_init__(self) -> None:
        for name, (low, high), degrees in (("promien", self.radius_m, False),
                                           ("kierunek", self.bearing_rad, True),
                                           ("obrot", self.yaw_rad, True)):
            shown = (f"{math.degrees(low):g}..{math.degrees(high):g} st" if degrees
                     else f"{low:g}..{high:g} m")
            if not (math.isfinite(low) and math.isfinite(high)):
                raise ValueError(f"obszar obiektu: {name} {shown} - potrzeba liczb skonczonych")
            if low > high:
                raise ValueError(f"obszar obiektu: {name} {shown} - poczatek wiekszy od konca")
        if self.radius_m[0] < 0:
            raise ValueError(f"obszar obiektu: promien {self.radius_m[0]:g} m - nie moze byc ujemny")

    @staticmethod
    def from_desk(desk: Desk) -> Area:
        """The robot's own cube ring, as its RL tasks place cubes, with any yaw."""
        return Area(desk.cube_radius, desk.cube_bearing, (-math.pi, math.pi))

    def to_json(self) -> dict[str, Any]:
        return {"radius_m": [float(value) for value in self.radius_m],
                "bearing_deg": [math.degrees(value) for value in self.bearing_rad],
                "yaw_deg": [math.degrees(value) for value in self.yaw_rad]}


def check_on_table(area: Area, obj: ObjectSpec, scene: sc.Scene) -> None:
    """ValueError when some pose of `area` puts the object's footprint off the table top.

    The table top is centred at the world origin (`scene.compose`). The footprint counts as a circle
    around the object's corners, so any yaw is covered.
    """
    sx, sy = scene.cfg.table.size
    T = scene.T_base2world
    base_yaw = math.atan2(T[1, 0], T[0, 0])
    half_x, half_y, _ = obj.half_m
    margin = math.hypot(half_x, half_y)

    for radius in area.radius_m:
        for bearing in _extreme_bearings(area.bearing_rad, base_yaw):
            x, y, _ = T[:3, :3] @ np.array([radius * math.cos(bearing), radius * math.sin(bearing), 0.0]) + T[:3, 3]
            if abs(x) > sx / 2.0 - margin or abs(y) > sy / 2.0 - margin:
                raise ValueError(f"obszar obiektu wychodzi poza blat {sx:.2f} x {sy:.2f} m: r {radius:.3f} m, "
                                 f"kierunek {math.degrees(bearing):.0f} st")


def _extreme_bearings(bearing_rad: tuple[float, float], base_yaw: float) -> list[float]:
    """The bounds plus every bearing inside them where a world x or y peaks: `k * pi / 2 - base_yaw`."""
    low, high = bearing_rad
    quarter = math.pi / 2.0
    first = math.ceil((low + base_yaw) / quarter)
    last = min(math.floor((high + base_yaw) / quarter), first + 3)

    return [low, high] + [k * quarter - base_yaw for k in range(first, last + 1)]


def check_jaws(spec: RobotSpec) -> None:
    """ValueError for a robot without two `jaw_bodies`: both rules read the jaws' contact sensors."""
    if len(spec.jaw_bodies) != 2 or not all(spec.jaw_bodies):
        raise ValueError(f"ramie {spec.name} nie podaje jaw_bodies - reguly touch i lift czytaja czujniki szczek")


# --------------------------------------------------------------------------- placement


@dataclass(frozen=True)
class Placement:
    """One episode's pose of the object in the base frame [m, rad], and the seed it came from."""

    seed: int
    x_m: float
    y_m: float
    radius_m: float
    bearing_rad: float
    yaw_rad: float

    def to_json(self) -> dict[str, Any]:
        """Rounded to 6 decimals, as the summary writes it, so the log's copy compares equal."""
        values = {"x_m": self.x_m, "y_m": self.y_m, "radius_m": self.radius_m,
                  "bearing_deg": math.degrees(self.bearing_rad), "yaw_deg": math.degrees(self.yaw_rad)}
        return {key: round(value, 6) for key, value in values.items()}


def draw_placement(area: Area, seed: int) -> Placement:
    """The pose `seed` gives, in the order and by the formula of soarm_lab's `sample_cubes`."""
    rng = np.random.default_rng(seed)
    radius = float(rng.uniform(*area.radius_m))
    bearing = float(rng.uniform(*area.bearing_rad))
    yaw = float(rng.uniform(*area.yaw_rad))

    return Placement(seed, radius * math.cos(bearing), radius * math.sin(bearing), radius, bearing, yaw)


def configure_object(twin: Any, obj: ObjectSpec, xy: tuple[float, float] = (0.0, 0.0)) -> None:
    """Adds the object and its jaw sensors to the twin's scene. Once per twin, before `connect`.

    Moving it later goes through `place_object`: a rebuild keeps the pose of a body of the same name.
    """
    twin.configure(objects=[obj.box(xy)], grasp_sensors=[OBJECT_BODY])


def place_object(twin: Any, obj: ObjectSpec, placement: Placement) -> None:
    """The object at rest on the table at the placement, turned by its yaw about z."""
    with twin.lock:
        scene = twin.scene
        model, data = scene.model, scene.data
        body = model.body(OBJECT_BODY).id
        joint = model.body_jntadr[body]
        position, velocity = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        T = scene.T_base2world

        half_yaw = placement.yaw_rad / 2.0
        turn = np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)])
        base, world = np.zeros(4), np.zeros(4)
        # MuJoCo's functions come from a C extension without type stubs.
        mujoco.mju_mat2Quat(base, T[:3, :3].ravel())  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mju_mulQuat(world, base, turn)  # pyright: ignore[reportAttributeAccessIssue]

        data.qpos[position:position + 3] = T[:3, :3] @ np.array([placement.x_m, placement.y_m, obj.half_m[2]]) + T[:3, 3]
        data.qpos[position + 3:position + 7] = world
        data.qvel[velocity:velocity + 6] = 0.0
        mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]


# --------------------------------------------------------------------------- what the physics says


@dataclass(frozen=True)
class SceneSample:
    """The object's true pose in the base frame and the contact count of each jaw."""

    position_m: np.ndarray
    rotation: np.ndarray
    contacts: tuple[float, float]


def sample(twin: Any) -> SceneSample:
    """The object now, from the physics."""
    with twin.lock:
        scene = twin.scene
        model, data = scene.model, scene.data
        body = model.body(OBJECT_BODY).id
        to_base = inverse(scene.T_base2world)
        position = to_base[:3, :3] @ data.xpos[body] + to_base[:3, 3]
        rotation = to_base[:3, :3] @ data.xmat[body].reshape(3, 3)
        contacts = tuple(float(data.sensordata[model.sensor(f"{OBJECT_BODY}_jaw{k}").adr[0]]) for k in range(2))

    return SceneSample(np.array(position), np.array(rotation), (contacts[0], contacts[1]))


def robot_contacts(twin: Any) -> int:
    """How many of the scene's current contacts pair the object with any part of the robot."""
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        own = model.geom(OBJECT_BODY).id
        count = 0
        for contact in data.contact[:data.ncon]:
            geoms = (int(contact.geom1), int(contact.geom2))
            if own not in geoms:
                continue
            other = geoms[1] if geoms[0] == own else geoms[0]
            if model.body(int(model.geom_bodyid[other])).name.startswith(sc.PREFIX):
                count += 1

    return count


class SimulationPoses:
    """The object's true pose for `StrategyRunner(objects=...)`, labelled `symulacja` and never old.

    It only reads arrays, so it cannot raise: an exception in `objects` at start would end the
    run as if the strategy had refused it.
    """

    def __init__(self, twin: Any, label: str):
        self._twin = twin
        self._label = label

    def objects(self) -> dict[str, ObjectPose]:
        now = sample(self._twin)
        return {self._label: ObjectPose(now.position_m, now.rotation, SIM_SOURCE, 0.0)}

    def begin(self) -> None:
        pass

    def advance(self, now: float) -> None:
        pass

    def end(self) -> None:
        pass


@dataclass
class Watch:
    """What happened to the object while the run went on, sampled after every physics step."""

    start: SceneSample
    most_contacts: list[float] = field(default_factory=lambda: [0.0, 0.0])
    #: The run time of the first jaw contact; None when none came.
    first_contact_s: float | None = None
    max_rise_m: float = 0.0

    def observe(self, now: SceneSample, run_t: float) -> None:
        self.most_contacts = [max(most, value) for most, value in zip(self.most_contacts, now.contacts)]
        if self.first_contact_s is None and max(now.contacts) > CONTACT_MIN:
            self.first_contact_s = run_t
        self.max_rise_m = max(self.max_rise_m, _rise(self.start, now))


# --------------------------------------------------------------------------- the rules


@dataclass(frozen=True)
class Verdict:
    """The scene's judgement of one run, and every measure it took, whatever the rule.

    The measures that need the look after settling are None without it.
    """

    ok: bool
    #: Why not, in Polish; "" when the rule holds.
    reason: str
    contacts_end: tuple[float, float]
    contacts_settled: tuple[float, float] | None
    most_contacts: tuple[float, float]
    first_contact_s: float | None
    push_m: float | None
    tilt_deg: float | None
    rise_end_m: float
    rise_settled_m: float | None
    max_rise_m: float
    position_end_m: tuple[float, float, float]
    position_settled_m: tuple[float, float, float] | None

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "contacts_end": list(self.contacts_end),
            "contacts_settled": None if self.contacts_settled is None else list(self.contacts_settled),
            "most_contacts": list(self.most_contacts),
            "first_contact_s": self.first_contact_s,
            "push_m": self.push_m,
            "tilt_deg": self.tilt_deg,
            "rise_end_m": self.rise_end_m,
            "rise_settled_m": self.rise_settled_m,
            "max_rise_m": self.max_rise_m,
            "position_end_m": list(self.position_end_m),
            "position_settled_m": None if self.position_settled_m is None else list(self.position_settled_m),
        }


def judge(rule: str, start: SceneSample, clash: bool, watch: Watch, end: SceneSample,
          settled: SceneSample | None) -> Verdict:
    """The scene's verdict under `rule` ("touch" or "lift"). The first failing check names the reason.

    `start` is the object as placed, `end` right after the run ended and `settled` `SETTLE_S` later
    (None when the scene could not run on). `clash`: the object touched the robot as placed.
    """
    if rule not in RULES:
        raise ValueError(f"regula {rule!r}: znam {', '.join(RULES)}")

    push = None if settled is None else float(np.linalg.norm((settled.position_m - start.position_m)[:2]))
    tilt = None if settled is None else _tilt_deg(start, settled)
    rise_end = _rise(start, end)
    rise_settled = None if settled is None else _rise(start, settled)

    checks = [(clash, "obiekt dotyka ramienia na starcie"), (settled is None, "brak pomiaru po uspokojeniu")]
    if rule == "touch":
        checks += _touch_checks(push, tilt, end, settled)
    else:
        checks += _lift_checks(rise_end, rise_settled, end, settled)
    reason = next((text for failed, text in checks if failed), "")

    return Verdict(
        ok=not reason,
        reason=reason,
        contacts_end=end.contacts,
        contacts_settled=None if settled is None else settled.contacts,
        most_contacts=(watch.most_contacts[0], watch.most_contacts[1]),
        first_contact_s=watch.first_contact_s,
        push_m=push,
        tilt_deg=tilt,
        rise_end_m=rise_end,
        rise_settled_m=rise_settled,
        max_rise_m=watch.max_rise_m,
        position_end_m=_triple(end.position_m),
        position_settled_m=None if settled is None else _triple(settled.position_m),
    )


def _touch_checks(push: float | None, tilt: float | None, end: SceneSample,
                  settled: SceneSample | None) -> list[tuple[bool, str]]:
    if settled is None or push is None or tilt is None:
        return []

    return [
        (max(end.contacts) <= CONTACT_MIN, "brak kontaktu na koncu"),
        (max(settled.contacts) <= CONTACT_MIN, f"brak kontaktu po {SETTLE_S:g} s"),
        (push >= TOUCH_PUSH_MAX_M, f"obiekt przesuniety o {push * 100:.1f} cm"),
        (tilt >= TOUCH_TILT_MAX_DEG, f"obiekt przechylony o {tilt:.0f} st"),
    ]


def _lift_checks(rise_end: float, rise_settled: float | None, end: SceneSample,
                 settled: SceneSample | None) -> list[tuple[bool, str]]:
    if settled is None or rise_settled is None:
        return []

    need = f"potrzeba {LIFT_RISE_MIN_M * 100:g} cm"
    return [
        (min(end.contacts) <= CONTACT_MIN, "nie obie szczeki na koncu"),
        (min(settled.contacts) <= CONTACT_MIN, f"nie obie szczeki po {SETTLE_S:g} s"),
        (rise_end < LIFT_RISE_MIN_M, f"uniesiony o {rise_end * 100:.1f} cm na koncu ({need})"),
        (rise_settled < LIFT_RISE_MIN_M, f"uniesiony o {rise_settled * 100:.1f} cm po {SETTLE_S:g} s ({need})"),
    ]


def rule_constants() -> Mapping[str, float]:
    """The constants both rules use, for a report: equal rule names mean equal judges."""
    return {"contact_min": CONTACT_MIN, "settle_s": SETTLE_S, "touch_push_max_m": TOUCH_PUSH_MAX_M,
            "touch_tilt_max_deg": TOUCH_TILT_MAX_DEG, "lift_rise_min_m": LIFT_RISE_MIN_M}


def _rise(start: SceneSample, now: SceneSample) -> float:
    return float(now.position_m[2] - start.position_m[2])


def _tilt_deg(start: SceneSample, now: SceneSample) -> float:
    cosine = float(np.clip(start.rotation[:, 2] @ now.rotation[:, 2], -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _triple(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
