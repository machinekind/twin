"""Robot descriptions: one folder per robot, shared by the simulation, the panel and calibration.

A robot is a folder holding `robot.yaml` and the MJCF model it names (meshes next to the model).
The description says only what the MJCF cannot: which joints form the arm in app order, which
one is the gripper, where the tool point (TCP) is, which body carries the calibration card, the
app's joint limits and speeds (tighter than the mechanical ones), jaw thresholds, the desk the
arm works on, and how the panel labels the joints and connects the arm. Ranges, masses and
meshes stay in the MJCF and are read from it.

App units are LeRobot's: joints in degrees, gripper 0..100. `kinematics.RobotKinematics` maps
the gripper linearly onto `RobotSpec.gripper_range_q` (closed, open) in MJCF units, and a backend
reports its own mapping (`gripper_range_q()`), so the twin can warn when the two disagree.

Where robots come from (`robot_folders`):

* installed packages, through the `lerobot_mp.robots` entry point group: the entry point's name
  is the robot's name and it loads to the path of the robot's folder;
* `assets/robots/<name>/` in this repository, for robots not yet moved to their own repository.

`robot_config(spec)` builds the twin configuration for one robot from its description.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import yaml

from ..paths import REPO_ROOT
from .config import JointConfig, TwinConfig, config_from_data, from_mapping, read_config_data

logger = logging.getLogger(__name__)

ROBOT_FILE = "robot.yaml"
ENTRY_POINT_GROUP = "lerobot_mp.robots"
LOCAL_ROBOTS = Path("assets") / "robots"


def resolve_asset(path: str | Path) -> Path:
    """A repository path, relative to the working directory first and then to the repository root.

    Same order as the 3D preview, so it works both from a clone and from `pip install -e .`.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for root in (Path.cwd(), REPO_ROOT):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return REPO_ROOT / candidate


@dataclass(frozen=True)
class JawThresholds:
    """Jaw thresholds in the MJCF units of the gripper joint (rad for a hinge, m for a slide)."""

    #: The jaw "stands still": it moved less than this between ticks.
    still: float = 0.02
    #: The jaw is wider than commanded / than closed by this much = something holds it apart.
    margin: float = 0.05
    #: The most the jaw may creep closed during a success streak.
    creep: float = 0.03
    #: Wider than this = surely empty.
    open: float = 0.40


@dataclass(frozen=True)
class Desk:
    """Where the arm works on the table [m, rad], in the base frame.

    Goal rings and cube placement of the RL tasks, the calibration wave, camera sampling in
    simulation and the top-down table map.
    """

    goal_radius: tuple[float, float] = (0.13, 0.32)
    goal_height: tuple[float, float] = (0.03, 0.22)
    cube_radius: tuple[float, float] = (0.15, 0.25)
    cube_bearing: tuple[float, float] = (-0.8, 0.8)
    wave_radius: tuple[float, float] = (0.12, 0.36)
    search_centre: tuple[float, float, float] = (0.2, 0.0, 0.05)
    search_dist: tuple[float, float] = (0.4, 0.9)
    camera_dist: tuple[float, float] = (0.45, 0.75)
    camera_centre: tuple[float, float, float] = (0.20, 0.0, 0.06)
    map_centre: tuple[float, float] = (0.22, 0.0)
    map_side: float = 0.56


@dataclass(frozen=True)
class Connection:
    """One way the panel offers to connect the arm: a backend plugin and its port."""

    label: str
    backend: str
    #: None = the port typed in the panel.
    port: str | None = None
    #: `platform.system()` this connection is the default for ("Darwin", "Linux", "Windows").
    platform: str | None = None


@dataclass(frozen=True)
class RobotSpec:
    """Everything the twin needs about an arm besides the MJCF itself."""

    name: str
    title: str
    #: MJCF of the arm alone (no table, no light), relative to `root`; the scene adds the rest.
    mjcf: str
    #: Joints in app order. Names as in the MJCF.
    joints: tuple[str, ...]
    #: Which joint is the gripper (app units 0..100); None = no gripper.
    gripper: str | None
    #: Tool point site: IK solves for its pose and the panel aims it.
    tcp_site: str
    #: Body that carries the calibration card held in the jaws.
    hand_body: str
    #: Base body: the frame in which camera poses and targets are given.
    base_body: str
    #: The robot's folder; `robot.yaml` never sets it.
    root: Path = Path()
    #: Rest pose in app units.
    home: dict[str, float] = field(default_factory=dict)
    #: Approach direction (from the wrist through the jaw tips) and the jaw closing axis, both in
    #: the `tcp_site` frame. The calibration card derives its nominal pose from them: it sticks out
    #: along the approach and its normal is the closing axis. A test checks them on the model's
    #: geometry instead of trusting the MJCF author's conventions.
    tcp_approach: tuple[float, float, float] = (1.0, 0.0, 0.0)
    tcp_closing: tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: Geoms of both jaw tips. The midpoint between them with the gripper closed is where the jaws
    #: hold the calibration card.
    fingertips: tuple[str, str] = ("", "")
    #: Joint that rolls the tool about the approach direction. The calibration wave sets it
    #: analytically so the card faces the camera.
    roll_joint: str | None = None
    #: Ranges [app units] the wave samples the other joints from: tighter than the mechanical ones,
    #: so the card circles above the table and not behind the arm.
    wave_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Cameras mounted on the arm (camera names in the MJCF).
    onboard_cameras: tuple[str, ...] = ()
    #: Bodies of both jaws (fixed, moving): grip contact sensors of the RL tasks.
    jaw_bodies: tuple[str, str] = ("", "")
    #: Joint limits [app units]: tighter than the mechanical ones in the MJCF.
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Highest commanded speed [app units / s].
    max_vel: dict[str, float] = field(default_factory=dict)
    #: Gripper 0..100 maps linearly onto (closed, open) in the gripper joint's MJCF units.
    gripper_range_q: tuple[float, float] | None = None
    #: MJCF joints coupled to another joint: joint -> (app joint, ratio). Kinematics writes them
    #: on every qpos write.
    mirror_joints: dict[str, tuple[str, float]] = field(default_factory=dict)
    jaw: JawThresholds = JawThresholds()
    desk: Desk = Desk()
    #: Staged home: joint groups that drive to the home pose one after another (the rest hold),
    #: each stage collision-checked in the twin before it moves. Empty = one ramp of all joints.
    home_stages: tuple[tuple[str, ...], ...] = ()
    #: Panel slider labels: joint -> (short label, full name for the hint).
    joint_labels: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: How the panel offers to connect this arm, besides the simulation.
    connections: tuple[Connection, ...] = ()

    @property
    def mjcf_path(self) -> Path:
        return self.root / self.mjcf

    @property
    def arm_joints(self) -> tuple[str, ...]:
        return tuple(j for j in self.joints if j != self.gripper)


def load_robot(folder: str | Path) -> RobotSpec:
    """The description in `folder/robot.yaml`, checked against itself and its model file."""
    folder = Path(folder).resolve()
    path = folder / ROBOT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{folder}: brak pliku {ROBOT_FILE}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: oczekiwano mapowania YAML")
    if "root" in data:
        raise ValueError(f"{path}: `root` wynika z polozenia folderu - usun to pole")
    spec = from_mapping(RobotSpec, {**data, "root": folder}, str(path))
    _check(spec, path)
    return spec


def _check(spec: RobotSpec, path: Path) -> None:
    joints = set(spec.joints)
    problems = []
    if not spec.joints or len(joints) != len(spec.joints):
        problems.append("`joints` musi byc niepusta lista bez powtorzen")
    if spec.gripper is not None and spec.gripper not in joints:
        problems.append(f"chwytak {spec.gripper!r} nie jest na liscie `joints`")
    if spec.gripper is not None and spec.gripper_range_q is None:
        problems.append("ramie z chwytakiem potrzebuje `gripper_range_q` (zamkniety, otwarty) w jednostkach MJCF")
    for table in ("limits", "max_vel"):
        missing = joints - set(getattr(spec, table))
        if missing:
            problems.append(f"`{table}` nie ma stawow {sorted(missing)}")
    named = {
        "home": set(spec.home),
        "wave_ranges": set(spec.wave_ranges),
        "joint_labels": set(spec.joint_labels),
        "home_stages": {j for stage in spec.home_stages for j in stage},
        "mirror_joints": {src for src, _ in spec.mirror_joints.values()},
        "roll_joint": {spec.roll_joint} - {None},
    }
    for table, names in named.items():
        if names - joints:
            problems.append(f"`{table}` wymienia stawy spoza `joints`: {sorted(names - joints)}")
    if not spec.mjcf_path.is_file():
        problems.append(f"nie ma pliku modelu {spec.mjcf_path}")
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))


@lru_cache(maxsize=1)
def robot_folders() -> dict[str, Path]:
    """Name -> folder of every robot the twin can load: installed packages, then this repository."""
    found: dict[str, Path] = {}

    def add(name: str, folder: Path, source: str) -> None:
        name = name.lower()
        if name in found:
            raise ValueError(f"Dwa ramiona o nazwie {name!r}: {found[name]} i {folder} ({source})")
        found[name] = folder

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        target = ep.load()
        if not isinstance(target, (str, os.PathLike)):
            raise TypeError(f"Wtyczka ramienia {ep.name!r} ({ep.value}) musi wskazywac sciezke folderu, "
                            f"a wskazuje {type(target).__name__}")
        add(ep.name, Path(target), f"pakiet {ep.dist.name if ep.dist else ep.value}")

    local = resolve_asset(LOCAL_ROBOTS)
    if local.is_dir():
        for path in sorted(local.glob(f"*/{ROBOT_FILE}")):
            add(path.parent.name, path.parent, str(local))
    return found


@lru_cache(maxsize=None)
def get_spec(name: str) -> RobotSpec:
    folders = robot_folders()
    try:
        folder = folders[name.lower()]
    except KeyError as exc:
        known = ", ".join(sorted(folders)) or "zadne"
        raise KeyError(f"Nieznane ramie {name!r}. Dostepne: {known}") from exc
    spec = load_robot(folder)
    if spec.name.lower() != name.lower():
        raise ValueError(f"{folder / ROBOT_FILE}: `name: {spec.name}`, a ramie jest zarejestrowane jako {name!r}")
    return spec


def robots() -> dict[str, RobotSpec]:
    """Every robot the twin can load, by name."""
    return {name: get_spec(name) for name in robot_folders()}


def reload_robots() -> None:
    """Forget the discovered robots: the next lookup scans installed packages and folders again."""
    robot_folders.cache_clear()
    get_spec.cache_clear()


def joint_configs(spec: RobotSpec) -> dict[str, JointConfig]:
    """JointConfig of every joint of the description, in its order."""
    return {
        n: JointConfig(min=spec.limits[n][0], max=spec.limits[n][1], max_vel=spec.max_vel[n])
        for n in spec.joints
    }


#: Files whose foreign joint tables were already reported (once per process).
_SKIP_WARNED: set[tuple[str, str]] = set()


def robot_config(
    spec: RobotSpec, path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> TwinConfig:
    """`load_config()` for this robot: joints and the home pose from the description.

    The YAML file (`path` or `LEROBOT_MP_CONFIG`) and `overrides` lie on top, as in `load_config`.
    Two rules:

    * The file's joint tables, `joints` and `safety.home`, apply only when every joint they name
      belongs to this robot. A file written for another arm would otherwise hand this one its
      limits or home pose; such tables are skipped with a warning. `overrides` always apply,
      strictly.
    * The description's home pose applies only when neither the file nor `overrides` has
      `safety.home`. A partial `home` replaces it whole, as in `load_config` (`SafetySupervisor`
      fills the missing joints).
    """
    data = _own_tables(read_config_data(path), spec, path)
    if not (_has_home(data) or _has_home(overrides or {})):
        safety = data.get("safety")
        data = dict(data, safety=dict(safety if isinstance(safety, dict) else {}, home=dict(spec.home)))
    return config_from_data(data, overrides, joints=joint_configs(spec))


def _has_home(data: dict[str, Any]) -> bool:
    safety = data.get("safety")
    return isinstance(safety, dict) and "home" in safety


def _own_tables(data: dict[str, Any], spec: RobotSpec, path: str | Path | None) -> dict[str, Any]:
    """File content without the joint tables that name joints this robot does not have."""
    data = dict(data)
    skipped = []
    joints = data.get("joints")
    if isinstance(joints, dict) and set(joints) - set(spec.joints):
        del data["joints"]
        skipped.append("joints")
    if _has_home(data) and set(data["safety"]["home"]) - set(spec.joints):
        data["safety"] = {k: v for k, v in data["safety"].items() if k != "home"}
        skipped.append("safety.home")
    key = (str(path), spec.name)
    if skipped and key not in _SKIP_WARNED:
        _SKIP_WARNED.add(key)
        logger.warning("Konfiguracja: sekcje %s pliku sa dla innego ramienia - %s bierze je z opisu",
                       ", ".join(skipped), spec.name)
    return data
