"""Fixture descriptions: static parts of a stand that come with a robot package, e.g. a camera mount.

A robot's folder may hold fixtures, one folder each: `fixtures/<name>/fixture.yaml` and the meshes it
names. A stand refers to one as `<robot>/<name>`. The description gives:

* `parts`: each with an optional mesh in metres, its colliders (boxes, centre and half sizes) and the
  cameras that sit on it (`cameras`, T_cam2part in OpenCV axes: x right, y down, z forward). All in
  the part's frame;
* `stack`: the parts from the bottom up. Each part's frame sits at the `top` of the one below it,
  so a stand can list another stack (`FixtureRecord.parts`) of the same parts;
* `placements`: named poses of the fixture's frame in the base frame of the robot whose folder holds
  it, e.g. left and right of the arm.

Collision checks see only the colliders: a mesh cannot grow by a clearance (`collision`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import yaml

from .config import from_mapping
from .kinematics import pose
from .robots import Matrix4, _is_rigid, robot_folders
from .scene import Collider, Fixture, FixturePart

FIXTURES_DIR = "fixtures"
FIXTURE_FILE = "fixture.yaml"


@dataclass(frozen=True)
class PartSpec:
    """One part of a fixture, in its own frame."""

    #: Mesh in metres, relative to the fixture's folder; None = the colliders are drawn instead.
    mesh: str | None = None
    rgba: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)
    #: Where the next part of the stack sits [m].
    top: tuple[float, float, float] = (0.0, 0.0, 0.0)
    colliders: dict[str, Collider] = field(default_factory=dict)
    #: Camera sites: T_cam2part in OpenCV axes.
    cameras: dict[str, Matrix4] = field(default_factory=dict)


@dataclass(frozen=True)
class Placement:
    """Where the fixture's frame stands in its robot's base frame [m] and its turn about the vertical [deg]."""

    xyz: tuple[float, float, float]
    yaw_deg: float = 0.0

    def T(self) -> np.ndarray:
        return yaw_pose(self.xyz, self.yaw_deg)


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    title: str
    parts: dict[str, PartSpec]
    #: The parts from the bottom up, as the fixture is usually built.
    stack: tuple[str, ...]
    placements: dict[str, Placement] = field(default_factory=dict)
    #: The fixture's folder and the robot whose folder holds it; neither is set in the file.
    root: Path = Path()
    robot: str = ""

    @property
    def ref(self) -> str:
        return f"{self.robot}/{self.name}"


def yaw_pose(xyz, yaw_deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return pose(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.asarray(xyz, float))


def load_fixture(folder: str | Path, robot: str) -> FixtureSpec:
    """The description in `folder/fixture.yaml`, checked against itself and its meshes."""
    folder = Path(folder).resolve()
    path = folder / FIXTURE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{folder}: brak pliku {FIXTURE_FILE}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: oczekiwano mapowania YAML")
    if "root" in data or "robot" in data:
        raise ValueError(f"{path}: `root` i `robot` wynikaja z polozenia folderu - usun te pola")
    spec = from_mapping(FixtureSpec, {**data, "root": folder, "robot": robot}, str(path))
    problems = []
    if spec.name != folder.name:
        problems.append(f"`name: {spec.name}`, a folder nazywa sie {folder.name}")
    if not spec.stack:
        problems.append("`stack` musi wymieniac co najmniej jedna czesc")
    problems += [f"`stack`: nie ma czesci {p!r}" for p in spec.stack if p not in spec.parts]
    for name, part in spec.parts.items():
        if part.mesh is not None and not (folder / part.mesh).is_file():
            problems.append(f"`parts.{name}.mesh`: nie ma pliku {part.mesh}")
        if part.mesh is None and not part.colliders:
            problems.append(f"`parts.{name}`: bez siatki i bez bryl kolizyjnych nie ma czego pokazac")
        problems += [f"`parts.{name}.colliders.{c}`: polowy bokow musza byc dodatnie"
                     for c, box in part.colliders.items() if min(box.size) <= 0]
        problems += [f"`parts.{name}.cameras.{c}`: poza musi byc przeksztalceniem sztywnym"
                     for c, T in part.cameras.items() if not _is_rigid(np.asarray(T, float))]
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
    return spec


@cache
def _load(folder: Path, robot: str) -> FixtureSpec:
    return load_fixture(folder, robot)


def fixture_folders() -> dict[str, Path]:
    """`<robot>/<name>` -> folder of every fixture the installed robot packages bring."""
    found = {}
    for robot, folder in robot_folders().items():
        for path in sorted((folder / FIXTURES_DIR).glob(f"*/{FIXTURE_FILE}")):
            found[f"{robot}/{path.parent.name}"] = path.parent
    return found


def get_fixture(ref: str) -> FixtureSpec:
    """The fixture `<robot>/<name>` from an installed robot package."""
    robot, _, name = ref.partition("/")
    folder = robot_folders().get(robot.lower())
    if not name or folder is None or not (folder / FIXTURES_DIR / name / FIXTURE_FILE).is_file():
        known = ", ".join(sorted(fixture_folders())) or "zadnego"
        raise KeyError(f"nieznane mocowanie {ref!r} (zainstalowane pakiety robotow maja: {known})")
    return _load((folder / FIXTURES_DIR / name).resolve(), robot.lower())


def stack_poses(spec: FixtureSpec, parts: Sequence[str] | None = None) -> list[tuple[str, np.ndarray]]:
    """(part, T_part2fixture) of the stack `parts` (default: the description's), from the bottom up."""
    parts = list(spec.stack if parts is None else parts)
    if not parts:
        raise ValueError(f"mocowanie {spec.ref}: pusta lista czesci")
    unknown = [p for p in parts if p not in spec.parts]
    if unknown:
        raise ValueError(f"mocowanie {spec.ref}: nie ma czesci {unknown} (sa: {', '.join(spec.parts)})")
    out, T = [], np.eye(4)
    for p in parts:
        out.append((p, T.copy()))
        T = T @ pose(np.eye(3), np.asarray(spec.parts[p].top, float))
    return out


def camera_site(spec: FixtureSpec, parts: Sequence[str] | None = None, site: str | None = None) -> np.ndarray:
    """T_cam2fixture of a camera site on the stack: `site`, or the only one when None. When a part with a
    camera site repeats, the top-most copy counts."""
    found = {}
    for part, T in stack_poses(spec, parts):
        for name, T_cam in spec.parts[part].cameras.items():
            found[name] = T @ np.asarray(T_cam, float)
    if site is None and len(found) == 1:
        return next(iter(found.values()))
    if site is None or site not in found:
        what = "podaj miejsce kamery" if site is None else f"brak miejsca kamery {site!r}"
        raise KeyError(f"mocowanie {spec.ref}: {what} (sa: {', '.join(sorted(found)) or 'zadne'})")
    return found[site]


def placed(spec: FixtureSpec, name: str, T_base: np.ndarray, parts: Sequence[str] | None = None,
           rgba: Sequence[float] | None = None) -> Fixture:
    """The fixture as part of the scene, its frame at `T_base` in the driven arm's base frame.

    A part listed more than once gets a number from the second on (`middle`, `middle2`).
    """
    if rgba is not None and len(rgba) != 4:
        raise ValueError(f"mocowanie {name}: rgba to 4 liczby, jest {list(rgba)}")
    colour = None if rgba is None else (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
    out, seen = [], {}
    for part, T in stack_poses(spec, parts):
        seen[part] = seen.get(part, 0) + 1
        p = spec.parts[part]
        out.append(FixturePart(part if seen[part] == 1 else f"{part}{seen[part]}", T,
                               None if p.mesh is None else spec.root / p.mesh, dict(p.colliders), colour or p.rgba))
    return Fixture(name, out, np.asarray(T_base, float))
