"""Stands: one folder that describes a physical stand and everything measured on it.

    stand.json                        the setup: robots, cameras with their controls, fixtures, table, card,
                                      look, the calibration wave's limits
    calib/poses/<time>.json           camera poses with the servo zero offsets they were fitted with
                                      (the workspace's pose presets, same format)
    calib/lens/<camera>/<time>.json   one camera's lens: K and distortion at one resolution
    calib/dynamics/<time>.json        identified servo dynamics
    station/                          `twin export` of the stand, for training

Calibration entries are never overwritten: a different result is a new file, and `calib` in
`stand.json` names the current entry of each kind. Stand folders live in their own repository;
the twin knows only the format (`validate`) and the commands (`twin stand ...`).

A workspace loaded from a stand stays bound to it (`Workspace.stand`): its pose presets are the
stand's, and every save writes the stand back (`sync`). docs/TWIN.md, "Stands", has the format.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, NoReturn

from .calib.card import Card
from .calib.session import WaveLimits
from .camera_controls import CameraControls
from .config import from_mapping
from .dynamics import Dynamics
from .scene import PREFIX, Look, Table
from .workspace import (
    ON_TABLE,
    WOBBLE_PX,
    CameraRecord,
    FixtureRecord,
    RobotRecord,
    Workspace,
    default_path,
    stamp_name,
)

logger = logging.getLogger(__name__)

#: The `stand.json` format this twin reads and writes.
FORMAT = 1
STAND_FILE = "stand.json"
POSES = Path("calib") / "poses"
LENS = Path("calib") / "lens"
DYNAMICS = Path("calib") / "dynamics"
STATION = "station"
#: Key order of a written `stand.json`.
_ORDER = ("format", "name", "description", "obsolete", "robots", "cameras", "fixtures", "table", "card", "look",
          "wave", "calib")
_CONTROLS = tuple(f.name for f in fields(CameraControls))
#: One writer at a time: panel callbacks save from several threads.
_LOCK = threading.Lock()


class StandError(ValueError):
    """A stand folder the twin cannot use; the message names the file, the key and the reason."""


# ------------------------------------------------------------------ validation
def _fail(where: str, what: str) -> NoReturn:
    raise StandError(f"{where}: {what}")


def _kind(value: Any) -> str:
    return {bool: "true/false", int: "liczba", float: "liczba", str: "tekst", list: "lista", dict: "obiekt",
            type(None): "null"}.get(type(value), type(value).__name__)


def _keys(obj: Any, where: str, required: tuple[str, ...], optional: tuple[str, ...]) -> None:
    if not isinstance(obj, dict):
        _fail(where, f"oczekiwano obiektu, jest {_kind(obj)}")
    missing = [k for k in required if k not in obj]
    if missing:
        _fail(where, f"brak pola {', '.join(missing)}")
    unknown = sorted(set(obj) - set(required) - set(optional))
    if unknown:
        _fail(where, f"nieznane pola {unknown}; znane: {', '.join([*required, *optional])}")


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _text(v: Any, where: str, *, empty: bool = False, null: bool = False) -> None:
    if v is None and null:
        return
    if not isinstance(v, str) or (not empty and not v.strip()):
        _fail(where, f"oczekiwano {'tekstu' if empty else 'niepustego tekstu'}, jest {v!r}")


def _number(v: Any, where: str) -> None:
    if not _is_num(v):
        _fail(where, f"oczekiwano liczby, jest {v!r}")


def _positive_int(v: Any, where: str) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        _fail(where, f"oczekiwano liczby calkowitej >= 1, jest {v!r}")


def _flag(v: Any, where: str) -> None:
    if not isinstance(v, bool):
        _fail(where, f"oczekiwano true albo false, jest {v!r}")


def _vector(v: Any, where: str, n: int) -> None:
    if not isinstance(v, list) or len(v) != n or not all(_is_num(x) for x in v):
        _fail(where, f"oczekiwano listy {n} liczb, jest {v!r}")


def _matrix(v: Any, where: str, n: int) -> None:
    if not isinstance(v, list) or len(v) != n or not all(
            isinstance(r, list) and len(r) == n and all(_is_num(x) for x in r) for r in v):
        _fail(where, f"oczekiwano macierzy {n}x{n} liczb")


def _numbers(v: Any, where: str) -> None:
    if not isinstance(v, dict) or not all(isinstance(k, str) and _is_num(x) for k, x in v.items()):
        _fail(where, f"oczekiwano obiektu {{nazwa: liczba}}, jest {v!r}")


def _file_name(v: Any, where: str) -> None:
    """A name that becomes a file or folder name in the stand: no separators, no leading dot."""
    if not isinstance(v, str) or not v or v.startswith(".") or re.search(r"[/\\\x00]", v):
        _fail(where, f"nazwa {v!r} nie moze byc pusta, zaczynac sie kropka ani zawierac / i \\")


def _date(v: Any, where: str) -> None:
    try:
        ok = isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) and time.strptime(v, "%Y-%m-%d")
    except ValueError:
        ok = False
    if not ok:
        _fail(where, f"oczekiwano daty RRRR-MM-DD, jest {v!r}")


def _dataclass(cls: Any, value: Any, where: str) -> Any:
    try:
        return from_mapping(cls, value, where)
    except (TypeError, ValueError) as exc:
        raise StandError(f"{where}: {exc}") from None


def obsolete_info(data: dict[str, Any], where: str = STAND_FILE) -> dict[str, str] | None:
    """`obsolete` as {"reason", "date"} ("" where not given); None for a current stand."""
    value = data.get("obsolete", False)
    if value is False or value is None:
        return None
    if value is True:
        return {"reason": "", "date": ""}
    w = f"{where}: obsolete"
    _keys(value, w, (), ("reason", "date"))
    _text(value.get("reason", ""), f"{w}.reason", empty=True)
    if "date" in value:
        _date(value["date"], f"{w}.date")
    return {"reason": value.get("reason", ""), "date": value.get("date", "")}


def _robots(robots: Any, where: str) -> None:
    if not isinstance(robots, list) or not robots:
        _fail(where, "oczekiwano listy z co najmniej jednym robotem: ramieniem sterowanym")
    driven, names = 0, []
    for i, r in enumerate(robots):
        w = f"{where}[{i}]"
        if not isinstance(r, dict):
            _fail(w, f"oczekiwano obiektu, jest {_kind(r)}")
        _flag(r.get("read_only", False), f"{w}.read_only")
        if r.get("read_only", False):
            _keys(r, w, ("name", "robot"), ("read_only", "xyz", "yaw_deg", "backend", "port", "joints",
                                            "base_drive"))
            name = r["name"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name + "/" == PREFIX:
                _fail(f"{w}.name", f"tylko litery, cyfry, _ i -, inna niz {PREFIX[:-1]!r}; jest {name!r}")
            names.append(name)
            if "xyz" in r:
                _vector(r["xyz"], f"{w}.xyz", 3)
            if "yaw_deg" in r:
                _number(r["yaw_deg"], f"{w}.yaw_deg")
            if "joints" in r:
                _numbers(r["joints"], f"{w}.joints")
            _flag(r.get("base_drive", False), f"{w}.base_drive")
        else:
            driven += 1
            extra = sorted(set(r) & {"name", "xyz", "yaw_deg", "joints"})
            if extra:
                _fail(w, f"ramie sterowane (read_only: false) jest poczatkiem ukladu stanowiska i w scenie "
                         f"nazywa sie {PREFIX[:-1]!r} - bez pol {extra}")
            _keys(r, w, ("robot",), ("read_only", "backend", "port", "home"))
            if r.get("home") is not None:
                _numbers(r["home"], f"{w}.home")
        _text(r["robot"], f"{w}.robot")
        _text(r.get("backend"), f"{w}.backend", null=True)
        _text(r.get("port"), f"{w}.port", null=True)
    if driven != 1:
        _fail(where, f"potrzebne dokladnie jedno ramie sterowane (read_only: false), jest {driven}")
    if len(set(names)) != len(names):
        _fail(where, f"nazwy robotow powtarzaja sie: {names}")


def _cameras(cams: Any, where: str) -> list[str]:
    if not isinstance(cams, list):
        _fail(where, f"oczekiwano listy, jest {_kind(cams)}")
    names: list[str] = []
    for i, c in enumerate(cams):
        w = f"{where}[{i}]"
        _keys(c, w, ("name", "source", "width", "height"),
              ("fps", "enabled", "controls", "sim_pose", "wobbly", "wobble_px", "fixture"))
        _file_name(c["name"], f"{w}.name")
        if c["name"] in names:
            _fail(f"{w}.name", f"kamera {c['name']!r} juz jest w stanowisku")
        names.append(c["name"])
        w = f"{where}[{i}] ({c['name']})"
        _text(c["source"], f"{w}.source")
        for key in ("width", "height", "fps"):
            if key in c:
                _positive_int(c[key], f"{w}.{key}")
        if "enabled" in c:
            _flag(c["enabled"], f"{w}.enabled")
        if "wobbly" in c:
            _flag(c["wobbly"], f"{w}.wobbly")
        if "wobble_px" in c and not (_is_num(c["wobble_px"]) and c["wobble_px"] > 0):
            _fail(f"{w}.wobble_px", f"oczekiwano liczby > 0, jest {c['wobble_px']!r}")
        if "controls" in c:
            _keys(c["controls"], f"{w}.controls", (), _CONTROLS)
            try:
                CameraControls(**c["controls"])
            except ValueError as exc:
                _fail(f"{w}.controls", str(exc))
        if c.get("sim_pose") is not None:
            if c["source"] != "sim":
                _fail(f"{w}.sim_pose", "tylko dla kamery symulowanej (source: sim)")
            _matrix(c["sim_pose"], f"{w}.sim_pose", 4)
        _text(c.get("fixture"), f"{w}.fixture", null=True)
    return names


_FIXTURE_KEYS = ("on", "placement", "parts", "xyz", "yaw_deg", "rgba")


def _fixtures(fixtures: Any, where: str, robots: list[str]) -> list[str]:
    """Checks `fixtures` against the stand's robot names; returns the fixture names."""
    if not isinstance(fixtures, list):
        _fail(where, f"oczekiwano listy, jest {_kind(fixtures)}")
    names: list[str] = []
    for i, f in enumerate(fixtures):
        w = f"{where}[{i}]"
        _keys(f, w, ("name", "fixture"), _FIXTURE_KEYS)
        name = f["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name + "/" == PREFIX:
            _fail(f"{w}.name", f"tylko litery, cyfry, _ i -, inna niz {PREFIX[:-1]!r}; jest {name!r}")
        if name in names or name in robots:
            _fail(f"{w}.name", f"nazwa {name!r} jest juz zajeta przez robota albo inne mocowanie")
        names.append(name)
        w = f"{where}[{i}] ({name})"
        _text(f["fixture"], f"{w}.fixture")
        if not re.fullmatch(r"[^/]+/[^/]+", f["fixture"]):
            _fail(f"{w}.fixture", f"oczekiwano <robot>/<mocowanie>, jest {f['fixture']!r}")
        on = f.get("on", PREFIX[:-1])
        if on not in (PREFIX[:-1], ON_TABLE, *robots):
            _fail(f"{w}.on", f"{PREFIX[:-1]!r}, {ON_TABLE!r} albo nazwa robota tylko do odczytu, jest {on!r}")
        if on == ON_TABLE and f.get("xyz") is None:
            _fail(f"{w}.xyz", "mocowanie na stole potrzebuje xyz (od srodka blatu)")
        _text(f.get("placement"), f"{w}.placement", null=True)
        parts = f.get("parts")
        if parts is not None and (not isinstance(parts, list) or not parts
                                  or not all(isinstance(p, str) and p for p in parts)):
            _fail(f"{w}.parts", f"oczekiwano niepustej listy nazw czesci, jest {parts!r}")
        if f.get("xyz") is not None:
            _vector(f["xyz"], f"{w}.xyz", 3)
        if "yaw_deg" in f:
            _number(f["yaw_deg"], f"{w}.yaw_deg")
        if f.get("rgba") is not None:
            _vector(f["rgba"], f"{w}.rgba", 4)
    return names


def _calib(calib: Any, where: str, cameras: list[str]) -> None:
    _keys(calib, where, (), ("poses", "lens", "dynamics"))
    for key in ("poses", "dynamics"):
        if calib.get(key) is not None:
            _file_name(calib[key], f"{where}.{key}")
    lens = calib.get("lens", {})
    if not isinstance(lens, dict):
        _fail(f"{where}.lens", f"oczekiwano obiektu {{kamera: wpis}}, jest {_kind(lens)}")
    for cam, entry in lens.items():
        if cam not in cameras:
            _fail(f"{where}.lens.{cam}", f"nie ma takiej kamery w cameras ({', '.join(cameras) or 'brak kamer'})")
        _file_name(entry, f"{where}.lens.{cam}")


def validate(data: Any, where: str = STAND_FILE) -> dict[str, Any]:
    """Checks a parsed `stand.json`; raises `StandError` naming the key and the problem."""
    _keys(data, where, ("format", "name", "robots"),
          ("description", "obsolete", "cameras", "fixtures", "table", "card", "look", "wave", "calib"))
    fmt = data["format"]
    if not isinstance(fmt, int) or isinstance(fmt, bool) or fmt < 1:
        _fail(f"{where}: format", f"oczekiwano liczby calkowitej >= 1, jest {fmt!r}")
    if fmt > FORMAT:
        _fail(f"{where}: format", f"format {fmt} jest nowszy niz ten blizniak (zna {FORMAT}) - zaktualizuj blizniaka")
    _text(data["name"], f"{where}: name")
    _text(data.get("description", ""), f"{where}: description", empty=True)
    obsolete_info(data, where)
    _robots(data["robots"], f"{where}: robots")
    cameras = _cameras(data.get("cameras", []), f"{where}: cameras")
    others = [r["name"] for r in data["robots"] if r.get("read_only", False)]
    fixtures = _fixtures(data.get("fixtures", []), f"{where}: fixtures", others)
    for i, c in enumerate(data.get("cameras", [])):
        if c.get("fixture") is not None and c["fixture"].partition("/")[0] not in fixtures:
            _fail(f"{where}: cameras[{i}] ({c['name']}).fixture",
                  f"nie ma mocowania {c['fixture'].partition('/')[0]!r} w fixtures ({', '.join(fixtures) or 'brak'})")
    if "table" in data:
        _dataclass(Table, data["table"], f"{where}: table")
    if "card" in data:
        _dataclass(Card, data["card"], f"{where}: card")
    if "look" in data:
        _dataclass(Look, data["look"], f"{where}: look")
    if "wave" in data:
        _dataclass(WaveLimits, data["wave"], f"{where}: wave")
    _calib(data.get("calib", {}), f"{where}: calib", cameras)
    return data


# ------------------------------------------------------------------ entries
def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StandError(f"{path}: nie ma takiego pliku") from None
    except json.JSONDecodeError as exc:
        raise StandError(f"{path}: zly JSON (linia {exc.lineno}, kolumna {exc.colno}): {exc.msg}") from None


def _dumps(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _write_text(path: Path, text: str) -> bool:
    """Writes `text` unless the file already holds it; through a file next to it. True = written."""
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return True


def _same(a: Any, b: Any) -> bool:
    """The same entry: equal apart from the top-level `time`."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return a == b
    return {k: v for k, v in a.items() if k != "time"} == {k: v for k, v in b.items() if k != "time"}


def write_entry(folder: Path, stem: str, body: dict[str, Any], prefer: str | None = None) -> tuple[str, bool]:
    """Stores a dated calibration entry; returns (its name, whether a file was written).

    An entry with the same content (`prefer` checked first) is reused. Otherwise the entry gets
    `stem`, or `stem-2`, `stem-3`... when that is taken: an existing file is never overwritten.
    """
    body = json.loads(json.dumps(body))
    names = ([prefer] if prefer else []) + sorted(p.stem for p in folder.glob("*.json"))
    for name in names:
        path = folder / f"{name}.json"
        try:
            if path.is_file() and _same(json.loads(path.read_text(encoding="utf-8")), body):
                return name, False
        except ValueError:
            continue
    folder.mkdir(parents=True, exist_ok=True)
    path, k = folder / f"{stem}.json", 2
    while path.exists():
        path, k = folder / f"{stem}-{k}.json", k + 1
    path.write_text(_dumps(body), encoding="utf-8")
    return path.stem, True


def _pose_entry(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    w = str(path)
    _keys(data, w, ("cameras",), ("time", "tag_size", "joint_offsets"))
    if not isinstance(data["cameras"], dict):
        _fail(f"{w}: cameras", "oczekiwano obiektu {kamera: poza}")
    for name, cam in data["cameras"].items():
        _keys(cam, f"{w}: cameras.{name}", ("T_cam2base", "calibration"), ())
        _matrix(cam["T_cam2base"], f"{w}: cameras.{name}.T_cam2base", 4)
        if not isinstance(cam["calibration"], dict):
            _fail(f"{w}: cameras.{name}.calibration", "oczekiwano obiektu")
    if "joint_offsets" in data:
        _numbers(data["joint_offsets"], f"{w}: joint_offsets")
    return data


def _lens_entry(path: Path, camera: str) -> dict[str, Any]:
    data = _read_json(path)
    w = str(path)
    _keys(data, w, ("camera", "width", "height", "K"), ("time", "dist", "intrinsics_from", "intrinsics_info"))
    if data["camera"] != camera:
        _fail(f"{w}: camera", f"wpis kamery {data['camera']!r}, a wskazuje go kamera {camera!r}")
    _positive_int(data["width"], f"{w}: width")
    _positive_int(data["height"], f"{w}: height")
    _matrix(data["K"], f"{w}: K", 3)
    dist = data.get("dist")
    if dist is not None and (not isinstance(dist, list) or not all(_is_num(x) for x in dist)):
        _fail(f"{w}: dist", "oczekiwano listy liczb albo null")
    if not isinstance(data.get("intrinsics_info", {}), dict):
        _fail(f"{w}: intrinsics_info", "oczekiwano obiektu")
    return data


def _dynamics_entry(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict):
        _fail(str(path), "oczekiwano obiektu")
    try:
        Dynamics.from_dict(data)
    except (TypeError, ValueError) as exc:
        raise StandError(f"{path}: {exc}") from None
    return data


def lens_body(rec: CameraRecord) -> dict[str, Any]:
    """A camera's lens as a `calib/lens` entry."""
    body: dict[str, Any] = {"camera": rec.name, "width": rec.width, "height": rec.height}
    if rec.intrinsics_info.get("time"):
        body["time"] = rec.intrinsics_info["time"]
    body.update(K=rec.K, dist=rec.dist, intrinsics_from=rec.intrinsics_from, intrinsics_info=rec.intrinsics_info)
    return body


# ------------------------------------------------------------------ the stand
@dataclass
class Stand:
    """A validated stand folder: `stand.json` and the calibration entries it names."""

    folder: Path
    data: dict[str, Any]
    poses: dict[str, Any] | None = None
    lens: dict[str, dict[str, Any]] = field(default_factory=dict)
    dynamics: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def description(self) -> str:
        return self.data.get("description", "")

    @property
    def obsolete(self) -> dict[str, str] | None:
        return obsolete_info(self.data)

    def obsolete_text(self) -> str:
        """The warning for an obsolete stand; "" for a current one."""
        info = self.obsolete
        if info is None:
            return ""
        when = f" od {info['date']}" if info["date"] else ""
        return f"stanowisko {self.name} jest NIEAKTUALNE{when}" + (f": {info['reason']}" if info["reason"] else "")

    def workspace(self) -> Workspace:
        """The workspace this stand describes, bound to it (`Workspace.stand`)."""
        d = copy.deepcopy(self.data)
        driven = next(r for r in d["robots"] if not r.get("read_only", False))
        others = [RobotRecord(r["name"], r["robot"], xyz=r.get("xyz", [0.0, 0.0, 0.0]), yaw_deg=r.get("yaw_deg", 0.0),
                              backend=r.get("backend"), port=r.get("port"), joints=dict(r.get("joints", {})),
                              base_drive=r.get("base_drive", False))
                  for r in d["robots"] if r.get("read_only", False)]
        fixtures = [FixtureRecord(**f) for f in d.get("fixtures", [])]
        ws = Workspace(robot=driven["robot"], backend=driven.get("backend") or "sim", port=driven.get("port"),
                       home=driven.get("home"), other_robots=others, fixtures=fixtures, stand=str(self.folder))
        if "table" in d:
            ws.table = d["table"]
        if "card" in d:
            ws.card = d["card"]
        if "look" in d:
            ws.look = from_mapping(Look, d["look"], "look")
        if "wave" in d:
            ws.wave = from_mapping(WaveLimits, d["wave"], "wave")
        for c in d.get("cameras", []):
            simulated = c["source"] == "sim"
            ws.cameras.append(CameraRecord(
                c["name"], c["source"], c["width"], c["height"], c.get("fps", 30), enabled=c.get("enabled", True),
                wobbly=c.get("wobbly", False), wobble_px=c.get("wobble_px", WOBBLE_PX),
                controls=CameraControls(**c.get("controls", {})), sim_pose=c.get("sim_pose") if simulated else None,
                fixture=c.get("fixture")))
        for cam, lens in copy.deepcopy(self.lens).items():
            rec = ws.camera(cam)
            rec.K, rec.dist = lens["K"], lens.get("dist")
            rec.intrinsics_from = lens.get("intrinsics_from", "szachownica")
            rec.intrinsics_info = dict(lens.get("intrinsics_info", {}))
        if self.poses is not None:
            for cam, entry in copy.deepcopy(self.poses["cameras"]).items():
                try:
                    rec = ws.camera(cam)
                except KeyError:
                    continue                            # a camera no longer on the stand
                rec.T_cam2base, rec.calibration = entry["T_cam2base"], dict(entry["calibration"])
            ws.joint_offsets = {k: float(v) for k, v in self.poses.get("joint_offsets", {}).items()}
            ws.calib_preset = d["calib"]["poses"]
        if self.dynamics is not None:
            ws.dynamics = dict(self.dynamics)
        ws.mount_cameras()
        return ws


def load(folder: str | Path) -> Stand:
    """Reads and checks a stand folder: `stand.json` and every calibration entry it names."""
    folder = Path(folder).expanduser().resolve()
    path = folder / STAND_FILE
    if not path.is_file():
        raise StandError(f"{folder}: brak {STAND_FILE} - to nie jest folder stanowiska")
    data = validate(_read_json(path), str(path))
    stand = Stand(folder, data)
    calib = data.get("calib", {})
    if calib.get("poses"):
        stand.poses = _pose_entry(folder / POSES / f"{calib['poses']}.json")
    for cam in data.get("cameras", []):
        entry = calib.get("lens", {}).get(cam["name"])
        if entry is None:
            continue
        lens = _lens_entry(folder / LENS / cam["name"] / f"{entry}.json", cam["name"])
        if (lens["width"], lens["height"]) != (cam["width"], cam["height"]):
            _fail(f"{path}: cameras ({cam['name']})",
                  f"rozdzielczosc {cam['width']}x{cam['height']}, a obiektyw {LENS.as_posix()}/{cam['name']}/"
                  f"{entry}.json jest z {lens['width']}x{lens['height']} - zmierz obiektyw ponownie (krok 1) "
                  f"albo usun calib.lens.{cam['name']}")
        stand.lens[cam["name"]] = lens
    if calib.get("dynamics"):
        stand.dynamics = _dynamics_entry(folder / DYNAMICS / f"{calib['dynamics']}.json")
    return stand


def stand_data(ws: Workspace, keep: dict[str, Any], calib: dict[str, Any]) -> dict[str, Any]:
    """`stand.json` for `ws`: name, description and `obsolete` from `keep`, the old `stand.json`.

    A workspace connected in simulation keeps the stand's own backend and port: a rehearsal in sim
    does not change what drives the real arm.
    """
    old = next((r for r in keep.get("robots", []) if not r.get("read_only", False)), {})
    backend, port = ws.backend, ws.port
    if backend == "sim" and (old.get("backend") or "sim") != "sim":
        backend, port = old["backend"], old.get("port")
    driven: dict[str, Any] = {"robot": ws.robot, "read_only": False, "backend": backend, "port": port}
    if ws.home is not None:
        driven["home"] = ws.home
    robots = [driven]
    for r in ws.other_robots:
        other: dict[str, Any] = {"name": r.name, "robot": r.robot, "read_only": True, "xyz": r.xyz,
                                 "yaw_deg": r.yaw_deg, "backend": r.backend, "port": r.port}
        if r.backend is None and r.joints:
            other["joints"] = r.joints              # a fixed pose; a live robot's last reading is not setup
        if r.base_drive:
            other["base_drive"] = True
        robots.append(other)
    cameras = []
    for c in ws.cameras:
        cam: dict[str, Any] = {"name": c.name, "source": c.source, "width": c.width, "height": c.height,
                               "fps": c.fps, "enabled": c.enabled}
        if c.wobbly:
            cam["wobbly"] = True
        if c.wobble_px != WOBBLE_PX:
            cam["wobble_px"] = c.wobble_px
        if not c.controls.empty:
            cam["controls"] = c.controls.to_dict()
        if c.simulated and c.sim_pose is not None:
            cam["sim_pose"] = c.sim_pose
        if c.fixture is not None:
            cam["fixture"] = c.fixture
        cameras.append(cam)
    out = {"format": FORMAT, "name": keep["name"], "description": keep.get("description", ""),
           "obsolete": keep.get("obsolete", False), "robots": robots, "cameras": cameras,
           "table": ws.table, "card": ws.card, "look": asdict(ws.look), "wave": asdict(ws.wave), "calib": calib}
    if ws.fixtures:
        out["fixtures"] = [{k: v for k, v in asdict(f).items() if v is not None} for f in ws.fixtures]
    return json.loads(json.dumps(out))


def _ordered(data: dict[str, Any]) -> dict[str, Any]:
    return {k: data[k] for k in _ORDER if k in data} | {k: v for k, v in data.items() if k not in _ORDER}


def sync(ws: Workspace) -> list[str]:
    """Writes the workspace back to the stand it is bound to; returns the files written.

    New calibration results become new entries, `stand.json` changes only when the setup or a
    current entry does, and `station/` is exported again when what it holds would change.
    """
    if ws.stand is None:
        raise ValueError("stanowisko robocze nie jest zwiazane z folderem stanowiska")
    with _LOCK:
        return _sync(ws, Path(ws.stand))


def _sync(ws: Workspace, folder: Path) -> list[str]:
    path = folder / STAND_FILE
    if not path.is_file():
        raise StandError(f"{folder}: brak {STAND_FILE} - folder przeniesiony albo usuniety? "
                         f"Wczytaj go z nowego miejsca: twin stand load <folder>")
    keep = validate(_read_json(path), str(path))
    validate(stand_data(ws, keep, {}), str(path))   # names become file names: checked before any is written
    old = keep.get("calib", {})
    written: list[str] = []
    calib: dict[str, Any] = {}
    entry = ws.preset_entry()
    if entry["cameras"] or ws.joint_offsets:
        name, new = write_entry(folder / POSES, stamp_name(entry["time"]), entry, prefer=ws.calib_preset)
        calib["poses"], ws.calib_preset = name, name
        written += [f"{POSES.as_posix()}/{name}.json"] if new else []
    else:
        ws.calib_preset = None
    lens = {}
    for rec in ws.cameras:
        if rec.K is None:
            continue
        stamp = rec.intrinsics_info.get("time") or time.strftime("%Y-%m-%dT%H:%M:%S")
        name, new = write_entry(folder / LENS / rec.name, stamp_name(stamp), lens_body(rec),
                                prefer=old.get("lens", {}).get(rec.name))
        lens[rec.name] = name
        written += [f"{LENS.as_posix()}/{rec.name}/{name}.json"] if new else []
    calib["lens"] = lens
    if ws.dynamics:
        name, new = write_entry(folder / DYNAMICS, stamp_name(time.strftime("%Y-%m-%dT%H:%M:%S")), ws.dynamics,
                                prefer=old.get("dynamics"))
        calib["dynamics"] = name
        written += [f"{DYNAMICS.as_posix()}/{name}.json"] if new else []
    data = _ordered(stand_data(ws, keep, calib))
    validate(data, str(path))
    if _write_text(path, _dumps(data)):
        written.append(STAND_FILE)
    if refresh_station(load(folder)):
        written.append(f"{STATION}/")
    return written


def refresh_station(stand: Stand, force: bool = False) -> bool:
    """Exports the stand to `station/` when its `station.json` would change (always with `force`).

    The export goes to a folder next to it first, so a failed export leaves the old station.
    """
    ws = stand.workspace()
    data = ws.to_dict()
    data.pop("stand", None)
    station = stand.folder / STATION
    current = station / "station.json"
    want = json.dumps(data, indent=2, ensure_ascii=False)
    if (not force and (station / "scene.xml").is_file() and current.is_file()
            and current.read_text(encoding="utf-8") == want):
        return False
    tmp = stand.folder / f".{STATION}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        ws.export(tmp)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(station, ignore_errors=True)
    tmp.rename(station)
    return True


def save(ws: Workspace, folder: str | Path, *, name: str | None = None, description: str | None = None,
         history: bool = True) -> tuple[list[str], list[str]]:
    """Saves the workspace as the stand in `folder` (a new one, or an update) and binds it there.

    A workspace not yet bound to this stand brings its pose preset history along (`history`), under
    the same names where they are free. A preset for none of the workspace's cameras stays behind.
    Returns the files written and the presets left behind.
    """
    folder = Path(folder).expanduser().resolve()
    path = folder / STAND_FILE
    if path.is_file():
        keep = validate(_read_json(path), str(path))
    elif (folder / ".git").exists() or any((p / STAND_FILE).is_file() for p in _children(folder)):
        raise StandError(f"{folder}: to repozytorium stanowisk, nie folder jednego stanowiska - "
                         f"podaj {folder / '<nazwa>'}")
    else:
        keep = {"format": FORMAT, "name": name or folder.name, "description": "", "obsolete": False,
                "robots": [{"robot": ws.robot, "read_only": False}]}
    if name is not None:
        keep["name"] = name
    if description is not None:
        keep["description"] = description
    written: list[str] = []
    skipped: list[str] = []
    if ws.stand is None or Path(ws.stand).resolve() != folder:
        src = ws.presets_dir()
        renamed = {}
        cameras = {c.name for c in ws.cameras}
        for p in sorted(src.glob("*.json")) if src.is_dir() and history else []:
            try:
                body = _pose_entry(p)
            except StandError as exc:
                logger.warning("Pose preset skipped: %s", exc)
                skipped.append(p.stem)
                continue
            if body["cameras"] and not cameras & set(body["cameras"]):
                skipped.append(p.stem)
                continue
            renamed[p.stem], new = write_entry(folder / POSES, p.stem, body)
            written += [f"{POSES.as_posix()}/{renamed[p.stem]}.json"] if new else []
        ws.calib_preset = renamed.get(ws.calib_preset) if ws.calib_preset is not None else None
    if _write_text(path, _dumps(_ordered(validate(keep, str(path))))):
        written.append(STAND_FILE)
    ws.stand = str(folder)
    try:
        written += sync(ws)
    finally:
        ws.save()                                       # bound, even when the stand is not complete
    return sorted(set(written)), skipped


def _children(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")) if folder.is_dir() else []


def _setup(ws: Workspace) -> dict[str, Any]:
    """What a stand holds of a workspace: no preset name, no binding, no live robot's last reading."""
    data = ws.to_dict()
    data.pop("calib_preset", None)
    data.pop("stand", None)
    for r in data.get("other_robots", []):
        if r["backend"] is not None:
            r["joints"] = {}
    return data


def load_into(folder: str | Path, path: str | Path | None = None) -> tuple[Workspace, Stand, Path | None]:
    """Loads the stand into the workspace file `path` (default `workspace/twin.json`) and binds it.

    A workspace file holding a different setup is copied to `<file>.<time>.bak` first. Returns the
    workspace, the stand and that copy (None when there was nothing to keep). The caller warns
    about an obsolete stand (`Stand.obsolete_text`).
    """
    stand = load(folder)
    ws = stand.workspace()
    target = Path(path) if path else default_path()
    backup = None
    if target.is_file():
        try:
            old = Workspace.load(target)
            if old.backend == "sim":                    # a rehearsal in sim is not a different setup
                old.backend, old.port = ws.backend, ws.port
            differs = _setup(old) != _setup(ws)
        except Exception:                               # noqa: BLE001 - an unreadable file is kept too
            differs = True
        if differs:
            backup = target.with_name(f"{target.name}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
            shutil.copy2(target, backup)
    ws.save(target)
    return ws, stand, backup


def export(folder: str | Path) -> Path:
    """Exports the stand to its `station/` folder, whatever is there now. Returns that folder."""
    stand = load(folder)
    with _LOCK:
        refresh_station(stand, force=True)
    return stand.folder / STATION


def mark_obsolete(folder: str | Path, reason: str = "", date: str | None = None, clear: bool = False) -> Stand:
    """Marks the stand obsolete (no longer matching reality), or current again with `clear`."""
    folder = Path(folder).expanduser().resolve()
    path = folder / STAND_FILE
    if not path.is_file():
        raise StandError(f"{folder}: brak {STAND_FILE} - to nie jest folder stanowiska")
    data = validate(_read_json(path), str(path))
    data["obsolete"] = False if clear else {"reason": reason, "date": date or time.strftime("%Y-%m-%d")}
    _write_text(path, _dumps(_ordered(validate(data, str(path)))))
    return load(folder)


@dataclass
class Summary:
    """One line of `twin stand list`."""

    folder: Path
    name: str = ""
    description: str = ""
    obsolete: dict[str, str] | None = None
    robots: list[str] = field(default_factory=list)
    cameras: int = 0
    #: Newest calibration entry (its file name, a date and time), "" without any.
    newest: str = ""
    error: str = ""


def list_stands(root: str | Path) -> list[Summary]:
    """Stand folders in `root` (or `root` itself): current ones by name, then obsolete, then broken."""
    root = Path(root).expanduser().resolve()
    folders = [root] if (root / STAND_FILE).is_file() else [p for p in _children(root) if (p / STAND_FILE).is_file()]
    out = []
    for folder in folders:
        try:
            stand = load(folder)
        except StandError as exc:
            out.append(Summary(folder, name=folder.name, error=str(exc)))
            continue
        entries = [p.stem for p in (folder / "calib").rglob("*.json")] if (folder / "calib").is_dir() else []
        robots = [r["robot"] + (" (tylko odczyt)" if r.get("read_only") else "") for r in stand.data["robots"]]
        out.append(Summary(folder, stand.name, stand.description, stand.obsolete, robots,
                           len(stand.data.get("cameras", [])), max(entries, default="")))
    return sorted(out, key=lambda s: (bool(s.error), s.obsolete is not None, s.name.lower()))
