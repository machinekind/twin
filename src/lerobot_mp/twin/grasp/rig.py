"""Per-rig measured values of the grasp student: `workspace/grasp.json`.

A separate file from `workspace/twin.json` on purpose: `Workspace.load` refuses unknown keys, so
a new key there would break every older reader on the rig. Written by the `grasp-bench` tools
and by the operator's editor; read by every grasp command.

    rig = Rig.load()                                   # missing file = empty rig
    rig.missing("shadow", simulated=False, now=time.time())   # [] = may run

Every reader returns plain values (None when not measured). A malformed value raises `RigError`
with a Polish message, already in `load`, so a bad file fails before anything connects.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ...paths import data_path
from .contract import (
    DOT_MAX_PX,
    FREEZE_DISABLED,
    LATENCY_MAX_S,
    OBJECT_HEIGHT_DEFAULT,
    TABLE_TOUCH_FAR,
    TEACH_MAX_AGE_S,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
    TOUCH_TOLERANCE,
    WRIST_DIST_2026_09_26,
    WRIST_K_2026_09_26,
)

logger = logging.getLogger(__name__)

DEFAULT_PATH = "workspace/grasp.json"          # resolved with lerobot_mp.paths.data_path
VERSION = 1
SECTIONS = ("table", "grasp", "gripper", "wrist", "latency", "guard", "overrides")
MODES = ("dry", "sim", "shadow", "real")
OBJECT_HEIGHT_RANGE = (0.005, 0.10)            # m
EFFORT_ABS_DEFAULT = 12.0                      # backend effort units, guessed


class RigError(ValueError):
    """Malformed per-rig file or value; the message is Polish ASCII."""


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _plain(value: Any) -> Any:
    """A JSON-ready copy: the teach tools measure with numpy, the file holds plain numbers."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


def _parse_stamp(value: str, where: str) -> float:
    # Python 3.10 (still allowed by pyproject) does not read the "Z" suffix.
    text = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        t = datetime.fromisoformat(text)
    except (TypeError, ValueError) as exc:
        raise RigError(f"{where}: to nie jest data ISO 8601: {value!r}") from exc
    if t.tzinfo is None:                        # the teach tools write UTC
        t = t.replace(tzinfo=timezone.utc)
    return t.timestamp()


@dataclass
class Rig:
    path: Path | None = None
    version: int = VERSION
    table: dict = field(default_factory=dict)
    grasp: dict = field(default_factory=dict)
    gripper: dict = field(default_factory=dict)
    wrist: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    guard: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)       # name -> note written by the operator

    # ------------------------------------------------------------------ file
    @staticmethod
    def load(path: str | Path | None = None) -> Rig:
        """Reads the file; a missing file is an empty rig with that path, not an error."""
        target = Path(path) if path else data_path(DEFAULT_PATH)
        if not target.is_file():
            logger.info("no per-rig file %s: nothing measured yet", target)
            return Rig(path=target)
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RigError(f"{target}: nie da sie odczytac pliku JSON ({exc})") from exc
        if not isinstance(data, dict):
            raise RigError(f"{target}: oczekiwano obiektu JSON na najwyzszym poziomie")
        unknown = set(data) - {"version", *SECTIONS}
        if unknown:
            raise RigError(f"{target}: nieznane pola {sorted(unknown)}")
        version = data.get("version", VERSION)
        if isinstance(version, bool) or not isinstance(version, int) or version != VERSION:
            raise RigError(f"{target}: wersja {version!r}, obslugiwana jest tylko {VERSION}")
        sections: dict[str, dict] = {}
        for name in SECTIONS:
            value = data.get(name)
            if value is None:
                value = {}
            if not isinstance(value, dict):
                raise RigError(f"{target}: pole {name!r} musi byc obiektem, jest {value!r}")
            sections[name] = value
        rig = Rig(path=target, version=version, **sections)
        try:
            rig._check()
        except RigError as exc:
            raise RigError(f"{target}: {exc}") from exc
        return rig

    def save(self) -> None:
        """Writes to a `.tmp` file next to it, then replaces: an interrupted write keeps the old file.

        Only a rig with a path (every `load` sets one) is written. `for_sim()` has none, so its
        stand-in values (dot 0 px, latency 0 s, shadow episodes "sim") never reach the rig's file.
        """
        if not self.path:
            raise RigError(
                "Rig bez sciezki (np. Rig.for_sim()) nie jest zapisywany - wczytaj plik przez Rig.load")
        self._check()
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(target)
        self.path = target

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"version": self.version}
        for name in SECTIONS:
            out[name] = _plain(getattr(self, name))
        return out

    @staticmethod
    def for_sim() -> Rig:
        """Every value the sim needs, nothing measured on a bench; the grasp point comes from the scene."""
        return Rig(
            table={"touch_tcp_z": TOUCH_EXPECTED_Z},
            grasp={"object_height": OBJECT_HEIGHT_DEFAULT},
            gripper={"stroke_close_s_operator": None, "freeze_threshold": FREEZE_DISABLED},
            wrist={"source": "sim", "dot_error_px": 0.0},
            latency={"camera_s": 0.0, "total_s": 0.0},
            overrides={"shadow_done": "sim"},
        )

    def _check(self) -> None:
        """Runs every reader once: raises RigError on the first malformed value."""
        if isinstance(self.version, bool) or self.version != VERSION:
            raise RigError(f"wersja {self.version!r}, obslugiwana jest tylko {VERSION}")
        self.touch_tcp_z()
        self.believed()
        self.grasp_yaw()
        self._tcp("table")
        self.stroke_s()
        self.freeze_threshold()
        self.wrist_source()
        self.intrinsics()
        self.upside_down()
        self.object_height()
        self.camera_latency_s()
        self.taught_age_s(0.0)
        self.effort_abs()
        self._num("wrist", "dot_error_px")
        self._num("latency", "total_s")
        self._num("latency", "actuation_s")
        self._num("latency", "infer_p95_s")
        joints = self.grasp.get("joint_offsets_deg")
        if joints is not None:
            self._vector(joints, 6, "grasp.joint_offsets_deg")

    # ------------------------------------------------------------------ plain values
    def _num(self, section: str, key: str) -> float | None:
        value = getattr(self, section).get(key)
        if value is None:
            return None
        if not _is_number(value) or not math.isfinite(value):
            raise RigError(f"{section}.{key}: oczekiwano skonczonej liczby, jest {value!r}")
        return float(value)

    @staticmethod
    def _vector(value: Any, n: int, where: str) -> np.ndarray:
        value = _plain(value)
        if not isinstance(value, list) or len(value) != n or not all(_is_number(v) for v in value):
            raise RigError(f"{where}: oczekiwano {n} liczb, jest {value!r}")
        v = np.array(value, dtype=np.float64)
        if not np.all(np.isfinite(v)):
            raise RigError(f"{where}: oczekiwano skonczonych liczb, jest {value!r}")
        return v

    def _tcp(self, section: str) -> np.ndarray | None:
        value = getattr(self, section).get("tcp")
        return None if value is None else self._vector(value, 3, f"{section}.tcp")

    def _override(self, name: str) -> bool:
        return bool(self.overrides.get(name))

    # ------------------------------------------------------------------ typed readers
    def touch_tcp_z(self) -> float | None:
        """Measured TCP z with the fingertips touching the table (m, base frame)."""
        return self._num("table", "touch_tcp_z")

    def believed(self) -> np.ndarray | None:
        """(3,) the believed grasp: taught xy, z = touch_tcp_z + TIP_CLEAR (training's tip clearance)."""
        tcp, touch = self._tcp("grasp"), self.touch_tcp_z()
        if tcp is None or touch is None:
            return None
        return np.array([tcp[0], tcp[1], touch + TIP_CLEAR])

    def grasp_yaw(self) -> float | None:
        return self._num("grasp", "yaw")

    def stroke_s(self) -> float | None:
        """The slowest recorded close stroke (s)."""
        strokes = [s for s in (self._num("gripper", "stroke_close_s_effort"),
                               self._num("gripper", "stroke_close_s_operator")) if s is not None]
        if not strokes:
            return None
        if min(strokes) <= 0.0:
            raise RigError(f"gripper.stroke_close_s_*: czas ruchu szczek musi byc dodatni, jest {strokes}")
        return max(strokes)

    def jaw_units_s(self) -> float | None:
        stroke = self.stroke_s()
        return None if stroke is None else 100.0 / stroke

    def freeze_threshold(self) -> float | None:
        return self._num("gripper", "freeze_threshold")

    def wrist_source(self) -> str | None:
        """Device path or index of the wrist camera, as a string; None when not set."""
        value = self.wrist.get("source")
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int, np.integer)):
            raise RigError(f"wrist.source: oczekiwano sciezki albo numeru kamery, jest {value!r}")
        text = str(value).strip()
        return text or None

    def intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        """(K, dist) of the upright raw frame: the rig's own pair, or the contract default."""
        K, dist = _plain(self.wrist.get("K")), _plain(self.wrist.get("dist"))
        if K is None and dist is None:
            return WRIST_K_2026_09_26.copy(), WRIST_DIST_2026_09_26.copy()
        # K and dist come from one calibration; half of a new one next to half of the default is wrong.
        if K is None or dist is None:
            raise RigError("wrist.K i wrist.dist podaje sie razem (z jednej kalibracji)")
        if not isinstance(K, list) or len(K) != 3:
            raise RigError(f"wrist.K: oczekiwano macierzy 3x3, jest {K!r}")
        K_arr = np.stack([self._vector(row, 3, "wrist.K") for row in K])
        if not isinstance(dist, list) or len(dist) not in (4, 5, 8, 12, 14):
            raise RigError(f"wrist.dist: oczekiwano 4, 5, 8, 12 albo 14 wspolczynnikow, jest {dist!r}")
        return K_arr, self._vector(dist, len(dist), "wrist.dist")

    def upside_down(self) -> bool:
        """The raw frame is rotated 180 degrees (the camera is mounted upside down); default True."""
        value = _plain(self.wrist.get("upside_down", True))
        if value is None:
            return True
        if not isinstance(value, bool):
            raise RigError(f"wrist.upside_down: oczekiwano true albo false, jest {value!r}")
        return value

    def object_height(self) -> float | None:
        return self._num("grasp", "object_height")

    def camera_latency_s(self) -> float:
        """Exposure to frame stamp (s); 0.0 while not measured (pairing by the stamp)."""
        value = self._num("latency", "camera_s")
        return 0.0 if value is None else value

    def taught_age_s(self, now: float) -> dict[str, float | None]:
        """Seconds since the table and the grasp point were taught; `now` is `time.time()`."""
        ages: dict[str, float | None] = {}
        for section in ("table", "grasp"):
            stamp = getattr(self, section).get("taught_at")
            if stamp is None:
                ages[section] = None
            elif not isinstance(stamp, str):
                raise RigError(f"{section}.taught_at: oczekiwano daty ISO 8601, jest {stamp!r}")
            else:
                ages[section] = now - _parse_stamp(stamp, f"{section}.taught_at")
        return ages

    def effort_abs(self) -> float:
        value = self._num("guard", "effort_abs")
        return EFFORT_ABS_DEFAULT if value is None else value

    # ------------------------------------------------------------------ refusals
    def missing(self, mode: str, simulated: bool, *, now: float | None = None) -> list[str]:
        """Refusal reasons for `mode` on this rig, in a fixed order; empty = may run.

        A reason that needs a value refused earlier in the list is not repeated: without a taught
        table there is no `table touch far` and no age of the table touch. Sim mode needs nothing
        from the file (on a real backend the run refuses it as `not armed` before asking).
        """
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if simulated or mode == "sim":
            return []
        moving = mode in ("shadow", "real")
        out: list[str] = []
        touch = self.touch_tcp_z()
        grasp_tcp, table_tcp = self._tcp("grasp"), self._tcp("table")
        if touch is None:
            out.append("table not taught")
        elif abs(touch - TOUCH_EXPECTED_Z) > TOUCH_TOLERANCE and not self._override("table"):
            out.append("table touch implausible")
        if grasp_tcp is None or self.grasp_yaw() is None:
            out.append("grasp point not taught")
        if touch is not None and grasp_tcp is not None and (
                table_tcp is None or math.hypot(*(table_tcp[:2] - grasp_tcp[:2])) > TABLE_TOUCH_FAR):
            out.append("table touch far")
        if now is not None and not self._override("taught"):
            ages = self.taught_age_s(now)
            taught = {"table": touch is not None, "grasp": grasp_tcp is not None}
            if any(taught[k] and (ages[k] is None or ages[k] > TEACH_MAX_AGE_S) for k in taught):
                out.append("teaching old")
        if moving:
            height = self.object_height()
            if height is None or not OBJECT_HEIGHT_RANGE[0] <= height <= OBJECT_HEIGHT_RANGE[1]:
                out.append("object height not set")
        if self.wrist_source() is None:
            out.append("wrist camera not set")
        if moving:
            dot = self._num("wrist", "dot_error_px")
            if dot is None:
                out.append("dot not checked")
            elif dot > DOT_MAX_PX and not self._override("dot"):
                out.append("dot off")
            total = self._num("latency", "total_s")
            if total is None:
                out.append("latency not measured")
            elif total > LATENCY_MAX_S and not self._override("latency"):
                out.append("latency high")
        if mode == "real":
            if self.stroke_s() is None:
                out.append("jaw stroke not measured")
            if self.freeze_threshold() is None:
                out.append("gripper freeze undecided")
            if not self._override("shadow_done"):
                out.append("no shadow episodes")
        return out
