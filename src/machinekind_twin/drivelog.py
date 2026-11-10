"""The drive log of another robot: its odometry, where the twin put its base and the tag sightings that
corrected it, as JSON Lines (docs/EPISODES.md, "Drive log of another robot")."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import basefix
from .episode import JOINT_DIGITS, METRE_DIGITS, SECOND_DIGITS, _rounded
from .robots import BaseTag
from .scene import _quat, _rotmat
from .workspace import RobotRecord

#: The header's `format`: what tells a drive log from an episode log.
FORMAT = "machinekind-twin-other-robot"
UNITS = {"t": "s", "position": "m", "quaternion": "w, x, y, z", "velocity": "m/s, m/s, rad/s",
         "drift_m": "m", "drift_deg": "deg", "path_m": "m"}


def start(folder: Path, rec: RobotRecord, tag: BaseTag | None) -> dict[str, Any]:
    """A new log of the robot `rec` in `folder`, its header written."""
    now = datetime.now().astimezone()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"other-{rec.name}-{now:%Y%m%d-%H%M%S}.jsonl"
    # Line buffered: a twin that dies mid-drive leaves every line but the one it was writing.
    log = {"file": path.open("x", encoding="utf-8", buffering=1), "path": path, "t0": time.monotonic()}
    log.update(_totals(rec.T_base()))

    _write(log, {"format": FORMAT, "version": 1, "robot": rec.name, "description": rec.robot,
                 "xyz": [float(v) for v in rec.xyz], "yaw_deg": float(rec.yaw_deg),
                 "base_tag": None if tag is None else {"id": tag.id, "size": tag.size},
                 "started": now.isoformat(timespec="seconds"), "units": UNITS})
    return log


def pose_line(log: dict[str, Any], t: float, odometry: np.ndarray, base: np.ndarray,
              velocity: tuple[float, float, float] | None) -> None:
    """A base pose the robot reported at `t` (`time.monotonic()`), and the base velocity command running."""
    _line(log, {"t": _rounded(t - log["t0"], SECOND_DIGITS), "odometry": _pose(odometry),
                "base": _pose(base), "velocity": _rounded(velocity, METRE_DIGITS)})


def tag_line(log: dict[str, Any], t: float, measured: np.ndarray, cameras: int, drift_m: float,
             drift_deg: float) -> None:
    """A sighting of the base tag in a frame taken at `t`, and the drift right after its correction."""
    _line(log, {"t": _rounded(t - log["t0"], SECOND_DIGITS), "tag": {
        "base": _pose(measured), "cameras": cameras, "drift_m": _rounded(drift_m, METRE_DIGITS),
        "drift_deg": _rounded(drift_deg, JOINT_DIGITS)}})


def end(log: dict[str, Any]) -> None:
    """The last line, from the running totals, whose `t` is the drive's length. The caller closes the file."""
    done = _done(log)
    _write(log, {"t": _rounded(time.monotonic() - log["t0"], SECOND_DIGITS), "end": {
        "path_m": done["path_m"], "sightings": done["sightings"], "drift_m": done["drift_m"],
        "drift_deg": done["drift_deg"]}})


def close(log: dict[str, Any]) -> None:
    """The file closed. On a full disk the buffered rest is lost; the file is closed all the same."""
    try:
        log["file"].close()
    except OSError:
        pass


def summary(path: Path) -> dict[str, Any] | None:
    """A drive log read back: its length [s], the path by odometry [m], the sightings, and the final and
    largest drift [m, deg]. None when the file is not a drive log."""
    with path.open("rb") as file:
        try:
            header = json.loads(file.readline())
        except ValueError:
            return None                                 # a compressed episode log, or no JSON at all
        if not isinstance(header, dict) or header.get("format") != FORMAT:
            return None
        record = RobotRecord(header["robot"], header["description"], xyz=header["xyz"],
                             yaw_deg=header["yaw_deg"])
        totals = _totals(record.T_base())
        for line in file:                               # one at a time: a long drive need not fit in memory
            if line.strip():
                _count(totals, json.loads(line))
    return _done(totals)


def _totals(record: np.ndarray) -> dict[str, Any]:
    """Running totals before the first line, of a robot whose record puts its fixed frame at `record`."""
    return {"record": record, "t": 0.0, "xy": None, "path_m": 0.0, "sightings": 0, "drift_m": 0.0,
            "drift_deg": 0.0, "max_drift_m": 0.0, "max_drift_deg": 0.0}


def _line(log: dict[str, Any], line: dict[str, Any]) -> None:
    _write(log, line)
    _count(log, line)


def _count(totals: dict[str, Any], line: dict[str, Any]) -> None:
    """One line into the running totals: the log counts what it writes (`_line`), `summary` what it reads,
    so the end line and `summary` agree."""
    totals["t"] = line["t"]
    if "tag" in line:
        totals["sightings"] += 1
    if "odometry" not in line:
        return

    # In the plane: a walking trunk bobs up and down, and that covers no distance.
    xy = np.asarray(line["odometry"]["position"][:2], float)
    if totals["xy"] is not None:
        totals["path_m"] += float(np.linalg.norm(xy - totals["xy"]))
    totals["xy"] = xy

    # From the line itself, not from the last sighting: the robot walks on, and the drift with it.
    shift, turn = basefix.planar_error(totals["record"] @ _matrix(line["odometry"]), _matrix(line["base"]))
    totals["drift_m"] = float(np.linalg.norm(shift))
    totals["drift_deg"] = float(np.degrees(abs(turn)))
    totals["max_drift_m"] = max(totals["max_drift_m"], totals["drift_m"])
    totals["max_drift_deg"] = max(totals["max_drift_deg"], totals["drift_deg"])


def _done(totals: dict[str, Any]) -> dict[str, Any]:
    return {"duration_s": totals["t"], "path_m": _rounded(totals["path_m"], METRE_DIGITS),
            "sightings": totals["sightings"], "drift_m": _rounded(totals["drift_m"], METRE_DIGITS),
            "drift_deg": _rounded(totals["drift_deg"], JOINT_DIGITS),
            "max_drift_m": _rounded(totals["max_drift_m"], METRE_DIGITS),
            "max_drift_deg": _rounded(totals["max_drift_deg"], JOINT_DIGITS)}


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rotmat(pose["quaternion"])
    T[:3, 3] = pose["position"]
    return T


def _pose(T: np.ndarray) -> dict[str, Any]:
    return {"position": _rounded(T[:3, 3], METRE_DIGITS),
            "quaternion": _rounded(_quat(T[:3, :3]), METRE_DIGITS)}


def _write(log: dict[str, Any], line: dict[str, Any]) -> None:
    log["file"].write(json.dumps(line, separators=(",", ":")) + "\n")
