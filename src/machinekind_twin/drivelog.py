"""The drive log of another robot: its odometry, where the twin put its base and the tag sightings that
corrected it, as JSON Lines (docs/EPISODES.md, "Drive log of another robot")."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .episode import JOINT_DIGITS, METRE_DIGITS, SECOND_DIGITS, _rounded
from .robots import BaseTag
from .scene import _quat
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

    _write(log, {"format": FORMAT, "version": 1, "robot": rec.name, "description": rec.robot,
                 "xyz": [float(v) for v in rec.xyz], "yaw_deg": float(rec.yaw_deg),
                 "base_tag": None if tag is None else {"id": tag.id, "size": tag.size},
                 "started": now.isoformat(timespec="seconds"), "units": UNITS})
    return log


def pose_line(log: dict[str, Any], t: float, odometry: np.ndarray, base: np.ndarray,
              velocity: tuple[float, float, float] | None) -> None:
    """A base pose the robot reported at `t` (`time.monotonic()`), and the base velocity command running."""
    _write(log, {"t": _rounded(t - log["t0"], SECOND_DIGITS), "odometry": _pose(odometry),
                 "base": _pose(base), "velocity": _rounded(velocity, METRE_DIGITS)})


def tag_line(log: dict[str, Any], t: float, measured: np.ndarray, cameras: int, drift_m: float,
             drift_deg: float) -> None:
    """A sighting of the base tag in frames taken at `t`, and the drift after the correction it made."""
    _write(log, {"t": _rounded(t - log["t0"], SECOND_DIGITS), "tag": {
        "base": _pose(measured), "cameras": cameras, "drift_m": _rounded(drift_m, METRE_DIGITS),
        "drift_deg": _rounded(drift_deg, JOINT_DIGITS)}})


def end(log: dict[str, Any]) -> None:
    """The last line, whose `t` is the drive's length, and the file closed."""
    # Read back from the file, so the end line says what `summary` and `twin episode` say.
    done = summary(log["path"])
    assert done is not None
    _write(log, {"t": _rounded(time.monotonic() - log["t0"], SECOND_DIGITS), "end": {
        "path_m": _rounded(done["path_m"], METRE_DIGITS), "sightings": done["sightings"],
        "drift_m": done["drift_m"], "drift_deg": done["drift_deg"]}})
    log["file"].close()


def summary(path: Path) -> dict[str, Any] | None:
    """A drive log read back: its length [s], the path by odometry [m], the sightings, and the final and
    largest drift [m, deg], None without a sighting. None when the file is not a drive log."""
    with path.open("rb") as file:
        try:
            header = json.loads(file.readline())
        except ValueError:
            return None                                 # a compressed episode log, or no JSON at all
        if not isinstance(header, dict) or header.get("format") != FORMAT:
            return None
        lines = [json.loads(line) for line in file if line.strip()]

    # In the plane: a walking trunk bobs up and down, and that covers no distance.
    walked = [line["odometry"]["position"][:2] for line in lines if "odometry" in line]
    steps = np.diff(np.reshape(walked, (-1, 2)), axis=0)
    drift_m = [line["tag"]["drift_m"] for line in lines if "tag" in line]
    drift_deg = [line["tag"]["drift_deg"] for line in lines if "tag" in line]

    return {"duration_s": lines[-1]["t"] if lines else 0.0,
            "path_m": float(np.linalg.norm(steps, axis=1).sum()),
            "sightings": len(drift_m),
            "drift_m": drift_m[-1] if drift_m else None, "drift_deg": drift_deg[-1] if drift_deg else None,
            "max_drift_m": max(drift_m, default=None), "max_drift_deg": max(drift_deg, default=None)}


def _pose(T: np.ndarray) -> dict[str, Any]:
    return {"position": _rounded(T[:3, 3], METRE_DIGITS),
            "quaternion": _rounded(_quat(T[:3, :3]), METRE_DIGITS)}


def _write(log: dict[str, Any], line: dict[str, Any]) -> None:
    log["file"].write(json.dumps(line, separators=(",", ":")) + "\n")
