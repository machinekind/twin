"""The detector contract: camera frames in, what is seen where in each image out.

A detector (a colour threshold, a neural network, a vision-language model behind a CLI) looks at
camera frames by camera name and says what it sees and where IN THE IMAGE: a label, a pixel and,
when it can, a box, a silhouette mask and the object's physical size. It knows nothing about
coordinates: no intrinsics, no camera poses, no robot, no clock. The twin does the geometry. It
turns each pixel into a ray through the calibrated camera, intersects the rays of two cameras or
one ray with the table, fits a box of the given size to the masks, tracks each object over time
and hands strategies `Observation.objects[label]` in the robot's base frame, with `age_s`.

A detector from another package depends on this package, never on the twin. It subclasses
`Detector` and registers under the `machinekind_twin.detectors` entry point group
(`entry_points.DETECTORS`). The twin builds it from its settings exactly like a strategy.

The twin calls `detect` on a thread of its own, one call at a time, starting a call no sooner
than `interval_s` after the previous one started. A slow detector (seconds per answer) only makes
the poses older. Strategies never wait for it.

Pixels follow OpenCV: (0, 0) is the centre of the top-left pixel, u grows right, v grows down,
in the resolution of the frame the detector was given. A detector that works on a resized copy
maps its answer back.

Labels are compared stripped and lower-cased. One object per label: when one camera reports a
label twice, the twin keeps the surer detection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .strategy import Frame, S

__all__ = ["DetectRequest", "Detection", "Detector", "DetectorInfo"]


@dataclass(frozen=True)
class DetectorInfo:
    """What a detector learns once, when the twin starts it."""

    #: Cameras the twin could place detections from AT START: enabled, calibrated, trusted (or
    #: simulated), fixed. The set may change during a run (a camera gets calibrated or disabled);
    #: `DetectRequest.frames` lists the cameras of each call and is the one to follow.
    cameras: tuple[str, ...]
    #: The robot's name. Never needed for geometry.
    robot: str
    #: True when every camera is rendered by the simulation (ideal pinhole, no distortion).
    simulated: bool


@dataclass(frozen=True)
class DetectRequest:
    """One call's input."""

    #: Frames by camera name, fetched only when read. Iterating and `len` give the cameras
    #: offered in this call. Reading a camera without a frame younger than a second raises
    #: KeyError; `name in frames` and `frames.get(name)` fetch it and tell whether it is there.
    #: Images are RGB uint8 (height, width, 3), raw, never undistorted.
    frames: Mapping[str, Frame]
    #: Call number since `start`, from 0.
    step: int


@dataclass(frozen=True, eq=False)
class Detection:
    """One object seen in one camera's frame. Compared by identity (it may hold an array)."""

    #: What the object is: the key of `Observation.objects`, e.g. "cube" or "red pen".
    label: str
    #: The camera whose frame this is: a key of `DetectRequest.frames`.
    camera: str
    #: (u, v) [px]: for an object on the table, the centre of its footprint; for a held one, its centre.
    pixel: tuple[float, float]
    #: (u0, v0, u1, v1) [px]; None = not known. The twin only shows it.
    box: tuple[float, float, float, float] | None = None
    #: How sure the detector is, 0..1.
    confidence: float = 1.0
    #: Bool image of the frame's shape, True on the object's silhouette; None = not given.
    mask: np.ndarray | None = None
    #: Height above the table top [m] of the point `pixel` marks; None = on the table (0), or
    #: half the object's height when `size_m` is given. One camera alone cannot tell depth; this fixes it.
    height_m: float | None = None
    #: The object's size [m] as (x, y, z) of a box resting on the table; None = unknown. With a
    #: `mask` the twin fits a box of this size to the masks of every camera, which also gives yaw.
    size_m: tuple[float, float, float] | None = None
    #: How far off `pixel` may be, in THIS frame's pixels; None = the detector's `pixel_error_px`.
    #: A detector that answers on a resized copy scales its error back with the pixel.
    pixel_error_px: float | None = None
    #: Age of the frames this detection was made from, relative to this call's frames [s]. 0 means
    #: this call's frames. A detector that repeats an earlier answer sets it to the time between
    #: that answer's frame and this call's frame from `camera`. The twin adds it to the published
    #: `age_s`.
    age_s: float = 0.0


class Detector(Protocol[S]):
    """An object detector: a protocol with defaults, meant to be subclassed.

    Subclass it as `class Mine(Detector[MineSettings])` and define `detect`. The base supplies
    the rest. `settings` works as for a `Strategy`: the twin builds the dataclass from the panel's
    form, prefilled from the config section named like the detector, and creates the detector as
    `cls(settings)`, or `cls()` without settings. `interval_s` and `pixel_error_px` are read from
    the instance after construction, so a detector may set them from its settings. The twin
    checks a detector by its attributes, never with isinstance.

    `start(info)` runs once before the first `detect` and may raise to refuse. `stop(reason)` runs
    exactly once after a successful `start`, possibly from another thread while `detect` runs; a
    detector that blocks on a subprocess should then make `detect` return or raise soon. The twin
    discards whatever `detect` returns after `stop`. A `detect` that raises is logged and shown in
    the panel; the twin calls again later, waiting longer after each consecutive failure.
    """

    #: The settings dataclass; None = the detector takes no settings.
    settings: type[S] | None = None
    #: Shortest time between the starts of two `detect` calls [s].
    interval_s: float = 1.0
    #: How far off this detector's pixels may be [px], in the frames it is given. The twin pairs
    #: two cameras' rays of one label only when they pass this close, scaled by depth.
    pixel_error_px: float = 3.0

    def start(self, info: DetectorInfo) -> None:
        """Before the first `detect`. Raise to refuse."""

    def stop(self, reason: str) -> None:
        """Once, after the last `detect`, or while one runs on the twin's thread."""

    def detect(self, request: DetectRequest) -> Sequence[Detection]: ...
