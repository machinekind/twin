"""Camera controls (focus, exposure) set on a V4L2 camera when the twin opens it.

A UVC camera that refocuses changes its focal length, so a lens calibration holds only with
autofocus off and a fixed focus. The twin sets these controls itself through the V4L2
`VIDIOC_S_CTRL` ioctl on the device node (Linux only, no extra dependencies). Elsewhere, and for
sources that are not a V4L2 device, nothing is applied.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: `_IOWR('V', 28, struct v4l2_control)`; `struct v4l2_control` is {__u32 id; __s32 value}.
VIDIOC_S_CTRL = 0xC008561C
_CAMERA_CLASS = 0x009A0900
CID_EXPOSURE_AUTO = _CAMERA_CLASS + 1
CID_EXPOSURE_ABSOLUTE = _CAMERA_CLASS + 2
CID_FOCUS_ABSOLUTE = _CAMERA_CLASS + 10
CID_FOCUS_AUTO = _CAMERA_CLASS + 12
#: `V4L2_CID_EXPOSURE_AUTO` menu values UVC cameras support: manual and aperture priority (auto).
EXPOSURE_MANUAL = 1
EXPOSURE_AUTO = 3


@dataclass
class CameraControls:
    """Controls the twin sets on every open. None leaves the camera's own setting."""

    autofocus: bool | None = None
    #: Fixed focus, in the camera's own units (often 0-255). Implies autofocus off.
    focus_absolute: int | None = None
    auto_exposure: bool | None = None
    #: Fixed exposure time [100 us]. Implies auto exposure off.
    exposure_absolute: int | None = None

    def __post_init__(self) -> None:
        for key in ("autofocus", "auto_exposure"):
            value = getattr(self, key)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{key}: true albo false, jest {value!r}")
        for key in ("focus_absolute", "exposure_absolute"):
            value = getattr(self, key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{key}: liczba calkowita >= 0, jest {value!r}")
        if self.autofocus and self.focus_absolute is not None:
            raise ValueError("focus_absolute dziala tylko z wylaczonym autofokusem (autofocus: false)")
        if self.auto_exposure and self.exposure_absolute is not None:
            raise ValueError("exposure_absolute dziala tylko z wylaczona auto ekspozycja (auto_exposure: false)")

    @property
    def empty(self) -> bool:
        return all(v is None for v in asdict(self).values())

    def to_dict(self) -> dict[str, Any]:
        """Only the controls that are set."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def v4l2(self) -> list[tuple[str, int, int]]:
        """(name, control id, value) in the order to set them: a mode before the value it unlocks."""
        out = []
        if self.autofocus is not None or self.focus_absolute is not None:
            on = bool(self.autofocus) and self.focus_absolute is None
            out.append(("autofocus", CID_FOCUS_AUTO, int(on)))
        if self.focus_absolute is not None:
            out.append(("focus_absolute", CID_FOCUS_ABSOLUTE, self.focus_absolute))
        if self.auto_exposure is not None or self.exposure_absolute is not None:
            auto = bool(self.auto_exposure) and self.exposure_absolute is None
            out.append(("auto_exposure", CID_EXPOSURE_AUTO, EXPOSURE_AUTO if auto else EXPOSURE_MANUAL))
        if self.exposure_absolute is not None:
            out.append(("exposure_absolute", CID_EXPOSURE_ABSOLUTE, self.exposure_absolute))
        return out


@dataclass
class Applied:
    """What `apply` did: controls set, controls the camera refused, or why nothing was tried."""

    ok: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: str = ""

    def text(self) -> str:
        """One line for the panel (Polish, like the rest of it)."""
        if self.skipped:
            return f"ustawienia kamery pominiete: {self.skipped}"
        parts = [f"{n} ok" for n in self.ok] + [f"{n}: {why}" for n, why in self.failed.items()]
        return "ustawienia kamery: " + (", ".join(parts) or "brak")


def device_node(source: str | int) -> str | None:
    """The V4L2 node of a camera source: "2" -> /dev/video2, a /dev/... path as it is; None otherwise."""
    text = str(source)
    if text.isdigit():
        return f"/dev/video{text}"
    return text if text.startswith("/dev/") else None


def apply(source: str | int, controls: CameraControls | None, *, platform: str | None = None,
          ioctl: Callable[[int, int, bytes], Any] | None = None,
          opener: Callable[[str], int] | None = None, closer: Callable[[int], None] = os.close) -> Applied:
    """Sets `controls` on the camera `source`. Never raises: a refused control is reported in `failed`.

    Off Linux, or for a source that is not a V4L2 device (a video file, a stream, "sim"), nothing is
    tried. `ioctl` and `opener` stand in for `fcntl.ioctl` and `os.open` in tests.
    """
    wanted = controls.v4l2() if controls is not None else []
    if not wanted:
        return Applied()
    node = device_node(source)
    if node is None:
        return Applied(skipped="zrodlo to nie kamera V4L2")
    if not (platform or sys.platform).startswith("linux"):
        logger.debug("Camera controls for %s skipped: V4L2 only exists on Linux", source)
        return Applied(skipped="tylko Linux (V4L2)")
    if ioctl is None:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Linux always has fcntl
            return Applied(skipped="brak fcntl")
        ioctl = fcntl.ioctl
    opener = opener or (lambda path: os.open(path, os.O_RDWR | os.O_NONBLOCK))
    try:
        fd = opener(node)
    except OSError as exc:
        logger.warning("Camera %s: cannot open %s to set controls: %s", source, node, exc)
        return Applied(failed={n: exc.strerror or str(exc) for n, _, _ in wanted})
    out = Applied()
    try:
        for name, cid, value in wanted:
            try:
                ioctl(fd, VIDIOC_S_CTRL, struct.pack("=Ii", cid, value))
                out.ok.append(name)
            except OSError as exc:
                out.failed[name] = exc.strerror or str(exc)
                logger.warning("Camera %s refused %s=%s: %s", source, name, value, out.failed[name])
    finally:
        closer(fd)
    return out
