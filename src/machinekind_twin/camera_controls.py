"""Camera controls (focus, exposure) set on a camera when the twin opens it.

A UVC camera that refocuses changes its focal length, so a lens calibration holds only with
autofocus off and a fixed focus. The twin sets these controls itself and reads each one back,
with one backend per OS and no extra dependencies:

* Linux: the V4L2 `VIDIOC_S_CTRL` / `VIDIOC_G_CTRL` ioctls on the device node.
* macOS: UVC class requests to the camera's USB device through IOKit (`uvc_mac`).
* Windows: OpenCV's DirectShow properties on the open capture (focus only).

Sources that are not a camera (a video file, a stream, "sim") get nothing.
"""

from __future__ import annotations

import errno
import logging
import os
import struct
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: `_IOWR('V', 28, struct v4l2_control)`; `struct v4l2_control` is {__u32 id; __s32 value}.
VIDIOC_S_CTRL = 0xC008561C
#: `_IOWR('V', 27, struct v4l2_control)`.
VIDIOC_G_CTRL = 0xC008561B
#: `_IOWR('V', 36, struct v4l2_queryctrl)`.
VIDIOC_QUERYCTRL = 0xC0445624
#: `struct v4l2_queryctrl`: id, type, name[32], minimum, maximum, step, default_value, flags, reserved[2].
_QUERYCTRL = struct.Struct("=II32siiiiI2I")
_CTRL_FLAG_DISABLED = 0x0001
_CAMERA_CLASS = 0x009A0900
CID_EXPOSURE_AUTO = _CAMERA_CLASS + 1
CID_EXPOSURE_ABSOLUTE = _CAMERA_CLASS + 2
CID_FOCUS_ABSOLUTE = _CAMERA_CLASS + 10
CID_FOCUS_AUTO = _CAMERA_CLASS + 12
#: `V4L2_CID_EXPOSURE_AUTO` menu values UVC cameras support: manual and aperture priority (auto).
EXPOSURE_MANUAL = 1
EXPOSURE_AUTO = 3
_V4L2_CID = {"autofocus": CID_FOCUS_AUTO, "focus_absolute": CID_FOCUS_ABSOLUTE,
             "auto_exposure": CID_EXPOSURE_AUTO, "exposure_absolute": CID_EXPOSURE_ABSOLUTE}


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

    def wanted(self) -> list[tuple[str, int]]:
        """(name, value) in the order to set them, a mode before the value it unlocks; modes are 1 on, 0 off."""
        out = []
        if self.autofocus is not None or self.focus_absolute is not None:
            out.append(("autofocus", int(bool(self.autofocus) and self.focus_absolute is None)))
        if self.focus_absolute is not None:
            out.append(("focus_absolute", self.focus_absolute))
        if self.auto_exposure is not None or self.exposure_absolute is not None:
            out.append(("auto_exposure", int(bool(self.auto_exposure) and self.exposure_absolute is None)))
        if self.exposure_absolute is not None:
            out.append(("exposure_absolute", self.exposure_absolute))
        return out

    def v4l2(self) -> list[tuple[str, int, int]]:
        """(name, control id, value) in the order to set them, with V4L2's values."""
        return [(name, _V4L2_CID[name], _to_v4l2(name, value)) for name, value in self.wanted()]


@dataclass
class Applied:
    """What `apply` did: controls set and read back, controls that failed, or why nothing was tried."""

    #: Set, and the camera reports the value that was set.
    ok: list[str] = field(default_factory=list)
    #: Refused, not read back, or read back as another value; the reason is for the panel.
    failed: dict[str, str] = field(default_factory=dict)
    skipped: str = ""
    #: Values the camera reported after the set, in `CameraControls.wanted` units.
    read: dict[str, int] = field(default_factory=dict)
    #: Controls the camera does not have (also in `failed`).
    missing: list[str] = field(default_factory=list)
    #: The backend that talked to the camera ("V4L2", "UVC", "DirectShow").
    backend: str = ""

    def text(self) -> str:
        """One line for the panel (Polish, like the rest of it)."""
        if self.skipped:
            return f"ustawienia kamery pominiete: {self.skipped}"
        parts = [f"{n} ok" for n in self.ok] + [f"{n}: {why}" for n, why in self.failed.items()]
        head = "ustawienia kamery" + (f" ({self.backend})" if self.backend else "")
        return head + ": " + (", ".join(parts) or "brak")


@dataclass(frozen=True)
class Range:
    """The values a control takes: `lo`..`hi` in steps of `step`."""

    lo: int
    hi: int
    step: int = 1


class ControlError(Exception):
    """A control the camera refused or did not report; the message is for the panel."""

    def __init__(self, message: str, *, missing: bool = False):
        super().__init__(message)
        #: The camera does not have this control at all.
        self.missing = missing


class NoControls(Exception):
    """The source has no controls the twin can reach; the message says why (for the panel)."""


class Device(Protocol):
    """One camera's controls, by `CameraControls.wanted` names and units."""

    backend: str

    def set(self, name: str, value: int) -> None: ...

    def get(self, name: str) -> int: ...

    def range(self, name: str) -> Range | None: ...

    def close(self) -> None: ...


def device_node(source: str | int) -> str | None:
    """The V4L2 node of a camera source: "2" -> /dev/video2, a /dev/... path as it is; None otherwise."""
    text = str(source)
    if text.isdigit():
        return f"/dev/video{text}"
    return text if text.startswith("/dev/") else None


def open_device(source: str | int, *, platform: str | None = None, cap: Any = None,
                ioctl: Callable[[int, int, bytes], Any] | None = None,
                opener: Callable[[str], int] | None = None, closer: Callable[[int], None] = os.close) -> Device:
    """The controls of camera `source` on this OS. Raises `NoControls` or `ControlError`.

    `cap` is the open `cv2.VideoCapture` of the source (Windows sets controls through it).
    `ioctl`, `opener` and `closer` stand in for `fcntl.ioctl`, `os.open` and `os.close` in tests.
    """
    platform = platform or sys.platform
    text = str(source)
    if platform.startswith("linux"):
        node = device_node(source)
        if node is None:
            raise NoControls("zrodlo to nie kamera V4L2")
        return V4L2Device(node, ioctl=ioctl, opener=opener, closer=closer)
    if platform == "darwin":
        if not text.isdigit():
            raise NoControls("zrodlo to nie kamera (na macOS kamera to numer)")
        from . import uvc_mac

        return uvc_mac.open_camera(int(text))
    if platform.startswith("win"):
        if not text.isdigit():
            raise NoControls("zrodlo to nie kamera")
        if cap is None:
            raise NoControls("na Windows tylko przy otwartym strumieniu kamery")
        return DirectShowDevice(cap)
    raise NoControls(f"brak sterowania kamera na {platform}")


def apply(source: str | int, controls: CameraControls | None, *, platform: str | None = None, cap: Any = None,
          ioctl: Callable[[int, int, bytes], Any] | None = None,
          opener: Callable[[str], int] | None = None, closer: Callable[[int], None] = os.close,
          device: Callable[[], Device] | None = None) -> Applied:
    """Sets `controls` on the camera `source` and reads each one back. Never raises.

    A refused control, or one that reads back another value, is reported in `failed`; a source with
    no reachable controls in `skipped`. `device` opens a stand-in `Device` (tests); the other
    keywords go to `open_device`.
    """
    wanted = controls.wanted() if controls is not None else []
    if not wanted:
        return Applied()
    try:
        dev = device() if device is not None else open_device(source, platform=platform, cap=cap, ioctl=ioctl,
                                                              opener=opener, closer=closer)
    except NoControls as exc:
        logger.debug("Camera controls for %s skipped: %s", source, exc)
        return Applied(skipped=str(exc))
    except ControlError as exc:
        logger.warning("Camera %s: cannot reach its controls: %s", source, exc)
        return Applied(failed={n: str(exc) for n, _ in wanted})
    out = Applied(backend=dev.backend)
    try:
        for name, value in wanted:
            try:
                dev.set(name, value)
            except ControlError as exc:
                out.failed[name] = str(exc)
                if exc.missing:
                    out.missing.append(name)
                logger.warning("Camera %s refused %s=%s: %s", source, name, value, exc)
                continue
            try:
                got = dev.get(name)
            except ControlError as exc:
                out.failed[name] = f"ustawione, ale bez odczytu ({exc})"
                continue
            out.read[name] = got
            if got == value:
                out.ok.append(name)
            else:
                out.failed[name] = f"odczyt {got} zamiast {value}"
                logger.warning("Camera %s: %s set to %s reads back %s", source, name, value, got)
    finally:
        dev.close()
    return out


def focus_range(source: str | int, *, platform: str | None = None, cap: Any = None,
                device: Callable[[], Device] | None = None) -> Range | None:
    """The values the camera's focus takes; None when the camera or the backend cannot tell."""
    try:
        dev = device() if device is not None else open_device(source, platform=platform, cap=cap)
    except (NoControls, ControlError):
        return None
    try:
        return dev.range("focus_absolute")
    except ControlError:
        return None
    finally:
        dev.close()


def lens_focus_problem(focus: int, done: Applied | None) -> str:
    """Why a lens measured at `focus` does not hold on a camera whose controls came out as `done`.

    "" = it holds, or nothing was applied yet (`done` None).
    """
    if done is None:
        return ""
    if done.skipped:
        return f"ostrosci {focus} nie da sie ustawic ({done.skipped})"
    for name in ("autofocus", "focus_absolute"):
        # A camera with a manual focus and no autofocus has nothing to turn off.
        if name in done.failed and not (name == "autofocus" and name in done.missing):
            return f"ostrosc nie zablokowana ({name}: {done.failed[name]})"
    if done.read.get("autofocus", 0):
        return "autofokus wlaczony"
    got = done.read.get("focus_absolute")
    if got is None:
        return "ostrosc nie ustawiona"
    if got != focus:
        return f"ostrosc {got}, a obiektyw zmierzony przy {focus}"
    return ""


# ------------------------------------------------------------------ Linux
def _to_v4l2(name: str, value: int) -> int:
    if name == "auto_exposure":
        return EXPOSURE_AUTO if value else EXPOSURE_MANUAL
    return value


def _from_v4l2(name: str, value: int) -> int:
    if name == "auto_exposure":
        return int(value != EXPOSURE_MANUAL)
    return value


class V4L2Device:
    """Controls of a V4L2 device node."""

    backend = "V4L2"

    def __init__(self, node: str, *, ioctl: Callable[[int, int, bytes], Any] | None = None,
                 opener: Callable[[str], int] | None = None, closer: Callable[[int], None] = os.close):
        if ioctl is None:
            import fcntl

            ioctl = fcntl.ioctl
        self._ioctl = ioctl
        self._closer = closer
        opener = opener or (lambda path: os.open(path, os.O_RDWR | os.O_NONBLOCK))
        try:
            self.fd = opener(node)
        except OSError as exc:
            raise ControlError(exc.strerror or str(exc)) from None

    def set(self, name: str, value: int) -> None:
        cid = _V4L2_CID[name]
        try:
            self._ioctl(self.fd, VIDIOC_S_CTRL, struct.pack("=Ii", cid, _to_v4l2(name, value)))
        except OSError as exc:
            absent = exc.errno == errno.EINVAL and self._query(cid) is None
            raise ControlError(exc.strerror or str(exc), missing=absent) from None

    def get(self, name: str) -> int:
        try:
            buf = self._ioctl(self.fd, VIDIOC_G_CTRL, struct.pack("=Ii", _V4L2_CID[name], 0))
            return _from_v4l2(name, struct.unpack("=Ii", bytes(buf))[1])
        except OSError as exc:
            raise ControlError(exc.strerror or str(exc)) from None
        except (TypeError, struct.error):
            raise ControlError("brak odczytu") from None

    def range(self, name: str) -> Range | None:
        q = self._query(_V4L2_CID[name])
        return None if q is None else Range(q[0], q[1], max(1, q[2]))

    def _query(self, cid: int) -> tuple[int, int, int] | None:
        """(minimum, maximum, step) of a control; None when the device has no such control, or it is disabled."""
        try:
            buf = self._ioctl(self.fd, VIDIOC_QUERYCTRL, _QUERYCTRL.pack(cid, 0, b"", 0, 0, 0, 0, 0, 0, 0))
        except OSError:
            return None
        _, _, _, lo, hi, step, _, flags, _, _ = _QUERYCTRL.unpack(bytes(buf))
        return None if flags & _CTRL_FLAG_DISABLED else (lo, hi, step)

    def close(self) -> None:
        self._closer(self.fd)


# ------------------------------------------------------------------ Windows
class DirectShowDevice:
    """Focus through OpenCV's DirectShow properties on an open capture; the stream owns the capture."""

    backend = "DirectShow"

    def __init__(self, cap: Any):
        import cv2

        self.cap = cap
        self._props = {"autofocus": cv2.CAP_PROP_AUTOFOCUS, "focus_absolute": cv2.CAP_PROP_FOCUS}

    def set(self, name: str, value: int) -> None:
        prop = self._props.get(name)
        if prop is None:
            raise ControlError("na Windows tylko ostrosc")
        if not self.cap.set(prop, float(value)):
            raise ControlError("kamera odrzucila (DirectShow)")

    def get(self, name: str) -> int:
        value = float(self.cap.get(self._props[name]))
        if value < 0:
            raise ControlError("brak odczytu (DirectShow)")
        # DirectShow reports the focus mode as flags: 1 = auto, 2 = manual.
        return int(value) & 1 if name == "autofocus" else round(value)

    def range(self, name: str) -> Range | None:
        return None

    def close(self) -> None:
        pass
