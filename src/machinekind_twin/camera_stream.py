"""Camera capture on a separate thread.

The key detail: for real-time control we do not want to queue frames. The thread reads the
camera as fast as it can and keeps *only the latest* frame, so the latency does not grow
when detection is slower than the camera.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import camera_controls
from .config import CameraConfig


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    index: int


def index_backend(platform: str | None = None) -> int:
    """The OpenCV backend for a camera index, so the index means what `camera_controls` takes it for.

    DirectShow on Windows (it also opens a camera in a fraction of a second, where the default MSMF
    takes seconds), AVFoundation on macOS, any (V4L2) elsewhere.
    """
    platform = platform or sys.platform
    if platform.startswith("win"):
        return cv2.CAP_DSHOW
    if platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def _is_device_path(source: int | str) -> bool:
    """A video device path in /dev (V4L2), e.g. /dev/video3 or /dev/v4l/by-id/usb-...-video-index0."""
    return isinstance(source, str) and source.startswith("/dev/")


def _is_video_file(source: int | str) -> bool:
    return isinstance(source, str) and not source.isdigit() and not _is_device_path(source)


class CameraStream:
    """Non-blocking frame source (USB camera, a /dev/... device path or a video file)."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._stop = threading.Event()
        self._index = 0
        self._is_file = _is_video_file(cfg.source)
        self._error: str | None = None
        #: What the last open did with `cfg.controls`.
        self.controls_applied = camera_controls.Applied()

    # ---------------------------------------------------------------- setup
    def open(self) -> "CameraStream":
        source: int | str = self.cfg.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            self._is_file = False

        if _is_device_path(source):
            # A device path (/dev/videoN or the stable /dev/v4l/by-id/...) is a camera, not a
            # file. The V4L2 index shifts every time something is plugged into USB; by-id does not.
            cap = cv2.VideoCapture(str(source), cv2.CAP_V4L2)
        elif isinstance(source, int):
            cap = cv2.VideoCapture(source, index_backend())
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(
                f"Nie udalo sie otworzyc zrodla obrazu: {self.cfg.source!r}. "
                "Sprawdz zrodlo kamery (zakladka Kamery) albo sciezke do pliku wideo."
            )

        if not self._is_file:
            if self.cfg.fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            # A small buffer means fresher frames (not every backend supports it).
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:  # pragma: no cover - backend dependent
                pass
            # After the format: some cameras reset their controls when it changes.
            self.controls_applied = camera_controls.apply(self.cfg.source, self.cfg.controls, cap=cap)

        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self

    def apply_controls(self, controls: camera_controls.CameraControls | None) -> camera_controls.Applied:
        """Sets `controls` on the open camera now (Windows needs its open capture for that)."""
        return camera_controls.apply(self.cfg.source, controls, cap=self._cap)

    def focus_range(self) -> camera_controls.Range | None:
        return camera_controls.focus_range(self.cfg.source, cap=self._cap)

    def __enter__(self) -> "CameraStream":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----------------------------------------------------------------- loop
    def _run(self) -> None:
        assert self._cap is not None
        # A video file plays at about its own FPS, so a demo does not run at disk speed.
        file_fps = self._cap.get(cv2.CAP_PROP_FPS) if self._is_file else 0.0
        frame_period = 1.0 / file_fps if self._is_file and file_fps > 1e-3 else 0.0

        while not self._stop.is_set():
            loop_start = time.perf_counter()
            ok, image = self._cap.read()
            if not ok:
                if self._is_file and self.cfg.loop_video:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._error = "Koniec strumienia obrazu."
                break

            if self.cfg.mirror:
                image = cv2.flip(image, 1)

            self._index += 1
            frame = Frame(image=image, timestamp=time.monotonic(), index=self._index)
            with self._lock:
                self._latest = frame

            if frame_period:
                sleep_for = frame_period - (time.perf_counter() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)

    # ----------------------------------------------------------------- read
    def read(self) -> Frame | None:
        """Returns the latest frame (or None if there is none yet)."""
        with self._lock:
            return self._latest

    def wait_for_frame(self, timeout: float = 5.0) -> Frame:
        """Waits for the first frame; raises on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
            if self._error:
                raise RuntimeError(self._error)
            time.sleep(0.01)
        raise TimeoutError(
            f"Kamera {self.cfg.source!r} nie dostarczyla klatki w ciagu {timeout:.0f} s."
        )

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
