"""The wrist camera as the grasp policy sees it: its 160x120 frame and the believed pixel.

The policy was trained on MuJoCo renders of an ideal pinhole camera on `gripper_link`
(f = FOCAL_PX, principal point at the image centre, upright). The real camera delivers a
1920x1080 frame through a wide, strongly distorting lens, mounted upside down. `WristCamera`
turns that frame into the training geometry with one remap through a 640x480 intermediate,
with the 180 degree rotation folded into the map, then an area resize to 160x120. The
model card's centre crop is not used: it is 5 to 27 px off at 160x120.

`project` is training's `believed_uv`: a base-frame point through the nominal camera pose
(`T_GRIPPER_CAMERA`) into the 160x120 image.

Frame sources:

* `LiveWristSource` opens the one wrist camera through the twin's camera layer, on a private
  `CameraRecord` that never joins `ws.cameras`. It is the only place in `grasp/` that opens a
  device, and only through its injected `opener`.
* `SimWristSource` renders the hand camera of the twin's scene (`Scene.render_hand`).

`Frame.t` is the stamp the source gives. For a live camera that is taken after read and
decode, so it is later than the exposure; the driver pairs the frame with the joints at
`t - camera latency`. This module reports `t` as it is and corrects nothing.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np

from .. import cameras
from ..kinematics import inverse
from ..workspace import CameraRecord
from .contract import (
    CX,
    CY,
    DOT_MAX_PX,
    FOCAL_PX,
    IMG_H,
    IMG_W,
    NEAR_Z,
    RAW_H,
    RAW_W,
    T_GRIPPER_CAMERA,
    WRIST_DIST_2026_09_26,
    WRIST_K_2026_09_26,
)

if TYPE_CHECKING:
    from ..kinematics import RobotKinematics
    from ..runtime import Twin
    from ..scene import HandCamera, HandCameraJitter, Scene
    from .rig import Rig

logger = logging.getLogger(__name__)

#: The body `T_GRIPPER_CAMERA` is measured in.
HAND_BODY = "gripper_link"
#: The remap goes to four times the policy size, then an area resize brings it down. Remapping
#: straight to 160x120 would take one sample per roughly 7x7 raw pixels and alias.
MID_SCALE = 4
MID_W, MID_H = MID_SCALE * IMG_W, MID_SCALE * IMG_H
#: Share of the policy frame that may lie past the sensor's edge before the intrinsics are refused.
MAX_OUTSIDE = 0.005
#: Frame rate asked of the wrist camera.
LIVE_FPS = 30


@dataclass(frozen=True)
class WristIntrinsics:
    K: np.ndarray                      # (3, 3), refers to the UPRIGHT 1920x1080 image
    dist: np.ndarray                   # (5,) k1 k2 p1 p2 k3, same frame
    width: int = RAW_W
    height: int = RAW_H
    upside_down: bool = True           # the device delivers the frame rotated by 180 degrees


@dataclass(frozen=True)
class Frame:
    image: np.ndarray                  # (120, 160, 3) uint8 RGB, policy frame
    #: Stamp of the raw frame. A live camera's hub stamps it AFTER read and decode, so it is
    #: later than the exposure.
    t: float
    seq: int                           # increases when t changes


class FrameSource(Protocol):
    def latest(self) -> Frame | None:
        """Newest frame, None before the first one; raises CameraError when the stream died."""
        ...

    def close(self) -> None: ...


class CameraError(RuntimeError):
    pass


def project(T_hand2base: np.ndarray, p_base: np.ndarray, *, T_cam2hand: np.ndarray = T_GRIPPER_CAMERA,
            f: float = FOCAL_PX, cx: float = CX, cy: float = CY) -> np.ndarray:
    """Pixel (u, v) of a base-frame point, float32; NaN pair when the point is not in front of the camera."""
    T_cam2base = np.asarray(T_hand2base, float) @ np.asarray(T_cam2hand, float)
    x, y, z = (inverse(T_cam2base) @ np.append(np.asarray(p_base, float), 1.0))[:3]
    if not z > NEAR_Z:
        return np.array([np.nan, np.nan], np.float32)
    return np.array([f * x / z + cx, f * y / z + cy], np.float32)


def _default_intrinsics() -> WristIntrinsics:
    return WristIntrinsics(WRIST_K_2026_09_26.copy(), WRIST_DIST_2026_09_26.copy())


def wrist_intrinsics(rig: Rig) -> WristIntrinsics:
    K, dist = rig.intrinsics()
    return WristIntrinsics(np.asarray(K, float), np.asarray(dist, float).ravel(),
                           upside_down=rig.upside_down())


class WristCamera:
    """The wrist camera's geometry: raw frame to policy frame, and base-frame points to pixels.

    Raises `CameraError` when the intrinsics would show more than `MAX_OUTSIDE` of the policy
    frame from outside the sensor (less is logged as a warning), or show it through a lens model
    that folds.
    """

    def __init__(self, intr: WristIntrinsics | None = None, *,
                 T_cam2hand: np.ndarray = T_GRIPPER_CAMERA) -> None:
        self.intr = intr if intr is not None else _default_intrinsics()
        self.T_cam2hand = np.asarray(T_cam2hand, float)
        self._m1, self._m2 = _raw_maps(self.intr)

    def policy_frame(self, raw_rgb: np.ndarray) -> np.ndarray:
        """Raw (1080, 1920, 3) uint8 RGB as the device delivers it -> (120, 160, 3) uint8 RGB."""
        shape = (self.intr.height, self.intr.width, 3)
        if raw_rgb.shape != shape or raw_rgb.dtype != np.uint8:
            # The hub does not check the delivered size: a driver that ignores the request
            # delivers another resolution and nothing else complains.
            raise CameraError(f"wrist frame is {raw_rgb.shape} {raw_rgb.dtype}, expected {shape} uint8")
        mid = cv2.remap(raw_rgb, self._m1, self._m2, cv2.INTER_LINEAR)
        return cv2.resize(mid, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

    def believed_uv(self, kin: RobotKinematics, joints: Mapping[str, float],
                    believed: np.ndarray) -> np.ndarray:
        return project(_hand_pose(kin, joints), believed, T_cam2hand=self.T_cam2hand)

    def cube_top_uv(self, kin: RobotKinematics, joints: Mapping[str, float], centre_xy: np.ndarray,
                    top_z: float) -> np.ndarray:
        p = np.array([float(centre_xy[0]), float(centre_xy[1]), float(top_z)])
        return project(_hand_pose(kin, joints), p, T_cam2hand=self.T_cam2hand)


def _hand_pose(kin: RobotKinematics, joints: Mapping[str, float]) -> np.ndarray:
    # `to_q` fills a missing joint with its home value, which would give a plausible wrong pixel.
    missing = [n for n in kin.spec.joints if n != kin.spec.gripper and n not in joints]
    if missing:
        raise KeyError(f"joints missing for the wrist camera pose: {', '.join(missing)}")
    return kin.body(HAND_BODY, joints)


def _raw_maps(intr: WristIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    f = MID_SCALE * FOCAL_PX
    K_mid = np.array([[f, 0.0, MID_SCALE * (CX + 0.5) - 0.5],
                      [0.0, f, MID_SCALE * (CY + 0.5) - 0.5],
                      [0.0, 0.0, 1.0]])
    K = np.asarray(intr.K, float)
    dist = np.asarray(intr.dist, float).ravel()
    mx, my = cv2.initUndistortRectifyMap(K, dist, None, K_mid, (MID_W, MID_H), cv2.CV_32FC1)
    _check_map(mx, my, intr)
    if intr.upside_down:
        # K and dist describe the upright image; the rotation is folded into the map
        # instead of undistorting the raw frame with unrotated intrinsics.
        mx = (intr.width - 1) - mx
        my = (intr.height - 1) - my
    return cv2.convertMaps(mx.astype(np.float32), my.astype(np.float32), cv2.CV_16SC2)


def _check_map(mx: np.ndarray, my: np.ndarray, intr: WristIntrinsics) -> None:
    # A K and dist from the rig file are not checked anywhere else, and cv2.remap complains about
    # neither fault: a view past the sensor turns into a black band the policy never saw, and a
    # lens model that folds (a poor fit with a large k2 or k3) samples the wrong raw pixels.
    if not (np.isfinite(mx).all() and np.isfinite(my).all()):
        raise CameraError("wrist camera K and dist give no pixel for part of the policy frame")
    edges = {"left": mx < 0, "right": mx > intr.width - 1, "top": my < 0, "bottom": my > intr.height - 1}
    outside = edges["left"] | edges["right"] | edges["top"] | edges["bottom"]
    if outside.any():
        n = int(outside.sum())
        where = ", ".join(k for k, m in edges.items() if m.any())
        msg = (f"wrist camera K and dist: {n} of {MID_W}x{MID_H} px ({100.0 * outside.mean():.2f} %) "
               f"of the policy frame lie outside the {intr.width}x{intr.height} sensor "
               f"({where} edge of the upright image)")
        # A recalibration can shift the view by a few pixels past one edge; a thin black strip
        # harms the policy less than a run that cannot start.
        if outside.mean() > MAX_OUTSIDE:
            raise CameraError(msg)
        logger.warning("%s; they stay black", msg)
    if not ((np.diff(mx, axis=1) > 0).all() and (np.diff(my, axis=0) > 0).all()):
        raise CameraError("wrist camera K and dist: the lens model folds inside the policy frame")


class LiveWristSource:
    """FrameSource over the twin's camera layer, for the one wrist camera."""

    def __init__(self, source: str, camera: WristCamera, *,
                 opener: Callable[[CameraRecord], Any] = cameras.open_live) -> None:
        self.camera = camera
        self.record = CameraRecord(name="wrist", source=str(source), width=RAW_W, height=RAW_H, fps=LIVE_FPS)
        try:
            self._live = opener(self.record)
        except Exception as exc:
            raise CameraError(f"wrist camera {source!r} did not open: {exc}") from exc
        self._frame: Frame | None = None
        self._closed = False
        logger.info("wrist camera %s opened, %dx%d at %d fps requested", source, RAW_W, RAW_H, LIVE_FPS)

    def latest(self) -> Frame | None:
        if self._closed:
            raise CameraError("wrist camera closed")
        try:
            rgb, t, err = self._live.rgb_t()
        except Exception as exc:
            # A camera backend that raises inside its read is a dead stream to the grasp driver,
            # which handles only CameraError.
            raise CameraError(f"wrist camera read failed: {exc}") from exc
        if err:
            raise CameraError(f"wrist camera: {err}")
        if rgb is None:
            return self._frame
        t = float(t)
        if self._frame is not None and t == self._frame.t:
            return self._frame
        seq = 1 if self._frame is None else self._frame.seq + 1
        self._frame = Frame(self.camera.policy_frame(rgb), t, seq)
        return self._frame

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._live.close()


class SimWristSource:
    """FrameSource rendering the hand camera of the twin's scene.

    The stamp is taken BEFORE the render, so in sim it is the exposure time and the camera
    latency is 0. With `connect(threaded=False)` pass the twin's own clock.
    """

    def __init__(self, twin: Twin, *, name: str = "wrist", jitter: HandCameraJitter | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.twin = twin
        self.name = name
        self.jitter = jitter
        self.clock = clock
        self._frame: Frame | None = None
        with twin.lock:
            scene = twin.scene
        cam = self._hand_camera(scene)
        if (cam.width, cam.height) != (IMG_W, IMG_H):
            raise CameraError(f"hand camera {name!r} renders {cam.width}x{cam.height}, "
                              f"the policy needs {IMG_W}x{IMG_H}")

    def _hand_camera(self, scene: Scene) -> HandCamera:
        try:
            return scene.hand_camera(self.name)
        except KeyError:
            raise CameraError(f"scene has no hand camera {self.name!r}") from None

    def _render(self, scene: Scene) -> np.ndarray:
        self._hand_camera(scene)
        return scene.render_hand(self.name, self.jitter)

    def latest(self) -> Frame | None:
        t = float(self.clock())
        if self._frame is not None and t == self._frame.t:
            return self._frame
        try:
            image = self.twin.render_with(self._render)
        except CameraError:
            raise
        except RuntimeError as exc:
            raise CameraError(f"sim wrist render failed: {exc}") from exc
        seq = 1 if self._frame is None else self._frame.seq + 1
        self._frame = Frame(image, t, seq)
        return self._frame

    def close(self) -> None:
        pass


_DOT = (0, 255, 0)
_EXTRA = (255, 0, 255)
_TEXT = (255, 255, 255)


def draw_dump(image: np.ndarray, uv: np.ndarray, *, scale: int = 4, radius_px: float = DOT_MAX_PX,
              extra: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    """The policy frame upscaled by `scale` (nearest), with the believed pixel and its dot limit drawn.

    A cross marks `uv`, a circle of `radius_px` (policy pixels) around it shows the dot check's
    limit. `extra` points (for example the projected cube top) are small labelled squares.
    """
    img = np.asarray(image, np.uint8)
    out = np.ascontiguousarray(np.repeat(np.repeat(img, scale, axis=0), scale, axis=1))
    h, w = img.shape[:2]
    far = 8 * scale * max(h, w)                        # keeps a pixel near the camera plane drawable
    notes: list[str] = []

    def at(p: np.ndarray) -> tuple[int, int] | None:
        u, v = float(p[0]), float(p[1])
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        x, y = round((u + 0.5) * scale - 0.5), round((v + 0.5) * scale - 0.5)
        return min(max(x, -far), far), min(max(y, -far), far)

    c = at(uv)
    if c is None:
        notes.append("no pixel")
    else:
        if not (-0.5 <= uv[0] <= w - 0.5 and -0.5 <= uv[1] <= h - 0.5):
            notes.append(f"off frame at ({float(uv[0]):.1f}, {float(uv[1]):.1f})")
        arm = 3 * scale
        cv2.line(out, (c[0] - arm, c[1]), (c[0] + arm, c[1]), _DOT, 1, cv2.LINE_AA)
        cv2.line(out, (c[0], c[1] - arm), (c[0], c[1] + arm), _DOT, 1, cv2.LINE_AA)
        cv2.circle(out, c, round(radius_px * scale), _DOT, 1, cv2.LINE_AA)

    for name, p in (extra or {}).items():
        e = at(p)
        if e is None:
            notes.append(f"{name}: no pixel")
            continue
        cv2.rectangle(out, (e[0] - scale, e[1] - scale), (e[0] + scale, e[1] + scale), _EXTRA, 1, cv2.LINE_AA)
        cv2.putText(out, name, (e[0] + 2 * scale, e[1] - 2 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _EXTRA, 1,
                    cv2.LINE_AA)

    for k, text in enumerate(notes):
        cv2.putText(out, text, (6, 16 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TEXT, 1, cv2.LINE_AA)
    return out
