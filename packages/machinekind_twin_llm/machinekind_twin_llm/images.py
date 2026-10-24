"""The images a question sends, and the still check that spares a question.

`encode` writes one JPEG per camera, scaled down from the camera's frame. `thumbnail`, `changed_pixels`
and `unchanged` tell whether a camera's image changed since the frames the last question was asked
on, so the detector can repeat that answer instead of asking again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["CHANGED_LEVELS", "JPEG_QUALITY", "STILL_MAX_CHANGED_PX", "THUMB_LONG_PX", "SentImage",
           "changed_pixels", "encode", "thumbnail", "unchanged"]

#: JPEG quality of the images sent.
JPEG_QUALITY = 90
#: Long edge of the grey thumbnails the still check compares [px]. At 256 px a 60 degree lens
#: gives 4.5 mm per thumbnail pixel at 1 m, and a 3 cm cube covers about 45 pixels.
THUMB_LONG_PX = 256
#: A thumbnail pixel counts as changed when its grey level moved by this much or more, with the
#: image's mean shift (auto exposure) taken out [grey levels, 0..255]. Area downscaling and the
#: 3x3 blur leave sensor noise at about one level, so 16 is far above it.
CHANGED_LEVELS = 16.0
#: Changed thumbnail pixels a camera may have while its image still counts as unchanged. A
#: blinking LED or a flickering pixel stays under it; a 3 cm cube at 1 m moving or leaving is over.
STILL_MAX_CHANGED_PX = 15


@dataclass(frozen=True)
class SentImage:
    """One camera's image as the model gets it."""

    #: The camera's name in the twin. Never part of the file name.
    camera: str
    #: The JPEG in the call directory, `img-<k>.jpg`.
    path: Path
    #: (width, height) of the JPEG [px].
    sent_size: tuple[int, int]
    #: (width, height) of the camera's frame the JPEG was scaled from [px].
    frame_size: tuple[int, int]


def _rgb(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3 or img.shape[0] < 1 or img.shape[1] < 1:
        raise ValueError(f"expected an RGB image of shape (height, width, 3), got {img.shape}")
    return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)


def encode(call: Path, frames: Sequence[tuple[str, np.ndarray]], max_side_px: int) -> list[SentImage]:
    """Write each (camera, RGB frame) as `call/img-<k>.jpg`, k from 1, in the given order.

    A frame whose long edge exceeds `max_side_px` is scaled down with INTER_AREA, keeping its
    aspect ratio. A smaller one is sent at its own size. File names never contain camera names.
    """
    if max_side_px < 1:
        raise ValueError(f"max_side_px must be at least 1, not {max_side_px}")
    sent = []
    for i, (camera, image) in enumerate(frames):
        img = _rgb(image)
        h, w = img.shape[:2]
        scale = min(1.0, max_side_px / max(w, h))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if scale < 1.0:
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        path = call / f"img-{i + 1}.jpg"
        if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise OSError(f"could not write {path}")
        sent.append(SentImage(camera, path, (int(bgr.shape[1]), int(bgr.shape[0])), (int(w), int(h))))
    return sent


def thumbnail(image: np.ndarray) -> np.ndarray:
    """A small grey copy of an RGB frame for `changed_pixels`: long edge `THUMB_LONG_PX` (a smaller
    frame keeps its size), blurred 3x3, float32."""
    grey = cv2.cvtColor(_rgb(image), cv2.COLOR_RGB2GRAY)
    h, w = grey.shape
    scale = THUMB_LONG_PX / max(w, h)
    if scale < 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(grey.astype(np.float32), (3, 3), 0)


def changed_pixels(before: np.ndarray, now: np.ndarray) -> int:
    """How many pixels of one camera's thumbnail moved by `CHANGED_LEVELS` or more, after the two
    images' mean brightness is matched. Thumbnails of different shapes count as changed throughout."""
    if before.shape != now.shape:
        return int(now.size)
    diff = np.abs((now - float(now.mean())) - (before - float(before.mean())))
    return int(np.count_nonzero(diff >= CHANGED_LEVELS))


def unchanged(before: Mapping[str, np.ndarray], now: Mapping[str, np.ndarray],
              max_changed_px: int = STILL_MAX_CHANGED_PX) -> bool:
    """True when both hold thumbnails of the same cameras and no camera has more than
    `max_changed_px` changed pixels."""
    if not before or set(before) != set(now):
        return False
    return all(changed_pixels(before[name], now[name]) <= max_changed_px for name in now)
