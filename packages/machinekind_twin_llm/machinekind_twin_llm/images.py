"""The images a question sends, and the still check that spares a question.

`encode` writes one JPEG per camera, scaled down from the camera's frame. `thumbnail`,
`largest_change_px` and `unchanged` tell whether a camera's image changed since earlier frames.
When no camera's image changed, the detector repeats an answer instead of asking again.

The check compares grey thumbnails with their mean brightness matched. This keeps auto exposure
out. A pixel's change must also exceed a share of its distance from the mean. This keeps a gain
change out on a high-contrast scene. The largest blob of changed pixels decides whether an image
changed. A moving object changes one patch. Noise, banding and a flickering camera change single
pixels or small groups.

The arm moving does not count either. `arm_thumbnail` shrinks the twin's arm mask to the thumbnail
and widens it over the arm's edge and shadow. `hide_arm` marks the pixels under it as not shown. A
pixel hidden now is left out. A pixel hidden at the last question counts as changed once it shows.
The last answer never saw it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["ARM_DILATE_PX", "BLOB_JOIN_PX", "CHANGED_LEVELS", "GAIN_TOLERANCE", "JPEG_QUALITY",
           "STILL_MAX_BLOB_PX", "THUMB_LONG_PX", "SentImage", "arm_thumbnail", "encode", "hide_arm",
           "largest_change_px", "thumbnail", "unchanged"]

#: JPEG quality of the images sent.
JPEG_QUALITY = 90
#: Long edge of the grey thumbnails the still check compares [px]. At 256 px a 60 degree lens
#: gives 4.5 mm per thumbnail pixel at 1 m, and a 3 cm cube covers about 45 pixels.
THUMB_LONG_PX = 256
#: A thumbnail pixel counts as changed when its grey level moved by this much or more, with the
#: image's mean shift (auto exposure) taken out [grey levels, 0..255]. Area downscaling and the
#: 3x3 blur leave sensor noise at about one level, so 16 is far above it.
CHANGED_LEVELS = 16.0
#: Share of a pixel's distance from the mean added to its change threshold, `CHANGED_LEVELS`. A
#: gain change then does not count. On a 160 table with dark cables and white paper, a 1.15 gain
#: makes blobs of 250 to 2200 thumbnail pixels without it. With it, the gain makes none.
GAIN_TOLERANCE = 0.2
#: Changed pixels this close in both row and column form one blob [thumbnail px]. The two edges
#: of a moving object then count together. 5 px is 2.3 cm at 1 m.
BLOB_JOIN_PX = 5
#: Largest blob of changed thumbnail pixels in an image that counts as unchanged. Measured on 320
#: to 1920 px wide frames of a 100..140 table. A 3 cm cube 80 levels darker at 1 m makes at least
#: 39 when it moves 1 cm, and at least 49 when it leaves. A blinking LED of 2x2 thumbnail pixels
#: makes at most 14. Noise of +-20 levels per pixel, a +12 step and a 1.15 gain make none. Every
#: 1 cm move along an axis counts once the cube is 39 to 64 levels from the table. The figure
#: depends on the frame width and on where the cube's edges fall on the thumbnail pixels.
STILL_MAX_BLOB_PX = 24
#: How far the arm mask is widened on the thumbnail [px]. This covers the arm's edge and shadow.
#: 5 px is 2.3 cm at 1 m.
ARM_DILATE_PX = 5


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


def _thumb_size(h: int, w: int) -> tuple[int, int]:
    """(width, height) of the thumbnail of an h x w frame."""
    scale = THUMB_LONG_PX / max(w, h)
    if scale >= 1.0:
        return w, h
    return max(1, round(w * scale)), max(1, round(h * scale))


def thumbnail(image: np.ndarray) -> np.ndarray:
    """A small grey float32 copy of an RGB frame for `largest_change_px`, blurred 3x3. Its long
    edge is `THUMB_LONG_PX`. A smaller frame keeps its size."""
    grey = cv2.cvtColor(_rgb(image), cv2.COLOR_RGB2GRAY)
    h, w = grey.shape
    size = _thumb_size(h, w)
    if size != (w, h):
        grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(grey.astype(np.float32), (3, 3), 0)


def arm_thumbnail(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """A frame's bool arm mask shrunk to the thumbnail `shape`, then widened by `ARM_DILATE_PX`.
    Before widening, a thumbnail pixel is True when the arm covers any part of it."""
    h, w = shape[:2]
    small = cv2.resize(np.asarray(mask, bool).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_AREA) > 0
    side = 2 * ARM_DILATE_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    return cv2.dilate(small.astype(np.uint8), kernel) > 0


def hide_arm(thumb: np.ndarray, arm: np.ndarray) -> np.ndarray:
    """A copy of `thumb` with NaN where the thumbnail-sized arm mask `arm` is True. NaN marks a
    pixel the frame does not show."""
    hidden = thumb.copy()
    hidden[arm] = np.nan
    return hidden


def largest_change_px(before: np.ndarray, now: np.ndarray) -> int:
    """The area of the largest blob of changed pixels between two thumbnails of one camera
    [thumbnail px].

    NaN marks a pixel the frame does not show. A pixel that is NaN in `now` is left out. A pixel
    that is NaN only in `before` counts as changed. The mean brightness is matched over the pixels
    both thumbnails show. A pixel both show counts as changed when it moved by `CHANGED_LEVELS`
    plus `GAIN_TOLERANCE` times its smaller distance from the mean. Changed pixels up to
    `BLOB_JOIN_PX` apart in both row and column form one blob. Only changed pixels count towards a
    blob's area. Thumbnails of different shapes count as changed throughout. So does a `now` with
    no pixel shown.
    """
    if before.shape != now.shape:
        return int(now.size)
    shown = ~np.isnan(now)
    if not shown.any():
        return int(now.size)
    both = shown & ~np.isnan(before)
    changed = shown & ~both
    if both.any():
        db = np.where(both, before - float(before[both].mean()), 0.0)
        dn = np.where(both, now - float(now[both].mean()), 0.0)
        limit = CHANGED_LEVELS + GAIN_TOLERANCE * np.minimum(np.abs(db), np.abs(dn))
        changed |= both & (np.abs(dn - db) >= limit)
    if not changed.any():
        return 0
    join = np.ones((BLOB_JOIN_PX, BLOB_JOIN_PX), np.uint8)
    labels = cv2.connectedComponents(cv2.dilate(changed.astype(np.uint8), join), connectivity=8)[1]
    return int(np.bincount(labels[changed].astype(np.intp)).max())


def unchanged(before: Mapping[str, np.ndarray], now: Mapping[str, np.ndarray],
              max_blob_px: int = STILL_MAX_BLOB_PX) -> bool:
    """True when both hold thumbnails of the same cameras and no camera's largest blob of changed
    pixels exceeds `max_blob_px`."""
    if not before or set(before) != set(now):
        return False
    return all(largest_change_px(before[name], now[name]) <= max_blob_px for name in now)
