"""The images a question sends, and the still check that spares a question.

`encode` writes one JPEG per camera, scaled down from the camera's frame. `thumbnail`,
`largest_change_px` and `unchanged` tell whether a camera's image changed since other frames, so
the detector can repeat an answer instead of asking again.

The check compares grey thumbnails with their mean brightness matched, so auto exposure does not
count. A pixel's change must also exceed a share of its contrast to the mean, so a gain change does
not count on a high-contrast scene. What decides is the largest patch of changed pixels, not their
total: an object moving makes one patch, while noise, banding and a flickering camera scatter
single pixels. The arm moving is no change either. `arm_thumbnail` shrinks the twin's arm mask to
the thumbnail and widens it over the arm's edge and shadow. `union_masks` joins the masks of both
frames compared, so neither the place the arm left nor the one it entered counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["ARM_DILATE_PX", "BLOB_JOIN_PX", "CHANGED_LEVELS", "GAIN_TOLERANCE", "JPEG_QUALITY",
           "STILL_MAX_BLOB_PX", "THUMB_LONG_PX", "SentImage", "arm_thumbnail", "encode", "largest_change_px",
           "thumbnail", "unchanged", "union_masks"]

#: JPEG quality of the images sent.
JPEG_QUALITY = 90
#: Long edge of the grey thumbnails the still check compares [px]. At 256 px a 60 degree lens
#: gives 4.5 mm per thumbnail pixel at 1 m, and a 3 cm cube covers about 45 pixels.
THUMB_LONG_PX = 256
#: A thumbnail pixel counts as changed when its grey level moved by this much or more, with the
#: image's mean shift (auto exposure) taken out [grey levels, 0..255]. Area downscaling and the
#: 3x3 blur leave sensor noise at about one level, so 16 is far above it.
CHANGED_LEVELS = 16.0
#: Share of a pixel's distance from the mean that its change may add on top of `CHANGED_LEVELS`,
#: so a gain change does not count. A 1.15 gain on a scene of a 160 table, dark cables and white
#: paper makes blobs of 250 to 2200 thumbnail pixels without it and 0 with it.
GAIN_TOLERANCE = 0.2
#: Changed pixels this close to each other [thumbnail px] belong to one blob, so the leading and
#: trailing edges of a moving object count together.
BLOB_JOIN_PX = 2
#: Largest blob of changed thumbnail pixels a camera may have while its image counts as unchanged.
#: Measured at 320 to 1920 px wide frames on a 100..140 table: a 3 cm cube 80 levels darker at
#: 1 m moving 1 cm makes at least 39, leaving at least 50; a blinking 2x2 thumbnail pixel LED makes
#: at most 13; noise of +-20 levels per pixel, a +12 step and a 1.15 gain at most 5.
#: A cube 55 levels or less from the table can move 1 cm along an axis unseen (14).
STILL_MAX_BLOB_PX = 24
#: The arm mask is widened by this much on the thumbnail [px], over the arm's edge and shadow:
#: about 2.3 cm at 1 m.
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
    """A small grey copy of an RGB frame for `largest_change_px`: long edge `THUMB_LONG_PX` (a
    smaller frame keeps its size), blurred 3x3, float32."""
    grey = cv2.cvtColor(_rgb(image), cv2.COLOR_RGB2GRAY)
    h, w = grey.shape
    size = _thumb_size(h, w)
    if size != (w, h):
        grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(grey.astype(np.float32), (3, 3), 0)


def arm_thumbnail(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """A bool arm mask of a frame shrunk to the thumbnail `shape`, True wherever the arm covers part
    of a thumbnail pixel, then widened by `ARM_DILATE_PX`."""
    h, w = shape[:2]
    small = cv2.resize(np.asarray(mask, bool).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_AREA) > 0
    side = 2 * ARM_DILATE_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    return cv2.dilate(small.astype(np.uint8), kernel) > 0


def union_masks(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per camera, the pixels either mapping masks. A camera in one mapping only keeps its mask;
    masks of different shapes keep `b`'s."""
    union = {**a, **b}
    for name in set(a) & set(b):
        if a[name].shape == b[name].shape:
            union[name] = a[name] | b[name]
    return union


def largest_change_px(before: np.ndarray, now: np.ndarray, ignore: np.ndarray | None = None) -> int:
    """The area of the largest blob of changed pixels between two thumbnails of one camera, pixels
    in `ignore` left out [thumbnail px].

    The mean brightness of the pixels left in is matched first. A pixel counts as changed when it
    moved by `CHANGED_LEVELS` plus `GAIN_TOLERANCE` of its smaller distance from the mean. Changed
    pixels within `BLOB_JOIN_PX` of each other form one blob, and only changed pixels count towards
    its area. Thumbnails of different shapes, and an image with nothing left in, count as changed
    throughout.
    """
    if before.shape != now.shape:
        return int(now.size)
    keep = np.ones(now.shape, bool) if ignore is None else ~ignore
    if not keep.any():
        return int(now.size)
    db = before - float(before[keep].mean())
    dn = now - float(now[keep].mean())
    limit = CHANGED_LEVELS + GAIN_TOLERANCE * np.minimum(np.abs(db), np.abs(dn))
    changed = keep & (np.abs(dn - db) >= limit)
    if not changed.any():
        return 0
    join = np.ones((2 * BLOB_JOIN_PX + 1,) * 2, np.uint8)
    labels = cv2.connectedComponents(cv2.dilate(changed.astype(np.uint8), join), connectivity=8)[1]
    return int(np.bincount(labels[changed].astype(np.intp)).max())


def unchanged(before: Mapping[str, np.ndarray], now: Mapping[str, np.ndarray],
              ignore: Mapping[str, np.ndarray] | None = None, max_blob_px: int = STILL_MAX_BLOB_PX) -> bool:
    """True when both hold thumbnails of the same cameras and no camera's largest blob of changed
    pixels, those in its `ignore` mask left out, exceeds `max_blob_px`."""
    if not before or set(before) != set(now):
        return False
    masks = ignore or {}
    return all(largest_change_px(before[name], now[name], masks.get(name)) <= max_blob_px for name in now)
