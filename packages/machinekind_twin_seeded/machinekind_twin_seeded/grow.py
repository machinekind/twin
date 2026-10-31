"""Growing a block from a pixel: its colour's mask, the mask's blobs, and the blob nearest the pixel.

Pure functions on images, no threads and no state; the detector keeps the tracks. Pixels follow
OpenCV: (0, 0) is the centre of the top-left pixel, u grows right, v grows down.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Blob", "blobs", "box_distance", "colour_mask", "nearest"]

#: The opening that removes specks and one-pixel bridges, as the example cube detector does.
_OPEN = np.ones((3, 3), np.uint8)


def colour_mask(hsv: np.ndarray, ranges: Sequence[tuple[tuple[int, int, int], tuple[int, int, int]]]
                ) -> np.ndarray:
    """The pixels of an HSV image (OpenCV's, H 0..180) in any of `ranges`: uint8 (height, width),
    255 on the colour, opened 3x3."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= np.asarray(cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)), np.uint8)
    return np.asarray(cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN), np.uint8)


@dataclass(frozen=True)
class Blob:
    """One connected patch of a mask (8-connectivity)."""

    #: Its number in the label image `blobs` returns; `labels == index` is its mask.
    index: int
    #: The mean of its pixels (u, v) [px].
    centroid: tuple[float, float]
    #: (u0, v0, u1, v1) [px], the first and the last pixel column and row, both included.
    box: tuple[int, int, int, int]
    #: Its pixels [px].
    area: int

    @property
    def side(self) -> int:
        """Its box's long side [px]."""
        return max(self.box[2] - self.box[0] + 1, self.box[3] - self.box[1] + 1)


def blobs(mask: np.ndarray, min_area_px: int) -> tuple[np.ndarray, list[Blob]]:
    """The label image of `mask` (uint8, non-zero on) and its blobs of at least `min_area_px` pixels.

    The background, component 0, is never a blob.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found: list[Blob] = []
    for i in range(1, int(n)):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        centroid = (float(centroids[i, 0]), float(centroids[i, 1]))
        found.append(Blob(i, centroid, (x, y, x + w - 1, y + h - 1), area))
    return labels, found


def box_distance(point: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    """How far `point` lies from the nearest point of `box` (u0, v0, u1, v1) [px]; 0 inside."""
    u, v = point
    du = max(box[0] - u, 0.0, u - box[2])
    dv = max(box[1] - v, 0.0, v - box[3])
    return math.hypot(du, dv)


def nearest(found: Sequence[Blob], anchor: tuple[float, float], radius_px: float,
            max_side_px: float | None = None) -> Blob | None:
    """The blob whose box lies nearest `anchor`, within `radius_px`; a tie goes to the larger blob.

    With `max_side_px`, a blob whose box's long side is longer is left out (two touching blocks
    of one colour). None when no blob qualifies.
    """
    best: tuple[float, int, Blob] | None = None
    for blob in found:
        if max_side_px is not None and blob.side > max_side_px:
            continue
        d = box_distance(anchor, blob.box)
        if d > radius_px:
            continue
        if best is None or (d, -blob.area) < (best[0], best[1]):
            best = (d, -blob.area, blob)
    return None if best is None else best[2]
