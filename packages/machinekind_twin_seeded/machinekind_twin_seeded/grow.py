"""Growing a block from a pixel: its colour's mask, the mask's blobs, the blob nearest the pixel,
and one blob shared among the blocks it holds.

Pure functions on images, no threads and no state; the detector keeps the tracks. Pixels follow
OpenCV: (0, 0) is the centre of the top-left pixel, u grows right, v grows down.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Blob", "assign", "blobs", "box_distance", "colour_mask", "nearest", "part", "split"]

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


def split(mask: np.ndarray, anchors: Sequence[tuple[float, float]], free: Sequence[bool] | None = None,
          iterations: int = 12) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """One blob shared among the blocks it holds (touching or stacked blocks of one colour): each
    pixel goes to the nearest of as many centres, started at `anchors` (u, v) and moved to the mean
    of their pixels until they settle (k-means on the pixels' positions).

    The centres follow the blob's outline rather than its blocks when one moves more than half way
    to the nearest other anchor: a flat block lying on another is then cut along its length. So
    then the anchors stay, and only those marked `free` move, each along the line from the nearest
    anchor that stays. Returns one bool mask per anchor, in order (a mask may be empty), and the
    centres."""
    v, u = np.nonzero(mask)
    pts = np.column_stack([u, v]).astype(float)
    start = np.asarray(anchors, float).reshape(-1, 2)
    n = len(start)
    loose = [False] * n if free is None else [bool(f) for f in free]
    owner, centres = _kmeans(pts, start, [True] * n, [None] * n, iterations)
    if n > 1:
        gaps = np.linalg.norm(start[:, None] - start[None], axis=-1) + np.diag(np.full(n, np.inf))
        reach = gaps.min(1) / 2
        if any(not loose[k] and np.linalg.norm(centres[k] - start[k]) > reach[k] for k in range(n)):
            stay = [k for k in range(n) if not loose[k]]
            along: list[np.ndarray | None] = []
            for k in range(n):
                d = None
                if loose[k] and stay:
                    d = start[k] - start[min(stay, key=lambda j: float(gaps[k, j]))]
                    d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else None
                along.append(d)
            owner, centres = _kmeans(pts, start, loose, along, iterations)
    parts = []
    for k in range(n):
        one = np.zeros(mask.shape, bool)
        one[v[owner == k], u[owner == k]] = True
        parts.append(one)
    return parts, [(float(c[0]), float(c[1])) for c in centres]


def _kmeans(pts: np.ndarray, start: np.ndarray, moving: Sequence[bool], along: Sequence[np.ndarray | None],
            iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """Each point's nearest centre, and the centres: those `moving` moved to the mean of their
    points until they settle, along their `along` direction when it is given; the others stay."""
    centres = start.copy()
    owner = np.zeros(len(pts), int)
    for _ in range(max(iterations, 1)):
        owner = ((pts[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
        moved = centres.copy()
        for k in range(len(centres)):
            if not moving[k] or not (owner == k).any():
                continue
            mean = pts[owner == k].mean(0)
            d = along[k]
            moved[k] = mean if d is None else start[k] + float((mean - start[k]) @ d) * d
        if np.abs(moved - centres).max(initial=0.0) < 0.25:
            break
        centres = moved
    return owner, centres


def part(mask: np.ndarray) -> tuple[tuple[float, float], tuple[int, int, int, int], int] | None:
    """A mask's centroid (u, v), box (u0, v0, u1, v1) and long side [px]; None when it is empty."""
    v, u = np.nonzero(mask)
    if len(u) == 0:
        return None
    box = (int(u.min()), int(v.min()), int(u.max()), int(v.max()))
    return (float(u.mean()), float(v.mean())), box, max(box[2] - box[0] + 1, box[3] - box[1] + 1)


def assign(cost: np.ndarray, gate: float) -> list[tuple[int, int]]:
    """The (row, column) pairs of least total cost, each row and column in one pair at most and
    every pair's cost under `gate`; more pairs win while each costs less than `gate`. Exact over
    the sets of columns taken; meant for the handful of blocks of one label in one camera."""
    cost = np.asarray(cost, float)
    if cost.ndim != 2 or 0 in cost.shape:
        return []
    if cost.shape[1] > cost.shape[0]:
        return sorted((i, j) for j, i in assign(cost.T, gate))
    n, m = cost.shape
    layers: list[dict[int, tuple[float, int, int]]] = []
    prev: dict[int, tuple[float, int, int]] = {0: (0.0, 0, -1)}
    for i in range(n):
        cur: dict[int, tuple[float, int, int]] = {}
        for taken, (value, _, _) in prev.items():
            if taken not in cur or value < cur[taken][0]:
                cur[taken] = (value, taken, -1)
            for j in range(m):
                c = float(cost[i, j])
                if taken >> j & 1 or not c < gate:
                    continue
                now, v = taken | 1 << j, value + c - gate
                if now not in cur or v < cur[now][0]:
                    cur[now] = (v, taken, j)
        layers.append(cur)
        prev = cur
    taken = min(prev, key=lambda k: prev[k][0])
    pairs = []
    for i in range(n - 1, -1, -1):
        _, before, j = layers[i][taken]
        if j >= 0:
            pairs.append((i, j))
        taken = before
    return sorted(pairs)
