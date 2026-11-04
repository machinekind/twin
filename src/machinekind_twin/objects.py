"""Where a detector's pixels are in the robot's base frame: camera models, rays and `locate`.

A detector says what it sees and where IN THE IMAGE (`machinekind_twin_api.Detection`). This
module does the geometry. It builds a pinhole model (with distortion) of every camera the
perception may use, turns a pixel into a ray in the base frame, and places each object one of
three ways:

- "sylwetka": every camera gave a mask and the object's size is known. The box fit
  (`CubeDetector.fit_masks`) gives position and yaw from all masks at once, for a box of that size
  resting on the table or on another block. A block another one partly hides is fitted again with
  the other objects' masks as pixels it may be hidden behind.
- "triangulacja": two or more cameras saw the object and their rays pass close enough to each
  other for the detectors' pixel error. The point nearest to all rays.
- "promien": one camera, or rays that disagree. The surest detection's ray meets the plane at
  the height the detection gives (`height_m`, else half of `size_m`, else the table top).

Several objects may share a label. A camera that reports a label more than once sees that many
objects (one reported twice in one spot counts once, `_distinct`), and the cameras' detections of
the label pair by geometry: those whose rays meet are one object (`_objects`). Each object is
placed from its own detections only, so the masks of two blocks never meet in one fit. With one
detection per camera, the label is one object, as it always was.

A box fit off the table counts only when the block rests on another placed or tracked block
(`_supported`). One camera cannot tell a block on the table from one on top of a stack along the
same ray, so a block one camera places, with no tracked pose yet, is left out while it could rest
on another block: its mask touches another object's (`_touches`), or its ray passes over another
block (`_could_rest_higher`). Two cameras place it.

The geometry only ever uses the CALIBRATED pose, never `sim_pose`, so simulated perception tests
the calibration just as `TableMapper` does.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from machinekind_twin_api.detector import Detection

from .perception import CubeDetector, TableMapper, perception_cameras
from .robots import Desk
from .workspace import Workspace

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_BASE_M",
    "MIN_RANGE_M",
    "MIN_RAY_GAP_M",
    "SUPPORT_MARGIN_M",
    "SUPPORT_TOL_M",
    "Z_RANGE_M",
    "CameraModel",
    "Located",
    "Resting",
    "assign",
    "camera_key",
    "camera_models",
    "locate",
    "mapper_for",
    "pixel_ray",
    "ray_plane",
    "ray_tolerance_m",
    "triangulate",
]

#: A triangulated point outside these heights above the table is not an object on the table [m].
Z_RANGE_M = (-0.03, 0.6)
#: The highest surface the box fit looks for a block resting on [m]: a stack of a few blocks.
MAX_BASE_M = 0.15
#: A block rests on another when its bottom is this close to the other's top, and on the table
#: when its bottom is this close to the table [m].
SUPPORT_TOL_M = 0.006
#: How far past a block's footprint the centre of a block resting on it may lie [m].
SUPPORT_MARGIN_M = 0.005
#: Masks this close in an image touch [px].
TOUCH_PX = 3
#: Detections of one label from two cameras are one object when their rays pass at most this many
#: times their tolerance apart.
PAIR_GATE = 3.0
#: An object of this call is a tracked block of its label when its rays pass this close to it [m].
CLAIM_M = 0.05
#: Two detections of one label in one camera are one object reported twice when their masks overlap
#: this much (IoU), or their boxes do and the boxes' centres lie within `SAME_SHIFT` of the shorter
#: box side; without masks or boxes, when their pixels lie this close [px]. The box of a block
#: resting on another of its size overlaps the lower one's by most of its area, its centre a
#: block's height higher.
SAME_IOU = 0.5
SAME_SHIFT = 0.15
SAME_PX = 2.0
#: `assign` is exact up to this many rows or columns on its smaller side.
ASSIGN_EXACT = 10
#: Rounds of fitting touching blocks again to their masks together, at most.
REFINE_ROUNDS = 2
#: A box fitted again further than this from where it was is not taken [m].
REFINE_M = 0.01
#: A round that moves no box more than this ends the fitting [m].
REFINE_STILL_M = 0.0005
#: Floor under the ray-pairing tolerance [m].
MIN_RAY_GAP_M = 0.01
#: A point closer than this in front of a camera is not something the camera sees [m].
MIN_RANGE_M = 0.05
#: How far a pixel may lie outside its frame and still count as in it [px].
PIXEL_MARGIN_PX = 2.0
#: Rays closer to parallel than this (smallest over largest eigenvalue of the normal matrix)
#: give no point.
_ILL_CONDITIONED = 1e-6


@dataclass(frozen=True, eq=False)
class CameraModel:
    """One fixed camera as the geometry sees it. Compared by identity (it holds arrays)."""

    name: str
    #: 3x3 intrinsics at `size`.
    K: np.ndarray
    #: OpenCV distortion (k1 k2 p1 p2 k3 ...); None = an ideal pinhole (simulation).
    dist: np.ndarray | None
    #: 4x4 camera-to-base transform: the CALIBRATED pose, never `sim_pose`.
    T_cam2base: np.ndarray
    #: (width, height) [px] the camera record asks for; `K` is for this size.
    size: tuple[int, int]


def camera_models(ws: Workspace) -> dict[str, CameraModel]:
    """The cameras the geometry may place detections from, in workspace order.

    `perception_cameras(ws)` (enabled, calibrated, trusted or simulated) minus any camera named
    like a camera on the arm (`RobotSpec.cameras`): such a camera rides on the arm and has no fixed
    pose. A simulated camera may not take such a name at all, so this only ever drops real ones.
    """
    on_arm = ws.spec().cameras
    out = {}
    for c in perception_cameras(ws):
        if c.name in on_arm:
            continue
        K, dist = c.intrinsics()
        out[c.name] = CameraModel(c.name, K, dist, np.asarray(c.T_cam2base, float), (int(c.width), int(c.height)))
    return out


def _frozen(value) -> tuple | None:
    """A nested list or array as nested tuples of floats, so it compares and hashes."""
    if value is None:
        return None
    arr = np.asarray(value, float)
    return tuple(map(float, arr.ravel())) + (arr.shape,)


def camera_key(ws: Workspace) -> tuple:
    """What the camera models and the mapper depend on: equal keys, equal models.

    Per camera, in workspace order: name, source, enabled, trusted, size, `T_cam2base`, K and
    distortion. The robot is in it too: it names the cameras on the arm and the table map.
    """
    cams = []
    for c in ws.cameras:
        K, dist = c.intrinsics()
        cams.append((c.name, c.source, c.enabled, c.trusted, int(c.width), int(c.height),
                     _frozen(c.T_cam2base), _frozen(K), _frozen(dist)))
    return (ws.robot, tuple(cams))


def mapper_for(models: Mapping[str, CameraModel], desk: Desk) -> TableMapper:
    """The table map over exactly these cameras, so the Map tab and the geometry see one camera set."""
    cams = {n: (m.K, m.dist, m.T_cam2base) for n, m in models.items()}
    return TableMapper(cams, desk.map_centre, desk.map_side)


def pixel_ray(cam: CameraModel, u: float, v: float, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The ray through pixel (u, v) as (origin, unit direction) in the base frame.

    `image_size` is the (width, height) of the frame the pixel is in. A frame of another size than
    `cam.size` is taken as a uniform resize of the camera's image: the pixel is scaled to
    `cam.size` first, centre to centre, (u + 0.5) * W / w - 0.5.
    """
    W, H = cam.size
    w, h = image_size
    uu = (float(u) + 0.5) * W / w - 0.5
    vv = (float(v) + 0.5) * H / h - 0.5
    if cam.dist is None:
        x, y, _ = np.linalg.solve(cam.K, np.array([uu, vv, 1.0]))
    else:
        pts = np.array([[[uu, vv]]], np.float64)
        x, y = cv2.undistortPoints(pts, cam.K, cam.dist).reshape(2)
    d_cam = np.array([x, y, 1.0])
    R, o = cam.T_cam2base[:3, :3], cam.T_cam2base[:3, 3]
    d = R @ (d_cam / np.linalg.norm(d_cam))
    return o.copy(), d / np.linalg.norm(d)


def ray_plane(o: np.ndarray, d: np.ndarray, z: float) -> np.ndarray | None:
    """The ray's point on the plane at height z; None when the ray runs along the plane or meets
    it less than `MIN_RANGE_M` in front of the origin (`d` is a unit vector)."""
    if abs(d[2]) < 1e-6:
        return None
    s = (z - o[2]) / d[2]
    if s < MIN_RANGE_M:
        return None
    return o + s * d


def triangulate(rays: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float] | None:
    """The least-squares point nearest to two or more rays and the largest point-to-ray distance.

    None for fewer than two rays or rays too close to parallel to cross anywhere definite.
    """
    if len(rays) < 2:
        return None
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    eig = np.linalg.eigvalsh(A)
    if eig[0] < _ILL_CONDITIONED * eig[-1]:
        return None
    p = np.linalg.solve(A, b)
    gap = max(float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p - o))) for o, d in rays)
    return p, gap


def ray_tolerance_m(cam: CameraModel, depth_m: float, pixel_error_px: float) -> float:
    """How far a ray may miss the object at `depth_m` for a pixel error in the camera's own pixels."""
    return max(MIN_RAY_GAP_M, pixel_error_px * depth_m / float(cam.K[0, 0]))


@dataclass(eq=False)
class Located:
    """One label placed in the base frame from one detector call."""

    label: str
    pos: np.ndarray
    #: 3x3; identity for the point paths, which know no orientation.
    rot: np.ndarray
    confidence: float
    n_cameras: int
    #: The oldest date among the detections used, on the Perception clock, which is time.monotonic
    #: in the panel. A detection's date is its frame's capture time minus its `age_s`. "promien"
    #: uses only the most confident detection.
    t: float
    size_m: tuple[float, float, float] | None
    #: How it was placed, as the panel shows it: "sylwetka", "triangulacja" or "promien".
    method: str
    #: The largest `Detection.age_s` among the detections used [s]. `t` already subtracts each
    #: detection's own age.
    age_s: float = 0.0


@dataclass(frozen=True, eq=False)
class Resting:
    """A block as stacking sees it: its centre and rotation in the base frame and its size [m]."""

    pos: np.ndarray
    rot: np.ndarray
    size_m: tuple[float, float, float]
    #: The detector's label; "" = its key is its label.
    label: str = ""

    @property
    def base(self) -> float:
        """Its bottom's height above the table [m]."""
        return float(self.pos[2]) - self.size_m[2] / 2

    @property
    def top(self) -> float:
        """Its top's height above the table [m]."""
        return float(self.pos[2]) + self.size_m[2] / 2

    def under(self, p: np.ndarray, margin: float = SUPPORT_MARGIN_M) -> bool:
        """Whether its footprint, widened by `margin`, lies under the point `p`."""
        local = np.asarray(self.rot, float).T @ (np.asarray(p, float) - np.asarray(self.pos, float))
        return abs(local[0]) <= self.size_m[0] / 2 + margin and abs(local[1]) <= self.size_m[1] / 2 + margin

    def holds(self, other: Resting) -> bool:
        """Whether `other` rests on this block."""
        return abs(other.base - self.top) <= SUPPORT_TOL_M and self.under(other.pos)


def _resting(loc: Located) -> Resting | None:
    return None if loc.size_m is None else Resting(np.asarray(loc.pos, float), np.asarray(loc.rot, float),
                                                   loc.size_m)


# ------------------------------------------------------------------ validation
_warned: set[str] = set()


def _finite(*xs: float) -> bool:
    return all(isinstance(x, (int, float, np.integer, np.floating)) and math.isfinite(float(x)) for x in xs)


def _problem(det: Detection, cameras: Mapping[str, CameraModel],
             frames: Mapping[str, tuple[tuple[int, int], float]]) -> str:
    """Why this detection cannot be used ("" = it can)."""
    if not isinstance(det.label, str) or not det.label.strip():
        return "empty label"
    if det.camera not in cameras:
        return "camera not among the perception cameras"
    if det.camera not in frames:
        return "camera without a frame in this call"
    (w, h), _ = frames[det.camera]
    try:
        u, v = det.pixel
    except (TypeError, ValueError):
        return "pixel is not (u, v)"
    if not _finite(u, v):
        return "pixel not finite"
    m = PIXEL_MARGIN_PX
    if not (-m <= u <= w - 1 + m and -m <= v <= h - 1 + m):
        return "pixel outside the frame"
    if not _finite(det.confidence):
        return "confidence not finite"
    if det.mask is not None:
        mask = det.mask
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != (h, w):
            return "mask is not a bool image of the frame's shape"
    if det.size_m is not None:
        try:
            ok = len(det.size_m) == 3 and _finite(*det.size_m) and all(s > 0 for s in det.size_m)
        except TypeError:
            ok = False
        if not ok:
            return "size_m is not three positive numbers"
    if det.height_m is not None and not _finite(det.height_m):
        return "height_m not finite"
    if det.pixel_error_px is not None and not (_finite(det.pixel_error_px) and det.pixel_error_px >= 0):
        return "pixel_error_px not a finite non-negative number"
    if not (_finite(det.age_s) and det.age_s >= 0):
        return "age_s not a finite non-negative number"
    return ""


def _drop(reason: str, det: Detection) -> None:
    if reason in _warned:
        logger.debug("detection dropped (%s): %r on %r", reason, det.label, det.camera)
        return
    _warned.add(reason)
    logger.warning("detection dropped (%s): %r on %r; further drops for this reason are logged at debug",
                   reason, det.label, det.camera)


_warned_size: set[str] = set()


def _full_size(dets: list[Detection], cameras: Mapping[str, CameraModel],
               frames: Mapping[str, tuple[tuple[int, int], float]]) -> bool:
    """True when every camera's frame has its record's size. Logs a camera that differs once, as
    the object then loses its yaw to triangulation (a webcam negotiating another resolution)."""
    full = True
    for d in dets:
        got, want = frames[d.camera][0], cameras[d.camera].size
        if got == want:
            continue
        full = False
        if d.camera not in _warned_size:
            _warned_size.add(d.camera)
            logger.warning("camera %r delivers %dx%d frames, its record says %dx%d: no silhouette fit "
                           "(no yaw) from it until the sizes match", d.camera, *got, *want)
    return full


# ------------------------------------------------------------------ locate
def _height(det: Detection) -> float:
    """The height above the table of the point the pixel marks (the contract's `height_m` rule)."""
    if det.height_m is not None:
        return float(det.height_m)
    if det.size_m is not None:
        return float(det.size_m[2]) / 2
    return 0.0


def locate(dets: Sequence[Detection], cameras: Mapping[str, CameraModel],
           frames: Mapping[str, tuple[tuple[int, int], float]], *, pixel_error_px: float,
           mapper: TableMapper | None,
           occluders: Callable[[dict[str, float]], dict[str, np.ndarray] | None] | None,
           box: Callable[[tuple[float, float, float], float], CubeDetector],
           tracked: Mapping[str, Resting] | None = None) -> tuple[list[Located], int]:
    """Places one detector call's detections in the base frame.

    `cameras` are the models the call may use; `frames[camera]` is ((width, height), capture time)
    of the frame the detector read. `pixel_error_px` is the detector's own, used for a detection
    that gives none. `mapper` (over `cameras`) enables the silhouette fit, `occluders` gives the arm
    masks for the capture times of the cameras used, and `box(size_m, base)` the fit for a box of
    that size resting `base` [m] above the table. `tracked[key]` is where a tracked block is now
    (`Resting.label` its label): its base is the fit's base for the object that is that block when
    the cameras cannot tell its height, and it may hold other blocks. Returns the placed objects,
    several per label when the detector saw several, and the number of detections dropped as invalid.
    """
    dropped = 0
    by_label: dict[str, dict[str, list[Detection]]] = {}
    for det in dets:
        reason = _problem(det, cameras, frames)
        if reason:
            dropped += 1
            _drop(reason, det)
            continue
        by_label.setdefault(det.label.strip().lower(), {}).setdefault(det.camera, []).append(det)
    tracked = tracked or {}
    groups: list[tuple[str, dict[str, Detection]]] = []
    for label, per_cam in by_label.items():
        several = sum(_label_of(key, r) == label for key, r in tracked.items()) > 1
        per_cam = {camera: _distinct(found) for camera, found in per_cam.items()}
        groups += [(label, g) for g in _objects(per_cam, cameras, frames, pixel_error_px, several)]
    claims = _claims(groups, tracked, cameras, frames, pixel_error_px)

    # Every object's mask per camera: what may hide a block resting under or behind another.
    masks: dict[str, dict[int, np.ndarray]] = {}
    for i, (_, g) in enumerate(groups):
        for camera, det in g.items():
            if det.mask is not None:
                masks.setdefault(camera, {})[i] = det.mask
    placed: dict[int, Located] = {}
    for i, (label, g) in enumerate(groups):
        others = {camera: [m for j, m in by.items() if j != i] for camera, by in masks.items()}
        key = claims[i]
        base = tracked[key].base if key is not None else 0.0
        base = base if base > SUPPORT_TOL_M else 0.0
        try:
            loc = _place(label, list(g.values()), cameras, frames, pixel_error_px, mapper, occluders, box,
                         others, base)
        except Exception:  # one bad object never costs the others
            logger.exception("placing %r failed", label)
            loc = None
        if loc is not None:
            placed[i] = loc

    def unplaced() -> dict[str, Resting]:
        taken = {claims[i] for i in placed}
        return {key: r for key, r in tracked.items() if key not in taken}

    if mapper is not None:
        placed = _refine(placed, groups, claims, tracked, cameras, frames, mapper, occluders, box)
    placed = _supported(placed, unplaced())
    placed = _once(placed)
    blocks: dict[int | str, Resting] = {i: r for i, loc in placed.items() if (r := _resting(loc)) is not None}
    blocks |= unplaced()
    kept = []
    for i, loc in placed.items():
        seen = list(groups[i][1])
        origins = [cameras[camera].T_cam2base[:3, 3] for camera in seen]
        if claims[i] is None and (
                (len(seen) < 2 and _touches(i, seen, masks))
                or (loc.n_cameras < 2 and _could_rest_higher(i, loc, origins, blocks))):
            logger.debug("%r: one camera, and it may rest on another block; waiting for two", loc.label)
            continue
        kept.append(loc)
    return kept, dropped


# ------------------------------------------------------------------ several objects of one label
def assign(cost: np.ndarray, gate: float) -> list[tuple[int, int]]:
    """The (row, column) pairs of least total cost, each row and column in one pair at most and
    every pair's cost under `gate`; more pairs win while each costs less than `gate`.

    Exact for up to `ASSIGN_EXACT` on the smaller side, the cheapest pair first beyond that."""
    cost = np.asarray(cost, float)
    if cost.ndim != 2 or 0 in cost.shape:
        return []
    if cost.shape[1] > cost.shape[0]:
        return sorted((i, j) for j, i in assign(cost.T, gate))
    n, m = cost.shape
    if m > ASSIGN_EXACT:
        pairs: list[tuple[int, int]] = []
        rows: set[int] = set()
        cols: set[int] = set()
        for c, i, j in sorted((float(cost[i, j]), i, j) for i in range(n) for j in range(m)):
            if c < gate and i not in rows and j not in cols:
                pairs.append((i, j))
                rows.add(i)
                cols.add(j)
        return sorted(pairs)
    # Row by row over the sets of columns taken: (cost, the set before, the column taken or -1).
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


def _label_of(key: str, r: Resting) -> str:
    return r.label or key


def _box_iou(a, b) -> float:
    try:
        au0, av0, au1, av1 = (float(x) for x in a)
        bu0, bv0, bu1, bv1 = (float(x) for x in b)
    except (TypeError, ValueError):
        return 0.0
    w = min(au1, bu1) - max(au0, bu0)
    h = min(av1, bv1) - max(av0, bv0)
    inter = max(w, 0.0) * max(h, 0.0)
    union = (au1 - au0) * (av1 - av0) + (bu1 - bu0) * (bv1 - bv0) - inter
    return inter / union if union > 0 else 0.0


def _box_shift(a, b) -> float:
    """How far apart two boxes' centres lie, over the shorter side of either box."""
    try:
        au0, av0, au1, av1 = (float(x) for x in a)
        bu0, bv0, bu1, bv1 = (float(x) for x in b)
    except (TypeError, ValueError):
        return math.inf
    side = min(au1 - au0, av1 - av0, bu1 - bu0, bv1 - bv0)
    shift = math.hypot((au0 + au1 - bu0 - bu1) / 2, (av0 + av1 - bv0 - bv1) / 2)
    return shift / side if side > 0 else math.inf


def _same(a: Detection, b: Detection) -> bool:
    """Whether two detections of one label in one camera are one object reported twice: their masks
    overlap by `SAME_IOU`, their boxes do so with centres within `SAME_SHIFT`, or without either
    their pixels lie within `SAME_PX`."""
    if a.mask is not None and b.mask is not None:
        inter = int(np.count_nonzero(a.mask & b.mask))
        union = int(np.count_nonzero(a.mask | b.mask))
        return union > 0 and inter / union >= SAME_IOU
    if a.box is not None and b.box is not None:
        return _box_iou(a.box, b.box) >= SAME_IOU and _box_shift(a.box, b.box) <= SAME_SHIFT
    return math.hypot(a.pixel[0] - b.pixel[0], a.pixel[1] - b.pixel[1]) <= SAME_PX


def _distinct(dets: list[Detection]) -> list[Detection]:
    """One camera's detections of one label, surest first, each object once (`_same`)."""
    kept: list[Detection] = []
    for det in sorted(dets, key=_conf, reverse=True):
        if not any(_same(det, other) for other in kept):
            kept.append(det)
    return kept


_Ray = tuple[CameraModel, tuple[np.ndarray, np.ndarray], float]


def _ray(det: Detection, cameras: Mapping[str, CameraModel],
         frames: Mapping[str, tuple[tuple[int, int], float]], pixel_error_px: float) -> _Ray:
    """The detection's camera, its pixel's ray, and its pixel error in the camera's own pixels."""
    cam = cameras[det.camera]
    frame_size = frames[det.camera][0]
    err = pixel_error_px if det.pixel_error_px is None else float(det.pixel_error_px)
    # The error is in the frame's pixels; the tolerance wants the camera's own.
    return cam, pixel_ray(cam, det.pixel[0], det.pixel[1], frame_size), err * cam.size[0] / frame_size[0]


def _meet(rays: Sequence[_Ray]) -> tuple[np.ndarray, float] | None:
    """Where two or more rays meet and how far they miss it for their tolerances (1 = just within):
    None when they meet nowhere in front of their cameras at a height an object on the table may have.

    Two rays: the point sits halfway between them, so a gap under the mean tolerance is the rays
    passing within the sum of both cameras' tolerances. More rays: the same rule."""
    hit = triangulate([r for _, r, _ in rays])
    if hit is None:
        return None
    p, gap = hit
    depths = [float((p - o) @ d) for _, (o, d), _ in rays]
    if min(depths) < MIN_RANGE_M or not Z_RANGE_M[0] <= p[2] <= Z_RANGE_M[1]:
        return None
    tol = float(np.mean([ray_tolerance_m(cam, depth, err) for (cam, _, err), depth in zip(rays, depths, strict=True)]))
    return p, gap / tol


def _objects(per_cam: Mapping[str, list[Detection]], cameras: Mapping[str, CameraModel],
             frames: Mapping[str, tuple[tuple[int, int], float]], pixel_error_px: float,
             several: bool) -> list[dict[str, Detection]]:
    """One label's objects in one call, each at most one detection per camera.

    One detection per camera without masks, and at most one tracked object of the label: they are
    one object, as they always were. Otherwise the cameras' detections pair by geometry: each
    camera's join the objects of the cameras before it whose rays they meet within `PAIR_GATE`
    tolerances, the closest pairs first (`assign`), and the rest are objects of their own. So the
    masks of two blocks never go to one fit, even when each camera sees only one of them."""
    masked = any(d.mask is not None for found in per_cam.values() for d in found)
    if not several and not masked and all(len(found) == 1 for found in per_cam.values()):
        return [{camera: found[0] for camera, found in per_cam.items()}]
    groups: list[dict[str, Detection]] = []
    rays: list[list[_Ray]] = []
    for camera in [c for c in cameras if c in per_cam]:
        found = per_cam[camera]
        mine = [_ray(d, cameras, frames, pixel_error_px) for d in found]
        cost = np.array([[_miss(rays[g] + [r]) for r in mine] for g in range(len(groups))]).reshape(
            len(groups), len(mine))
        joined = set()
        for g, j in assign(cost, PAIR_GATE):
            groups[g][camera] = found[j]
            rays[g].append(mine[j])
            joined.add(j)
        for j, det in enumerate(found):
            if j not in joined:
                groups.append({camera: det})
                rays.append([mine[j]])
    return groups


def _miss(rays: Sequence[_Ray]) -> float:
    met = _meet(rays)
    return math.inf if met is None else met[1]


def _claims(groups: Sequence[tuple[str, Mapping[str, Detection]]], tracked: Mapping[str, Resting],
            cameras: Mapping[str, CameraModel], frames: Mapping[str, tuple[tuple[int, int], float]],
            pixel_error_px: float) -> list[str | None]:
    """Per object of this call, the key of the tracked block it is; None for one not tracked yet.

    A label with one object here and one tracked block pairs them, as one label always did.
    Otherwise each object takes the tracked block of its label that lies nearest its rays, within
    `CLAIM_M`."""
    out: list[str | None] = [None] * len(groups)
    for label in {label for label, _ in groups}:
        mine = [i for i, (other, _) in enumerate(groups) if other == label]
        keys = [key for key, r in tracked.items() if _label_of(key, r) == label]
        if not keys:
            continue
        if len(mine) == 1 and len(keys) == 1:
            out[mine[0]] = keys[0]
            continue
        cost = np.zeros((len(mine), len(keys)))
        for a, i in enumerate(mine):
            rays = [_ray(d, cameras, frames, pixel_error_px)[1] for d in groups[i][1].values()]
            for b, key in enumerate(keys):
                cost[a, b] = np.mean([_off_ray(tracked[key].pos, r) for r in rays])
        for a, b in assign(cost, CLAIM_M):
            out[mine[a]] = keys[b]
    return out


def _off_ray(p: np.ndarray, ray: tuple[np.ndarray, np.ndarray]) -> float:
    """How far the point lies from the ray [m]."""
    o, d = ray
    r = np.asarray(p, float) - o
    return float(np.linalg.norm(r - (r @ d) * d))


def _once(placed: dict[int, Located]) -> dict[int, Located]:
    """`placed` with one object where two of a label were placed in one spot: the one more cameras
    saw, then the surer one, stays."""
    order = sorted(placed, key=lambda i: (-placed[i].n_cameras, -placed[i].confidence, i))
    kept: list[int] = []
    for i in order:
        loc = placed[i]
        if any(placed[k].label == loc.label and _one_spot(placed[k], loc) for k in kept):
            logger.debug("%r: placed twice in one spot; kept once", loc.label)
            continue
        kept.append(i)
    return {i: placed[i] for i in sorted(kept)}


def _one_spot(a: Located, b: Located) -> bool:
    """Whether two placements of one label are one block. Two blocks of one size lying flat cannot
    share their footprint centres closer than their narrower side unless one rests on the other,
    a block's height higher; half of each is the bar."""
    d = np.asarray(b.pos, float) - np.asarray(a.pos, float)
    if a.size_m is None:
        return float(np.linalg.norm(d)) < MIN_RAY_GAP_M
    return float(np.hypot(d[0], d[1])) < min(a.size_m[:2]) / 2 and abs(float(d[2])) < a.size_m[2] / 2


def _supported(placed: dict[int, Located], ground: Mapping[str, Resting]) -> dict[int, Located]:
    """`placed` without the box fits that rest on nothing: a fit off the table needs a block under
    it, one on the table, a tracked one this call did not place (`ground`), or one resting on those."""
    holding: dict[int | str, Resting] = {**ground}
    waiting: dict[int, Resting] = {}
    for i, loc in placed.items():
        r = _resting(loc)
        if r is None:
            continue
        if loc.method == "sylwetka" and r.base > SUPPORT_TOL_M:
            waiting[i] = r
        else:
            holding[i] = r
    while True:
        found = [i for i, r in waiting.items() if any(other != i and s.holds(r) for other, s in holding.items())]
        if not found:
            break
        for i in found:
            holding[i] = waiting.pop(i)
    for i in waiting:
        logger.debug("%r: off the table on no block; left out", placed[i].label)
    return {i: loc for i, loc in placed.items() if i not in waiting}


def _touches(i: int, cams: list[str], masks: Mapping[str, Mapping[int, np.ndarray]]) -> bool:
    """Whether object i's mask in one of `cams` touches another object's mask there."""
    for camera in cams:
        by = masks.get(camera, {})
        own = by.get(i)
        if own is not None and any(other != i and _near(own, m) for other, m in by.items()):
            return True
    return False


def _near(own: np.ndarray, other: np.ndarray) -> bool:
    """Whether two masks of one image touch: lie within `TOUCH_PX` of each other."""
    if other.shape != own.shape:
        return False
    v, u = np.nonzero(own)
    if len(u) == 0:
        return False
    k = TOUCH_PX
    y0, y1 = max(int(v.min()) - k, 0), int(v.max()) + k + 1
    x0, x1 = max(int(u.min()) - k, 0), int(u.max()) + k + 1
    kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    near = cv2.dilate(np.asarray(own[y0:y1, x0:x1], np.uint8), kernel) > 0
    return bool((near & np.asarray(other[y0:y1, x0:x1], bool)).any())


def _could_rest_higher(i: int, loc: Located, origins: list[np.ndarray], blocks: Mapping[int | str, Resting]) -> bool:
    """Whether object i, placed from one camera, could rest on another block instead: a camera's ray
    through it passes over that block at the height it would rest there."""
    r = _resting(loc)
    if r is None:
        return False
    for key, s in blocks.items():
        # A block standing on this one, or one it already rests on, is no other place to be.
        if key == i or (s.base >= r.top - SUPPORT_TOL_M and r.under(s.pos)) \
                or abs(s.top - r.base) <= SUPPORT_TOL_M:
            continue
        z = s.top + r.size_m[2] / 2
        for o in origins:
            d = r.pos - o
            if abs(d[2]) < 1e-9:
                continue
            if s.under(o + (z - o[2]) / d[2] * d):
                return True
    return False


_Pose = tuple[np.ndarray, np.ndarray, tuple[float, float, float]]


def _refine(placed: dict[int, Located], groups: Sequence[tuple[str, Mapping[str, Detection]]],
            claims: Sequence[str | None], tracked: Mapping[str, Resting], cameras: Mapping[str, CameraModel],
            frames: Mapping[str, tuple[tuple[int, int], float]], mapper: TableMapper,
            occluders: Callable[[dict[str, float]], dict[str, np.ndarray] | None] | None,
            box: Callable[[tuple[float, float, float], float], CubeDetector]) -> dict[int, Located]:
    """Blocks whose masks touch, fitted again together. A detector that splits one colour's blob
    between touching or stacked blocks cannot tell where one ends, and a block in front hides part
    of the one behind. So each box is fitted again, from where it is, to the touching masks taken
    together, with the silhouettes of the other boxes as pixels that say nothing about it: the boxes
    end up covering the masks between them. A box resting on another takes that one's top as its
    base. A box starts from this call's fit, else from the tracked pose of the block it is, else
    from a fit to its masks and those of the boxes it touches, resting on one of them or on the
    table (`_beside`). A block with none of these stays as it was, and so does one whose new fit
    lies more than `REFINE_M` from its start."""
    poses: dict[int, _Pose] = {}
    masked: dict[int, dict[str, np.ndarray]] = {}
    sizes: dict[int, tuple[float, float, float]] = {}
    for i, (_, g) in enumerate(groups):
        dets = sorted(g.values(), key=_conf, reverse=True)
        if not all(d.mask is not None and d.size_m is not None for d in dets) \
                or not _full_size(dets, cameras, frames):
            continue
        masked[i] = {c: d.mask for c, d in g.items() if d.mask is not None}
        sizes[i] = size = _size3(dets[0])
        loc = placed.get(i)
        if loc is not None and loc.method == "sylwetka":
            poses[i] = (np.asarray(loc.pos, float), np.asarray(loc.rot, float), size)
        elif loc is None and (key := claims[i]) is not None:
            r = tracked[key]
            poses[i] = (np.asarray(r.pos, float), np.asarray(r.rot, float), size)
    out = dict(placed)
    arm: dict[tuple, dict[str, np.ndarray] | None] = {}

    def arm_at(times: dict[str, float]) -> dict[str, np.ndarray] | None:
        key = tuple(sorted(times.items()))
        if key not in arm:
            arm[key] = occluders(dict(times)) if occluders is not None else None
        return arm[key]

    for i in [i for i in masked if i not in poses and i not in placed]:
        near = [j for j in poses if any(c in masked[j] and _near(m, masked[j][c]) for c, m in masked[i].items())]
        loc = _beside(groups[i][0], masked[i], sizes[i], [poses[j] for j in near], [masked[j] for j in near],
                      _dates(groups[i][1].values(), frames), _age(groups[i][1].values()), cameras, mapper,
                      arm_at, box) if near else None
        if loc is not None:
            poses[i] = (loc.pos, loc.rot, sizes[i])
            out[i] = loc
    own = {i: masked[i] for i in poses}
    for cluster in _clusters(own):
        start = {i: poses[i] for i in cluster}
        for _ in range(REFINE_ROUNDS):
            moved = 0.0
            union, others = _together(cluster, poses, own, cameras)
            for i in cluster:
                label, g = groups[i]
                pos, rot, size = poses[i]
                times = _dates(g.values(), frames)
                occ = arm_at(times)
                masks = {c: union[c] for c in own[i]}
                hidden = {c: others[i][c] | (np.asarray(occ[c], bool) if occ and c in occ else False) for c in masks}
                base = _rest_base(i, pos, rot, size, poses, out, tracked, claims)
                fitter = box(size, min(max(base, 0.0), MAX_BASE_M))
                yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
                fit = fitter.fit_silhouette(masks, mapper.cameras, (float(pos[0]), float(pos[1]), yaw), hidden)
                if fit is None or np.linalg.norm(np.asarray(fit.pos, float) - start[i][0]) > REFINE_M:
                    continue
                moved = max(moved, float(np.linalg.norm(np.asarray(fit.pos, float) - pos)))
                poses[i] = (np.asarray(fit.pos, float), np.asarray(fit.rot, float), size)
                # The other boxes hide part of it, not its masks: the cameras that saw it still do.
                seen = max(int(fit.n_cameras), out[i].n_cameras if i in out else 0)
                out[i] = Located(label, poses[i][0], poses[i][1], float(fit.confidence), seen,
                                 min(times.values()), size, "sylwetka", _age(g.values()))
            if moved < REFINE_STILL_M:
                break
    return out


def _beside(label: str, masks: Mapping[str, np.ndarray], size: tuple[float, float, float], near: Sequence[_Pose],
            near_masks: Sequence[Mapping[str, np.ndarray]], times: dict[str, float], age_s: float,
            cameras: Mapping[str, CameraModel], mapper: TableMapper,
            arm_at: Callable[[dict[str, float]], dict[str, np.ndarray] | None],
            box: Callable[[tuple[float, float, float], float], CubeDetector]) -> Located | None:
    """A block whose own masks gave no fit, fitted to them joined with the masks of the placed
    boxes they touch, those boxes' silhouettes hiding what they cover: a split of one colour's blob
    may give it pixels of its neighbours, or theirs of it. It is tried resting on each of those
    boxes, highest first, then on the table; the first fit wins."""
    occ = arm_at(times)
    joined: dict[str, np.ndarray] = {}
    hidden: dict[str, np.ndarray] = {}
    for c, m in masks.items():
        joined[c] = np.logical_or.reduce([m] + [nm[c] for nm in near_masks if c in nm and nm[c].shape == m.shape])
        cover = [_silhouette(cameras[c], *pose, m.shape) for pose in near]
        if occ and c in occ:
            cover.append(np.asarray(occ[c], bool))
        hidden[c] = np.logical_or.reduce(cover)
    tops = sorted({Resting(*pose).top for pose in near}, reverse=True)
    for base in [b for b in tops if b <= MAX_BASE_M] + [0.0]:
        fit = box(size, base).fit_masks(joined, mapper, hidden, t=min(times.values()))
        if fit is not None:
            return Located(label, np.asarray(fit.pos, float), np.asarray(fit.rot, float), float(fit.confidence),
                           int(fit.n_cameras), min(times.values()), size, "sylwetka", age_s)
    return None


def _dates(dets: Iterable[Detection], frames: Mapping[str, tuple[tuple[int, int], float]]) -> dict[str, float]:
    """Each detection's date by camera: its frame's capture time minus its `age_s`."""
    return {d.camera: float(frames[d.camera][1]) - float(d.age_s) for d in dets}


def _age(dets: Iterable[Detection]) -> float:
    """The largest `age_s` among `dets` [s]."""
    return max((float(d.age_s) for d in dets), default=0.0)


def _size3(det: Detection) -> tuple[float, float, float]:
    assert det.size_m is not None
    x, y, z = (float(s) for s in det.size_m)
    return x, y, z


def _clusters(own: Mapping[int, Mapping[str, np.ndarray]]) -> list[list[int]]:
    """The groups of two or more objects whose masks touch in some camera, directly or through others."""
    keys = sorted(own)
    links = {i: {j for j in keys if j != i and any(c in own[j] and _near(m, own[j][c]) for c, m in own[i].items())}
             for i in keys}
    seen: set[int] = set()
    out = []
    for i in keys:
        if i in seen:
            continue
        group, todo = [], [i]
        seen.add(i)
        while todo:
            k = todo.pop()
            group.append(k)
            for j in links[k] - seen:
                seen.add(j)
                todo.append(j)
        if len(group) > 1:
            out.append(sorted(group))
    return out


def _silhouette(cam: CameraModel, pos: np.ndarray, rot: np.ndarray, size: tuple[float, float, float],
                shape: tuple[int, ...]) -> np.ndarray:
    """The box's silhouette in the camera's image of this shape: the hull of its projected corners."""
    half = np.asarray(size, float) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half @ \
        np.asarray(rot, float).T + np.asarray(pos, float)
    Ti = np.linalg.inv(cam.T_cam2base)
    out = np.zeros(shape[:2], np.uint8)
    if ((corners @ Ti[:3, :3].T + Ti[:3, 3])[:, 2] < MIN_RANGE_M).any():
        return out > 0
    px, _ = cv2.projectPoints(corners, cv2.Rodrigues(Ti[:3, :3])[0], Ti[:3, 3], cam.K, cam.dist)
    hull = cv2.convexHull(px.reshape(-1, 2).astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(out, hull, 1)
    return out > 0


def _together(cluster: list[int], poses: Mapping[int, _Pose], own: Mapping[int, Mapping[str, np.ndarray]],
              cameras: Mapping[str, CameraModel]) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    """Per camera, the cluster's masks taken together; per object and camera, the silhouettes of
    the cluster's other boxes."""
    union: dict[str, np.ndarray] = {}
    others: dict[int, dict[str, np.ndarray]] = {i: {} for i in cluster}
    for camera in sorted({c for i in cluster for c in own[i]}):
        here = [i for i in cluster if camera in own[i]]
        union[camera] = np.logical_or.reduce([own[i][camera] for i in here])
        shape = union[camera].shape
        sils = {i: _silhouette(cameras[camera], *poses[i], shape) for i in cluster}
        for i in here:
            rest = [sils[j] for j in cluster if j != i]
            others[i][camera] = np.logical_or.reduce(rest) if rest else np.zeros(shape, bool)
    return union, others


def _rest_base(i: int, pos: np.ndarray, rot: np.ndarray, size: tuple[float, float, float],
               poses: Mapping[int, _Pose], placed: Mapping[int, Located], tracked: Mapping[str, Resting],
               claims: Sequence[str | None]) -> float:
    """The base of object i's box: the top of the highest block that holds it, else where it is."""
    me = Resting(pos, rot, size)
    base = me.base
    if base <= SUPPORT_TOL_M:
        return base
    under = [Resting(*poses[j]) for j in poses if j != i]
    under += [r for j, loc in placed.items() if j != i and j not in poses and (r := _resting(loc)) is not None]
    taken = {claims[j] for j in placed} | {claims[j] for j in poses}
    under += [r for key, r in tracked.items() if key not in taken]
    tops = [s.top for s in under if s.holds(me)]
    return max(tops) if tops else base


def _conf(det: Detection) -> float:
    return min(1.0, max(0.0, float(det.confidence)))


def _hidden(masks: Mapping[str, np.ndarray], occ: Mapping[str, np.ndarray] | None,
            others: Mapping[str, list[np.ndarray]]) -> dict[str, np.ndarray] | None:
    """The arm masks widened by the other objects' masks, per camera of `masks`; None when no other
    object has a mask there."""
    out: dict[str, np.ndarray] = {}
    for camera, mask in masks.items():
        hide = [m for m in others.get(camera, []) if m.shape == mask.shape]
        if occ is not None and camera in occ:
            hide.append(np.asarray(occ[camera], bool))
        if hide:
            out[camera] = np.logical_or.reduce(hide)
    if not any(others.get(camera) for camera in masks):
        return None
    return out


def _place(label: str, dets: list[Detection], cameras: Mapping[str, CameraModel],
           frames: Mapping[str, tuple[tuple[int, int], float]], pixel_error_px: float,
           mapper: TableMapper | None,
           occluders: Callable[[dict[str, float]], dict[str, np.ndarray] | None] | None,
           box: Callable[[tuple[float, float, float], float], CubeDetector],
           others: Mapping[str, list[np.ndarray]] | None = None, base: float = 0.0) -> Located | None:
    dets = sorted(dets, key=_conf, reverse=True)
    best = dets[0]
    size = None if best.size_m is None else tuple(float(s) for s in best.size_m)
    size3 = None if size is None else (size[0], size[1], size[2])
    times = _dates(dets, frames)
    age = _age(dets)

    # The silhouette fit runs on raw masks with each camera's K, so every frame must be full size.
    if (mapper is not None and size3 is not None
            and all(d.mask is not None for d in dets) and _full_size(dets, cameras, frames)):
        masks = {d.camera: d.mask for d in dets if d.mask is not None}
        occ = occluders(dict(times)) if occluders is not None else None
        fitter = box(size3, min(max(base, 0.0), MAX_BASE_M))
        fit = fitter.fit_masks(masks, mapper, occ, t=min(times.values()))
        if fit is None:
            # Partly hidden by another block (one resting on it, or one in front): the other
            # objects' pixels say nothing about this one.
            hidden = _hidden(masks, occ, others or {})
            if hidden is not None:
                fit = fitter.fit_masks(masks, mapper, hidden, t=min(times.values()))
        if fit is None:
            return None
        return Located(label, np.asarray(fit.pos, float), np.asarray(fit.rot, float), float(fit.confidence),
                       int(fit.n_cameras), min(times.values()), size3, "sylwetka", age)

    rays = [_ray(d, cameras, frames, pixel_error_px) for d in dets]
    if len(rays) >= 2:
        met = _meet(rays)
        if met is not None and met[1] <= 1.0:
            return Located(label, met[0], np.eye(3), float(np.mean([_conf(d) for d in dets])), len(dets),
                           min(times.values()), size3, "triangulacja", age)

    o, d = rays[0][1]
    p = ray_plane(o, d, _height(best))
    if p is None:
        return None
    return Located(label, p, np.eye(3), _conf(best), 1, times[best.camera], size3, "promien",
                   float(best.age_s))
