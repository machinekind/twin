"""Where a detector's pixels are in the robot's base frame: camera models, rays and `locate`.

A detector says what it sees and where IN THE IMAGE (`machinekind_twin_api.Detection`). This
module does the geometry. It builds a pinhole model (with distortion) of every camera the
perception may use, turns a pixel into a ray in the base frame, and places each label one of
three ways:

- "sylwetka": every camera gave a mask and the object's size is known. The box fit
  (`CubeDetector.fit_masks`) gives position and yaw from all masks at once, for a box of that size
  resting on the table or on another block. A block another one partly hides is fitted again with
  the other labels' masks as pixels it may be hidden behind.
- "triangulacja": two or more cameras saw the label and their rays pass close enough to each
  other for the detectors' pixel error. The point nearest to all rays.
- "promien": one camera, or rays that disagree. The surest detection's ray meets the plane at
  the height the detection gives (`height_m`, else half of `size_m`, else the table top).

A box fit off the table counts only when the block rests on another placed or tracked block
(`_supported`). One camera cannot tell a block on the table from one on top of a stack along the
same ray, so a block one camera places, with no tracked pose yet, is left out while it could rest
on another block: its mask touches another label's (`_touches`), or its ray passes over another
block (`_could_rest_higher`). Two cameras place it.

The geometry only ever uses the CALIBRATED pose, never `sim_pose`, so simulated perception tests
the calibration just as `TableMapper` does.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
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
    #: Capture time of the OLDEST frame used, on the Perception clock (time.monotonic in the panel).
    t: float
    size_m: tuple[float, float, float] | None
    #: How it was placed, as the panel shows it: "sylwetka", "triangulacja" or "promien".
    method: str


@dataclass(frozen=True, eq=False)
class Resting:
    """A block as stacking sees it: its centre and rotation in the base frame and its size [m]."""

    pos: np.ndarray
    rot: np.ndarray
    size_m: tuple[float, float, float]

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
    that size resting `base` [m] above the table. `tracked[label]` is where a tracked block is
    now: its base is the fit's base for that label when the cameras cannot tell its height, and it
    may hold other blocks. Returns the placed objects and the number of detections dropped as invalid.
    """
    dropped = 0
    by_label: dict[str, dict[str, Detection]] = {}
    for det in dets:
        reason = _problem(det, cameras, frames)
        if reason:
            dropped += 1
            _drop(reason, det)
            continue
        label = det.label.strip().lower()
        per_cam = by_label.setdefault(label, {})
        kept = per_cam.get(det.camera)
        if kept is None or _conf(det) > _conf(kept):
            per_cam[det.camera] = det

    # Every label's mask per camera: what may hide a block resting under or behind another.
    masks: dict[str, dict[str, np.ndarray]] = {}
    for label, per_cam in by_label.items():
        for camera, det in per_cam.items():
            if det.mask is not None:
                masks.setdefault(camera, {})[label] = det.mask
    tracked = tracked or {}
    out = []
    for label, per_cam in by_label.items():
        others = {camera: [m for other, m in by.items() if other != label] for camera, by in masks.items()}
        base = tracked[label].base if label in tracked else 0.0
        base = base if base > SUPPORT_TOL_M else 0.0
        try:
            placed = _place(label, list(per_cam.values()), cameras, frames, pixel_error_px, mapper,
                            occluders, box, others, base)
        except Exception:  # one bad label never costs the others
            logger.exception("placing %r failed", label)
            placed = None
        if placed is not None:
            out.append(placed)
    out = _supported(out, tracked)
    blocks = {loc.label: r for loc in out if (r := _resting(loc)) is not None}
    blocks |= {label: r for label, r in tracked.items() if label not in blocks}
    kept = []
    for loc in out:
        seen = list(by_label[loc.label])
        origins = [cameras[camera].T_cam2base[:3, 3] for camera in seen]
        if loc.label not in tracked and (
                (len(seen) < 2 and _touches(loc.label, seen, masks))
                or (loc.n_cameras < 2 and _could_rest_higher(loc, origins, blocks))):
            logger.debug("%r: one camera, and it may rest on another block; waiting for two", loc.label)
            continue
        kept.append(loc)
    return kept, dropped


def _supported(placed: list[Located], tracked: Mapping[str, Resting]) -> list[Located]:
    """`placed` without the box fits that rest on nothing: a fit off the table needs a block under
    it, one on the table, a tracked one this call did not place, or one resting on those."""
    ground = {label: r for label, r in tracked.items() if label not in {loc.label for loc in placed}}
    waiting: dict[str, Resting] = {}
    for loc in placed:
        r = _resting(loc)
        if r is None:
            continue
        if loc.method == "sylwetka" and r.base > SUPPORT_TOL_M:
            waiting[loc.label] = r
        else:
            ground[loc.label] = r
    while True:
        found = [label for label, r in waiting.items()
                 if any(other != label and s.holds(r) for other, s in ground.items())]
        if not found:
            break
        for label in found:
            ground[label] = waiting.pop(label)
    for label in waiting:
        logger.debug("%r: off the table on no block; left out", label)
    return [loc for loc in placed if loc.label not in waiting]


def _touches(label: str, cams: list[str], masks: Mapping[str, Mapping[str, np.ndarray]]) -> bool:
    """Whether the label's mask in one of `cams` touches another label's mask there."""
    for camera in cams:
        by = masks.get(camera, {})
        own = by.get(label)
        if own is None:
            continue
        v, u = np.nonzero(own)
        if len(u) == 0:
            continue
        k = TOUCH_PX
        y0, y1 = max(int(v.min()) - k, 0), int(v.max()) + k + 1
        x0, x1 = max(int(u.min()) - k, 0), int(u.max()) + k + 1
        kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
        near = cv2.dilate(np.asarray(own[y0:y1, x0:x1], np.uint8), kernel) > 0
        for other, m in by.items():
            if other != label and m.shape == own.shape and (near & np.asarray(m[y0:y1, x0:x1], bool)).any():
                return True
    return False


def _could_rest_higher(loc: Located, origins: list[np.ndarray], blocks: Mapping[str, Resting]) -> bool:
    """Whether a block placed from one camera could rest on another block instead: a camera's ray
    through it passes over that block at the height it would rest there."""
    r = _resting(loc)
    if r is None:
        return False
    for label, s in blocks.items():
        # A block standing on this one, or one it already rests on, is no other place to be.
        if label == loc.label or (s.base >= r.top - SUPPORT_TOL_M and r.under(s.pos)) \
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


def _conf(det: Detection) -> float:
    return min(1.0, max(0.0, float(det.confidence)))


def _hidden(masks: Mapping[str, np.ndarray], occ: Mapping[str, np.ndarray] | None,
            others: Mapping[str, list[np.ndarray]]) -> dict[str, np.ndarray] | None:
    """The arm masks widened by the other labels' masks, per camera of `masks`; None when no other
    label has a mask there."""
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
    times = {d.camera: float(frames[d.camera][1]) for d in dets}

    # The silhouette fit runs on raw masks with each camera's K, so every frame must be full size.
    if (mapper is not None and size3 is not None
            and all(d.mask is not None for d in dets) and _full_size(dets, cameras, frames)):
        masks = {d.camera: d.mask for d in dets if d.mask is not None}
        occ = occluders(dict(times)) if occluders is not None else None
        fitter = box(size3, min(max(base, 0.0), MAX_BASE_M))
        fit = fitter.fit_masks(masks, mapper, occ, t=min(times.values()))
        if fit is None:
            # Partly hidden by another block (one resting on it, or one in front): the other
            # labels' pixels say nothing about this one.
            hidden = _hidden(masks, occ, others or {})
            if hidden is not None:
                fit = fitter.fit_masks(masks, mapper, hidden, t=min(times.values()))
        if fit is None:
            return None
        return Located(label, np.asarray(fit.pos, float), np.asarray(fit.rot, float), float(fit.confidence),
                       int(fit.n_cameras), min(times.values()), size3, "sylwetka")

    rays = []
    for d in dets:
        cam = cameras[d.camera]
        frame_size = frames[d.camera][0]
        err = pixel_error_px if d.pixel_error_px is None else float(d.pixel_error_px)
        # The error is in the frame's pixels; the tolerance wants the camera's own.
        err_cam = err * cam.size[0] / frame_size[0]
        rays.append((cam, pixel_ray(cam, d.pixel[0], d.pixel[1], frame_size), err_cam))

    if len(rays) >= 2:
        hit = triangulate([r for _, r, _ in rays])
        if hit is not None:
            p, gap = hit
            depths = [float((p - o) @ d) for _, (o, d), _ in rays]
            # Two rays: the point sits halfway between them, so `gap` under the mean tolerance is
            # the rays passing within the sum of both cameras' tolerances. More rays: the same rule.
            tol = float(np.mean([ray_tolerance_m(cam, depth, err)
                                  for (cam, _, err), depth in zip(rays, depths, strict=True)]))
            if gap <= tol and min(depths) >= MIN_RANGE_M and Z_RANGE_M[0] <= p[2] <= Z_RANGE_M[1]:
                return Located(label, p, np.eye(3), float(np.mean([_conf(d) for d in dets])), len(dets),
                               min(times.values()), size3, "triangulacja")

    o, d = rays[0][1]
    p = ray_plane(o, d, _height(best))
    if p is None:
        return None
    return Located(label, p, np.eye(3), _conf(best), 1, times[best.camera], size3, "promien")
