"""The perception pieces the detectors build on: the camera rule, the tracker's capture time, the
mapper shared between threads, and `fit_masks`, the box fit for masks a detector made.

`fit_masks` runs the fit of `detect_frames` on masks of the raw frames; only its start differs (the
masks warped onto the top face's plane instead of a colour threshold of the warped frames). The
render test holds the two paths to each other and to the physics on the `cube_scene` stand. The box
fit on synthetic masks (the convex hull of a shape's projected points, no render) holds the cube to
what the cube-only fit gave, and places a cuboid and a lying cylinder with their yaw.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest
from conftest import BOT
from test_objects import A, B, H, W

from machinekind_twin.perception import (
    CubeDetection,
    CubeDetector,
    CubeTracker,
    ObjectTracker,
    TableMapper,
    perception_cameras,
)
from machinekind_twin.workspace import CameraRecord, Workspace

K = [[560.0, 0.0, 322.0], [0.0, 560.0, 236.0], [0.0, 0.0, 1.0]]
#: Above the base, looking straight down (OpenCV axes: z down, y along the base's -y).
TOP_DOWN = [[1.0, 0.0, 0.0, 0.2], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.6], [0.0, 0.0, 0.0, 1.0]]

#: How far the two paths may land apart. Measured over 60 poses on `cube_scene` (x 0.14..0.30 m,
#: y -0.08..0.08 m, yaw 0..65 deg): at most 1.04 mm and 4.4 deg. The yaw spread is the fit's own:
#: from a slightly different start Nelder-Mead stops at another point of a flat IoU ridge.
PATHS_APART_M = 0.0015
PATHS_APART_DEG = 5.0
#: How far `detect_frames` lands from the physics on `cube_scene`, measured over the same 60 poses:
#: at most 1.89 mm and 6.7 deg (`fit_masks`: 1.89 mm and 6.1 deg). The cube example's test adds
#: 1 mm and 1 deg to these.
LEGACY_ERROR_M = 0.0025
LEGACY_ERROR_DEG = 7.5
#: The poses the render test checks: (x [m], y [m], yaw [deg]).
POSES = [(0.18, 0.0, 0.0), (0.26, -0.08, 40.0), (0.14, 0.08, 65.0)]


def yaw_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def yaw_apart_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Yaw difference of two cube rotations, modulo the cube's 90 deg symmetry."""
    return abs((yaw_deg(Ra) - yaw_deg(Rb) + 45.0) % 90.0 - 45.0)


# ------------------------------------------------------------------ the camera rule
def _camera(name: str, source: str = "0", **kw) -> CameraRecord:
    kw.setdefault("T_cam2base", TOP_DOWN)
    return CameraRecord(name, source, 640, 480, K=K, **kw)


def test_perception_cameras_are_enabled_calibrated_and_trusted_unless_simulated():
    ws = Workspace(robot=BOT)
    ws.add_camera(_camera("trusted", calibration={"trusted": True}))
    ws.add_camera(_camera("untrusted"))
    ws.add_camera(_camera("uncalibrated", T_cam2base=None, calibration={"trusted": True}))
    ws.add_camera(_camera("disabled", enabled=False, calibration={"trusted": True}))
    ws.add_camera(_camera("sim", "sim", sim_pose=TOP_DOWN))

    assert [c.name for c in perception_cameras(ws)] == ["trusted", "sim"]
    assert [c.name for c in perception_cameras(ws, only_trusted=False)] == ["trusted", "untrusted", "sim"]
    assert list(TableMapper.from_workspace(ws).cameras) == ["trusted", "sim"]
    assert list(TableMapper.from_workspace(ws, only_trusted=False).cameras) == ["trusted", "untrusted", "sim"]


# ------------------------------------------------------------------ the mapper across threads
def test_at_height_builds_one_mapper_per_height_when_threads_ask_at_once():
    K_ = np.asarray(K)
    mapper = TableMapper({"top": (K_, None, np.asarray(TOP_DOWN))})
    n = 8
    ready = threading.Barrier(n)
    got: list[TableMapper] = []

    def ask():
        ready.wait()
        got.append(mapper.at_height(0.03))

    threads = [threading.Thread(target=ask) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == n and all(m is got[0] for m in got)
    assert got[0].table_z == pytest.approx(0.03)


# ------------------------------------------------------------------ the tracker
def test_cube_tracker_is_the_object_tracker_under_its_old_name():
    assert CubeTracker is ObjectTracker
    from machinekind_twin.robots import get_spec

    assert type(ObjectTracker.for_spec(get_spec(BOT))) is ObjectTracker


def test_t_seen_is_the_capture_time_of_the_last_accepted_measurement():
    tr = ObjectTracker()
    far = np.eye(4)
    far[:3, 3] = (0.0, 0.0, 10.0)
    assert tr.t_seen == -np.inf

    seen = CubeDetection(np.array([0.2, 0.0, 0.015]), np.eye(3), 1.0, 0.9, n_cameras=2, t=4.5)
    tr.update(seen, far, 0.0, 0.0, 0.0, now=5.0)
    assert tr.source == "kamery" and tr.t_seen == 4.5

    # Nothing seen: the last pose is held, its capture time stays.
    tr.update(None, far, 0.0, 0.0, 0.0, now=6.0)
    assert tr.source == "ostatnie widziane" and tr.t_seen == 4.5

    # A measurement without a capture time counts from the update that brings it.
    undated = CubeDetection(np.array([0.21, 0.0, 0.015]), np.eye(3), 1.0, 0.9, n_cameras=2)
    tr.update(undated, far, 0.0, 0.0, 0.0, now=7.0)
    assert tr.t_seen == 7.0

    # A capture time of 0 is a time (a virtual clock starting at 0), not "none given".
    at_zero = CubeDetection(np.array([0.21, 0.0, 0.015]), np.eye(3), 1.0, 0.9, n_cameras=2, t=0.0)
    tr.update(at_zero, far, 0.0, 0.0, 0.0, now=8.0)
    assert tr.t_seen == 0.0


# ------------------------------------------------------------------ fit_masks against detect_frames
@pytest.mark.render
@pytest.mark.parametrize("x, y, yaw", POSES)
def test_fit_masks_lands_where_detect_frames_does(cube_scene, x, y, yaw):
    cube_scene.place(x, y, yaw)
    frames, occ = cube_scene.frames(), cube_scene.occluders()
    pos, R = cube_scene.truth()
    mapper = TableMapper.from_workspace(cube_scene.ws)
    det = CubeDetector()

    legacy = det.detect_frames(frames, mapper, occ, t=3.0)
    masks = det.fit_masks({n: det.mask(img) > 0 for n, img in frames.items()}, mapper, occ, t=3.0)

    assert legacy is not None and masks is not None
    assert masks.n_cameras == legacy.n_cameras == 2
    assert masks.t == legacy.t == 3.0
    assert np.linalg.norm(masks.pos - legacy.pos) <= PATHS_APART_M
    assert yaw_apart_deg(masks.rot, legacy.rot) <= PATHS_APART_DEG
    for got in (legacy, masks):
        assert np.linalg.norm(got.pos - pos) <= LEGACY_ERROR_M
        assert yaw_apart_deg(got.rot, R) <= LEGACY_ERROR_DEG


@pytest.mark.render
def test_fit_masks_ignores_masks_of_cameras_the_mapper_does_not_know(cube_scene):
    cube_scene.place(0.2, 0.0, 20.0)
    frames, occ = cube_scene.frames(), cube_scene.occluders()
    mapper = TableMapper.from_workspace(cube_scene.ws)
    det = CubeDetector()
    masks = {n: det.mask(img) > 0 for n, img in frames.items()}

    alone = det.fit_masks(masks, mapper, occ)
    with_stranger = det.fit_masks({**masks, "stranger": np.ones((480, 640), bool)}, mapper, occ)

    assert alone is not None and with_stranger is not None
    assert np.array_equal(alone.pos, with_stranger.pos)


# ------------------------------------------------------------------ the box fit on synthetic masks
#: The map the synthetic tests fit on: `test_objects`' camera a (distorted) and b (pinhole).
MAPPER = TableMapper({"a": (A.K, A.dist, A.T_cam2base), "b": (B.K, None, B.T_cam2base)},
                     centre_xy=(0.22, 0.0), side=0.56)
#: A 30 mm cube, a 50x25x25 mm cuboid lying on a long side and that cuboid's centre on the table [m].
CUBE_HALF = (0.015, 0.015, 0.015)
CUBOID_HALF = (0.025, 0.0125, 0.0125)
CUBOID_AT = (0.22, 0.03)
#: The cube-only fit of origin/main (fb3838f, before the box fit) on `hull_mask`s of a 30 mm cube at
#: `POSES`: centre (x, y, z) [m], yaw [rad] and IoU, recorded on macOS arm64 with OpenCV 5.0.0.
CUBE_GOLDEN = {
    (0.18, 0.0, 0.0): ((0.17998124408100719, -0.00017129495160211357, 0.015),
                       -0.010667874469638128, 1.0),
    (0.26, -0.08, 40.0): ((0.2598834471553878, -0.08018597502741454, 0.015),
                          0.7050773992116826, 1.0),
    (0.14, 0.08, 65.0): ((0.14010934508379694, 0.07999583975146626, 0.015),
                         -0.4392508898565072, 0.9958847736625515),
}
#: How far another OpenCV build may move the golden fit: (position [m], yaw [rad], IoU). Mask
#: rasterisation and remap rounding differ between builds (the Mac against the DGX's aarch64) and
#: change Nelder-Mead's path; a wrong height or yaw fold moves the fit by millimetres or tens of
#: degrees.
GOLDEN_ATOL = (5e-4, np.radians(0.5), 5e-3)


def rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_points(half, x: float, y: float, yaw: float) -> np.ndarray:
    """The 8 corners of a box of these half sides resting on the table at (x, y), turned by `yaw`."""
    hx, hy, hz = half
    local = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return local @ rz(yaw).T + np.array([x, y, hz])


def cylinder_points(length: float, diameter: float, x: float, y: float, yaw: float,
                    n: int = 72) -> np.ndarray:
    """A cylinder lying on the table at (x, y), its axis along `yaw`: its two end rings of n points."""
    r = diameter / 2
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([np.zeros(n), r * np.cos(a), r * np.sin(a)])
    local = np.vstack([ring - [length / 2, 0.0, 0.0], ring + [length / 2, 0.0, 0.0]])
    return local @ rz(yaw).T + np.array([x, y, r])


def hull_mask(points: np.ndarray, cam) -> np.ndarray:
    """The silhouette of a convex shape in one camera: its points projected, their hull filled."""
    T = np.linalg.inv(cam.T_cam2base)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    dist = np.zeros(5) if cam.dist is None else cam.dist
    uv, _ = cv2.projectPoints(np.asarray(points, float).reshape(-1, 1, 3), rvec, T[:3, 3], cam.K, dist)
    hull = cv2.convexHull(uv.reshape(-1, 2).astype(np.float32)).astype(np.int32)
    img = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(img, hull, 1)
    return img > 0


def masks_of(points: np.ndarray) -> dict[str, np.ndarray]:
    return {cam.name: hull_mask(points, cam) for cam in (A, B)}


def map_pixels(points_xy: np.ndarray, mapper: TableMapper) -> np.ndarray:
    """Base-frame (x, y) to map pixels (u, v): the inverse of `TableMapper.to_base`."""
    s = mapper.side / mapper.n
    x0, y0 = mapper.centre[0] - mapper.side / 2, mapper.centre[1] + mapper.side / 2
    p = np.asarray(points_xy, float)
    return np.column_stack([(p[:, 0] - x0) / s - 0.5, (y0 - p[:, 1]) / s - 0.5])


def fit(det: CubeDetector, masks: dict[str, np.ndarray]) -> CubeDetection | None:
    return det.fit_masks(masks, MAPPER)


def turn_apart_deg(a_deg: float, b_deg: float, period_deg: float) -> float:
    """How far apart two yaws are, modulo a symmetry's period [deg]."""
    return abs((a_deg - b_deg + period_deg / 2) % period_deg - period_deg / 2)


@pytest.mark.parametrize("x, y, yaw", POSES)
def test_the_box_fit_of_a_cube_matches_main_and_the_two_constructors_agree(x, y, yaw):
    masks = masks_of(box_points(CUBE_HALF, x, y, np.radians(yaw)))
    by_side = fit(CubeDetector(cube_half=0.015), masks)
    by_sides = fit(CubeDetector(half_size=CUBE_HALF), masks)

    assert by_side is not None and by_sides is not None
    pos, yaw_rad, iou = CUBE_GOLDEN[(x, y, yaw)]
    atol_m, atol_rad, atol_iou = GOLDEN_ATOL
    np.testing.assert_allclose(by_side.pos, pos, rtol=0, atol=atol_m)
    assert turn_apart_deg(yaw_deg(by_side.rot), np.degrees(yaw_rad), 90.0) <= np.degrees(atol_rad)
    assert abs(by_side.confidence - iou) <= atol_iou
    # A cube given by its three sides is the cube given by one, to the bit.
    assert np.array_equal(by_sides.pos, by_side.pos) and np.array_equal(by_sides.rot, by_side.rot)
    assert by_sides.confidence == by_side.confidence and by_sides.n_cameras == by_side.n_cameras


@pytest.mark.parametrize("yaw", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
def test_a_cuboid_is_placed_with_its_yaw_from_two_cameras(yaw):
    x, y = CUBOID_AT
    got = fit(CubeDetector(half_size=CUBOID_HALF), masks_of(box_points(CUBOID_HALF, x, y, np.radians(yaw))))

    assert got is not None and got.n_cameras == 2
    assert np.linalg.norm(got.pos - [x, y, CUBOID_HALF[2]]) <= 0.001
    assert turn_apart_deg(yaw_deg(got.rot), yaw, 180.0) <= 2.0
    assert got.confidence >= 0.95


def test_a_cuboid_s_yaw_is_folded_to_a_half_turn_and_a_cube_s_to_a_quarter():
    assert CubeDetector(half_size=CUBOID_HALF).yaw_period == np.pi
    assert CubeDetector(cube_half=0.015).yaw_period == np.pi / 2
    assert CubeDetector(half_size=(0.0125, 0.0125, 0.025)).yaw_period == np.pi / 2

    x, y = CUBOID_AT
    got = fit(CubeDetector(half_size=CUBOID_HALF), masks_of(box_points(CUBOID_HALF, x, y, np.radians(170.0))))

    assert got is not None
    assert -90.0 <= yaw_deg(got.rot) < 90.0
    assert abs(yaw_deg(got.rot) - -10.0) <= 2.0


def test_a_lying_cylinder_as_its_bounding_box_fits_in_a_band_below_a_box():
    # The box fit knows no cylinder: a detector lists one lying as its bounding box (L, D, D). Seen
    # across its axis a cylinder is D high from any elevation while a box grows to 1.41 D from 45
    # degrees, so the fit's IoU stays well under a box's and around the 0.75 gate. Which yaws fall
    # under the gate depends on the fit's start, so only the band and "at least one" are pinned. A
    # model of the two end rings instead of 8 corners fits at IoU 0.98-1.00, but the twin would
    # need to know the shape: an API field the contract does not have.
    x, y = CUBOID_AT
    det = CubeDetector(half_size=CUBOID_HALF, min_iou=0.0)
    fits = {yaw: fit(det, masks_of(cylinder_points(0.05, 0.025, x, y, np.radians(yaw))))
            for yaw in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)}

    confidences = []
    for got in fits.values():
        assert got is not None
        confidences.append(got.confidence)
    assert all(0.70 <= c <= 0.85 for c in confidences), confidences
    assert min(confidences) < 0.75 <= CubeDetector().min_iou
    assert max(confidences) >= 0.78

    across = fits[90.0]
    assert across is not None and across.confidence >= 0.78
    assert np.linalg.norm(across.pos - [x, y, CUBOID_HALF[2]]) <= 0.002
    assert turn_apart_deg(yaw_deg(across.rot), 90.0, 180.0) <= 2.0


def _top_face(half, yaw_deg_: float, mapper: TableMapper) -> np.ndarray:
    """A box's footprint at (0.22, 0.03) drawn on the map by its BASE-frame yaw (uint8, 255 inside)."""
    x, y = CUBOID_AT
    corners = box_points(half, x, y, np.radians(yaw_deg_))[::2, :2]  # the four bottom corners
    uv = map_pixels(corners, mapper)
    hull = cv2.convexHull(np.round(uv * 16).astype(np.int32))
    img = np.zeros((mapper.n, mapper.n), np.uint8)
    cv2.fillConvexPoly(img, hull, 255, shift=4)
    return img


def test_the_start_from_the_top_face_turns_the_long_side_along_x():
    cuboid = CubeDetector(half_size=CUBOID_HALF)
    top = cuboid._top_mapper(MAPPER)

    for drawn in (60.0, -60.0):
        start = cuboid._from_mask(_top_face(CUBOID_HALF, drawn, top), top)
        assert start is not None
        # Outside a cube's [-45, 45): the half-turn fold keeps the long side's direction.
        assert abs(yaw_deg(start.rot) - drawn) <= 3.0
        assert np.linalg.norm(start.pos - [*CUBOID_AT, CUBOID_HALF[2]]) <= 0.003

    # Listed with its long side second, the box's x side is the rectangle's short side.
    across = (CUBOID_HALF[1], CUBOID_HALF[0], CUBOID_HALF[2])
    start = CubeDetector(half_size=across)._from_mask(_top_face(across, 30.0, top), top)
    assert start is not None
    assert abs(yaw_deg(start.rot) - 30.0) <= 3.0

    cube = CubeDetector(cube_half=0.015)
    start = cube._from_mask(_top_face(CUBE_HALF, 60.0, top), top)
    assert start is not None
    assert abs(yaw_deg(start.rot) - -30.0) <= 3.0


# ------------------------------------------------------------------ blocks on blocks
#: Where the stacked cube rests: on a 29 mm cube [m].
ON_CUBE = 0.029


def stacked(half, x: float, y: float, yaw_deg_: float, base: float) -> np.ndarray:
    """A box's corners resting `base` above the table."""
    return box_points(half, x, y, np.radians(yaw_deg_)) + [0.0, 0.0, base]


@pytest.mark.parametrize("half, base", [(CUBE_HALF, ON_CUBE), (CUBOID_HALF, ON_CUBE), (CUBE_HALF, 0.014)])
def test_with_the_base_searched_a_box_resting_on_another_is_placed_at_its_height(half, base):
    x, y, yaw = 0.21, 0.02, 25.0
    masks = masks_of(stacked(half, x, y, yaw, base))

    got = fit(CubeDetector(half_size=half, max_base_z=0.15), masks)

    assert got is not None and got.n_cameras == 2
    assert np.linalg.norm(got.pos - [x, y, base + half[2]]) <= 0.0015
    period = 90.0 if half[0] == half[1] else 180.0
    assert turn_apart_deg(yaw_deg(got.rot), yaw, period) <= 2.0
    # Without the search the box on the table cannot cover both cameras' masks.
    legacy = fit(CubeDetector(half_size=half), masks)
    assert legacy is None or np.linalg.norm(legacy.pos - got.pos) > 0.01


@pytest.mark.parametrize("x, y, yaw", POSES)
def test_with_the_base_searched_a_box_on_the_table_stays_on_the_table(x, y, yaw):
    masks = masks_of(box_points(CUBE_HALF, x, y, np.radians(yaw)))

    got = fit(CubeDetector(half_size=CUBE_HALF, max_base_z=0.15), masks)

    assert got is not None and got.pos[2] == CUBE_HALF[2]
    assert np.linalg.norm(got.pos - [x, y, CUBE_HALF[2]]) <= 0.001


def test_one_camera_cannot_tell_the_height_so_base_z_gives_it():
    x, y, yaw = 0.21, 0.02, 25.0
    masks = {"a": masks_of(stacked(CUBE_HALF, x, y, yaw, ON_CUBE))["a"]}
    truth = [x, y, ON_CUBE + CUBE_HALF[2]]

    on_table = fit(CubeDetector(half_size=CUBE_HALF, max_base_z=0.15), masks)
    on_cube = fit(CubeDetector(half_size=CUBE_HALF, max_base_z=0.15, base_z=ON_CUBE), masks)

    assert on_table is None or np.linalg.norm(on_table.pos - truth) > 0.02
    assert on_cube is not None and on_cube.n_cameras == 1
    assert np.linalg.norm(on_cube.pos - truth) <= 0.003


def test_masks_of_two_blocks_one_per_camera_make_no_box_in_the_air():
    masks = {"a": masks_of(box_points(CUBE_HALF, 0.24, 0.04, 0.0))["a"],
             "b": masks_of(box_points(CUBE_HALF, 0.18, -0.03, 0.0))["b"]}

    ungated = fit(CubeDetector(half_size=CUBE_HALF, max_base_z=0.15, lift_iou=0.0), masks)
    got = fit(CubeDetector(half_size=CUBE_HALF, max_base_z=0.15), masks)

    # The rays meet in the air, where a box covers both masks only partly.
    assert ungated is not None and ungated.pos[2] > 0.03 and ungated.confidence < 0.85
    assert got is None
