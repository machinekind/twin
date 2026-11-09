"""ChArUco boards: the spec and geometry, corner labels, detection on a synthetic view, solvePnP, the board fit
on the table and the intrinsics with their verdict. OpenCV tests skip without OpenCV."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from machinekind_twin_api import boards
from machinekind_twin_api.boards import BoardSpec, board_rotation, corner_label, corner_of, fit_board

PACKAGE = Path(__file__).resolve().parents[1]


def _cv2():
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco"):
        pytest.skip("OpenCV without the aruco module")
    return cv2


def test_it_imports_nothing_but_numpy_and_the_standard_library():
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(PACKAGE)!r})\n"
        "before = set(sys.modules)\n"
        "import machinekind_twin_api.boards\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "print(sorted(new - set(sys.stdlib_module_names)))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "['machinekind_twin_api', 'numpy']"


# --------------------------------------------------------------------------- the spec
@pytest.mark.parametrize("change, words", [
    ({"squares_x": 1}, "at least 2"),
    ({"squares_x": 2, "squares_y": 2}, "6 squares"),
    ({"square_m": 0.0}, "square_m"),
    ({"marker_m": 0.05}, "marker_m"),
    ({"dictionary": "6X6"}, "dictionary"),
    ({"thickness_m": -0.001}, "thickness_m"),
])
def test_a_bad_board_spec_says_what_is_wrong(change, words):
    spec = BoardSpec(**change)
    with pytest.raises(ValueError, match=words):
        spec.check()


def test_inner_corners_follow_opencvs_numbering():
    spec = BoardSpec()
    assert spec.n_corners == 8 and spec.size_m == pytest.approx((0.135, 0.225))
    want = [[0.045, 0.045], [0.09, 0.045], [0.045, 0.09], [0.09, 0.18]]
    assert spec.corner_xy([0, 1, 2, 7]) == pytest.approx(np.array(want))


def test_the_board_spec_matches_opencvs_board():
    cv2 = _cv2()
    spec = BoardSpec(square_m=0.04, marker_m=0.03)
    corners = np.asarray(spec.cv_board(cv2).getChessboardCorners(), float)
    assert spec.corner_obj(range(spec.n_corners)) == pytest.approx(corners)
    with pytest.raises(ValueError, match="no dictionary"):
        BoardSpec(dictionary="DICT_NOPE").cv_board(cv2)


def test_a_corner_label_never_ends_in_a_bare_number_and_reads_back():
    assert corner_label("charuco", 3) == "charuco c3"
    assert corner_label(" Tablica  A ", 12) == "tablica a c12"
    assert corner_of("charuco", "Charuco  c5") == 5
    assert corner_of("charuco", "charuco c5 2") is None
    assert corner_of("charuco", "red cube") is None and corner_of("charuco", "charuco cx") is None


# --------------------------------------------------------------------------- the board on the table
def test_the_board_fit_recovers_yaw_origin_and_scale():
    spec = BoardSpec()
    ids = list(range(spec.n_corners))
    yaw, origin = math.radians(7.0), np.array([0.13, 0.11])
    R = board_rotation(yaw)
    base = spec.corner_xy(ids) @ R[:2, :2].T + origin
    fit = fit_board(spec.corner_xy(ids), base)
    assert fit.yaw == pytest.approx(yaw, abs=1e-9) and fit.origin == pytest.approx(origin, abs=1e-9)
    assert fit.scale == pytest.approx(1.0) and fit.rms_m < 1e-9
    # A board printed smaller than its set size: the scale gives it away.
    small = fit_board(spec.corner_xy(ids), (spec.corner_xy(ids) * 41.2 / 45) @ R[:2, :2].T + origin)
    assert small.scale == pytest.approx(41.2 / 45)


def test_a_board_lying_face_up_points_its_z_axis_into_the_table():
    R = board_rotation(math.radians(30.0))
    assert R[:, 2] == pytest.approx([0.0, 0.0, -1.0])
    assert np.linalg.det(R) == pytest.approx(1.0)


# --------------------------------------------------------------------------- a synthetic view
def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.column_stack([x, np.cross(z, x), z]), eye
    return T


K = np.array([[900.0, 0.0, 639.5], [0.0, 900.0, 359.5], [0.0, 0.0, 1.0]])
SIZE = (1280, 720)


def synthetic(cv2, spec: BoardSpec, T_board_cam: np.ndarray, px_per_m: float = 6000.0,
              sub: int = 4) -> np.ndarray:
    """The board as the camera sees it: the printed image warped by the exact plane homography at `sub` times
    the resolution, then averaged down, as a lens and a sensor would."""
    board = spec.image(cv2, px_per_m)
    # Board pixel (u, v) -> board point ((u + 0.5) / s, (v + 0.5) / s, 0) -> camera pixel, `sub` times finer.
    s = px_per_m
    A = np.array([[1 / s, 0.0, 0.5 / s], [0.0, 1 / s, 0.5 / s], [0.0, 0.0, 1.0]])
    up = np.array([[sub, 0.0, (sub - 1) / 2], [0.0, sub, (sub - 1) / 2], [0.0, 0.0, 1.0]])
    R, t = T_board_cam[:3, :3], T_board_cam[:3, 3]
    H = up @ K @ np.column_stack([R[:, 0], R[:, 1], t]) @ A
    big = cv2.warpPerspective(board, H, (SIZE[0] * sub, SIZE[1] * sub), flags=cv2.INTER_LINEAR,
                              borderValue=180)
    img = cv2.resize(big, SIZE, interpolation=cv2.INTER_AREA)
    return np.repeat(img[:, :, None], 3, axis=2)


def project(T_board_cam: np.ndarray, obj: np.ndarray) -> np.ndarray:
    c = obj @ T_board_cam[:3, :3].T + T_board_cam[:3, 3]
    return (c[:, :2] / c[:, 2:3]) * np.diag(K)[:2] + K[:2, 2]


def a_view(spec: BoardSpec, distance: float = 0.3) -> np.ndarray:
    """A board pose in the camera: looked at from `distance`, tilted."""
    centre = np.array([*np.array(spec.size_m) / 2, 0.0])
    T_cam_board = look_at(centre + np.array([0.08, -0.06, -distance]), centre, up=(0.0, 1.0, 0.0))
    return np.linalg.inv(T_cam_board)


def test_corners_are_found_within_a_third_of_a_pixel():
    cv2 = _cv2()
    spec = BoardSpec()
    T = a_view(spec)
    _, detector = boards.charuco_detector(spec, cv2)
    ids, pixels = boards.detect_corners(synthetic(cv2, spec, T), detector, cv2)
    assert sorted(ids) == list(range(spec.n_corners))
    truth = project(T, spec.corner_obj(ids))
    assert np.abs(pixels - truth).max() < 0.3


def test_a_printed_board_seen_head_on_gives_each_corner_at_its_pixel():
    """The board image at 3000 px/m on a grey page: corner k at its board position times the scale."""
    cv2 = _cv2()
    spec = BoardSpec()
    board = spec.image(cv2, 3000.0)
    page = np.full((board.shape[0] + 200, board.shape[1] + 300), 150, np.uint8)
    page[100:100 + board.shape[0], 150:150 + board.shape[1]] = board
    _, detector = boards.charuco_detector(spec, cv2)
    ids, pixels = boards.detect_corners(np.repeat(page[:, :, None], 3, axis=2), detector, cv2)
    assert sorted(ids) == list(range(spec.n_corners))
    want = spec.corner_xy(ids) * 3000.0 - 0.5 + np.array([150, 100])
    assert pixels == pytest.approx(want, abs=0.3)
    empty = np.full((480, 640, 3), 128, np.uint8)
    assert len(boards.detect_corners(empty, detector, cv2)[0]) == 0


def test_solvepnp_recovers_the_board_within_a_tenth_of_a_millimetre():
    cv2 = _cv2()
    spec = BoardSpec()
    T = a_view(spec, 0.25)
    board, detector = boards.charuco_detector(spec, cv2, min_markers=1)
    points = boards.board_points(synthetic(cv2, spec, T), board, detector, cv2)
    view = boards.board_pose(points, K, None, cv2)
    assert view.ok and view.points >= 6 and view.reproj_px < 0.5
    corners = spec.corner_obj(range(spec.n_corners))
    seen = corners @ view.R_cb.T + view.t_cb
    true = corners @ T[:3, :3].T + T[:3, 3]
    assert np.linalg.norm(seen - true, axis=1).max() < 1e-4
    angle = math.degrees(math.acos(min(1.0, (np.trace(view.R_cb.T @ T[:3, :3]) - 1) / 2)))
    assert angle < 0.05


def test_hidden_corners_are_left_out_and_marker_corners_fill_in():
    cv2 = _cv2()
    spec = BoardSpec()
    T = a_view(spec, 0.25)
    board, detector = boards.charuco_detector(spec, cv2, min_markers=1)
    image = synthetic(cv2, spec, T)
    full = boards.board_points(image, board, detector, cv2)
    assert full.n_corners == spec.n_corners and full.n_markers == 0
    # A mask over the left half of the frame, as the jaws would hide it.
    mask = np.zeros((SIZE[1], SIZE[0]), bool)
    px = project(T, spec.corner_obj(range(spec.n_corners)))
    mask[:, : int(np.median(px[:, 0]))] = True
    part = boards.board_points(image, board, detector, cv2, mask=mask)
    kept = project(T, part.obj)
    hidden = mask[kept[:, 1].round().astype(int), kept[:, 0].round().astype(int)]
    assert part.n_corners < spec.n_corners and not hidden.any()
    assert part.n_corners >= boards.MARKERS_BELOW or part.n_markers > 0


def test_too_few_or_collinear_points_are_unusable():
    cv2 = _cv2()
    spec = BoardSpec()
    obj = spec.corner_obj([0, 2, 4, 6, 1, 3])
    few = boards.BoardPoints(obj[:5], np.zeros((5, 2)), 5, 0)
    assert "za malo" in boards.board_pose(few, K, None, cv2).problem
    line = boards.BoardPoints(spec.corner_obj([0, 2, 4, 6, 0, 2]), np.zeros((6, 2)), 6, 0)
    assert "linii" in boards.board_pose(line, K, None, cv2).problem


# --------------------------------------------------------------------------- intrinsics
def test_the_intrinsics_come_back_from_tilted_views_and_need_a_spread_of_tilts():
    cv2 = _cv2()
    spec = BoardSpec()
    board, detector = boards.charuco_detector(spec, cv2, min_markers=1)
    centre = np.array([*np.array(spec.size_m) / 2, 0.0])
    views, flat = [], []
    rng = np.random.default_rng(3)
    for k in range(12):
        tilt = math.radians(-25 + 50 * k / 11)
        eye = centre + np.array([0.3 * math.sin(tilt), rng.uniform(-0.03, 0.03), -0.3 * math.cos(tilt)])
        T = np.linalg.inv(look_at(eye, centre + np.r_[rng.uniform(-0.02, 0.02, 2), 0.0], up=(0.0, 1.0, 0.0)))
        views.append(boards.board_points(synthetic(cv2, spec, T), board, detector, cv2, all_markers=True))
        if abs(tilt) < math.radians(6):
            flat.append(views[-1])
    result = boards.calibrate(views, SIZE, 1000.0, cv2)
    assert result.trusted, result.reason
    assert result.K[0, 0] == pytest.approx(900.0, rel=0.01) and result.rms_px < 0.5
    # Views at one tilt cannot tell focal length from distance.
    narrow = boards.calibrate(flat * 6, SIZE, 1000.0, cv2)
    assert not narrow.trusted and "rozrzut pochylen" in narrow.reason
    assert not boards.calibrate(views[:3], SIZE, 1000.0, cv2).trusted


def _projected_views(cv2, n: int, noisy: dict[int, float], seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Board corners projected by `K` from `n` tilted poses, 0.1 px noise, more noise in the views `noisy`."""
    spec = BoardSpec(squares_x=9, squares_y=6, square_m=0.028, marker_m=0.021)
    obj = spec.corner_obj(range(spec.n_corners))
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        tilt = math.radians(-25 + 50 * i / max(1, n - 1))
        centre = np.array([*np.array(spec.size_m) / 2, 0.0])
        eye = centre + np.array([0.4 * math.sin(tilt), rng.uniform(-0.05, 0.05), -0.4 * math.cos(tilt)])
        T = np.linalg.inv(look_at(eye, centre, up=(0.0, 1.0, 0.0)))
        px = project(T, obj) + rng.normal(0.0, noisy.get(i, 0.1), (len(obj), 2))
        out.append((obj, px))
    return out


def test_views_far_off_the_rest_are_left_out_and_the_rest_fitted_again():
    cv2 = _cv2()
    views = _projected_views(cv2, 12, {3: 4.0, 7: 4.0})
    fit = boards.fit_intrinsics(views, SIZE, cv2, max_rms_px=0.6, outlier_factor=2.0, min_kept=8)
    assert sorted(i for i, _ in fit.dropped) == [3, 7] and fit.kept == [i for i in range(12) if i not in (3, 7)]
    assert len(fit.per_view_px) == len(fit.kept) == len(fit.rvecs)
    assert fit.K[0, 0] == pytest.approx(900.0, rel=0.01) and fit.rms_px < 0.3
    # Never below `min_kept` views.
    floor = boards.fit_intrinsics(views, SIZE, cv2, max_rms_px=0.6, outlier_factor=2.0, min_kept=11)
    assert len(floor.kept) == 11 and len(floor.dropped) == 1


def test_a_limit_on_the_fits_leaves_the_last_view_out_without_fitting_again():
    cv2 = _cv2()
    views = _projected_views(cv2, 12, {1: 6.0, 4: 5.0, 9: 4.0})
    K0 = np.array([[1000.0, 0.0, 639.5], [0.0, 1000.0, 359.5], [0.0, 0.0, 1.0]])
    fit = boards.fit_intrinsics(views, SIZE, cv2, K0=K0, max_fits=2)
    assert [i for i, _ in fit.dropped] == [1, 4]
    # The second fit still held view 4: its errors cover one view more than are kept.
    assert len(fit.per_view_px) == len(fit.kept) + 1 == 11
    full = boards.fit_intrinsics(views, SIZE, cv2, K0=K0)
    assert sorted(i for i, _ in full.dropped) == [1, 4, 9] and len(full.per_view_px) == len(full.kept) == 9
