"""The detector geometry (`objects.py`): camera models from the workspace, pixel rays, the three
ways to place a label, and what `locate` drops. Synthetic cameras and projected pixels, no render.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import cv2
import numpy as np
import pytest
from conftest import BOT
from machinekind_twin_api.detector import Detection

from machinekind_twin import objects
from machinekind_twin.objects import (
    MIN_RAY_GAP_M,
    Z_RANGE_M,
    CameraModel,
    camera_key,
    camera_models,
    locate,
    mapper_for,
    pixel_ray,
    ray_plane,
    ray_tolerance_m,
    triangulate,
)
from machinekind_twin.perception import CubeDetection
from machinekind_twin.robots import get_spec
from machinekind_twin.workspace import CameraRecord, Workspace

W, H = 640, 480
K = np.array([[600.0, 0.0, 322.0], [0.0, 600.0, 236.0], [0.0, 0.0, 1.0]])
DIST = np.array([0.1, -0.2, 0.0, 0.0, 0.05])


def look_at(eye, target) -> np.ndarray:
    """Camera-to-base pose of a camera at `eye` looking at `target` (OpenCV axes, y roughly down)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = eye
    return T


#: In front of the arm, looking at the table; with distortion.
A = CameraModel("a", K, DIST, look_at([0.2, -0.45, 0.45], [0.2, 0.0, 0.0]), (W, H))
#: From the right, looking at the table; an ideal pinhole.
B = CameraModel("b", K, None, look_at([0.65, 0.1, 0.4], [0.2, 0.0, 0.0]), (W, H))
CAMS = {"a": A, "b": B}


def project(cam: CameraModel, p) -> tuple[float, float]:
    T_base2cam = np.linalg.inv(cam.T_cam2base)
    rvec, _ = cv2.Rodrigues(T_base2cam[:3, :3])
    dist = np.zeros(5) if cam.dist is None else cam.dist
    uv, _ = cv2.projectPoints(np.asarray(p, float).reshape(1, 3), rvec, T_base2cam[:3, 3], cam.K, dist)
    u, v = uv.reshape(2)
    return float(u), float(v)


def ray_miss(ray, p) -> float:
    o, d = ray
    r = np.asarray(p, float) - o
    return float(np.linalg.norm(r - (r @ d) * d))


def frames(t_a: float = 1.0, t_b: float = 2.0, size=(W, H)) -> dict:
    return {"a": (size, t_a), "b": (size, t_b)}


def run(dets, *, pixel_error_px: float = 3.0, fr=None, mapper=None, occluders=None, cube=None):
    return locate(dets, CAMS, frames() if fr is None else fr, pixel_error_px=pixel_error_px,
                  mapper=mapper, occluders=occluders, cube=cube or (lambda side: pytest.fail("no fit expected")))


# ------------------------------------------------------------------ camera models
def _record(name: str, source: str = "0", **kw) -> CameraRecord:
    kw.setdefault("T_cam2base", A.T_cam2base.tolist())
    kw.setdefault("calibration", {"trusted": True})
    return CameraRecord(name, source, W, H, K=K.tolist(), **kw)


def test_camera_models_are_the_perception_cameras_without_the_ones_on_the_arm():
    assert "wrist" in get_spec(BOT).cameras
    ws = Workspace(robot=BOT)
    ws.add_camera(_record("front", dist=DIST.tolist()))
    ws.add_camera(_record("untrusted", calibration={}))
    ws.add_camera(_record("wrist"))
    ws.add_camera(_record("sim", "sim", sim_pose=B.T_cam2base.tolist(), calibration={}))

    models = camera_models(ws)
    assert list(models) == ["front", "sim"]
    front = models["front"]
    assert front.size == (W, H)
    np.testing.assert_allclose(front.K, K)
    np.testing.assert_allclose(front.dist, DIST)
    # The calibrated pose, never the simulation's truth.
    np.testing.assert_allclose(models["sim"].T_cam2base, A.T_cam2base)
    assert models["sim"].dist is None


def test_the_camera_key_changes_with_what_the_models_depend_on_only():
    ws = Workspace(robot=BOT)
    ws.add_camera(_record("front"))
    key = camera_key(ws)
    assert hash(key) == hash(camera_key(ws)) and key == camera_key(ws)

    ws.cameras[0].fps = 15
    assert camera_key(ws) == key
    for change in (lambda c: setattr(c, "enabled", False), lambda c: setattr(c, "calibration", {}),
                   lambda c: setattr(c, "T_cam2base", B.T_cam2base.tolist()),
                   lambda c: setattr(c, "dist", DIST.tolist()), lambda c: setattr(c, "width", 1280)):
        ws = Workspace(robot=BOT)
        ws.add_camera(_record("front"))
        change(ws.cameras[0])
        assert camera_key(ws) != key


def test_the_mapper_covers_exactly_the_models_on_the_robot_s_table():
    desk = get_spec(BOT).desk
    mapper = mapper_for(CAMS, desk)
    assert list(mapper.cameras) == ["a", "b"]
    assert mapper.centre == tuple(desk.map_centre) and mapper.side == desk.map_side
    assert mapper.cameras["a"][0] is A.K and mapper.cameras["b"][1] is None


# ------------------------------------------------------------------ rays
@pytest.mark.parametrize("cam", [A, B], ids=["distorted", "pinhole"])
@pytest.mark.parametrize("p", [(0.2, 0.0, 0.0), (0.3, 0.12, 0.05), (0.08, -0.1, 0.15)])
def test_a_pixel_ray_passes_through_the_point_it_was_projected_from(cam, p):
    u, v = project(cam, p)
    assert 0 <= u < W and 0 <= v < H
    o, d = pixel_ray(cam, u, v, (W, H))
    np.testing.assert_allclose(o, cam.T_cam2base[:3, 3])
    assert np.linalg.norm(d) == pytest.approx(1.0)
    assert ray_miss((o, d), p) < 1e-6


@pytest.mark.parametrize("cam", [A, B], ids=["distorted", "pinhole"])
def test_a_half_size_frame_gives_the_same_ray(cam):
    u, v = project(cam, (0.3, 0.12, 0.05))
    full = pixel_ray(cam, u, v, (W, H))
    half = pixel_ray(cam, (u + 0.5) / 2 - 0.5, (v + 0.5) / 2 - 0.5, (W // 2, H // 2))
    np.testing.assert_allclose(half[0], full[0])
    np.testing.assert_allclose(half[1], full[1], atol=1e-9)


def test_a_ray_meets_a_plane_only_ahead_and_not_along_it():
    o = np.array([0.0, 0.0, 0.5])
    np.testing.assert_allclose(ray_plane(o, np.array([0.0, 0.0, -1.0]), 0.1), [0.0, 0.0, 0.1])
    assert ray_plane(o, np.array([0.0, 0.0, 1.0]), 0.1) is None
    assert ray_plane(o, np.array([1.0, 0.0, 0.0]), 0.1) is None
    assert ray_plane(o, np.array([0.0, 0.0, -1.0]), 0.47) is None  # 3 cm in front


def test_triangulate_needs_two_rays_that_are_not_parallel():
    o, d = np.zeros(3), np.array([1.0, 0.0, 0.0])
    assert triangulate([(o, d)]) is None
    assert triangulate([(o, d), (o + [0.0, 0.1, 0.0], d)]) is None
    p, gap = triangulate([(o, d), (np.array([0.5, -1.0, 0.02]), np.array([0.0, 1.0, 0.0]))])
    np.testing.assert_allclose(p, [0.5, 0.0, 0.01])
    assert gap == pytest.approx(0.01)


def test_the_ray_tolerance_grows_with_depth_above_a_floor():
    assert ray_tolerance_m(A, 0.5, 3.0) == MIN_RAY_GAP_M
    assert ray_tolerance_m(A, 0.6, 40.0) == pytest.approx(0.04)


# ------------------------------------------------------------------ the point paths
P = np.array([0.24, 0.04, 0.02])


def test_two_cameras_triangulate_the_point_within_half_a_millimetre():
    dets = [Detection("pen", "a", project(A, P), confidence=0.8),
            Detection("pen", "b", project(B, P), confidence=0.6)]
    placed, dropped = run(dets)
    assert dropped == 0 and len(placed) == 1
    obj = placed[0]
    assert obj.method == "triangulacja" and obj.n_cameras == 2
    assert np.linalg.norm(obj.pos - P) < 5e-4
    np.testing.assert_array_equal(obj.rot, np.eye(3))
    assert obj.confidence == pytest.approx(0.7)
    assert obj.t == 1.0  # the oldest frame


def test_a_resized_frame_triangulates_the_same_point():
    u, v = project(B, P)
    dets = [Detection("pen", "a", project(A, P)),
            Detection("pen", "b", ((u + 0.5) / 2 - 0.5, (v + 0.5) / 2 - 0.5))]
    fr = {"a": ((W, H), 1.0), "b": ((W // 2, H // 2), 1.5)}
    (obj,), _ = run(dets, fr=fr)
    assert obj.method == "triangulacja" and np.linalg.norm(obj.pos - P) < 5e-4


def test_one_camera_hits_the_plane_at_half_the_object_s_height():
    on_table = np.array([0.24, 0.04, 0.015])
    (obj,), _ = run([Detection("cube", "a", project(A, on_table), size_m=(0.03, 0.03, 0.03))])
    assert obj.method == "promien" and obj.n_cameras == 1
    np.testing.assert_allclose(obj.pos, on_table, atol=1e-6)
    assert obj.size_m == (0.03, 0.03, 0.03)


def test_one_camera_without_a_height_hits_the_table_top():
    foot = np.array([0.24, 0.04, 0.0])
    (obj,), _ = run([Detection("pen", "b", project(B, foot))])
    assert obj.method == "promien"
    np.testing.assert_allclose(obj.pos, foot, atol=1e-6)
    assert obj.t == 2.0


def test_one_camera_hits_the_plane_at_the_given_height():
    p = np.array([0.24, 0.04, 0.1])
    (obj,), _ = run([Detection("pen", "a", project(A, p), height_m=0.1, size_m=(0.01, 0.01, 0.15))])
    np.testing.assert_allclose(obj.pos, p, atol=1e-6)


def _disagreeing(err_a: float | None = None, err_b: float | None = None) -> list[Detection]:
    """Camera b's pixel 40 px off the point camera a sees; a is the surer one."""
    u, v = project(B, P)
    return [Detection("pen", "a", project(A, P), confidence=0.9, height_m=float(P[2]), pixel_error_px=err_a),
            Detection("pen", "b", (u + 40.0, v), confidence=0.5, height_m=float(P[2]), pixel_error_px=err_b)]


def test_rays_that_disagree_for_the_pixel_error_fall_back_to_the_surer_ray():
    (obj,), _ = run(_disagreeing(), pixel_error_px=3.0)
    assert obj.method == "promien" and obj.n_cameras == 1
    np.testing.assert_allclose(obj.pos, P, atol=1e-6)
    assert obj.confidence == 0.9 and obj.t == 1.0


def test_a_larger_pixel_error_pairs_the_same_rays():
    (obj,), _ = run(_disagreeing(), pixel_error_px=40.0)
    assert obj.method == "triangulacja" and obj.n_cameras == 2


def test_a_detection_s_own_pixel_error_overrides_the_detector_s():
    (obj,), _ = run(_disagreeing(err_a=40.0, err_b=40.0), pixel_error_px=3.0)
    assert obj.method == "triangulacja"
    (obj,), _ = run(_disagreeing(err_a=3.0, err_b=3.0), pixel_error_px=40.0)
    assert obj.method == "promien"


def test_a_point_below_the_table_is_not_an_object_on_it():
    below = np.array([0.24, 0.04, Z_RANGE_M[0] - 0.05])
    dets = [Detection("pen", "a", project(A, below), confidence=0.9),
            Detection("pen", "b", project(B, below), confidence=0.5)]
    (obj,), _ = run(dets)
    assert obj.method == "promien" and obj.pos[2] == pytest.approx(0.0)


def test_a_point_behind_a_camera_is_not_triangulated():
    # Rays that cross 10 cm behind camera a: a's ray to P extended backwards, and a camera c
    # looking straight at that point.
    eye = A.T_cam2base[:3, 3]
    behind = eye + 0.1 * (eye - P) / np.linalg.norm(eye - P)
    C = CameraModel("c", K, None, look_at([0.6, -0.3, 0.5], behind), (W, H))
    cams = {"a": A, "c": C}
    dets = [Detection("pen", "a", project(A, P), confidence=0.9, height_m=float(P[2])),
            Detection("pen", "c", project(C, behind), confidence=0.5)]
    hit = triangulate([pixel_ray(A, *project(A, P), (W, H)), pixel_ray(C, *project(C, behind), (W, H))])
    assert hit is not None and np.linalg.norm(hit[0] - behind) < 1e-6 and hit[1] < 1e-6
    (obj,), _ = locate(dets, cams, {"a": ((W, H), 1.0), "c": ((W, H), 1.0)}, pixel_error_px=3.0,
                       mapper=None, occluders=None, cube=lambda side: pytest.fail("no fit expected"))
    assert obj.method == "promien"
    np.testing.assert_allclose(obj.pos, P, atol=1e-6)


# ------------------------------------------------------------------ age
def test_a_detection_s_age_moves_its_capture_time_back():
    (obj,), _ = run([Detection("pen", "a", project(A, P), height_m=float(P[2]), age_s=2.5)])

    assert obj.method == "promien"
    assert obj.t == -1.5
    assert obj.age_s == 2.5


def test_two_cameras_of_one_age_triangulate_at_the_aged_capture_time():
    dets = [Detection("pen", "a", project(A, P), age_s=10.0),
            Detection("pen", "b", project(B, P), age_s=10.0)]
    (obj,), _ = run(dets)

    assert obj.method == "triangulacja" and obj.n_cameras == 2
    assert np.linalg.norm(obj.pos - P) < 5e-4
    assert obj.t == 1.0 - 10.0
    assert obj.age_s == 10.0


def test_the_oldest_aged_frame_dates_a_triangulated_point():
    dets = [Detection("pen", "a", project(A, P)),
            Detection("pen", "b", project(B, P), age_s=1.5)]
    (obj,), _ = run(dets)

    assert obj.method == "triangulacja"
    assert obj.t == 0.5
    assert obj.age_s == 1.5


@pytest.mark.parametrize("age", [-0.5, float("nan"), float("inf")])
def test_an_age_that_is_not_a_finite_non_negative_number_is_dropped(age):
    placed, dropped = run([Detection("pen", "a", project(A, P), age_s=age)])

    assert placed == [] and dropped == 1


# ------------------------------------------------------------------ labels and what is dropped
def test_labels_are_compared_stripped_and_lower_cased():
    dets = [Detection("Red Pen ", "a", project(A, P), height_m=float(P[2])),
            Detection("red pen", "b", project(B, P), height_m=float(P[2]))]
    (obj,), _ = run(dets)
    assert obj.label == "red pen" and obj.method == "triangulacja"


def test_per_camera_the_surer_detection_of_a_label_wins():
    far = np.array([0.1, -0.1, 0.0])
    dets = [Detection("pen", "a", project(A, far), confidence=0.3),
            Detection("pen", "a", project(A, P), confidence=0.8, height_m=float(P[2])),
            Detection("pen", "a", project(A, far), confidence=0.5)]
    (obj,), dropped = run(dets)
    assert dropped == 0 and obj.n_cameras == 1 and obj.confidence == 0.8
    np.testing.assert_allclose(obj.pos, P, atol=1e-6)


def test_confidence_is_clipped_to_zero_to_one():
    (obj,), _ = run([Detection("pen", "a", project(A, P), confidence=3.0)])
    assert obj.confidence == 1.0


def test_bad_detections_are_dropped_and_counted():
    good = Detection("pen", "a", project(A, P))
    bad = [
        Detection("ghost", "nowhere", (10.0, 10.0)),
        Detection("ghost", "a", (float("nan"), 10.0)),
        Detection("ghost", "a", (W + 5.0, 10.0)),
        Detection("ghost", "a", (10.0, 10.0), mask=np.zeros((H // 2, W // 2), bool)),
        Detection("ghost", "a", (10.0, 10.0), mask=np.zeros((H, W), np.uint8)),
        Detection("  ", "a", (10.0, 10.0)),
        Detection("ghost", "a", (10.0, 10.0), size_m=(0.03, -0.03, 0.03)),
        Detection("ghost", "a", (10.0, 10.0), pixel_error_px=-1.0),
        Detection("ghost", "a", (10.0, 10.0), confidence=float("nan")),
    ]
    placed, dropped = run([good, *bad])
    assert dropped == len(bad)
    assert [o.label for o in placed] == ["pen"]


def test_a_camera_without_a_frame_in_this_call_is_dropped():
    # A camera that left the set during the call: a model, but no frame read.
    placed, dropped = run([Detection("pen", "b", project(B, P))], fr={"a": ((W, H), 1.0)})
    assert placed == [] and dropped == 1


def test_a_pixel_just_outside_the_frame_still_counts():
    (obj,), dropped = run([Detection("pen", "a", (-1.5, H + 0.5))])
    assert dropped == 0 and obj.method == "promien"


# ------------------------------------------------------------------ the silhouette path
class FakeCube:
    """Records the fit's input; answers with a fixed detection (or none)."""

    calls: ClassVar[list] = []

    def __init__(self, side: float, answer: CubeDetection | None):
        self.side, self.answer = side, answer

    def fit_masks(self, masks, mapper, occluders=None, t=0.0):
        FakeCube.calls.append((self.side, masks, mapper, occluders, t))
        return self.answer


def _masks(size=(0.03, 0.03, 0.03)) -> list[Detection]:
    m = np.zeros((H, W), bool)
    m[200:220, 300:320] = True
    return [Detection("cube", "a", project(A, P), confidence=0.9, mask=m, size_m=size),
            Detection("cube", "b", project(B, P), confidence=0.7, mask=m.copy(), size_m=size)]


def test_masks_of_a_cube_go_to_the_silhouette_fit():
    FakeCube.calls = []
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    answer = CubeDetection(np.array([0.2, 0.01, 0.015]), yaw, 1.0, 0.88, n_cameras=2)
    mapper = mapper_for(CAMS, get_spec(BOT).desk)
    occ = {"a": np.zeros((H, W), bool)}
    seen_times = []

    def occluders(times):
        seen_times.append(times)
        return occ

    dets = _masks()
    (obj,), dropped = run(dets, mapper=mapper, occluders=occluders, cube=lambda side: FakeCube(side, answer))
    assert dropped == 0
    assert obj.method == "sylwetka" and obj.n_cameras == 2 and obj.confidence == 0.88
    np.testing.assert_allclose(obj.pos, answer.pos)
    np.testing.assert_allclose(obj.rot, yaw)
    assert obj.t == 1.0 and obj.size_m == (0.03, 0.03, 0.03)
    assert seen_times == [{"a": 1.0, "b": 2.0}]
    (side, masks, used_mapper, used_occ, t), = FakeCube.calls
    assert side == pytest.approx(0.03) and used_mapper is mapper and used_occ is occ and t == 1.0
    assert masks["a"] is dets[0].mask and masks["b"] is dets[1].mask


def test_the_silhouette_fit_and_its_arm_masks_use_the_aged_times():
    FakeCube.calls = []
    answer = CubeDetection(np.array([0.2, 0.01, 0.015]), np.eye(3), 1.0, 0.88, n_cameras=2)
    mapper = mapper_for(CAMS, get_spec(BOT).desk)
    seen_times = []

    def occluders(times):
        seen_times.append(times)

    dets = [replace(d, age_s=3.0) for d in _masks()]
    (obj,), _ = run(dets, mapper=mapper, occluders=occluders, cube=lambda side: FakeCube(side, answer))

    assert obj.method == "sylwetka"
    assert seen_times == [{"a": -2.0, "b": -1.0}]
    (*_, t), = FakeCube.calls
    assert t == -2.0
    assert obj.t == -2.0
    assert obj.age_s == 3.0


def test_a_gated_out_fit_places_nothing():
    mapper = mapper_for(CAMS, get_spec(BOT).desk)
    placed, dropped = run(_masks(), mapper=mapper, cube=lambda side: FakeCube(side, None))
    assert placed == [] and dropped == 0


@pytest.mark.parametrize("why", ["no mapper", "not a cube", "a mask missing", "a resized frame"])
def test_without_everything_the_fit_needs_the_point_paths_place_the_object(why):
    mapper = mapper_for(CAMS, get_spec(BOT).desk)
    dets = _masks()
    fr = None
    if why == "no mapper":
        mapper = None
    elif why == "not a cube":
        dets = _masks(size=(0.03, 0.03, 0.06))
    elif why == "a mask missing":
        dets[1] = Detection("cube", "b", dets[1].pixel, size_m=(0.03, 0.03, 0.03))
    else:
        dets[1] = Detection("cube", "b", ((dets[1].pixel[0] + 0.5) / 2 - 0.5, (dets[1].pixel[1] + 0.5) / 2 - 0.5),
                            size_m=(0.03, 0.03, 0.03), mask=np.zeros((H // 2, W // 2), bool))
        fr = {"a": ((W, H), 1.0), "b": ((W // 2, H // 2), 2.0)}
    (obj,), _ = run(dets, mapper=mapper, fr=fr)
    assert obj.method in ("triangulacja", "promien")


def test_a_resized_frame_is_logged_once_per_camera(caplog, monkeypatch):
    monkeypatch.setattr(objects, "_warned_size", set())
    mapper = mapper_for(CAMS, get_spec(BOT).desk)
    dets = _masks()
    dets[1] = Detection("cube", "b", ((dets[1].pixel[0] + 0.5) / 2 - 0.5, (dets[1].pixel[1] + 0.5) / 2 - 0.5),
                        size_m=(0.03, 0.03, 0.03), mask=np.zeros((H // 2, W // 2), bool))
    fr = {"a": ((W, H), 1.0), "b": ((W // 2, H // 2), 2.0)}
    with caplog.at_level("WARNING", logger="machinekind_twin.objects"):
        for _ in range(3):
            (obj,), _ = run(dets, mapper=mapper, fr=fr)
            assert obj.method == "triangulacja"
    warned = [r.getMessage() for r in caplog.records if "its record says" in r.getMessage()]
    assert warned == [f"camera 'b' delivers {W // 2}x{H // 2} frames, its record says {W}x{H}: "
                      "no silhouette fit (no yaw) from it until the sizes match"]
