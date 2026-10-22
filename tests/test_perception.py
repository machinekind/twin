"""The perception pieces the detectors build on: the camera rule, the tracker's capture time, the
mapper shared between threads, and `fit_masks`, the cube fit for masks a detector made.

`fit_masks` runs the fit of `detect_frames` on masks of the raw frames; only its start differs (the
masks warped onto the top face's plane instead of a colour threshold of the warped frames). The
render test holds the two paths to each other and to the physics on the `cube_scene` stand.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from conftest import BOT

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
