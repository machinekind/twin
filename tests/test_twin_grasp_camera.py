"""The wrist camera adapter: projection against training, the policy frame, the frame sources."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mujoco")

from lerobot_mp.twin.grasp import camera as cam  # noqa: E402
from lerobot_mp.twin.grasp.contract import (  # noqa: E402
    ARM_BASE_WORLD,
    CX,
    CY,
    FOCAL_PX,
    FOVY_DEG,
    HEAT_SIGMA_FRAC,
    IMG_H,
    IMG_W,
    NEAR_Z,
    RAW_H,
    RAW_W,
    T_GRIPPER_CAMERA,
    WRIST_DIST_2026_09_26,
    WRIST_K_2026_09_26,
)
from lerobot_mp.twin.grasp.rig import Rig  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics, pose  # noqa: E402
from lerobot_mp.twin.robots import A1X  # noqa: E402

FIX = Path(os.environ.get("GRASP_FIXTURES", Path(__file__).parent / "fixtures" / "grasp"))
RERECORD = "re-record the fixtures (docs/GRASP.md)"
ARM = tuple(f"arm_joint{k}" for k in range(1, 7))
BASE = np.array(ARM_BASE_WORLD)


def _fixture(name: str, *keys: str) -> dict[str, np.ndarray]:
    path = FIX / name
    if not path.is_file():
        pytest.fail(f"{RERECORD}: {path} is missing")
    z = np.load(path)
    missing = [k for k in keys if k not in z.files]
    if missing:
        pytest.fail(f"{RERECORD}: {name} has no {', '.join(missing)}")
    return {k: z[k] for k in keys}


def _joints(q: np.ndarray) -> dict[str, float]:
    return dict(zip(ARM, np.degrees(q).tolist()))


def _believed_base(name: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if name == "projection.npz":
        z = _fixture(name, "q", "believed_base", "uv", "uv_fk")
        return z, z["believed_base"]
    z = _fixture(name, "q", "believed_pos", "uv", "uv_fk")
    return z, z["believed_pos"] - BASE


def _assert_same_pixels(got: np.ndarray, want: np.ndarray, tol: float) -> None:
    want = np.asarray(want, float)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    ok = ~np.isnan(want[:, 0])
    err = np.abs(got[ok] - want[ok]).max()
    assert err <= tol, err


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics(A1X)


# ------------------------------------------------------------------ projection
def test_project_reproduces_the_recorded_training_pixels():
    z = _fixture("projection.npz", "gl_pos", "gl_R", "believed_base", "uv")
    uv = np.array([cam.project(pose(R, p - BASE), b)
                   for p, R, b in zip(z["gl_pos"], z["gl_R"], z["believed_base"])])
    assert uv.dtype == np.float32
    assert np.isnan(z["uv"][:, 0]).any()
    u, v = z["uv"][:, 0], z["uv"][:, 1]
    assert ((u < -0.5) | (u > IMG_W - 0.5) | (v < -0.5) | (v > IMG_H - 0.5)).any()   # off frame, kept
    _assert_same_pixels(uv, z["uv"], 1e-4)


@pytest.mark.parametrize("name", ["projection.npz", "shard.npz"])
def test_believed_uv_from_the_joints_matches_training_at_the_same_joints(kin, name):
    z, believed = _believed_base(name)
    wc = cam.WristCamera()
    uv = np.array([wc.believed_uv(kin, _joints(q), b) for q, b in zip(z["q"], believed)])
    _assert_same_pixels(uv, z["uv_fk"], 1e-4)


@pytest.mark.parametrize("name", ["projection.npz", "shard.npz"])
def test_training_pixels_lag_the_joints_by_about_a_pixel(name):
    """Training's own dot is computed from poses one 2 ms substep older than its joints.

    Checked within 640 px of the image centre, where the recording measured 1.06 px. A few
    projection samples lie further out (u about -1100, shallow depth): there the same 0.3 mm
    lag moves the pixel by up to 11 px, far beyond anything the heat channel shows.
    """
    z, _ = _believed_base(name)
    uv, uv_fk = np.asarray(z["uv"], float), np.asarray(z["uv_fk"], float)
    known = ~np.isnan(uv[:, 0])
    near = known & (np.hypot(uv[:, 0] - CX, uv[:, 1] - CY) <= 640.0)
    reach = 3 * HEAT_SIGMA_FRAC * IMG_W
    seen = (np.abs(uv[:, 0] - CX) < IMG_W / 2 + reach) & (np.abs(uv[:, 1] - CY) < IMG_H / 2 + reach)
    assert near.sum() >= 0.95 * known.sum() and (near & seen).sum() == seen.sum() > 50
    lag = np.abs(uv[near] - uv_fk[near]).max()
    assert lag <= 1.2, lag


def test_project_has_no_pixel_at_or_behind_the_camera_plane():
    T_hand = np.eye(4)
    T_cam = T_GRIPPER_CAMERA
    for depth in (-0.10, 0.0, 0.9 * NEAR_Z):
        p = (T_cam @ np.array([0.01, 0.02, depth, 1.0]))[:3]
        uv = cam.project(T_hand, p)
        assert uv.dtype == np.float32 and uv.shape == (2,) and np.isnan(uv).all()
    p = (T_cam @ np.array([0.01, -0.02, 0.20, 1.0]))[:3]
    uv = cam.project(T_hand, p)
    assert uv == pytest.approx([FOCAL_PX * 0.05 + CX, -FOCAL_PX * 0.1 + CY], abs=1e-4)
    assert np.isnan(cam.project(T_hand, np.array([np.nan, 0.0, 0.0]))).all()


def test_believed_uv_needs_all_six_arm_joints(kin):
    joints = dict(A1X.home)
    wc = cam.WristCamera()
    believed = np.array([0.30, 0.0, 0.04])
    arm_only = {k: v for k, v in joints.items() if k != "gripper"}
    assert np.array_equal(wc.believed_uv(kin, arm_only, believed), wc.believed_uv(kin, joints, believed),
                          equal_nan=True)
    del joints["arm_joint6"]
    with pytest.raises(KeyError, match="arm_joint6"):
        wc.believed_uv(kin, joints, believed)
    with pytest.raises(KeyError):
        wc.cube_top_uv(kin, joints, believed[:2], 0.03)


def test_cube_top_uv_is_the_projection_of_the_top_face_centre(kin):
    wc = cam.WristCamera()
    joints = {"arm_joint1": 90.0, "arm_joint2": 80.0, "arm_joint3": -100.0, "arm_joint4": 20.0,
              "arm_joint5": 0.0, "arm_joint6": 0.0}
    top = np.array([0.02, 0.31, 0.03])
    got = wc.cube_top_uv(kin, joints, top[:2], float(top[2]))
    want = cam.project(kin.body("gripper_link", joints), top)
    assert np.array_equal(got, want, equal_nan=True)


# ------------------------------------------------------------------ policy frame
@pytest.fixture(scope="module")
def upright_dots() -> tuple[np.ndarray, np.ndarray]:
    """An analytic dot pattern seen through the default lens: the UPRIGHT raw frame and the dots.

    Every raw pixel is taken back to the ideal pinhole by the inverse of the lens model; the
    brightness there is a sum of Gaussian dots placed at known policy-frame pixels.
    """
    K, dist = WRIST_K_2026_09_26, WRIST_DIST_2026_09_26
    ys, xs = np.mgrid[0:RAW_H, 0:RAW_W].astype(np.float64)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    ideal = cv2.undistortPoints(pts, K, dist, None, np.eye(3), np.eye(3), crit).reshape(RAW_H, RAW_W, 2)
    u, v = FOCAL_PX * ideal[..., 0] + CX, FOCAL_PX * ideal[..., 1] + CY

    dots = np.array([(du, dv) for dv in np.linspace(12.3, 107.7, 5) for du in np.linspace(14.2, 145.9, 6)])
    sigma = 1.5
    img = np.zeros((RAW_H, RAW_W))
    for du, dv in dots:
        img += np.exp(-((u - du) ** 2 + (v - dv) ** 2) / (2 * sigma ** 2))
    grey = np.clip(np.round(255.0 * img), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(grey[:, :, None], 3, axis=2)), dots


def _centroids(img: np.ndarray, dots: np.ndarray, half: int = 6) -> np.ndarray:
    grey = img[:, :, 1].astype(float)
    out = []
    for du, dv in dots:
        u0, v0 = int(round(du)), int(round(dv))
        win = grey[v0 - half:v0 + half + 1, u0 - half:u0 + half + 1]
        yy, xx = np.mgrid[v0 - half:v0 + half + 1, u0 - half:u0 + half + 1]
        out.append(((win * xx).sum() / win.sum(), (win * yy).sum() / win.sum()))
    return np.array(out)


@pytest.mark.parametrize("upside_down", [True, False])
def test_the_policy_frame_undoes_the_lens_and_the_upside_down_mount(upright_dots, upside_down):
    upright, dots = upright_dots
    raw = np.ascontiguousarray(upright[::-1, ::-1]) if upside_down else upright
    K, dist = WRIST_K_2026_09_26, WRIST_DIST_2026_09_26
    wc = cam.WristCamera(cam.WristIntrinsics(K, dist, upside_down=upside_down))
    img = wc.policy_frame(raw)
    assert img.shape == (IMG_H, IMG_W, 3) and img.dtype == np.uint8
    err = np.hypot(*(_centroids(img, dots) - dots).T)
    assert err.max() <= 0.25, err.max()

    # The same frame read with the rotation left out, or added, does not put the dots there.
    wrong = cam.WristCamera(cam.WristIntrinsics(K, dist, upside_down=not upside_down)).policy_frame(raw)
    at = np.round(dots).astype(int)
    assert img[at[:, 1], at[:, 0], 1].mean() > 200
    assert wrong[at[:, 1], at[:, 0], 1].mean() < 100


def test_the_default_wrist_camera_uses_the_calibrated_intrinsics(upright_dots):
    upright, _ = upright_dots
    raw = np.ascontiguousarray(upright[::-1, ::-1])
    default = cam.WristCamera()
    assert default.intr.upside_down and (default.intr.width, default.intr.height) == (RAW_W, RAW_H)
    explicit = cam.WristCamera(cam.WristIntrinsics(WRIST_K_2026_09_26, WRIST_DIST_2026_09_26))
    assert np.array_equal(default.policy_frame(raw), explicit.policy_frame(raw))


def test_a_frame_of_the_wrong_size_or_type_raises():
    wc = cam.WristCamera()
    for bad in (np.zeros((480, 640, 3), np.uint8), np.zeros((RAW_H, RAW_W, 3), np.float32),
                np.zeros((RAW_H, RAW_W), np.uint8), np.zeros((RAW_W, RAW_H, 3), np.uint8)):
        with pytest.raises(cam.CameraError):
            wc.policy_frame(bad)
    assert wc.policy_frame(np.zeros((RAW_H, RAW_W, 3), np.uint8)).shape == (IMG_H, IMG_W, 3)


def test_intrinsics_that_look_past_the_sensor_or_fold_are_refused():
    """A K and dist from the rig file must fill the policy frame from the sensor, through a lens
    that does not fold.

    `cv2.remap` complains about neither: it fills black where the map leaves the sensor and
    samples wherever a folding lens model points.
    """
    K = WRIST_K_2026_09_26
    for upside_down in (True, False):
        cam.WristCamera(cam.WristIntrinsics(K, WRIST_DIST_2026_09_26, upside_down=upside_down))
    # The calibrated K without its barrel distortion shows a strip above the sensor's top row.
    outside = (r"5120 of 640x480 px \(1\.67 %\) of the policy frame lie outside the 1920x1080 sensor "
               r"\(top edge")
    with pytest.raises(cam.CameraError, match=outside):
        cam.WristCamera(cam.WristIntrinsics(K, np.zeros(5)))
    # A negative k2 turns the barrel back inward towards the corners, inside the sensor.
    with pytest.raises(cam.CameraError, match="folds"):
        cam.WristCamera(cam.WristIntrinsics(K, np.array([-0.648, -0.5, 0.0, 0.0, 0.0])))
    with pytest.raises(cam.CameraError, match="no pixel"):
        cam.WristCamera(cam.WristIntrinsics(K, np.array([np.nan, 0.0, 0.0, 0.0, 0.0])))


def test_intrinsics_a_few_pixels_past_the_sensor_are_accepted_with_a_warning(caplog):
    # The calibrated view's top row samples the upright raw frame at y = 36; a principal point
    # 38 px higher puts a strip of 146 samples at 640x480 above the sensor.
    K = WRIST_K_2026_09_26.copy()
    K[1, 2] -= 38.0
    with caplog.at_level("WARNING", logger=cam.__name__):
        wc = cam.WristCamera(cam.WristIntrinsics(K, WRIST_DIST_2026_09_26))
    assert "146 of 640x480 px" in caplog.text and "top edge" in caplog.text
    img = wc.policy_frame(np.full((RAW_H, RAW_W, 3), 200, np.uint8))
    assert (img[0] < 200).any() and (img[2:] == 200).all()
    caplog.clear()
    with caplog.at_level("WARNING", logger=cam.__name__):
        cam.WristCamera()
    assert "outside" not in caplog.text


def test_wrist_intrinsics_carry_upside_down_and_k_from_the_rig_file(tmp_path):
    path = tmp_path / "grasp.json"
    path.write_text(json.dumps({"version": 1, "wrist": {"upside_down": False}}), encoding="utf-8")
    intr = cam.wrist_intrinsics(Rig.load(path))
    assert intr.upside_down is False
    assert np.array_equal(intr.K, WRIST_K_2026_09_26) and np.array_equal(intr.dist, WRIST_DIST_2026_09_26)

    K = [[1200.0, 0.0, 950.0], [0.0, 1210.0, 530.0], [0.0, 0.0, 1.0]]
    dist = [-0.5, 0.3, 0.0, 0.0, -0.1]
    path.write_text(json.dumps({"version": 1, "wrist": {"K": K, "dist": dist}}), encoding="utf-8")
    intr = cam.wrist_intrinsics(Rig.load(path))
    assert intr.upside_down is True
    assert np.array_equal(intr.K, np.array(K)) and np.array_equal(intr.dist, np.array(dist))
    assert cam.wrist_intrinsics(Rig.load(tmp_path / "none.json")).upside_down is True


# ------------------------------------------------------------------ frame sources
class _FakeLive:
    """What `cameras.open_live` returns, scripted: `rgb_t()` answers in order, the last one repeats."""

    def __init__(self, answers: list[tuple[np.ndarray | None, float, str | None]]) -> None:
        self.answers = list(answers)
        self.closed = False

    def rgb_t(self) -> tuple[np.ndarray | None, float, str | None]:
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def no_device(monkeypatch):
    """Anything that would open a real camera fails the test."""
    from lerobot_mp.twin import cameras
    from lerobot_mp.vision import camera as vision_camera

    def refuse(*a, **kw):
        raise AssertionError("a camera device was opened")
    monkeypatch.setattr(cv2, "VideoCapture", refuse)
    monkeypatch.setattr(cameras, "open_live", refuse)
    monkeypatch.setattr(vision_camera.CameraStream, "open", refuse)


def test_the_live_source_opens_only_through_its_opener_and_converts_each_frame_once(no_device):
    raw1 = np.full((RAW_H, RAW_W, 3), 40, np.uint8)
    raw2 = np.full((RAW_H, RAW_W, 3), 90, np.uint8)
    live = _FakeLive([(None, 0.0, None), (raw1, 10.0, None), (raw1, 10.0, None), (raw2, 10.05, None),
                      (None, 0.0, "Koniec strumienia obrazu.")])
    records = []

    def opener(record):
        records.append(record)
        return live

    wc = cam.WristCamera()
    converted = []
    real = wc.policy_frame
    wc.policy_frame = lambda raw: converted.append(raw) or real(raw)
    src = cam.LiveWristSource("3", wc, opener=opener)

    (rec,) = records
    assert (rec.name, rec.source, rec.width, rec.height, rec.fps) == ("wrist", "3", RAW_W, RAW_H, 30)
    assert src.latest() is None                                       # before the first frame
    first = src.latest()
    assert (first.t, first.seq) == (10.0, 1)
    assert first.image.shape == (IMG_H, IMG_W, 3) and first.image.dtype == np.uint8
    assert src.latest() is first                                      # same stamp: not converted again
    assert len(converted) == 1
    second = src.latest()
    assert (second.t, second.seq) == (10.05, 2) and len(converted) == 2
    assert second.image.mean() > first.image.mean()
    with pytest.raises(cam.CameraError, match="Koniec strumienia"):
        src.latest()
    src.close()
    assert live.closed


def test_the_live_source_turns_a_failed_open_into_a_camera_error(no_device):
    def opener(record):
        raise RuntimeError("Nie mozna otworzyc kamery: 7")
    with pytest.raises(cam.CameraError, match="Nie mozna otworzyc"):
        cam.LiveWristSource("7", cam.WristCamera(), opener=opener)


def test_the_live_source_rejects_a_frame_of_the_wrong_size(no_device):
    live = _FakeLive([(np.zeros((480, 640, 3), np.uint8), 1.0, None)])
    src = cam.LiveWristSource("0", cam.WristCamera(), opener=lambda record: live)
    with pytest.raises(cam.CameraError):
        src.latest()


def test_the_live_source_reports_a_failing_read_and_a_closed_stream_as_camera_errors(no_device):
    class Broken(_FakeLive):
        def rgb_t(self):
            raise OSError("USB transfer failed")

    live = Broken([(None, 0.0, None)])
    src = cam.LiveWristSource("0", cam.WristCamera(), opener=lambda record: live)
    with pytest.raises(cam.CameraError, match="USB transfer failed"):
        src.latest()
    src.close()
    src.close()
    assert live.closed
    with pytest.raises(cam.CameraError, match="closed"):
        src.latest()


@pytest.mark.render
def test_the_sim_source_renders_the_hand_camera_with_the_stamp_taken_before_the_render():
    from lerobot_mp.twin import scene as sc
    from lerobot_mp.twin.runtime import Twin
    from lerobot_mp.twin.workspace import Workspace

    twin = Twin(Workspace(robot="a1x"))
    try:
        with pytest.raises(cam.CameraError, match="scene has no hand camera 'wrist'"):
            cam.SimWristSource(twin)
        twin.configure(hand_cameras=[sc.HandCamera("wrist", T_GRIPPER_CAMERA, FOVY_DEG)])

        events: list[str] = []
        now = [5.0]

        def clock() -> float:
            events.append("clock")
            return now[0]
        render_with = twin.render_with

        def spy(fn):
            events.append("render")
            return render_with(fn)
        twin.render_with = spy

        src = cam.SimWristSource(twin, clock=clock)
        fr = src.latest()
        assert events == ["clock", "render"]
        assert (fr.t, fr.seq) == (5.0, 1)
        assert fr.image.shape == (IMG_H, IMG_W, 3) and fr.image.dtype == np.uint8
        assert np.array_equal(fr.image, render_with(lambda s: s.render_hand("wrist")))
        assert src.latest() is fr                                     # the clock did not move
        now[0] = 5.05
        assert src.latest().seq == 2

        jittered = cam.SimWristSource(twin, clock=clock,
                                      jitter=sc.HandCameraJitter(np.array([0.01, 0.0, 0.0]), np.zeros(3)))
        assert not np.array_equal(jittered.latest().image, fr.image)
        now[0] = 5.10
        assert np.array_equal(src.latest().image, fr.image)           # the nominal mount is back
    finally:
        twin.close()


def test_the_sim_source_turns_a_lost_camera_a_wrong_size_and_a_stopped_render_thread_into_camera_errors():
    """Against a stand-in for the twin: its lock, its scene and `render_with`. Nothing renders."""
    import threading
    from types import SimpleNamespace

    from lerobot_mp.twin import scene as sc

    wrist = sc.build(sc.SceneConfig(A1X, hand_cameras=[sc.HandCamera("wrist", T_GRIPPER_CAMERA, FOVY_DEG)]))
    big = sc.build(sc.SceneConfig(A1X, hand_cameras=[sc.HandCamera("wrist", T_GRIPPER_CAMERA, FOVY_DEG,
                                                                    4 * IMG_W, 4 * IMG_H)]))
    bare = sc.build(sc.SceneConfig(A1X))
    twin = SimpleNamespace(lock=threading.Lock(), scene=big, render_with=lambda fn: fn(twin.scene))

    with pytest.raises(cam.CameraError, match="640x480"):
        cam.SimWristSource(twin)
    twin.scene = wrist
    src = cam.SimWristSource(twin, clock=lambda: 1.0)
    twin.scene = bare                                                 # a rebuild without the camera
    with pytest.raises(cam.CameraError, match="scene has no hand camera 'wrist'"):
        src.latest()

    def stopped(fn):
        raise RuntimeError("watek renderujacy zatrzymany")
    twin.render_with = stopped
    with pytest.raises(cam.CameraError, match="watek renderujacy zatrzymany"):
        src.latest()


# ------------------------------------------------------------------ dump
def test_draw_dump_marks_the_pixel_and_says_when_there_is_none():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 60, (IMG_H, IMG_W, 3), dtype=np.uint8)
    before = image.copy()
    uv = np.array([40.0, 70.0], np.float32)
    out = cam.draw_dump(image, uv, extra={"cube top": np.array([120.0, 30.0])})
    assert out.shape == (4 * IMG_H, 4 * IMG_W, 3) and out.dtype == np.uint8
    assert np.array_equal(image, before)
    # Nearest-neighbour upscale away from the marks.
    assert np.array_equal(out[400:, 400:], np.repeat(np.repeat(image, 4, 0), 4, 1)[400:, 400:])
    cu, cv = 4 * 40.5 - 0.5, 4 * 70.5 - 0.5
    green = (out[:, :, 1] > 150) & (out[:, :, 0] < 100)
    assert green[int(cv), int(cu) - 6:int(cu) + 7].any()                             # cross
    assert green[int(cv) - 2:int(cv) + 3, int(cu + 32) - 2:int(cu + 32) + 3].any()   # the 8 px circle
    magenta = (out[:, :, 0] > 150) & (out[:, :, 2] > 150) & (out[:, :, 1] < 100)
    assert magenta[4 * 30 - 8:4 * 30 + 9, 4 * 120 - 8:4 * 120 + 9].any()

    none = cam.draw_dump(image, np.array([np.nan, np.nan], np.float32))
    green = (none[:, :, 1] > 150) & (none[:, :, 0] < 100)
    assert not green.any()
    white = (none > 200).all(axis=2)
    assert white[:30, :120].any() and not white[30:].any()
