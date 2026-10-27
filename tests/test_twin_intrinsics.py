"""Intrynsyki z ChArUco: bramka zaufania K dla sesji bez pochylenia tablicy."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from machinekind_twin.calib.intrinsics import Board, Collector  # noqa: E402


def _synthetic_session(max_tilt_deg: float, seed: int, n: int = 14) -> Collector:
    """Rogi tablicy rzutowane znanym K (bez renderu): przesuwana po kadrze, pochylona do `max_tilt_deg`."""
    W, H = 640, 480
    Kt = np.array([[610.0, 0, 322], [0, 610.0, 238], [0, 0, 1]])
    dist = np.array([0.08, -0.15, 0, 0, 0.0])
    rng = np.random.default_rng(seed)
    board = Board()
    col = Collector(board, (W, H))
    obj = np.asarray(board.cv().getChessboardCorners(), float)
    for _ in range(3000):
        if len(col.views) >= n:
            break
        axis = np.r_[rng.normal(size=2), 0.0]
        axis /= np.linalg.norm(axis)
        R = cv2.Rodrigues(axis * np.radians(rng.uniform(0, max_tilt_deg)))[0]
        z = rng.uniform(0.25, 0.55)
        c = np.linalg.inv(Kt) @ np.array([rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H, 1.0]) * z
        px, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], c - R @ obj.mean(0), Kt, dist)
        px = px.reshape(-1, 2) + rng.normal(0, 0.15, (len(obj), 2))
        ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        if ok.sum() >= 12:
            col.detect = lambda img, o=obj[ok], p=px[ok]: (o, p)
            col.add(None)
    return col


def test_board_held_parallel_to_the_camera_is_not_trusted():
    """Tablica tylko przesuwana (pochylenie do 2 st.): fx wychodzilo 424-27340 zamiast 610 - i "zaufane"."""
    for seed in range(3):
        res = _synthetic_session(2.0, seed).solve()
        assert not res.trusted and "rownolegla" in res.reason
    # Ta sama sesja z pochylaniem o 30 st.: K w 1%.
    res = _synthetic_session(30.0, 0).solve()
    assert res.trusted, res.reason
    assert abs(res.K[0, 0] - 610.0) < 6.0
    assert res.dropped == []


def test_views_that_disagree_with_the_rest_are_left_out():
    """Two smeared views (4 px noise) pull the fit above 0.6 px; left out, K is back within 1%."""
    col = _synthetic_session(30.0, 0)
    rng = np.random.default_rng(1)
    for i in (3, 7):
        col.views[i].img = col.views[i].img + rng.normal(0, 4.0, col.views[i].img.shape)
    res = col.solve()
    assert res.trusted, res.reason
    assert sorted(i for i, _ in res.dropped) == [3, 7]
    assert res.n_views == len(col.views) - 2 and len(res.per_view_px) == res.n_views
    assert abs(res.K[0, 0] - 610.0) < 6.0
    col.outlier_factor = float("inf")                   # the same session, nothing left out
    res = col.solve()
    assert not res.trusted and "residuum" in res.reason and res.dropped == []


def test_a_board_is_taken_only_once_it_stands_still():
    """A live camera: a view is taken from a frame whose corners barely moved since the previous one."""
    board = Board()
    obj = np.asarray(board.cv().getChessboardCorners(), float)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    px = cv2.projectPoints(obj, np.zeros(3), np.array([-0.126, -0.084, 0.4]), K, None)[0].reshape(-1, 2)
    col = Collector(board, (640, 480), still_px=2.0)
    frames = iter([px, px + 0.5, px + 20.0, px + 20.4])
    col.detect = lambda img: (obj, next(frames))
    ok, why = col.add(None)                             # first sighting: nothing to compare with
    assert not ok and "ruchu" in why and col.moving
    ok, why = col.add(None)                             # held still
    assert ok and not col.moving
    ok, why = col.add(None)                             # moved 28 px
    assert not ok and "ruchu" in why and col.moving
    ok, why = col.add(None)                             # still again, but the same view as before
    assert not ok and "ruchu" not in why and not col.moving
    col.detect = lambda img: None                       # out of view: the next sighting starts over
    assert not col.add(None)[0] and not col.moving
    col.detect = lambda img: (obj, px)
    assert "ruchu" in col.add(None)[1]


def test_the_lens_job_keeps_the_frames_it_took_and_the_fit(tmp_path, monkeypatch):
    import json
    import threading
    import time

    from machinekind_twin.calib import intrinsics
    from machinekind_twin.ui import jobs

    board = Board.of("7x5", 0.024)
    sheet = board.board_image(3000)                     # head-on, about 500 x 360 px
    frame = np.full((480, 640), 255, np.uint8)
    y0, x0 = (480 - sheet.shape[0]) // 2, (640 - sheet.shape[1]) // 2
    frame[y0:y0 + sheet.shape[0], x0:x0 + sheet.shape[1]] = sheet
    frame = np.dstack([frame] * 3)

    class Hub:
        def frame(self, name):
            return frame.copy()                         # a new frame each time, as from a camera

    fit = intrinsics.Result(np.eye(3), np.zeros(5), 0.4, [0.3, 0.5], 2, 0.3, (640, 480), False, "za malo",
                            [(1, 2.5)])
    monkeypatch.setattr(intrinsics.Collector, "solve", lambda self: fit)
    out = tmp_path / "kam1-sesja"
    solve = threading.Event()
    job = jobs.Job("intrynsyki")
    job.start(lambda j: jobs.run_intrinsics(j, Hub(), "kam1", board, (640, 480), period=0.01, solve=solve,
                                            save_to=out))
    deadline = time.monotonic() + 5.0
    while not (out / "kadr-00.png").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    solve.set()
    assert job.wait(5.0) and job.state == jobs.DONE and job.result is fit
    assert cv2.imread(str(out / "kadr-00.png")).shape == (480, 640, 3)
    data = json.loads((out / "sesja.json").read_text())
    assert data["board"]["name"] == "7x5" and data["board"]["square_m"] == pytest.approx(0.024)
    assert [(v["view"], v["px"], v["left_out"]) for v in data["views"]] == [(0, 0.3, False), (1, 2.5, True),
                                                                          (2, 0.5, False)]
