"""Intrynsyki z ChArUco: bramka zaufania K dla sesji bez pochylenia tablicy."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from lerobot_mp.twin.calib.intrinsics import Board, Collector  # noqa: E402


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
