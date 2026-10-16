"""Stanowisko blizniaka: zaufanie do pozy kamery po kalibracji (bramka percepcji).

Bez renderu i bez sprzetu - to jest warstwa danych, na ktorej stoi reszta.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from conftest import BOT  # noqa: E402

from lerobot_mp.twin.calib.handeye import CameraFit, Fit  # noqa: E402
from lerobot_mp.twin.workspace import CameraRecord, Workspace, nominal_K  # noqa: E402


def charuco_camera(name: str = "front", trusted: bool = True) -> CameraRecord:
    return CameraRecord(name, K=nominal_K(640, 480).tolist(), dist=[0.1, -0.05, 0.0, 0.0, 0.0],
                        intrinsics_from="szachownica",
                        intrinsics_info={"rms_px": 0.3, "trusted": trusted, "reason": "" if trusted else "malo kadrow"})


def trusted_fit(name: str = "front") -> Fit:
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.2]
    return Fit({name: CameraFit(T, 0.18, 0.6, 30, 12.5, trusted=True, reason="")})


def test_real_camera_pose_is_not_trusted_without_a_trusted_charuco_K():
    """Dopasowanie z nominalnym K przechodzi prog residuum, a poza jest przesunieta o centymetry."""
    ws = Workspace(robot=BOT)
    ws.add_camera(CameraRecord("nominalna"))                       # krok 1 pominiety
    ws.add_camera(charuco_camera("slaba", trusted=False))           # sesja ChArUco niezaufana
    ws.add_camera(CameraRecord("sym", "sim", K=nominal_K(640, 480).tolist(), intrinsics_from="symulacja"))
    fit = Fit({n: trusted_fit(n).cameras[n] for n in ("nominalna", "slaba", "sym")})
    ws.apply_fit(fit)
    assert not ws.camera("nominalna").trusted and "nominalne" in ws.camera("nominalna").calibration["reason"]
    assert not ws.camera("slaba").trusted and "malo kadrow" in ws.camera("slaba").calibration["reason"]
    assert ws.camera("sym").trusted                                 # symulowana zna swoje K
