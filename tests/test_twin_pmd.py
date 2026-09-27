"""Kamera glebi pmd jako kamera stanowiska - na udawanym SDK (prawdziwa kamera: `pmd_view.py`).

K odczytane z xyz ma trafiac w K, ktorym te xyz policzono; strumien `pmd[:serial]` ma isc przez
`CameraHub` jak kazda inna kamera (kadr = amplituda, K do rekordu przy otwarciu); probe ma
oddawac K; zrodla `ueye`/USB nie moga trafic do sterownika pmd.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from lerobot_mp.twin import cameras, pmd
from lerobot_mp.twin.workspace import CameraRecord, Workspace

K_TRUE = np.array([[104.0, 0.0, 81.95], [0.0, 103.2, 59.46], [0.0, 0.0, 1.0]])
W, H = 165, 120


def synthetic_xyz(K: np.ndarray, depth: float = 0.8) -> np.ndarray:
    """xyz, jakie SDK oddaje dla plaskiej sciany w `depth` przed kamera o K."""
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.dstack([x, y, np.full_like(x, depth)]).astype(np.float32)


class FakeNano:
    """`pmd.CamboardNano` bez sprzetu: sciana 0,8 m, amplituda z gradientem, kilka pikseli oflagowanych."""

    opened: list[FakeNano] = []

    def __init__(self, sdk, serial="", calib=None, integration_us=None, illumination=True):
        if serial and serial != "10340935":
            raise pmd.PMDError("pmdOpen failed (4098): no such camera")
        self.serial, self.width, self.height, self.calibrated = "10340935", W, H, "Yes"
        self.closed, self.updates = False, 0
        FakeNano.opened.append(self)

    def update(self):
        self.updates += 1
        time.sleep(0.005)

    def xyz(self):
        return synthetic_xyz(K_TRUE)

    def distances(self):
        return np.linalg.norm(self.xyz(), axis=2)

    def amplitudes(self):
        return np.tile(np.linspace(0, 900, W, dtype=np.float32), (H, 1))

    def flags(self):
        f = np.zeros((H, W), np.uint32)
        f[0, :5] = pmd.PMD_FLAG_SATURATED
        return f

    def set_averaging(self, n):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def fake_sdk(monkeypatch):
    FakeNano.opened = []
    monkeypatch.setattr(pmd, "CamboardNano", FakeNano)
    monkeypatch.setattr(pmd, "shared_sdk", lambda sdk_dir=None: object())
    yield


def test_K_read_from_xyz_matches_the_K_that_made_them():
    K = pmd.intrinsics_from_xyz(synthetic_xyz(K_TRUE))
    assert np.allclose(K, K_TRUE, atol=1e-3)
    # Z bliska tez, i przy pikselach bez punktu (z = 0 - SDK tak oznacza brak pomiaru).
    xyz = synthetic_xyz(K_TRUE, depth=0.25)
    xyz[:10, :, :] = 0.0
    assert np.allclose(pmd.intrinsics_from_xyz(xyz), K_TRUE, atol=1e-3)
    with pytest.raises(ValueError):
        pmd.intrinsics_from_xyz(np.zeros((H, W, 3)))


def test_amplitude_becomes_a_grey_rgb_frame_with_flagged_pixels_darkened():
    amp = np.tile(np.linspace(0, 900, W, dtype=np.float32), (H, 1))
    flags = np.zeros((H, W), np.uint32)
    flags[3, 100] = pmd.PMD_FLAG_SATURATED
    rgb = pmd.amplitude_rgb(amp, flags)
    assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
    assert rgb[5, 0, 0] == 0 and rgb[5, W - 1, 0] == 255 and (rgb[5, :, 0] == rgb[5, :, 2]).all()
    assert rgb[3, 100, 0] < rgb[5, 100, 0] * 0.5                 # nasycony piksel sciemniony
    assert np.all(np.diff(rgb[5, :, 0].astype(int)) >= 0)         # monotoniczna krzywa tonalna


def test_source_parsing_keeps_pmd_apart_from_usb_and_ueye():
    assert pmd.is_pmd("pmd") and pmd.is_pmd("pmd:10340935")
    assert not pmd.is_pmd("ueye") and not pmd.is_pmd("/dev/video2") and not pmd.is_pmd("0")
    assert pmd.parse_source("pmd") == "" and pmd.parse_source("pmd:10340935") == "10340935"
    with pytest.raises(ValueError):
        pmd.parse_source("ueye:1")
    with pytest.raises(ValueError):
        pmd.parse_source("pmd:")


def test_probe_opens_the_camera_once_and_returns_its_factory_K(fake_sdk):
    found = pmd.probe_devices()
    assert len(found) == 1 and found[0]["source"] == "pmd:10340935"
    assert (found[0]["width"], found[0]["height"]) == (W, H)
    assert np.allclose(found[0]["K"], K_TRUE, atol=1e-3)
    assert len(FakeNano.opened) == 1 and FakeNano.opened[0].closed
    # Bez SDK: pusta lista, bez wyjatku (panel szuka kamer takze tam, gdzie pmd nie ma).
    pmd.shared_sdk = lambda sdk_dir=None: (_ for _ in ()).throw(pmd.PMDError("PMDSDK not found"))
    assert pmd.probe_devices() == []


def test_pmd_camera_streams_through_the_hub_like_any_other_camera(fake_sdk):
    ws = Workspace()
    ws.add_camera(CameraRecord("pmd1", "pmd:10340935", W, H))
    ws.add_camera(CameraRecord("pmd_zla", "pmd:1", W, H))
    hub = cameras.CameraHub(ws, stale_after=2.0)
    try:
        hub.sync()
        rec = ws.camera("pmd1")
        assert rec.K is not None and rec.intrinsics_from == "pmd" and rec.intrinsics_problem() == ""
        assert np.allclose(rec.K, K_TRUE, atol=1e-3)
        assert "pmdOpen failed" in (hub.error("pmd_zla") or "")
        deadline = time.monotonic() + 3.0
        img = None
        while img is None and time.monotonic() < deadline:
            img, t = hub.frame_t("pmd1")
            time.sleep(0.02)
        assert img is not None and img.shape == (H, W, 3) and t > 0
        assert hub.error("pmd1") is None
        live = hub._live["pmd1"]
        dist, xyz, flags, t2 = live.depth_t()
        assert dist.shape == (H, W) and xyz.shape == (H, W, 3) and flags.shape == (H, W) and t2 > 0
        assert abs(float(dist[H // 2, W // 2]) - 0.8) < 0.01
        assert ws.camera("pmd1").view() is None          # bez pozy nie ma jej w scenie - poza z tagow
    finally:
        hub.close()
    assert all(c.closed for c in FakeNano.opened if c.serial == "10340935")
    # Zwykle zrodla nadal ida przez OpenCV / uEye, nie przez sterownik pmd.
    assert not pmd.is_pmd(CameraRecord("brio", "/dev/video4").source)
