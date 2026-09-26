"""Kamera IDS uEye jako zrodlo kadru stanowiska - na atrapie pyueye, bez sprzetu.

Atrapa udaje jedna kamere 8x4 BGR8: sprawdza parsowanie zrodla, otwarcie przez
CameraHub, kolejnosc kanalow (BGR z kamery -> RGB w kadrze), zerwanie strumienia
i czytelny blad, gdy sterownik nie widzi kamery.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import types

import numpy as np
import pytest

from lerobot_mp.twin import cameras as cams
from lerobot_mp.twin import ueye as ue
from lerobot_mp.twin.workspace import CameraRecord, Workspace

W, H = 8, 4


class _FakeUEye(types.ModuleType):
    """Minimalne `pyueye.ueye`: jedna kamera, klatka na zadanie (`push_frame`)."""

    IS_SUCCESS = 0
    IS_CM_BGR8_PACKED = 1
    IS_AOI_IMAGE_GET_AOI = 2
    IS_DONT_WAIT = 0
    IS_FORCE_VIDEO_STOP = 0x4000
    IS_SET_EVENT_FRAME = 2
    IS_SET_ENABLE_AUTO_SHUTTER = 0x8802
    IS_SET_AUTO_SHUTTER_MAX = 0x8004
    IS_SET_ENABLE_AUTO_GAIN = 0x8800
    INT = ctypes.c_int
    HIDS = ctypes.c_uint
    double = ctypes.c_double
    c_mem_p = ctypes.c_char_p

    class IS_RECT(ctypes.Structure):
        _fields_ = [("s32X", ctypes.c_int), ("s32Y", ctypes.c_int), ("s32Width", ctypes.c_int),
                    ("s32Height", ctypes.c_int)]

    class SENSORINFO(ctypes.Structure):
        _fields_ = [("nMaxWidth", ctypes.c_uint), ("nMaxHeight", ctypes.c_uint)]

    class UEYE_CAMERA_INFO(ctypes.Structure):
        _fields_ = [("dwCameraID", ctypes.c_uint), ("dwDeviceID", ctypes.c_uint),
                    ("dwSensorID", ctypes.c_uint), ("dwInUse", ctypes.c_uint),
                    ("SerNo", ctypes.c_char * 16), ("Model", ctypes.c_char * 16)]

    def __init__(self, n_cameras: int = 1, init_ret: int = 0):
        super().__init__("pyueye.ueye")
        self.n_cameras, self.init_ret = n_cameras, init_ret
        self.frame_ready = threading.Event()
        self.bgr = np.zeros((H, W, 3), np.uint8)
        self.calls: list[str] = []
        self.pitch = W * 3 + 4                       # wiersz z wyrownaniem, jak w prawdziwym buforze

    def UEYE_CAMERA_LIST(self, uci):
        info = self.UEYE_CAMERA_INFO

        class L(ctypes.Structure):
            _fields_ = [("dwCount", ctypes.c_uint), ("uci", info * self.n_cameras)]
        return L()                                    # jak pyueye: instancja, nie klasa

    def push_frame(self, bgr: np.ndarray) -> None:
        self.bgr = bgr
        self.frame_ready.set()

    # -- API uzywane przez twin.ueye --
    def sizeof(self, x):
        return ctypes.sizeof(x)

    def is_GetNumberOfCameras(self, n):
        n.value = self.n_cameras
        return 0

    def is_GetCameraList(self, cl):
        for i in range(self.n_cameras):
            cl.uci[i].dwCameraID, cl.uci[i].Model, cl.uci[i].SerNo = i + 1, b"UI-1240ML-C", b"4103"
        return 0

    def is_InitCamera(self, h, wnd):
        self.calls.append("init")
        return self.init_ret

    def is_ExitCamera(self, h):
        self.calls.append("exit")
        return 0

    def is_GetSensorInfo(self, h, si):
        si.nMaxWidth, si.nMaxHeight = W, H
        return 0

    def is_SetColorMode(self, h, mode):
        return 0

    def is_AOI(self, h, cmd, rect, size):
        rect.s32Width, rect.s32Height = W, H
        return 0

    def is_AllocImageMem(self, h, w, hgt, bpp, mem, mem_id):
        return 0

    def is_SetImageMem(self, h, mem, mem_id):
        return 0

    def is_InquireImageMem(self, h, mem, mem_id, w, hgt, bits, pitch):
        pitch.value = self.pitch
        return 0

    def is_SetFrameRate(self, h, fps, actual):
        return 0

    def is_SetAutoParameter(self, h, what, a, b):
        self.calls.append(f"auto:{what:#x}")
        return 0

    def is_EnableEvent(self, h, which):
        return 0

    def is_DisableEvent(self, h, which):
        return 0

    def is_CaptureVideo(self, h, wait):
        return 0

    def is_StopLiveVideo(self, h, mode):
        return 0

    def is_FreeImageMem(self, h, mem, mem_id):
        return 0

    def is_WaitEvent(self, h, which, timeout_ms):
        ok = self.frame_ready.wait(timeout_ms / 1000.0)
        self.frame_ready.clear()
        return 0 if ok else -1

    def get_data(self, mem, w, hgt, bits, pitch, copy):
        rows = np.zeros((hgt, self.pitch), np.uint8)
        rows[:, : w * 3] = self.bgr.reshape(hgt, w * 3)
        return rows.ravel()


@pytest.fixture
def fake(monkeypatch):
    def install(**kw):
        mod = _FakeUEye(**kw)
        pkg = types.ModuleType("pyueye")
        pkg.ueye = mod
        monkeypatch.setitem(sys.modules, "pyueye", pkg)
        monkeypatch.setitem(sys.modules, "pyueye.ueye", mod)
        return mod
    return install


def test_source_parsing():
    assert ue.parse_source("ueye") == 0 and ue.parse_source("ueye:3") == 3
    assert ue.is_ueye("ueye:1") and not ue.is_ueye("1") and not ue.is_ueye("sim")
    with pytest.raises(ValueError):
        ue.parse_source("2")


def test_probe_lists_the_camera_with_its_sensor_size(fake):
    fake()
    found = ue.probe_devices()
    assert found == [dict(source="ueye", label=f"ueye: UI-1240ML-C {W}x{H}", width=W, height=H,
                          model="UI-1240ML-C", serial="4103")]
    assert any(d["source"] == "ueye" for d in cams.probe_all() if d["source"].startswith("ueye"))


def test_hub_opens_ueye_and_delivers_rgb_frames(fake):
    mod = fake()
    ws = Workspace()
    ws.add_camera(CameraRecord("ids", "ueye", W, H))
    hub = cams.CameraHub(ws, stale_after=5.0)
    hub.sync()
    try:
        assert hub.error("ids") is None and isinstance(hub._live["ids"], ue.UEyeLive)
        assert hub.frame("ids") is None                    # jeszcze bez klatki - nie blad
        bgr = np.zeros((H, W, 3), np.uint8)
        bgr[..., 0], bgr[..., 2] = 200, 50                 # B=200, R=50
        mod.push_frame(bgr)
        for _ in range(50):
            img, t = hub.frame_t("ids")
            if img is not None:
                break
            threading.Event().wait(0.02)
        assert img is not None and img.shape == (H, W, 3) and t > 0
        assert img[0, 0].tolist() == [50, 0, 200]          # RGB
        assert any(c.startswith("auto:") for c in mod.calls)   # automatyczna migawka wlaczona
    finally:
        hub.close()
    assert "exit" in mod.calls


def test_hub_reports_a_missing_camera_instead_of_crashing(fake):
    fake(n_cameras=0)
    ws = Workspace()
    ws.add_camera(CameraRecord("ids", "ueye"))
    hub = cams.CameraHub(ws)
    hub.sync()
    assert "nie widzi zadnej kamery" in (hub.error("ids") or "")
    assert hub.frame("ids") is None


def test_init_failure_releases_the_handle(fake):
    mod = fake(init_ret=3)
    with pytest.raises(RuntimeError, match="is_InitCamera"):
        ue.UEyeLive(CameraRecord("ids", "ueye:2"))
    assert mod.calls.count("exit") == 0                    # nie bylo czego zwalniac


def test_lost_stream_is_reported(fake, monkeypatch):
    fake()
    monkeypatch.setattr(ue, "NO_FRAME_TIMEOUT_S", 0.0)
    monkeypatch.setattr(ue, "_WAIT_MS", 10)
    live = ue.UEyeLive(CameraRecord("ids", "ueye"))
    try:
        for _ in range(100):
            if live.error:
                break
            threading.Event().wait(0.01)
        assert live.error and "brak klatek" in live.error
        assert live.rgb_t() == (None, 0.0, live.error)
    finally:
        live.close()
