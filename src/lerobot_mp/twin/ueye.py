"""Kamera IDS uEye (USB / GigE) jako zrodlo kadru stanowiska - przez `pyueye` i `libueye_api`.

Zrodlo w rekordzie kamery: `ueye` albo `ueye:<id>`. `id` 0 (domyslne) to pierwsza
wolna kamera - tak numeruje libueye; `id` 1.. to numer kamery z IDS Camera Manager.
OpenCV tej kamery nie otworzy: sterownik IDS nie daje urzadzenia V4L2.

Kamera IDS sama z siebie nie ma zadnej automatyki - w swietle pokoju obraz jest
czarny. Tu wlacza sie automatyczna migawke ograniczona do `MAX_EXPOSURE_MS`
(ruch ramienia zostaje ostry) i automatyczne wzmocnienie: te same ustawienia,
z ktorymi kamera byla kalibrowana (realant, `ueye_view.py`).

Wymaga IDS Software Suite (demon `ueyeusbd` / `ueyeethd` i `libueye_api`) oraz
`pip install -e ".[ueye]"`.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .workspace import CameraRecord

logger = logging.getLogger(__name__)

#: Gorna granica automatycznej migawki [ms] - dluzsza rozmazuje poruszajace sie ramie.
MAX_EXPOSURE_MS = 20.0
#: Po tylu sekundach bez klatki strumien uznaje sie za zerwany (kamera odpieta).
NO_FRAME_TIMEOUT_S = 5.0
_WAIT_MS = 1000


def parse_source(source: str) -> int:
    """`ueye` -> 0, `ueye:3` -> 3. Inne zrodlo to blad."""
    s = str(source).strip()
    if s == "ueye":
        return 0
    if s.startswith("ueye:") and s[5:].isdigit():
        return int(s[5:])
    raise ValueError(f"zrodlo {source!r} to nie kamera uEye (oczekiwane 'ueye' albo 'ueye:<id>')")


def is_ueye(source: str) -> bool:
    return str(source) == "ueye" or str(source).startswith("ueye:")


def _api():
    try:
        from pyueye import ueye
    except ImportError as exc:  # pragma: no cover - zalezy od srodowiska
        raise RuntimeError("brak pakietu pyueye: pip install -e \".[ueye]\" "
                           "(i IDS Software Suite z libueye_api)") from exc
    return ueye


def camera_count() -> int:
    """Ile kamer widzi sterownik IDS (0 takze, gdy demon ueyeusbd nie dziala)."""
    ueye = _api()
    n = ueye.INT()
    ueye.is_GetNumberOfCameras(n)
    return int(n.value)


def probe_devices() -> list[dict]:
    """Kamery uEye, ktore widzi sterownik: [{source, label, width, height, model, serial}].

    Rozdzielczosc to pelny sensor (`is_GetSensorInfo`), po krotkim otwarciu kamery -
    kamera juz uzywana (`dwInUse`) jest pomijana. Bez pyueye / sterownika: [].
    """
    try:
        ueye = _api()
        n = camera_count()
    except Exception as exc:
        logger.debug("uEye: brak sterownika/pyueye: %s", exc)
        return []
    if n == 0:
        return []
    cl = ueye.UEYE_CAMERA_LIST(ueye.UEYE_CAMERA_INFO * n)   # pyueye oddaje gotowa instancje
    if isinstance(cl, type):
        cl = cl()
    cl.dwCount = n
    if ueye.is_GetCameraList(cl) != ueye.IS_SUCCESS:
        return []
    found = []
    for i in range(n):
        info = cl.uci[i]
        if int(info.dwInUse):
            continue
        cam_id = int(info.dwCameraID)
        model = bytes(info.Model).split(b"\0", 1)[0].decode(errors="replace")
        serial = bytes(info.SerNo).split(b"\0", 1)[0].decode(errors="replace")
        width = height = 0
        h = ueye.HIDS(cam_id)
        if ueye.is_InitCamera(h, None) == ueye.IS_SUCCESS:
            try:
                si = ueye.SENSORINFO()
                if ueye.is_GetSensorInfo(h, si) == ueye.IS_SUCCESS:
                    width, height = int(si.nMaxWidth), int(si.nMaxHeight)
            finally:
                ueye.is_ExitCamera(h)
        # Jedna kamera: "ueye" (pierwsza wolna) dziala niezaleznie od numeru z Camera Managera.
        source = "ueye" if n == 1 else f"ueye:{cam_id}"
        found.append(dict(source=source, label=f"{source}: {model} {width}x{height}".rstrip(),
                          width=width, height=height, model=model, serial=serial))
    return found


class UEyeLive:
    """Strumien z jednej kamery uEye - ten sam interfejs, co `cameras._Live`.

    Watek czeka na zdarzenie klatki (`is_WaitEvent`), kopiuje bufor i pamieta
    tylko najnowszy kadr z chwila jego wykonania (`time.monotonic`).
    """

    def __init__(self, record: CameraRecord, max_exposure_ms: float = MAX_EXPOSURE_MS):
        self._ueye = ueye = _api()
        cam_id = parse_source(record.source)
        if camera_count() == 0:
            raise RuntimeError("uEye: sterownik IDS nie widzi zadnej kamery (ueyeusbd dziala? kabel USB?)")
        self.h = ueye.HIDS(cam_id)
        ret = ueye.is_InitCamera(self.h, None)
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"uEye: is_InitCamera({cam_id}) kod {ret} - kamera zajeta albo zly numer")
        self.mem = ueye.c_mem_p()
        self.mem_id = ueye.INT()
        self.bpp = 24
        try:
            self._check(ueye.is_SetColorMode(self.h, ueye.IS_CM_BGR8_PACKED), "is_SetColorMode")
            rect = ueye.IS_RECT()
            self._check(ueye.is_AOI(self.h, ueye.IS_AOI_IMAGE_GET_AOI, rect, ueye.sizeof(rect)), "is_AOI")
            self.width, self.height = int(rect.s32Width), int(rect.s32Height)
            self._check(ueye.is_AllocImageMem(self.h, self.width, self.height, self.bpp, self.mem,
                                              self.mem_id), "is_AllocImageMem")
            self._check(ueye.is_SetImageMem(self.h, self.mem, self.mem_id), "is_SetImageMem")
            self.pitch = ueye.INT()
            w, hgt, bits = ueye.INT(), ueye.INT(), ueye.INT()
            self._check(ueye.is_InquireImageMem(self.h, self.mem, self.mem_id, w, hgt, bits, self.pitch),
                        "is_InquireImageMem")
            if record.fps:
                actual = ueye.double()
                ueye.is_SetFrameRate(self.h, ueye.double(float(record.fps)), actual)
            on, dummy = ueye.double(1.0), ueye.double(0.0)
            ueye.is_SetAutoParameter(self.h, ueye.IS_SET_ENABLE_AUTO_SHUTTER, on, dummy)
            if max_exposure_ms:
                ueye.is_SetAutoParameter(self.h, ueye.IS_SET_AUTO_SHUTTER_MAX, ueye.double(max_exposure_ms),
                                         dummy)
            ueye.is_SetAutoParameter(self.h, ueye.IS_SET_ENABLE_AUTO_GAIN, on, dummy)
            self._check(ueye.is_EnableEvent(self.h, ueye.IS_SET_EVENT_FRAME), "is_EnableEvent")
            self._check(ueye.is_CaptureVideo(self.h, ueye.IS_DONT_WAIT), "is_CaptureVideo")
        except Exception:
            ueye.is_ExitCamera(self.h)
            raise
        if (self.width, self.height) != (record.width, record.height):
            logger.warning("uEye %s: kamera daje %dx%d, rekord ma %dx%d - K liczy sie dla kadru z kamery",
                           record.name, self.width, self.height, record.width, record.height)
        self._lock = threading.Lock()
        self._frame: tuple[np.ndarray, float] | None = None
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"ueye-{record.name}", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ strumien
    def _read_bgr(self) -> np.ndarray:
        ueye = self._ueye
        raw = ueye.get_data(self.mem, self.width, self.height, self.bpp, self.pitch, copy=True)
        rows = np.reshape(raw, (self.height, int(self.pitch.value)))
        return rows[:, : self.width * 3].reshape(self.height, self.width, 3)

    def _run(self) -> None:
        ueye = self._ueye
        last = time.monotonic()
        while not self._stop.is_set():
            if ueye.is_WaitEvent(self.h, ueye.IS_SET_EVENT_FRAME, _WAIT_MS) != ueye.IS_SUCCESS:
                if time.monotonic() - last > NO_FRAME_TIMEOUT_S:
                    self._error = f"brak klatek z uEye od {NO_FRAME_TIMEOUT_S:.0f} s (kamera odpieta?)"
                    return
                continue
            try:
                rgb = np.ascontiguousarray(self._read_bgr()[:, :, ::-1])
            except Exception as exc:                      # bufor zwolniony przy zamykaniu
                if not self._stop.is_set():
                    self._error = f"blad odczytu uEye: {exc}"
                return
            last = time.monotonic()
            with self._lock:
                self._frame = (rgb, last)

    # ------------------------------------------------------------ interfejs jak _Live
    @property
    def error(self) -> str | None:
        return self._error

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    def rgb_t(self) -> tuple[np.ndarray | None, float, str | None]:
        err = self._error
        if err is None and not self.is_running:
            err = "watek kamery uEye zatrzymany"
        if err is not None:
            return None, 0.0, err
        with self._lock:
            frame = self._frame
        if frame is None:
            return None, 0.0, None
        return frame[0], frame[1], None

    def rgb(self) -> np.ndarray | None:
        return self.rgb_t()[0]

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2 * _WAIT_MS / 1000.0)
        ueye = self._ueye
        ueye.is_DisableEvent(self.h, ueye.IS_SET_EVENT_FRAME)
        ueye.is_StopLiveVideo(self.h, ueye.IS_FORCE_VIDEO_STOP)
        ueye.is_FreeImageMem(self.h, self.mem, self.mem_id)
        ueye.is_ExitCamera(self.h)

    def _check(self, ret: int, what: str) -> None:
        if ret != self._ueye.IS_SUCCESS:
            raise RuntimeError(f"uEye: {what} kod {ret}")
