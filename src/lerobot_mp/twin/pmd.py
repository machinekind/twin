"""Kamera glebi pmd[vision] CamBoard nano jako kamera stanowiska (zrodlo `pmd` albo `pmd:<serial>`).

    from lerobot_mp.twin import pmd
    sdk = pmd.PMDSDK(*pmd.find_sdk())
    cam = pmd.CamboardNano(sdk, "", "pmd/Calibration_file")
    cam.update(); dist, amp, xyz = cam.distances(), cam.amplitudes(), cam.xyz()
    K = pmd.intrinsics_from_xyz(xyz)             # K obiektywu z kalibracji fabrycznej kamery

CamBoard nano (USB 1c28:c007) to urzadzenie bez sterownika V4L2 - czyta sie je przez
wlasnosciowe PMDSDK2 (`libpmdaccess2.so` + wtyczki `camboardnano.L64.pap` /
`camboardnanoproc.L64.ppp` z plyty kamery) przez ctypes, bez kompilacji. SDK szukane w
`--sdk-dir` / `$PMDDIR`, `pmd/Linux/Linux64bit/PMDSDK` (rozpakowany Linux64bit.tar.gz;
w klonie wystarczy dowiazanie `pmd/` do katalogu z plyty, jest w .gitignore),
`/usr/local/pmd` i `./PMDSDK`. Wtyczka potrzebuje starego libusb-0.1.so.4 - kopia obok
libpmdaccess2.so jest ladowana sama. Regula udev `10-pmd-ubuntu.rules` z SDK daje dostep
zwyklemu uzytkownikowi. Kalibracja `<serial>.dat` z `pmd/Calibration_file`.

Jako kamera stanowiska (`PMDLive`, ten sam interfejs co `cameras._Live`): kadr RGB to
obraz AMPLITUDY (szary, krzywa pierwiastkowa) - prawdziwy obraz natezenia przez ten sam
obiektyw, wiec K i poza sa jego. Odleglosci, xyz i flagi ostatniej klatki sa obok
(`depth_t`) do pozniejszego wlaczenia chmury punktow. K nie jest nominalne: xyz z SDK to
dokladny model otworkowy (u = fx x/z + cx, residuum 0,002 px), wiec `intrinsics_from_xyz`
odczytuje K z kalibracji fabrycznej kamery przy otwarciu (`intrinsics_from = "pmd"`).
Poza: z tagow na obudowie (`tagbody`), przepisywana do rekordu przez panel.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import threading
import time

import cv2
import numpy as np

from .workspace import CameraRecord

logger = logging.getLogger(__name__)

PMD_OK = 0

# Bity flag piksela (pmdsdk2.h). Kazdy z czterech pierwszych = piksel nie do uzytku.
PMD_FLAG_INVALID = 0x01
PMD_FLAG_SATURATED = 0x02
PMD_FLAG_INCONSISTENT = 0x04
PMD_FLAG_LOW_SIGNAL = 0x08
PMD_FLAG_SBI_ACTIVE = 0x10
BAD_PIXEL_MASK = PMD_FLAG_INVALID | PMD_FLAG_SATURATED | PMD_FLAG_INCONSISTENT | PMD_FLAG_LOW_SIGNAL

PMDHandle = ctypes.c_uint

#: Proximity dla pmdGetValidIntegrationTime.
AT_LEAST, AT_MOST, CLOSE_TO = 0, 1, 2

#: Matryca CamBoard nano, jednoznaczny zasieg przy 30 MHz.
NANO_RESOLUTION = (165, 120)
WRAP_M = 5.0
#: Katalog z kalibracja per numer seryjny (`<serial>.dat`) - jak w `pmd_view.py`.
DEFAULT_CALIB = "pmd/Calibration_file"

SDK_ROOTS = [os.environ.get("PMDDIR"), "pmd/Linux/Linux64bit/PMDSDK", "/usr/local/pmd", "PMDSDK"]


class PMDImageData(ctypes.Structure):
    _fields_ = [
        ("subType", ctypes.c_uint),
        ("numColumns", ctypes.c_uint),
        ("numRows", ctypes.c_uint),
        ("numSubImages", ctypes.c_uint),
        ("integrationTime", ctypes.c_int * 4),
        ("modulationFrequency", ctypes.c_int * 4),
        ("offset", ctypes.c_int * 4),
        ("pixelAspectRatio", ctypes.c_int),
        ("pixelOrigin", ctypes.c_int),
        ("timeStampHi", ctypes.c_int),
        ("timeStampLo", ctypes.c_uint),
        ("reserved", ctypes.c_char * 24),
        ("userData0", ctypes.c_uint),
    ]


class _DDUnion(ctypes.Union):
    _fields_ = [("img", PMDImageData), ("fill", ctypes.c_char * 108)]


class PMDDataDescription(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("PID", ctypes.c_uint),
        ("DID", ctypes.c_uint),
        ("type", ctypes.c_uint),
        ("size", ctypes.c_uint),
        ("subHeaderType", ctypes.c_uint),
        ("u", _DDUnion),
        ("_pad", ctypes.c_char * 512),  # zapas, gdyby nowsze SDK powiekszylo strukture
    ]


class PMDError(RuntimeError):
    pass


def find_sdk(sdk_dir: str | None = None) -> tuple[str, str]:
    """(sciezka libpmdaccess2.so, katalog wtyczek) albo PMDError z podpowiedzia."""
    roots = [d for d in [sdk_dir] + SDK_ROOTS if d]
    for root in roots:
        libs = glob.glob(os.path.join(root, "*", "libpmdaccess2.so*")) + glob.glob(
            os.path.join(root, "libpmdaccess2.so*"))
        plugins = glob.glob(os.path.join(root, "*", "camboardnano.L64.pap")) + glob.glob(
            os.path.join(root, "camboardnano.L64.pap"))
        if libs and plugins:
            return os.path.abspath(sorted(libs)[0]), os.path.abspath(os.path.dirname(plugins[0]))
    raise PMDError(
        "PMDSDK not found (looked in %s).\n"
        "Unpack pmd's Linux64bit.tar.gz and point --sdk-dir or $PMDDIR at its PMDSDK folder "
        "(containing bin/libpmdaccess2.so and camboardnano.L64.pap / camboardnanoproc.L64.ppp)."
        % ", ".join(roots))


_SDK: PMDSDK | None = None
_SDK_LOCK = threading.Lock()


def shared_sdk(sdk_dir: str | None = None) -> PMDSDK:
    """Jedna zaladowana biblioteka na proces (ctypes.CDLL drugi raz to ta sama .so)."""
    global _SDK
    with _SDK_LOCK:
        if _SDK is None:
            _SDK = PMDSDK(*find_sdk(sdk_dir))
        return _SDK


class PMDSDK:
    """ctypes do garsci funkcji pmdsdk2.h, ktorych potrzebuje podglad i strumien."""

    def __init__(self, lib_path: str, plugin_dir: str):
        # Wtyczka camboardnano z 2013 r. potrzebuje libusb-0.1.so.4, ktorego nowe dystrybucje nie maja.
        # Kopia obok SDK jest ladowana globalnie, zeby dlopen wtyczki ja znalazl.
        for usb in glob.glob(os.path.join(os.path.dirname(lib_path), "libusb-0.1.so*")) + glob.glob(
                os.path.join(plugin_dir, "libusb-0.1.so*")):
            ctypes.CDLL(usb, mode=ctypes.RTLD_GLOBAL)
            break
        self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        self.plugin_dir = plugin_dir
        L = self.lib
        c_char_p, c_size_t, c_uint = ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint
        L.pmdOpen.argtypes = [ctypes.POINTER(PMDHandle), c_char_p, c_char_p, c_char_p, c_char_p]
        L.pmdClose.argtypes = [PMDHandle]
        L.pmdUpdate.argtypes = [PMDHandle]
        L.pmdGetLastError.argtypes = [PMDHandle, c_char_p, c_size_t]
        L.pmdGetSourceDataDescription.argtypes = [PMDHandle, ctypes.POINTER(PMDDataDescription)]
        for name in ("pmdGetDistances", "pmdGetAmplitudes", "pmdGet3DCoordinates"):
            getattr(L, name).argtypes = [PMDHandle, ctypes.c_void_p, c_size_t]
        L.pmdGetFlags.argtypes = [PMDHandle, ctypes.c_void_p, c_size_t]
        L.pmdSourceCommand.argtypes = [PMDHandle, c_char_p, c_size_t, c_char_p]
        L.pmdProcessingCommand.argtypes = [PMDHandle, c_char_p, c_size_t, c_char_p]
        L.pmdGetIntegrationTime.argtypes = [PMDHandle, ctypes.POINTER(c_uint), c_uint]
        L.pmdSetIntegrationTime.argtypes = [PMDHandle, c_uint, c_uint]
        L.pmdGetValidIntegrationTime.argtypes = [PMDHandle, ctypes.POINTER(c_uint), c_uint, ctypes.c_int, c_uint]

    def plugin(self, name: str) -> bytes:
        # SDK samo dokleja .L64.pap / .ppp; pelna sciezka uniezaleznia od katalogu biezacego.
        return os.path.join(self.plugin_dir, name).encode()


class CamboardNano:
    def __init__(self, sdk: PMDSDK, serial: str = "", calib: str | None = None,
                 integration_us: int | None = None, illumination: bool = True):
        self.sdk, self.L = sdk, sdk.lib
        self.h = PMDHandle(0)
        ret = self.L.pmdOpen(ctypes.byref(self.h), sdk.plugin("camboardnano"), serial.encode(),
                             sdk.plugin("camboardnanoproc"), b"")
        if ret != PMD_OK:
            raise PMDError("pmdOpen failed (%d): %s" % (ret, self._last_error()))
        try:
            self._setup(serial, calib, integration_us, illumination)
        except PMDError:
            self.close()
            raise

    def _setup(self, serial, calib, integration_us, illumination):
        self.serial = self.source_command("GetSerialNumber")
        if serial and self.serial != serial:
            raise PMDError("opened camera %s, not the requested %s" % (self.serial, serial))
        if calib and os.path.isdir(calib):
            calib = os.path.join(calib, self.serial + ".dat")
            if not os.path.isfile(calib):
                logger.warning("pmd: brak pliku kalibracji %s (SDK bierze cwd / domyslne)", calib)
                calib = None
        if calib:
            self.source_command("LoadCalibrationData " + os.path.abspath(calib))
        self.calibrated = self.source_command("IsCalibrationDataLoaded")
        # Ustawienia obnizajace prad LED ida przed pierwsza klatka: oswietlacz startuje z pierwszym pmdUpdate.
        if integration_us is not None:
            self.set_integration_time(integration_us)
        if not illumination:
            self.source_command("SetIllumination 0 0")
        self.check(self.L.pmdUpdate(self.h), "pmdUpdate")
        dd = PMDDataDescription()
        self.check(self.L.pmdGetSourceDataDescription(self.h, ctypes.byref(dd)), "pmdGetSourceDataDescription")
        self.width, self.height = int(dd.img.numColumns), int(dd.img.numRows)
        if not (0 < self.width < 4096 and 0 < self.height < 4096):
            self.width, self.height = NANO_RESOLUTION
        self.n = self.width * self.height

    # --- SDK ----------------------------------------------------------------
    def _last_error(self) -> str:
        buf = ctypes.create_string_buffer(256)
        self.L.pmdGetLastError(self.h, buf, len(buf))
        return buf.value.decode(errors="replace")

    def check(self, ret: int, what: str) -> None:
        if ret != PMD_OK:
            raise PMDError("%s failed (%d): %s" % (what, ret, self._last_error()))

    def source_command(self, cmd: str) -> str:
        buf = ctypes.create_string_buffer(256)
        self.check(self.L.pmdSourceCommand(self.h, buf, len(buf), cmd.encode()), cmd)
        return buf.value.decode(errors="replace").strip()

    def processing_command(self, cmd: str) -> str:
        buf = ctypes.create_string_buffer(256)
        self.check(self.L.pmdProcessingCommand(self.h, buf, len(buf), cmd.encode()), cmd)
        return buf.value.decode(errors="replace").strip()

    def _get_array(self, fn, dtype, per_pixel: int = 1) -> np.ndarray:
        shape = (self.height, self.width) if per_pixel == 1 else (self.height, self.width, per_pixel)
        arr = np.empty(shape, dtype)
        self.check(fn(self.h, arr.ctypes.data, arr.nbytes), fn.__name__)
        return arr

    # --- dane ---------------------------------------------------------------
    def update(self) -> None:
        self.check(self.L.pmdUpdate(self.h), "pmdUpdate")

    def distances(self) -> np.ndarray:
        """Odleglosc radialna na piksel, float32 [m]."""
        return self._get_array(self.L.pmdGetDistances, np.float32)

    def amplitudes(self) -> np.ndarray:
        return self._get_array(self.L.pmdGetAmplitudes, np.float32)

    def flags(self) -> np.ndarray:
        return self._get_array(self.L.pmdGetFlags, np.uint32)

    def xyz(self) -> np.ndarray:
        """(H, W, 3) punkt na piksel w ukladzie kamery (OpenCV: x w prawo, y w dol, z do przodu)."""
        return self._get_array(self.L.pmdGet3DCoordinates, np.float32, 3)

    # --- ustawienia ---------------------------------------------------------
    def get_integration_time(self) -> int:
        t = ctypes.c_uint()
        self.check(self.L.pmdGetIntegrationTime(self.h, ctypes.byref(t), 0), "pmdGetIntegrationTime")
        return int(t.value)

    def set_integration_time(self, us: int) -> int:
        valid = ctypes.c_uint()
        self.check(self.L.pmdGetValidIntegrationTime(self.h, ctypes.byref(valid), 0, CLOSE_TO, int(us)),
                   "pmdGetValidIntegrationTime")
        self.check(self.L.pmdSetIntegrationTime(self.h, 0, valid), "pmdSetIntegrationTime")
        return int(valid.value)

    def set_averaging(self, frames: int) -> None:
        if frames:
            self.processing_command("SetAveraging On")
            self.processing_command("SetAveragingFrames %d" % frames)
        else:
            self.processing_command("SetAveraging Off")

    def set_bilateral_filter(self, enable: bool) -> None:
        self.processing_command("SetBilateralFilter %s" % ("On" if enable else "Off"))

    def close(self) -> None:
        if self.h.value:
            self.L.pmdClose(self.h)
            self.h = PMDHandle(0)


def unwrap(dist: np.ndarray) -> np.ndarray:
    """Odleglosci tuz pod zawinieciem 5 m (szum wokol 0 m) jako ujemne."""
    d = dist.copy()
    d[d > WRAP_M * 0.9] -= WRAP_M
    return d


# ---------------------------------------------------------------- K i obraz
def intrinsics_from_xyz(xyz: np.ndarray, min_z: float = 0.05) -> np.ndarray:
    """K obiektywu z tego, jak SDK rzutuje piksele w przestrzen: u = fx x/z + cx, v = fy y/z + cy.

    xyz z kalibracji fabrycznej to dokladny model otworkowy (residuum 0,002 px na CamBoard
    nano 10340935), wiec najmniejsze kwadraty po pikselach z z > `min_z` daja K bez
    szachownicy. Punkty licza sie takze dla pikseli oflagowanych - geometria nie zalezy
    od sygnalu - wiec flag sie tu nie potrzebuje.
    """
    xyz = np.asarray(xyz, np.float64)
    H, W = xyz.shape[:2]
    v, u = np.mgrid[0:H, 0:W]
    z = xyz[..., 2]
    m = np.isfinite(xyz).all(axis=2) & (z > min_z)
    if m.sum() < 16:
        raise ValueError("za malo pikseli z z > %.2f m, zeby dopasowac K" % min_z)
    x, y, z = xyz[..., 0][m], xyz[..., 1][m], z[m]
    fx, cx = np.linalg.lstsq(np.column_stack([x / z, np.ones_like(x)]), u[m].astype(float), rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.column_stack([y / z, np.ones_like(y)]), v[m].astype(float), rcond=None)[0]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def amplitude_rgb(amp: np.ndarray, flags: np.ndarray | None = None) -> np.ndarray:
    """Obraz amplitudy jako RGB uint8: krzywa pierwiastkowa (ciemne partie widoczne),
    skala do 99,5 percentyla; piksele oflagowane (nasycone, niespojne) sciemnione."""
    a = np.sqrt(np.maximum(np.asarray(amp, np.float32), 0))
    top = float(np.percentile(a, 99.5)) if a.size else 1.0
    g = (255 * np.clip(a / max(top, 1e-6), 0, 1)).astype(np.uint8)
    if flags is not None:
        bad = (np.asarray(flags) & (PMD_FLAG_SATURATED | PMD_FLAG_INCONSISTENT)) != 0
        g[bad] = (g[bad] * 0.35).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(g[:, :, None], 3, axis=2))


# ---------------------------------------------------------------- zrodlo `pmd`
def is_pmd(source: str) -> bool:
    return str(source) == "pmd" or str(source).startswith("pmd:")


def parse_source(source: str) -> str:
    """`pmd` -> "" (pierwsza kamera), `pmd:10340935` -> numer seryjny. Inne zrodlo to blad."""
    s = str(source).strip()
    if s == "pmd":
        return ""
    if s.startswith("pmd:") and s[4:]:
        return s[4:]
    raise ValueError(f"zrodlo {source!r} to nie kamera pmd (oczekiwane 'pmd' albo 'pmd:<serial>')")


def probe_devices(calib: str = DEFAULT_CALIB) -> list[dict]:
    """Kamera CamBoard nano do dodania: [{source, label, width, height, K, serial, model}] albo [].

    Otwiera kamere na chwile (jedna klatka, K z xyz) i zamyka. Bez SDK, bez kamery albo z
    kamera zajeta przez inny proces (takze przez ten panel, gdy juz jest w stanowisku): [].
    """
    try:
        sdk = shared_sdk()
    except PMDError as exc:
        logger.debug("pmd: %s", exc)
        return []
    try:
        cam = CamboardNano(sdk, "", calib)
    except PMDError as exc:
        logger.debug("pmd: kamera nie otwarta: %s", exc)
        return []
    try:
        cam.update()
        K = intrinsics_from_xyz(cam.xyz())
        serial, w, h = cam.serial, cam.width, cam.height
    finally:
        cam.close()
    return [dict(source=f"pmd:{serial}", label=f"pmd:{serial}: CamBoard nano {w}x{h} (glebia)",
                 width=w, height=h, K=K.tolist(), serial=serial, model="CamBoard nano")]


class PMDLive:
    """Strumien z CamBoard nano - ten sam interfejs, co `cameras._Live` (+ `depth_t`, `K`).

    Watek wola `pmdUpdate` (kamera oddaje ~45-85 klatek/s zaleznie od czasu integracji),
    pamieta najnowszy kadr: amplituda jako RGB oraz odleglosci, xyz i flagi.
    """

    def __init__(self, record: CameraRecord, calib: str = DEFAULT_CALIB,
                 integration_us: int | None = None, averaging: int = 0):
        serial = parse_source(record.source)
        self.cam = CamboardNano(shared_sdk(), serial, calib, integration_us=integration_us)
        self.serial, self.width, self.height = self.cam.serial, self.cam.width, self.cam.height
        if averaging:
            self.cam.set_averaging(averaging)
        try:
            self.K = intrinsics_from_xyz(self.cam.xyz())
        except ValueError as exc:                          # pierwsza klatka bez punktow - K pozniej z rekordu
            logger.warning("pmd %s: K z xyz nie wyszlo (%s)", record.name, exc)
            self.K = None
        if (self.width, self.height) != (record.width, record.height):
            logger.warning("pmd %s: kamera daje %dx%d, rekord ma %dx%d", record.name, self.width, self.height,
                           record.width, record.height)
        self._lock = threading.Lock()
        self._frame: tuple[np.ndarray, float] | None = None
        self._depth: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"pmd-{record.name}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # pmdUpdate blokuje do klatki albo zwraca blad (kamera odpieta) - nie ma "cichego" braku klatek.
        while not self._stop.is_set():
            try:
                self.cam.update()
                amp, flags = self.cam.amplitudes(), self.cam.flags()
                dist, xyz = self.cam.distances(), self.cam.xyz()
            except PMDError as exc:
                if not self._stop.is_set():
                    self._error = f"blad odczytu pmd: {exc}"
                return
            now = time.monotonic()
            rgb = amplitude_rgb(amp, flags)
            with self._lock:
                self._frame = (rgb, now)
                self._depth = (dist, xyz, flags, now)

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
            err = "watek kamery pmd zatrzymany"
        if err is not None:
            return None, 0.0, err
        with self._lock:
            frame = self._frame
        if frame is None:
            return None, 0.0, None
        return frame[0], frame[1], None

    def rgb(self) -> np.ndarray | None:
        return self.rgb_t()[0]

    def depth_t(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        """(odleglosci [m], xyz (H, W, 3) w ukladzie kamery, flagi, chwila) ostatniej klatki albo None."""
        with self._lock:
            return self._depth

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cam.close()
