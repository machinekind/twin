"""Camera focus and exposure on every OS, and the focus lock of the lens step.

The backends talk to fakes: a UVC camera behind a control-request transport (macOS), an OpenCV
capture (Windows); the V4L2 fake lives with the stand tests. The focus lock runs against a camera
whose image blurs with the distance from its sharpest focus.
"""

from __future__ import annotations

import json
import struct
import threading
import time

import cv2
import numpy as np
import pytest

from machinekind_twin import camera_controls as cc
from machinekind_twin import uvc_mac
from machinekind_twin.calib import focus as fl
from machinekind_twin.camera_controls import Applied, CameraControls, ControlError, Range
from machinekind_twin.uvc_mac import CameraTerminal, UvcCamera

C920_BITS = 0b100000101000101110          # AE mode and priority, exposure, focus, zoom, pan/tilt, autofocus
WRIST_BITS = 0b100010101000001110         # the same without a manual focus
C920, WRIST_A, WRIST_B, BUILT_IN = ("0x120000046d082d", "0x11000010bb2b08", "0x112000010bb2b08",
                                     "3F45E80A-0176-46F7-B185-BB9E2C0E82E3")


# ------------------------------------------------------------------ macOS: UVC through IOKit
def uvc_config(bits: int, *, interface: int = 0, terminal: int = 1) -> bytes:
    """A webcam's configuration descriptor: an IAD, the VideoControl interface with a header and a
    camera terminal, then a VideoStreaming interface with a decoy class descriptor."""
    body = bytes([8, 0x0B, interface, 2, 0x0E, 0x03, 0, 0])
    body += bytes([9, 0x04, interface, 0, 1, 0x0E, 0x01, 0, 0])
    body += bytes([13, 0x24, 0x01, 0x00, 0x01, 0x33, 0x00, 0, 0, 0, 0, 1, 1])
    body += bytes([18, 0x24, 0x02, terminal, 0x01, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 3]) + bits.to_bytes(3, "little")
    body += bytes([9, 0x04, interface + 1, 0, 1, 0x0E, 0x02, 0, 0])
    body += bytes([18, 0x24, 0x02, 9, 0x01, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0xFF, 0xFF, 0xFF])
    head = struct.pack("<BBHBBBBB", 9, 0x02, 9 + len(body), 2, 1, 0, 0x80, 0xFA)
    return head + body


#: Camera terminal control selector -> its size in bytes.
SIZES = {0x02: 1, 0x04: 4, 0x06: 2, 0x08: 1}
#: The C920's focus range.
C920_FOCUS = Range(0, 250, 5)


class FakeUvc:
    """A UVC camera behind `UvcCamera`'s transport: descriptors, current values and ranges."""

    def __init__(self, bits: int = C920_BITS, *, ae_modes: int = 0x09, stall: tuple[int, ...] = (),
                 reads: dict[int, int] | None = None):
        self.config = uvc_config(bits)
        self.values = {0x02: 0x08, 0x04: 166, 0x06: 0, 0x08: 1}
        self.ranges = {0x04: (3, 2047, 1), 0x06: (0, 250, 5), 0x08: (0, 1, 1)}
        self.ae_modes, self.stall, self.reads = ae_modes, stall, reads or {}
        self.calls: list[tuple[int, int, int, int, bytes | int]] = []

    def __call__(self, kind: int, code: int, value: int, index: int, data: bytes | int) -> bytes:
        self.calls.append((kind, code, value, index, data))
        if kind == 0x80:
            assert isinstance(data, int)
            return self.config[:data]
        selector = value >> 8
        if selector in self.stall:
            raise ControlError("kamera odrzucila zadanie (STALL)")
        size = SIZES[selector]
        if kind == uvc_mac.SET_REQUEST:
            assert isinstance(data, bytes) and len(data) == size and code == uvc_mac.SET_CUR
            self.values[selector] = int.from_bytes(data, "little")
            return b""
        if code == uvc_mac.GET_CUR:
            return self.reads.get(selector, self.values[selector]).to_bytes(size, "little")
        if selector == 0x02 and code == uvc_mac.GET_RES:
            return bytes([self.ae_modes])
        lo, hi, res = self.ranges[selector]
        return {uvc_mac.GET_MIN: lo, uvc_mac.GET_MAX: hi, uvc_mac.GET_RES: res}[code].to_bytes(size, "little")


class FakeUSB:
    """IOKit's part: one fake camera per USB location; `asked` lists the addresses opened."""

    def __init__(self, cameras: dict[int, FakeUvc]):
        self.cameras = cameras
        self.asked: list[tuple[int, int, int]] = []

    def camera(self, location: int, vendor: int, product: int) -> UvcCamera:
        self.asked.append((location, vendor, product))
        fake = self.cameras[location]
        terminal = uvc_mac.camera_terminal(fake.config)
        assert terminal is not None
        return UvcCamera(fake, terminal)


def test_a_usb_cameras_unique_id_gives_its_usb_port_and_ids():
    assert uvc_mac.usb_address(C920) == (0x120000, 0x046D, 0x082D)
    assert uvc_mac.usb_address(WRIST_B) == (0x1120000, 0x10BB, 0x2B08)
    assert uvc_mac.usb_address(BUILT_IN) is None


def test_the_camera_terminal_is_found_in_the_video_control_interface_only():
    assert uvc_mac.camera_terminal(uvc_config(C920_BITS, interface=2, terminal=4)) == CameraTerminal(2, 4, C920_BITS)
    no_video = struct.pack("<BBHBBBBB", 9, 2, 18, 1, 1, 0, 0x80, 0xFA) + bytes([9, 4, 0, 0, 1, 1, 1, 0, 0])
    assert uvc_mac.camera_terminal(no_video) is None


def test_uvc_requests_go_to_the_camera_terminal_with_little_endian_values():
    fake = FakeUvc()
    cam = UvcCamera(fake, CameraTerminal(0, 1, C920_BITS))
    cam.set("autofocus", 0)
    cam.set("focus_absolute", 300)
    assert fake.calls[-2:] == [(0x21, 0x01, 0x0800, 0x0100, b"\x00"), (0x21, 0x01, 0x0600, 0x0100, b"\x2c\x01")]
    assert (cam.get("autofocus"), cam.get("focus_absolute")) == (0, 300)
    assert cam.range("focus_absolute") == Range(0, 250, 5)
    cam.set("exposure_absolute", 70000)
    assert fake.calls[-1][4] == (70000).to_bytes(4, "little") and cam.get("exposure_absolute") == 70000


def test_uvc_auto_exposure_uses_a_mode_the_camera_has():
    fake = FakeUvc()
    cam = UvcCamera(fake, CameraTerminal(0, 1, C920_BITS))
    cam.set("auto_exposure", 0)
    assert fake.values[0x02] == uvc_mac.AE_MANUAL and cam.get("auto_exposure") == 0
    cam.set("auto_exposure", 1)
    assert fake.values[0x02] == uvc_mac.AE_APERTURE and cam.get("auto_exposure") == 1
    full_auto_only = FakeUvc(ae_modes=uvc_mac.AE_MANUAL | uvc_mac.AE_AUTO)
    UvcCamera(full_auto_only, CameraTerminal(0, 1, C920_BITS)).set("auto_exposure", 1)
    assert full_auto_only.values[0x02] == uvc_mac.AE_AUTO


def test_a_control_the_terminal_lacks_is_missing_and_a_stall_is_a_refusal():
    cam = UvcCamera(FakeUvc(WRIST_BITS), CameraTerminal(0, 1, WRIST_BITS))
    with pytest.raises(ControlError) as missing:
        cam.set("focus_absolute", 10)
    assert missing.value.missing
    stalled = UvcCamera(FakeUvc(stall=(0x06,)), CameraTerminal(0, 1, C920_BITS))
    with pytest.raises(ControlError, match="STALL") as refused:
        stalled.set("focus_absolute", 10)
    assert not refused.value.missing


def test_two_identical_cameras_each_get_only_their_own_settings():
    a, b, c920 = FakeUvc(WRIST_BITS), FakeUvc(WRIST_BITS), FakeUvc()
    usb = FakeUSB({0x110000: a, 0x1120000: b, 0x120000: c920})
    ids = [WRIST_A, WRIST_B, C920, BUILT_IN]          # OpenCV's order: sorted by unique ID

    def opener(index):
        return lambda: uvc_mac.open_camera(index, ids=lambda: ids, usb=usb)

    done = cc.apply("1", CameraControls(autofocus=False, exposure_absolute=50), device=opener(1))
    assert done.ok == ["autofocus", "auto_exposure", "exposure_absolute"] and done.backend == "UVC"
    assert usb.asked == [(0x1120000, 0x10BB, 0x2B08)]
    assert (b.values[0x08], b.values[0x04]) == (0, 50) and (a.values[0x08], a.values[0x04]) == (1, 166)
    done = cc.apply("2", CameraControls(focus_absolute=40), device=opener(2))
    assert done.ok == ["autofocus", "focus_absolute"] and done.read == {"autofocus": 0, "focus_absolute": 40}
    assert c920.values[0x06] == 40 and a.values == FakeUvc(WRIST_BITS).values
    assert "USB" in cc.apply("3", CameraControls(autofocus=False), device=opener(3)).skipped
    assert "numer 7" in cc.apply("7", CameraControls(autofocus=False), device=opener(7)).failed["autofocus"]


def test_on_macos_a_camera_index_goes_to_the_uvc_backend(monkeypatch):
    fake = FakeUvc()
    opened = []

    def open_camera(index):
        opened.append(index)
        return UvcCamera(fake, CameraTerminal(0, 1, C920_BITS))

    monkeypatch.setattr(uvc_mac, "open_camera", open_camera)
    done = cc.apply("2", CameraControls(autofocus=False, focus_absolute=0), platform="darwin")
    assert opened == [2] and done.ok == ["autofocus", "focus_absolute"]
    assert cc.focus_range("2", platform="darwin") == Range(0, 250, 5)
    assert cc.apply("film.mp4", CameraControls(autofocus=False), platform="darwin").skipped


def test_a_value_read_back_different_from_the_one_set_is_a_failure():
    fake = FakeUvc(reads={0x06: 35})
    done = cc.apply("2", CameraControls(focus_absolute=30), device=lambda: UvcCamera(fake, CameraTerminal(0, 1, C920_BITS)))
    assert done.ok == ["autofocus"] and done.failed == {"focus_absolute": "odczyt 35 zamiast 30"}
    assert done.read["focus_absolute"] == 35 and "odczyt 35 zamiast 30" in done.text()


# ------------------------------------------------------------------ Linux: read back through V4L2
def test_v4l2_reads_each_control_back_and_reports_its_focus_range():
    from test_stand import FakeIoctl

    ioctl = FakeIoctl(reads={cc.CID_FOCUS_ABSOLUTE: 35})
    done = cc.apply("/dev/video0", CameraControls(focus_absolute=30, auto_exposure=True), platform="linux",
                    ioctl=ioctl, opener=lambda path: 4, closer=lambda fd: None)
    assert done.ok == ["autofocus", "auto_exposure"] and done.backend == "V4L2"
    assert done.failed == {"focus_absolute": "odczyt 35 zamiast 30"} and done.read["auto_exposure"] == 1
    missing = cc.apply("/dev/video0", CameraControls(focus_absolute=30), platform="linux",
                       ioctl=FakeIoctl(refuse={cc.CID_FOCUS_ABSOLUTE}), opener=lambda path: 4, closer=lambda fd: None)
    assert missing.missing == ["focus_absolute"]
    dev = cc.V4L2Device("/dev/video0", ioctl=FakeIoctl(), opener=lambda path: 4, closer=lambda fd: None)
    assert dev.range("focus_absolute") == Range(0, 250, 5)


# ------------------------------------------------------------------ Windows: DirectShow
class FakeCapture:
    """An OpenCV DirectShow capture: the focus mode reads back as flags (1 auto, 2 manual)."""

    def __init__(self):
        self.props = {cv2.CAP_PROP_AUTOFOCUS: 3.0, cv2.CAP_PROP_FOCUS: 0.0}

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_AUTOFOCUS:
            self.props[prop] = 3.0 if round(value) == 1 else 2.0
        elif prop in self.props:
            self.props[prop] = value
        else:
            return False
        return True

    def get(self, prop):
        return self.props.get(prop, -1.0)


def test_windows_sets_focus_through_the_open_capture_and_reads_it_back():
    cap = FakeCapture()
    done = cc.apply("0", CameraControls(focus_absolute=45, exposure_absolute=100), platform="win32", cap=cap)
    assert done.ok == ["autofocus", "focus_absolute"] and done.backend == "DirectShow"
    assert done.read == {"autofocus": 0, "focus_absolute": 45} and cap.props[cv2.CAP_PROP_FOCUS] == 45
    assert done.failed == {"auto_exposure": "na Windows tylko ostrosc", "exposure_absolute": "na Windows tylko ostrosc"}
    assert cc.apply("0", CameraControls(autofocus=True), platform="win32", cap=cap).read == {"autofocus": 1}
    assert cc.apply("0", CameraControls(autofocus=False), platform="win32").skipped   # no open capture
    assert cc.focus_range("0", platform="win32", cap=cap) is None


# ------------------------------------------------------------------ does the lens hold?
def test_a_lens_holds_only_with_autofocus_off_and_its_own_focus():
    ok = Applied(ok=["autofocus", "focus_absolute"], read={"autofocus": 0, "focus_absolute": 30})
    assert cc.lens_focus_problem(30, ok) == "" and cc.lens_focus_problem(30, None) == ""
    assert "obiektyw zmierzony przy 30" in cc.lens_focus_problem(30, Applied(ok=ok.ok, read={"autofocus": 0,
                                                                                            "focus_absolute": 45}))
    assert "macOS" in cc.lens_focus_problem(30, Applied(skipped="brak sterowania na macOS"))
    assert "odczyt" in cc.lens_focus_problem(30, Applied(failed={"focus_absolute": "odczyt 35 zamiast 30"}))
    assert cc.lens_focus_problem(30, Applied(ok=["autofocus"], read={"autofocus": 1})) == "autofokus wlaczony"
    no_af = Applied(ok=["focus_absolute"], failed={"autofocus": "brak"}, missing=["autofocus"],
                    read={"focus_absolute": 30})
    assert cc.lens_focus_problem(30, no_af) == ""


# ------------------------------------------------------------------ the focus lock
class Rig:
    """A camera whose image is sharpest at focus `best` and blurs with the distance from it."""

    def __init__(self, best: int = 120, *, focus: bool = True, autofocus: bool = True, skipped: str = "",
                 refuse_af: bool = False, misread: int | None = None, flat: bool = False):
        self.best, self.has_focus, self.has_af, self.skipped = best, focus, autofocus, skipped
        self.refuse_af, self.misread = refuse_af, misread
        self.focus, self.af, self.t = 0, 1, 0.0
        rng = np.random.default_rng(0)
        self.base = np.full((240, 320), 128, np.uint8) if flat else rng.integers(0, 255, (240, 320), np.uint8)
        self.set_calls: list[CameraControls] = []

    def set_controls(self, controls: CameraControls) -> Applied:
        self.set_calls.append(controls)
        if self.skipped:
            return Applied(skipped=self.skipped)
        out = Applied(backend="fake")
        for name, value in controls.wanted():
            if (name == "autofocus" and not self.has_af) or (name == "focus_absolute" and not self.has_focus):
                out.failed[name], out.missing = "brak w tej kamerze", out.missing + [name]
                continue
            if name == "autofocus" and self.refuse_af:
                out.failed[name] = "kamera odrzucila"
                continue
            if name == "autofocus":
                self.af = value
            else:
                self.focus = value
            got = self.misread if name == "focus_absolute" and self.misread is not None else value
            out.read[name] = got
            (out.ok.append(name) if got == value else out.failed.__setitem__(name, f"odczyt {got} zamiast {value}"))
        return out

    def grab(self) -> tuple[np.ndarray, float]:
        self.t += 0.04
        sigma = 0.3 + abs(self.focus - self.best) / 15
        return np.dstack([cv2.GaussianBlur(self.base, (0, 0), sigma)] * 3), self.t

    def clock(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt

    def lock(self, wanted=None, rng=C920_FOCUS, **kw) -> fl.FocusLock:
        return fl.lock_focus(self.set_controls, self.grab, wanted=wanted, focus_range=rng, clock=self.clock,
                             sleep=self.sleep, **kw)


def test_sharpness_drops_with_blur_and_the_grid_follows_the_cameras_step():
    img = np.random.default_rng(1).integers(0, 255, (120, 160), np.uint8)
    assert fl.sharpness(img) > fl.sharpness(cv2.GaussianBlur(img, (0, 0), 2.0)) * 5
    assert fl.grid(Range(0, 250, 5), 0, 250, 11) == list(range(0, 251, 25))
    assert fl.grid(Range(0, 250, 5), 101, 124) == [105, 110, 115, 120]
    assert fl.grid(Range(3, 20, 4), 0, 100) == [3, 7, 11, 15, 19]


def test_without_a_focus_in_the_record_the_sweep_picks_the_sharpest():
    rig = Rig(best=120)
    seen = []
    lock = rig.lock(report=lambda text, part: seen.append(part))
    assert lock.focus == 120 and lock.autofocus_off and "przeszukania" in lock.note
    values = [v for v, _ in lock.swept]
    assert set(range(0, 251, 25)) <= set(values) and 120 in values
    assert rig.set_calls[0] == CameraControls(autofocus=False)
    assert rig.set_calls[-1] == CameraControls(autofocus=False, focus_absolute=120) and rig.focus == 120
    assert seen == sorted(seen) and 0.0 <= seen[0] and seen[-1] < 1.0


def test_the_focus_in_the_record_is_set_without_a_sweep():
    rig = Rig()
    lock = rig.lock(wanted=0)
    assert (lock.focus, lock.swept) == (0, []) and "z ustawien kamery" in lock.note
    assert rig.set_calls == [CameraControls(autofocus=False), CameraControls(autofocus=False, focus_absolute=0)]


def test_a_camera_without_a_focus_or_controls_is_measured_as_before():
    assert Rig(skipped="kamera nie jest kamera USB (UVC)").lock().focus is None
    wrist = Rig(focus=False).lock()
    assert (wrist.focus, wrist.autofocus_off) == (None, True) and "nie ma recznej ostrosci" in wrist.note
    fixed = Rig(focus=False, autofocus=False).lock()
    assert (fixed.focus, fixed.autofocus_off) == (None, False) and "stala" in fixed.note
    manual_only = Rig(autofocus=False).lock(wanted=30)
    assert manual_only.focus == 30


@pytest.mark.parametrize(("rig", "wanted", "match"), [
    (Rig(refuse_af=True), 0, "autofokus nie wylacza sie"),
    (Rig(misread=5), 0, "odczyt 5 zamiast 0"),
    (Rig(focus=False), 30, "nie ma recznej ostrosci"),
    (Rig(flat=True), None, "szachownice"),
])
def test_a_focus_that_does_not_lock_stops_the_lens_step(rig, wanted, match):
    with pytest.raises(fl.FocusLockError, match=match):
        rig.lock(wanted=wanted)


def test_the_sweep_stops_when_the_operator_cancels_or_frames_stop():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(fl.FocusLockError, match="przerwane"):
        Rig().lock(cancel=cancel)
    frozen = Rig()
    frozen.grab = lambda: (None, 0.0)                   # type: ignore[method-assign]
    with pytest.raises(fl.FocusLockError, match="klatek"):
        frozen.lock()


# ------------------------------------------------------------------ the lens job
class FocusHub:
    """The camera hub's part the lens step uses, over a `Rig`; frames are stamped with the real clock."""

    def __init__(self, rig: Rig):
        self.rig = rig
        self.applied: list[CameraControls | None] = []

    def apply_controls(self, name, controls=None):
        self.applied.append(controls)
        return self.rig.set_controls(controls) if controls is not None else Applied()

    def focus_range(self, name):
        return Range(0, 250, 5)

    def frame_t(self, name):
        return self.rig.grab()[0], time.monotonic()

    def frame(self, name):
        return None


def test_the_lens_job_locks_the_focus_before_its_first_view(tmp_path, monkeypatch):
    from machinekind_twin.calib import intrinsics
    from machinekind_twin.ui import jobs

    rig = Rig(best=60)
    hub = FocusHub(rig)
    fit = intrinsics.Result(np.eye(3), np.zeros(5), 0.3, [0.3] * 12, 12, 0.7, (640, 480), True)
    monkeypatch.setattr(intrinsics.Collector, "solve", lambda self: fit)
    solve = threading.Event()
    solve.set()
    job = jobs.Job("intrynsyki")
    step = jobs.focus_lock_step(hub, "kam1", None, settle=0.0, samples=1)
    job.start(lambda j: jobs.run_intrinsics(j, hub, "kam1", intrinsics.Board(), (640, 480), period=0.01,
                                            solve=solve, save_to=tmp_path / "sesja", lock_focus=step))
    assert job.wait(10.0) and job.state == jobs.DONE and job.result is fit
    assert job.data["focus"].focus == 60 and rig.focus == 60 and job.data["last"] is not None
    assert hub.applied[0] == CameraControls(autofocus=False)
    saved = json.loads((tmp_path / "sesja" / "sesja.json").read_text())
    assert saved["focus"]["focus"] == 60 and max(saved["focus"]["swept"], key=lambda vs: vs[1])[0] == 60


def test_a_lens_job_whose_focus_does_not_lock_measures_nothing():
    from machinekind_twin.calib import intrinsics
    from machinekind_twin.ui import jobs

    def refuse(job):
        raise fl.FocusLockError("ostrosc 0 nie ustawia sie: odczyt 5 zamiast 0")

    job = jobs.Job("intrynsyki")
    job.start(lambda j: jobs.run_intrinsics(j, FocusHub(Rig()), "kam1", intrinsics.Board(), (640, 480),
                                            period=0.01, lock_focus=refuse))
    assert job.wait(5.0) and job.state == jobs.FAILED and "collector" not in job.data
    assert job.data["focus_error"] == "ostrosc 0 nie ustawia sie: odczyt 5 zamiast 0"
