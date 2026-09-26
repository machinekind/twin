"""Live viewer for pmd[vision] CamBoard nano time-of-flight cameras.

Shows the distance image (colour-mapped) and the amplitude image (grey) of
every opened camera in an OpenCV window, 165x120 px upscaled.
Keys: q / Esc = quit, s = save distance/amplitude/xyz (.npy) + PNG of each
camera, + / - = integration time up/down, e = toggle auto integration time,
i = toggle invalid-pixel masking, f = toggle vertical flip, a = toggle
temporal averaging.

Display: the distance image is colour-mapped (near = red, far = blue) over
the 2..98 percentile range of the valid pixels (or --min-dist/--max-dist),
with a colour bar; flagged pixels (invalid, low signal, inconsistent,
saturated) are black. Values just below the 5 m wrap-around are shown as
negative (they are noise around 0 m). The amplitude image uses a square-root
tone curve so dim regions stay visible.

The CamBoard nano (USB id 1c28:c007) is a vendor-specific USB device with no
Linux kernel driver: it can only be read through pmd's proprietary PMDSDK2
(libpmdaccess2.so + camboardnano.L64.pap / camboardnanoproc.L64.ppp plugins,
shipped with the camera). This script talks to that library with ctypes, so
nothing needs to be compiled. The SDK is looked up in --sdk-dir, $PMDDIR,
pmd/Linux/Linux64bit/PMDSDK (the unpacked Linux64bit.tar.gz from the pmd
disk), /usr/local/pmd and ./PMDSDK. The plugin needs the old libusb-0.1.so.4;
a copy placed next to libpmdaccess2.so is preloaded automatically. Copy the
SDK's 10-pmd-ubuntu.rules to /etc/udev/rules.d so a normal user may open the
camera. The per-camera <serial>.dat calibration file is taken from --calib
(default: the pmd/Calibration_file directory).
"""

import argparse
import ctypes
import glob
import os
import sys
import time

import cv2
import numpy as np

PMD_OK = 0
PMD_FILE_NOT_FOUND = 4097
PMD_COULD_NOT_OPEN = 4098

# Pixel flag bits (pmdsdk2.h). Any of the first four marks a pixel as unusable.
PMD_FLAG_INVALID = 0x01
PMD_FLAG_SATURATED = 0x02
PMD_FLAG_INCONSISTENT = 0x04
PMD_FLAG_LOW_SIGNAL = 0x08
PMD_FLAG_SBI_ACTIVE = 0x10
BAD_PIXEL_MASK = PMD_FLAG_INVALID | PMD_FLAG_SATURATED | PMD_FLAG_INCONSISTENT | PMD_FLAG_LOW_SIGNAL

PMDHandle = ctypes.c_uint

# Proximity enum for pmdGetValidIntegrationTime
AT_LEAST, AT_MOST, CLOSE_TO = 0, 1, 2


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
        ("_pad", ctypes.c_char * 512),  # headroom in case a newer SDK grew the struct
    ]


class PMDError(RuntimeError):
    pass


SDK_ROOTS = [os.environ.get("PMDDIR"), "pmd/Linux/Linux64bit/PMDSDK", "/usr/local/pmd", "PMDSDK"]


def find_sdk(sdk_dir=None):
    """Return (path to libpmdaccess2.so, plugin dir) or raise with a helpful message."""
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


class PMDSDK:
    """ctypes binding of the handful of pmdsdk2.h functions the viewer needs."""

    def __init__(self, lib_path, plugin_dir):
        # The 2013 camboardnano plugin needs libusb-0.1.so.4, which modern distros no longer
        # ship. If a copy sits next to the SDK, preload it globally so the plugin's dlopen resolves.
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

    def plugin(self, name):
        # The SDK appends .L64.pap / .ppp itself; a path prefix avoids depending on the cwd.
        return os.path.join(self.plugin_dir, name).encode()


class CamboardNano:
    def __init__(self, sdk, serial="", calib=None, integration_us=None, illumination=True):
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
                print("warning: no calibration file %s (SDK falls back to the cwd / defaults)" % calib)
                calib = None
        if calib:
            self.source_command("LoadCalibrationData " + os.path.abspath(calib))
        self.calibrated = self.source_command("IsCalibrationDataLoaded")
        # Settings that lower the LED current go in before the first frame is requested:
        # the illumination switches on with the first pmdUpdate.
        if integration_us is not None:
            self.set_integration_time(integration_us)
        if not illumination:
            self.source_command("SetIllumination 0 0")
        self.check(self.L.pmdUpdate(self.h), "pmdUpdate")
        dd = PMDDataDescription()
        self.check(self.L.pmdGetSourceDataDescription(self.h, ctypes.byref(dd)), "pmdGetSourceDataDescription")
        self.width, self.height = int(dd.img.numColumns), int(dd.img.numRows)
        if not (0 < self.width < 4096 and 0 < self.height < 4096):
            self.width, self.height = 165, 120  # CamBoard nano sensor size
        self.n = self.width * self.height

    # --- SDK plumbing ---------------------------------------------------
    def _last_error(self):
        buf = ctypes.create_string_buffer(256)
        self.L.pmdGetLastError(self.h, buf, len(buf))
        return buf.value.decode(errors="replace")

    def check(self, ret, what):
        if ret != PMD_OK:
            raise PMDError("%s failed (%d): %s" % (what, ret, self._last_error()))

    def source_command(self, cmd):
        buf = ctypes.create_string_buffer(256)
        self.check(self.L.pmdSourceCommand(self.h, buf, len(buf), cmd.encode()), cmd)
        return buf.value.decode(errors="replace").strip()

    def processing_command(self, cmd):
        buf = ctypes.create_string_buffer(256)
        self.check(self.L.pmdProcessingCommand(self.h, buf, len(buf), cmd.encode()), cmd)
        return buf.value.decode(errors="replace").strip()

    def _get_array(self, fn, dtype, per_pixel=1):
        arr = np.empty((self.height, self.width) if per_pixel == 1 else (self.height, self.width, per_pixel), dtype)
        self.check(fn(self.h, arr.ctypes.data, arr.nbytes), fn.__name__)
        return arr

    # --- data -----------------------------------------------------------
    def update(self):
        self.check(self.L.pmdUpdate(self.h), "pmdUpdate")

    def distances(self):
        """Radial distance per pixel, float32 (metres as delivered by the SDK)."""
        return self._get_array(self.L.pmdGetDistances, np.float32)

    def amplitudes(self):
        return self._get_array(self.L.pmdGetAmplitudes, np.float32)

    def flags(self):
        return self._get_array(self.L.pmdGetFlags, np.uint32)

    def xyz(self):
        return self._get_array(self.L.pmdGet3DCoordinates, np.float32, 3)

    # --- settings -------------------------------------------------------
    def get_integration_time(self):
        t = ctypes.c_uint()
        self.check(self.L.pmdGetIntegrationTime(self.h, ctypes.byref(t), 0), "pmdGetIntegrationTime")
        return int(t.value)

    def set_integration_time(self, us):
        valid = ctypes.c_uint()
        self.check(self.L.pmdGetValidIntegrationTime(self.h, ctypes.byref(valid), 0, CLOSE_TO, int(us)),
                   "pmdGetValidIntegrationTime")
        self.check(self.L.pmdSetIntegrationTime(self.h, 0, valid), "pmdSetIntegrationTime")
        return int(valid.value)

    def set_averaging(self, frames):
        if frames:
            self.processing_command("SetAveraging On")
            self.processing_command("SetAveragingFrames %d" % frames)
        else:
            self.processing_command("SetAveraging Off")

    def set_bilateral_filter(self, enable):
        self.processing_command("SetBilateralFilter %s" % ("On" if enable else "Off"))

    def close(self):
        if self.h.value:
            self.L.pmdClose(self.h)
            self.h = PMDHandle(0)


def open_cameras(sdk, serials, count, calib, **kw):
    """Open the given serials, or `count` cameras in the order the SDK finds them."""
    cams = []
    if serials:
        for s in serials:
            cams.append(CamboardNano(sdk, s, calib, **kw))
    else:
        for _ in range(count):
            try:
                cam = CamboardNano(sdk, "", calib, **kw)
            except PMDError as e:
                if cams:
                    break
                raise e
            if any(c.serial == cam.serial for c in cams):  # SDK handed us the same one again
                cam.close()
                break
            cams.append(cam)
    return cams


WRAP_M = 5.0  # unambiguous range at the nano's fixed 30 MHz modulation


def unwrap(dist):
    """Fold distances just below the 5 m wrap-around (noise around 0 m) back to negative values."""
    d = dist.copy()
    d[d > WRAP_M * 0.9] -= WRAP_M
    return d


def display_range(dist, good, min_dist, max_dist):
    """Colour-map range: fixed if given, otherwise the 2..98 percentile of the valid pixels."""
    if min_dist is not None and max_dist is not None:
        return min_dist, max_dist
    if good.sum() < 20:
        return 0.0, max_dist or 2.0
    lo, hi = np.percentile(dist[good], [2, 98])
    lo = min_dist if min_dist is not None else max(lo, 0.0)
    hi = max_dist if max_dist is not None else max(hi, lo + 0.05)
    return float(lo), float(hi)


def colorbar(height, lo, hi, scale, width=None):
    width = width or max(6 * scale, 24)
    ramp = np.linspace(255, 0, height, dtype=np.float32).astype(np.uint8)[:, None]
    bar = cv2.applyColorMap(np.repeat(ramp, width, axis=1), cv2.COLORMAP_JET)
    bar = np.hstack([bar, np.zeros((height, 12 * scale, 3), np.uint8)])
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.11 * scale
    text_h = cv2.getTextSize("0.00", font, fs, 1)[0][1]
    for frac, val in ((0.0, hi), (0.5, (lo + hi) / 2), (1.0, lo)):
        y = int(frac * (height - 1))
        y = min(max(y + text_h // 2, text_h + 2), height - 2)  # baseline, kept inside the image
        cv2.putText(bar, "%.2f" % val, (width + 3, y), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def render(dist, amp, flags, lo, hi, mask_invalid, scale, status=None):
    """Distance (colour-mapped, near = red) next to amplitude (grey), upscaled smoothly, with a
    colour bar and a status strip. dist is taken as already unwrapped."""
    h, w = dist.shape
    bad = (flags & BAD_PIXEL_MASK) != 0 if mask_invalid else np.zeros(dist.shape, bool)
    size = (w * scale, h * scale)

    # Interpolate the data (not the colours) so the upscaling stays smooth; invalid pixels are
    # filled with the median before interpolation so they do not bleed into their neighbours.
    d = dist.astype(np.float32).copy()
    if bad.any() and (~bad).any():
        d[bad] = np.median(d[~bad])
    d_big = cv2.resize(d, size, interpolation=cv2.INTER_CUBIC)
    n = (255 * (1.0 - np.clip((d_big - lo) / max(hi - lo, 1e-6), 0, 1))).astype(np.uint8)
    d_img = cv2.applyColorMap(n, cv2.COLORMAP_JET)
    bad_big = cv2.resize(bad.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    d_img[bad_big] = 0

    a = np.sqrt(np.maximum(amp, 0))  # gamma 0.5: lifts the dark parts of the amplitude image
    top = np.percentile(a[~bad], 99.5) if (~bad).any() else a.max()
    a_big = cv2.resize(a, size, interpolation=cv2.INTER_CUBIC)
    a8 = (255 * np.clip(a_big / max(top, 1e-6), 0, 1)).astype(np.uint8)
    a_img = cv2.cvtColor(a8, cv2.COLOR_GRAY2BGR)
    a_img[bad_big] = (a_img[bad_big] * 0.35).astype(np.uint8)

    gap = np.zeros((size[1], 2 * scale, 3), np.uint8)
    img = np.hstack([d_img, colorbar(size[1], lo, hi, scale), gap, a_img])
    if status:
        strip_h = 8 * scale
        strip = np.zeros((strip_h, img.shape[1], 3), np.uint8)
        fs = 0.13 * scale
        for i, line in enumerate(status):
            cv2.putText(strip, line, (2 * scale, int((i + 0.8) * strip_h / len(status))), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (255, 255, 255), 1, cv2.LINE_AA)
        img = np.vstack([strip, img])
    return img


class AutoIntegration:
    """Simple exposure control: raise the integration time while many pixels are flagged as low
    signal, lower it as soon as some saturate. Steps are applied at most every `period` frames."""

    def __init__(self, cam, period=5, lo_us=50, hi_us=2000):
        self.cam, self.period, self.lo_us, self.hi_us = cam, period, lo_us, hi_us
        self.n = 0

    def step(self, flags):
        self.n += 1
        if self.n % self.period:
            return None
        sat = ((flags & PMD_FLAG_SATURATED) != 0).mean()
        low = ((flags & PMD_FLAG_LOW_SIGNAL) != 0).mean()
        t = self.cam.get_integration_time()
        if sat > 0.005:
            new = int(t * 0.7)
        elif low > 0.05:
            new = int(t * 1.3) + 10
        else:
            return None
        new = min(max(new, self.lo_us), self.hi_us)
        return self.cam.set_integration_time(new) if new != t else None


def save_frames(cams, outdir, flip, max_dist, mask_invalid):
    """Save distance/amplitude/xyz arrays (.npy) and a rendered PNG of each camera's current frame."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(outdir, exist_ok=True)
    for cam in cams:
        base = os.path.join(outdir, "pmd_%s_%s" % (cam.serial, stamp))
        dist, amp, flags, xyz = cam.distances(), cam.amplitudes(), cam.flags(), cam.xyz()
        if flip:
            dist, amp, flags, xyz = dist[::-1], amp[::-1], flags[::-1], xyz[::-1]
        np.save(base + "_dist.npy", dist)
        np.save(base + "_amp.npy", amp)
        np.save(base + "_xyz.npy", xyz)
        du = unwrap(dist)
        good = (flags & BAD_PIXEL_MASK) == 0
        lo, hi = display_range(du, good, None, max_dist)
        cv2.imwrite(base + ".png", render(du, amp, flags, lo, hi, mask_invalid, 2))
        print("saved", base + "_{dist,amp,xyz}.npy / .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdk-dir", default=None, help="PMDSDK root (default: $PMDDIR, /usr/local/pmd, ./PMDSDK)")
    ap.add_argument("--serial", action="append", default=[], help="camera serial to open (repeatable)")
    ap.add_argument("--count", type=int, default=1, help="how many cameras to open when no --serial is given")
    ap.add_argument("--calib", default="pmd/Calibration_file",
                    help="camera calibration: a <serial>.dat file or a directory holding one per serial")
    ap.add_argument("--integration", type=int, default=None, help="integration time in us (camera default otherwise)")
    ap.add_argument("--averaging", type=int, default=0, help="temporal averaging window in frames (0 = off)")
    ap.add_argument("--no-bilateral", action="store_true", help="disable the SDK's bilateral distance filter")
    ap.add_argument("--no-illumination", action="store_true",
                    help="keep the LEDs off (no usable depth; for USB power debugging)")
    ap.add_argument("--min-dist", type=float, default=None, help="near end of the colour map in m (default: auto, 2nd percentile)")
    ap.add_argument("--max-dist", type=float, default=None, help="far end of the colour map in m (default: auto, 98th percentile)")
    ap.add_argument("--scale", type=int, default=6, help="display upscaling factor (165x120 -> 990x720 per image)")
    ap.add_argument("--auto-integration", action="store_true",
                    help="adjust the integration time automatically (up while pixels are low-signal, down on saturation)")
    ap.add_argument("--no-flip", action="store_true", help="do not flip images vertically (raw SDK orientation)")
    ap.add_argument("--outdir", default=".", help="where 's' saves frames")
    ap.add_argument("--frames", type=int, default=0, help="exit after this many frames (0 = run until q)")
    args = ap.parse_args()

    try:
        lib, plugin_dir = find_sdk(args.sdk_dir)
        print("PMDSDK: %s, plugins in %s" % (lib, plugin_dir))
        sdk = PMDSDK(lib, plugin_dir)
        cams = open_cameras(sdk, args.serial, args.count, args.calib,
                            integration_us=args.integration, illumination=not args.no_illumination)
    except (PMDError, OSError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    if not cams:
        print("error: no CamBoard nano opened", file=sys.stderr)
        return 1

    for cam in cams:
        cam.auto = AutoIntegration(cam) if args.auto_integration else None
        cam.set_averaging(args.averaging)
        if args.no_bilateral:
            cam.set_bilateral_filter(False)
        print("CamBoard nano %s: %dx%d, integration %d us, calibration loaded: %s" % (
            cam.serial, cam.width, cam.height, cam.get_integration_time(), cam.calibrated))

    flip, mask_invalid, averaging = not args.no_flip, True, args.averaging
    counts = {cam.serial: 0 for cam in cams}
    t0 = time.time()
    fps = 0.0
    try:
        while True:
            for cam in cams:
                cam.update()
                dist, amp, flags = cam.distances(), cam.amplitudes(), cam.flags()
                if flip:
                    dist, amp, flags = dist[::-1], amp[::-1], flags[::-1]
                if cam.auto is not None:
                    cam.auto.step(flags)
                du = unwrap(dist)
                good = (flags & BAD_PIXEL_MASK) == 0
                lo, hi = display_range(du, good, args.min_dist, args.max_dist)
                cy, cx = cam.height // 2, cam.width // 2
                centre = du[cy - 1:cy + 2, cx - 1:cx + 2]
                cgood = good[cy - 1:cy + 2, cx - 1:cx + 2]
                centre_txt = "%.1f cm" % (100 * centre[cgood].mean()) if cgood.any() else "invalid"
                status = [
                    "%s   %.1f fps   integration %d us%s   valid %.0f%%" % (
                        cam.serial, fps, cam.get_integration_time(), " (auto)" if cam.auto else "", 100 * good.mean()),
                    "centre %s   range %.2f..%.2f m%s   avg %s   flip %s" % (
                        centre_txt, lo, hi, "" if (args.min_dist is None and args.max_dist is None) else " (fixed)",
                        "%d" % averaging if averaging else "off", "on" if flip else "off"),
                ]
                img = render(du, amp, flags, lo, hi, mask_invalid, args.scale, status)
                strip = img.shape[0] - cam.height * args.scale
                cv2.drawMarker(img, (int((cx + 0.5) * args.scale), strip + int((cy + 0.5) * args.scale)),
                               (255, 255, 255), cv2.MARKER_CROSS, 3 * args.scale, 1)
                cv2.imshow("pmd " + cam.serial, img)
                counts[cam.serial] += 1

            n = counts[cams[0].serial]
            if args.frames and n >= args.frames:
                break
            if n % 10 == 0:
                now = time.time()
                fps = 10.0 / max(now - t0, 1e-6)
                t0 = now

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("s"):
                save_frames(cams, args.outdir, flip, args.max_dist, mask_invalid)
            elif k == ord("e"):
                for cam in cams:
                    cam.auto = None if cam.auto else AutoIntegration(cam)
                print("auto integration", "on" if cams[0].auto else "off")
            elif k in (ord("+"), ord("=")):
                for cam in cams:
                    print("%s: integration %d us" % (cam.serial, cam.set_integration_time(cam.get_integration_time() + 50)))
            elif k == ord("-"):
                for cam in cams:
                    print("%s: integration %d us" % (cam.serial, cam.set_integration_time(max(cam.get_integration_time() - 50, 10))))
            elif k == ord("i"):
                mask_invalid = not mask_invalid
                print("invalid-pixel masking", "on" if mask_invalid else "off")
            elif k == ord("f"):
                flip = not flip
            elif k == ord("a"):
                averaging = 0 if averaging else (args.averaging or 4)
                for cam in cams:
                    cam.set_averaging(averaging)
                print("averaging", "%d frames" % averaging if averaging else "off")
    finally:
        for cam in cams:
            cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
