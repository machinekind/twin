"""UVC camera controls on macOS: class requests to the camera's USB device through IOKit.

OpenCV's AVFoundation backend ignores focus and exposure, so the twin sends the UVC requests
itself: SET_CUR / GET_CUR on the camera terminal of the VideoControl interface, through
`IOUSBDeviceInterface::DeviceRequest`. That call needs no open of the device, so it works while
the system's UVC driver streams the camera. Only ctypes, no extra dependencies.

A camera source on macOS is an index into AVFoundation's video devices in the order OpenCV
numbers them (sorted by unique ID). A USB camera's unique ID is "0x" + its USB location ID
+ vendor ID + product ID, so the index leads to one USB port: two identical cameras never get
each other's settings.
"""

from __future__ import annotations

import ctypes
import re
import struct
from collections.abc import Callable
from ctypes import (
    POINTER,
    byref,
    c_bool,
    c_char_p,
    c_int,
    c_int32,
    c_int64,
    c_uint8,
    c_uint16,
    c_uint32,
    c_ulong,
    c_void_p,
)
from dataclasses import dataclass
from typing import Any

from .camera_controls import ControlError, NoControls, Range

#: Request types (UVC 1.5, 4.2.1): class request to an interface, host to device and back.
SET_REQUEST, GET_REQUEST = 0x21, 0xA1
SET_CUR, GET_CUR, GET_MIN, GET_MAX, GET_RES = 0x01, 0x81, 0x82, 0x83, 0x84
#: Camera terminal control selectors: name -> (selector, size [bytes], bit in the terminal's bmControls).
CONTROLS = {"auto_exposure": (0x02, 1, 1), "exposure_absolute": (0x04, 4, 3),
            "focus_absolute": (0x06, 2, 5), "autofocus": (0x08, 1, 17)}
#: CT_AE_MODE_CONTROL values: manual, auto, shutter priority, aperture priority.
AE_MANUAL, AE_AUTO, AE_SHUTTER, AE_APERTURE = 0x01, 0x02, 0x04, 0x08
#: IOReturn of a request the camera refused (the control pipe stalled).
PIPE_STALLED = 0xE000404F

#: Transport: (bmRequestType, bRequest, wValue, wIndex, data to send or the length to read) -> data read.
Request = Callable[[int, int, int, int, "bytes | int"], bytes]


@dataclass(frozen=True)
class CameraTerminal:
    """Where a UVC camera's controls are: the VideoControl interface and its camera terminal."""

    interface: int
    terminal: int
    #: The terminal's bmControls: one bit per control it has.
    controls: int


def usb_address(unique_id: str) -> tuple[int, int, int] | None:
    """(location ID, vendor ID, product ID) from a USB camera's AVFoundation unique ID; None for others."""
    m = re.fullmatch(r"0x([0-9a-fA-F]+)([0-9a-fA-F]{4})([0-9a-fA-F]{4})", unique_id)
    if m is None:
        return None
    return int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16)


def camera_terminal(config: bytes) -> CameraTerminal | None:
    """The first camera terminal in a USB configuration descriptor; None when there is none."""
    i, video_control = 0, None
    while i + 2 <= len(config):
        length, kind = config[i], config[i + 1]
        if length < 2 or i + length > len(config):
            break
        if kind == 0x04 and length >= 9:                # INTERFACE: number, class, subclass
            number, cls, sub = config[i + 2], config[i + 5], config[i + 6]
            video_control = number if (cls, sub) == (0x0E, 0x01) else None
        elif kind == 0x24 and video_control is not None and length >= 15 and config[i + 2] == 0x02:
            # CS_INTERFACE, VC_INPUT_TERMINAL; wTerminalType ITT_CAMERA
            if struct.unpack_from("<H", config, i + 4)[0] == 0x0201:
                size = config[i + 14]
                bits = int.from_bytes(config[i + 15:i + 15 + size], "little")
                return CameraTerminal(video_control, config[i + 3], bits)
        i += length
    return None


class UvcCamera:
    """A UVC camera's controls through a control-request transport (IOKit, or a fake in tests)."""

    backend = "UVC"

    def __init__(self, request: Request, terminal: CameraTerminal, close: Callable[[], None] = lambda: None):
        self._request = request
        self.terminal = terminal
        self._close = close

    def _control(self, name: str) -> tuple[int, int]:
        selector, size, bit = CONTROLS[name]
        if not self.terminal.controls >> bit & 1:
            raise ControlError("brak w tej kamerze", missing=True)
        return selector, size

    def _call(self, kind: int, request: int, selector: int, data: bytes | int) -> bytes:
        index = self.terminal.terminal << 8 | self.terminal.interface
        return self._request(kind, request, selector << 8, index, data)

    def _read(self, request: int, name: str) -> int:
        selector, size = self._control(name)
        data = self._call(GET_REQUEST, request, selector, size)
        if len(data) < size:
            raise ControlError(f"kamera oddala {len(data)} z {size} bajtow")
        return int.from_bytes(data[:size], "little")

    def set(self, name: str, value: int) -> None:
        selector, size = self._control(name)
        if name == "auto_exposure":
            value = self._auto_mode() if value else AE_MANUAL
        self._call(SET_REQUEST, SET_CUR, selector, int(value).to_bytes(size, "little"))

    def get(self, name: str) -> int:
        value = self._read(GET_CUR, name)
        return int(value != AE_MANUAL) if name == "auto_exposure" else value

    def range(self, name: str) -> Range | None:
        lo, hi = self._read(GET_MIN, name), self._read(GET_MAX, name)
        return Range(lo, hi, max(1, self._read(GET_RES, name)))

    def _auto_mode(self) -> int:
        """The auto exposure mode the camera has: aperture priority (what V4L2 calls auto), else full auto."""
        try:
            modes = self._read(GET_RES, "auto_exposure")      # for this control: a bitmap of the modes
        except ControlError:
            return AE_APERTURE
        return AE_APERTURE if modes & AE_APERTURE or not modes & AE_AUTO else AE_AUTO

    def close(self) -> None:
        self._close()


def open_camera(index: int, *, ids: Callable[[], list[str]] | None = None, usb: _IOKit | None = None) -> UvcCamera:
    """The controls of the camera OpenCV opens as `index`. Raises `NoControls` or `ControlError`."""
    try:
        found = (ids or video_device_ids)()
    except OSError as exc:
        raise ControlError(f"AVFoundation niedostepne: {exc}") from None
    if index >= len(found):
        raise ControlError(f"brak kamery numer {index} (AVFoundation zna {len(found)})")
    address = usb_address(found[index])
    if address is None:
        raise NoControls("kamera nie jest kamera USB (UVC)")
    try:
        usb = usb or _IOKit.shared()
    except OSError as exc:
        raise ControlError(f"IOKit niedostepny: {exc}") from None
    return usb.camera(*address)


# ------------------------------------------------------------------ AVFoundation
def video_device_ids() -> list[str]:
    """Unique IDs of AVFoundation's video devices, in the order OpenCV numbers cameras.

    OpenCV (4.7 and later) takes video and muxed devices and sorts them by unique ID.
    """
    objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
    avf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
    objc.objc_getClass.restype, objc.objc_getClass.argtypes = c_void_p, [c_char_p]
    objc.sel_registerName.restype, objc.sel_registerName.argtypes = c_void_p, [c_char_p]
    objc.objc_autoreleasePoolPush.restype = c_void_p
    objc.objc_autoreleasePoolPop.argtypes = [c_void_p]
    send_addr = ctypes.cast(objc.objc_msgSend, c_void_p).value or 0

    def send(obj: Any, selector: bytes, restype: Any = c_void_p, *args: tuple[Any, Any]) -> Any:
        fn = ctypes.CFUNCTYPE(restype, c_void_p, c_void_p, *(t for t, _ in args))(send_addr)
        return fn(obj, objc.sel_registerName(selector), *(v for _, v in args))

    pool = objc.objc_autoreleasePoolPush()
    try:
        device_class = objc.objc_getClass(b"AVCaptureDevice")
        ids = []
        for media in ("AVMediaTypeVideo", "AVMediaTypeMuxed"):
            kind = c_void_p.in_dll(avf, media).value
            devices = send(device_class, b"devicesWithMediaType:", c_void_p, (c_void_p, kind))
            count = send(devices, b"count", c_ulong)
            for i in range(int(count)):
                dev = send(devices, b"objectAtIndex:", c_void_p, (c_ulong, i))
                text = send(send(dev, b"uniqueID"), b"UTF8String", c_char_p)
                if isinstance(text, bytes):
                    ids.append(text.decode())
        return sorted(ids)
    finally:
        objc.objc_autoreleasePoolPop(pool)


# ------------------------------------------------------------------ IOKit
class _UUIDBytes(ctypes.Structure):
    _fields_ = [(f"b{i}", c_uint8) for i in range(16)]


class _DevRequest(ctypes.Structure):
    """`IOUSBDevRequest`."""

    _fields_ = [("bmRequestType", c_uint8), ("bRequest", c_uint8), ("wValue", c_uint16), ("wIndex", c_uint16),
                ("wLength", c_uint16), ("pData", c_void_p), ("wLenDone", c_uint32)]


_DEVICE_USER_CLIENT = "9DC7B780-9EC0-11D4-A54F-000A27052861"     # kIOUSBDeviceUserClientTypeID
_PLUGIN_INTERFACE = "C244E858-109C-11D4-91D4-0050E4C6426F"       # kIOCFPlugInInterfaceID
_DEVICE_INTERFACE = "5C8187D0-9EF3-11D4-8B45-000A27052861"       # kIOUSBDeviceInterfaceID (100)
#: Slots in the COM-style vtables (`IUNKNOWN_C_GUTS` first: reserved, QueryInterface, AddRef, Release).
_QUERY_INTERFACE, _RELEASE, _DEVICE_REQUEST = 1, 3, 26
_UTF8 = 0x08000100
_SINT64 = 4


class _IOKit:
    """IOKit and CoreFoundation through ctypes: finds a USB device and sends it control requests."""

    _shared: _IOKit | None = None

    @classmethod
    def shared(cls) -> _IOKit:
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def __init__(self) -> None:
        io = self.io = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/IOKit.framework/IOKit")
        cf = self.cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        io.IOServiceMatching.restype, io.IOServiceMatching.argtypes = c_void_p, [c_char_p]
        io.IOServiceGetMatchingServices.restype = c_int32
        io.IOServiceGetMatchingServices.argtypes = [c_uint32, c_void_p, POINTER(c_uint32)]
        io.IOIteratorNext.restype, io.IOIteratorNext.argtypes = c_uint32, [c_uint32]
        io.IOObjectRelease.restype, io.IOObjectRelease.argtypes = c_int32, [c_uint32]
        io.IORegistryEntryCreateCFProperty.restype = c_void_p
        io.IORegistryEntryCreateCFProperty.argtypes = [c_uint32, c_void_p, c_void_p, c_uint32]
        io.IOCreatePlugInInterfaceForService.restype = c_int32
        io.IOCreatePlugInInterfaceForService.argtypes = [c_uint32, c_void_p, c_void_p, POINTER(c_void_p),
                                                         POINTER(c_int32)]
        io.IODestroyPlugInInterface.restype, io.IODestroyPlugInInterface.argtypes = c_int32, [c_void_p]
        cf.CFStringCreateWithCString.restype = c_void_p
        cf.CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, c_uint32]
        cf.CFNumberGetValue.restype, cf.CFNumberGetValue.argtypes = c_bool, [c_void_p, c_int, c_void_p]
        cf.CFGetTypeID.restype, cf.CFGetTypeID.argtypes = c_ulong, [c_void_p]
        cf.CFNumberGetTypeID.restype = c_ulong
        cf.CFRelease.argtypes = [c_void_p]
        cf.CFUUIDCreateFromString.restype, cf.CFUUIDCreateFromString.argtypes = c_void_p, [c_void_p, c_void_p]
        cf.CFUUIDGetUUIDBytes.restype, cf.CFUUIDGetUUIDBytes.argtypes = _UUIDBytes, [c_void_p]
        # Kept for the life of the process.
        self._keys = {k: self._string(k) for k in ("locationID", "idVendor", "idProduct")}
        self._user_client = cf.CFUUIDCreateFromString(None, self._string(_DEVICE_USER_CLIENT))
        self._plugin = cf.CFUUIDCreateFromString(None, self._string(_PLUGIN_INTERFACE))
        self._device_iid = cf.CFUUIDGetUUIDBytes(cf.CFUUIDCreateFromString(None, self._string(_DEVICE_INTERFACE)))

    def _string(self, text: str) -> int:
        return self.cf.CFStringCreateWithCString(None, text.encode(), _UTF8)

    def _number(self, service: int, key: str) -> int | None:
        ref = self.io.IORegistryEntryCreateCFProperty(service, self._keys[key], None, 0)
        if not ref:
            return None
        try:
            if self.cf.CFGetTypeID(ref) != self.cf.CFNumberGetTypeID():
                return None
            out = c_int64()
            return out.value if self.cf.CFNumberGetValue(ref, _SINT64, byref(out)) else None
        finally:
            self.cf.CFRelease(ref)

    def _service(self, location: int, vendor: int, product: int) -> int:
        """The IOUSBHostDevice at `location`, checked against the vendor and product IDs; 0 when there is none."""
        found = c_uint32()
        kr = self.io.IOServiceGetMatchingServices(0, self.io.IOServiceMatching(b"IOUSBHostDevice"), byref(found))
        if kr:
            raise ControlError(f"IOKit: lista urzadzen USB niedostepna (0x{kr & 0xFFFFFFFF:08x})")
        hit = 0
        try:
            while not hit:
                service = self.io.IOIteratorNext(found.value)
                if not service:
                    break
                if (self._number(service, "locationID"), self._number(service, "idVendor"),
                        self._number(service, "idProduct")) == (location, vendor, product):
                    hit = service
                else:
                    self.io.IOObjectRelease(service)
        finally:
            self.io.IOObjectRelease(found.value)
        return hit

    @staticmethod
    def _method(obj: int, slot: int, restype: type | None, *argtypes: type) -> Callable[..., int]:
        table = ctypes.cast(ctypes.cast(obj, POINTER(c_void_p))[0], POINTER(c_void_p))
        return ctypes.CFUNCTYPE(restype, c_void_p, *argtypes)(table[slot])

    def camera(self, location: int, vendor: int, product: int) -> UvcCamera:
        """The UVC controls of the USB device at `location`. Raises `NoControls` or `ControlError`."""
        service = self._service(location, vendor, product)
        if not service:
            raise ControlError(f"brak urzadzenia USB {vendor:04x}:{product:04x} pod adresem 0x{location:x}")
        plugin, score = c_void_p(), c_int32()
        try:
            kr = self.io.IOCreatePlugInInterfaceForService(service, self._user_client, self._plugin, byref(plugin),
                                                          byref(score))
        finally:
            self.io.IOObjectRelease(service)
        if kr or not plugin.value:
            raise ControlError(f"IOKit: brak dostepu do urzadzenia USB (0x{kr & 0xFFFFFFFF:08x})")
        device = c_void_p()
        try:
            query = self._method(plugin.value, _QUERY_INTERFACE, c_int32, _UUIDBytes, POINTER(c_void_p))
            kr = query(plugin.value, self._device_iid, byref(device))
        finally:
            self.io.IODestroyPlugInInterface(plugin.value)
        if kr or not device.value:
            raise ControlError(f"IOKit: brak interfejsu urzadzenia USB (0x{kr & 0xFFFFFFFF:08x})")
        handle = device.value
        send = self._method(handle, _DEVICE_REQUEST, c_int32, POINTER(_DevRequest))
        release = self._method(handle, _RELEASE, c_uint32)

        def request(kind: int, code: int, value: int, index: int, data: bytes | int) -> bytes:
            out = isinstance(data, int)
            n = data if isinstance(data, int) else len(data)
            buf = (c_uint8 * max(1, n))(*(b"" if isinstance(data, int) else data))
            req = _DevRequest(kind, code, value, index, n, ctypes.cast(buf, c_void_p), 0)
            kr = send(handle, byref(req)) & 0xFFFFFFFF
            if kr == PIPE_STALLED:
                raise ControlError("kamera odrzucila zadanie (STALL)")
            if kr:
                raise ControlError(f"blad USB 0x{kr:08x}")
            return bytes(buf[:req.wLenDone]) if out else b""

        try:
            config = _config_descriptor(request)
            terminal = camera_terminal(config)
        except ControlError:
            release(handle)
            raise
        if terminal is None:
            release(handle)
            raise NoControls("urzadzenie USB bez terminala kamery UVC")

        def close() -> None:
            release(handle)

        return UvcCamera(request, terminal, close=close)


def _config_descriptor(request: Request) -> bytes:
    """The device's first configuration descriptor, with every interface and class descriptor in it."""
    head = request(0x80, 0x06, 0x0200, 0, 9)            # GET_DESCRIPTOR (CONFIGURATION, index 0)
    if len(head) < 4:
        raise ControlError("brak deskryptora konfiguracji USB")
    return request(0x80, 0x06, 0x0200, 0, struct.unpack_from("<H", head, 2)[0])
