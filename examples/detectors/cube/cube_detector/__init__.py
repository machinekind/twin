"""The twin's example detector: a coloured cube found by colour; the twin fits its pose to the silhouettes.

The twin finds `ColourCube` through the entry point in this package's `pyproject.toml`, builds
`CubeSettings` from the panel's form, which the config file's `cube:` section prefills, and
creates the detector as `ColourCube(settings)`. The package depends on the contracts
(`machinekind-twin-api`), never on the twin.

The detector only thresholds colour. Per camera it answers with the cube's silhouette mask, its
centre in the image and its edge length; the twin fits a cube of that size to the masks of every
camera, which gives the position, the yaw and the final confidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from machinekind_twin_api import Detection, Detector, DetectRequest

__all__ = ["COLORS", "ColourCube", "CubeSettings"]

#: HSV ranges per colour, OpenCV convention (H 0..180): a pixel counts when it is in any range.
#: Red wraps around H = 0, so it takes one range on each side of the circle.
COLORS: dict[str, tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]] = {
    "red": (((0, 120, 70), (12, 255, 255)), ((170, 120, 70), (180, 255, 255))),
    "green": (((40, 80, 60), (85, 255, 255)),),
    "blue": (((95, 100, 60), (130, 255, 255)),),
    "yellow": (((20, 100, 90), (35, 255, 255)),),
}

#: The largest edge the detector accepts [m].
MAX_SIZE_M = 0.2
#: The shortest interval between calls [s].
MIN_INTERVAL_S = 0.05


@dataclass
class CubeSettings:
    #: red, green, blue or yellow.
    color: str = "red"
    #: Edge length [m].
    size_m: float = 0.03
    #: The key strategies read it under (soarm_lab's policy reads "cube").
    label: str = "cube"
    #: Smallest blob that counts [px].
    min_area_px: int = 20
    #: Seconds between calls (the Map tab used 0.33 s).
    interval_s: float = 0.33


class ColourCube(Detector[CubeSettings]):
    """A cube of one colour, found by an HSV threshold; the twin fits its pose to the masks."""

    settings = CubeSettings
    #: The blob's centroid against the footprint's centre; the silhouette fit does the rest.
    pixel_error_px = 4.0

    def __init__(self, settings: CubeSettings):
        if settings.color not in COLORS:
            raise ValueError(f"cube: unknown color {settings.color!r} (known: {', '.join(COLORS)})")
        if not (math.isfinite(settings.size_m) and 0.0 < settings.size_m <= MAX_SIZE_M):
            raise ValueError(f"cube: size_m must be in (0, {MAX_SIZE_M}] m, got {settings.size_m}")
        if not (math.isfinite(settings.interval_s) and settings.interval_s >= MIN_INTERVAL_S):
            raise ValueError(f"cube: interval_s must be at least {MIN_INTERVAL_S} s, got {settings.interval_s}")

        self.config = settings
        self.interval_s = float(settings.interval_s)
        self._ranges = [(np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in COLORS[settings.color]]

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        """The colour's pixels in an RGB image: uint8 (height, width), 255 on the colour, opened 3x3."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in self._ranges:
            m |= np.asarray(cv2.inRange(hsv, lo, hi), np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def detect(self, request: DetectRequest) -> Sequence[Detection]:
        config = self.config
        size = float(config.size_m)
        found: list[Detection] = []
        for name in request.frames:
            frame = request.frames.get(name)
            if frame is None:
                continue
            m = self.mask(frame.image) > 0
            if int(m.sum()) < config.min_area_px:
                continue
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            blob = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(blob)
            mom = cv2.moments(blob)
            if mom["m00"] > 0:
                pixel = (mom["m10"] / mom["m00"], mom["m01"] / mom["m00"])
            else:
                pixel = (x + (w - 1) / 2.0, y + (h - 1) / 2.0)
            found.append(Detection(config.label, name, (float(pixel[0]), float(pixel[1])),
                                   box=(float(x), float(y), float(x + w - 1), float(y + h - 1)),
                                   confidence=1.0, mask=m, size_m=(size, size, size)))
        return found
