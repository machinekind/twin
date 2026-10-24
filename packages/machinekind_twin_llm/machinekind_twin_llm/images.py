"""The images a question sends: one JPEG per camera, scaled down from the camera's frame."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["JPEG_QUALITY", "SentImage", "encode"]

#: JPEG quality of the images sent.
JPEG_QUALITY = 90


@dataclass(frozen=True)
class SentImage:
    """One camera's image as the model gets it."""

    #: The camera's name in the twin. Never part of the file name.
    camera: str
    #: The JPEG in the call directory, `img-<k>.jpg`.
    path: Path
    #: (width, height) of the JPEG [px].
    sent_size: tuple[int, int]
    #: (width, height) of the camera's frame the JPEG was scaled from [px].
    frame_size: tuple[int, int]


def _rgb(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3 or img.shape[0] < 1 or img.shape[1] < 1:
        raise ValueError(f"expected an RGB image of shape (height, width, 3), got {img.shape}")
    return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)


def encode(call: Path, frames: Sequence[tuple[str, np.ndarray]], max_side_px: int) -> list[SentImage]:
    """Write each (camera, RGB frame) as `call/img-<k>.jpg`, k from 1, in the given order.

    A frame whose long edge exceeds `max_side_px` is scaled down with INTER_AREA, keeping its
    aspect ratio. A smaller one is sent at its own size. File names never contain camera names.
    """
    if max_side_px < 1:
        raise ValueError(f"max_side_px must be at least 1, not {max_side_px}")
    sent = []
    for i, (camera, image) in enumerate(frames):
        img = _rgb(image)
        h, w = img.shape[:2]
        scale = min(1.0, max_side_px / max(w, h))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if scale < 1.0:
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        path = call / f"img-{i + 1}.jpg"
        if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise OSError(f"could not write {path}")
        sent.append(SentImage(camera, path, (int(bgr.shape[1]), int(bgr.shape[0])), (int(w), int(h))))
    return sent
