"""The model's answer to Detections: checked item by item, scaled back to the camera's frame.

The model answers in the pixels of the image it was sent. A frame scaled down by `encode` maps back
with pixel centres kept: `u' = (u + 0.5) * W / w - 0.5`. An item that is malformed, names a camera
that was not sent, lies outside its image or is less sure than asked is dropped, never guessed at.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from machinekind_twin_api.detector import Detection

from .images import SentImage
from .prompt import normalise_label

__all__ = ["MAX_DETECTIONS", "MAX_LABEL", "parse_answer"]

logger = logging.getLogger(__name__)

#: Detections kept from one answer, surest first.
MAX_DETECTIONS = 30
#: Longest label accepted [characters].
MAX_LABEL = 64
#: How far outside its image a pixel may lie and still count [px]: -2..w+1.
_MARGIN_LO, _MARGIN_HI = 2.0, 1.0


def _number(value: Any) -> float | None:
    """A finite float, or None for anything else (bools, strings, NaN, infinities)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _wanted_label(label: str, wanted: Sequence[str]) -> str | None:
    """The wanted name `label` stands for: equal to it, or ending with it as a whole word
    ("the red pen" is "red pen"); the longest such name wins. None when no wanted name fits."""
    if label in wanted:
        return label
    fits = [w for w in wanted if label.endswith(" " + w)]
    return max(fits, key=len) if fits else None


def _cameras(sent: Sequence[SentImage]) -> tuple[dict[str, SentImage], dict[str, SentImage]]:
    """Sent images by exact camera name and by lower-cased name (the latter only where unique)."""
    exact = {s.camera: s for s in sent}
    folded: dict[str, list[SentImage]] = {}
    for s in sent:
        folded.setdefault(s.camera.casefold(), []).append(s)
    return exact, {k: v[0] for k, v in folded.items() if len(v) == 1}


def parse_answer(raw: Any, sent: Sequence[SentImage], wanted: Sequence[str] = (), min_confidence: float = 0.0,
                 pixel_error_px: float = 25.0) -> list[Detection]:
    """The Detections in `raw`, the model's structured answer, in the pixels of each camera's frame.

    `wanted` are the labels asked for; when given, an item must name one of them (or end with one as
    a whole word) and is reported under that name. `pixel_error_px` is in the SENT image's pixels;
    each Detection carries it scaled to its frame. Raises ValueError when `raw` has no objects list.
    """
    objects = raw.get("objects") if isinstance(raw, Mapping) else None
    if not isinstance(objects, list):
        raise ValueError("answer has no objects list")
    exact, folded = _cameras(sent)
    wants = list(dict.fromkeys(w for w in map(normalise_label, wanted) if w))
    dropped: Counter[str] = Counter()
    out: list[Detection] = []
    for item in objects:
        reason, det = _detection(item, exact, folded, wants, min_confidence, pixel_error_px)
        if det is None:
            dropped[reason] += 1
        else:
            out.append(det)
    if dropped:
        logger.debug("dropped %d of %d answer items: %s", sum(dropped.values()), len(objects),
                     ", ".join(f"{k} {n}" for k, n in sorted(dropped.items())))
    out.sort(key=lambda d: -d.confidence)
    return out[:MAX_DETECTIONS]


def _detection(item: Any, exact: Mapping[str, SentImage], folded: Mapping[str, SentImage], wanted: Sequence[str],
               min_confidence: float, pixel_error_px: float) -> tuple[str, Detection | None]:
    """One answer item as a Detection, or (why it was dropped, None)."""
    if not isinstance(item, Mapping):
        return "not an object", None
    label = item.get("label")
    label = normalise_label(label) if isinstance(label, str) else ""
    if not label or len(label) > MAX_LABEL:
        return "bad label", None
    camera = item.get("camera")
    s = (exact.get(camera) or folded.get(camera.casefold())) if isinstance(camera, str) else None
    if s is None:
        return "unknown camera", None
    u, v = _number(item.get("u")), _number(item.get("v"))
    if u is None or v is None:
        return "bad pixel", None
    box_raw = item.get("box")
    box: tuple[float, float, float, float] | None = None
    if box_raw is not None:
        corners = [_number(box_raw.get(k)) for k in ("u0", "v0", "u1", "v1")] if isinstance(box_raw, Mapping) else []
        if len(corners) != 4 or any(c is None for c in corners):
            return "bad box", None
        u0, v0, u1, v1 = (float(c) for c in corners if c is not None)
        box = (u0, v0, u1, v1)
    w, h = s.sent_size
    if not (-_MARGIN_LO <= u <= w + _MARGIN_HI and -_MARGIN_LO <= v <= h + _MARGIN_HI):
        return "outside the image", None
    confidence = _number(item.get("confidence"))
    if confidence is None:
        return "bad confidence", None
    if confidence < min_confidence:
        return "unsure", None
    if wanted:
        named = _wanted_label(label, wanted)
        if named is None:
            return "not asked for", None
        label = named

    fw, fh = s.frame_size
    sx, sy = fw / w, fh / h

    def back(x: float, scale: float, size: int) -> float:
        return min(max((x + 0.5) * scale - 0.5, 0.0), float(size - 1))

    pixel = (back(u, sx, fw), back(v, sy, fh))
    if box is not None:
        a, b = back(box[0], sx, fw), back(box[2], sx, fw)
        c, d = back(box[1], sy, fh), back(box[3], sy, fh)
        box = (min(a, b), min(c, d), max(a, b), max(c, d))
    return "", Detection(label=label, camera=s.camera, pixel=pixel, box=box,
                         confidence=min(max(confidence, 0.0), 1.0),
                         pixel_error_px=pixel_error_px * max(sx, sy))
