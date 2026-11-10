"""Focus lock for the lens step: autofocus off and a fixed focus before any view is taken.

A lens result holds only at the focus it was measured at, so step 1 first turns autofocus off
and sets a fixed focus: the one in the camera's record, or the sharpest one found by stepping
the focus through its range (`sweep`). The chosen focus goes into the record's `controls` and
into the lens result.

    lock = lock_focus(set_controls, grab, wanted=None, focus_range=Range(0, 250, 5))
    lock.focus                 # the focus set and read back; None for a camera without one

`set_controls` sets `CameraControls` on the camera and returns `Applied`; `grab` returns the
newest frame and the moment it was taken (`time.monotonic`).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..camera_controls import Applied, CameraControls, Range

#: The range swept when the camera does not report one (most UVC cameras use 0-255 or less).
DEFAULT_RANGE = Range(0, 250, 5)
#: Values tried on the first pass over the range.
COARSE_STEPS = 11


class FocusLockError(RuntimeError):
    """The focus could not be locked; the message is for the panel."""


@dataclass
class FocusLock:
    """What `lock_focus` did."""

    #: The focus set and read back; None when the camera has no focus to set.
    focus: int | None
    #: Whether autofocus was turned off (False for a camera without autofocus or without controls).
    autofocus_off: bool = False
    #: (focus, sharpness) of each value tried; empty when the focus came from the record.
    swept: list[tuple[int, float]] = field(default_factory=list)
    #: One line for the panel.
    note: str = ""


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian over the middle half of the frame: higher is sharper."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    return float(cv2.Laplacian(gray[h // 4:3 * h // 4, w // 4:3 * w // 4], cv2.CV_64F).var())


def grid(rng: Range, lo: int, hi: int, n: int | None = None) -> list[int]:
    """Values of `rng` between `lo` and `hi` on its step grid: every one, or about `n` spread evenly."""
    step = max(1, rng.step)
    first = rng.lo + -(-(max(lo, rng.lo) - rng.lo) // step) * step
    values = list(range(first, min(hi, rng.hi) + 1, step))
    if n is None or len(values) <= n:
        return values
    picks = np.linspace(0, len(values) - 1, n).round().astype(int)
    return [values[i] for i in sorted(set(picks.tolist()))]


def lock_focus(set_controls: Callable[[CameraControls], Applied],
               grab: Callable[[], tuple[np.ndarray | None, float]], *, wanted: int | None,
               focus_range: Range | None, settle: float = 0.8, samples: int = 3, timeout: float = 5.0,
               cancel: threading.Event | None = None, report: Callable[[str, float], None] = lambda text, part: None,
               clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> FocusLock:
    """Autofocus off, then focus `wanted` (from the camera's record) or the sharpest value of a sweep.

    Raises `FocusLockError` when the camera has a focus that does not lock or read back. A source
    without reachable controls, or a camera without a focus to set, gives a `FocusLock` with
    `focus` None: its lens is measured as before. `report(text, part)` follows the sweep.
    """
    done = set_controls(CameraControls(autofocus=False))
    if done.skipped:
        return FocusLock(None, note=f"ostrosc bez blokady: {done.skipped}")
    af_off = "autofocus" in done.ok
    if "autofocus" in done.failed and "autofocus" not in done.missing:
        raise FocusLockError(f"autofokus nie wylacza sie: {done.failed['autofocus']}")

    def settle_focus(value: int) -> Applied:
        return set_controls(CameraControls(autofocus=False, focus_absolute=value))

    if wanted is not None:
        value, swept, how = wanted, [], "z ustawien kamery"
    else:
        rng = focus_range or DEFAULT_RANGE
        first = settle_focus(rng.lo)
        if "focus_absolute" in first.missing:
            note = "autofokus wylaczony, kamera nie ma recznej ostrosci" if af_off else "kamera ze stala ostroscia"
            return FocusLock(None, autofocus_off=af_off, note=note)
        swept = sweep(settle_focus, grab, rng, settle=settle, samples=samples, timeout=timeout, cancel=cancel,
                      report=report, clock=clock, sleep=sleep)
        value, how = max(swept, key=lambda vs: vs[1])[0], "najostrzejsza z przeszukania"
    final = settle_focus(value)
    if "focus_absolute" in final.missing:
        if wanted is not None:
            raise FocusLockError(f"kamera nie ma recznej ostrosci, a w ustawieniach jest {wanted}")
        return FocusLock(None, autofocus_off=af_off, note="kamera nie ma recznej ostrosci")
    if "focus_absolute" not in final.ok:
        why = final.failed.get("focus_absolute") or final.skipped or "brak odczytu"
        raise FocusLockError(f"ostrosc {value} nie ustawia sie: {why}")
    return FocusLock(value, autofocus_off=True, swept=swept, note=f"ostrosc {value} ({how})")


def sweep(set_focus: Callable[[int], Applied], grab: Callable[[], tuple[np.ndarray | None, float]], rng: Range, *,
          settle: float = 0.8, samples: int = 3, timeout: float = 5.0, cancel: threading.Event | None = None,
          report: Callable[[str, float], None] = lambda text, part: None,
          clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
          ) -> list[tuple[int, float]]:
    """(focus, sharpness) over the range: a coarse pass, then every step around the sharpest value.

    Each value is set, given `settle` seconds, and scored on the median of `samples` frames taken
    after that. Raises `FocusLockError` when no frame comes, the operator cancels, or the image
    hardly changes with the focus (nothing in view to focus on).
    """
    coarse = grid(rng, rng.lo, rng.hi, COARSE_STEPS)
    scores: dict[int, float] = {}

    def score(values: list[int], start: float, share: float) -> None:
        for k, value in enumerate(values):
            if value in scores:
                continue
            if cancel is not None and cancel.is_set():
                raise FocusLockError("przerwane")
            report(f"szukam ostrosci: {value} ({rng.lo}-{rng.hi})", start + share * k / max(1, len(values)))
            done = set_focus(value)
            if "focus_absolute" not in done.ok:
                continue                                # a value the camera refuses is left out
            scores[value] = _measure(grab, clock() + settle, samples, timeout, clock, sleep)

    score(coarse, 0.0, 0.6)
    if not scores:
        raise FocusLockError("kamera nie przyjela zadnej wartosci ostrosci")
    best = max(scores, key=scores.__getitem__)
    gap = coarse[1] - coarse[0] if len(coarse) > 1 else rng.step
    score(grid(rng, best - gap + 1, best + gap - 1), 0.6, 0.4)
    out = sorted(scores.items())
    top, low = max(s for _, s in out), min(s for _, s in out)
    if top <= 0 or (top - low) / top < 0.1:
        raise FocusLockError("obraz prawie sie nie zmienia z ostroscia - poloz szachownice w srodku kadru")
    return out


def _measure(grab: Callable[[], tuple[np.ndarray | None, float]], after: float, samples: int, timeout: float,
             clock: Callable[[], float], sleep: Callable[[float], None]) -> float:
    """Median sharpness of `samples` distinct frames taken after the moment `after`."""
    deadline, seen, values = after + timeout, set(), []
    while len(values) < samples:
        now = clock()
        if now > deadline:
            raise FocusLockError("kamera nie daje nowych klatek")
        if now < after:
            sleep(after - now)
            continue
        img, t = grab()
        if img is not None and t >= after and t not in seen:
            seen.add(t)
            values.append(sharpness(img))
        else:
            sleep(0.03)
    return float(np.median(values))
