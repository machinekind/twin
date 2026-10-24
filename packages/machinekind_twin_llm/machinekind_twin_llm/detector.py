"""The LLM detector: each call sends the cameras' frames to the claude or codex CLI and reads back
what is where, in each frame's pixels.

`start` finds the CLI and refuses one that is not logged in or logged in with an API key. `detect`
asks one question per call, in a call directory of its own under the detector's work directory.
With `reuse_if_still` it repeats the last answer instead while no camera's image changed since the
frames of the last question. It also repeats it while the scene still moves: frames that changed
since the last question and since the previous call wait, for up to `SETTLE_MAX_SKIPS` calls, so a
hand passing through is not asked about mid-motion. Where the twin's arm masks hide the scene, each
camera keeps what its earlier frames showed there (`images.known_scene`), so the arm moving over the
table is no change while an object it leaves behind is. A question over the hourly cap fails. `stop`
cancels a running question: the CLI's process group is killed within 0.2 s.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from machinekind_twin_api import Detection, Detector, DetectorInfo, DetectRequest

from . import cli
from .answer import parse_answer
from .images import arm_thumbnail, encode, known_scene, thumbnail, unchanged
from .prompt import SCHEMA, build_prompt
from .settings import LlmSettings, names, validate

__all__ = ["CALL_WINDOW_S", "SETTLE_MAX_SKIPS", "LlmDetector"]

logger = logging.getLogger(__name__)

#: The window `max_calls_per_hour` counts questions in [s].
CALL_WINDOW_S = 3600.0
#: Consecutive calls a moving scene may repeat the last answer for before it is asked about anyway.
SETTLE_MAX_SKIPS = 2


class LlmDetector(Detector[LlmSettings]):
    """Objects a vision model names and places, asked through the claude or codex CLI on your subscription."""

    settings = LlmSettings

    def __init__(self, settings: LlmSettings | None = None):
        settings = settings if settings is not None else LlmSettings()
        validate(settings)
        self.config = settings
        self.interval_s = float(settings.interval_s)
        # A fallback only: each Detection carries the error scaled to its own frame.
        self.pixel_error_px = float(settings.pixel_error_px)
        self._cameras = names(settings.cameras)
        self._objects = names(settings.objects)
        self._exe = ""
        self._work: Path | None = None
        self._cancel = threading.Event()
        self._calls: deque[float] = deque()
        #: The still check, by camera (`images.known_scene`): what the frames showed when the last
        #: question was asked and at the previous call; then the calls repeated in a row while the
        #: scene moved, and the last answer and when it was asked.
        self._ref: dict[str, np.ndarray] = {}
        self._prev: dict[str, np.ndarray] = {}
        self._moving = 0
        self._last: list[Detection] = []
        self._last_t = -float("inf")

    def start(self, info: DetectorInfo) -> None:
        config = self.config
        if config.executable:
            given = str(Path(config.executable).expanduser())
            exe = shutil.which(given)
            if exe is None:
                raise RuntimeError(f"{given} not found or not executable; fix `executable` or leave it empty")
        else:
            exe = shutil.which(config.cli)
            if exe is None:
                raise RuntimeError(f"{config.cli} not found on PATH; install it or set `executable`")
        # Calls change working directory; keep the executable's original symlink name.
        exe = str(Path(exe).absolute())
        unknown = [name for name in self._cameras if name not in info.cameras]
        if unknown:
            raise ValueError(f"llm: unknown camera(s) in `cameras`: {', '.join(unknown)}; "
                             f"available: {', '.join(info.cameras) or 'none'}")
        method = cli.check_login(config.cli, exe)
        logger.info("llm: %s at %s, logged in as %s", config.cli, exe, method)
        self._exe = exe
        self._work = Path(tempfile.mkdtemp(prefix="twin-llm-"))
        self._cancel = threading.Event()
        self._calls.clear()
        self._ref, self._prev, self._moving = {}, {}, 0
        self._last, self._last_t = [], -float("inf")

    def stop(self, reason: str) -> None:
        self._cancel.set()
        if self._work is not None:
            shutil.rmtree(self._work, ignore_errors=True)

    def detect(self, request: DetectRequest) -> Sequence[Detection]:
        if self._work is None:
            raise RuntimeError("llm: detect before start")
        if self._cancel.is_set():
            raise cli.CliCancelled("stopped")
        config = self.config
        cameras = self._cameras or tuple(request.frames)
        frames: list[tuple[str, np.ndarray]] = []
        for name in cameras:
            frame = request.frames.get(name)
            if frame is not None:
                frames.append((name, frame.image))
        if not frames:
            raise RuntimeError(f"no fresh frame from cameras {', '.join(cameras) or '(none offered)'}")

        now = time.monotonic()
        scene = self._scene(request, frames) if config.reuse_if_still else {}
        reason = self._repeat_reason(scene, now)
        self._prev = scene
        if reason:
            logger.debug("llm: %s, repeating the answer from %.0f s ago", reason, now - self._last_t)
            return list(self._last)

        if config.max_calls_per_hour:
            while self._calls and now - self._calls[0] >= CALL_WINDOW_S:
                self._calls.popleft()
            if len(self._calls) >= config.max_calls_per_hour:
                raise RuntimeError(f"hourly cap of {config.max_calls_per_hour} questions reached")
        self._calls.append(now)

        try:
            with cli.call_dir(self._work) as call:
                sent = encode(call, frames, config.max_side_px)
                if config.cli == "claude":
                    prompt = build_prompt(sent, self._objects, config.prompt,
                                          image_ref="path" if config.claude_images == "read" else "order")
                    raw = cli.ask_claude(self._exe, sent, prompt, SCHEMA, cwd=call, model=config.model,
                                         effort=config.effort, route=config.claude_images,
                                         timeout_s=config.timeout_s, cancel=self._cancel)
                else:
                    prompt = build_prompt(sent, self._objects, config.prompt)
                    raw = cli.ask_codex(self._exe, sent, prompt, SCHEMA, cwd=call, model=config.model,
                                        effort=config.effort, timeout_s=config.timeout_s, cancel=self._cancel)
        except cli.CliCancelled:
            raise
        except Exception as exc:
            if self._cancel.is_set():  # stop removed the work directory under a running call
                raise cli.CliCancelled("stopped") from exc
            raise
        if self._cancel.is_set():
            raise cli.CliCancelled("stopped")

        dets = parse_answer(raw, sent, self._objects, config.min_confidence, config.pixel_error_px)
        self._ref, self._last, self._last_t, self._moving = scene, dets, now, 0
        return list(dets)

    def _scene(self, request: DetectRequest, frames: list[tuple[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """What each camera's frames have shown up to this call: the thumbnail of the frame sent,
        with the pixels its arm mask hides kept from the previous call. A mask of another shape than
        its frame is ignored."""
        scene: dict[str, np.ndarray] = {}
        for name, image in frames:
            thumb = thumbnail(image)
            mask = request.arm_masks.get(name)
            arm = arm_thumbnail(mask, thumb.shape) if mask is not None and mask.shape == image.shape[:2] else None
            scene[name] = known_scene(thumb, arm, self._prev.get(name))
        return scene

    def _repeat_reason(self, scene: dict[str, np.ndarray], now: float) -> str:
        """Why this call repeats the last answer instead of asking; "" = ask. Only an answer
        restarts the count of repeats for motion, so failed questions in a moving scene keep the
        runner's backoff."""
        if not scene or now - self._last_t >= self.config.reuse_max_age_s:
            return ""
        if unchanged(self._ref, scene):
            self._moving = 0
            return "no camera's image changed"
        if self._moving < SETTLE_MAX_SKIPS and not unchanged(self._prev, scene):
            self._moving += 1
            return f"the scene still moves ({self._moving} of {SETTLE_MAX_SKIPS})"
        return ""
