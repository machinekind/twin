"""What the operator sets for the LLM detector, and the checks those settings must pass.

The twin builds `LlmSettings` from the panel's form, which the config file's `llm:` section
prefills, and creates the detector as `LlmDetector(settings)`. `validate` runs in the detector's
constructor, so a setting the detector cannot work with is refused before anything starts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["CLAUDE_IMAGES", "CLIS", "EFFORTS", "MODELS", "LlmSettings", "names", "validate"]

#: The CLIs the detector can ask.
CLIS = ("codex", "claude")
#: Model picker choices by CLI; "" uses the CLI's default. Configs may name other models.
MODELS = {
    "codex": ("", "gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol",
              "gpt-5.6-terra", "gpt-5.6-luna"),
    "claude": ("", "sonnet", "opus", "haiku", "fable", "claude-sonnet-5-5", "claude-opus-5-5",
               "claude-haiku-5-5", "claude-fable-5-1"),
}
#: Reasoning efforts passed to the CLI; "" = the CLI's default.
EFFORTS = ("", "low", "medium", "high", "xhigh", "max")
#: How claude gets the images: inline over stream-json, or as files it opens with Read.
CLAUDE_IMAGES = ("stream", "read")
#: claude's model aliases; any "claude-..." name counts too.
CLAUDE_ALIASES = ("sonnet", "opus", "haiku", "fable")

#: Ranges the numbers must lie in, inclusive.
MIN_INTERVAL_S = 2.0
TIMEOUT_S = (5.0, 900.0)
MAX_SIDE_PX = (256, 4096)
PIXEL_ERROR_PX = (0.0, 500.0)
REUSE_MAX_AGE_S = (0.0, 3600.0)


@dataclass
class LlmSettings:
    #: Which locally installed CLI answers, on your subscription: "codex" or "claude".
    cli: str = field(default="codex", metadata={"choices": CLIS})
    #: The model as the CLI names it: "gpt-6-astra" (Astra), "sonnet", "claude-opus-5-5"; "" = the
    #: CLI's default.
    model: str = field(default="gpt-6-astra", metadata={"choices_from": "cli", "choices": MODELS})
    #: Reasoning effort passed to the CLI: low, medium, high, xhigh, max; "" = the CLI's default.
    effort: str = "low"
    #: What to look for, comma-separated ("red pen, cube"); "" = every movable object on the table.
    objects: str = ""
    #: Cameras to send, comma-separated; "" = every camera the twin offers in each call.
    cameras: str = ""
    #: Extra instructions appended to the built-in prompt.
    prompt: str = ""
    #: Shortest time between the starts of two questions [s]; at least 2.
    interval_s: float = 10.0
    #: Kill the CLI after this many seconds.
    timeout_s: float = 120.0
    #: Long edge of the images sent [px]; frames are scaled down, answers scaled back. Claude
    #: downscales above about 1568 px itself.
    max_side_px: int = 1280
    #: How far off the model's pixels may be, in the SENT image's pixels; scaled to each frame.
    pixel_error_px: float = 25.0
    #: Repeat the last answer, without asking, while no camera's image changed since the frames it
    #: was given, and for up to two calls while the scene still moves; the arm moving over the table
    #: is no change (see `images`).
    reuse_if_still: bool = True
    #: Ask again anyway once the last answer is this old [s], so a wrong answer does not stick.
    reuse_max_age_s: float = 60.0
    #: Ask at most this many times per hour; 0 = no cap.
    max_calls_per_hour: int = 0
    #: Drop answers the model is less sure of than this.
    min_confidence: float = 0.3
    #: The CLI binary; "" = `cli` found on PATH. Point it at ~/.local/bin/<cli> to bypass terminal
    #: shims.
    executable: str = ""
    #: claude only: "stream" sends images inline over stream-json; "read" names temp files the
    #: model opens with Read.
    claude_images: str = "stream"


def names(text: str) -> tuple[str, ...]:
    """A comma-separated setting as its items, stripped, empty ones and repeats left out."""
    return tuple(dict.fromkeys(item.strip() for item in text.split(",") if item.strip()))


def _within(value: float, lo: float, hi: float) -> bool:
    return math.isfinite(value) and lo <= value <= hi


def validate(settings: LlmSettings) -> None:
    """Refuse unsupported settings before the detector starts."""
    s = settings
    if s.cli not in CLIS:
        raise ValueError(f"llm: cli must be one of {', '.join(CLIS)}, not {s.cli!r}")
    if s.effort not in EFFORTS:
        raise ValueError(f"llm: effort must be one of {', '.join(e or repr(e) for e in EFFORTS)}, not {s.effort!r}")
    if not (math.isfinite(s.interval_s) and s.interval_s >= MIN_INTERVAL_S):
        raise ValueError(f"llm: interval_s must be at least {MIN_INTERVAL_S:g} s, got {s.interval_s}")
    if not _within(s.timeout_s, *TIMEOUT_S):
        raise ValueError(f"llm: timeout_s must be in [{TIMEOUT_S[0]:g}, {TIMEOUT_S[1]:g}] s, got {s.timeout_s}")
    if not MAX_SIDE_PX[0] <= s.max_side_px <= MAX_SIDE_PX[1]:
        raise ValueError(f"llm: max_side_px must be in [{MAX_SIDE_PX[0]}, {MAX_SIDE_PX[1]}] px, got {s.max_side_px}")
    if not _within(s.pixel_error_px, *PIXEL_ERROR_PX):
        raise ValueError(f"llm: pixel_error_px must be in [{PIXEL_ERROR_PX[0]:g}, {PIXEL_ERROR_PX[1]:g}] px, "
                         f"got {s.pixel_error_px}")
    if not _within(s.reuse_max_age_s, *REUSE_MAX_AGE_S):
        raise ValueError(f"llm: reuse_max_age_s must be in [{REUSE_MAX_AGE_S[0]:g}, {REUSE_MAX_AGE_S[1]:g}] s, "
                         f"got {s.reuse_max_age_s}")
    if not _within(s.min_confidence, 0.0, 1.0):
        raise ValueError(f"llm: min_confidence must be in [0, 1], got {s.min_confidence}")
    if s.max_calls_per_hour < 0:
        raise ValueError(f"llm: max_calls_per_hour must be 0 (no cap) or more, got {s.max_calls_per_hour}")
    if s.claude_images not in CLAUDE_IMAGES:
        raise ValueError(f"llm: claude_images must be one of {', '.join(CLAUDE_IMAGES)}, not {s.claude_images!r}")
    model = s.model.strip().lower()
    if s.cli == "claude" and model.startswith("gpt-"):
        raise ValueError(f"llm: model {s.model!r} is an OpenAI model and cli is claude; use a claude model "
                         "(sonnet, opus, ...) or \"\" for claude's default")
    if s.cli == "codex" and (model in CLAUDE_ALIASES or model.startswith("claude-")):
        raise ValueError(f"llm: model {s.model!r} is a Claude model and cli is codex; use an OpenAI model "
                         "(gpt-6-astra, ...) or \"\" for codex's default")
