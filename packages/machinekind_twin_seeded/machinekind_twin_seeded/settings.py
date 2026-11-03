"""What the operator sets for the seeded detector, how a label becomes a block, and the checks.

The twin builds `SeededSettings` from the panel's form, which the config file's `seeded:` section
prefills, and creates the detector as `SeededDetector(settings)`. `validate` runs in the
detector's constructor, so a setting the detector cannot work with is refused before anything
starts. The settings of the LLM detector it asks are mirrored here (`llm_settings`); that
detector's own checks run too, and their "llm: ..." messages pass through.

A label is "<colour> <shape>", as the model names a block ("red cube", "light blue cuboid",
"wooden cylinder"). The shape word gives the block's size (`SHAPES`), the colour word its OpenCV
HSV ranges (`COLOURS`). Two override tables, entries separated by `;`, take precedence:

* `labels_sizes`: `name=LxWxH`, the block's box as it rests [m], H its height; a lying cylinder
  as its bounding box (length x diameter x diameter);
* `labels_hsv`: `name=H0-H1,S0-S1,V0-V1`, OpenCV HSV ranges (H 0..180, S and V 0..255); several
  ranges of one name joined by `|` (red wraps around H = 0 and needs two).

`name` is a whole label, which wins, or a shape or colour word. Labels are keyed as the twin keys
them: inner whitespace collapsed, stripped, lower-cased.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from machinekind_twin_llm.prompt import normalise_label
from machinekind_twin_llm.settings import CLIS, MODELS, LlmSettings, names

__all__ = [
    "COLOURS", "COLOUR_WORDS", "SHAPES", "SHAPE_WORDS", "Block", "HsvRange", "SeededSettings", "Size",
    "Tables", "colour_word", "llm_settings", "naming_prompt", "parse_hsv", "parse_sizes", "resolve",
    "shape_word", "validate",
]

#: A block's box as it rests: (L, W, H) [m].
Size = tuple[float, float, float]
#: One HSV range: ((H0, S0, V0), (H1, S1, V1)), both ends included.
HsvRange = tuple[tuple[int, int, int], tuple[int, int, int]]

#: The shortest time between two colour-tracking calls [s]; the twin calls no faster anyway.
MIN_INTERVAL_S = 0.1
#: The shortest time between two LLM questions [s]; the LLM detector's own minimum.
MIN_SEED_INTERVAL_S = 2.0
#: The longest side a block may have [m].
MAX_SIZE_M = 0.3
#: The range of `seed_error_px` [px]; the LLM detector's `pixel_error_px` range.
SEED_ERROR_PX = (0.0, 500.0)
#: The range of `track_error_px` [px].
TRACK_ERROR_PX = (1.0, 500.0)
#: The largest H in OpenCV's HSV for 8-bit images (a half of the degrees).
H_MAX = 180
#: The largest S and V.
SV_MAX = 255

#: The blocks by shape, as they rest: (L, W, H) [m]. A cuboid lies flat, a cylinder on its side.
SHAPES: dict[str, Size] = {
    "cube": (0.029, 0.029, 0.029),
    "cuboid": (0.060, 0.029, 0.014),
    "cylinder": (0.060, 0.029, 0.029),
}
#: Words a label may use for a shape.
SHAPE_WORDS = {"cube": "cube", "cuboid": "cuboid", "brick": "cuboid", "cylinder": "cylinder"}
#: OpenCV HSV ranges by colour, as `labels_hsv` writes them. Paint is saturated; bare wood is
#: warm, pale and bright, and is only followed where it stands out from the table.
COLOURS = {
    "red": "0-10,100-255,60-255|170-180,100-255,60-255",
    "yellow": "18-35,110-255,90-255",
    "green": "40-85,70-255,40-255",
    "light blue": "86-125,50-255,70-255",
    "wooden": "5-25,25-105,120-255",
}
#: Words a label may use for a colour.
COLOUR_WORDS = {
    "red": "red", "yellow": "yellow", "green": "green", "light blue": "light blue", "blue": "light blue",
    "wooden": "wooden", "wood": "wooden", "natural": "wooden", "unpainted": "wooden",
}


@dataclass
class SeededSettings:
    #: Which locally installed CLI the seed question goes to, on your subscription: "codex" or "claude".
    cli: str = field(default="codex", metadata={"choices": CLIS})
    #: The model as the CLI names it: "gpt-6-astra", "sonnet", "claude-opus-5-5"; "" = the CLI's default.
    model: str = field(default="gpt-6-astra", metadata={"choices_from": "cli", "choices": MODELS})
    #: Reasoning effort passed to the CLI: low, medium, high, xhigh, max; "" = the CLI's default.
    effort: str = "low"
    #: The blocks to follow, comma-separated, each "<colour> <shape>" ("red cube, wooden cuboid");
    #: "" = every block the model names that way.
    labels: str = ""
    #: Size overrides, `name=LxWxH` [m] with `name` a label or a shape word, `;` between entries;
    #: "" = the shapes' sizes.
    labels_sizes: str = ""
    #: Colour overrides, `name=H0-H1,S0-S1,V0-V1` in OpenCV's HSV (H 0..180, S and V 0..255) with
    #: `name` a label or a colour word, several ranges joined by `|`, `;` between entries.
    labels_hsv: str = ""
    #: Cameras to track and to send, comma-separated; "" = every camera the twin offers.
    cameras: str = ""
    #: Extra instructions appended to the LLM's prompt.
    prompt: str = ""
    #: Shortest time between two colour-tracking calls [s]; at least 0.1.
    interval_s: float = 0.2
    #: Shortest time between the starts of two LLM questions [s]; at least 2.
    seed_interval_s: float = 10.0
    #: How far off the model's pixel may be, in the SENT image's pixels [px]; scaled to each frame, it
    #: is the radius within which a blob takes a fresh seed.
    seed_error_px: float = 25.0
    #: How far a followed blob may move between two calls, in the frame's pixels [px]; 1 to 500.
    track_error_px: float = 40.0
    #: A track without a blob for longer than this is dropped and waits for the next seed [s].
    lost_after_s: float = 3.0
    #: Smallest connected patch of the colour that counts as a blob [px].
    min_area_px: int = 30
    #: A blob whose long side exceeds this times the long side of the model's box is not one block
    #: (two touching blocks of one colour, or a block the colour of the table) [-]; at least 1.
    max_blob_ratio: float = 2.0
    #: Kill the CLI after this many seconds [s].
    timeout_s: float = 120.0
    #: Long edge of the images sent [px]; frames are scaled down, answers scaled back.
    max_side_px: int = 1280
    #: Ask at most this many times per hour; 0 = no cap.
    max_calls_per_hour: int = 0
    #: Seeds the model is less sure of than this are dropped [-].
    min_confidence: float = 0.3
    #: The CLI binary; "" = `cli` found on PATH.
    executable: str = ""
    #: claude only: "stream" sends images inline over stream-json; "read" names temp files the
    #: model opens with Read.
    claude_images: str = "stream"


@dataclass(frozen=True)
class Block:
    """What the detector follows a label as: its size and its colour's HSV ranges."""

    size: Size
    hsv: tuple[HsvRange, ...]


@dataclass(frozen=True)
class Tables:
    """The checked settings: the labels asked for and the override tables, keys normalised."""

    #: The labels asked for; empty = every block the model names.
    labels: tuple[str, ...]
    sizes: dict[str, Size]
    hsv: dict[str, tuple[HsvRange, ...]]


def shape_word(label: str) -> tuple[str, str]:
    """(the words before the shape, the shape) of a label, the shape being the last word naming one;
    ("label", "") without one."""
    words = normalise_label(label).split()
    for i in range(len(words) - 1, -1, -1):
        shape = SHAPE_WORDS.get(words[i])
        if shape:
            return " ".join(words[:i]), shape
    return " ".join(words), ""


def colour_word(phrase: str) -> str:
    """The colour a phrase names: the whole phrase, else its last word naming a colour; "" for none."""
    phrase = normalise_label(phrase)
    if phrase in COLOUR_WORDS:
        return COLOUR_WORDS[phrase]
    for word in reversed(phrase.split()):
        if word in COLOUR_WORDS:
            return COLOUR_WORDS[word]
    return ""


def _key(name: str) -> str:
    """An override's key: a shape or colour word as its canonical word, else the normalised label."""
    name = normalise_label(name)
    return SHAPE_WORDS.get(name) or COLOUR_WORDS.get(name) or name


def _entries(text: str, setting: str) -> list[tuple[str, str]]:
    """The `name=value` entries of a `;`-separated table, names normalised; empty entries skipped."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in text.split(";"):
        if not item.strip():
            continue
        name, sep, value = item.partition("=")
        name = normalise_label(name)
        if not sep or not name or not value.strip():
            raise ValueError(f"seeded: {setting}: {item.strip()!r} is not name=value")
        key = _key(name)
        if key in seen:
            raise ValueError(f"seeded: {setting} names {name!r} twice")
        seen.add(key)
        out.append((key, value.strip()))
    return out


def parse_sizes(text: str) -> dict[str, Size]:
    """`labels_sizes` as {key: (L, W, H)} [m]. Raises ValueError on a malformed entry."""
    sizes: dict[str, Size] = {}
    for key, value in _entries(text, "labels_sizes"):
        parts = value.lower().split("x")
        try:
            if len(parts) != 3:
                raise ValueError
            sizes[key] = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            raise ValueError(f"seeded: labels_sizes: {key}: {value!r} is not LxWxH in metres") from None
    return sizes


def _range(text: str, key: str) -> HsvRange:
    """One `H0-H1,S0-S1,V0-V1` range."""
    try:
        pairs = [part.split("-") for part in text.split(",")]
        if len(pairs) != 3 or any(len(p) != 2 for p in pairs):
            raise ValueError
        (h0, h1), (s0, s1), (v0, v1) = ((int(a), int(b)) for a, b in pairs)
    except ValueError:
        raise ValueError(f"seeded: labels_hsv: {key}: {text.strip()!r} is not H0-H1,S0-S1,V0-V1") from None
    return (h0, s0, v0), (h1, s1, v1)


def parse_hsv(text: str) -> dict[str, tuple[HsvRange, ...]]:
    """`labels_hsv` as {key: (((H0, S0, V0), (H1, S1, V1)), ...)}.

    Raises ValueError on a malformed entry.
    """
    return {key: tuple(_range(part, key) for part in value.split("|"))
            for key, value in _entries(text, "labels_hsv")}


#: The built-in colours, parsed.
_COLOUR_RANGES = parse_hsv(";".join(f"{name}={ranges}" for name, ranges in COLOURS.items()))


def resolve(label: str, sizes: dict[str, Size], hsv: dict[str, tuple[HsvRange, ...]]) -> Block | None:
    """The block a label names: an override for the whole label, else for its shape or colour word,
    else the built-in shape and colour. None when the size or the colour stays unknown."""
    label = normalise_label(label)
    phrase, shape = shape_word(label)
    colour = colour_word(phrase)
    size = sizes.get(label) or sizes.get(shape) or SHAPES.get(shape)
    ranges = hsv.get(label) or hsv.get(colour) or _COLOUR_RANGES.get(colour)
    if size is None or ranges is None:
        return None
    return Block(size, ranges)


def naming_prompt() -> str:
    """The line that tells the model what the blocks are and how to name them."""
    colours = ", ".join(COLOURS)
    shapes = ", ".join(SHAPES)
    return (f'The objects are wooden blocks. Name each block "<colour> <shape>": colour one of {colours} '
            f'("wooden" for bare wood), shape one of {shapes}. A cuboid is a flat bar; a cylinder lies '
            "on its side. Several blocks may share a name.")


def _at_least(value: float, lo: float) -> bool:
    return math.isfinite(value) and value >= lo


def _within(value: float, lo: float, hi: float) -> bool:
    return math.isfinite(value) and lo <= value <= hi


def validate(settings: SeededSettings) -> Tables:
    """Refuse settings the detector cannot work with; return the labels and the override tables.

    The LLM detector's own settings (`cli`, `model`, `timeout_s`, ...) are checked when it is built.
    """
    s = settings
    sizes = parse_sizes(s.labels_sizes)
    hsv = parse_hsv(s.labels_hsv)
    for key, size in sizes.items():
        if not all(_within(side, 0.0, MAX_SIZE_M) and side > 0.0 for side in size):
            raise ValueError(f"seeded: labels_sizes: {key}: every side must be in (0, {MAX_SIZE_M:g}] m, "
                             f"got {'x'.join(f'{side:g}' for side in size)}")
    for key, ranges in hsv.items():
        for lo, hi in ranges:
            if not (0 <= lo[0] <= H_MAX and 0 <= hi[0] <= H_MAX):
                raise ValueError(f"seeded: labels_hsv: {key}: H must be in 0..{H_MAX}, got {lo[0]}-{hi[0]}")
            if not all(0 <= v <= SV_MAX for v in (*lo[1:], *hi[1:])):
                raise ValueError(f"seeded: labels_hsv: {key}: S and V must be in 0..{SV_MAX}")
            if not all(a <= b for a, b in zip(lo, hi)):
                raise ValueError(f"seeded: labels_hsv: {key}: a range must run from low to high, got "
                                 f"{lo} above {hi}")
    labels = tuple(dict.fromkeys(normalise_label(label) for label in names(s.labels)))
    for label in labels:
        if not (label in sizes or shape_word(label)[1]):
            raise ValueError(f"seeded: labels: {label!r} names no shape ({', '.join(SHAPES)}) and "
                             "labels_sizes gives it no size")
        if resolve(label, sizes, hsv) is None:
            raise ValueError(f"seeded: labels: {label!r} names no colour ({', '.join(COLOURS)}) and "
                             "labels_hsv gives it no range")
    if not _at_least(s.interval_s, MIN_INTERVAL_S):
        raise ValueError(f"seeded: interval_s must be at least {MIN_INTERVAL_S:g} s, got {s.interval_s}")
    if not _at_least(s.seed_interval_s, MIN_SEED_INTERVAL_S):
        raise ValueError(f"seeded: seed_interval_s must be at least {MIN_SEED_INTERVAL_S:g} s, "
                         f"got {s.seed_interval_s}")
    if not _within(s.seed_error_px, *SEED_ERROR_PX):
        raise ValueError(f"seeded: seed_error_px must be in [{SEED_ERROR_PX[0]:g}, {SEED_ERROR_PX[1]:g}] px, "
                         f"got {s.seed_error_px}")
    if not _within(s.track_error_px, *TRACK_ERROR_PX):
        lo, hi = TRACK_ERROR_PX
        raise ValueError(f"seeded: track_error_px must be in [{lo:g}, {hi:g}] px, got {s.track_error_px}")
    if not _at_least(s.lost_after_s, 0.0):
        raise ValueError(f"seeded: lost_after_s must be 0 s or more, got {s.lost_after_s}")
    if s.min_area_px < 1:
        raise ValueError(f"seeded: min_area_px must be at least 1 px, got {s.min_area_px}")
    if not _at_least(s.max_blob_ratio, 1.0):
        raise ValueError(f"seeded: max_blob_ratio must be at least 1, got {s.max_blob_ratio}")
    return Tables(labels, sizes, hsv)


def llm_settings(settings: SeededSettings, labels: tuple[str, ...] | list[str] = ()) -> LlmSettings:
    """The LLM detector's settings: the mirrored fields, `labels` as its objects (none = every
    object), the naming line before `prompt`, `seed_interval_s` as its interval and `seed_error_px`
    as its pixel error. Reusing an answer for still frames stays off."""
    s = settings
    prompt = "\n".join(part for part in (naming_prompt(), s.prompt.strip()) if part)
    return LlmSettings(
        cli=s.cli, model=s.model, effort=s.effort, objects=", ".join(labels), cameras=s.cameras,
        prompt=prompt, interval_s=float(s.seed_interval_s), timeout_s=float(s.timeout_s),
        max_side_px=int(s.max_side_px), pixel_error_px=float(s.seed_error_px), reuse_if_still=False,
        max_calls_per_hour=int(s.max_calls_per_hour), min_confidence=float(s.min_confidence),
        executable=s.executable, claude_images=s.claude_images,
    )
