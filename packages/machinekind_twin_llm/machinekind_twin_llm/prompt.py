"""What the model is asked: the answer schema and the prompt.

The schema follows the strict mode of structured outputs (every property required, no extra
properties, a nullable box as `anyOf` with null), so the same document works as claude's
`--json-schema` and codex's `--output-schema`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .images import SentImage

__all__ = ["IMAGE_REFS", "SCHEMA", "build_prompt", "normalise_label"]

_NUM: dict[str, Any] = {"type": "number"}

#: The answer: one entry per object per image it is seen in, in that image's pixels.
SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["objects"],
    "properties": {"objects": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["label", "camera", "u", "v", "box", "confidence"],
        "properties": {
            "label": {"type": "string"}, "camera": {"type": "string"}, "u": _NUM, "v": _NUM,
            "box": {"anyOf": [{"type": "object", "additionalProperties": False,
                               "required": ["u0", "v0", "u1", "v1"],
                               "properties": {"u0": _NUM, "v0": _NUM, "u1": _NUM, "v1": _NUM}},
                              {"type": "null"}]},
            "confidence": _NUM}}}}}

#: How the prompt points at an image: "order" (the k-th image attached or inline) or "path" (a
#: file the model opens with the Read tool).
IMAGE_REFS = ("order", "path")


def normalise_label(label: str) -> str:
    """A label as the twin keys it: inner whitespace collapsed, stripped, lower-cased."""
    return " ".join(label.split()).lower()


def build_prompt(sent: Sequence[SentImage], objects: Sequence[str] = (), extra: str = "",
                 image_ref: str = "order") -> str:
    """The question about `sent`: which images there are, what to find, how to answer.

    `objects` are the labels to look for (normalised here); empty means every movable object on the
    table, named by the model. `extra`, stripped, is appended as the last line.
    """
    if image_ref not in IMAGE_REFS:
        raise ValueError(f"image_ref must be one of {', '.join(IMAGE_REFS)}, not {image_ref!r}")
    n = len(sent)
    lines = [
        "You are the object detector of a robot arm's workstation. "
        f"You get {n} photo{'' if n == 1 else 's'} of the same table, one per camera, taken at the same moment.",
    ]
    for k, s in enumerate(sent, start=1):
        w, h = s.sent_size
        where = f"image {k}" if image_ref == "order" else f"file {s.path.resolve()} (open it with the Read tool)"
        lines.append(f'- camera "{s.camera}": {w}x{h} px, {where}')
    wanted = list(dict.fromkeys(label for label in map(normalise_label, objects) if label))
    if wanted:
        lines.append(f"Find these objects: {', '.join(wanted)}. Report only these, with exactly these labels.")
    else:
        lines.append(
            "Find every movable object on the table: not the robot arm, not the table, not the cameras, "
            "not things fixed to the table. Label each with a short lowercase English name, colour plus "
            'noun, e.g. "red pen", the same name in every image.')
    lines += [
        "For every object, in every image where you can see it, give one entry:",
        "- label; camera: the camera name of that image",
        "- u, v: the pixel at the centre of where the object rests on the table (for an object held in "
        "the gripper, its centre), in that image's own pixels: u from the left edge (0..width-1), "
        "v from the top edge (0..height-1)",
        "- box: its bounding box in the same pixels, or null",
        "- confidence: 0..1",
        "An object you cannot see in an image gets no entry for that image. Do not guess positions.",
        "If you find nothing, return an empty list. Answer only with the JSON.",
    ]
    if extra.strip():
        lines.append(extra.strip())
    return "\n".join(lines)
