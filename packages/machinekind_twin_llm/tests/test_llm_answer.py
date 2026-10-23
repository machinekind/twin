"""The prompt, the schema, the images sent and the answer parsed back to Detections. No CLI, no model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from machinekind_twin_api import Detection
from machinekind_twin_llm import smoke
from machinekind_twin_llm.answer import MAX_DETECTIONS, parse_answer
from machinekind_twin_llm.images import THUMB_SIZE, SentImage, encode, thumbnail, unchanged
from machinekind_twin_llm.prompt import SCHEMA, build_prompt, normalise_label

#: A 1920x1440 frame sent as 1280x960: scale 1.5.
TOP = SentImage("top", Path("/call/img-1.jpg"), (1280, 960), (1920, 1440))
#: A frame sent at its own size.
SIDE = SentImage("side", Path("/call/img-2.jpg"), (640, 480), (640, 480))
SENT = [TOP, SIDE]


def item(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"label": "red pen", "camera": "top", "u": 100.0, "v": 200.0, "box": None,
                            "confidence": 0.9}
    base.update(kw)
    return base


def parse(*items: dict[str, Any], **kw: Any) -> list[Detection]:
    return parse_answer({"objects": list(items)}, SENT, **kw)


# --- answer ---------------------------------------------------------------------------------------


def test_pixels_scale_back_to_the_frame_with_pixel_centres_kept():
    (d,) = parse(item(u=639.5, v=479.5))  # the centre of the sent image

    assert d.camera == "top"
    assert d.pixel == pytest.approx((959.5, 719.5), abs=1e-9)  # the centre of the frame
    (d,) = parse(item(u=10.0, v=20.0))
    assert d.pixel == pytest.approx((10.5 * 1.5 - 0.5, 20.5 * 1.5 - 0.5), abs=1e-9)


def test_each_detection_carries_the_pixel_error_scaled_to_its_frame():
    by_camera = {d.camera: d for d in parse(item(), item(camera="side"), pixel_error_px=25.0)}

    assert by_camera["top"].pixel_error_px == pytest.approx(25.0 * 1.5)
    assert by_camera["side"].pixel_error_px == pytest.approx(25.0)


def test_a_box_is_scaled_reordered_and_clamped():
    (d,) = parse(item(box={"u0": 200.0, "v0": 300.0, "u1": 100.0, "v1": 100.0}))
    assert d.box == pytest.approx((100.5 * 1.5 - 0.5, 100.5 * 1.5 - 0.5, 200.5 * 1.5 - 0.5, 300.5 * 1.5 - 0.5))

    (d,) = parse(item(box={"u0": -50.0, "v0": -1.0, "u1": 5000.0, "v1": 960.0}))
    assert d.box == pytest.approx((0.0, 0.0, 1919.0, 1439.0))


def test_a_pixel_just_outside_the_image_is_clamped_into_the_frame():
    (lo,) = parse(item(u=-2.0, v=-1.0))
    (hi,) = parse(item(u=1281.0, v=961.0))

    assert lo.pixel == (0.0, 0.0)
    assert hi.pixel == (1919.0, 1439.0)


def test_a_detection_has_only_pixels_no_mask_size_or_height():
    (d,) = parse(item())

    assert isinstance(d, Detection)
    assert d.mask is None and d.size_m is None and d.height_m is None and d.box is None


@pytest.mark.parametrize("bad", [
    item(camera="front"),                  # not sent
    item(camera=None),
    item(u=math.nan),
    item(v=math.inf),
    item(u="100"),                          # a string, not a number
    item(u=True),
    item(u=-2.5),                           # outside -2..w+1
    item(u=1281.5),
    item(v=961.5),
    item(camera="side", u=642.0),           # outside that camera's own image
    item(confidence=0.2),                   # below min_confidence
    item(confidence=None),
    item(confidence=math.nan),
    item(label=""),
    item(label="   "),
    item(label=None),
    item(label="x" * 65),
    item(box={"u0": 1.0, "v0": 2.0, "u1": math.nan, "v1": 4.0}),
    item(box={"u0": 1.0, "v0": 2.0, "u1": 3.0}),
    item(box=[1.0, 2.0, 3.0, 4.0]),
    "red pen",
    None,
])
def test_a_bad_item_is_dropped_and_the_rest_kept(bad: Any):
    got = parse_answer({"objects": [bad, item(label="blue cube")]}, SENT, min_confidence=0.3)

    assert [d.label for d in got] == ["blue cube"]


def test_a_camera_name_matches_case_insensitively_and_is_reported_as_sent():
    (d,) = parse(item(camera="TOP"))

    assert d.camera == "top"


def test_labels_are_normalised():
    (d,) = parse(item(label="  Red   Pen "))

    assert d.label == "red pen"
    assert normalise_label("Red\tPen\n") == "red pen"
    assert len(parse(item(label="x" * 64))) == 1


def test_with_wanted_labels_only_those_are_reported_under_the_wanted_name():
    got = parse(item(label="The Red Pen"), item(label="red pen", camera="side"), item(label="blue cube"),
                item(label="pen"), item(label="redpen"), wanted=["Red Pen", "cube"])

    assert sorted((d.label, d.camera) for d in got) == [("cube", "top"), ("red pen", "side"), ("red pen", "top")]


def test_the_longest_wanted_name_wins():
    (d,) = parse(item(label="big red pen"), wanted=["pen", "red pen"])

    assert d.label == "red pen"


def test_confidence_is_clamped_to_0_1():
    (d,) = parse(item(confidence=1.7))
    assert d.confidence == 1.0

    (d,) = parse(item(confidence=-0.5), min_confidence=-1.0)
    assert d.confidence == 0.0


@pytest.mark.parametrize("raw", [{}, {"objects": None}, {"objects": {"label": "x"}}, [], "objects", None])
def test_an_answer_without_an_objects_list_raises(raw: Any):
    with pytest.raises(ValueError, match="answer has no objects list"):
        parse_answer(raw, SENT)


def test_an_empty_answer_gives_no_detections():
    assert parse() == []


def test_at_most_30_detections_are_kept_surest_first():
    items = [item(label=f"thing {i}", confidence=i / 100) for i in range(40)]

    got = parse(*items)

    assert len(got) == MAX_DETECTIONS == 30
    assert [d.confidence for d in got] == [i / 100 for i in range(39, 9, -1)]


# --- prompt and schema ----------------------------------------------------------------------------


def _strict(node: Any) -> None:
    """Every object schema lists all its properties as required and allows no others."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert sorted(node["required"]) == sorted(node["properties"])
        for value in node.values():
            _strict(value)
    elif isinstance(node, list):
        for value in node:
            _strict(value)


def test_the_schema_is_strict_and_the_smoke_run_uses_it():
    _strict(SCHEMA)
    item_schema = SCHEMA["properties"]["objects"]["items"]
    assert item_schema["required"] == ["label", "camera", "u", "v", "box", "confidence"]
    assert {"type": "null"} in item_schema["properties"]["box"]["anyOf"]
    assert smoke.SMOKE_SCHEMA is SCHEMA


def test_the_prompt_names_each_image_by_order_and_size():
    text = build_prompt(SENT)

    assert "You get 2 photos of the same table" in text
    assert '- camera "top": 1280x960 px, image 1' in text
    assert '- camera "side": 640x480 px, image 2' in text
    assert "every movable object on the table" in text
    assert text.endswith("Answer only with the JSON.")


def test_the_read_route_prompt_names_absolute_paths(tmp_path: Path):
    s = SentImage("top", tmp_path / "img-1.jpg", (320, 240), (640, 480))

    text = build_prompt([s], image_ref="path")

    assert "You get 1 photo of the same table" in text
    assert f'- camera "top": 320x240 px, file {(tmp_path / "img-1.jpg").resolve()} (open it with the Read tool)' in text


def test_the_prompt_asks_for_wanted_objects_by_their_normalised_labels_and_appends_extra():
    text = build_prompt(SENT, ["Red Pen", " cube", "", "red pen"], extra="  The table is white.  ")

    assert "Find these objects: red pen, cube. Report only these, with exactly these labels." in text
    assert "every movable object" not in text
    assert text.endswith("Answer only with the JSON.\nThe table is white.")


def test_an_unknown_image_ref_raises():
    with pytest.raises(ValueError, match="image_ref"):
        build_prompt(SENT, image_ref="inline")


# --- images ---------------------------------------------------------------------------------------


def test_encode_scales_large_frames_down_and_keeps_small_ones(tmp_path: Path):
    big = np.zeros((1440, 1920, 3), np.uint8)
    big[..., 0] = 255  # red in RGB
    small = np.full((240, 320, 3), 128, np.uint8)

    sent = encode(tmp_path, [("top camera", big), ("side", small)], max_side_px=1280)

    assert [s.camera for s in sent] == ["top camera", "side"]
    assert [s.path.name for s in sent] == ["img-1.jpg", "img-2.jpg"]
    assert [(s.sent_size, s.frame_size) for s in sent] == [((1280, 960), (1920, 1440)), ((320, 240), (320, 240))]
    bgr = cv2.imread(str(sent[0].path))
    assert bgr is not None and bgr.shape == (960, 1280, 3)
    assert sent[0].path.read_bytes()[:2] == b"\xff\xd8"
    b, g, r = (float(c) for c in bgr.reshape(-1, 3).mean(axis=0))
    assert r > 240 and g < 15 and b < 15  # written as BGR, so it decodes red


def test_encode_refuses_an_image_that_is_not_rgb(tmp_path: Path):
    with pytest.raises(ValueError, match="RGB"):
        encode(tmp_path, [("top", np.zeros((10, 10), np.uint8))], max_side_px=1280)


def test_unchanged_compares_thumbnails_of_the_same_cameras():
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    noisy = np.clip(scene.astype(int) + rng.integers(-2, 3, scene.shape), 0, 255).astype(np.uint8)
    moved = scene.copy()
    moved[100:300, 100:300] = 0

    before = {"top": thumbnail(scene), "side": thumbnail(scene)}

    assert thumbnail(scene).shape == (THUMB_SIZE[1], THUMB_SIZE[0])
    assert unchanged(before, {"top": thumbnail(noisy), "side": thumbnail(scene)})
    assert not unchanged(before, {"top": thumbnail(moved), "side": thumbnail(scene)})
    assert not unchanged(before, {"top": thumbnail(scene)})  # a camera fewer
    assert not unchanged({}, {})
