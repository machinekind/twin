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
from machinekind_twin_llm.images import (
    BLOB_JOIN_PX,
    STILL_MAX_BLOB_PX,
    THUMB_LONG_PX,
    SentImage,
    arm_thumbnail,
    encode,
    known_scene,
    largest_change_px,
    thumbnail,
    unchanged,
)
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


# --- the still check ----------------------------------------------------------------------------


def scene(rng: np.random.Generator) -> np.ndarray:
    """A 1280x960 table: textured grey with a 3 cm cube's worth of dark pixels at 1 m, 24x24 px."""
    table = rng.integers(100, 140, (960, 1280, 3), dtype=np.uint8)
    table[600:624, 100:124] = (40, 40, 40)
    return table


def test_a_thumbnail_is_a_small_grey_float_copy_with_the_frame_s_aspect():
    thumb = thumbnail(np.zeros((960, 1280, 3), np.uint8))

    assert thumb.shape == (THUMB_LONG_PX * 3 // 4, THUMB_LONG_PX) and thumb.dtype == np.float32
    assert thumbnail(np.zeros((120, 160, 3), np.uint8)).shape == (120, 160)


def test_sensor_noise_and_exposure_drift_do_not_count_as_a_change():
    rng = np.random.default_rng(0)
    before = scene(rng)
    noisy = np.clip(before.astype(int) + rng.integers(-6, 7, before.shape), 0, 255).astype(np.uint8)
    brighter = np.clip(before.astype(int) + 12, 0, 255).astype(np.uint8)

    assert largest_change_px(thumbnail(before), thumbnail(noisy)) == 0
    assert largest_change_px(thumbnail(before), thumbnail(brighter)) == 0
    assert unchanged({"top": thumbnail(before)}, {"top": thumbnail(noisy)})


def test_a_small_object_moving_or_leaving_counts_as_a_change():
    rng = np.random.default_rng(0)
    before = scene(rng)
    moved = scene(rng)
    moved[600:624, 100:124] = before[600:624, 200:224]
    moved[600:624, 140:164] = (40, 40, 40)
    gone = scene(rng)
    gone[600:624, 100:124] = before[600:624, 200:224]

    assert largest_change_px(thumbnail(before), thumbnail(moved)) > STILL_MAX_BLOB_PX
    assert largest_change_px(thumbnail(before), thumbnail(gone)) > STILL_MAX_BLOB_PX
    assert not unchanged({"top": thumbnail(before), "side": thumbnail(before)},
                         {"top": thumbnail(moved), "side": thumbnail(before)})


def test_a_few_flickering_pixels_are_not_a_change():
    before = np.full((480, 640, 3), 120, np.uint8)
    blink = before.copy()
    blink[10:14, 10:14] = 255                      # a status LED: 2x2 thumbnail pixels

    assert 0 < largest_change_px(thumbnail(before), thumbnail(blink)) <= STILL_MAX_BLOB_PX


def test_unchanged_needs_the_same_cameras_and_frame_shapes():
    frame = thumbnail(np.full((480, 640, 3), 120, np.uint8))
    other_shape = thumbnail(np.full((640, 480, 3), 120, np.uint8))

    assert largest_change_px(frame, other_shape) == other_shape.size
    assert not unchanged({"top": frame, "side": frame}, {"top": frame})
    assert not unchanged({"top": frame}, {"top": frame, "side": frame})
    assert not unchanged({}, {})
    assert not unchanged({"top": frame}, {"top": other_shape})


#: A 3 cm cube at 1 m through a 60 degree lens in a 1280 px wide frame [px]; 1 cm is 11 px.
CUBE_PX, CM_PX = 33, 11


def table(rng: np.random.Generator, w: int = 1280, h: int = 960) -> np.ndarray:
    return rng.integers(100, 141, (h, w, 3), dtype=np.uint8)


def with_box(img: np.ndarray, x: int, y: int, side: int, shade: int) -> np.ndarray:
    out = img.copy()
    out[y:y + side, x:x + side] = shade
    return out


def u8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def cables(rng: np.random.Generator) -> np.ndarray:
    """A high-contrast scene: a light table, ten dark cables 1 cm thick and a sheet of white paper."""
    img = np.full((960, 1280), 160, np.uint8)
    for _ in range(10):
        start = (int(rng.integers(0, 1280)), int(rng.integers(0, 960)))
        end = (int(rng.integers(0, 1280)), int(rng.integers(0, 960)))
        cv2.line(img, start, end, 20, 12)
    img[100:430, 130:450] = 245
    return np.repeat(img[..., None], 3, axis=2)


def test_scattered_changed_pixels_are_not_a_change():
    rng = np.random.default_rng(0)
    before = thumbnail(np.full((960, 1280, 3), 120, np.uint8))
    now = before.copy()
    spots = [(row, col) for row in range(10, 190, 30) for col in range(10, 250, 24)][:60]
    for row, col in spots:
        now[row, col] += rng.uniform(60, 120)

    assert len(spots) == 60
    assert largest_change_px(before, now) <= STILL_MAX_BLOB_PX
    assert unchanged({"top": before}, {"top": now})


def test_a_gain_change_on_a_high_contrast_scene_is_not_a_change():
    scene_now = cables(np.random.default_rng(0))
    before = thumbnail(scene_now)

    for gain in (1.15, 1 / 1.15):
        assert largest_change_px(before, thumbnail(u8(scene_now * gain))) <= STILL_MAX_BLOB_PX


def test_a_cube_moving_a_centimetre_or_leaving_counts_with_a_1_5x_margin():
    base = table(np.random.default_rng(0))
    before = thumbnail(with_box(base, 402, 301, CUBE_PX, 40))
    moves = [(CM_PX, 0), (-CM_PX, 0), (0, CM_PX), (0, -CM_PX), (CM_PX, CM_PX)]

    for dx, dy in moves:
        moved = thumbnail(with_box(base, 402 + dx, 301 + dy, CUBE_PX, 40))
        assert largest_change_px(before, moved) >= 1.5 * STILL_MAX_BLOB_PX, (dx, dy)
    assert largest_change_px(before, thumbnail(base)) >= 1.5 * STILL_MAX_BLOB_PX


def test_a_blinking_2x2_thumbnail_led_is_not_a_change():
    base = table(np.random.default_rng(0))
    before = thumbnail(base)

    for x, y in ((500, 300), (502, 300), (500, 303), (504, 304)):
        for shade in (255, 0):
            blink = thumbnail(with_box(base, x, y, 10, shade))
            assert 0 < largest_change_px(before, blink) <= STILL_MAX_BLOB_PX, (x, y, shade)


def test_noise_a_brightness_step_and_a_gain_change_are_not_a_change():
    rng = np.random.default_rng(0)
    still = with_box(table(rng), 402, 301, CUBE_PX, 40).astype(float)
    before = thumbnail(u8(still))
    noisy_before = thumbnail(u8(still + rng.integers(-20, 21, still.shape)))
    noisy_now = thumbnail(u8(still + rng.integers(-20, 21, still.shape)))

    assert largest_change_px(noisy_before, noisy_now) <= STILL_MAX_BLOB_PX
    for changed in (still + 12, still * 1.15, still / 1.15):
        assert largest_change_px(before, thumbnail(u8(changed))) <= STILL_MAX_BLOB_PX


def test_changed_pixels_join_one_blob_up_to_blob_join_px_apart():
    before = thumbnail(np.full((960, 1280, 3), 120, np.uint8))

    for gap, area in ((BLOB_JOIN_PX, 2), (BLOB_JOIN_PX + 1, 1)):
        for rows, cols in ((0, gap), (gap, 0), (gap, gap)):
            now = before.copy()
            now[50, 50] += 60
            now[50 + rows, 50 + cols] += 60
            assert largest_change_px(before, now) == area, (rows, cols)


def arm_at(img: np.ndarray, x: int, reach: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """`img` with a dark arm bar 80 px wide at column `x`, reaching `reach` px down from the top
    edge, and its bool mask."""
    mask = np.zeros(img.shape[:2], bool)
    mask[:reach, x:x + 80] = True
    out = img.copy()
    out[mask] = 30
    return out, mask


def test_the_pixels_the_arm_hides_keep_their_last_seen_value_or_none():
    rng = np.random.default_rng(0)
    seen = thumbnail(table(rng))
    now = thumbnail(table(rng))
    arm = np.zeros(now.shape, bool)
    arm[:100, 60:80] = True

    known = known_scene(now, arm, seen)
    np.testing.assert_array_equal(known[arm], seen[arm])
    np.testing.assert_array_equal(known[~arm], now[~arm])
    for previous in (None, seen[:, 1:]):
        first = known_scene(now, arm, previous)
        assert np.isnan(first[arm]).all()
        np.testing.assert_array_equal(first[~arm], now[~arm])
    np.testing.assert_array_equal(known_scene(now, None, seen), now)


def test_a_place_no_frame_showed_yet_counts_as_changed_once_seen():
    base = thumbnail(table(np.random.default_rng(0)))
    unseen = base.copy()
    unseen[:100, 60:80] = np.nan

    assert largest_change_px(unseen, base) > STILL_MAX_BLOB_PX
    assert largest_change_px(unseen, unseen) == 0
    assert largest_change_px(base, unseen) == 0


def test_the_arm_reaching_further_under_its_mask_is_no_change():
    base = table(np.random.default_rng(0))
    near_img, near_arm = arm_at(base, 300, reach=300)
    far_img, far_arm = arm_at(base, 300)
    near_thumb, far_thumb = thumbnail(near_img), thumbnail(far_img)
    near = known_scene(near_thumb, arm_thumbnail(near_arm, near_thumb.shape), None)
    far = known_scene(far_thumb, arm_thumbnail(far_arm, far_thumb.shape), near)

    assert largest_change_px(near_thumb, far_thumb) > STILL_MAX_BLOB_PX
    assert largest_change_px(near, far) == 0
    assert unchanged({"top": near}, {"top": far})

    cube_before = thumbnail(with_box(base, 1000, 800, CUBE_PX, 40))
    entering_img, entering_arm = arm_at(with_box(base, 1000 + CM_PX, 800, CUBE_PX, 40), 300)
    entering = thumbnail(entering_img)

    assert not unchanged({"top": cube_before},
                         {"top": known_scene(entering, arm_thumbnail(entering_arm, entering.shape), cube_before)})


def test_the_mean_is_matched_over_the_pixels_both_frames_show():
    base = table(np.random.default_rng(0))
    arm = np.zeros(base.shape[:2], bool)
    arm[:, :256] = True
    hidden = base.copy()
    hidden[arm] = 0
    before, now = thumbnail(base), thumbnail(hidden)

    assert largest_change_px(before, known_scene(now, arm_thumbnail(arm, now.shape), None)) == 0


def test_the_arm_s_edge_and_shadow_beside_its_mask_do_not_count_but_a_cube_further_off_does():
    base = table(np.random.default_rng(0))
    shadowed, arm = arm_at(base, 500)
    shadowed[:600, 580:595] = 70                   # a shadow 3 thumbnail pixels wide
    before, now = thumbnail(base), thumbnail(shadowed)
    mask = arm_thumbnail(arm, now.shape)
    cube = thumbnail(with_box(shadowed, 650, 300, CUBE_PX, 40))

    assert mask.shape == now.shape and mask.dtype == bool
    assert largest_change_px(before, now) > STILL_MAX_BLOB_PX
    assert largest_change_px(before, known_scene(now, mask, before)) == 0
    assert largest_change_px(before, known_scene(cube, mask, before)) > STILL_MAX_BLOB_PX


def test_an_image_no_frame_showed_yet_counts_as_changed():
    frame = thumbnail(np.full((480, 640, 3), 120, np.uint8))
    unseen = known_scene(frame, np.ones(frame.shape, bool), None)

    assert largest_change_px(frame, unseen) == frame.size
    assert largest_change_px(unseen, unseen) == frame.size
