"""The seeded detector's settings: labels as blocks, the override tables' grammar, the checks, and
the LLM detector's settings mirrored from them. No CLI runs."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest
from machinekind_twin_llm import LlmSettings
from machinekind_twin_seeded import SeededDetector, SeededSettings
from machinekind_twin_seeded.grow import colour_mask
from machinekind_twin_seeded.settings import (
    COLOURS,
    SHAPES,
    Block,
    llm_settings,
    naming_prompt,
    parse_hsv,
    parse_sizes,
    resolve,
)

#: A paint of each colour word, RGB.
PAINTS = {"red": (200, 30, 30), "yellow": (235, 200, 50), "green": (60, 165, 75),
          "light blue": (115, 185, 235), "wooden": (214, 189, 166)}
#: The twin simulation's table, RGB.
TABLE = (140, 128, 112)


def ranges(colour: str):
    return parse_hsv(f"{colour}={COLOURS[colour]}")[colour]


def matches(rgb: tuple[int, int, int], colour: str) -> bool:
    """Whether a patch of `rgb` passes the colour's ranges (the opening needs a patch, not a pixel)."""
    hsv = cv2.cvtColor(np.full((9, 9, 3), rgb, np.uint8), cv2.COLOR_RGB2HSV)
    return bool(colour_mask(hsv, ranges(colour))[4, 4])


@pytest.mark.parametrize("label, size", [
    ("red cube", (0.029, 0.029, 0.029)),
    ("light blue cuboid", (0.060, 0.029, 0.014)),
    ("wooden cylinder", (0.060, 0.029, 0.029)),
    ("Natural  Wood Cube", (0.029, 0.029, 0.029)),
    ("blue brick", (0.060, 0.029, 0.014)),
])
def test_a_colour_and_shape_label_resolves_to_the_shape_s_size_and_the_colour_s_ranges(label, size):
    block = resolve(label, {}, {})

    assert block is not None
    assert block.size == size
    assert block.hsv in {ranges(colour) for colour in COLOURS}


def test_labels_without_a_known_shape_or_colour_resolve_to_nothing():
    assert resolve("red pen", {}, {}) is None
    assert resolve("purple cube", {}, {}) is None
    assert resolve("cube", {}, {}) is None


def test_overrides_go_by_label_then_by_shape_or_colour_word():
    sizes = parse_sizes("cube=0.03x0.03x0.03; Red Cube = 0.031x0.031x0.031; pen=0.1x0.01x0.01")
    hsv = parse_hsv("wood=10-20,30-90,150-255;pen=100-120,80-255,50-255")

    assert resolve("red cube", sizes, hsv) == Block((0.031, 0.031, 0.031), ranges("red"))
    assert resolve("green cube", sizes, hsv) == Block((0.03, 0.03, 0.03), ranges("green"))
    assert resolve("wooden cuboid", sizes, hsv) == Block(SHAPES["cuboid"], (((10, 30, 150), (20, 90, 255)),))
    assert resolve("pen", sizes, hsv) == Block((0.1, 0.01, 0.01), (((100, 80, 50), (120, 255, 255)),))


def test_each_paint_passes_its_own_colour_only_and_the_table_passes_no_paint():
    for paint, rgb in PAINTS.items():
        assert [colour for colour in COLOURS if matches(rgb, colour)] == [paint]
    assert [colour for colour in COLOURS if matches(TABLE, colour) and colour != "wooden"] == []


def test_bare_wood_and_a_wood_coloured_table_pass_the_same_range():
    """Why bare wood needs the blob size guard: hue and saturation do not tell it from such a table."""
    assert matches(PAINTS["wooden"], "wooden") and matches(TABLE, "wooden")


def test_names_are_keyed_as_the_twin_keys_them_and_empty_entries_are_skipped():
    assert parse_sizes(" Red  Block = 0.03X0.02x0.01 ;; ") == {"red block": (0.03, 0.02, 0.01)}
    assert parse_sizes("wooden=0.03x0.02x0.01; Brick=0.05x0.03x0.01") == {
        "wooden": (0.03, 0.02, 0.01), "cuboid": (0.05, 0.03, 0.01)}


@pytest.mark.parametrize("kw, problem", [
    ({"interval_s": 0.05}, "seeded: interval_s must be at least 0.1 s"),
    ({"interval_s": float("nan")}, "seeded: interval_s must be at least 0.1 s"),
    ({"seed_interval_s": 1.9}, "seeded: seed_interval_s must be at least 2 s"),
    ({"labels": "red cube, red pen"}, "seeded: labels: 'red pen' names no shape"),
    ({"labels": "purple cube"}, "seeded: labels: 'purple cube' names no colour"),
    ({"labels_sizes": "cube=0x0.025x0.025"}, r"seeded: labels_sizes: cube: every side must be in \(0, 0.3\] m"),
    ({"labels_sizes": "cube=0.5x0.025x0.025"}, r"seeded: labels_sizes: cube: every side must be in \(0, 0.3\] m"),
    ({"labels_sizes": "cube=0.025x0.025"}, "seeded: labels_sizes: cube: '0.025x0.025' is not LxWxH"),
    ({"labels_hsv": "red=0-200,120-255,70-255"}, "seeded: labels_hsv: red: H must be in 0..180"),
    ({"labels_hsv": "red=0-12,255-120,70-255"}, "seeded: labels_hsv: red: a range must run from low to high"),
    ({"labels_hsv": "red=0-12,120-255"}, "seeded: labels_hsv: red: '0-12,120-255' is not H0-H1,S0-S1,V0-V1"),
    ({"labels_sizes": "cube=0.025x0.025x0.025;cube=0.03x0.03x0.03"}, "seeded: labels_sizes names 'cube' twice"),
    ({"labels_hsv": "wood=0-12,120-255,70-255;wooden=0-12,120-255,70-255"},
     "seeded: labels_hsv names 'wooden' twice"),
    ({"seed_error_px": 501.0}, r"seeded: seed_error_px must be in \[0, 500\] px"),
    ({"track_error_px": 0.0}, r"seeded: track_error_px must be in \[1, 500\] px"),
    ({"lost_after_s": -1.0}, "seeded: lost_after_s must be 0 s or more"),
    ({"min_area_px": 0}, "seeded: min_area_px must be at least 1 px"),
    ({"max_blob_ratio": 0.5}, "seeded: max_blob_ratio must be at least 1"),
    ({"cli": "gemini"}, "llm: cli must be one of codex, claude"),
    ({"timeout_s": 4.0}, r"llm: timeout_s must be in \[5, 900\]"),
])
def test_settings_the_detector_cannot_work_with_are_refused(kw: dict[str, Any], problem: str):
    with pytest.raises(ValueError, match=problem):
        SeededDetector(SeededSettings(**kw))


def test_a_listed_label_may_take_its_size_and_colour_from_the_overrides():
    det = SeededDetector(SeededSettings(labels="pen, red cube", labels_sizes="pen=0.1x0.01x0.01",
                                        labels_hsv="pen=100-120,80-255,50-255"))

    assert det._labels == ("pen", "red cube")
    assert det._llm.config.objects == "pen, red cube"


def test_the_llm_settings_mirror_the_seeded_ones():
    seeded = SeededSettings(cli="claude", model="sonnet", effort="medium", cameras="a", prompt=" be brief ",
                            seed_interval_s=4.0, seed_error_px=30.0, timeout_s=60.0, max_side_px=640,
                            max_calls_per_hour=100, min_confidence=0.5, executable="/bin/claude",
                            claude_images="read")

    got = llm_settings(seeded, ("red cube", "wooden cuboid"))

    assert got == LlmSettings(cli="claude", model="sonnet", effort="medium",
                              objects="red cube, wooden cuboid", cameras="a",
                              prompt=naming_prompt() + "\nbe brief", interval_s=4.0, timeout_s=60.0,
                              max_side_px=640, pixel_error_px=30.0, reuse_if_still=False,
                              max_calls_per_hour=100, min_confidence=0.5, executable="/bin/claude",
                              claude_images="read")


def test_the_naming_line_lists_every_colour_and_shape():
    line = naming_prompt()

    assert all(word in line for word in (*COLOURS, *SHAPES))


def test_the_contract_attributes_come_from_the_settings():
    det = SeededDetector(SeededSettings(interval_s=0.5))

    assert det.interval_s == 0.5
    assert det.pixel_error_px == 6.0
    assert SeededDetector.settings is SeededSettings
    assert det._llm.interval_s == 10.0
    assert det._llm.config.reuse_if_still is False


def test_the_defaults_follow_every_block_the_model_names():
    det = SeededDetector()

    assert det.config == SeededSettings()
    assert det.interval_s == 0.2
    assert det._labels == ()
    assert det._llm.config.objects == ""
    assert det._llm.config.prompt == naming_prompt()
