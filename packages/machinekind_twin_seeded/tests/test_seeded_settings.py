"""The seeded detector's settings: the label tables' grammar, the checks, and the LLM detector's
settings mirrored from them. No CLI runs."""

from __future__ import annotations

from typing import Any

import pytest
from machinekind_twin_llm import LlmSettings
from machinekind_twin_seeded import SeededDetector, SeededSettings
from machinekind_twin_seeded.settings import DEFAULT_HSV, DEFAULT_SIZES, llm_settings, parse_hsv, parse_sizes

LABELS = ("cube", "cuboid5", "cuboid85", "cylinder")


def test_the_defaults_parse_into_four_labels_with_sizes_and_ranges():
    sizes = parse_sizes(DEFAULT_SIZES)
    hsv = parse_hsv(DEFAULT_HSV)

    assert sorted(sizes) == list(LABELS) and sorted(hsv) == list(LABELS)
    assert sizes["cube"] == (0.025, 0.025, 0.025)
    assert sizes["cuboid85"] == (0.085, 0.025, 0.025)
    assert hsv["cube"] == (((0, 120, 70), (12, 255, 255)), ((170, 120, 70), (180, 255, 255)))
    assert hsv["cylinder"] == (((135, 60, 60), (165, 255, 255)),)


def test_labels_are_keyed_as_the_twin_keys_them_and_empty_entries_are_skipped():
    sizes = parse_sizes(" Red  Block = 0.03X0.02x0.01 ;; ")

    assert sizes == {"red block": (0.03, 0.02, 0.01)}


@pytest.mark.parametrize("kw, problem", [
    ({"interval_s": 0.05}, "seeded: interval_s must be at least 0.1 s"),
    ({"interval_s": float("nan")}, "seeded: interval_s must be at least 0.1 s"),
    ({"seed_interval_s": 1.9}, "seeded: seed_interval_s must be at least 2 s"),
    ({"labels_sizes": DEFAULT_SIZES + ";pen=0.1x0.01x0.01"},
     "seeded: labels_sizes and labels_hsv must name the same labels; only in labels_sizes: pen"),
    ({"labels_sizes": "cube=0x0.025x0.025", "labels_hsv": "cube=0-12,120-255,70-255"},
     r"seeded: labels_sizes: cube: every side must be in \(0, 0.3\] m"),
    ({"labels_sizes": "cube=0.5x0.025x0.025", "labels_hsv": "cube=0-12,120-255,70-255"},
     r"seeded: labels_sizes: cube: every side must be in \(0, 0.3\] m"),
    ({"labels_sizes": "cube=0.025x0.025", "labels_hsv": "cube=0-12,120-255,70-255"},
     "seeded: labels_sizes: cube: '0.025x0.025' is not LxWxH"),
    ({"labels_sizes": "cube=0.025x0.025x0.025", "labels_hsv": "cube=0-200,120-255,70-255"},
     "seeded: labels_hsv: cube: H must be in 0..180"),
    ({"labels_sizes": "cube=0.025x0.025x0.025", "labels_hsv": "cube=0-12,255-120,70-255"},
     "seeded: labels_hsv: cube: a range must run from low to high"),
    ({"labels_sizes": "cube=0.025x0.025x0.025", "labels_hsv": "cube=0-12,120-255"},
     "seeded: labels_hsv: cube: '0-12,120-255' is not H0-H1,S0-S1,V0-V1"),
    ({"labels_sizes": "cube=0.025x0.025x0.025;cube=0.03x0.03x0.03"},
     "seeded: labels_sizes names 'cube' twice"),
    ({"labels_sizes": "", "labels_hsv": ""}, "seeded: labels_sizes and labels_hsv name no label"),
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


def test_the_llm_settings_mirror_the_seeded_ones():
    seeded = SeededSettings(cli="claude", model="sonnet", effort="medium", cameras="a", prompt="be brief",
                            seed_interval_s=4.0, seed_error_px=30.0, timeout_s=60.0, max_side_px=640,
                            max_calls_per_hour=100, min_confidence=0.5, executable="/bin/claude",
                            claude_images="read")

    got = llm_settings(seeded, LABELS)

    assert got == LlmSettings(cli="claude", model="sonnet", effort="medium",
                              objects="cube, cuboid5, cuboid85, cylinder", cameras="a", prompt="be brief",
                              interval_s=4.0, timeout_s=60.0, max_side_px=640, pixel_error_px=30.0,
                              reuse_if_still=False, max_calls_per_hour=100, min_confidence=0.5,
                              executable="/bin/claude", claude_images="read")


def test_the_contract_attributes_come_from_the_settings():
    det = SeededDetector(SeededSettings(interval_s=0.5))

    assert det.interval_s == 0.5
    assert det.pixel_error_px == 6.0
    assert SeededDetector.settings is SeededSettings
    assert det._llm.interval_s == 10.0
    assert det._llm.config.objects == "cube, cuboid5, cuboid85, cylinder"
    assert det._llm.config.reuse_if_still is False


def test_the_defaults_are_valid():
    det = SeededDetector()

    assert det.config == SeededSettings()
    assert det.interval_s == 0.2
    assert det._labels == LABELS
