"""The built-in `charuco` detector: registered by the twin itself, one detection per inner corner, labels the
twin keeps apart, nothing without a board, and a package's own `charuco` replaces it."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from machinekind_twin_api.detector import DetectorInfo, DetectRequest, base_label
from machinekind_twin_api.strategy import Frame

from machinekind_twin import plugin_settings
from machinekind_twin.calib.intrinsics import Board
from machinekind_twin.charuco_detector import CharucoDetector, CharucoSettings
from machinekind_twin.detectors import ENTRY_POINT_GROUP, detectors, load_detector

INFO = DetectorInfo(cameras=("kamera1",), robot="testbot", simulated=True)
BUILT_IN = "machinekind_twin.charuco_detector:CharucoDetector"


def board_frame(cv2, detector: CharucoDetector) -> np.ndarray:
    """The board printed at 3000 px/m on a grey page, as an RGB frame."""
    board = detector.spec.image(cv2, 3000.0)
    page = np.full((board.shape[0] + 200, board.shape[1] + 300), 150, np.uint8)
    page[100:100 + board.shape[0], 150:150 + board.shape[1]] = board
    return np.repeat(page[:, :, None], 3, axis=2)


def test_each_inner_corner_is_one_detection_at_the_boards_height():
    import cv2

    detector = CharucoDetector(CharucoSettings(thickness_m=0.0005))
    detector.start(INFO)
    spec = detector.spec
    found = detector.detect(DetectRequest(frames={"kamera1": Frame(board_frame(cv2, detector), 0.0)}, step=0))
    assert sorted(d.label for d in found) == [f"charuco c{k}" for k in range(spec.n_corners)]
    for d in found:
        assert d.camera == "kamera1" and d.height_m == 0.0005 and d.mask is None and d.size_m is None
        k = int(d.label.split("c")[-1])
        u, v = spec.corner_xy([k])[0] * 3000.0 - 0.5 + np.array([150, 100])
        assert d.pixel == pytest.approx((u, v), abs=0.3)
    assert detector.interval_s == 0.5 and detector.pixel_error_px == 1.0


def test_the_defaults_are_the_3x5_sheet_of_twin_board():
    spec, sheet = CharucoDetector().spec, Board.of("3x5", 0.045)
    assert (spec.squares_x, spec.squares_y) == sheet.squares and spec.dictionary == sheet.spec.dictionary
    assert spec.square_m == sheet.square and spec.marker_m == pytest.approx(sheet.marker)


def test_the_labels_are_distinct_objects_for_the_twin():
    labels = [f"charuco c{k}" for k in range(12)]
    assert len({base_label(label) for label in labels}) == len(labels)
    assert base_label("charuco c3 2") == "charuco c3"


def test_frames_without_the_board_or_without_a_frame_give_nothing():
    detector = CharucoDetector()
    detector.start(INFO)
    empty = np.full((480, 640, 3), 128, np.uint8)
    assert list(detector.detect(DetectRequest(frames={"kamera1": Frame(empty, 0.0)}, step=0))) == []

    class Missing(dict):
        def __getitem__(self, name):
            raise KeyError(name)

    frames = Missing(kamera1=None)
    assert list(detector.detect(DetectRequest(frames=frames, step=1))) == []


def test_without_opencv_the_detector_refuses_to_start_and_says_why(monkeypatch):
    real = importlib.import_module

    def no_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("no cv2")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", no_cv2)
    with pytest.raises(RuntimeError, match="OpenCV"):
        CharucoDetector().start(INFO)


@pytest.mark.parametrize("change, words", [
    ({"square_m": 0.03, "marker_m": 0.035}, "marker_m"),
    ({"label_prefix": "  "}, "label_prefix"),
    ({"label_prefix": "board 2"}, "number"),
    ({"interval_s": 0.0}, "interval_s"),
    ({"pixel_error_px": -1.0}, "pixel_error_px"),
])
def test_bad_settings_say_what_is_wrong(change, words):
    with pytest.raises(ValueError, match=words):
        CharucoDetector(CharucoSettings(**change))


# --------------------------------------------------------------------------- registration
def test_the_twin_registers_the_detector_itself():
    assert detectors()["charuco"].value == BUILT_IN

    plugin = load_detector("charuco")

    assert plugin.cls is CharucoDetector
    assert plugin.source.startswith("machinekind-twin ")
    assert [f.name for f in plugin.fields] == ["squares_x", "squares_y", "square_m", "marker_m", "dictionary",
                                               "thickness_m", "label_prefix", "interval_s", "pixel_error_px"]


def test_a_package_s_own_charuco_replaces_the_built_in_one(install_dist):
    install_dist("stand-charuco", {ENTRY_POINT_GROUP: {"charuco": "stand_charuco:Detector"}})

    assert detectors()["charuco"].value == "stand_charuco:Detector"


def _entry(name: str, value: str, dist: str | None) -> SimpleNamespace:
    return SimpleNamespace(name=name, value=value, dist=None if dist is None else SimpleNamespace(name=dist))


@pytest.mark.parametrize("order", [0, 1], ids=["built-in-first", "package-first"])
def test_the_package_wins_whatever_the_order(monkeypatch, order):
    found = [_entry("charuco", BUILT_IN, "machinekind-twin"), _entry("charuco", "stand:Detector", "stand")]
    if order:
        found.reverse()
    monkeypatch.setattr(plugin_settings, "entry_points", lambda group: found)

    assert plugin_settings.registered(ENTRY_POINT_GROUP, "Dwa detektory")["charuco"].value == "stand:Detector"


def test_two_packages_besides_the_twin_still_clash(monkeypatch):
    found = [_entry("charuco", BUILT_IN, "machinekind-twin"), _entry("charuco", "one:Detector", "one"),
             _entry("charuco", "two:Detector", None)]
    monkeypatch.setattr(plugin_settings, "entry_points", lambda group: found)

    with pytest.raises(ValueError, match="Dwa detektory o nazwie 'charuco'"):
        plugin_settings.registered(ENTRY_POINT_GROUP, "Dwa detektory")
