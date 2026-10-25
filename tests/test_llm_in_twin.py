"""The LLM detector end to end through the twin's perception, against a fake codex CLI.

The fake codex (`packages/machinekind_twin_llm/tests/fake_cli.py`) answers where two synthetic
cameras see a known table point, in the pixels of the images the detector sent. Headless
`Perception` with an injected clock places the answer in the base frame and tracks it. No model,
no network, no render.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import LLM
from test_detection import MODELS, Clock, Frames, workspace
from test_objects import H, W, project

from machinekind_twin.detection import Perception
from machinekind_twin.detectors import load_detector

#: Where the pen lies: on the table (z = 0), as the prompt asks the model to point.
PEN = np.array([0.22, 0.03, 0.0])
#: The frames are 640x480; sent with max_side_px=320 they are 320x240, so the scale is 2.
SIDE = 320
SCALE = W / SIDE


def sent_pixel(cam: str, p: np.ndarray) -> tuple[float, float]:
    """Where camera `cam` sees `p`, in the sent image's pixels (pixel centres, like the detector)."""
    u, v = project(MODELS[cam], p)
    return (u + 0.5) / SCALE - 0.5, (v + 0.5) / SCALE - 0.5


@pytest.fixture
def fake_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    """The fake codex in a bin folder, answering with the pen seen by both cameras. (exe, log)"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "codex"
    exe.write_text(f"#!{sys.executable}\n" + (LLM / "tests" / "fake_cli.py").read_text(encoding="utf-8"),
                   encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    objects = []
    for cam in ("a", "b"):
        u, v = sent_pixel(cam, PEN)
        objects.append({"label": "red pen", "camera": cam, "u": u, "v": v,
                        "box": {"u0": u - 8, "v0": v - 4, "u1": u + 8, "v1": v + 4}, "confidence": 0.9})
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps({"objects": objects}), encoding="utf-8")
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_ANSWER", str(answer))
    monkeypatch.setenv("FAKE_MODE", "ok")
    monkeypatch.setenv("FAKE_AUTH", "subscription")
    monkeypatch.delenv("MACHINEKIND_TWIN_LLM_KEEP", raising=False)
    return str(exe), log


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def perception(clock: Clock):
    p = Perception(SimpleNamespace(), workspace(), frame=Frames(clock), hand=lambda: None, clock=clock)
    yield p
    p.stop_all("koniec testu")


def start_llm(perception: Perception, llm_package: str, exe: str):
    plugin = load_detector(llm_package)
    raw = {"cli": "codex", "executable": exe, "objects": "red pen", "max_side_px": SIDE}
    return perception.start_detector(plugin, plugin.check(raw), threaded=False)


def runs(log: Path) -> list[dict]:
    return [r for r in map(json.loads, log.read_text(encoding="utf-8").splitlines()) if "argv" in r]


def test_the_llm_answer_becomes_an_object_in_the_base_frame(perception: Perception, clock: Clock,
                                                            llm_package: str, fake_codex: tuple[str, Path]):
    exe, log = fake_codex
    runner = start_llm(perception, llm_package, exe)
    assert runner.detector.interval_s == 10.0

    wait = runner.step_once()

    assert wait == pytest.approx(10.0)
    assert runner.status.failures == 0, runner.status.last_error
    assert runner.status.last_labels == ("red pen",) and runner.status.dropped == 0
    (run,) = runs(log)
    assert [i["size"] for i in run["images"]] == [[SIDE, SIDE * H // W]] * 2

    clock.advance(0.4)
    pen = perception.objects()["red pen"]
    np.testing.assert_allclose(pen.position, PEN, atol=3e-3)
    assert pen.age_s == pytest.approx(0.4, abs=1e-6) and pen.age_s > 0
    assert pen.source == "kamery"
    (view,) = perception.snapshot()
    assert view.n_cameras == 2 and view.method == "triangulacja" and view.detector == "llm"

    clock.advance(0.8)                      # older than fresh_s = 1.0 s: no longer "seen now"
    pen = perception.objects()["red pen"]
    assert pen.age_s == pytest.approx(1.2, abs=1e-6)
    assert pen.source == "ostatnie widziane"
    np.testing.assert_allclose(pen.position, PEN, atol=3e-3)


def test_a_slow_answer_is_published_as_last_seen_with_its_age(perception: Perception, clock: Clock,
                                                              llm_package: str, fake_codex: tuple[str, Path]):
    exe, _ = fake_codex
    runner = start_llm(perception, llm_package, exe)
    ask = runner.detector.detect

    def slow(request):
        found = ask(request)           # the frames were read at the start of the call
        clock.advance(12.0)            # the model took 12 s to answer
        return found

    runner.detector.detect = slow  # type: ignore[method-assign]
    runner.step_once()

    assert runner.status.last_latency_s == pytest.approx(12.0)
    pen = perception.objects()["red pen"]
    assert pen.source == "ostatnie widziane"
    assert pen.age_s == pytest.approx(12.0, abs=1e-6)
    np.testing.assert_allclose(pen.position, PEN, atol=3e-3)

    clock.advance(30.0)                     # held for 3 * (interval + latency) = 66 s
    assert perception.objects()["red pen"].age_s == pytest.approx(42.0, abs=1e-6)


def test_a_still_scene_repeats_the_answer_and_publishes_kamery(perception: Perception, clock: Clock,
                                                              llm_package: str, fake_codex: tuple[str, Path]):
    exe, log = fake_codex
    runner = start_llm(perception, llm_package, exe)
    runner.step_once()
    clock.advance(10.0)

    runner.step_once()

    assert len(runs(log)) == 1                  # no second question for unchanged frames
    assert runner.status.calls == 2 and runner.status.failures == 0
    clock.advance(0.4)
    pen = perception.objects()["red pen"]
    assert pen.source == "kamery"
    assert pen.age_s == pytest.approx(0.4, abs=1e-6)
    np.testing.assert_allclose(pen.position, PEN, atol=3e-3)
