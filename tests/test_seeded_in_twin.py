"""The seeded detector end to end through the twin's perception, against a fake codex CLI.

Two synthetic cameras (`test_objects`' a, distorted, and b, pinhole) see a red 50x25x25 mm block
on grey: its silhouette is the filled hull of its projected corners. The fake codex
(`packages/machinekind_twin_llm/tests/fake_cli.py`) answers the block's footprint centre in the
pixels of the images the detector sent. The detector follows the block by colour, hands the twin
its masks and size, and headless `Perception` fits the box: position and yaw in the base frame.
No model, no network, no render.

The plugin's `create` cannot pass a clock, so each test sets the detector's `_clock` to the test
clock right after `start_detector`; the detector reads it only in `detect`. The worker runs a real
subprocess, so `cycle` waits for its answer on the wall clock.
"""

from __future__ import annotations

import json
import math
import stat
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from conftest import LLM
from test_detection import Clock, workspace
from test_objects import A, B, H, W, project
from test_perception import box_points, hull_mask

from machinekind_twin.detection import DetectorRunner, Perception
from machinekind_twin.detectors import load_detector

#: The block: 50x25x25 mm lying on a long side, its half sides [m].
HALF = (0.025, 0.0125, 0.0125)
LABEL = "cuboid5"
RED, GREY = (200, 30, 30), (120, 120, 120)
#: The frames are 640x480; sent with max_side_px=320 they are 320x240, so the scale is 2.
SIDE = 320
SCALE = W / SIDE
CAMERAS = {"a": A, "b": B}


def sent_pixel(u: float, v: float) -> tuple[float, float]:
    """A frame pixel in the sent image's pixels (pixel centres, like the detector)."""
    return (u + 0.5) / SCALE - 0.5, (v + 0.5) / SCALE - 0.5


def masks(x: float, y: float, yaw_deg: float) -> dict[str, np.ndarray]:
    """The block's silhouette in each camera, resting on the table at (x, y) turned by `yaw_deg`."""
    points = box_points(HALF, x, y, math.radians(yaw_deg))
    return {name: hull_mask(points, cam) for name, cam in CAMERAS.items()}


def frames_of(x: float | None = None, y: float = 0.0, yaw_deg: float = 0.0) -> dict[str, np.ndarray]:
    """Each camera's RGB frame: the red block on grey, or grey alone without `x`."""
    out = {}
    for name in CAMERAS:
        img = np.full((H, W, 3), GREY, np.uint8)
        if x is not None:
            img[masks(x, y, yaw_deg)[name]] = RED
        out[name] = img
    return out


class Synthetic:
    """A frame source stamping on the test clock: the current images, captured now."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.images: dict[str, np.ndarray] = {}

    def __call__(self, name: str) -> tuple[np.ndarray | None, float]:
        image = self.images.get(name)
        return (None, 0.0) if image is None else (image, self.clock())


class FakeCodex:
    def __init__(self, exe: Path, log: Path, answer: Path) -> None:
        self.exe = str(exe)
        self._log = log
        self._answer = answer

    def answers_block(self, x: float, y: float, yaw_deg: float) -> None:
        """The model points at the block's footprint centre in both cameras, with the silhouette's box."""
        objects = []
        for name, cam in CAMERAS.items():
            u, v = sent_pixel(*project(cam, (x, y, 0.0)))
            vs, us = np.nonzero(masks(x, y, yaw_deg)[name])
            u0, v0 = sent_pixel(float(us.min()), float(vs.min()))
            u1, v1 = sent_pixel(float(us.max()), float(vs.max()))
            objects.append({"label": LABEL, "camera": name, "u": u, "v": v,
                            "box": {"u0": u0, "v0": v0, "u1": u1, "v1": v1}, "confidence": 0.9})
        self._answer.write_text(json.dumps({"objects": objects}), encoding="utf-8")

    def runs(self) -> int:
        if not self._log.is_file():
            return 0
        return sum("argv" in json.loads(line) for line in self._log.read_text(encoding="utf-8").splitlines())


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCodex:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "codex"
    exe.write_text(f"#!{sys.executable}\n" + (LLM / "tests" / "fake_cli.py").read_text(encoding="utf-8"),
                   encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps({"objects": []}), encoding="utf-8")
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_ANSWER", str(answer))
    monkeypatch.setenv("FAKE_MODE", "ok")
    monkeypatch.setenv("FAKE_AUTH", "subscription")
    monkeypatch.delenv("MACHINEKIND_TWIN_LLM_KEEP", raising=False)
    return FakeCodex(exe, log, answer)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def source(clock: Clock) -> Synthetic:
    return Synthetic(clock)


@pytest.fixture
def perception(clock: Clock, source: Synthetic) -> Iterator[Perception]:
    p = Perception(SimpleNamespace(), workspace(), frame=source, hand=lambda: None, clock=clock)
    yield p
    started = list(p.running().values())
    p.stop_all("koniec testu")
    for runner in started:                  # stop does not join the worker; the test does
        thread = getattr(runner.detector, "_thread", None)
        if thread is not None:
            thread.join(5.0)


def start_seeded(perception: Perception, clock: Clock, seeded_package: str, fake: FakeCodex,
                 **kw: Any) -> DetectorRunner:
    plugin = load_detector(seeded_package)
    raw = {"cli": "codex", "executable": fake.exe, "max_side_px": SIDE,
           "labels_sizes": "cuboid5=0.05x0.025x0.025",
           "labels_hsv": "cuboid5=0-12,120-255,70-255|170-180,120-255,70-255",
           "interval_s": 0.2, "seed_interval_s": 2.0}
    raw.update(kw)
    runner = perception.start_detector(plugin, plugin.check(raw), threaded=False)
    runner.detector._clock = clock  # type: ignore[attr-defined]
    return runner


def wait_for_the_answer(runner: DetectorRunner, timeout: float = 5.0) -> None:
    det: Any = runner.detector
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with det._lock:
            if not det._busy:
                return
        time.sleep(0.02)
    pytest.fail(f"the seed question gave no answer in {timeout} s")


def cycle(runner: DetectorRunner, clock: Clock) -> None:
    """One call, the worker's answer to it when it asked, and the next call 0.2 s later."""
    runner.step_once()
    wait_for_the_answer(runner)
    clock.advance(0.2)
    runner.step_once()


def yaw_of(rotation: np.ndarray) -> float:
    return math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))


def half_turn_apart(a_deg: float, b_deg: float) -> float:
    return abs((a_deg - b_deg + 90.0) % 180.0 - 90.0)


def test_the_seed_becomes_a_box_with_yaw_in_the_base_frame(perception: Perception, clock: Clock,
                                                           source: Synthetic, seeded_package: str,
                                                           fake: FakeCodex):
    x, y, yaw = 0.22, 0.03, 35.0
    fake.answers_block(x, y, yaw)
    source.images = frames_of(x, y, yaw)
    runner = start_seeded(perception, clock, seeded_package, fake)

    cycle(runner, clock)

    assert runner.status.failures == 0, runner.status.last_error
    assert runner.status.dropped == 0
    assert runner.status.last_labels == (LABEL,)
    assert fake.runs() == 1
    block = perception.objects()[LABEL]
    np.testing.assert_allclose(block.position, (x, y, HALF[2]), atol=3e-3)
    assert half_turn_apart(yaw_of(block.rotation), yaw) <= 3.0
    assert block.source == "kamery"
    assert block.age_s <= 0.3
    (view,) = perception.snapshot()
    assert view.method == "sylwetka" and view.n_cameras == 2 and view.detector == "seeded"
    assert view.size_m == (0.05, 0.025, 0.025)


def test_the_pose_follows_the_block_between_questions(perception: Perception, clock: Clock, source: Synthetic,
                                                      seeded_package: str, fake: FakeCodex):
    x, y, yaw = 0.22, 0.03, 35.0
    fake.answers_block(x, y, yaw)
    source.images = frames_of(x, y, yaw)
    runner = start_seeded(perception, clock, seeded_package, fake, seed_interval_s=60.0)
    cycle(runner, clock)

    for step in range(1, 4):                # 2 cm along x per call
        moved = x + 0.02 * step
        source.images = frames_of(moved, y, yaw)
        clock.advance(0.2)
        runner.step_once()
        block = perception.objects()[LABEL]
        np.testing.assert_allclose(block.position, (moved, y, HALF[2]), atol=3e-3)
        assert block.source == "kamery"
        assert half_turn_apart(yaw_of(block.rotation), yaw) <= 3.0
    assert fake.runs() == 1


def test_a_hidden_block_ages_to_last_seen_and_expires(perception: Perception, clock: Clock, source: Synthetic,
                                                      seeded_package: str, fake: FakeCodex):
    x, y, yaw = 0.22, 0.03, 35.0
    fake.answers_block(x, y, yaw)
    source.images = frames_of(x, y, yaw)
    runner = start_seeded(perception, clock, seeded_package, fake, seed_interval_s=60.0)
    cycle(runner, clock)
    assert perception.objects()[LABEL].source == "kamery"

    source.images = frames_of()             # the block is gone from both cameras
    ages = []
    for _ in range(3):
        clock.advance(0.2)
        runner.step_once()
        block = perception.objects()[LABEL]
        assert block.source == "ostatnie widziane"
        ages.append(block.age_s)
    assert ages == sorted(ages) and ages[-1] == pytest.approx(0.6, abs=1e-6)

    clock.advance(2.5)                      # 3.1 s since the last sighting
    runner.step_once()
    assert runner.detector._tracks == {}  # type: ignore[attr-defined]
    assert LABEL in perception.objects()
    clock.advance(3.0)                      # 6.1 s: past hold_s
    runner.step_once()
    assert LABEL not in perception.objects()
    assert runner.status.failures == 0
    assert fake.runs() == 1


def test_a_failed_question_shows_in_the_runner_status(perception: Perception, clock: Clock, source: Synthetic,
                                                      seeded_package: str, fake: FakeCodex,
                                                      monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_MODE", "exit1")
    source.images = frames_of(0.22, 0.03, 35.0)
    runner = start_seeded(perception, clock, seeded_package, fake)

    cycle(runner, clock)

    assert runner.status.failures == 1
    assert runner.status.last_error.startswith("RuntimeError: seeded: CliError")
    assert LABEL not in perception.objects()
