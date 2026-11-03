"""SeededDetector against the LLM package's fake codex (`machinekind_twin_llm/tests/fake_cli.py`):
start and stop, the seed, the colour track between questions, losing and dropping a track,
failed questions. No model, no network.

The detector's timers run on an injected clock that the tests advance by hand. The worker runs a
real subprocess, so a test that waits for an answer polls the worker on the wall clock.
"""

from __future__ import annotations

import json
import stat
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from machinekind_twin_api import Detection, DetectorInfo, DetectRequest, Frame
from machinekind_twin_llm.cli import KEEP_ENV
from machinekind_twin_seeded import SeededDetector, SeededSettings

#: The LLM package's fake CLI, read where it lives: the twin's tests/test_llm_in_twin.py does the same.
FAKE = Path(__file__).resolve().parents[2] / "machinekind_twin_llm" / "tests" / "fake_cli.py"
INFO = DetectorInfo(cameras=("top", "side"), robot="testbot", simulated=True)

#: Frames 640x480, sent with max_side_px=320 as 320x240: a sent pixel maps back with scale 2, and
#: seed_error_px 25 is 50 px in the frame.
W, H, SIDE = 640, 480, 320
SCALE = W / SIDE
RED = (200, 30, 30)
#: The squares' side [px].
SQUARE = 40


class Fake:
    def __init__(self, bin_dir: Path, log: Path, answer: Path) -> None:
        self.bin = bin_dir
        self._log = log
        self._answer = answer

    def exe(self, cli: str) -> str:
        return str(self.bin / cli)

    def runs(self) -> list[dict[str, Any]]:
        if not self._log.is_file():
            return []
        lines = self._log.read_text(encoding="utf-8").splitlines()
        return [r for r in map(json.loads, lines) if "argv" in r]

    def answers(self, *items: dict[str, Any]) -> None:
        """What the fake answers from its next run on."""
        self._answer.write_text(json.dumps({"objects": list(items)}), encoding="utf-8")


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fake:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    body = FAKE.read_text(encoding="utf-8")
    for name in ("claude", "codex"):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "fake.log"
    answer = tmp_path / "answer.json"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_ANSWER", str(answer))
    monkeypatch.setenv("FAKE_MODE", "ok")
    monkeypatch.setenv("FAKE_AUTH", "subscription")
    monkeypatch.delenv(KEEP_ENV, raising=False)
    made = Fake(bin_dir, log, answer)
    made.answers()
    return made


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


class Frames(Mapping[str, Frame]):
    """Like the twin's lazy frames: every offered camera iterates; one without a frame raises KeyError."""

    def __init__(self, offered: Mapping[str, np.ndarray | None]) -> None:
        self._offered = dict(offered)

    def __getitem__(self, name: str) -> Frame:
        image = self._offered[name]
        if image is None:
            raise KeyError(name)
        return Frame(image=image, age_s=0.0)

    def __iter__(self) -> Iterator[str]:
        return iter(self._offered)

    def __len__(self) -> int:
        return len(self._offered)


def image(*squares: tuple[int, int], size_px: int = SQUARE, colour: tuple[int, int, int] = RED) -> np.ndarray:
    """A grey frame with squares of `colour` at these top-left corners (u, v)."""
    img = np.full((H, W, 3), 120, np.uint8)
    for u, v in squares:
        img[v:v + size_px, u:u + size_px] = colour
    return img


def sent(u: float, v: float) -> tuple[float, float]:
    """A frame pixel in the sent image's pixels (pixel centres kept, like the detector)."""
    return (u + 0.5) / SCALE - 0.5, (v + 0.5) / SCALE - 0.5


def seed(u0: int, v0: int, label: str = "red cube", camera: str = "top", size_px: int = SQUARE) -> dict[str, Any]:
    """The model's answer item for a square at (u0, v0): its centre and its box, in sent pixels."""
    u, v = sent(u0 + (size_px - 1) / 2, v0 + (size_px - 1) / 2)
    a, b = sent(u0, v0)
    c, d = sent(u0 + size_px - 1, v0 + size_px - 1)
    return {"label": label, "camera": camera, "u": u, "v": v, "box": {"u0": a, "v0": b, "u1": c, "v1": d},
            "confidence": 0.9}


def request(step: int = 0, **offered: np.ndarray | None) -> DetectRequest:
    return DetectRequest(frames=Frames(offered or {"top": image(), "side": image()}), step=step)


def top(*squares: tuple[int, int]) -> DetectRequest:
    """A call offering the top camera only, with these squares."""
    return request(top=image(*squares))


def settings(fake: Fake, **kw: Any) -> SeededSettings:
    base: dict[str, Any] = {"cli": "codex", "model": "gpt-6-astra", "executable": fake.exe("codex"),
                            "max_side_px": SIDE}
    base.update(kw)
    return SeededSettings(**base)


@pytest.fixture
def started(fake: Fake) -> Iterator[Callable[..., SeededDetector]]:
    """Builds a detector on a fresh `Clock` and starts it; every one is stopped and joined afterwards."""
    made: list[SeededDetector] = []

    def start(**kw: Any) -> SeededDetector:
        det = SeededDetector(settings(fake, **kw), clock=Clock())
        det.start(INFO)
        made.append(det)
        return det

    yield start
    for det in made:
        det.stop("koniec testu")
        if det._thread is not None:
            det._thread.join(5.0)


def clock_of(det: SeededDetector) -> Clock:
    clock = det._clock
    assert isinstance(clock, Clock)
    return clock


def answered(det: SeededDetector, timeout: float = 5.0) -> None:
    """Waits until the worker has stored its answer (the question runs a subprocess)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with det._lock:
            if not det._busy:
                return
        time.sleep(0.02)
    pytest.fail(f"the worker gave no answer in {timeout} s")


def ended(det: SeededDetector, timeout: float = 3.0) -> None:
    """Joins the worker (stop does not) and checks that it ended."""
    assert det._thread is not None
    det._thread.join(timeout)
    assert not det._thread.is_alive()


def centre(u0: int, v0: int, size_px: int = SQUARE) -> tuple[float, float]:
    return u0 + (size_px - 1) / 2, v0 + (size_px - 1) / 2


# --- start and stop -------------------------------------------------------------------------------


def test_start_starts_the_llm_and_the_worker_and_stop_ends_both(started: Callable[..., SeededDetector]):
    det = started()
    work = det._llm._work

    assert work is not None and work.is_dir()
    assert det._thread is not None and det._thread.is_alive()
    assert det._thread.name == "detektor-seed:codex" and det._thread.daemon
    t0 = time.monotonic()
    det.stop("test")
    assert time.monotonic() - t0 < 0.1
    ended(det)
    assert not work.exists()
    with pytest.raises(RuntimeError, match="seeded: stopped"):
        det.detect(request())


def test_start_refuses_when_the_llm_refuses(fake: Fake, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_AUTH", "logged_out")
    det = SeededDetector(settings(fake))

    with pytest.raises(RuntimeError, match="is not logged in"):
        det.start(INFO)
    assert det._thread is None


def test_detect_before_start_raises(fake: Fake):
    with pytest.raises(RuntimeError, match="seeded: detect before start"):
        SeededDetector(settings(fake)).detect(request())


def test_stop_during_a_question_cancels_it_and_the_worker_ends_within_three_seconds(
        fake: Fake, started: Callable[..., SeededDetector], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_MODE", "sleep")
    det = started()
    det.detect(top((300, 200)))
    deadline = time.monotonic() + 10.0
    while not fake.runs() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert fake.runs()

    t0 = time.monotonic()
    det.stop("test")
    assert time.monotonic() - t0 < 0.1     # it does not join
    ended(det, 3.0)
    assert det._answer is None


# --- seeds and tracks -----------------------------------------------------------------------------


def test_the_first_call_asks_and_the_seed_grows_a_track_on_the_next(
        fake: Fake, started: Callable[..., SeededDetector]):
    fake.answers(seed(300, 200))
    det = started()

    assert det.detect(top((300, 200))) == []
    answered(det)
    assert len(fake.runs()) == 1
    (run,) = fake.runs()
    assert [i["size"] for i in run["images"]] == [[SIDE, SIDE * H // W]]
    clock_of(det).advance(0.2)
    (found,) = det.detect(top((300, 200)))

    assert isinstance(found, Detection)
    assert (found.label, found.camera) == ("red cube", "top")
    assert found.mask is not None and found.mask.dtype == bool and found.mask.shape == (H, W)
    assert found.mask[200:240, 300:340].all() and int(found.mask.sum()) == SQUARE * SQUARE
    assert found.box == (300.0, 200.0, 339.0, 239.0)
    assert found.size_m == (0.029, 0.029, 0.029)
    assert found.pixel_error_px == 6.0 and found.confidence == 1.0
    assert found.pixel == pytest.approx(centre(300, 200), abs=1.0)


def test_the_track_follows_the_blob_between_questions_and_never_asks_sooner_than_seed_interval_s(
        fake: Fake, started: Callable[..., SeededDetector]):
    fake.answers(seed(200, 200))
    det = started(seed_interval_s=2.0)
    clock = clock_of(det)
    det.detect(top((200, 200)))
    answered(det)

    for k in range(1, 12):
        clock.advance(0.2)
        u = 200 + 20 * k
        (found,) = det.detect(top((u, 200)))
        assert found.pixel == pytest.approx(centre(u, 200), abs=1.0)
        if k <= 9:                          # 1.8 s on the detector's clock: no second question
            assert det._asks == 1
        elif k == 11:                       # 2.2 s
            assert det._asks == 2
    answered(det)
    assert len(fake.runs()) == 2


def test_a_blob_farther_than_the_seed_error_is_not_snapped(fake: Fake,
                                                           started: Callable[..., SeededDetector]):
    item = seed(500, 200)                   # 200 px right of the square, beyond 50 px
    item["box"] = None
    fake.answers(item)
    det = started(seed_interval_s=60.0)
    clock = clock_of(det)
    det.detect(top((300, 200)))
    answered(det)

    clock.advance(0.2)
    assert det.detect(top((300, 200))) == []
    assert ("red cube", "top") in det._tracks
    clock.advance(2.9)                      # 2.9 s after the answer arrived
    assert det.detect(top((300, 200))) == []
    assert ("red cube", "top") in det._tracks
    clock.advance(0.2)                      # 3.1 s
    assert det.detect(top((300, 200))) == []
    assert det._tracks == {}
    clock.advance(0.2)
    assert det.detect(top((500, 200))) == []   # under the old pixel now, but nothing seeds it
    assert len(fake.runs()) == 1


def test_a_fresh_seed_gets_lost_after_s_from_the_answer_s_arrival_not_from_the_question(
        fake: Fake, started: Callable[..., SeededDetector]):
    fake.answers(seed(300, 200))
    det = started(seed_interval_s=60.0)
    clock = clock_of(det)
    det.detect(top((300, 200)))
    clock.advance(5.0)                      # the question took 5 s on the detector's clock
    answered(det)

    assert det.detect(top()) == []          # drained now; the square is gone
    assert ("red cube", "top") in det._tracks
    clock.advance(2.9)
    assert det.detect(top()) == []
    assert ("red cube", "top") in det._tracks
    clock.advance(0.2)
    assert det.detect(top()) == []
    assert det._tracks == {}


def test_a_lost_blob_omits_the_label_and_the_next_answer_reseeds(
        fake: Fake, started: Callable[..., SeededDetector]):
    fake.answers(seed(300, 200))
    det = started(seed_interval_s=4.5)
    clock = clock_of(det)
    det.detect(top((300, 200)))
    answered(det)
    clock.advance(0.2)
    assert len(det.detect(top((300, 200)))) == 1

    clock.advance(0.2)
    assert det.detect(top()) == []          # gone: the label is omitted
    assert det._tracks[("red cube", "top")].lost
    clock.advance(3.1)                      # 3.3 s since the last sighting
    assert det.detect(top()) == []
    assert det._tracks == {}

    clock.advance(1.5)                      # 5.0 s since the first question: the next one goes out
    assert det.detect(top((300, 200))) == []
    assert det._asks == 2
    answered(det)
    clock.advance(0.2)
    (found,) = det.detect(top((300, 200)))
    assert found.label == "red cube"


def test_a_live_track_outranks_a_new_seed(fake: Fake, started: Callable[..., SeededDetector]):
    fake.answers(seed(200, 200))
    det = started(seed_interval_s=2.0)
    clock = clock_of(det)
    det.detect(top((200, 200)))
    answered(det)
    for u in (220, 240, 260):               # the block moves 60 px away from the seed
        clock.advance(0.2)
        det.detect(top((u, 200)))
    clock.advance(1.6)                      # 2.2 s: the second question, still pointing at (200, 200)
    det.detect(top((260, 200)))
    assert det._asks == 2
    answered(det)

    clock.advance(0.2)
    # Another block of the colour now lies where the model points: a new seed would snap to it.
    found = det.detect(top((260, 200), (180, 200)))

    assert [d.pixel for d in found] == [pytest.approx(centre(260, 200), abs=1.0)]
    assert det._tracks[("red cube", "top")].unconfirmed == 0


def test_three_answers_without_the_label_drop_a_live_track(fake: Fake,
                                                           started: Callable[..., SeededDetector]):
    fake.answers(seed(300, 200))
    det = started(seed_interval_s=2.0)
    clock = clock_of(det)
    det.detect(top((300, 200)))
    answered(det)
    fake.answers()                          # from now on the model sees nothing
    clock.advance(0.2)
    assert len(det.detect(top((300, 200)))) == 1

    for unconfirmed in (1, 2):
        clock.advance(2.0)
        assert len(det.detect(top((300, 200)))) == 1     # asks again
        answered(det)
        clock.advance(0.2)
        assert len(det.detect(top((300, 200)))) == 1     # the empty answer, drained
        assert det._tracks[("red cube", "top")].unconfirmed == unconfirmed
    clock.advance(2.0)
    det.detect(top((300, 200)))
    answered(det)
    clock.advance(0.2)

    assert det.detect(top((300, 200))) == []             # the third empty answer drops the track
    assert det._tracks == {}
    assert det._asks == 4


def test_a_failed_question_is_raised_only_when_nothing_is_tracked(
        fake: Fake, started: Callable[..., SeededDetector], monkeypatch: pytest.MonkeyPatch):
    fake.answers(seed(300, 200))
    monkeypatch.setenv("FAKE_MODE", "exit1")
    det = started()
    assert det.detect(top((300, 200))) == []
    answered(det)
    clock_of(det).advance(0.2)
    with pytest.raises(RuntimeError, match="seeded: CliError: codex exit 1"):
        det.detect(top((300, 200)))

    monkeypatch.setenv("FAKE_MODE", "ok")
    tracked = started(seed_interval_s=2.0)
    clock = clock_of(tracked)
    tracked.detect(top((300, 200)))
    answered(tracked)
    clock.advance(0.2)
    assert len(tracked.detect(top((300, 200)))) == 1
    monkeypatch.setenv("FAKE_MODE", "exit1")
    clock.advance(2.0)
    tracked.detect(top((300, 200)))         # asks again; that question fails
    answered(tracked)
    clock.advance(0.2)

    assert len(tracked.detect(top((300, 200)))) == 1
    assert tracked._last_error.startswith("CliError: codex exit 1")


def test_no_frame_at_all_raises(fake: Fake, started: Callable[..., SeededDetector]):
    det = started()

    with pytest.raises(RuntimeError, match="seeded: no fresh frame from cameras top, side"):
        det.detect(request(top=None, side=None))
    assert det._asks == 0
    assert fake.runs() == []


def test_the_frames_handed_to_the_model_are_copies(fake: Fake, started: Callable[..., SeededDetector],
                                                   monkeypatch: pytest.MonkeyPatch):
    det = started()
    got: dict[str, np.ndarray] = {}

    def record(request: DetectRequest) -> list[Detection]:
        for name in request.frames:
            got[name] = request.frames[name].image
        return []

    monkeypatch.setattr(det._llm, "detect", record)
    img = image((300, 200))
    det.detect(request(top=img, side=None))
    answered(det)

    assert list(got) == ["top"]
    assert not np.shares_memory(got["top"], img)
    img[:] = 0
    assert np.array_equal(got["top"], image((300, 200)))
    assert fake.runs() == []


# --- the real blocks -----------------------------------------------------------------------------

#: Paints, RGB: so-arm101's light blue and bare wood, and the twin simulation's table.
LIGHT_BLUE, WOOD, TABLE = (115, 185, 235), (214, 189, 166), (140, 128, 112)


def painted(background: tuple[int, int, int], *squares: tuple[int, int, tuple[int, int, int]]) -> DetectRequest:
    """A call offering the top camera only: squares (u, v, colour) on a plain background."""
    img = np.full((H, W, 3), background, np.uint8)
    for u, v, colour in squares:
        img[v:v + SQUARE, u:u + SQUARE] = colour
    return request(top=img)


def seeded_once(det: SeededDetector, call: DetectRequest) -> list[Detection]:
    """The asking call, the answer, and the call 0.2 s later that grows the seeds."""
    det.detect(call)
    answered(det)
    clock_of(det).advance(0.2)
    return list(det.detect(call))


def test_every_label_the_model_names_as_a_block_is_followed_with_its_shape_s_size(
        fake: Fake, started: Callable[..., SeededDetector]):
    fake.answers(seed(100, 100, "light blue cuboid"), seed(300, 100, "red pen"), seed(500, 100, "Red Cube"))
    call = painted((120, 120, 120), (100, 100, LIGHT_BLUE), (300, 100, RED), (500, 100, RED))

    found = {d.label: d for d in seeded_once(started(), call)}

    assert sorted(found) == ["light blue cuboid", "red cube"]
    assert found["light blue cuboid"].size_m == (0.060, 0.029, 0.014)
    assert found["light blue cuboid"].pixel == pytest.approx(centre(100, 100), abs=1.0)
    assert found["red cube"].size_m == (0.029, 0.029, 0.029)


def test_with_labels_set_only_those_are_followed(fake: Fake, started: Callable[..., SeededDetector]):
    fake.answers(seed(100, 100, "light blue cuboid"), seed(500, 100, "red cube"))
    call = painted((120, 120, 120), (100, 100, LIGHT_BLUE), (500, 100, RED))

    det = started(labels="red cube")

    assert [d.label for d in seeded_once(det, call)] == ["red cube"]
    assert det._llm.config.objects == "red cube"


def test_bare_wood_is_followed_on_a_table_of_another_colour(fake: Fake, started: Callable[..., SeededDetector]):
    fake.answers(seed(300, 200, "wooden cube"))

    (found,) = seeded_once(started(), painted((120, 120, 120), (300, 200, WOOD)))

    assert found.label == "wooden cube"
    assert found.mask is not None and int(found.mask.sum()) == SQUARE * SQUARE


@pytest.mark.parametrize("with_box", [True, False])
def test_bare_wood_on_a_wood_coloured_table_is_not_followed(fake: Fake, started: Callable[..., SeededDetector],
                                                           with_box: bool):
    item = seed(300, 200, "wooden cube")
    if not with_box:
        item["box"] = None
    fake.answers(item)
    det = started()

    assert seeded_once(det, painted(TABLE, (300, 200, WOOD))) == []
    assert det._tracks[("wooden cube", "top")].seen_at is None
