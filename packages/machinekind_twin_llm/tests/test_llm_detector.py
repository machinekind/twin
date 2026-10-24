"""LlmDetector against a fake claude and codex (tests/fake_cli.py): settings, start, detect, stop.
No model, no network."""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from machinekind_twin_api import Detection, DetectorInfo, DetectRequest, Frame
from machinekind_twin_llm import LlmDetector, LlmSettings
from machinekind_twin_llm.cli import KEEP_ENV, CliCancelled

FAKE = Path(__file__).with_name("fake_cli.py")
INFO = DetectorInfo(cameras=("top", "side"), robot="testbot", simulated=True)

#: Frames 1024x768, sent with max_side_px=512 as 512x384: scale 2.
W, H, SIDE = 1024, 768, 512
#: The model's answer in the SENT image's pixels.
ANSWER = {"objects": [
    {"label": "Red Pen", "camera": "top", "u": 100.0, "v": 50.0,
     "box": {"u0": 90.0, "v0": 40.0, "u1": 110.0, "v1": 60.0}, "confidence": 0.9},
    {"label": "red pen", "camera": "side", "u": 200.0, "v": 150.0, "box": None, "confidence": 0.8},
    {"label": "blue cup", "camera": "top", "u": 10.0, "v": 10.0, "box": None, "confidence": 0.1},
]}


class Fake:
    def __init__(self, bin_dir: Path, log: Path) -> None:
        self.bin = bin_dir
        self._log = log

    def exe(self, cli: str) -> str:
        return str(self.bin / cli)

    def runs(self) -> list[dict[str, Any]]:
        if not self._log.is_file():
            return []
        lines = self._log.read_text(encoding="utf-8").splitlines()
        return [r for r in map(json.loads, lines) if "argv" in r]


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fake:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    body = FAKE.read_text(encoding="utf-8")
    for name in ("claude", "codex"):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(ANSWER), encoding="utf-8")
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_ANSWER", str(answer))
    monkeypatch.setenv("FAKE_MODE", "ok")
    monkeypatch.setenv("FAKE_AUTH", "subscription")
    monkeypatch.delenv(KEEP_ENV, raising=False)
    return Fake(bin_dir, log)


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


def image(shade: int = 120) -> np.ndarray:
    img = np.full((H, W, 3), shade, np.uint8)
    img[300:400, 400:500] = (200, 30, 30)
    return img


def request(step: int = 0, arms: Mapping[str, np.ndarray] | None = None, **offered: np.ndarray | None) -> DetectRequest:
    return DetectRequest(frames=Frames(offered or {"top": image(), "side": image()}), step=step,
                         arm_masks={} if arms is None else arms)


def with_dark_box(x: int) -> np.ndarray:
    """`image()` with a dark 40 px box at column `x`: frames that differ for each `x` 60 px apart."""
    img = image()
    img[600:640, x:x + 40] = (40, 40, 40)
    return img


def with_arm(x: int, reach: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """`image()` with a dark arm bar 80 px wide at column `x`, reaching `reach` px down from the top
    edge, and its bool mask."""
    mask = np.zeros((H, W), bool)
    mask[:reach, x:x + 80] = True
    img = image()
    img[mask] = (30, 30, 30)
    return img, mask


def settings(fake: Fake, **kw: Any) -> LlmSettings:
    cli = kw.pop("cli", "codex")
    base: dict[str, Any] = {"cli": cli, "model": "" if cli == "claude" else "gpt-6-astra",
                            "executable": fake.exe(cli), "max_side_px": SIDE}
    base.update(kw)
    return LlmSettings(**base)


def started(fake: Fake, **kw: Any) -> LlmDetector:
    det = LlmDetector(settings(fake, **kw))
    det.start(INFO)
    return det


# --- settings -------------------------------------------------------------------------------------


def test_the_defaults_are_valid_and_set_the_contract_attributes():
    det = LlmDetector()

    assert det.config == LlmSettings()
    assert det.interval_s == 10.0
    assert det.pixel_error_px == 25.0
    assert LlmDetector.settings is LlmSettings


@pytest.mark.parametrize("kw, problem", [
    ({"cli": "gemini"}, "cli must be one of codex, claude"),
    ({"effort": "ultra"}, "effort must be one of"),
    ({"interval_s": 1.9}, "interval_s must be at least 2"),
    ({"interval_s": float("nan")}, "interval_s must be at least 2"),
    ({"timeout_s": 4.0}, r"timeout_s must be in \[5, 900\]"),
    ({"timeout_s": 901.0}, r"timeout_s must be in \[5, 900\]"),
    ({"max_side_px": 255}, r"max_side_px must be in \[256, 4096\]"),
    ({"max_side_px": 4097}, r"max_side_px must be in \[256, 4096\]"),
    ({"pixel_error_px": -1.0}, r"pixel_error_px must be in \[0, 500\]"),
    ({"reuse_max_age_s": -1.0}, r"reuse_max_age_s must be in \[0, 3600\]"),
    ({"reuse_max_age_s": float("inf")}, r"reuse_max_age_s must be in \[0, 3600\]"),
    ({"pixel_error_px": 501.0}, r"pixel_error_px must be in \[0, 500\]"),
    ({"min_confidence": -0.1}, r"min_confidence must be in \[0, 1\]"),
    ({"min_confidence": 1.1}, r"min_confidence must be in \[0, 1\]"),
    ({"max_calls_per_hour": -1}, "max_calls_per_hour must be 0"),
    ({"claude_images": "inline"}, "claude_images must be one of stream, read"),
    ({"cli": "claude", "model": "gpt-6-astra"}, "is an OpenAI model and cli is claude"),
    ({"cli": "codex", "model": "sonnet"}, "is a Claude model and cli is codex"),
    ({"cli": "codex", "model": "Fable"}, "is a Claude model and cli is codex"),
    ({"cli": "codex", "model": "claude-opus-5-5"}, "is a Claude model and cli is codex"),
])
def test_settings_the_detector_cannot_work_with_are_refused(kw: dict[str, Any], problem: str):
    with pytest.raises(ValueError, match=problem):
        LlmDetector(LlmSettings(**kw))


@pytest.mark.parametrize("kw", [
    {"cli": "claude", "model": "sonnet"}, {"cli": "claude", "model": "claude-opus-5-5"},
    {"cli": "claude", "model": ""}, {"cli": "codex", "model": ""}, {"effort": ""}, {"effort": "max"},
    {"interval_s": 2.0}, {"timeout_s": 5.0}, {"max_side_px": 4096}, {"pixel_error_px": 0.0},
])
def test_settings_at_the_edges_are_accepted(kw: dict[str, Any]):
    LlmDetector(LlmSettings(**kw))


# --- start ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("cli", ["codex", "claude"])
@pytest.mark.parametrize("auth", ["subscription", "oauth_token"])
def test_start_accepts_a_subscription_login(fake: Fake, monkeypatch: pytest.MonkeyPatch, cli: str, auth: str):
    monkeypatch.setenv("FAKE_AUTH", auth)

    det = started(fake, cli=cli)

    assert det._work is not None and det._work.is_dir()
    det.stop("test")
    assert not det._work.exists()


@pytest.mark.parametrize("cli", ["codex", "claude"])
@pytest.mark.parametrize("auth, problem", [("apikey", "not a subscription"), ("logged_out", "is not logged in")])
def test_start_refuses_an_api_key_or_no_login(fake: Fake, monkeypatch: pytest.MonkeyPatch, cli: str, auth: str,
                                              problem: str):
    monkeypatch.setenv("FAKE_AUTH", auth)

    with pytest.raises(RuntimeError, match=problem):
        started(fake, cli=cli)


def test_start_refuses_a_missing_executable(fake: Fake, tmp_path: Path):
    with pytest.raises(RuntimeError, match="not found or not executable"):
        started(fake, executable=str(tmp_path / "no-such-codex"))


def test_start_refuses_a_cli_missing_from_path(fake: Fake, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    with pytest.raises(RuntimeError, match="codex not found on PATH; install it or set `executable`"):
        started(fake, executable="")


def test_start_finds_the_cli_on_path(fake: Fake, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", f"{fake.bin}{os.pathsep}{os.environ.get('PATH', '')}")

    det = started(fake, executable="")

    assert det._exe == fake.exe("codex")
    det.stop("test")


@pytest.mark.parametrize("source", ["executable", "path"])
def test_relative_cli_paths_still_work_in_the_call_directory(
        fake: Fake, monkeypatch: pytest.MonkeyPatch, source: str):
    monkeypatch.chdir(fake.bin.parent)
    if source == "path":
        monkeypatch.setenv("PATH", "bin")
    det = started(fake, executable="./bin/codex" if source == "executable" else "")
    try:
        assert det.detect(request())
        (run,) = fake.runs()
        assert Path(run["cwd"]) != fake.bin.parent
        assert Path(det._exe).is_absolute()
    finally:
        det.stop("test")


def test_start_with_an_unknown_camera_lists_the_available_ones(fake: Fake):
    with pytest.raises(ValueError, match=r"unknown camera\(s\) in `cameras`: back; available: top, side"):
        started(fake, cameras="top, back")


# --- detect ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("cli, route", [("codex", "stream"), ("claude", "stream"), ("claude", "read")])
def test_detect_end_to_end_returns_detections_in_frame_pixels(fake: Fake, cli: str, route: str):
    det = started(fake, cli=cli, claude_images=route, objects="red pen")

    found = det.detect(request())

    assert all(isinstance(d, Detection) for d in found)
    by_camera = {d.camera: d for d in found}
    assert sorted(by_camera) == ["side", "top"]
    assert [d.label for d in found] == ["red pen", "red pen"]  # the blue cup was not asked for
    top = by_camera["top"]
    assert top.pixel == pytest.approx((100.5 * 2 - 0.5, 50.5 * 2 - 0.5))
    assert top.box == pytest.approx((90.5 * 2 - 0.5, 40.5 * 2 - 0.5, 110.5 * 2 - 0.5, 60.5 * 2 - 0.5))
    assert top.pixel_error_px == pytest.approx(25.0 * 2)
    assert by_camera["side"].pixel == pytest.approx((200.5 * 2 - 0.5, 150.5 * 2 - 0.5))
    (run,) = fake.runs()
    assert [i["size"] for i in run["images"]] == [[SIDE, SIDE * H // W]] * 2
    assert "red pen" in run["stdin"]
    assert det._work is not None and not any(det._work.iterdir())  # the call directory is gone
    det.stop("test")


def test_detect_drops_answers_below_min_confidence(fake: Fake):
    det = started(fake, min_confidence=0.85)

    assert [(d.label, d.camera) for d in det.detect(request())] == [("red pen", "top")]
    det.stop("test")


def test_a_camera_offered_without_a_frame_is_not_sent(fake: Fake):
    det = started(fake)

    found = det.detect(request(top=image(), side=None))

    (run,) = fake.runs()
    assert len(run["images"]) == 1
    assert {d.camera for d in found} == {"top"}  # the answer's "side" item names a camera not sent
    det.stop("test")


def test_the_cameras_setting_sends_only_those(fake: Fake):
    det = started(fake, cameras="side")

    found = det.detect(request())

    (run,) = fake.runs()
    assert len(run["images"]) == 1
    assert {d.camera for d in found} == {"side"}
    det.stop("test")


def test_no_frame_at_all_raises(fake: Fake):
    det = started(fake)

    with pytest.raises(RuntimeError, match="no fresh frame from cameras top, side"):
        det.detect(request(top=None, side=None))
    assert fake.runs() == []
    det.stop("test")


def test_unchanged_images_repeat_the_last_answer_without_asking(fake: Fake):
    det = started(fake)

    first = det.detect(request(0))
    again = det.detect(request(1))

    assert len(fake.runs()) == 1
    assert again == first

    det.detect(request(2, top=image(shade=20), side=image()))
    det.detect(request(3, top=image(shade=20), side=image()))

    assert len(fake.runs()) == 2
    det.stop("test")


def test_a_small_object_moving_in_one_camera_asks_again_once_it_settles(fake: Fake):
    det = started(fake)
    before = image()
    before[600:624, 100:124] = (40, 40, 40)
    moved = image()
    moved[600:624, 140:164] = (40, 40, 40)

    det.detect(request(0, top=before, side=image()))
    det.detect(request(1, top=moved, side=image()))
    det.detect(request(2, top=moved, side=image()))

    assert len(fake.runs()) == 2
    det.stop("test")


def test_an_answer_older_than_reuse_max_age_s_is_not_repeated(fake: Fake):
    det = started(fake, reuse_max_age_s=30.0)
    det.detect(request(0))
    det._last_t -= 31.0

    det.detect(request(1))

    assert len(fake.runs()) == 2
    det.stop("test")


def test_without_reuse_every_call_asks(fake: Fake):
    det = started(fake, reuse_if_still=False)

    det.detect(request(0))
    det.detect(request(1))

    assert len(fake.runs()) == 2
    det.stop("test")


def test_a_repeat_is_not_a_question_for_the_hourly_cap(fake: Fake):
    det = started(fake, max_calls_per_hour=1)
    det.detect(request(0))

    assert det.detect(request(1)) == det.detect(request(2))
    det.detect(request(3, top=image(shade=20), side=image()))
    with pytest.raises(RuntimeError, match="hourly cap of 1 questions reached"):
        det.detect(request(4, top=image(shade=20), side=image()))
    assert len(fake.runs()) == 1
    det.stop("test")


def test_a_change_waits_one_call_and_asks_once_two_calls_agree(fake: Fake):
    det = started(fake)
    first = det.detect(request(0, top=with_dark_box(100), side=image()))

    waiting = det.detect(request(1, top=with_dark_box(400), side=image()))

    assert len(fake.runs()) == 1
    assert waiting == first

    det.detect(request(2, top=with_dark_box(400), side=image()))

    assert len(fake.runs()) == 2
    det.stop("test")


def test_a_scene_that_keeps_moving_is_asked_after_two_repeats(fake: Fake):
    det = started(fake)
    counts = []

    for k in range(7):
        det.detect(request(k, top=with_dark_box(100 + 60 * k), side=image()))
        counts.append(len(fake.runs()))

    assert counts == [1, 1, 1, 2, 2, 2, 3]
    det.stop("test")


def test_without_an_answer_yet_a_moving_scene_is_asked_about(fake: Fake, monkeypatch: pytest.MonkeyPatch):
    det = started(fake)
    monkeypatch.setenv("FAKE_MODE", "exit1")
    with pytest.raises(Exception, match="boom"):
        det.detect(request(0, top=with_dark_box(100), side=image()))
    monkeypatch.setenv("FAKE_MODE", "ok")

    found = det.detect(request(1, top=with_dark_box(400), side=image()))

    assert len(fake.runs()) == 2
    assert {d.label for d in found} == {"red pen"}
    det.stop("test")


def test_an_old_answer_is_asked_again_while_the_scene_moves(fake: Fake):
    det = started(fake)
    det.detect(request(0, top=with_dark_box(100), side=image()))
    det._last_t -= det.config.reuse_max_age_s + 1

    det.detect(request(1, top=with_dark_box(400), side=image()))

    assert len(fake.runs()) == 2
    det.stop("test")


def test_a_still_frame_restarts_the_count_of_repeats_for_motion(fake: Fake):
    det = started(fake)
    counts = []

    for k, x in enumerate((100, 400, 100, 220, 280, 340)):
        det.detect(request(k, top=with_dark_box(x), side=image()))
        counts.append(len(fake.runs()))

    assert counts == [1, 1, 1, 1, 1, 2]
    det.stop("test")


def test_a_failed_question_while_the_scene_moves_is_asked_again_on_the_next_call(fake: Fake,
                                                                                  monkeypatch: pytest.MonkeyPatch):
    det = started(fake)
    det.detect(request(0, top=with_dark_box(100), side=image()))
    det.detect(request(1, top=with_dark_box(160), side=image()))
    det.detect(request(2, top=with_dark_box(220), side=image()))
    monkeypatch.setenv("FAKE_MODE", "exit1")
    with pytest.raises(Exception, match="boom"):
        det.detect(request(3, top=with_dark_box(280), side=image()))
    with pytest.raises(Exception, match="boom"):
        det.detect(request(4, top=with_dark_box(340), side=image()))
    monkeypatch.setenv("FAKE_MODE", "ok")

    det.detect(request(5, top=with_dark_box(400), side=image()))

    assert len(fake.runs()) == 4
    det.stop("test")


def test_the_arm_reaching_further_under_its_mask_repeats_the_answer(fake: Fake):
    reaches = [with_arm(200, reach) for reach in (200, 300, 400, 500, 500)]
    masked = started(fake)

    for k, (img, mask) in enumerate(reaches):
        masked.detect(request(k, arms={"top": mask}, top=img, side=image()))

    assert len(fake.runs()) == 1
    masked.stop("test")

    unmasked = started(fake)
    for k, (img, _) in enumerate(reaches):
        unmasked.detect(request(k, top=img, side=image()))

    assert len(fake.runs()) == 1 + 2
    unmasked.stop("test")


def arm_calls(fake: Fake, det: LlmDetector, columns: tuple[int, ...], start: int = 0) -> list[int]:
    """Calls `det` with the arm at each of `columns` in turn, nothing else moving; the questions
    asked so far after each call."""
    counts = []
    for k, x in enumerate(columns, start):
        img, mask = with_arm(x)
        det.detect(request(k, arms={"top": mask}, top=img, side=image()))
        counts.append(len(fake.runs()))
    return counts


def test_the_arm_leaving_its_place_at_the_question_is_asked_about_once_it_settles(fake: Fake):
    det = started(fake)

    assert arm_calls(fake, det, (100, 600, 600)) == [1, 1, 2]
    det.stop("test")


def test_the_arm_moving_over_the_table_after_a_question_without_it_repeats_the_answer(fake: Fake):
    det = started(fake)
    det.detect(request(0, arms={"top": np.zeros((H, W), bool)}, top=image(), side=image()))

    assert arm_calls(fake, det, (350, 600, 600, 100, 350, 600), start=1) == [1] * 6
    det.stop("test")


def test_a_place_the_arm_hid_at_the_question_is_asked_about_once_it_shows(fake: Fake):
    """Whatever lay there before the arm came, the answer was given without seeing it."""
    det = started(fake)
    no_arm = np.zeros((H, W), bool)
    arm = np.zeros((H, W), bool)
    arm[:450, 380:520] = True
    covered = with_dark_box(100)
    covered[arm] = (30, 30, 30)
    uncovered = with_dark_box(100)

    det.detect(request(0, arms={"top": no_arm}, top=image(), side=image()))
    det.detect(request(1, arms={"top": arm}, top=covered, side=image()))
    det.detect(request(2, arms={"top": arm}, top=covered, side=image()))

    assert len(fake.runs()) == 2

    det.detect(request(3, arms={"top": no_arm}, top=uncovered, side=image()))
    det.detect(request(4, arms={"top": no_arm}, top=uncovered, side=image()))

    assert len(fake.runs()) == 3
    det.stop("test")


def test_an_object_the_arm_uncovers_is_asked_about_and_bare_table_is_not(fake: Fake):
    counts = []
    for left in (True, False):
        det = started(fake)
        holding, arm = with_arm(200)
        released = image()
        if left:
            released[400:440, 220:260] = (40, 40, 40)
        no_arm = {"top": np.zeros((H, W), bool)}
        det.detect(request(0, arms=no_arm, top=image(), side=image()))
        det.detect(request(1, arms={"top": arm}, top=holding, side=image()))
        before = len(fake.runs())
        det.detect(request(2, arms=no_arm, top=released, side=image()))
        det.detect(request(3, arms=no_arm, top=released, side=image()))
        counts.append(len(fake.runs()) - before)
        det.stop("test")

    assert counts == [1, 0]


def test_a_mask_of_the_wrong_shape_is_ignored(fake: Fake):
    det = started(fake)

    for k, reach in enumerate((200, 300, 300)):
        img, mask = with_arm(200, reach)
        det.detect(request(k, arms={"top": np.pad(mask, ((0, 0), (0, 1)))}, top=img, side=image()))

    assert len(fake.runs()) == 2
    det.stop("test")


class RecordingMasks(Mapping[str, np.ndarray]):
    """Arm masks for every camera that record which cameras were read and hold no mask."""

    def __init__(self) -> None:
        self.read: list[str] = []

    def __getitem__(self, name: str) -> np.ndarray:
        self.read.append(name)
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return iter(("top", "side"))

    def __len__(self) -> int:
        return 2


def test_arm_masks_are_read_only_for_the_cameras_sent_and_only_with_reuse(fake: Fake):
    top_only = RecordingMasks()
    det = started(fake, cameras="top")
    det.detect(request(0, arms=top_only))
    det.stop("test")

    no_reuse = RecordingMasks()
    det = started(fake, reuse_if_still=False)
    det.detect(request(0, arms=no_reuse))
    det.stop("test")

    assert top_only.read == ["top"]
    assert no_reuse.read == []


def test_the_hourly_cap_refuses_the_next_question(fake: Fake):
    det = started(fake, max_calls_per_hour=1, reuse_if_still=False)
    det.detect(request(0))

    with pytest.raises(RuntimeError, match="hourly cap of 1 questions reached"):
        det.detect(request(1))
    assert len(fake.runs()) == 1
    det.stop("test")


def test_keep_env_keeps_a_copy_of_the_call(fake: Fake, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    keep = tmp_path / "keep"
    monkeypatch.setenv(KEEP_ENV, str(keep))
    det = started(fake)

    det.detect(request())

    (call,) = keep.iterdir()
    assert sorted(p.name for p in call.iterdir()) == [
        "img-1.jpg", "img-2.jpg", "last.txt", "schema.json", "stderr.txt", "stdin.txt", "stdout.txt"]
    det.stop("test")


def test_detect_before_start_raises(fake: Fake):
    with pytest.raises(RuntimeError, match="before start"):
        LlmDetector(settings(fake)).detect(request())


# --- stop -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("cli", ["codex", "claude"])
def test_stop_during_a_question_cancels_it_and_removes_the_work_dir(fake: Fake, monkeypatch: pytest.MonkeyPatch,
                                                                     cli: str):
    det = started(fake, cli=cli)
    work = det._work
    assert work is not None
    monkeypatch.setenv("FAKE_MODE", "sleep")
    raised: list[BaseException] = []

    def ask() -> None:
        try:
            det.detect(request())
        except BaseException as e:  # noqa: BLE001 - the test inspects it
            raised.append(e)

    thread = threading.Thread(target=ask)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not fake.runs() and time.monotonic() < deadline:
        time.sleep(0.05)
    t0 = time.monotonic()
    det.stop("test")
    thread.join(5.0)

    assert not thread.is_alive()
    assert time.monotonic() - t0 < 3.0
    assert len(raised) == 1 and isinstance(raised[0], CliCancelled)
    assert not work.exists()
    with pytest.raises(CliCancelled):
        det.detect(request())
