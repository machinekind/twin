"""Detector threads and the perception service (`detection.py`): `DetectorRunner` driven by
`step_once` on an injected clock, the request's lazy frames, failures and backoff, stopping, and
`Perception` with several detectors, the tick, the occluder wrapper and the hand. No render; one
wall-clock test.
"""

from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace

import machinekind_twin_api
import numpy as np
import pytest
from conftest import BOT
from machinekind_twin_api.detector import Detection
from test_objects import A, B, H, K, W, project

from machinekind_twin import detection
from machinekind_twin.clock import VirtualClock
from machinekind_twin.detection import DetectorRunner, Perception, _TwinHand
from machinekind_twin.detectors import DetectorPlugin
from machinekind_twin.kinematics import RobotKinematics
from machinekind_twin.objects import CameraModel
from machinekind_twin.robots import get_spec
from machinekind_twin.runtime import Twin
from machinekind_twin.strategy import Action, StrategyRunner
from machinekind_twin.workspace import CameraRecord, Workspace

SPEC = get_spec(BOT)
#: A point on the table where the test objects rest.
REST = np.array([0.20, 0.0, 0.015])
MODELS = {"a": A, "b": B}
IMAGE = np.zeros((H, W, 3), np.uint8)


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


class Frames:
    """A frame source stamping on the test clock: per camera, the frame's age, None = no image."""

    def __init__(self, clock, ages: dict[str, float | None] | None = None):
        self.clock = clock
        self.ages: dict[str, float | None] = dict(ages or {"a": 0.0, "b": 0.0})
        self.reads: list[str] = []

    def __call__(self, name: str):
        self.reads.append(name)
        age = self.ages.get(name)
        if age is None:
            return None, 0.0
        return IMAGE, self.clock() - age


class Scripted:
    """A detector as another package writes it, with scripted answers.

    Each answer is a list of detections (given after reading every offered frame), an exception
    to raise, or a callable taking the request.
    With `block`, a call waits on it after setting `entered`; `stop` releases it.
    """

    settings = None

    def __init__(self, *answers, interval_s: float = 0.5, refuse: str = "", block: bool = False):
        self.answers = list(answers)
        self.interval_s = interval_s
        self.pixel_error_px = 3.0
        self.refuse = refuse
        self.block = threading.Event() if block else None
        self.entered = threading.Event()
        self.requests = []
        self.info = None
        self.stops: list[tuple[str, str]] = []

    def start(self, info):
        self.info = info
        if self.refuse:
            raise RuntimeError(self.refuse)

    def stop(self, reason):
        self.stops.append((reason, threading.current_thread().name))
        if self.block is not None:
            self.block.set()

    def detect(self, request):
        self.requests.append(request)
        self.entered.set()
        if self.block is not None:
            self.block.wait(5.0)
        answer = self.answers.pop(0) if self.answers else []
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return answer(request)
        for name in request.frames:         # a real detector looks before it answers
            request.frames.get(name)
        return answer


def seen(label: str = "x", cams=("a", "b"), p=REST, **kw) -> list[Detection]:
    return [Detection(label, c, project(MODELS[c], p), height_m=float(p[2]), **kw) for c in cams]


def record(name: str, cam: CameraModel, trusted: bool = True) -> CameraRecord:
    return CameraRecord(name, "0", W, H, K=K.tolist(), dist=None if cam.dist is None else cam.dist.tolist(),
                        T_cam2base=cam.T_cam2base.tolist(), calibration={"trusted": True} if trusted else {})


def workspace(**trusted: bool) -> Workspace:
    ws = Workspace(robot=BOT)
    for name, cam in MODELS.items():
        ws.add_camera(record(name, cam, trusted.get(name, True)))
    return ws


def plugin(detector, name: str = "fake") -> DetectorPlugin:
    return DetectorPlugin(name=name, cls=lambda: detector, settings=None, fields=(), source="test", summary="")  # type: ignore[arg-type]


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def frames(clock: Clock) -> Frames:
    return Frames(clock)


@pytest.fixture
def perception(clock: Clock, frames: Frames):
    p = Perception(SimpleNamespace(), workspace(), frame=frames, hand=lambda: None, clock=clock)
    yield p
    p.stop_all("koniec testu")


def runner_for(detector, clock, frames, cameras=("a", "b"), sink=None) -> DetectorRunner:
    info = detection.DetectorInfo(tuple(cameras), BOT, False)
    fed = []

    def default_sink(name, dets, read, latency, interval):
        fed.append((name, dets, dict(read), latency, interval))
        return 0

    r = DetectorRunner("fake", detector, info=info, cameras=lambda: tuple(cameras), frame=frames,
                       sink=sink or default_sink, clock=clock)
    r.fed = fed  # type: ignore[attr-defined]
    return r


# ------------------------------------------------------------------ re-exports
def test_the_contract_types_are_the_api_package_s():
    assert detection.Detection is machinekind_twin_api.Detection
    assert detection.DetectRequest is machinekind_twin_api.DetectRequest
    assert detection.Detector is machinekind_twin_api.Detector
    assert detection.DetectorInfo is machinekind_twin_api.DetectorInfo


# ------------------------------------------------------------------ the request's frames
def test_step_once_feeds_the_objects(perception: Perception, clock: Clock):
    det = Scripted(seen())
    runner = perception.start_detector(plugin(det), {}, threaded=False)
    assert det.info.cameras == ("a", "b") and det.info.robot == SPEC.name and not det.info.simulated
    assert perception.objects() == {}

    runner.step_once()
    clock.advance(0.05)
    obj = perception.objects()["x"]
    assert obj.source == "kamery"
    np.testing.assert_allclose(obj.position, REST, atol=1e-3)
    assert obj.age_s == pytest.approx(0.05, abs=1e-6)
    assert runner.status.calls == 1 and runner.status.last_labels == ("x",)
    assert runner.status.cameras == ("a", "b")


def test_a_detection_s_age_is_part_of_the_published_age(perception: Perception, clock: Clock):
    runner = perception.start_detector(plugin(Scripted(seen(age_s=5.0))), {}, threaded=False)
    runner.step_once()
    obj = perception.objects()["x"]

    assert obj.age_s == pytest.approx(5.0, abs=1e-6)
    assert obj.source == "ostatnie widziane"


def test_frames_are_fetched_only_when_read(clock: Clock, frames: Frames):
    def read_b(req):
        _ = req.frames["b"]
        return []

    det = Scripted(read_b)
    runner = runner_for(det, clock, frames)
    runner.start(threaded=False)
    runner.step_once()
    assert frames.reads == ["b"]
    assert runner.fed[0][2] == {"b": ((W, H), clock.t)}


def test_a_frame_older_than_a_second_is_not_offered(clock: Clock, frames: Frames):
    frames.ages = {"a": 1.01, "b": 0.99}
    def read_all(req):
        _ = [req.frames.get(n) for n in req.frames]
        return []

    det = Scripted(read_all)
    runner = runner_for(det, clock, frames)
    runner.start(threaded=False)
    runner.step_once()
    req = det.requests[0]
    assert "a" not in req.frames and "b" in req.frames
    assert req.frames["b"].age_s == pytest.approx(0.99)
    assert set(runner.fed[0][2]) == {"b"}


def test_a_camera_without_a_frame_is_listed_but_missing(clock: Clock, frames: Frames):
    frames.ages = {"a": None, "b": 0.0}
    det = Scripted(lambda req: [])
    runner = runner_for(det, clock, frames)
    runner.start(threaded=False)
    runner.step_once()
    req = det.requests[0]
    assert list(req.frames) == ["a", "b"] and len(req.frames) == 2
    assert "a" not in req.frames
    assert req.frames.get("a") is None
    with pytest.raises(KeyError):
        req.frames["a"]
    with pytest.raises(KeyError):
        req.frames["nope"]
    assert req.step == 0


def test_each_call_offers_the_cameras_trusted_now(clock: Clock, frames: Frames):
    ws = workspace(b=False)
    p = Perception(SimpleNamespace(), ws, frame=frames, hand=lambda: None, clock=clock)
    det = Scripted(lambda req: [], lambda req: [], lambda req: [])
    runner = p.start_detector(plugin(det), {}, threaded=False)
    assert det.info.cameras == ("a",)

    runner.step_once()
    ws.cameras[1].calibration = {"trusted": True}
    runner.step_once()
    ws.cameras[0].calibration = {}
    runner.step_once()

    assert [tuple(r.frames) for r in det.requests] == [("a",), ("a", "b"), ("b",)]
    assert det.info.cameras == ("a",)          # the set at start, as the contract says
    p.stop_all("koniec testu")


def test_a_detection_from_a_camera_that_left_is_dropped(clock: Clock, frames: Frames):
    ws = workspace()

    def leave(req):
        _ = req.frames["a"], req.frames["b"]
        ws.cameras[0].enabled = False          # "a" leaves while the detector thinks
        return seen(cams=("a", "b"))

    p = Perception(SimpleNamespace(), ws, frame=frames, hand=lambda: None, clock=clock)
    runner = p.start_detector(plugin(Scripted(leave)), {}, threaded=False)
    runner.step_once()
    assert runner.status.dropped == 1
    clock.advance(0.05)
    assert p.objects()["x"].source == "kamery"
    assert p.snapshot()[0].n_cameras == 1
    p.stop_all("koniec testu")


# ------------------------------------------------------------------ failures and pacing
@pytest.mark.parametrize("fail", [False, True])
def test_the_current_call_is_visible_until_success_or_failure(clock: Clock, frames: Frames, fail: bool):
    clock.t = 0.0
    det = Scripted(RuntimeError("model down") if fail else [], block=True)
    runner = runner_for(det, clock, frames)
    runner.start(threaded=False)
    assert runner.status.call_started is None
    thread = threading.Thread(target=runner.step_once)
    thread.start()
    try:
        assert det.entered.wait(5.0)
        assert runner.status.call_started == 0.0
        assert runner.status.cameras == ("a", "b")
        assert runner.status.calls == 0
        clock.advance(7.0)
        assert runner.status.call_started == 0.0
        assert det.block is not None
        det.block.set()
        thread.join(5.0)
        assert not thread.is_alive()
        assert runner.status.call_started is None
        assert runner.status.calls == 1
        assert runner.status.last_latency_s == 7.0
        assert runner.status.failures == int(fail)
        assert runner.status.last_error == ("RuntimeError: model down" if fail else "")
    finally:
        assert det.block is not None
        det.block.set()
        thread.join(5.0)
        runner.stop("test")


def test_the_wait_after_a_success_is_the_interval(clock: Clock, frames: Frames):
    runner = runner_for(Scripted([], interval_s=0.33), clock, frames)
    runner.start(threaded=False)
    assert runner.step_once() == pytest.approx(0.33)
    assert runner.status.next_try == pytest.approx(clock.t + 0.33)
    runner.detector.interval_s = 0.01
    assert runner.step_once() == pytest.approx(DetectorRunner.MIN_INTERVAL_S)


def test_a_failing_detect_backs_off_and_keeps_the_tracks(perception: Perception, clock: Clock):
    det = Scripted(seen(), RuntimeError("model down"), RuntimeError("model down"), seen(), interval_s=0.5)
    runner = perception.start_detector(plugin(det), {}, threaded=False)
    assert runner.step_once() == pytest.approx(0.5)
    clock.advance(0.05)
    assert "x" in perception.objects()

    first = runner.step_once()
    assert runner.status.failures == 1
    assert runner.status.last_error == "RuntimeError: model down"
    second = runner.step_once()
    assert runner.status.failures == 2
    assert first == pytest.approx(1.0) and second == pytest.approx(2.0 * first)
    clock.advance(0.05)
    assert "x" in perception.objects()

    assert runner.step_once() == pytest.approx(0.5)
    assert runner.status.failures == 0 and runner.status.calls == 4


def test_the_backoff_is_bounded(clock: Clock, frames: Frames):
    det = Scripted(*[RuntimeError("no")] * 12, interval_s=10.0)
    runner = runner_for(det, clock, frames)
    runner.start(threaded=False)
    waits = [runner.step_once() for _ in range(12)]
    assert waits[0] == 10.0 and waits[1] == 20.0
    assert max(waits) == DetectorRunner.MAX_BACKOFF_S


def test_start_raising_refuses(perception: Perception):
    det = Scripted(refuse="not logged in")
    with pytest.raises(RuntimeError, match="not logged in"):
        perception.start_detector(plugin(det), {})
    assert perception.running() == {}
    assert det.stops == []


def test_stop_while_starting_gets_the_detector_its_one_stop(perception: Perception):
    det = Scripted()
    starting, release = threading.Event(), threading.Event()
    det.start = lambda info: (starting.set(), release.wait(5.0))
    errors = []

    def start():
        try:
            perception.start_detector(plugin(det), {})
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=start, name="detektor-start:fake")
    thread.start()
    assert starting.wait(5.0)
    assert perception.running()["fake"].status.starting
    perception.stop_detector("fake", "operator")
    release.set()
    thread.join(5.0)

    assert errors == ["detektor zatrzymany w trakcie startu"]
    assert det.stops == [("operator", "detektor-start:fake")]
    assert perception.running() == {}
    assert det.requests == []


def test_a_stop_while_the_old_detector_stops_cancels_its_replacement(perception: Perception):
    old = Scripted()
    perception.start_detector(plugin(old), {}, threaded=False)
    in_stop, release = threading.Event(), threading.Event()
    old.stop = lambda reason: (in_stop.set(), release.wait(5.0))
    new = Scripted(seen())
    errors = []

    def restart():
        try:
            perception.start_detector(plugin(new), {})
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=restart, name="detektor-start:fake")
    thread.start()
    assert in_stop.wait(5.0)                 # the replacement is registered, the old one stops
    replacement = perception.running()["fake"]
    perception.stop_detector("fake", "operator")
    release.set()
    thread.join(5.0)

    assert errors == ["detector runner 'fake' was already started or stopped"]
    assert perception.running() == {}
    assert new.info is None and new.requests == [] and new.stops == []
    assert replacement._thread is None
    assert not replacement.status.running and replacement.status.stopped_because == "operator"


def test_a_nan_interval_raises_before_the_detector_starts(clock: Clock, frames: Frames):
    det = Scripted(interval_s=math.nan)
    runner = runner_for(det, clock, frames)
    with pytest.raises(ValueError, match="interval_s"):
        runner.start(threaded=False)
    assert det.info is None
    det = Scripted()
    det.pixel_error_px = -1.0
    with pytest.raises(ValueError, match="pixel_error_px"):
        runner_for(det, clock, frames).start(threaded=False)


def test_stop_during_a_blocked_detect_discards_the_late_answer(clock: Clock, frames: Frames):
    det = Scripted(seen(), block=True)
    runner = runner_for(det, clock, frames)
    runner.start()
    assert det.entered.wait(5.0)

    runner.stop("operator")                  # the fake's stop releases the blocked call
    assert det.stops == [("operator", threading.current_thread().name)]
    assert runner._thread is not None and not runner._thread.is_alive()
    assert runner.fed == []
    assert not runner.status.running and runner.status.stopped_because == "operator"
    runner.stop("again")
    assert len(det.stops) == 1


def test_stop_clears_the_current_call_before_an_uncooperative_detector_returns(clock: Clock, frames: Frames):
    det = Scripted(seen(), block=True)
    det.stop = lambda reason: None
    runner = runner_for(det, clock, frames)
    runner.start(threaded=False)
    thread = threading.Thread(target=runner.step_once)
    thread.start()
    try:
        assert det.entered.wait(5.0)
        assert runner.status.call_started == clock.t
        runner.stop("operator", wait=False)
        assert runner.status.call_started is None
        assert thread.is_alive()
        clock.advance(7.0)
        assert det.block is not None
        det.block.set()
        thread.join(5.0)
        assert not thread.is_alive()
        assert runner.status.call_started is None
        assert not runner.status.running
        assert runner.status.calls == 0
        assert runner.status.last_done == 0.0
        assert runner.fed == []
    finally:
        assert det.block is not None
        det.block.set()
        thread.join(5.0)
        runner.stop("test")


def test_stop_during_camera_listing_cannot_restart_a_call(clock: Clock, frames: Frames):
    entered, release = threading.Event(), threading.Event()

    def cameras():
        entered.set()
        release.wait(5.0)
        return ("a",)

    det = Scripted(seen())
    runner = runner_for(det, clock, frames)
    runner._cameras = cameras
    runner.start(threaded=False)
    thread = threading.Thread(target=runner.step_once)
    thread.start()
    try:
        assert entered.wait(5.0)
        assert runner.status.call_started == clock.t
        runner.stop("operator", wait=False)
        assert runner.status.call_started is None
        release.set()
        thread.join(5.0)
        assert not thread.is_alive()
        assert runner.status.call_started is None
        assert runner.status.cameras == ()
        assert runner.status.calls == 0
        assert det.requests == []
        assert runner.fed == []
    finally:
        release.set()
        thread.join(5.0)
        runner.stop("test")


# ------------------------------------------------------------------ several detectors
def test_two_detectors_feed_their_own_labels(perception: Perception, clock: Clock):
    one = perception.start_detector(plugin(Scripted(seen("cube")), "one"), {}, threaded=False)
    two = perception.start_detector(plugin(Scripted(seen("pen", p=REST + [0.05, 0.05, 0.0])), "two"), {},
                                    threaded=False)
    assert set(perception.running()) == {"one", "two"}
    one.step_once()
    two.step_once()
    clock.advance(0.05)
    assert set(perception.objects()) == {"cube", "pen"}
    assert {v.label: v.detector for v in perception.snapshot()} == {"cube": "one", "pen": "two"}

    perception.stop_detector("one", "operator")
    clock.advance(0.05)
    assert set(perception.objects()) == {"pen"}


def test_starting_a_running_name_replaces_it(perception: Perception, clock: Clock):
    old = Scripted(seen())
    first = perception.start_detector(plugin(old), {}, threaded=False)
    first.step_once()
    clock.advance(0.05)
    assert "x" in perception.objects()

    new = Scripted()
    second = perception.start_detector(plugin(new), {}, threaded=False)
    assert perception.running() == {"fake": second}
    assert old.stops == [("ponownie z nowymi ustawieniami", threading.current_thread().name)]
    assert not first.status.running and second.status.running
    assert first.step_once() == math.inf
    clock.advance(0.05)
    assert perception.objects() == {}


def test_stop_all_joins_every_runner(perception: Perception):
    dets = [Scripted(interval_s=0.1), Scripted(interval_s=0.1)]
    runners = [perception.start_detector(plugin(d, f"d{i}"), {}) for i, d in enumerate(dets)]
    assert all(d.entered.wait(5.0) for d in dets)

    perception.stop_all("koniec")
    assert perception.running() == {}
    for d, r in zip(dets, runners, strict=True):
        assert [s for s, _ in d.stops] == ["koniec"]
        assert r._thread is not None and not r._thread.is_alive()


# ------------------------------------------------------------------ stepping
def test_tick_steps_only_when_nobody_did(perception: Perception, clock: Clock):
    runner = perception.start_detector(plugin(Scripted(seen())), {}, threaded=False)
    runner.step_once()
    clock.advance(0.05)
    perception.objects()
    first = perception.tracks.last_step()

    clock.advance(0.1)
    perception.tick(idle_s=0.15)
    assert perception.tracks.last_step() == first

    clock.advance(0.06)
    perception.tick(idle_s=0.15)
    assert perception.tracks.last_step() == clock.t
    assert perception.snapshot()[0].label == "x"


def test_the_occluder_wrapper_swallows_failures(clock: Clock, frames: Frames, caplog):
    calls = []

    def broken(times):
        calls.append(times)
        raise RuntimeError("renderer closed")

    p = Perception(SimpleNamespace(), workspace(), frame=frames, occluders=broken, hand=lambda: None, clock=clock)
    assert p._occluders({"a": 1.0}) is None
    assert p._occluders({"a": 2.0}) is None
    assert len(calls) == 2
    assert sum("renderer closed" in r.getMessage() for r in caplog.records if r.levelname == "WARNING") == 1
    masks = {"a": np.zeros((H, W), bool)}
    q = Perception(SimpleNamespace(), workspace(), frame=frames, occluders=lambda t: masks, hand=lambda: None,
                   clock=clock)
    assert q._occluders({"a": 1.0}) is masks


def test_the_mapper_is_shared_and_rebuilt_only_on_a_camera_change(perception: Perception):
    m = perception.mapper()
    assert m is not None and set(m.cameras) == {"a", "b"}
    assert perception.mapper() is m
    assert set(perception.cameras()) == {"a", "b"}
    perception.ws.cameras[1].enabled = False
    m2 = perception.mapper()
    assert m2 is not m and m2 is not None and set(m2.cameras) == {"a"}


# ------------------------------------------------------------------ the hand
class FakeTwin:
    def __init__(self, connected: bool = True, limits=None, command=None):
        self.connected = connected
        self.status = SimpleNamespace(measured={"j1": 10.0, "j2": 30.0, "gripper": 40.0}, command=command or {})
        self._limits = limits

    def joints(self):
        return dict(SPEC.home)

    def joint_limits(self):
        if self._limits is None:
            raise RuntimeError("no supervisor")
        return self._limits


def test_the_closed_jaw_comes_from_the_limits_that_hold():
    kin = RobotKinematics(SPEC)
    g = SPEC.joints.index("gripper")
    twin = FakeTwin(limits={"j1": (-90.0, 90.0), "j2": (-90.0, 90.0), "gripper": (10.0, 100.0)},
                    command={"gripper": 20.0})
    hand = _TwinHand(twin, Workspace(robot=BOT))()
    assert hand is not None
    joints = twin.status.measured
    assert hand.grip_closed == pytest.approx(kin.to_q({**joints, "gripper": 10.0})[g])
    assert hand.grip_closed != pytest.approx(kin.to_q({"gripper": 0.0})[g])
    assert hand.grip_q == pytest.approx(kin.to_q(joints)[g])
    assert hand.grip_cmd == pytest.approx(kin.to_q({**joints, "gripper": 20.0})[g])
    np.testing.assert_allclose(hand.T_tcp, kin.tcp(joints))


def test_the_closed_jaw_falls_back_to_gripper_zero_without_limits():
    kin = RobotKinematics(SPEC)
    g = SPEC.joints.index("gripper")
    hand = _TwinHand(FakeTwin(limits=None), Workspace(robot=BOT))()
    assert hand is not None
    assert hand.grip_closed == pytest.approx(kin.to_q({"gripper": 0.0})[g])
    assert hand.grip_cmd == pytest.approx(hand.grip_q)       # nothing commanded: the measured jaw


def test_a_disconnected_arm_has_no_hand():
    assert _TwinHand(FakeTwin(connected=False), Workspace(robot=BOT))() is None


# ------------------------------------------------------------------ with a strategy
def test_a_strategy_sees_the_objects(clock: Clock, frames: Frames):
    twin = Twin(Workspace(robot=BOT), clock=VirtualClock(t0=0.0))
    try:
        twin.connect("sim", threaded=False)
        p = Perception(twin, workspace(), frame=frames, hand=lambda: None, clock=clock)
        runner = p.start_detector(plugin(Scripted(seen())), {}, threaded=False)
        runner.step_once()
        clock.advance(0.05)

        seen_objects = []

        class Look:
            hz = 10.0

            def act(self, obs):
                seen_objects.append(dict(obs.objects))
                return Action()

        strategy = StrategyRunner(twin, Look(), objects=p.objects)
        strategy.start(threaded=False)
        assert strategy.step_once(0.0)
        strategy.stop()
        assert seen_objects[-1]["x"].source == "kamery"
        np.testing.assert_allclose(seen_objects[-1]["x"].position, REST, atol=1e-3)
        p.stop_all("koniec testu")
    finally:
        twin.close()


# ------------------------------------------------------------------ wall clock
def test_objects_answer_while_a_detector_blocks(frames: Frames):
    frames.clock = time.monotonic
    det = Scripted(seen(), block=False)
    blocked = threading.Event()
    release = threading.Event()

    def hang(req):
        blocked.set()
        release.wait(5.0)
        return []

    det.answers.append(hang)
    det.interval_s = DetectorRunner.MIN_INTERVAL_S
    p = Perception(SimpleNamespace(), workspace(), frame=frames, hand=lambda: None)
    try:
        p.start_detector(plugin(det), {})
        assert blocked.wait(5.0)
        t0 = time.perf_counter()
        got = [p.objects() for _ in range(100)]
        assert time.perf_counter() - t0 < 1.0
        assert all("x" in g for g in got)
    finally:
        release.set()
        p.stop_all("koniec testu")
