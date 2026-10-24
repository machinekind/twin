"""Tracking (`tracks.py`): the old tracker per label, `age_s`, the bounded source rewrite and the far
hand. An injected clock, no sleeping, no render.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import BOT

from machinekind_twin.kinematics import pose
from machinekind_twin.objects import Located
from machinekind_twin.perception import ObjectTracker
from machinekind_twin.robots import get_spec
from machinekind_twin.tracks import FAR_HAND, Hand, Tracks

SPEC = get_spec(BOT)
JAW = SPEC.jaw
#: Where the test object rests: the centre of a 3 cm cube on the table.
REST = np.array([0.20, 0.0, 0.015])
#: The cube detector's interval and its freshness bound (twice the interval).
CUBE_INTERVAL = 0.33
CUBE_FRESH = 0.66


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def tracks(clock: Clock) -> Tracks:
    return Tracks(SPEC, clock)


def located(t: float, label: str = "cube", pos=REST, confidence: float = 0.9, n_cameras: int = 2,
            method: str = "sylwetka", age_s: float = 0.0) -> Located:
    return Located(label, np.array(pos, float), np.eye(3), confidence, n_cameras, t, (0.03, 0.03, 0.03), method,
                   age_s)


def hand_at(tcp, grip_q: float, grip_cmd: float, grip_closed: float = 0.0) -> Hand:
    return Hand(pose(np.eye(3), np.array(tcp, float)), grip_q, grip_cmd, grip_closed)


#: A jaw wider than the cube and commanded open: holds nothing, squeezes nothing.
OPEN = 2.0 * JAW.open
#: A jaw stopped on something: well wider than commanded (closed) and than the empty closed jaw.
SQUEEZE = 4.0 * JAW.margin


# ------------------------------------------------------------------ timing rules
def test_fresh_s_is_twice_the_interval_within_its_bounds():
    assert Tracks.fresh_s_for(0.33) == pytest.approx(0.66)
    assert Tracks.fresh_s_for(10.0) == pytest.approx(1.0)
    assert Tracks.fresh_s_for(0.1) == pytest.approx(0.5)


def test_hold_s_scales_with_latency():
    assert Tracks.hold_s_for(0.33, 0.01) == Tracks.HOLD_S
    assert Tracks.hold_s_for(10.0, 20.0) == pytest.approx(90.0)


# ------------------------------------------------------------------ source and age
def test_the_measuring_step_publishes_kamery_with_the_capture_age(tracks: Tracks, clock: Clock):
    t = clock.t
    clock.advance(0.1)
    tracks.observe("cube", [located(t)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    obj = tracks.step(None)["cube"]
    assert obj.source == "kamery"
    assert obj.age_s == pytest.approx(clock.t - t, abs=1e-6)
    np.testing.assert_allclose(obj.position, REST)
    np.testing.assert_allclose(obj.rotation, np.eye(3))


def test_kamery_holds_while_the_answer_has_the_label_and_the_pose_is_fresh(tracks: Tracks, clock: Clock):
    t = clock.t
    tracks.observe("cube", [located(t)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    assert tracks.step(None)["cube"].source == "kamery"
    ages = []
    # The tracker itself says "ostatnie widziane" on the steps without a new measurement.
    for _ in range(6):
        clock.advance(0.1)
        obj = tracks.step(None)["cube"]
        assert obj.source == "kamery"
        ages.append(obj.age_s)
    assert ages == sorted(ages) and ages[-1] == pytest.approx(0.6, abs=1e-6)
    # The label is still in the latest answer, but the pose is older than fresh_s.
    clock.t = t + CUBE_FRESH + 0.01
    obj = tracks.step(None)["cube"]
    assert obj.source == "ostatnie widziane"
    assert obj.age_s == pytest.approx(CUBE_FRESH + 0.01, abs=1e-6)


def test_a_measurement_old_on_arrival_is_ostatnie_widziane_on_its_accept_step(tracks: Tracks, clock: Clock):
    t = clock.t
    clock.advance(20.0)
    tracks.observe("red pen", [located(t, "red pen", method="triangulacja")], latency_s=20.0, interval_s=10.0)
    obj = tracks.step(None)["red pen"]
    assert obj.source == "ostatnie widziane"
    assert obj.age_s == pytest.approx(20.0, abs=1e-6)


def test_an_answer_without_the_label_ends_kamery(tracks: Tracks, clock: Clock):
    tracks.observe("cube", [located(clock.t)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    assert tracks.step(None)["cube"].source == "kamery"
    clock.advance(0.1)
    tracks.observe("cube", [], latency_s=0.05, interval_s=CUBE_INTERVAL)
    obj = tracks.step(None)["cube"]
    assert obj.source == "ostatnie widziane"
    assert obj.age_s == pytest.approx(0.1, abs=1e-6)


def test_a_measurement_is_fed_once(tracks: Tracks, clock: Clock, monkeypatch: pytest.MonkeyPatch):
    fed = []
    update = ObjectTracker.update

    def spy(self, det, *args):
        fed.append(det is not None)
        return update(self, det, *args)

    monkeypatch.setattr(ObjectTracker, "update", spy)
    tracks.observe("cube", [located(clock.t)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    for _ in range(5):
        tracks.step(None)
        clock.advance(0.05)
    assert fed == [True, False, False, False, False]
    tracks.observe("cube", [located(clock.t)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    tracks.step(None)
    assert fed[-1] is True


def test_a_slow_detector_keeps_its_track_until_the_next_answer(clock: Clock):
    slow, fast = Tracks(SPEC, clock), Tracks(SPEC, clock)
    t = clock.t
    clock.advance(20.0)
    slow.observe("detector", [located(t)], latency_s=20.0, interval_s=10.0)
    fast.observe("detector", [located(t)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    assert "cube" in slow.step(None)
    # A 20 s old measurement is past the cube detector's hold, so it is never fed.
    assert fast.step(None) == {}
    clock.advance(30.0)
    obj = slow.step(None)["cube"]
    assert obj.source == "ostatnie widziane"
    assert obj.age_s == pytest.approx(50.0, abs=1e-6)


def test_repeats_of_one_answer_hold_its_pose_through_a_60_s_still_scene_and_one_interval(
        tracks: Tracks, clock: Clock):
    t0 = clock.t
    tracks.observe("llm", [located(t0)], latency_s=0.0, interval_s=10.0)
    tracks.step(None)
    for k in range(1, 6):
        clock.t = t0 + 10.0 * k
        tracks.observe("llm", [located(t0, age_s=10.0 * k)], latency_s=0.0, interval_s=10.0)
        obj = tracks.step(None)["cube"]

        assert obj.source == "ostatnie widziane"
        assert obj.age_s == pytest.approx(10.0 * k, abs=1e-6)

    clock.t = t0 + 70.0
    obj = tracks.step(None)["cube"]

    assert obj.age_s == pytest.approx(70.0, abs=1e-6)
    np.testing.assert_allclose(obj.position, REST)


def test_a_repeated_answer_s_age_counts_as_latency_for_the_hold(tracks: Tracks, clock: Clock):
    t = clock.t
    clock.advance(50.0)
    tracks.observe("llm", [located(t, age_s=50.0)], latency_s=0.0, interval_s=10.0)
    fed_at = clock.t
    obj = tracks.step(None)["cube"]

    assert obj.source == "ostatnie widziane"
    assert obj.age_s == pytest.approx(50.0, abs=1e-6)

    clock.t = fed_at + Tracks.hold_s_for(10.0, 50.0) - 0.1

    assert "cube" in tracks.step(None)

    clock.advance(0.2)

    assert "cube" not in tracks.step(None)


def test_a_pose_expires_after_hold_s_and_its_track_goes(tracks: Tracks, clock: Clock):
    tracks.observe("cube", [located(clock.t)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    assert "cube" in tracks.step(None)
    clock.advance(Tracks.HOLD_S - 0.1)
    assert "cube" in tracks.step(None)
    clock.advance(0.2)
    assert tracks.step(None) == {}
    assert tracks.snapshot() == []
    clock.advance(Tracks.HOLD_S + 0.1)
    tracks.step(None)
    assert tracks._tracks == {}


def test_confidence_at_the_gate_is_ignored(tracks: Tracks, clock: Clock):
    tracks.observe("cube", [located(clock.t, confidence=0.4)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    assert tracks.step(None) == {}
    clock.advance(0.1)
    tracks.observe("cube", [located(clock.t, confidence=0.41)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    assert "cube" in tracks.step(None)


# ------------------------------------------------------------------ the hand
def test_a_squeezing_hand_carries_the_object_and_drops_it_to_the_table(tracks: Tracks, clock: Clock):
    tcp = REST + [0.0, 0.0, 0.01]
    tracks.observe("cube", [located(clock.t)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    assert tracks.step(hand_at(tcp, OPEN, OPEN))["cube"].source == "kamery"
    clock.advance(0.05)
    tracks.step(hand_at(tcp, SQUEEZE, 0.0))          # closing: the jaw still moves
    clock.advance(0.05)
    obj = tracks.step(hand_at(tcp, SQUEEZE, 0.0))["cube"]    # stopped on the cube: held
    assert obj.source == "w dloni" and obj.age_s == 0.0
    np.testing.assert_allclose(obj.position, REST, atol=1e-9)
    clock.advance(0.05)
    obj = tracks.step(hand_at(tcp + [0.02, 0.0, 0.05], SQUEEZE, 0.0))["cube"]
    assert obj.source == "w dloni" and obj.age_s == 0.0
    np.testing.assert_allclose(obj.position, REST + [0.02, 0.0, 0.05], atol=1e-9)
    clock.advance(0.05)
    obj = tracks.step(hand_at(tcp + [0.02, 0.0, 0.05], OPEN, OPEN))["cube"]
    assert obj.source == "ostatnie widziane (upuszczona)"
    np.testing.assert_allclose(obj.position, [REST[0] + 0.02, REST[1], REST[2]], atol=1e-9)
    assert obj.age_s > 0.0


def test_an_empty_jaw_closed_to_its_hardware_limit_never_holds(clock: Clock):
    # The jaw closes no further than the hardware allows, short of the MJCF limit (0 here) that a
    # closed command asks for. With the hardware limit as `grip_closed` the empty closed jaw does
    # not look stopped on something; with the MJCF limit it would (G1).
    hardware_closed = 3.0 * JAW.margin
    tcp = REST + [0.0, 0.0, 0.01]
    for grip_closed, held in ((hardware_closed, False), (0.0, True)):
        tracks = Tracks(SPEC, clock)
        tracks.observe("cube", [located(clock.t)], latency_s=0.05, interval_s=CUBE_INTERVAL)
        sources = []
        for _ in range(10):
            sources.append(tracks.step(hand_at(tcp, hardware_closed, 0.0, grip_closed))["cube"].source)
            clock.advance(0.05)
        assert ("w dloni" in sources) is held


def test_without_a_hand_nothing_is_held_and_one_camera_near_the_base_counts(tracks: Tracks, clock: Clock):
    near_base = [0.05, 0.0, 0.015]
    tracks.observe("cube", [located(clock.t, pos=near_base, n_cameras=1, method="promien")],
                   latency_s=0.05, interval_s=CUBE_INTERVAL)
    obj = tracks.step(None)["cube"]
    assert obj.source == "kamery"
    np.testing.assert_allclose(obj.position, near_base)
    for _ in range(10):
        clock.advance(0.05)
        assert tracks.step(None)["cube"].source != "w dloni"
    np.testing.assert_allclose(FAR_HAND.T_tcp[:3, 3], [0.0, 0.0, 10.0])


def test_a_hand_at_the_object_holds_back_a_single_camera(tracks: Tracks, clock: Clock):
    # The same single-camera detection next to the TCP: the tracker cannot tell it from a lifted
    # cube, so a hand that stays put does not accept it at once.
    near_base = np.array([0.05, 0.0, 0.015])
    tracks.observe("cube", [located(clock.t, pos=near_base, n_cameras=1, method="promien")],
                   latency_s=0.05, interval_s=CUBE_INTERVAL)
    assert tracks.step(hand_at(near_base + [0.0, 0.0, 0.03], 0.0, 0.0)) == {}


def test_step_falls_back_to_the_far_hand_when_reading_the_hand_fails(tracks: Tracks, clock: Clock):
    def broken() -> Hand:
        raise RuntimeError("arm gone")

    tracks.observe("cube", [located(clock.t, n_cameras=1)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    assert tracks.step(broken)["cube"].source == "kamery"


# ------------------------------------------------------------------ several detectors, rate, clear
def test_clear_removes_only_that_detectors_tracks(tracks: Tracks, clock: Clock):
    tracks.observe("cube", [located(clock.t)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    tracks.observe("llm", [located(clock.t, "red pen", pos=[0.25, 0.05, 0.005])], latency_s=0.05, interval_s=1.0)
    assert set(tracks.step(None)) == {"cube", "red pen"}
    assert {v.detector for v in tracks.snapshot()} == {"cube", "llm"}
    tracks.clear("llm")
    assert [v.label for v in tracks.snapshot()] == ["cube"]
    assert set(tracks.step(None)) == {"cube"}
    clock.advance(0.05)
    assert set(tracks.step(None)) == {"cube"}
    tracks.clear()
    assert tracks.snapshot() == [] and tracks.step(None) == {}


def test_the_newer_capture_wins_a_shared_label(tracks: Tracks, clock: Clock):
    t = clock.t
    tracks.observe("a", [located(t + 0.1, pos=[0.20, 0.0, 0.015])], latency_s=0.05, interval_s=CUBE_INTERVAL)
    tracks.observe("b", [located(t, pos=[0.30, 0.0, 0.015])], latency_s=0.05, interval_s=CUBE_INTERVAL)
    clock.advance(0.2)
    np.testing.assert_allclose(tracks.step(None)["cube"].position, [0.20, 0.0, 0.015])
    assert tracks.snapshot()[0].detector == "a"


def test_two_steps_within_10_ms_read_the_hand_once(tracks: Tracks, clock: Clock):
    calls = []

    def hand() -> Hand | None:
        calls.append(clock.t)
        return None

    tracks.observe("cube", [located(clock.t)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    first = tracks.step(hand)
    clock.advance(0.005)
    second = tracks.step(hand)
    assert len(calls) == 1 and tracks.last_step() == calls[0]
    assert second["cube"].age_s == first["cube"].age_s
    clock.advance(0.006)
    tracks.step(hand)
    assert len(calls) == 2


def test_snapshot_never_steps(tracks: Tracks, clock: Clock):
    tracks.observe("cube", [located(clock.t)], latency_s=0.05, interval_s=CUBE_INTERVAL)
    assert tracks.snapshot() == []
    tracks.step(None)
    clock.advance(1.0)
    view = tracks.snapshot()[0]
    assert (view.label, view.source, view.age_s, view.n_cameras, view.method) == ("cube", "kamery", 0.0, 2, "sylwetka")
    assert view.size_m == (0.03, 0.03, 0.03) and view.confidence == 0.9
