"""Backend `a1x`: read-only gripper state and command frames of another sender on the CAN link.

Everything runs on the `FakeA1XBus` of `test_a1x.py` with its virtual clock; no CAN link is opened.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("galaxeo")

from galaxeo import protocol as P  # noqa: E402
from galaxeo.bus import Frame  # noqa: E402
from test_a1x import ARM, STEP, Clock, FakeA1XBus, connected, make_arm, run  # noqa: E402

from lerobot_mp.robot.a1x import ECHO_KEEP, FOREIGN_WINDOW_S, A1XArm  # noqa: E402
from lerobot_mp.robot.base import GripperState, RobotBackend, RobotInfo  # noqa: E402

FOREIGN = "inny nadawca na laczu CAN (ramki rozkazu spoza blizniaka)"
#: Default stream: 2 p_des/s over the 2.6 p_des between open (-2.0) and closed (0.6).
RATE = 2.0 * 100.0 / 2.6


def state(arm: A1XArm) -> GripperState:
    gs = arm.gripper_state()
    assert gs is not None
    return gs


def snapshot(arm: A1XArm) -> dict[str, str]:
    return {k: repr(v) for k, v in vars(arm).items()}


def open_stream(arm, bus, clock) -> None:
    """A stream really sent at 100: the first command must differ from the reported opening."""
    arm.send_joints({"gripper": 99.0})
    run(arm, bus, clock, 0.1)
    arm.send_joints({"gripper": 100.0})
    run(arm, bus, clock, 0.1)
    assert state(arm).streamed == pytest.approx(100.0, abs=1e-9)


# ------------------------------------------------------------------ gripper state
def test_gripper_state_is_none_before_connect_and_after_disconnect():
    arm, bus, clock = make_arm()
    assert arm.gripper_state() is None
    bus.tick(n=60)
    arm.connect()
    assert isinstance(arm.gripper_state(), GripperState)
    arm.disconnect()
    assert arm.gripper_state() is None


def test_commanded_is_none_until_a_command_passes_the_backends_gate():
    arm, bus, clock = connected()
    run(arm, bus, clock, 0.05)
    gs = state(arm)
    assert gs.commanded is None and gs.target is None and gs.streamed is None
    assert gs.frozen is False
    assert gs.position is None and gs.measured is False
    assert gs.effort == pytest.approx(0.0, abs=1.0 / P.EFF_SCALE)
    assert gs.t == clock.t and gs.age_s == pytest.approx(0.0, abs=1e-12)
    assert gs.rate == pytest.approx(RATE, rel=1e-12)
    assert gs.force_threshold == arm.acfg.gripper_force
    with pytest.raises(dataclasses.FrozenInstanceError):
        gs.commanded = 50.0                                                     # type: ignore[misc]
    # 100 equals the reported opening right after connect: nothing streams and nothing is commanded
    arm.send_joints({"gripper": 100.0})
    run(arm, bus, clock, 0.1)
    assert state(arm).commanded is None and not bus.frames(P.GRIP_ID)
    arm.send_joints({"gripper": 60.0})
    run(arm, bus, clock, STEP)
    gs = state(arm)
    assert gs.commanded == 60.0 and gs.target == pytest.approx(60.0, abs=1e-9)
    assert gs.streamed == pytest.approx(100.0 - RATE * STEP, abs=1e-9)


def test_streamed_closes_at_the_stream_rate_while_commanded_is_already_zero():
    arm, bus, clock = connected()
    open_stream(arm, bus, clock)
    arm.send_joints({"gripper": 0.0})
    samples = []
    for _ in range(20):                                                          # 0.8 s of the 1.3 s stroke
        run(arm, bus, clock, 0.04)
        gs = state(arm)
        assert gs.commanded == 0.0 and gs.target == pytest.approx(0.0, abs=1e-9)
        samples.append((clock.t, gs.streamed))
    assert all(b[1] < a[1] for a, b in zip(samples, samples[1:]))
    (t0, s0), (t1, s1) = samples[0], samples[-1]
    assert (s0 - s1) / (t1 - t0) == pytest.approx(RATE, rel=0.05)
    assert 30.0 < samples[-1][1] < 60.0
    assert arm.read_joints()["gripper"] == 0.0                                   # the command, not the jaw


def test_effort_above_the_threshold_while_closing_freezes_the_target_at_the_stream():
    arm, bus, clock = connected()
    arm.send_joints({"gripper": 0.0})
    run(arm, bus, clock, 0.4)
    assert state(arm).frozen is False
    bus.grip_effort = 2.0
    run(arm, bus, clock, 0.05)
    gs = state(arm)
    assert gs.frozen is True
    assert gs.target == gs.streamed
    assert 0.0 < gs.streamed < 100.0
    assert gs.commanded == 0.0
    assert gs.effort == pytest.approx(2.0, abs=1.0 / P.EFF_SCALE)
    assert gs.effort > gs.force_threshold
    run(arm, bus, clock, 0.3)
    assert state(arm).streamed == gs.streamed                                    # the stream holds


def test_stale_feedback_gives_no_effort_and_age_grows_while_the_bus_is_silent():
    arm, bus, clock = connected()
    arm.send_joints({"gripper": 0.0})
    run(arm, bus, clock, 0.1)
    gs = state(arm)
    assert gs.effort is not None and gs.age_s == pytest.approx(0.0, abs=1e-12)
    t_last = gs.t
    bus.silent = True
    stale_s = arm.acfg.stale_ms / 1000.0
    ages = []
    for _ in range(8):
        run(arm, bus, clock, 0.04)
        gs = state(arm)
        assert gs.t == t_last
        assert gs.age_s == pytest.approx(clock.t - t_last, abs=1e-9)
        assert (gs.effort is None) == (gs.age_s > stale_s)
        ages.append(gs.age_s)
    assert all(b > a for a, b in zip(ages, ages[1:]))
    assert ages[-1] > stale_s and gs.effort is None
    assert gs.target is None and gs.streamed is None and gs.commanded == 0.0    # the stream was dropped


def test_position_is_none_without_gripper_feedback_points_and_measured_with_them():
    arm, bus, clock = connected()
    run(arm, bus, clock, 0.05)
    gs = state(arm)
    assert gs.position is None and gs.measured is False
    # Feedback points on the p_des endpoints: the fake jaw's group 7 reads as the opening.
    arm, bus, clock = connected(gripper_fb_closed=0.6, gripper_fb_open=-2.0)
    arm.send_joints({"gripper": 0.0})
    run(arm, bus, clock, 0.2)
    gs = state(arm)
    assert gs.measured is True
    assert gs.position == pytest.approx((bus.q[6] - 0.6) * 100.0 / -2.6, abs=0.05)


def test_a_thousand_calls_change_no_backend_field_and_send_no_frame():
    arm, bus, clock = connected()
    arm.send_joints({"arm_joint1": 30.0, "gripper": 0.0})
    run(arm, bus, clock, 0.3)

    def thousand_calls_change_nothing():
        bus.tick(n=3)                                                            # queued frames stay queued
        before, n_tx, n_rx = snapshot(arm), len(bus.tx), len(bus.rx)
        for _ in range(1000):
            arm.gripper_state()
        assert snapshot(arm) == before
        assert len(bus.tx) == n_tx and len(bus.rx) == n_rx

    thousand_calls_change_nothing()
    # Stale feedback that no `read_joints` has seen yet: unlike a read, this must not set `_link_silent`.
    bus.silent = True
    run(arm, bus, clock, 0.3, read=False)
    assert arm._link_silent is False and state(arm).effort is None
    thousand_calls_change_nothing()
    assert arm._link_silent is False


def test_gripper_state_never_raises_and_logs_the_error_once(monkeypatch, caplog):
    arm, bus, clock = connected()

    def broken():
        raise RuntimeError("zepsute")

    monkeypatch.setattr(arm, "_fresh_fb", broken)
    with caplog.at_level("ERROR", logger="lerobot_mp.robot.a1x"):
        assert arm.gripper_state() is None
        assert arm.gripper_state() is None
    assert len([r for r in caplog.records if "stanu chwytaka" in r.getMessage()]) == 1


def test_the_base_backend_has_no_gripper_state_and_sees_no_other_sender():
    class Bare(RobotBackend):
        info = RobotInfo(name="bare")

        def connect(self) -> None:
            pass

        def disconnect(self) -> None:
            pass

        def read_joints(self) -> dict[str, float]:
            return {}

        def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
            return {}

        @property
        def is_connected(self) -> bool:
            return True

    b = Bare()
    assert b.gripper_state() is None
    assert b.foreign_frames() == 0


# ------------------------------------------------------------- another sender
@pytest.mark.parametrize("can_id, payload", [
    (P.CMD_ID, P.encode_arm([0.1] * 6, 20.0, 1.0)),
    (P.GRIP_ID, P.encode_gripper(0.3, 25.0, 1.0)),
    (P.FF_ID, P.encode_ff(1)),
], ids=["arm", "gripper", "function"])
def test_a_command_frame_the_backend_never_sent_is_counted_and_reported_for_half_a_second(can_id, payload):
    assert FOREIGN_WINDOW_S == 0.5
    arm, bus, clock = connected()
    run(arm, bus, clock, 0.05)
    assert arm.foreign_frames() == 0 and arm.faults() == []
    pose = arm.read_joints()
    bus.rx.append(Frame(can_id, payload))
    run(arm, bus, clock, STEP)
    t_seen = clock.t
    assert arm.foreign_frames() == 1
    assert arm.faults() == [FOREIGN]
    assert arm.read_joints() == pose                                             # not decoded as feedback
    assert bus.tx == []                                                          # and nothing was sent
    run(arm, bus, clock, FOREIGN_WINDOW_S - 0.01)
    assert clock.t - t_seen < FOREIGN_WINDOW_S and arm.faults() == [FOREIGN]
    run(arm, bus, clock, 0.02)
    assert arm.faults() == []
    assert arm.foreign_frames() == 1                                             # counted since connect
    bus.rx.append(Frame(can_id, payload))
    run(arm, bus, clock, STEP)
    assert arm.foreign_frames() == 2 and arm.faults() == [FOREIGN]


def test_an_echo_of_the_backends_own_payloads_is_not_another_sender():
    assert ECHO_KEEP == 16
    arm, bus, clock = connected()
    arm.enable()                                                                 # FF 1, 5, 6 went out
    arm.send_joints({"arm_joint1": 40.0, "gripper": 0.0})
    run(arm, bus, clock, 0.2)                                                    # moving: payloads differ
    cmd, grip = bus.frames(P.CMD_ID), bus.frames(P.GRIP_ID)
    assert len(set(cmd[-ECHO_KEEP - 1:])) == ECHO_KEEP + 1
    echoes = [
        (P.CMD_ID, cmd[-1]), (P.GRIP_ID, grip[-1]), (P.FF_ID, b"\x06"), (P.FF_ID, b"\x01"),
        (P.CMD_ID, cmd[-ECHO_KEEP]),                                             # oldest one still remembered
        (P.CMD_ID, cmd[-1] + bytes(64 - len(cmd[-1]))),                          # padded to an FD length
        (P.GRIP_ID, grip[-1] + bytes(12 - len(grip[-1]))),
    ]
    bus.rx.extend(Frame(i, d) for i, d in echoes)
    run(arm, bus, clock, STEP)
    assert arm.foreign_frames() == 0 and arm.faults() == []
    # Older than the remembered payloads, or padded with anything but zeros: another sender.
    cmd = bus.frames(P.CMD_ID)
    bus.rx.append(Frame(P.CMD_ID, cmd[-ECHO_KEEP - 1]))
    bus.rx.append(Frame(P.GRIP_ID, bus.frames(P.GRIP_ID)[-1] + b"\x00\x01"))
    run(arm, bus, clock, STEP)
    assert arm.foreign_frames() == 2 and arm.faults() == [FOREIGN]


def test_through_the_twin_a_foreign_command_frame_ends_in_stop(monkeypatch):
    pytest.importorskip("mujoco")
    from lerobot_mp.control.safety import SafetyState
    from lerobot_mp.twin import robots
    from lerobot_mp.twin import runtime as rt
    from lerobot_mp.twin.workspace import Workspace

    spec = robots.REGISTRY["a1x"]
    clock = Clock()
    bus = FakeA1XBus([spec.home.get(n, 0.0) for n in ARM])
    bus.tick(n=60)
    made: dict[str, A1XArm] = {}

    def sleep(secs):
        clock.t += secs
        bus.tick(secs)

    def create(cfg):
        made["arm"] = A1XArm(cfg, bus=bus, clock=clock, threaded=False, sleep=sleep, connect_timeout_s=0.05)
        return made["arm"]

    monkeypatch.setattr(rt, "create_backend", create)
    twin = rt.Twin(Workspace(robot="a1x"))
    dt = 0.02

    def run_twin(secs):
        for _ in range(int(round(secs / dt))):
            for _ in range(int(round(dt / STEP))):
                clock.t += STEP
                bus.tick(STEP)
                made["arm"].io_step(clock.t)
            twin.step(dt)

    try:
        twin.connect("a1x", port="fake", threaded=False)
        run_twin(0.5)
        twin.set_engaged(True)
        twin.set_target({"arm_joint1": twin.status.measured["arm_joint1"] + 10.0})
        run_twin(0.2)
        assert twin.safety_state is not SafetyState.ESTOP and not twin.status.faults
        bus.rx.append(Frame(P.CMD_ID, P.encode_arm([0.0] * 6, 20.0, 1.0)))
        run_twin(dt)
        assert twin.safety_state is SafetyState.ESTOP
        assert FOREIGN in twin.status.faults and FOREIGN in twin.status.error
        run_twin(FOREIGN_WINDOW_S + 0.1)
        assert made["arm"].faults() == []
        assert twin.safety_state is SafetyState.ESTOP                            # STOP stays until cleared
    finally:
        twin.close()
