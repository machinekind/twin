"""Safety regressions through the real A1X backend, with an in-memory CAN bus."""

import pytest
import struct

from test_a1x import ARM, P, Clock, FakeA1XBus, connected, run as _drive


def run(arm, bus, clock, seconds):
    _drive(arm, bus, clock, seconds, renew=False)


def test_unknown_gripper_cannot_open_a_held_object_on_first_command():
    arm, bus, clock = connected()
    arm.send_joints({"gripper": 100.0})  # Display placeholder is not a measurement.
    run(arm, bus, clock, 0.05)
    assert not bus.frames(P.GRIP_ID)
    with pytest.raises(RuntimeError, match="gripper.*feedback"):
        arm.send_joints({"arm_joint1": 40.0, "gripper": 0.0})
    run(arm, bus, clock, 0.05)
    assert not bus.tx  # Reject the entire command, not just its gripper component.
    assert not arm.gripper_known()


@pytest.mark.parametrize("raw,opening", [(.6, 0.0), (-.7, 50.0), (-2.0, 100.0)])
def test_calibrated_feedback_endpoints_and_midpoint_are_measurements(raw, opening):
    arm, bus, clock = connected(bus=FakeA1XBus(grip_q=raw), gripper_fb_closed=.6, gripper_fb_open=-2)
    assert arm.gripper_measured() and arm.gripper_known()
    assert arm.read_joints()["gripper"] == pytest.approx(opening, abs=.03)
    assert not bus.tx


def test_first_calibrated_squeeze_starts_at_held_opening_without_reopening():
    arm, bus, clock = connected(bus=FakeA1XBus(grip_q=-.7), gripper_fb_closed=.6, gripper_fb_open=-2)
    arm.send_joints({"gripper": 0.0})
    run(arm, bus, clock, .02)
    commands = [struct.unpack_from(">h", frame)[0] / P.POS_SCALE for frame in bus.frames(P.GRIP_ID)]
    assert commands and -.7 <= commands[0] <= -.68
    assert commands == sorted(commands)  # More positive closes; an open-start would be near -2.


def test_fresh_feedback_does_not_keep_an_abandoned_command_alive():
    arm, bus, clock = connected()
    arm.send_joints({"arm_joint1": 100.0})
    run(arm, bus, clock, 0.30)
    count = len(bus.tx)
    run(arm, bus, clock, 0.30)
    assert len(bus.tx) == count
    assert any("command timeout" in f for f in arm.faults())
    assert not arm.link_silent  # Telemetry is healthy; the controller is not.
    assert arm.send_joints({"arm_joint1": 100.0}) == {}
    run(arm, bus, clock, 0.10)
    assert len(bus.tx) == count


def test_cancellation_is_latched_and_recovery_does_not_replay_old_targets():
    arm, bus, clock = connected()
    arm.send_joints({"arm_joint1": 100.0})
    run(arm, bus, clock, 0.05)
    arm.cancel_commands("injected controller failure")
    count = len(bus.tx)
    assert arm.send_joints({"arm_joint1": 100.0}) == {}
    run(arm, bus, clock, 0.10)
    assert len(bus.tx) == count
    arm.resume_commands()
    run(arm, bus, clock, 0.10)
    assert len(bus.tx) == count
    arm.send_joints({"arm_joint1": 15.0})
    run(arm, bus, clock, 0.05)
    assert len(bus.tx) > count
    assert not bus.frames(P.FF_ID)  # No disable or re-enable sequence.


def test_sender_io_exception_cancels_targets_and_remains_latched_after_recovery(monkeypatch):
    arm, bus, clock = connected()
    arm.send_joints({"arm_joint1": 100.0})
    run(arm, bus, clock, .02)
    receive = bus.recv

    def broken_recv(timeout=0):
        arm._stop.set()
        raise OSError("injected CAN receive failure")

    monkeypatch.setattr(bus, "recv", broken_recv)
    arm._loop()
    assert not arm.goal_positions()
    count = len(bus.tx)
    monkeypatch.setattr(bus, "recv", receive)
    monkeypatch.setattr(arm._stop, "wait", lambda timeout: arm._stop.set())
    arm._stop.clear()
    arm._loop()  # One healthy I/O iteration clears the bus error, not the stop latch.
    assert not arm._bus_error
    assert arm.send_joints({"arm_joint1": 100.0}) == {}
    run(arm, bus, clock, .02)
    assert len(bus.tx) == count
    arm.resume_commands()
    run(arm, bus, clock, .02)
    assert len(bus.tx) == count


def test_reconnect_discards_previous_targets_without_transmitting():
    arm, bus, clock = connected()
    arm.send_joints({"arm_joint1": 100.0})
    run(arm, bus, clock, .02)
    arm.disconnect()
    count = len(bus.tx)
    bus.tick(n=60)
    arm.connect()
    run(arm, bus, clock, .02)
    assert not arm.goal_positions()
    assert len(bus.tx) == count


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_command_is_rejected_before_any_component_is_applied(value):
    arm, bus, clock = connected()
    with pytest.raises(ValueError, match="finite"):
        arm.send_joints({"arm_joint1": 40.0, "arm_joint2": value})
    run(arm, bus, clock, 0.05)
    assert not bus.tx


@pytest.fixture
def controlled_twin(monkeypatch):
    pytest.importorskip("mujoco")
    from lerobot_mp.robot.a1x import A1XArm
    from lerobot_mp.twin import runtime
    from lerobot_mp.twin.robots import A1X
    from lerobot_mp.twin.workspace import Workspace

    clock = Clock()
    bus = FakeA1XBus([A1X.home[n] for n in ARM], grip_q=-0.96)
    bus.tick(n=60)
    made = []

    def create(cfg):
        cfg.a1x.gripper_fb_closed, cfg.a1x.gripper_fb_open = 0.6, -2.0
        arm = A1XArm(cfg, bus=bus, clock=clock, threaded=False)
        made.append(arm)
        return arm

    monkeypatch.setattr(runtime, "create_backend", create)
    twin = runtime.Twin(Workspace(robot="a1x"))
    twin.connect("a1x", port="fake", threaded=False)
    arm = made[0]

    def advance(seconds):
        for _ in range(round(seconds / 0.02)):
            _drive(arm, bus, clock, 0.02, renew=False)
            twin.step(0.02)

    try:
        yield twin, arm, bus, clock, advance
    finally:
        twin.close()


def test_controller_exception_cancels_the_real_backend_stream(controlled_twin, monkeypatch):
    twin, arm, bus, clock, advance = controlled_twin
    twin.set_engaged(True)
    twin.set_target({"arm_joint1": 20.0})
    advance(0.10)
    assert bus.frames(P.CMD_ID)

    def fail(_dt):
        raise RuntimeError("injected supervisor failure")

    monkeypatch.setattr(arm, "step", fail)
    advance(0.02)
    assert not twin.connected
    assert "injected supervisor failure" in twin.status.error
    count = len(bus.tx)
    _drive(arm, bus, clock, 0.5, renew=False)
    assert len(bus.tx) == count
    assert not bus.frames(P.FF_ID)


def test_stop_and_clear_do_not_restart_a_previous_move(controlled_twin):
    twin, arm, bus, clock, advance = controlled_twin
    twin.set_engaged(True)
    twin.set_target({"arm_joint1": 20.0})
    advance(0.10)
    twin.estop("operator STOP")
    count = len(bus.tx)
    advance(0.10)
    assert len(bus.tx) == count
    twin.clear_estop()
    advance(0.10)
    assert not twin.status.engaged
    # Clearing STOP may hold the measured pose, but cannot restore the old goal.
    assert abs(twin.status.measured["arm_joint1"]) < 10.0
    twin.set_engaged(True)
    twin.set_target({"arm_joint1": 7.0})
    advance(0.3)
    assert twin.status.measured["arm_joint1"] == pytest.approx(7.0, abs=1.0)
