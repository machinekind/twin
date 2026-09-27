"""`SpecsReach` driving the `a1x` backend over a fake CAN bus (`FakeA1XBus`), no hardware.

The twin runs with `threaded=False` and the test ticks its loop and the bus (`Rig.pump`). The virtual
clock follows the wall clock because the driver's worker thread is real (ramps, waiting for arrival).
The relay is the fake from `test_twin_specsreach.py`.
"""

from __future__ import annotations

import struct
import threading
import time

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("websockets")
pytest.importorskip("galaxeo")

from galaxeo import protocol as P  # noqa: E402
from test_a1x import LSB_DEG, STEP, Clock, FakeA1XBus  # noqa: E402
from test_twin_specsreach import FakeRelay  # noqa: E402

from lerobot_mp.robot.a1x import A1XArm  # noqa: E402
from lerobot_mp.twin import robots  # noqa: E402
from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics  # noqa: E402
from lerobot_mp.twin.reach import ReachVolume  # noqa: E402
from lerobot_mp.twin.specsreach import SpecsReach  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

ARM = tuple(f"arm_joint{i}" for i in range(1, 7))
DT = 0.02                                      # twin loop period (4 bus ticks)


class WallBus(FakeA1XBus):
    """Fake bus with a wall: joint `wall[0]` can't be commanded past `wall[1]` [rad] in direction `wall[2]`."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.wall: tuple[int, float, float] | None = None       # (joint index, limit, direction sign)

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None:
        super().send(can_id, data, fd=fd)
        if self.wall is not None and self.p_des is not None:
            j, lim, sign = self.wall
            if sign * (self.p_des[j] - lim) > 0:
                self.p_des[j] = lim


class Rig:
    """Twin on the fake A1X, plus driver and relay; `pump` runs the loop at wall-clock pace."""

    def __init__(self, monkeypatch, relay: FakeRelay, offset: dict[str, float] | None = None):
        spec = robots.REGISTRY["a1x"]
        self.kin = RobotKinematics(spec)
        self.vol = ReachVolume.from_workspace(Workspace(robot="a1x"))
        sol = self.kin.ik(self.vol.center, seed=spec.home, tol=0.003)
        assert sol.ok
        self.start = {k: float(sol.joints[k]) for k in ARM}  # from home every target is a > 45 deg jump
        self.clock = Clock()
        # `offset` [deg]: start this far from the pose above the volume centre (for `start_center`).
        self.bus = WallBus([self.start[k] + (offset or {}).get(k, 0.0) for k in ARM])
        self.bus.tick(n=60)
        self.arm: A1XArm | None = None

        def sleep(secs):
            self.clock.t += secs
            self.bus.tick(secs)

        def create(cfg):
            self.arm = A1XArm(cfg, bus=self.bus, clock=self.clock, threaded=False, sleep=sleep,
                              connect_timeout_s=0.05)
            return self.arm

        monkeypatch.setattr(rt, "create_backend", create)
        self.twin = rt.Twin(Workspace(robot="a1x"))
        self.twin.connect("a1x", port="fake", threaded=False)
        self.relay = relay
        self.faults: list[str] = []
        self.driver: SpecsReach | None = None
        self.pump(0.5)

    def tick(self) -> None:
        assert self.arm is not None
        for _ in range(int(round(DT / STEP))):
            self.clock.t += STEP
            self.bus.tick(STEP)
            self.arm.io_step(self.clock.t)
        self.twin.step(DT)
        self.faults.extend(self.twin.status.faults)

    def pump(self, secs: float, until=None) -> bool:
        """Tick the loop for `secs` of wall time or until `until()`; True if `until` was met."""
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < secs:
            if until is not None and until():
                return True
            self.tick()
            n += 1
            ahead = t0 + n * DT - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)
        return until is not None and bool(until())

    def connect_driver(self, **kw) -> SpecsReach:
        kw.setdefault("mode", "real")
        kw.setdefault("real_ok", True)
        kw.setdefault("hold_off_s", 0.0)
        d = SpecsReach(self.twin, relay_url=self.relay.url, **kw)
        d.start()
        self.driver = d
        self.relay.wait_client()
        self.pump(0.3)
        return d

    def state(self, ts: int, states: set[str]) -> dict | None:
        with self.relay.cond:
            hit = [s for s in self.relay.states if s["ts"] == ts and s["state"] in states]
        return hit[-1] if hit else None

    def wait_state(self, ts: int, states: set[str], secs: float) -> dict:
        assert self.pump(secs, lambda: self.state(ts, states) is not None), \
            f"brak stanu {states}: {[(s['state'], s.get('reason')) for s in self.relay.states]}"
        st = self.state(ts, states)
        assert st is not None
        return st

    def cmd_frames(self, since: int = 0) -> list[list[float]]:
        return [self.bus.pdes_deg(d) for i, d, _ in self.bus.tx[since:] if i == P.CMD_ID]

    def close(self) -> None:
        if self.driver is not None:
            # The worker may be waiting on the loop (arrival), so keep ticking until the driver stops.
            import threading
            t = threading.Thread(target=self.driver.stop)
            t.start()
            self.pump(6.0, lambda: not t.is_alive())
            t.join(1.0)
        self.twin.close()


@pytest.fixture
def relay():
    r = FakeRelay()
    yield r
    r.close()


@pytest.fixture
def rig(monkeypatch, relay):
    r = Rig(monkeypatch, relay)
    yield r
    r.close()


def far_target(rig: Rig) -> np.ndarray:
    """Hand target ~20 deg of J1 from the volume centre, so the ramp takes over a second."""
    return rig.vol.center + np.array([0.0, 0.2, -0.05])


@pytest.mark.parametrize("mode,real_ok", [("sim", True), ("real", False)])
def test_sim_mode_or_real_without_consent_on_the_a1x_is_not_armed(rig, mode, real_ok):
    rig.connect_driver(mode=mode, real_ok=real_ok)
    ts = rig.relay.target(rig.vol.center + np.array([0.02, 0.02, 0.0]))
    st = rig.wait_state(ts, {"rejected", "moving", "reached"}, 5.0)
    assert (st["state"], st.get("reason")) == ("rejected", "not armed")
    rig.pump(0.3)
    assert rig.cmd_frames() == []
    assert rig.twin.owner is None


def test_effort_spike_mid_ramp_backs_off_slowly_toward_the_earlier_pose(rig):
    d = rig.connect_driver()
    trip: dict = {}
    real_backoff = d._backoff

    def spy(t):
        rig.bus.eff[:6] = [0.0] * 6                                # force gone, so backing off doesn't trip again
        trip["hit"] = d._impact
        trip["tx"] = len(rig.bus.tx)
        return real_backoff(t)
    d._backoff = spy
    ts = rig.relay.target(far_target(rig))
    rig.wait_state(ts, {"moving"}, 5.0)
    rig.pump(0.6)
    before = dict(rig.twin.status.measured)
    rig.bus.eff[0] = 60.0                                           # > threshold of 20 in 0x052 frame units
    st = rig.wait_state(ts, {"reached", "hold", "backoff", "rejected"}, 15.0)
    assert (st["state"], st.get("reason")) == ("backoff", "impact effort arm_joint1"), rig.relay.states_for(ts)
    hit = trip["hit"]
    assert hit is not None and hit.backoff_pose
    frames = rig.cmd_frames(trip["tx"])
    assert frames
    j1 = [f[0] for f in frames]
    # J1 moves back toward the pre-hit pose and ends at the backoff pose.
    assert abs(j1[-1] - hit.backoff_pose["arm_joint1"]) < 0.5
    ahead = np.sign(before["arm_joint1"] - rig.start["arm_joint1"])              # direction of travel
    assert ahead != 0 and (j1[-1] - j1[0]) * ahead < -2.0
    # Slow: smoothstep peak 1.5 x 8 = 12 deg/s (+ test clock jitter) over 0.2 s windows,
    # never above the supervisor's 15 deg/s cap.
    win = int(round(0.2 / STEP))
    rates = [abs(j1[i + win] - j1[i]) / 0.2 for i in range(len(j1) - win)]
    assert rates and max(rates) <= 13.0, max(rates)
    # The backend chases the loop command in jumps (90 deg/s, every 4 frames), so only windowed speed is meaningful.
    win = int(round(0.1 / STEP))
    fast = [abs(j1[i + win] - j1[i]) / 0.1 for i in range(len(j1) - win)]
    assert max(fast) <= 15.0 + 2 * LSB_DEG / 0.1, max(fast)
    assert rig.twin.safety_state.value != "estop"
    assert rig.twin.owner is None


def test_stall_at_a_wall_trips_the_guard_before_the_backend_deaf_fault(rig):
    d = rig.connect_driver()
    tripped: list[float] = []
    real_backoff = d._backoff

    def spy(t):
        tripped.append(rig.clock.t)
        return real_backoff(t)
    d._backoff = spy
    target = far_target(rig)
    ts = rig.relay.target(target)
    rig.wait_state(ts, {"moving"}, 5.0)
    rig.pump(0.3)
    # J1 gets stuck 2 deg further along its direction of travel.
    q = rig.bus.q[0]
    sign = 1.0 if rig.bus.p_des[0] >= q else -1.0
    rig.bus.wall = (0, q + sign * np.radians(2.0), sign)
    st = rig.wait_state(ts, {"reached", "hold", "backoff", "rejected"}, 15.0)
    rig.bus.wall = None
    assert (st["state"], st.get("reason")) == ("backoff", "impact lag arm_joint1"), rig.relay.states_for(ts)
    assert tripped
    assert not any("gluche" in f for f in rig.faults), rig.faults
    rig.pump(1.5)                                                   # no servo fault after the backoff either
    assert not any("gluche" in f for f in rig.faults), rig.faults
    assert rig.twin.safety_state.value != "estop"


def test_stop_mid_ramp_holds_releases_and_streams_only_the_held_pose(rig):
    d = rig.connect_driver()
    ts = rig.relay.target(far_target(rig))
    rig.wait_state(ts, {"moving"}, 5.0)
    rig.pump(0.6)
    rig.twin.estop("test")
    st = rig.wait_state(ts, {"reached", "hold", "backoff", "rejected"}, 5.0)
    assert (st["state"], st.get("reason")) == ("hold", "stop")
    assert rig.twin.owner is None
    rig.pump(0.2)
    n = len(rig.bus.tx)
    rig.pump(1.0)
    after = rig.cmd_frames(n)
    held = after[0] if after else None
    assert all(max(abs(a - b) for a, b in zip(f, held)) <= 2 * LSB_DEG for f in after)   # no motion
    assert d._worker is not None and d._worker.is_alive()


def test_relay_loss_mid_move_in_real_mode_holds(rig):
    d = rig.connect_driver()
    ts = rig.relay.target(far_target(rig))
    rig.wait_state(ts, {"moving"}, 5.0)
    rig.pump(0.4)
    rig.relay.drop_clients()
    assert rig.pump(3.0, lambda: (d.last_state or {}).get("state") == "hold")
    assert d.last_state["reason"] == "relay"
    assert rig.twin.owner is None
    rig.pump(0.3)
    cmd = dict(rig.twin.status.command)
    rig.pump(0.8)
    assert max(abs(rig.twin.status.command[k] - cmd[k]) for k in ARM) < 0.2


@pytest.mark.parametrize("left_deg", [6.0, 7.0, 8.0])
def test_start_center_stall_near_the_end_holds_before_the_backend_deaf_fault(monkeypatch, relay, left_deg):
    """`start_center` at 10 deg/s; J1 gets stuck 2 deg further when `left_deg` of the ramp remains.

    The error ends at about 4-6 deg: up to the old 6 deg lag threshold, above the backend's 3 deg "deaf"
    threshold. With 6 deg / 0.3 s (the normal-move setting) the backend had already counted up to 0.74 s
    of its 1 s deaf timeout; with 4 deg / 0.2 s, only up to 0.34 s.
    """
    rig = Rig(monkeypatch, relay, offset={"arm_joint1": -15.0})
    try:
        d = SpecsReach(rig.twin, relay_url=relay.url, mode="real", real_ok=True)
        goal = rig.start["arm_joint1"]
        trip: dict = {}
        real_interrupt = d._interrupt

        def spy(cause):
            if cause == "impact" and "deaf_age" not in trip:
                # How long the backend has been counting a joint as "deaf" when the guard trips.
                trip["deaf_age"] = max((rig.clock.t - t0 for t0 in rig.arm._deaf_since.values()), default=0.0)
            real_interrupt(cause)
        d._interrupt = spy
        out: dict = {}
        th = threading.Thread(target=lambda: out.setdefault("r", d.start_center(10.0)))
        th.start()
        assert rig.pump(5.0, lambda: goal - rig.twin.status.measured["arm_joint1"] < left_deg)
        rig.bus.wall = (0, rig.bus.q[0] + np.radians(2.0), 1.0)
        assert rig.pump(6.0, lambda: not th.is_alive()), "start_center nie skonczyl"
        th.join(1.0)
        assert out["r"] == "impact lag arm_joint1", (out, rig.faults)
        assert not any("gluche" in f for f in rig.faults), rig.faults
        assert trip["deaf_age"] <= 1.0 - 0.3, trip                     # >= 0.3 s before the backend fault
        rig.pump(1.5)                                                   # holds: no backoff and no fault
        rig.bus.wall = None
        assert not any("gluche" in f for f in rig.faults), rig.faults
        meas, cmd = rig.twin.status.measured, rig.twin.status.command
        assert max(abs(cmd[k] - meas[k]) for k in ARM) < 1.0, (cmd, meas)
        assert goal - meas["arm_joint1"] > 2.0                         # stayed at the wall
        assert rig.twin.safety_state.value != "estop"
        assert rig.twin.owner is None
    finally:
        rig.close()


def grip_pdes(rig: Rig, since: int = 0) -> list[float]:
    """p_des of every 0x051 frame sent since tx index `since`."""
    return [struct.unpack_from(">h", d, 0)[0] / P.POS_SCALE for i, d, _ in rig.bus.tx[since:] if i == P.GRIP_ID]


def test_fist_close_then_open_then_close_drives_0x051_and_done_waits_for_the_slew(rig):
    """Domyslny A1X: grupa 7 liczy sie dopiero przy sterowanym chwytaku, wiec po Polacz chwytak jest
    zalozony zamkniety (`gripper_start` 0): "zamknij" nic nie wysyla, "otworz" idzie od razu na
    `gripper_open_pdes` (zgadniete rozwarcie: bez ramki ku zamknietym), od tej chwili pomiar, "zamknij"
    rampa do `gripper_closed_pdes` (nigdy dalej: +0,6 przegrzalo silnik), a `done` przychodzi dopiero,
    gdy strumien doszedl do celu."""
    a = rig.arm.acfg
    rig.bus.grip_rest_reading = -2.88                                   # resting group 7 after power-up
    d = rig.connect_driver()
    assert not rig.arm.gripper_measured()
    assert rig.twin.status.measured["gripper"] == pytest.approx(0.0)
    # The specs driver of the teleop branch waits `gripper_travel_s()` itself; here the test sets it.
    d.grip_wait_s = rig.arm.gripper_travel_s()
    assert d.grip_wait_s == pytest.approx(abs(a.gripper_open_pdes - a.gripper_closed_pdes) / a.gripper_rate_pdes_s + 0.2)
    assert rig.arm.gripper_drivable()                                   # port "fake", not XCAN

    def run_grip(action: str) -> tuple[dict, int]:
        n = len(rig.bus.tx)
        ts = rig.relay.grip(action)
        done: list[dict] = []

        def finished() -> bool:
            with rig.relay.cond:
                done[:] = [g for g in rig.relay.grips if g["ts"] == ts and g["state"] in ("done", "rejected", "hold")]
            return bool(done)
        assert rig.pump(10.0, finished), rig.relay.grips
        return done[-1], n

    st, n = run_grip("close")
    assert st["state"] == "done", st
    assert grip_pdes(rig, n) == []                                       # already (assumed) closed

    lsb = 2 / P.POS_SCALE
    st, n = run_grip("open")
    assert st["state"] == "done", st
    ps = grip_pdes(rig, n)
    assert ps and ps[-1] == pytest.approx(a.gripper_open_pdes, abs=lsb)  # reached before `done`
    assert ps[0] == pytest.approx(a.gripper_open_pdes, abs=lsb)         # a guess: straight to open, no slam
    assert rig.arm.gripper_measured()                                   # driven: group 7 counts now
    assert all(b <= a_ + lsb for a_, b in zip(ps, ps[1:]))
    assert rig.twin.status.measured["gripper"] == pytest.approx(100.0 * (a.gripper_open_pdes - a.gripper_fb_closed)
                                                                / (a.gripper_fb_open - a.gripper_fb_closed), abs=0.5)

    st, n = run_grip("close")
    assert st["state"] == "done", st
    ps = grip_pdes(rig, n)
    assert ps and ps[-1] == pytest.approx(a.gripper_closed_pdes, abs=lsb)
    assert max(ps) <= a.gripper_closed_pdes + lsb                        # never past the closed p_des
    assert all(b >= a_ - lsb for a_, b in zip(ps, ps[1:]))
    assert max(b - a_ for a_, b in zip(ps, ps[1:])) <= 2.0 * STEP + lsb  # the backend's slew
    assert rig.twin.status.command["gripper"] == pytest.approx(0.0)
    assert rig.twin.safety_state.value != "estop"


# ------------------------------------------------------------------ follow streams (teleop) on the A1X
STREAM = 2_000_000
FOLLOW_PERIOD = 0.05                                            # the glasses send every 50 ms


def stream_states(rig: Rig, stream: int = STREAM) -> list[tuple[str, str | None, str | None]]:
    return [(s["state"], s.get("reason"), s.get("detail")) for s in rig.relay.stream_states(stream)]


def send_stream(rig: Rig, points, seq0: int = 0, stream: int = STREAM, until=None) -> int:
    """Samples every 50 ms, ticking the loop in between; stops early when `until()`. Returns the next seq."""
    seq = seq0
    for p in points:
        rig.relay.follow(stream, seq, p)
        seq += 1
        if rig.pump(FOLLOW_PERIOD, until):
            break
    return seq


def test_follow_stream_moves_the_real_arm_with_bounded_0x050_steps(rig):
    d = rig.connect_driver()
    assert d.follow_v_max == pytest.approx(0.10)                   # the real arm follows slower
    c = rig.vol.center
    n = len(rig.bus.tx)
    line = [c + np.array([0.0, -0.08 + 0.16 * i / 59, -0.02]) for i in range(60)]
    seq = send_stream(rig, line)
    seq = send_stream(rig, [line[-1]] * 20, seq0=seq)
    kinds = stream_states(rig)
    assert kinds[0][0] == "accepted" and all(k[0] in ("accepted", "moving") for k in kinds), kinds
    frames = rig.cmd_frames(n)
    assert len(frames) > 100
    # One 0x050 frame never steps more than one loop tick of the follower's 60 deg/s cap (plus rounding),
    # whatever the backend does between loop ticks.
    step = max(max(abs(a - b) for a, b in zip(f, g)) for f, g in zip(frames, frames[1:]))
    assert step <= 60.0 * DT + 2 * LSB_DEG, step
    tcp = rig.kin.tcp(rig.twin.status.measured)[:3, 3]
    assert np.linalg.norm(tcp - line[-1]) < 0.015
    assert not rig.faults and rig.twin.safety_state.value != "estop"
    rig.relay.follow(STREAM, seq, active=False)
    assert rig.pump(2.0, lambda: rig.twin.owner is None)
    assert stream_states(rig)[-1][:2] == ("hold", "exit")


def test_follow_stream_into_a_wall_ends_in_impact_and_backs_off_without_a_fault(rig):
    d = rig.connect_driver()
    trip: dict = {}
    real_hold_now = d._hold_now

    def spy():
        trip.setdefault("grips", rig.relay.grips_for(g))
        trip.setdefault("tx", len(rig.bus.tx))
        trip.setdefault("q", rig.bus.pdes_deg(next(dd for i, dd, _ in reversed(rig.bus.tx) if i == P.CMD_ID)))
        real_hold_now()
    d._hold_now = spy
    real_backoff = d._follow_backoff

    def backoff_spy(st, hit):
        trip["hit"] = hit
        return real_backoff(st, hit)
    d._follow_backoff = backoff_spy
    c = rig.vol.center
    far = c + np.array([0.0, 0.15, -0.02])                          # J1 turns towards +y
    seq = send_stream(rig, [far] * 6)
    # A gripper command keeps the worker in its settle wait while the arm hits the wall: the hold must not
    # wait for the worker.
    g = rig.relay.grip("close")
    seq = send_stream(rig, [far] * 4, seq0=seq)
    assert rig.relay.grips_for(g) == ["accepted"]
    assert stream_states(rig)[-1][0] == "moving", stream_states(rig)
    # J1 gets stuck 2 deg further along its direction of travel.
    q = rig.bus.q[0]
    sign = 1.0 if rig.bus.p_des[0] >= q else -1.0
    wall = q + sign * np.radians(2.0)
    rig.bus.wall = (0, wall, sign)

    def ended():
        return any(k[0] == "hold" for k in stream_states(rig))
    send_stream(rig, [far] * 60, seq0=seq, until=ended)
    rig.pump(0.3)
    kinds = stream_states(rig)
    assert kinds[-1][:2] == ("hold", "impact"), kinds
    assert "tx" in trip and trip["grips"] == ["accepted"], trip       # the worker was busy at the impact
    tick = int(round(DT / STEP))
    window = rig.cmd_frames(trip["tx"])[: int(round(0.3 / STEP))]
    # From one loop tick after the impact on, the J1 command never goes further towards the wall...
    j1_at = trip["q"][0]
    after = [f[0] for f in window[tick:]]
    assert after and max(sign * (v - j1_at) for v in after) <= 2 * LSB_DEG, (j1_at, after[:10])
    # ... and it stops pushing: from three loop ticks on (the backend slews p_des at 90 deg/s) the command
    # holds J1 where it stands at the wall, while the worker still waits for the gripper.
    near = [f[0] for f in window[3 * tick:]]
    assert near and max(sign * (v - np.degrees(wall)) for v in near) < 0.5, (np.degrees(wall), near[:10])
    # The backoff: the arm ends at the guard's backoff pose, J1 away from the wall, and is released.
    assert rig.pump(8.0, lambda: rig.twin.owner is None)
    rig.bus.wall = None
    pose = trip["hit"].backoff_pose
    measured = rig.twin.status.measured
    assert pose and max(abs(measured[k] - pose[k]) for k in ARM) < 1.0, (measured, pose)
    j1 = np.degrees(rig.bus.q[0])
    assert sign * (np.degrees(wall) - j1) > 1.0, (np.degrees(wall), j1)
    assert not any("gluche" in f for f in rig.faults), rig.faults
    rig.pump(1.0)
    assert not any("gluche" in f for f in rig.faults), rig.faults
    assert rig.twin.safety_state.value != "estop"
    assert d._stream is None and d._follow_handle is None


def test_follow_stream_with_the_gripper_loaded_ends_in_impact_and_stops_advancing(rig):
    """2026-09-27: the fingertips pressed into the table during a stream and the gripper motor overheated,
    with no gripper command sent. The gripper's effort above its rest + `gripper_force` ends the stream."""
    d = rig.connect_driver()
    assert d.follow_floor_m == pytest.approx(0.01)                  # the floor stays at 1 cm on the real arm
    c = rig.vol.center
    far = c + np.array([0.0, 0.15, -0.02])
    trip: dict = {}
    real_hold_now = d._hold_now

    def spy():
        cmd = rig.twin.status.command
        trip.setdefault("dist", float(np.linalg.norm(rig.kin.tcp(cmd)[:3, 3] - d.volume.clamp(far))))
        real_hold_now()
    d._hold_now = spy
    seq = send_stream(rig, [far] * 6)
    st = d._stream
    assert st is not None and st.grip_rest == pytest.approx(0.0, abs=0.01)
    floor = st.follower.setpoint(np.array([c[0], c[1], rig.vol.lo[2]]))
    assert floor[2] == pytest.approx(rig.vol.lo[2] + 0.01)
    assert stream_states(rig)[-1][0] in ("accepted", "moving"), stream_states(rig)
    rig.bus.grip_effort = 3.0                                       # the jaws press on the table

    def ended():
        return any(k[0] == "hold" for k in stream_states(rig))
    send_stream(rig, [far] * 40, seq0=seq, until=ended)
    kinds = stream_states(rig)
    assert kinds[-1] == ("hold", "impact", "gripper load"), kinds
    assert "dist" in trip
    # The command never gets closer to the stream's point after the hold (the backoff goes back).
    closest = np.inf

    def track():
        nonlocal closest
        cmd = rig.twin.status.command
        closest = min(closest, float(np.linalg.norm(rig.kin.tcp(cmd)[:3, 3] - d.volume.clamp(far))))
        return rig.twin.owner is None
    rig.bus.grip_effort = 0.0
    assert rig.pump(8.0, track)
    assert closest > trip["dist"] - 0.005, (closest, trip["dist"])
    assert rig.twin.safety_state.value != "estop"
    assert d._stream is None and d._follow_handle is None
    # No gripper frame went out.
    assert not any(i == P.GRIP_ID for i, _, _ in rig.bus.tx)
