"""Follow streams from the glasses (teleop): `SpecsReach` + `twin/follow.py` on the simulated A1X.

The twin runs with `threaded=False` and the test ticks its loop (`Bench.pump`), so the follower (a tick
listener, all times from the loop's clock) steps deterministically; the relay is the fake from
`test_twin_specsreach.py`. The driver's network and worker threads are real, so samples are sent
between pumps at wall-clock pace (20 Hz, as the glasses do).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("websockets")

from test_twin_specsreach import FakeRelay  # noqa: E402

from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin.follow import Follower, Tracker, down_rotation, nearest_yaw  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics  # noqa: E402
from lerobot_mp.twin.reach import ReachVolume  # noqa: E402
from lerobot_mp.twin.specsreach import SpecsReach, parse_follow  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

DT = 0.02
PERIOD = 0.05                                          # the glasses' follow period (20 Hz)
STREAM = 1_000_000


class Bench:
    """Twin (threaded=False) with the arm parked above the box centre, a driver and the fake relay."""

    def __init__(self, relay: FakeRelay, **kw):
        self.relay = relay
        self.twin = rt.Twin(Workspace(robot="a1x"))
        self.twin.connect("sim", threaded=False)
        self.spec = self.twin.spec
        self.arm = tuple(j for j in self.spec.joints if j != self.spec.gripper)
        self.kin = RobotKinematics(self.spec)
        self.vol = ReachVolume.from_workspace(self.twin.workspace)
        self.pump(0.5)
        sol = self.kin.ik(self.vol.center, seed=self.spec.home, tol=0.003)
        assert sol.ok
        self.twin.claim("test")
        self.twin.set_engaged(True, owner="test")
        self.twin.set_target({**{k: sol.joints[k] for k in self.arm}, "gripper": 100.0}, owner="test")
        self.pump(2.0)
        self.twin.release("test")
        # What the follower asks for (arm joints, per tick), its tracker's point and velocity then, and
        # what the loop sent (per tick).
        self.asked: list[dict] = []
        self.points: list[tuple[np.ndarray, np.ndarray]] = []
        self.sent: list[dict] = []
        real_set_target = self.twin.set_target

        def spy(joints, owner=None):
            real_set_target(joints, owner=owner)
            st = self.driver._stream
            follower = st.follower if st is not None else None
            if owner == "specs" and set(joints) == set(self.arm) and follower is not None:
                self.asked.append(dict(joints))
                self.points.append((follower.point, follower.tracker.v.copy()))
        self.twin.set_target = spy
        self.twin.add_tick_listener(lambda s: self.sent.append(dict(s.sent)))
        kw.setdefault("mode", "sim")
        self.driver = SpecsReach(self.twin, relay_url=relay.url, **kw)
        self.driver.start()
        relay.wait_client()
        self.pump(0.3)

    def pump(self, secs: float, until=None) -> bool:
        """Tick the loop for `secs` of wall time or until `until()`; True if `until` was met."""
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < secs:
            if until is not None and until():
                return True
            self.twin.step(DT)
            n += 1
            ahead = t0 + n * DT - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)
        return until is not None and bool(until())

    def tcp(self) -> np.ndarray:
        return self.kin.tcp(self.twin.status.measured)[:3, 3]

    def near(self, p, tol: float = 0.005) -> bool:
        """The follower's last command puts the TCP within `tol` of `p`, and the measured TCP within 15 mm
        (a servo under load sags: in sim the TCP ends up to ~1 cm low; the follower doesn't trim)."""
        asked = self.kin.tcp(self.asked[-1])[:3, 3]
        return bool(np.linalg.norm(asked - p) < tol and np.linalg.norm(self.tcp() - p) < 0.015)

    def states(self, stream: int = STREAM) -> list[tuple[str, str | None]]:
        return [(s["state"], s.get("reason")) for s in self.relay.stream_states(stream)]

    def details(self, stream: int = STREAM) -> list[tuple[str, str | None]]:
        return [(s["state"], s.get("detail")) for s in self.relay.stream_states(stream)]

    def state(self, stream: int, states: set[str], reason: str | None = None) -> dict | None:
        hit = [s for s in self.relay.stream_states(stream)
               if s["state"] in states and (reason is None or s.get("reason") == reason)]
        return hit[-1] if hit else None

    def wait(self, stream: int, states: set[str], secs: float, reason: str | None = None) -> dict:
        assert self.pump(secs, lambda: self.state(stream, states, reason) is not None), self.states(stream)
        st = self.state(stream, states, reason)
        assert st is not None
        return st

    def stream(self, points, stream: int = STREAM, seq0: int = 0, yaw: float | None = None,
               check=None) -> int:
        """Sends `points` as samples every 50 ms, pumping in between; returns the next seq."""
        seq = seq0
        for p in points:
            self.relay.follow(stream, seq, p, yaw=yaw)
            seq += 1
            self.pump(PERIOD)
            if check is not None:
                check(p)
        return seq

    def close(self) -> None:
        self.driver.stop()
        self.twin.close()


@pytest.fixture
def relay():
    r = FakeRelay()
    yield r
    r.close()


@pytest.fixture
def bench(relay):
    b = Bench(relay)
    yield b
    b.close()


def largest_step(poses: list[dict], joints) -> float:
    return max((max(abs(b[k] - a[k]) for k in joints) for a, b in zip(poses, poses[1:])), default=0.0)


# The joint rate the follower may ask for: 0.2 m/s at ~0.25 m from the shoulder is ~50 deg/s, and the tool's
# rotation (1 rad/s, only while it turns towards pointing down) adds up to ~60 deg/s on the wrist.
ASK_DEG_PER_TICK = 2.6


def test_parse_follow_contract():
    s = parse_follow({"type": "follow", "ts": 5, "session": "s", "trusted": True, "stream": 3, "seq": 2,
                      "active": True, "pos": [0.3, 0.0, 0.1], "yaw": None})
    assert s is not None and s.stream == 3 and s.seq == 2 and s.active and s.yaw is None and s.trusted
    stop = parse_follow({"type": "follow", "stream": 3, "seq": 3, "active": False})
    assert stop is not None and not stop.active and stop.pos is None
    for bad in ({"stream": 3, "seq": 1, "active": True},                 # no pos
                {"stream": 3, "seq": 1.5, "active": True, "pos": [0, 0, 0]},
                {"stream": None, "seq": 1, "active": False},
                {"stream": 3, "seq": 1, "active": "yes"}):
        assert parse_follow(bad) is None


def test_tracker_bounds_speed_and_acceleration_and_stops_on_the_target():
    tr = Tracker(np.zeros(3), v_max=0.2)
    target = np.array([0.3, 0.0, 0.0])
    ps, vs = [tr.p.copy()], [0.0]
    for _ in range(200):
        p, v = tr.propose(target, DT)
        tr.commit(p, v)
        ps.append(p.copy())
        vs.append(float(np.linalg.norm(v)))
    speeds = [float(np.linalg.norm(b - a)) / DT for a, b in zip(ps, ps[1:])]
    assert max(speeds) <= 0.2 + 1e-9
    assert max(abs(b - a) for a, b in zip(vs, vs[1:])) <= 4 * 0.2 * DT + 1e-9
    assert np.allclose(tr.p, target, atol=1e-6) and vs[-1] < 1e-6


def test_nearest_yaw_takes_the_symmetric_grasp_nearer_the_jaws():
    assert nearest_yaw(0.1, 0.0) == pytest.approx(0.1)
    assert nearest_yaw(np.pi - 0.1, 0.0) == pytest.approx(-0.1)          # the same grasp turned by pi
    assert nearest_yaw(-3.0, 3.0) == pytest.approx(3.0 + (2 * np.pi - 6.0))       # across the wrap


def test_follower_holds_on_collision_without_moving_its_point():
    b_spec = rt.Twin(Workspace(robot="a1x")).spec
    kin = RobotKinematics(b_spec)
    vol = ReachVolume.from_spec(b_spec)
    sol = kin.ik(vol.center, rotation=down_rotation(b_spec, vol.center), seed=b_spec.home, tol=0.003)

    class Wall:
        def config_clear(self, joints):
            return "kolizja: sciana" if kin.tcp(joints)[1, 3] > 0.02 else None

    f = Follower(kin, Wall(), b_spec, limits=b_spec.limits, joints=sol.joints, volume=vol)
    target = f.setpoint(vol.center + [0.0, 0.1, 0.0])
    outs = [f.step(DT, target, None) for _ in range(100)]
    assert any(o.joints is not None for o in outs) and outs[-1].fail == "collision"
    assert f.point[1] <= 0.021
    back = f.step(DT, f.setpoint(vol.center), None)
    assert back.joints is not None


def test_a_line_at_20_hz_is_followed_within_3_cm_with_bounded_joint_steps(bench):
    c = bench.vol.center
    line = [c + np.array([0.0, -0.1 + 0.2 * i / 59, -0.02]) for i in range(60)]
    errors: list[float] = []
    n = {"i": 0}

    def check(p):
        n["i"] += 1
        if n["i"] > 10:                                 # 0.5 s to catch up with the first point
            errors.append(float(np.linalg.norm(bench.tcp() - p)))
    seq = bench.stream(line, check=check)
    seq = bench.stream([line[-1]] * 20, seq0=seq)       # 1 s of the same point
    assert max(errors) < 0.03, max(errors)
    assert bench.near(line[-1])
    kinds = bench.states()
    assert kinds[0] == ("accepted", None) and ("moving", None) in kinds
    assert all(k[0] in ("accepted", "moving") for k in kinds), kinds
    moving = [s for s in bench.relay.stream_states(STREAM) if s["state"] == "moving"]
    assert len(moving) <= 2 * 4 + 1                     # at most 2 Hz over ~4 s
    assert bench.asked and largest_step(bench.asked, bench.arm) <= ASK_DEG_PER_TICK
    sent = [s for s in bench.sent if s]
    assert largest_step(sent[-200:], bench.arm) <= 60.0 * DT + 0.05     # the follower's speed cap
    assert bench.twin.owner == "specs"
    st = bench.relay.stream_states(STREAM)[-1]
    assert list(st)[0] == "type" and st["stream"] == STREAM and st["mode"] == "sim" and "tcp" in st


def test_one_second_of_identical_samples_stays_moving(bench):
    p = bench.vol.center + np.array([0.03, 0.0, 0.0])
    bench.stream([p] * 40)                                # 2 s, the glasses send every 50 ms changed or not
    assert all(k[0] in ("accepted", "moving") for k in bench.states()), bench.states()
    assert sum(k[0] == "moving" for k in bench.states()) >= 2          # the 1 s heartbeat


def test_far_entry_point_is_a_ramp(bench):
    c = bench.vol.center
    far = c + np.array([0.1, 0.25, -0.08])
    v_max, a_max = 0.2, 0.8
    bench.stream([far] * 50)
    assert bench.near(far)
    # The tracker's commanded point: speed within v_max, acceleration within a_max, from standstill.
    tracked = [p for p, _ in bench.points]
    vels = [v for _, v in bench.points]
    speeds = [float(np.linalg.norm(b - a)) / DT for a, b in zip(tracked, tracked[1:])]
    assert max(speeds) <= v_max + 1e-9, max(speeds)
    assert float(np.linalg.norm(vels[0])) <= a_max * DT + 1e-9          # the first tick: one step of a_max
    accel = [float(np.linalg.norm(b - a)) / DT for a, b in zip(vels, vels[1:])]
    assert max(accel) <= a_max + 1e-6, max(accel)
    # The FK of what IK made of it: each point is within ik_tol (3 mm) of the tracker's.
    pts = [bench.kin.tcp(q)[:3, 3] for q in bench.asked]
    fk = [float(np.linalg.norm(b - a)) / DT for a, b in zip(pts, pts[1:])]
    assert max(fk) <= v_max + 0.003 / DT, max(fk)
    assert largest_step(bench.asked, bench.arm) <= ASK_DEG_PER_TICK


def test_active_false_holds_with_exit_and_the_arm_stays(bench):
    p = bench.vol.center + np.array([0.02, 0.05, 0.0])
    seq = bench.stream([p] * 20)
    bench.relay.follow(STREAM, seq, active=False)
    st = bench.wait(STREAM, {"hold"}, 2.0)
    assert st["reason"] == "exit" and st["ts"] == STREAM + seq
    assert bench.pump(1.0, lambda: bench.twin.owner is None)
    a = dict(bench.twin.status.measured)
    bench.pump(0.5)
    b = bench.twin.status.measured
    assert max(abs(a[k] - b[k]) for k in bench.arm) < 0.3
    # Later samples of the ended stream are ignored; the glasses start a new stream to move again.
    bench.stream([p + [0.05, 0.0, 0.0]] * 5, seq0=seq + 1)
    assert bench.states()[-1] == ("hold", "exit")


def test_a_sample_10_cm_outside_ends_the_stream(bench):
    c = bench.vol.center
    seq = bench.stream([c] * 10)
    bench.relay.follow(STREAM, seq, bench.vol.hi + [0.1, 0.0, 0.0])
    st = bench.wait(STREAM, {"hold"}, 2.0)
    assert st["reason"] == "outside volume"


def test_start_outside_the_box_is_rejected_and_the_stream_stays_refused(bench):
    out = bench.vol.hi + np.array([0.03, 0.0, 0.0])
    bench.stream([out] * 3)
    assert bench.states() == [("rejected", "outside volume")]
    bench.stream([bench.vol.center] * 5, seq0=3)         # same stream: still refused
    assert bench.states() == [("rejected", "outside volume")]
    bench.stream([bench.vol.center] * 5, stream=STREAM + 7)
    assert bench.states(STREAM + 7)[0] == ("accepted", None)


def test_a_wall_in_the_way_pauses_the_stream_then_it_resumes(bench):
    d = bench.driver
    kin, checker = d._follow_tools()                    # the follower's own checker: add a wall at y > c.y + 4 cm
    wall_y = float(bench.vol.center[1]) + 0.04
    clear = checker.config_clear
    checker.config_clear = lambda j: "kolizja: sciana" if kin.tcp(j)[1, 3] > wall_y else clear(j)
    c = bench.vol.center
    seq = bench.stream([c] * 10)
    seq = bench.stream([c + [0.0, 0.1, 0.0]] * 20, seq0=seq)           # 1 s into the wall
    # A pause is not a hold (a hold ends the stream: the glasses stop sending).
    assert ("moving", "paused: collision") in bench.details(), bench.details()
    assert not any(k[0] == "hold" for k in bench.states()), bench.states()
    assert bench.tcp()[1] < wall_y + 0.005
    bench.stream([c + [0.0, -0.05, 0.0]] * 20, seq0=seq)               # back out: moving again
    details = bench.details()
    assert details[-1][0] == "moving" and not (details[-1][1] or "").startswith("paused"), details
    assert not any(k[0] == "hold" for k in bench.states()), bench.states()
    assert bench.near((c + [0.0, -0.05, 0.0]))


def test_two_seconds_in_a_wall_end_the_stream_with_collision(bench):
    d = bench.driver
    kin, checker = d._follow_tools()
    wall_y = float(bench.vol.center[1]) + 0.04
    clear = checker.config_clear
    checker.config_clear = lambda j: "kolizja: sciana" if kin.tcp(j)[1, 3] > wall_y else clear(j)
    c = bench.vol.center
    seq = bench.stream([c] * 10)
    bench.stream([c + [0.0, 0.1, 0.0]] * 70, seq0=seq)                 # 3.5 s pressing into the wall
    st = bench.state(STREAM, {"hold"})
    assert st is not None and st["reason"] == "collision", bench.details()
    kinds = bench.details()
    assert ("moving", "paused: collision") in kinds and kinds.index(("moving", "paused: collision")) < \
        kinds.index(("hold", st.get("detail")))


def test_gripper_mid_stream_closes_and_the_stream_keeps_the_arm(bench):
    c = bench.vol.center
    assert bench.twin.status.command["gripper"] > 95.0
    seq = bench.stream([c] * 10)
    g = bench.relay.grip("close")
    t0 = time.monotonic()
    line = [c + np.array([0.0, 0.004 * i, 0.0]) for i in range(30)]
    closed_at: list[float] = []

    def check(_):
        if not closed_at and bench.twin.status.command["gripper"] < 1.0:
            closed_at.append(time.monotonic() - t0)
        assert bench.twin.owner == "specs"
    seq = bench.stream(line, seq0=seq, check=check)
    assert closed_at and closed_at[0] < 0.6, closed_at
    assert bench.relay.grips_for(g) == ["accepted", "done"]
    assert bench.twin.status.command["gripper"] < 1.0
    assert bench.states()[-1] == ("moving", None) and not any(k[0] == "hold" for k in bench.states())
    assert np.linalg.norm(bench.tcp() - line[-1]) < 0.02            # the arm kept following meanwhile
    bench.relay.follow(STREAM, seq, active=False)
    bench.wait(STREAM, {"hold"}, 2.0, reason="exit")
    assert bench.pump(1.0, lambda: bench.twin.owner is None)
    assert bench.twin.status.command["gripper"] < 1.0


def test_gripper_during_a_stream_that_preempted_a_target_is_done(bench):
    """A stream interrupting a target leaves the driver's `_cancel` set; a gripper command then must not
    see it as its own interrupt (hold) while the jaws do move."""
    c = bench.vol.center
    ts = bench.relay.target(c + np.array([0.0, 0.08, -0.04]))
    assert bench.pump(3.0, lambda: "moving" in bench.relay.states_for(ts)), bench.relay.states_for(ts)
    seq = bench.stream([c] * 10)
    assert bench.relay.states_for(ts)[-1] == "hold"
    assert bench.states()[0] == ("accepted", None)
    g = bench.relay.grip("close")
    seq = bench.stream([c] * 40, seq0=seq)
    assert bench.relay.grips_for(g) == ["accepted", "done"], bench.relay.grips_for(g)
    assert bench.twin.status.command["gripper"] < 1.0
    assert bench.states()[-1] == ("moving", None) and not any(k[0] == "hold" for k in bench.states())
    assert bench.twin.owner == "specs"


def test_a_grab_during_a_stream_ends_it_and_runs(bench):
    c = bench.vol.center
    bench.stream([c] * 10)
    ts = bench.relay.grab_object([0.30, -0.05, 0.015], size=[0.03, 0.03])
    st = bench.wait(STREAM, {"hold"}, 2.0)
    assert st["reason"] == "retargeted"
    done = {"held", "missed", "rejected", "hold", "backoff"}
    assert bench.pump(30.0, lambda: bool(set(bench.relay.states_for(ts)) & done)), bench.relay.states_for(ts)
    states = bench.relay.states_for(ts)
    assert states[0] == "accepted" and states[-1] in ("held", "missed"), \
        [(s["state"], s.get("reason"), s.get("detail")) for s in bench.relay.stream_states(ts)]


def test_a_target_during_a_stream_ends_it_and_runs(bench):
    c = bench.vol.center
    bench.stream([c] * 10)
    p = c + np.array([0.04, -0.06, -0.03])
    ts = bench.relay.target(p)
    st = bench.wait(STREAM, {"hold"}, 2.0)
    assert st["reason"] == "retargeted"
    assert bench.pump(15.0, lambda: bench.relay.states_for(ts)[-1:] in (["reached"], ["rejected"], ["hold"]))
    assert bench.relay.states_for(ts)[-1] == "reached", [(s["state"], s.get("reason"), s.get("detail")) for s in bench.relay.states if s["ts"] == ts]
    assert np.linalg.norm(bench.tcp() - p) < 0.005


def test_silence_ends_the_stream_with_timeout_and_releases_the_arm_later(bench):
    c = bench.vol.center
    bench.stream([c] * 10)
    t_last = time.monotonic()
    st = bench.wait(STREAM, {"hold"}, 2.0)
    waited = time.monotonic() - t_last
    assert st["reason"] == "timeout" and 0.5 < waited < 1.2, waited
    assert bench.twin.owner == "specs"                  # released 2 s after the last sample
    assert bench.pump(3.0, lambda: bench.twin.owner is None)
    assert 1.5 < time.monotonic() - t_last < 3.0


def test_a_reordered_sample_is_ignored(bench):
    c = bench.vol.center
    seq = bench.stream([c] * 10)
    bench.relay.follow(STREAM, seq + 5, c + [0.0, 0.05, 0.0])
    bench.pump(0.1)
    bench.relay.follow(STREAM, seq + 2, c + [0.0, -0.1, 0.0])           # older: dropped
    bench.pump(0.1)
    bench.stream([c + [0.0, 0.05, 0.0]] * 20, seq0=seq + 6)
    assert bench.driver._stream is not None and bench.driver._stream.sample.seq == seq + 25
    assert bench.near((c + [0.0, 0.05, 0.0]))
    ys = [bench.kin.tcp(q)[1, 3] for q in bench.asked]
    assert min(ys) > float(c[1]) - 0.01                 # never went towards the dropped sample


def test_a_new_stream_id_replaces_the_running_stream(bench):
    c = bench.vol.center
    bench.stream([c] * 10)
    other = STREAM + 500
    bench.stream([c + [0.0, 0.05, 0.0]] * 20, stream=other)
    assert bench.states()[-1] == ("hold", "retargeted")
    kinds = bench.states(other)
    assert kinds[0] == ("accepted", None) and kinds[-1] == ("moving", None), kinds
    assert bench.near((c + [0.0, 0.05, 0.0]))


def test_dry_mode_accepts_then_holds_and_nothing_moves(relay):
    b = Bench(relay, mode="dry")
    try:
        before = dict(b.twin.status.measured)
        b.stream([b.vol.center + [0.05, 0.05, 0.0]] * 10)
        assert b.states() == [("accepted", None), ("hold", None)]
        after = b.twin.status.measured
        assert max(abs(after[k] - before[k]) for k in b.arm) < 0.3
        assert b.asked == [] and b.twin.owner is None
    finally:
        b.close()


def test_mode_change_ends_the_stream(bench):
    bench.stream([bench.vol.center] * 10)
    bench.driver.set_mode("dry", False)
    st = bench.wait(STREAM, {"hold"}, 2.0)
    assert st["reason"] == "mode"
    assert bench.pump(1.0, lambda: bench.twin.owner is None)
