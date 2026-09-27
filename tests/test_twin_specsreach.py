"""Specs -> relay -> `SpecsReach` driver -> A1X arm in simulation (no hardware).

The twin runs without a loop thread on a `VirtualClock` that the driver shares, and the tests step it
(`twinbench.Bench.pump`): a simulated second costs a few milliseconds and the timings repeat to the tick.
The relay is an in-process fake (`twinbench.FakeRelay`, a `websockets.sync.server` on a free port) that
sends a trusted registration for session "s1" on connect; the websocket and the driver's network thread
are real. The CLI tests run the real CLI in a subprocess (marker `cli`; skip them with `-m "not cli"`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("websockets")

from twinbench import END, Bench, FakeRelay  # noqa: E402,F401 - FakeRelay: other test files import it from here

from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin.clock import VirtualClock  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics  # noqa: E402
from lerobot_mp.twin.reach import GRAB_ZONES, ReachVolume  # noqa: E402
from lerobot_mp.twin.scene import Box  # noqa: E402
from lerobot_mp.twin.specsreach import SpecsReach  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="module")
def clock():
    """One time axis for the whole module: it only grows, and stands still between tests."""
    c = VirtualClock()
    yield c
    c.close()
    assert c.late == 0, f"VirtualClock: {c.late} barrier(s) timed out in this module (see the warning in the log)"


@pytest.fixture(scope="module")
def twin(clock):
    tw = rt.Twin(Workspace(robot="a1x"), clock=clock)
    tw.connect("sim", threaded=False)
    yield tw
    tw.close()


@pytest.fixture(scope="module")
def vol(twin):
    return ReachVolume.from_workspace(twin.workspace)


@pytest.fixture(scope="module")
def kin(twin):
    return RobotKinematics(twin.spec)


@pytest.fixture
def relay():
    r = FakeRelay()
    yield r
    r.close()


@pytest.fixture
def bench(twin, relay):
    b = Bench(twin, relay)
    yield b
    b.close()                                            # stops the drivers the test started


def ready(bench: Bench, kin, vol) -> None:
    """Park the arm above the volume centre (no move if it is already there)."""
    twin = bench.twin
    if twin.safety_state is not None and twin.safety_state.value == "estop":
        twin.clear_estop()
    twin.configure(objects=None)
    sol = kin.ik(vol.center, seed=twin.spec.home, tol=0.003)
    assert sol.ok
    bench.ramp_to({k: v for k, v in sol.joints.items() if k != twin.spec.gripper})


def tcp(kin, twin) -> np.ndarray:
    return kin.tcp(twin.status.measured or twin.joints())[:3, 3]


def test_hand_target_inside_volume_is_reached(bench, kin, vol, relay):
    twin = bench.twin
    ready(bench, kin, vol)
    d = bench.driver()
    first = bench.overlays()[0]
    kinds = {o["id"]: o["kind"] for o in first["objects"]}
    assert kinds == {"reach": "volume", "grab": "volume", "tcp": "sphere"}
    vol_obj = next(o for o in first["objects"] if o["id"] == "reach")
    assert vol_obj["alpha"] == pytest.approx(0.35)
    # The grab zone, drawn faintly by the glasses: marks inside it can be grabbed.
    grab = next(o for o in first["objects"] if o["id"] == "grab")
    lo, hi = (np.array(v) for v in GRAB_ZONES["a1x"])
    assert grab["label"] == "grab" and grab["alpha"] == pytest.approx(0.12)
    assert grab["pos"] == pytest.approx((lo + hi) / 2, abs=1e-4)
    assert grab["size"] == pytest.approx(hi - lo, abs=1e-4)
    assert len(grab["color"]) == 3
    p = vol.center + np.array([0.05, 0.08, -0.05])
    ts = bench.target(p)
    bench.wait_state(ts, {"moving", "rejected"}, max_sim_s=5.0)
    # While moving, the supervisor caps speed at 1.5 x ramp speed + 5: 20 deg/s -> 35.
    assert twin._supervisor.speed_cap == pytest.approx(35.0)
    st = bench.wait_state(ts, END, max_sim_s=10.0)
    assert st["state"] == "reached", st
    assert bench.states_for(ts)[:2] == ["accepted", "moving"]
    assert st["mode"] == "sim" and list(st)[0] == "type"
    assert np.linalg.norm(tcp(kin, twin) - p) < 0.005
    assert twin.owner is None                              # released; the clutch holds the target
    assert bench.pump(until=lambda: twin._speed_cap is None, max_sim_s=2.0)
    assert twin._supervisor.speed_cap is None
    # The TCP sphere keeps coming every 0.2 s of the driver's clock, and the wire carried it all.
    for _ in range(2):
        bench.tcp_overlay()
    bench.wait_real(lambda: relay.saw(ts, {"reached"}), what="reached u przekaznika")
    Bench.wait_real(lambda: sum(o["id"] == "tcp" for m in relay.overlays for o in m["objects"]) >= 3)
    ids = [o["id"] for m in relay.overlays for o in m["objects"]]
    assert "target" in ids and relay.overlays[0]["objects"][0]["kind"] == "volume"
    assert d.status()["phase"] == "idle"


@pytest.mark.parametrize("case", ["outside", "session", "no depth", "stale", "untrusted"])
def test_rejections(bench, kin, vol, relay, case):
    twin = bench.twin
    d = bench.driver()
    before = dict(twin.status.measured)
    if case == "outside":
        ts, why = bench.target(vol.hi + [0.05, 0.0, 0.0]), "outside volume"
    elif case == "session":
        ts, why = bench.target(vol.center, session="stara"), "session"
    elif case == "no depth":
        ts, why = bench.target(vol.center, source="object", label="cup"), "no depth"
    elif case == "stale":
        # Detection last actually seen 0.8 s ago (the glasses lost the object): don't move.
        ts = bench.target(vol.center, source="object", label="cup", depthMeasured=True, age=0.8)
        why = "stale"
    else:
        relay.trusted = False
        bench.driver()
        d.stop()
        ts, why = bench.target(vol.center), "untrusted"
    st = bench.wait_state(ts, {"rejected", "reached"}, max_sim_s=5.0)
    assert (st["state"], st.get("reason")) == ("rejected", why)
    if case == "stale":
        assert "0.80 s" in st["detail"], st
    after = twin.status.measured
    assert max(abs(after[k] - before[k]) for k in d.arm) < 0.5


def test_object_target_goes_above_it_pointing_down(bench, kin, vol):
    twin = bench.twin
    ready(bench, kin, vol)
    bench.driver()
    p = vol.center + np.array([0.0, -0.05, -0.08])
    ts = bench.target(p, source="object", label="cup", depthMeasured=True, age=0.2)
    st = bench.wait_state(ts, END, max_sim_s=12.0)
    assert st["state"] == "reached", st
    T = kin.tcp(twin.status.measured)
    assert np.linalg.norm(T[:3, 3] - (p + [0.0, 0.0, 0.08])) < 0.005
    approach = T[:3, :3] @ np.asarray(twin.spec.tcp_approach)
    assert approach[2] < -0.95                          # gripper points down


def test_object_grasp_descends_closes_and_lifts(bench, kin, vol):
    twin = bench.twin
    ready(bench, kin, vol)
    bench.driver(grasp=True)
    p = vol.center + np.array([0.0, 0.05, -0.1])
    ts = bench.target(p, source="object", label="cup", depthMeasured=True)
    st = bench.wait_state(ts, END, max_sim_s=25.0)
    assert st["state"] == "reached", (st, bench.states_for(ts))
    assert bench.states_for(ts).count("moving") == 3            # above, down, up
    # The down/up legs aren't corrected for sag under load, so it remains (up to ~8 mm in sim).
    assert np.linalg.norm(tcp(kin, twin) - (p + [0.0, 0.0, 0.08])) < 0.008
    assert twin.status.command["gripper"] < 5.0


def test_fist_closes_then_opens_the_gripper(bench, kin, vol, relay):
    twin = bench.twin
    ready(bench, kin, vol)
    bench.driver()
    ts = bench.grip("close")
    st = bench.wait_grip(ts, max_sim_s=10.0)
    assert st["state"] == "done" and st["action"] == "close" and st["mode"] == "sim", st
    assert bench.grips_for(ts) == ["accepted", "done"]
    assert twin.status.command["gripper"] < 5.0
    assert twin.owner != "specs"                                  # released, still holding
    assert bench.tcp_overlay().get("label") == "closed"            # the TCP sphere says so

    ts = bench.grip("open")
    st = bench.wait_grip(ts, max_sim_s=10.0)
    assert st["state"] == "done", st
    assert twin.status.command["gripper"] > 95.0
    # A malformed action is dropped without a reply.
    bench.grip("squeeze")
    bench.pump(sim_s=0.5)
    assert len(bench._sent("gripperState")) == 4
    Bench.wait_real(lambda: len(relay.grips) == 4, what="4 stany chwytaka u przekaznika")


def test_fist_during_a_move_closes_the_gripper_once_the_arm_is_there(bench, kin, vol):
    twin = bench.twin
    ready(bench, kin, vol)
    twin.set_target({"gripper": 100.0}, owner=None)
    bench.driver()
    ts = bench.target(vol.center + np.array([0.05, 0.05, 0.05]))
    bench.wait_state(ts, {"moving"}, max_sim_s=10.0)
    g = bench.grip("close")
    st = bench.wait_grip(g, max_sim_s=25.0)
    assert st["state"] == "done", (st, bench.grips_for(g))
    assert bench.states_for(ts)[-1] == "reached", bench.states_for(ts)
    kinds = [(e["type"], e["state"]) for e in bench.events()]
    # The target finished before the gripper command was even looked at.
    assert kinds.index(("targetState", "reached")) < kinds.index(("gripperState", "accepted"))
    assert twin.status.command["gripper"] < 5.0


@pytest.mark.parametrize("mode,real_ok,want", [("dry", False, "hold"), ("real", False, "rejected")])
def test_gripper_in_dry_mode_only_checks_and_real_needs_consent(bench, kin, vol, mode, real_ok, want):
    twin = bench.twin
    ready(bench, kin, vol)
    twin.set_target({"gripper": 100.0}, owner=None)
    assert bench.pump(until=lambda: twin.status.command["gripper"] > 99.0, max_sim_s=2.0)   # open before we start
    bench.driver(mode=mode, real_ok=real_ok)
    ts = bench.grip("close")
    st = bench.wait_grip(ts, max_sim_s=10.0)
    assert st["state"] == want, st
    if want == "rejected":
        assert st["reason"] == "not armed"
    else:
        assert bench.grips_for(ts) == ["accepted", "hold"]
    bench.pump(sim_s=0.3)
    assert twin.status.command["gripper"] > 95.0                   # nothing moved


def test_dry_mode_accepts_then_holds_and_does_not_move(bench, kin, vol):
    twin = bench.twin
    ready(bench, kin, vol)
    d = bench.driver(mode="dry")
    before = dict(twin.status.measured)
    ts = bench.target(vol.center + [0.08, 0.1, -0.05])
    bench.wait_state(ts, {"hold", "rejected"}, max_sim_s=5.0)
    bench.pump(sim_s=0.5)
    assert bench.states_for(ts) == ["accepted", "hold"]
    assert bench._sent("targetState")[-1]["mode"] == "dry"
    after = twin.status.measured
    assert max(abs(after[k] - before[k]) for k in d.arm) < 0.5
    assert any(o["id"] == "target" for m in bench.overlays() for o in m["objects"])   # planned target still shown


def test_new_target_mid_move_retargets(bench, kin, vol):
    twin = bench.twin
    ready(bench, kin, vol)
    bench.driver(speed_deg_s=10.0)
    p1 = vol.center + np.array([0.0, 0.2, -0.05])
    p2 = vol.center + np.array([0.08, -0.15, -0.03])
    ts1 = bench.target(p1)
    bench.wait_state(ts1, {"moving", "rejected"}, max_sim_s=5.0)
    bench.pump(sim_s=0.6)
    ts2 = bench.target(p2)
    st = bench.wait_state(ts2, END, max_sim_s=15.0)
    assert st["state"] == "reached", st
    assert "reached" not in bench.states_for(ts1)
    old = [x for x in bench._sent("targetState") if x["ts"] == ts1 and x["state"] == "hold"]
    assert old and old[-1]["reason"] == "retargeted"
    assert np.linalg.norm(tcp(kin, twin) - p2) < 0.005


def test_trust_holds_for_the_session_and_a_new_session_starts_without_it(bench, kin, vol):
    d = bench.driver()
    assert d.status()["session"] == "s1"
    # A wobbly update in the same session (the wearer glanced at a tag from afar) keeps the trust.
    d._on_message(json.dumps({"type": "registration", "session": "s1", "trusted": False}))
    assert d.status()["session"] == "s1"
    # The Lens restarted: nothing is trusted until its new session registers.
    d._on_message(json.dumps({"type": "registration", "session": "s2", "trusted": False}))
    assert d.status()["session"] is None
    ts = bench.target(vol.center)
    st = bench.wait_state(ts, {"rejected", "reached"}, max_sim_s=5.0)
    assert (st["state"], st.get("reason")) == ("rejected", "untrusted")


def test_start_center_goes_above_the_volume_centre_in_stages(bench, kin, vol, relay):
    twin = bench.twin
    if twin.safety_state is not None and twin.safety_state.value == "estop":
        twin.clear_estop()
    twin.configure(objects=None)
    bench.ramp_to({k: v for k, v in twin.spec.home.items() if k != twin.spec.gripper})
    d = bench.attach(SpecsReach(twin, relay_url=relay.url, mode="sim"))
    assert bench.call(lambda: d.start_center(30.0)) is None
    assert np.linalg.norm(tcp(kin, twin) - vol.center) < 0.01
    assert twin.owner is None and twin._speed_cap is None


# ------------------------------------------------------------------ grab (select mode)
GRAB_AT = (0.30, -0.05)
GRAB_END = {"held", "missed", "rejected", "hold", "backoff"}


def grab_log(bench: Bench, ts) -> list[tuple[str, str | None, str | None]]:
    return [(s["state"], s.get("reason"), s.get("detail")) for s in bench.stream_states(ts)]


def test_grab_of_a_cube_on_the_table_opens_hovers_descends_closes_lifts_and_holds_it(bench, kin, vol):
    twin = bench.twin
    ready(bench, kin, vol)
    h = 0.015
    twin.configure(objects=[Box("cube", (h, h, h), GRAB_AT, rgba=(0.85, 0.25, 0.2, 1.0))])
    try:
        d = bench.driver()
        ts = bench.grab_object([*GRAB_AT, h], size=[2 * h, 2 * h])
        st = bench.wait_stream(ts, GRAB_END, max_sim_s=40.0)
        log = grab_log(bench, ts)
        assert st["state"] == "held", log
        assert log[0][0] == "accepted"
        legs = [detail for state, _, detail in log if state == "moving"]
        assert legs == ["open", "hover", "descend", "close", "lift"], log
        assert st["stream"] == ts and st["ts"] == ts and list(st)[0] == "type" and st["mode"] == "sim"
        # Lifted back to the hover point (tips at h: TCP h + 0.033, hover 8 cm above), jaws closed on the cube.
        hover = np.array([*GRAB_AT, h + 0.033 + 0.08])
        assert np.linalg.norm(tcp(kin, twin) - hover) < 0.012
        assert twin.status.command["gripper"] < 5.0
        with twin.lock:
            cube_z = float(twin.scene.data.body("cube").xpos[2])
        assert cube_z > h + 0.03, cube_z                     # it came up with the gripper
        assert d.status()["grab"]["state"] == "held"
        assert twin.owner is None                            # released, holding the object up
    finally:
        bench.close()
        twin.configure(objects=None)


def test_grab_closing_on_nothing_is_missed_and_does_not_reopen(bench, kin, vol):
    twin = bench.twin
    ready(bench, kin, vol)
    bench.driver()
    ts = bench.grab_object([*GRAB_AT, 0.015], size=[0.03, 0.03])
    st = bench.wait_stream(ts, GRAB_END, max_sim_s=40.0)
    log = grab_log(bench, ts)
    assert st["state"] == "missed", log
    assert [detail for state, _, detail in log if state == "moving"] == ["open", "hover", "descend", "close",
                                                                         "lift"]
    bench.pump(sim_s=0.5)
    assert twin.status.command["gripper"] < 5.0            # never re-opens after `missed`


@pytest.mark.parametrize("case", ["outside grab zone", "too wide", "no depth", "stale"])
def test_grab_refusals(bench, kin, vol, case):
    twin = bench.twin
    ready(bench, kin, vol)
    bench.driver()
    before = dict(twin.status.measured)
    if case == "outside grab zone":
        ts = bench.grab_object([0.45, 0.0, 0.015], size=[0.03, 0.03])           # in the reach box, not the zone
    elif case == "too wide":
        ts = bench.grab_object([*GRAB_AT, 0.015], size=[0.15, 0.05])            # 80 % of 15 cm > 9 cm
    elif case == "no depth":
        ts = bench.grab_object([*GRAB_AT, 0.015], depth=False)
    else:
        ts = bench.grab_object([*GRAB_AT, 0.015], age=0.9)
    st = bench.wait_stream(ts, GRAB_END, max_sim_s=5.0)
    assert (st["state"], st.get("reason")) == ("rejected", case), grab_log(bench, ts)
    if case == "too wide":
        assert "15 cm" in st["detail"]
    bench.pump(sim_s=0.3)
    after = twin.status.measured
    assert max(abs(after[k] - before[k]) for k in twin.status.measured) < 0.5


def test_grab_without_a_drivable_gripper_is_refused(bench, kin, vol, monkeypatch):
    twin = bench.twin
    ready(bench, kin, vol)
    monkeypatch.setattr(twin.backend, "gripper_drivable", lambda: False, raising=False)
    d = bench.driver()
    assert d.status()["gripper_drivable"] is False
    ts = bench.grab_object([*GRAB_AT, 0.015], size=[0.03, 0.03])
    st = bench.wait_stream(ts, GRAB_END, max_sim_s=5.0)
    assert (st["state"], st.get("reason"), st.get("detail")) == ("rejected", "no gripper", "gripper disabled")
    g = bench.grip("close")
    gs = bench.wait_grip(g, {"rejected", "done", "hold"}, max_sim_s=5.0)
    assert (gs["state"], gs.get("reason")) == ("rejected", "no gripper")


def test_grab_in_dry_mode_accepts_then_holds_and_nothing_moves(bench, kin, vol):
    twin = bench.twin
    ready(bench, kin, vol)
    bench.driver(mode="dry")
    before = dict(twin.status.measured)
    grip = twin.status.command["gripper"]
    ts = bench.grab_object([*GRAB_AT, 0.015], size=[0.03, 0.03])
    st = bench.wait_stream(ts, GRAB_END, max_sim_s=5.0)
    assert [(s, r) for s, r, _ in grab_log(bench, ts)] == [("accepted", None), ("hold", None)], st
    bench.pump(sim_s=0.5)
    after = twin.status.measured
    assert max(abs(after[k] - before[k]) for k in before) < 0.5
    assert twin.status.command["gripper"] == pytest.approx(grip)
    assert twin.owner is None


def _dist_to_box(p, lo, hi) -> float:
    return float(np.linalg.norm(p - np.clip(p, lo, hi)))


@pytest.mark.parametrize("rule", ["effort", "lag"])
def test_impact_on_a_static_wall_backs_off_without_stop(bench, kin, vol, rule):
    twin = bench.twin
    ready(bench, kin, vol)
    d = bench.driver()
    if rule == "lag":
        d.sim_effort_frac = 100.0                          # effort never trips, leaving only the lag rule
    try:
        d._tools()                                         # build the collision checker before the wall exists
        c = vol.center
        half = (0.12, 0.01, 0.15)
        xy = (float(c[0]), float(c[1]) + 0.09)
        wall = Box("wall", size=half, xy=xy, rgba=(0.5, 0.5, 0.5, 1.0), free=False)
        twin.configure(objects=[wall])
        with twin.lock:
            m = twin.scene.model
            assert m.body_jntnum[m.body("wall").id] == 0
        lo = np.array([xy[0] - half[0], xy[1] - half[1], 0.0])
        hi = np.array([xy[0] + half[0], xy[1] + half[1], 2 * half[2]])
        trip_tcp: list[np.ndarray] = []
        real_backoff = d._backoff

        hits: list = []

        trip_q: list[dict] = []

        def spy(t):
            trip_tcp.append(tcp(kin, twin))
            trip_q.append(dict(twin.status.measured))
            hits.append(d._impact)
            return real_backoff(t)
        d._backoff = spy
        ts = bench.target(c + np.array([0.0, 0.2, -0.05]))
        st = bench.wait_state(ts, END, max_sim_s=15.0)
        assert st["state"] == "backoff", (st, bench.states_for(ts))
        assert st["reason"] == f"impact {rule} arm_joint1", st
        assert trip_tcp
        # Backoff goes to the pose from >= 0.4 s earlier that is >= 3 deg away on the most-moved joint
        # (the guard's rule), not necessarily J1, which hit. The TCP ends farther from the wall than the
        # ~2 mm rebound alone; a forearm hit moves the TCP only a little.
        measured = twin.status.measured
        assert max(abs(measured[j] - trip_q[0][j]) for j in d.arm) >= 2.5, (measured, trip_q[0])
        assert _dist_to_box(tcp(kin, twin), lo, hi) > _dist_to_box(trip_tcp[0], lo, hi) + 0.005
        pose = hits[0].backoff_pose
        assert max(abs(measured[j] - pose[j]) for j in d.arm) < 2.0, (measured, pose)
        assert twin.safety_state.value != "estop"
        assert twin.owner is None
    finally:
        d.stop()
        twin.configure(objects=None)


def _moving(bench: Bench, kin, vol, **kw):
    """Start a driver on a long move (J1 ~20 deg at 10 deg/s) and return mid-move."""
    ready(bench, kin, vol)
    kw.setdefault("speed_deg_s", 10.0)
    d = bench.driver(**kw)
    ts = bench.target(vol.center + np.array([0.0, 0.2, -0.05]))
    bench.wait_state(ts, {"moving"}, max_sim_s=5.0)
    bench.pump(sim_s=0.5)
    return d, ts


def test_stop_mid_move_holds_and_releases_the_arm(bench, kin, vol):
    twin = bench.twin
    d, ts = _moving(bench, kin, vol)
    try:
        twin.estop("test")
        st = bench.wait_state(ts, END, max_sim_s=5.0)
        assert (st["state"], st.get("reason")) == ("hold", "stop")
        assert twin.owner is None
        assert bench.stands(d.arm)
        assert d.last_state["state"] != "reached"
        assert d._worker is not None and d._worker.is_alive()          # the worker thread did not crash
    finally:
        d.stop()
        twin.clear_estop()


def test_command_moved_during_the_real_hold_off_holds_with_moved(bench, kin, vol):
    # Real mode on the sim backend: a hand target waits `hold_off_s`, and meanwhile the panel moves J1 by 45 deg.
    twin = bench.twin
    ready(bench, kin, vol)
    bench.driver(mode="real", real_ok=True, hold_off_s=1.0)
    cmd0 = dict(twin.status.command)
    ts = bench.target(vol.center + np.array([0.0, -0.08, -0.03]))       # J1 turns opposite to the panel
    t_rx = bench.clock.now()                                             # received (the clock stood still)
    bench.wait_state(ts, {"accepted", "rejected"}, max_sim_s=5.0)
    assert bench.states_for(ts) == ["accepted"]
    twin.claim("panel")
    twin.set_target({"arm_joint1": cmd0["arm_joint1"] + 45.0}, owner="panel")
    twin.release("panel")
    st = bench.wait_state(ts, {"moving", "reached", "hold", "backoff", "rejected"}, max_sim_s=5.0)
    assert (st["state"], st.get("reason")) == ("hold", "moved"), bench.states_for(ts)
    assert "rozkaz zmieniony" in st.get("detail", ""), st
    assert "moving" not in bench.states_for(ts)
    # The whole hold-off, not a moment more: planning takes no simulated time, so 1 s of the shared clock
    # after the target came in (+ a few ticks).
    assert 1.0 - 1e-6 <= bench.clock.now() - t_rx <= 1.0 + 3 * 0.02
    assert twin.owner is None


def test_relay_loss_mid_move_in_real_mode_holds(bench, kin, vol, relay):
    # Real mode on the sim backend: only real mode holds on relay loss (sim mode keeps going).
    twin = bench.twin
    d, ts = _moving(bench, kin, vol, mode="real", real_ok=True, hold_off_s=0.0)
    relay.drop_clients()
    # The network thread notices in real time; the clock stands still meanwhile.
    Bench.wait_real(lambda: d._cancel_cause == "relay", what="utrata przekaznika u kierowcy")
    assert bench.pump(until=lambda: (d.last_state or {}).get("state") == "hold", max_sim_s=3.0)
    assert (d.last_state["state"], d.last_state.get("reason")) == ("hold", "relay")
    assert twin.owner is None
    assert bench.stands(d.arm)
    assert d.last_state["state"] != "reached"


def test_mode_change_mid_move_holds(bench, kin, vol):
    twin = bench.twin
    d, ts = _moving(bench, kin, vol)
    d.set_mode("dry", False)
    st = bench.wait_state(ts, END, max_sim_s=5.0)
    assert (st["state"], st.get("reason")) == ("hold", "mode")
    assert twin.owner is None
    assert bench.stands(d.arm)
    assert d.last_state["state"] != "reached"


def test_exception_in_the_worker_holds_and_the_driver_takes_the_next_target(bench, kin, vol, monkeypatch):
    twin = bench.twin
    d, ts = _moving(bench, kin, vol)
    real_check = d._check

    def boom(gen):
        raise ValueError("atrapa bledu")
    monkeypatch.setattr(d, "_check", boom)
    st = bench.wait_state(ts, END, max_sim_s=5.0)
    assert (st["state"], st.get("reason")) == ("hold", "error")
    assert "atrapa bledu" in st["detail"]
    assert twin.owner is None
    assert bench.stands(d.arm)
    assert twin.safety_state.value != "estop"
    monkeypatch.setattr(d, "_check", real_check)
    p = vol.center + np.array([0.03, 0.05, -0.02])
    ts2 = bench.target(p)
    st = bench.wait_state(ts2, END, max_sim_s=15.0)
    assert st["state"] == "reached", (st, bench.states_for(ts2))
    assert np.linalg.norm(tcp(kin, twin) - p) < 0.005


def _env() -> dict[str, str]:
    import os
    env = {k: v for k, v in os.environ.items() if k != "LEROBOT_MP_CONFIG"}
    env["PYTHONPATH"] = str(SRC)
    return env


@pytest.mark.cli
def test_cli_specs_help():
    out = subprocess.run([sys.executable, "-m", "lerobot_mp.twin.cli", "specs", "--help"], capture_output=True,
                         text=True, env=_env(), timeout=60)
    assert out.returncode == 0, out.stderr
    for flag in ("--relay", "--mode", "--grasp", "--trusted-only", "--i-know", "--seconds", "--follow-speed"):
        assert flag in out.stdout


def test_cli_specs_refuses_real_without_i_know(relay):
    from lerobot_mp.twin.cli import main
    assert main(["specs", "--mode", "real", "--relay", relay.url, "--seconds", "1"]) == 2


@pytest.mark.cli
def test_cli_specs_runs_against_the_relay_in_sim(relay, tmp_path):
    ws = tmp_path / "twin.json"
    ws.write_text(json.dumps({"robot": "a1x", "backend": "sim"}), encoding="utf-8")
    # One second is enough: the overlay goes out right after the driver connects.
    out = subprocess.run([sys.executable, "-m", "lerobot_mp.twin.cli", "specs", "--relay", relay.url, "--mode",
                          "sim", "--seconds", "1", "--workspace", str(ws)], capture_output=True, text=True,
                         env=_env(), timeout=90)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "tryb sim" in out.stdout
    assert relay.overlays and relay.overlays[0]["objects"][0]["kind"] == "volume"
