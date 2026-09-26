"""Specs -> relay -> `SpecsReach` driver -> A1X arm in simulation (no hardware).

The relay is an in-process fake (`websockets.sync.server`) on a free port. On connect it sends a
trusted registration for session "s1"; it records overlays and target states, and tests inject targets.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("websockets")

from websockets.sync.server import serve  # noqa: E402

from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics  # noqa: E402
from lerobot_mp.twin.reach import ReachVolume  # noqa: E402
from lerobot_mp.twin.scene import Box  # noqa: E402
from lerobot_mp.twin.specsreach import SpecsReach  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"


class FakeRelay:
    """Fake of the `/twin` side of `server.js`."""

    def __init__(self, session: str = "s1", trusted: bool = True):
        self.session, self.trusted = session, trusted
        self.overlays: list[dict] = []
        self.states: list[dict] = []
        self.clients: list = []
        self.cond = threading.Condition()
        self.server = serve(self._handler, "127.0.0.1", 0)
        self.port = self.server.socket.getsockname()[1]
        self.url = f"ws://127.0.0.1:{self.port}/twin"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def _handler(self, ws) -> None:
        ws.send(json.dumps({"type": "registration", "T_world2base": np.eye(4).tolist(), "n": 30,
                            "trusted": self.trusted, "reason": "ok", "session": self.session}))
        with self.cond:
            self.clients.append(ws)
            self.cond.notify_all()
        try:
            for text in ws:
                msg = json.loads(text)
                assert next(iter(msg)) == "type", text             # the real relay reads the type from the first key
                with self.cond:
                    if msg["type"] == "overlay":
                        self.overlays.append(msg)
                    elif msg["type"] == "targetState":
                        self.states.append(msg)
                    self.cond.notify_all()
        finally:
            with self.cond:
                if ws in self.clients:
                    self.clients.remove(ws)

    def wait_client(self, timeout: float = 5.0) -> None:
        with self.cond:
            assert self.cond.wait_for(lambda: self.clients, timeout), "kierowca nie polaczyl sie z przekaznikiem"

    def target(self, pos, *, source: str = "hand", session: str = "s1", ts: int | None = None, **extra) -> int:
        ts = ts if ts is not None else int(time.time() * 1000) + len(self.states)
        msg = {"type": "target", "source": source, **extra, "pos": [float(v) for v in pos],
               "session": session, "ts": ts, "trusted": True}
        with self.cond:
            clients = list(self.clients)
        for ws in clients:
            ws.send(json.dumps(msg))
        return ts

    def states_for(self, ts: int) -> list[str]:
        with self.cond:
            return [s["state"] for s in self.states if s["ts"] == ts]

    def wait_state(self, ts: int, states: set[str], timeout: float) -> dict:
        def found():
            with_ts = [s for s in self.states if s["ts"] == ts and s["state"] in states]
            return with_ts[-1] if with_ts else None
        with self.cond:
            ok = self.cond.wait_for(found, timeout)
            assert ok, f"brak stanu {states} dla {ts}: {[(s['ts'], s['state'], s.get('reason')) for s in self.states]}"
            return found()

    def drop_clients(self) -> None:
        """Simulates the relay dying or the network dropping."""
        with self.cond:
            clients = list(self.clients)
        for ws in clients:
            ws.close()

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture(scope="module")
def twin():
    tw = rt.Twin(Workspace(robot="a1x"))
    tw.connect("sim", threaded=True)
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


def ready(twin, kin, vol) -> None:
    """Park the arm above the volume centre: from home every target is a > 45 deg jump, rejected as "jump"."""
    if twin.safety_state is not None and twin.safety_state.value == "estop":
        twin.clear_estop()
    twin.configure(objects=None)
    sol = kin.ik(vol.center, seed=twin.spec.home, tol=0.003)
    assert sol.ok
    goal = {k: v for k, v in sol.joints.items() if k != twin.spec.gripper}
    twin.move(goal, 2.0, owner="test")
    twin.release("test")


def driver(twin, relay, **kw) -> SpecsReach:
    kw.setdefault("mode", "sim")
    d = SpecsReach(twin, relay_url=relay.url, **kw)
    d.start()
    relay.wait_client()
    time.sleep(0.2)                                      # let the registration reach the driver
    return d


def tcp(kin, twin) -> np.ndarray:
    return kin.tcp(twin.status.measured or twin.joints())[:3, 3]


def test_hand_target_inside_volume_is_reached(twin, kin, vol, relay):
    ready(twin, kin, vol)
    d = driver(twin, relay)
    try:
        first = relay.overlays[0]
        kinds = {o["id"]: o["kind"] for o in first["objects"]}
        assert kinds == {"reach": "volume", "tcp": "sphere"}
        vol_obj = next(o for o in first["objects"] if o["id"] == "reach")
        assert vol_obj["alpha"] == pytest.approx(0.35)
        p = vol.center + np.array([0.05, 0.08, -0.05])
        ts = relay.target(p)
        relay.wait_state(ts, {"moving", "rejected"}, 5.0)
        # While moving, the supervisor caps speed at 1.5 x ramp speed + 5: 20 deg/s -> 35.
        assert twin._supervisor.speed_cap == pytest.approx(35.0)
        st = relay.wait_state(ts, {"reached", "rejected", "hold", "backoff"}, 10.0)
        assert st["state"] == "reached", st
        assert relay.states_for(ts)[:2] == ["accepted", "moving"]
        assert st["mode"] == "sim" and list(st)[0] == "type"
        assert np.linalg.norm(tcp(kin, twin) - p) < 0.005
        assert twin.owner is None                          # released; the clutch holds the target
        deadline = time.monotonic() + 2.0
        while twin._speed_cap is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert twin._speed_cap is None and twin._supervisor.speed_cap is None
        ids = [o["id"] for m in relay.overlays for o in m["objects"]]
        assert ids.count("tcp") >= 3 and "target" in ids
    finally:
        d.stop()


@pytest.mark.parametrize("case", ["outside", "session", "no depth", "stale", "untrusted"])
def test_rejections(twin, kin, vol, relay, case):
    d = driver(twin, relay)
    try:
        before = dict(twin.status.measured)
        if case == "outside":
            ts, why = relay.target(vol.hi + [0.05, 0.0, 0.0]), "outside volume"
        elif case == "session":
            ts, why = relay.target(vol.center, session="stara"), "session"
        elif case == "no depth":
            ts, why = relay.target(vol.center, source="object", label="cup"), "no depth"
        elif case == "stale":
            # Detection last actually seen 0.8 s ago (the glasses lost the object): don't move.
            ts = relay.target(vol.center, source="object", label="cup", depthMeasured=True, age=0.8)
            why = "stale"
        else:
            relay.trusted = False
            d2 = driver(twin, relay)
            d.stop()
            d = d2
            ts, why = relay.target(vol.center), "untrusted"
        st = relay.wait_state(ts, {"rejected", "reached"}, 5.0)
        assert (st["state"], st.get("reason")) == ("rejected", why)
        if case == "stale":
            assert "0.80 s" in st["detail"], st
        after = twin.status.measured
        assert max(abs(after[k] - before[k]) for k in d.arm) < 0.5
    finally:
        d.stop()


def test_object_target_goes_above_it_pointing_down(twin, kin, vol, relay):
    ready(twin, kin, vol)
    d = driver(twin, relay)
    try:
        p = vol.center + np.array([0.0, -0.05, -0.08])
        ts = relay.target(p, source="object", label="cup", depthMeasured=True, age=0.2)
        st = relay.wait_state(ts, {"reached", "rejected", "hold", "backoff"}, 12.0)
        assert st["state"] == "reached", st
        T = kin.tcp(twin.status.measured)
        assert np.linalg.norm(T[:3, 3] - (p + [0.0, 0.0, 0.08])) < 0.005
        approach = T[:3, :3] @ np.asarray(twin.spec.tcp_approach)
        assert approach[2] < -0.95                          # gripper points down
    finally:
        d.stop()


def test_object_grasp_descends_closes_and_lifts(twin, kin, vol, relay):
    ready(twin, kin, vol)
    d = driver(twin, relay, grasp=True)
    try:
        p = vol.center + np.array([0.0, 0.05, -0.1])
        ts = relay.target(p, source="object", label="cup", depthMeasured=True)
        st = relay.wait_state(ts, {"reached", "rejected", "hold", "backoff"}, 25.0)
        assert st["state"] == "reached", (st, relay.states_for(ts))
        assert relay.states_for(ts).count("moving") == 3            # above, down, up
        # The down/up legs aren't corrected for sag under load, so it remains (up to ~8 mm in sim).
        assert np.linalg.norm(tcp(kin, twin) - (p + [0.0, 0.0, 0.08])) < 0.008
        assert twin.status.command["gripper"] < 5.0
    finally:
        d.stop()


def test_dry_mode_accepts_then_holds_and_does_not_move(twin, kin, vol, relay):
    ready(twin, kin, vol)
    d = driver(twin, relay, mode="dry")
    try:
        before = dict(twin.status.measured)
        ts = relay.target(vol.center + [0.08, 0.1, -0.05])
        relay.wait_state(ts, {"hold", "rejected"}, 5.0)
        time.sleep(0.5)
        assert relay.states_for(ts) == ["accepted", "hold"]
        assert relay.states[-1]["mode"] == "dry"
        after = twin.status.measured
        assert max(abs(after[k] - before[k]) for k in d.arm) < 0.5
        assert any(o["id"] == "target" for m in relay.overlays for o in m["objects"])   # planned target still shown
    finally:
        d.stop()


def test_new_target_mid_move_retargets(twin, kin, vol, relay):
    ready(twin, kin, vol)
    d = driver(twin, relay, speed_deg_s=10.0)
    try:
        p1 = vol.center + np.array([0.0, 0.2, -0.05])
        p2 = vol.center + np.array([0.08, -0.15, -0.03])
        ts1 = relay.target(p1)
        relay.wait_state(ts1, {"moving", "rejected"}, 5.0)
        time.sleep(0.6)
        ts2 = relay.target(p2)
        st = relay.wait_state(ts2, {"reached", "rejected", "hold", "backoff"}, 15.0)
        assert st["state"] == "reached", st
        assert "reached" not in relay.states_for(ts1)
        old = [x for x in relay.states if x["ts"] == ts1 and x["state"] == "hold"]
        assert old and old[-1]["reason"] == "retargeted"
        assert np.linalg.norm(tcp(kin, twin) - p2) < 0.005
    finally:
        d.stop()


def test_trust_holds_for_the_session_and_a_new_session_starts_without_it(twin, kin, vol, relay):
    d = driver(twin, relay)
    try:
        assert d.status()["session"] == "s1"
        # A wobbly update in the same session (the wearer glanced at a tag from afar) keeps the trust.
        d._on_message(json.dumps({"type": "registration", "session": "s1", "trusted": False}))
        assert d.status()["session"] == "s1"
        # The Lens restarted: nothing is trusted until its new session registers.
        d._on_message(json.dumps({"type": "registration", "session": "s2", "trusted": False}))
        assert d.status()["session"] is None
        ts = relay.target(vol.center)
        st = relay.wait_state(ts, {"rejected", "reached"}, 5.0)
        assert (st["state"], st.get("reason")) == ("rejected", "untrusted")
    finally:
        d.stop()


def test_start_center_goes_above_the_volume_centre_in_stages(twin, kin, vol, relay):
    if twin.safety_state is not None and twin.safety_state.value == "estop":
        twin.clear_estop()
    twin.configure(objects=None)
    twin.move({k: v for k, v in twin.spec.home.items() if k != twin.spec.gripper}, 2.0, owner="test")
    twin.release("test")
    d = SpecsReach(twin, relay_url=relay.url, mode="sim")
    assert d.start_center(30.0) is None
    assert np.linalg.norm(tcp(kin, twin) - vol.center) < 0.01
    assert twin.owner is None and twin._speed_cap is None


def _dist_to_box(p, lo, hi) -> float:
    return float(np.linalg.norm(p - np.clip(p, lo, hi)))


@pytest.mark.parametrize("rule", ["effort", "lag"])
def test_impact_on_a_static_wall_backs_off_without_stop(twin, kin, vol, relay, rule):
    ready(twin, kin, vol)
    d = driver(twin, relay)
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
        ts = relay.target(c + np.array([0.0, 0.2, -0.05]))
        st = relay.wait_state(ts, {"reached", "rejected", "hold", "backoff"}, 15.0)
        assert st["state"] == "backoff", (st, relay.states_for(ts))
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


def _moving(twin, kin, vol, relay, **kw):
    """Start a driver on a long move (J1 ~20 deg at 10 deg/s) and return mid-move."""
    ready(twin, kin, vol)
    kw.setdefault("speed_deg_s", 10.0)
    d = driver(twin, relay, **kw)
    ts = relay.target(vol.center + np.array([0.0, 0.2, -0.05]))
    relay.wait_state(ts, {"moving"}, 5.0)
    time.sleep(0.5)
    return d, ts


def _stands(twin, d, secs: float = 0.5) -> bool:
    """After 0.5 s to brake, the arm stays still for `secs` instead of continuing to the target."""
    time.sleep(0.5)
    a = dict(twin.status.measured)
    time.sleep(secs)
    b = twin.status.measured
    return max(abs(a[k] - b[k]) for k in d.arm) < 0.5


def test_stop_mid_move_holds_and_releases_the_arm(twin, kin, vol, relay):
    d, ts = _moving(twin, kin, vol, relay)
    try:
        twin.estop("test")
        st = relay.wait_state(ts, {"reached", "hold", "backoff", "rejected"}, 5.0)
        assert (st["state"], st.get("reason")) == ("hold", "stop")
        assert twin.owner is None
        assert _stands(twin, d)
        assert d.last_state["state"] != "reached"
        assert d._worker is not None and d._worker.is_alive()          # the worker thread did not crash
    finally:
        d.stop()
        twin.clear_estop()


def test_command_moved_during_the_real_hold_off_holds_with_moved(twin, kin, vol, relay):
    # Real mode on the sim backend: a hand target waits `hold_off_s`, and meanwhile the panel moves J1 by 45 deg.
    ready(twin, kin, vol)
    d = driver(twin, relay, mode="real", real_ok=True, hold_off_s=1.0)
    try:
        cmd0 = dict(twin.status.command)
        ts = relay.target(vol.center + np.array([0.0, -0.08, -0.03]))       # J1 turns opposite to the panel
        relay.wait_state(ts, {"accepted", "rejected"}, 5.0)
        assert relay.states_for(ts) == ["accepted"]
        twin.claim("panel")
        twin.set_target({"arm_joint1": cmd0["arm_joint1"] + 45.0}, owner="panel")
        twin.release("panel")
        st = relay.wait_state(ts, {"moving", "reached", "hold", "backoff", "rejected"}, 5.0)
        assert (st["state"], st.get("reason")) == ("hold", "moved"), relay.states_for(ts)
        assert "rozkaz zmieniony" in st.get("detail", ""), st
        assert "moving" not in relay.states_for(ts)
        assert twin.owner is None
    finally:
        d.stop()


def test_relay_loss_mid_move_in_real_mode_holds(twin, kin, vol, relay):
    # Real mode on the sim backend: only real mode holds on relay loss (sim mode keeps going).
    d, ts = _moving(twin, kin, vol, relay, mode="real", real_ok=True, hold_off_s=0.0)
    try:
        relay.drop_clients()
        deadline = time.monotonic() + 3.0
        while (d.last_state or {}).get("state") != "hold" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (d.last_state["state"], d.last_state.get("reason")) == ("hold", "relay")
        assert twin.owner is None
        assert _stands(twin, d)
        assert d.last_state["state"] != "reached"
    finally:
        d.stop()


def test_mode_change_mid_move_holds(twin, kin, vol, relay):
    d, ts = _moving(twin, kin, vol, relay)
    try:
        d.set_mode("dry", False)
        st = relay.wait_state(ts, {"reached", "hold", "backoff", "rejected"}, 5.0)
        assert (st["state"], st.get("reason")) == ("hold", "mode")
        assert twin.owner is None
        assert _stands(twin, d)
        assert d.last_state["state"] != "reached"
    finally:
        d.stop()


def test_exception_in_the_worker_holds_and_the_driver_takes_the_next_target(twin, kin, vol, relay, monkeypatch):
    d, ts = _moving(twin, kin, vol, relay)
    try:
        real_check = d._check

        def boom(gen):
            raise ValueError("atrapa bledu")
        monkeypatch.setattr(d, "_check", boom)
        st = relay.wait_state(ts, {"reached", "hold", "backoff", "rejected"}, 5.0)
        assert (st["state"], st.get("reason")) == ("hold", "error")
        assert "atrapa bledu" in st["detail"]
        assert twin.owner is None
        assert _stands(twin, d)
        assert twin.safety_state.value != "estop"
        monkeypatch.setattr(d, "_check", real_check)
        p = vol.center + np.array([0.03, 0.05, -0.02])
        ts2 = relay.target(p)
        st = relay.wait_state(ts2, {"reached", "rejected", "hold", "backoff"}, 15.0)
        assert st["state"] == "reached", (st, relay.states_for(ts2))
        assert np.linalg.norm(tcp(kin, twin) - p) < 0.005
    finally:
        d.stop()


def _env() -> dict[str, str]:
    import os
    env = {k: v for k, v in os.environ.items() if k != "LEROBOT_MP_CONFIG"}
    env["PYTHONPATH"] = str(SRC)
    return env


def test_cli_specs_help():
    out = subprocess.run([sys.executable, "-m", "lerobot_mp.twin.cli", "specs", "--help"], capture_output=True,
                         text=True, env=_env(), timeout=60)
    assert out.returncode == 0, out.stderr
    for flag in ("--relay", "--mode", "--grasp", "--allow-untrusted", "--i-know", "--seconds"):
        assert flag in out.stdout


def test_cli_specs_refuses_real_without_i_know(relay):
    from lerobot_mp.twin.cli import main
    assert main(["specs", "--mode", "real", "--relay", relay.url, "--seconds", "1"]) == 2


def test_cli_specs_runs_against_the_relay_in_sim(relay, tmp_path):
    ws = tmp_path / "twin.json"
    ws.write_text(json.dumps({"robot": "a1x", "backend": "sim"}), encoding="utf-8")
    out = subprocess.run([sys.executable, "-m", "lerobot_mp.twin.cli", "specs", "--relay", relay.url, "--mode",
                          "sim", "--seconds", "3", "--workspace", str(ws)], capture_output=True, text=True,
                         env=_env(), timeout=90)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "tryb sim" in out.stdout
    assert relay.overlays and relay.overlays[0]["objects"][0]["kind"] == "volume"
