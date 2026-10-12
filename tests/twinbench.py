"""Test bench: a twin stepped on a `VirtualClock`, the fake relay, and helpers that drive `SpecsReach` on them.

The twin runs without a loop thread (`connect(threaded=False)`) on a `VirtualClock`, and the driver reads
the same clock (`SpecsReach.clock` defaults to `twin.clock`). `Bench.pump` moves time: each tick advances
the clock, so the driver's worker computes its next ramp command and parks again, and then runs one loop
tick. Nothing waits in wall time, so ten simulated seconds of motion cost a fraction of a second and the
ramp, arrival, hold-off and backoff timings repeat to the tick.

The relay, the websocket and the driver's network thread stay real. After sending a message the bench
waits in real time, without moving the clock, until the driver has handled it (`target`, `grip`). It
records every message the driver sends at the moment the driver sends it (`sent`), so `wait_state` and
`states_for` don't depend on network latency. The relay's own lists (`relay.states`, `relay.overlays`)
are what actually crossed the wire.

    bench = Bench.sim("a1x", relay)
    d = bench.driver(mode="sim")
    ts = bench.target(p)
    st = bench.wait_state(ts, {"reached", "hold"}, max_sim_s=10.0)
"""

from __future__ import annotations

import itertools
import json
import math
import threading
import time
from collections.abc import Callable, Mapping

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from lerobot_mp.twin import runtime as rt
from lerobot_mp.twin.ant import add_ant
from lerobot_mp.twin.clock import VirtualClock
from lerobot_mp.twin.runtime import smoothstep
from lerobot_mp.twin.specsreach import SpecsReach
from lerobot_mp.twin.workspace import Workspace

DT = 0.02                                      # twin loop period, 50 Hz
END = frozenset({"reached", "rejected", "hold", "backoff"})
GRIP_END = frozenset({"done", "rejected", "hold"})


class FakeRelay:
    """Fake of the `/twin` side of `server.js`: sends a registration on connect, records what it gets."""

    def __init__(self, session: str = "s1", trusted: bool = True):
        self.session, self.trusted = session, trusted
        self.overlays: list[dict] = []
        self.states: list[dict] = []
        self.grips: list[dict] = []
        #: Every targetState and gripperState in the order received.
        self.events: list[dict] = []
        self.clients: list = []
        self.cond = threading.Condition()
        # Opaque, unique message timestamps (two targets in the same millisecond must not share one).
        self._ts = itertools.count(int(time.time() * 1000))
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
                        self.events.append(msg)
                    elif msg["type"] == "gripperState":
                        self.grips.append(msg)
                        self.events.append(msg)
                    self.cond.notify_all()
        finally:
            with self.cond:
                if ws in self.clients:
                    self.clients.remove(ws)

    def wait_client(self, timeout: float = 5.0) -> None:
        with self.cond:
            assert self.cond.wait_for(lambda: self.clients, timeout), "kierowca nie polaczyl sie z przekaznikiem"

    def _broadcast(self, msg: dict) -> None:
        with self.cond:
            clients = list(self.clients)
        for ws in clients:
            try:
                ws.send(json.dumps(msg))
            except ConnectionClosed:                                # a driver that just stopped; the relay drops it
                pass

    def target(self, pos, *, source: str = "hand", session: str = "s1", ts: int | None = None, **extra) -> int:
        ts = ts if ts is not None else next(self._ts)
        self._broadcast({"type": "target", "source": source, **extra, "pos": [float(v) for v in pos],
                         "session": session, "ts": ts, "trusted": True})
        return ts

    def grip(self, action: str, *, ts: int | None = None) -> int:
        """A gripper command from the glasses (a left-hand pinch)."""
        ts = ts if ts is not None else next(self._ts)
        self._broadcast({"type": "gripper", "action": action, "session": self.session, "ts": ts})
        return ts

    def send(self, msg: dict) -> None:
        self._broadcast(msg)

    def follow(self, stream: int, seq: int, pos=None, *, active: bool = True, yaw: float | None = None,
               ts: int | None = None, session: str = "s1") -> int:
        """One sample of a follow stream (teleop): `stream` = ts of its first sample, `seq` from 0."""
        ts = ts if ts is not None else stream + seq
        msg = {"type": "follow", "ts": ts, "session": session, "trusted": True, "stream": stream, "seq": seq,
               "active": active}
        if active:
            msg["pos"] = [round(float(v), 4) for v in pos]
            msg["yaw"] = yaw
        self._broadcast(msg)
        return ts

    def grab_object(self, pos, *, size=None, yaw=None, label: str = "cup", depth: bool = True, age: float = 0.1,
                    ts: int | None = None, session: str = "s1") -> int:
        """A pinch on a detection in the grab zone."""
        ts = ts if ts is not None else next(self._ts)
        obj = {"pos": [float(v) for v in pos], "surface": [float(v) for v in pos], "ray": [1.0, 0.0, 0.0],
               "size": size, "yaw": yaw, "label": label, "id": 7}
        self._broadcast({"type": "grab", "ts": ts, "session": session, "trusted": True, "stream": ts,
                         "policy": "topdown", "object": obj, "depthMeasured": depth, "age": age})
        return ts

    def stream_states(self, stream: int) -> list[dict]:
        with self.cond:
            return [s for s in self.states if s.get("stream") == stream]

    def saw(self, ts: int, states) -> dict | None:
        """The last targetState for `ts` in `states` that reached the relay, or None (never blocks)."""
        with self.cond:
            hit = [s for s in self.states if s["ts"] == ts and s["state"] in states]
        return hit[-1] if hit else None

    def saw_grip(self, ts: int, states) -> dict | None:
        with self.cond:
            hit = [g for g in self.grips if g["ts"] == ts and g["state"] in states]
        return hit[-1] if hit else None

    def states_for(self, ts: int) -> list[str]:
        with self.cond:
            return [s["state"] for s in self.states if s["ts"] == ts]

    def grips_for(self, ts: int) -> list[str]:
        with self.cond:
            return [g["state"] for g in self.grips if g["ts"] == ts]

    def wait_state(self, ts: int, states, timeout: float) -> dict:
        """Real-time wait: only for a twin with its own loop thread (a stepped twin needs `Bench.wait_state`)."""
        with self.cond:
            ok = self.cond.wait_for(lambda: self.saw(ts, states), timeout)   # the condition's lock is an RLock
            assert ok, f"brak stanu {states} dla {ts}: {[(s['ts'], s['state'], s.get('reason')) for s in self.states]}"
        found = self.saw(ts, states)
        assert found is not None
        return found

    def wait_grip(self, ts: int, states, timeout: float) -> dict:
        """Real-time wait: only for a twin with its own loop thread (a stepped twin needs `Bench.wait_grip`)."""
        def found():
            return any(g["ts"] == ts and g["state"] in states for g in self.grips)
        with self.cond:
            ok = self.cond.wait_for(found, timeout)
            assert ok, f"brak stanu chwytaka {states} dla {ts}: {[(g['ts'], g['state']) for g in self.grips]}"
        hit = self.saw_grip(ts, states)
        assert hit is not None
        return hit

    def drop_clients(self) -> None:
        """Simulates the relay dying or the network dropping."""
        with self.cond:
            clients = list(self.clients)
        for ws in clients:
            ws.close()

    def close(self) -> None:
        self.server.shutdown()


class Bench:
    """A stepped twin on a `VirtualClock`, optionally with the fake relay and `SpecsReach` drivers on it."""

    #: A pump that takes longer than this in real time fails instead of hanging the suite.
    real_timeout_s = 60.0

    def __init__(self, twin: rt.Twin, relay: FakeRelay | None = None):
        assert isinstance(twin.clock, VirtualClock), "Bench needs a twin on a VirtualClock (connect(threaded=False))"
        self.twin = twin
        self.clock: VirtualClock = twin.clock
        self.relay = relay
        #: Every message the attached drivers sent, recorded when they sent it (before the network).
        self.sent: list[dict] = []
        #: Every message the attached drivers received, recorded once handled.
        self.handled: list[dict] = []
        self.drivers: list[SpecsReach] = []
        self._lock = threading.Lock()
        # Barrier timeouts before this bench (a module-scoped clock outlives many benches).
        self._late0 = self.clock.late

    @classmethod
    def sim(cls, robot: str = "a1x", relay: FakeRelay | None = None, *, t0: float = 100.0,
            ws: Workspace | None = None) -> Bench:
        """A simulated `robot` (or workspace `ws`) on a fresh `VirtualClock` starting at `t0` s."""
        tw = rt.Twin(ws if ws is not None else Workspace(robot=robot), clock=VirtualClock(t0=t0))
        tw.connect("sim", threaded=False)
        return cls(tw, relay)

    # ------------------------------------------------------------ time
    def step(self, dt: float = DT) -> None:
        """One loop tick of `dt` (the clock first, then the loop), then the drivers' stream reports go out."""
        self.twin.step(dt)
        self.flush_reports()

    def flush_reports(self) -> None:
        """Sends the attached drivers' queued follow-stream reports now, from this thread.

        The driver queues a stream's `targetState` on the loop or worker thread and its network thread sends
        it within a receive slice of real time; flushing here puts it in `sent` on the tick it happened."""
        for d in self.drivers:
            d._flush_follow_reports()

    def pump(self, *, until: Callable[[], object] | None = None, sim_s: float | None = None,
             max_sim_s: float = 10.0, dt: float = DT) -> bool:
        """Steps the loop `dt` at a time for `sim_s` simulated seconds (`max_sim_s` if not given), stopping
        early once `until()` holds. Returns whether `until` held; True without `until`."""
        if until is not None and until():
            return True
        span = sim_s if sim_s is not None else max_sim_s
        ticks = max(1, round(span / dt)) if sim_s is not None else max(1, math.ceil(span / dt - 1e-9))
        deadline = time.monotonic() + self.real_timeout_s
        for _ in range(ticks):
            self.step(dt)
            if until is not None and until():
                return True
            if time.monotonic() > deadline:
                raise AssertionError(f"pump: {self.real_timeout_s:.0f} s of real time for {span} s simulated "
                                     f"(clock at {self.clock.now():.2f}, late barriers {self.clock.late})")
        return until is None or bool(until())

    def pump_to(self, t: float, *, until: Callable[[], object] | None = None, dt: float = DT) -> bool:
        """Steps the loop until the clock reaches `t` (a tick that would overshoot it by less than half a tick
        still counts: a cadence that is not a multiple of `dt` alternates, 3-2-3-2 ticks for 50 ms at 20 ms),
        stopping early once `until()` holds. Returns whether `until` held; True without `until`."""
        return self.pump(until=lambda: self.clock.now() >= t - dt / 2 or (until is not None and until()),
                         max_sim_s=max(0.0, t - self.clock.now()) + dt) and (until is None or bool(until()))

    @staticmethod
    def wait_real(pred: Callable[[], object], timeout: float = 10.0, what: str = "") -> None:
        """Waits in real time, without moving the clock, for something the network threads do."""
        deadline = time.monotonic() + timeout
        while not pred():
            assert time.monotonic() < deadline, f"nie doczekano sie: {what or pred}"
            time.sleep(0.0005)

    def call(self, fn: Callable[[], object], *, max_sim_s: float = 30.0):
        """Runs a blocking call (`start_center`, `Twin.move`) in a thread while pumping; returns its result.

        The thread is a clock member for the call, so each `advance` waits until it is back in a clock wait
        and the call's steps land on the same ticks whatever the thread scheduling."""
        out: dict = {}

        def run():
            try:
                with self.clock.member():
                    out["result"] = fn()
            except BaseException as exc:                            # noqa: BLE001 - re-raised below
                out["error"] = exc
        th = threading.Thread(target=run, name="bench-call", daemon=True)
        th.start()
        assert self.pump(until=lambda: not th.is_alive(), max_sim_s=max_sim_s), \
            f"{getattr(fn, '__name__', fn)} nie skonczylo w {max_sim_s} s symulowanych"
        th.join(5.0)
        if "error" in out:
            raise out["error"]
        return out.get("result")

    # ------------------------------------------------------------ the arm
    def ramp_to(self, goal: Mapping[str, float], *, duration: float = 2.0, owner: str = "test",
                tol_deg: float = 1.0, max_sim_s: float = 5.0) -> None:
        """`Twin.move` for a stepped twin: a smoothstep ramp as `owner` to `goal`, then waits until the arm
        stands within `tol_deg`, and releases it (engaged, holding the goal). No ramp if already there."""
        tw = self.twin
        joints = list(goal)
        cmd = tw.status.command or tw.joints()
        start = {k: float(cmd[k]) for k in joints}
        if max(abs(start[k] - goal[k]) for k in joints) > 0.5:
            tw.claim(owner)
            try:
                tw.set_engaged(True, owner=owner)
                n = max(1, round(duration / DT))
                for i in range(1, n + 1):
                    s = smoothstep(i / n)
                    tw.set_target({k: start[k] + s * (goal[k] - start[k]) for k in joints}, owner=owner)
                    self.step()
            finally:
                tw.release(owner)
        prev: list[dict] = [dict(tw.status.measured)]

        def arrived() -> bool:
            m = dict(tw.status.measured)
            still = max(abs(m[k] - prev[0][k]) for k in joints) / DT < 2.0
            prev[0] = m
            return still and max(abs(m[k] - goal[k]) for k in joints) < tol_deg
        assert self.pump(until=arrived, max_sim_s=max_sim_s), (dict(tw.status.measured), dict(goal))

    def stands(self, joints, *, secs: float = 0.5, brake_s: float = 0.5, tol_deg: float = 0.5) -> bool:
        """After `brake_s` to brake, the arm stays within `tol_deg` for `secs` (simulated)."""
        self.pump(sim_s=brake_s)
        a = dict(self.twin.status.measured)
        self.pump(sim_s=secs)
        b = self.twin.status.measured
        return max(abs(a[k] - b[k]) for k in joints) < tol_deg

    # ------------------------------------------------------------ the driver
    def attach(self, d: SpecsReach) -> SpecsReach:
        """Records what `d` sends (`sent`) and receives (`handled`), from inside the driver's threads."""
        real_send, real_on_message = d._send, d._on_message

        def send(msg: dict) -> bool:
            with self._lock:
                self.sent.append(msg)
            return real_send(msg)

        def on_message(text) -> None:
            real_on_message(text)
            try:
                msg = json.loads(text)
            except (TypeError, ValueError):
                msg = {"type": "?", "raw": text}
            with self._lock:
                self.handled.append(msg if isinstance(msg, dict) else {"type": "?", "raw": msg})

        # The bench flushes stream reports from its own thread (`flush_reports`): one at a time with the network
        # thread, and the network thread's kinematics (`_tcp_now`) under the same lock.
        real_flush, real_tcp_now = d._flush_follow_reports, d._tcp_now
        net_lock = threading.RLock()

        def flush() -> None:
            with net_lock:
                real_flush()

        def tcp_now(kin=None):
            with net_lock:
                return real_tcp_now(kin)

        d._send = send
        d._on_message = on_message
        d._flush_follow_reports = flush
        d._tcp_now = tcp_now
        self.drivers.append(d)
        return d

    def driver(self, **kw) -> SpecsReach:
        """Starts an attached `SpecsReach` (mode "sim" by default) and waits until it has the registration."""
        assert self.relay is not None
        kw.setdefault("mode", "sim")
        return self.start_driver(self.attach(SpecsReach(self.twin, relay_url=self.relay.url, **kw)))

    def start_driver(self, d: SpecsReach) -> SpecsReach:
        """Starts an attached driver and waits (real time) until it has handled the relay's registration."""
        assert self.relay is not None
        n = self._handled_count("registration")
        d.start()
        self.relay.wait_client()
        self.wait_real(lambda: self._handled_count("registration") > n, what="rejestracja u kierowcy")
        return d

    def _handled_count(self, kind: str) -> int:
        with self._lock:
            return sum(1 for m in self.handled if m.get("type") == kind)

    def _was_handled(self, kind: str, ts: int) -> bool:
        with self._lock:
            return any(m.get("type") == kind and m.get("ts") == ts for m in self.handled)

    def target(self, pos, **kw) -> int:
        """Sends a target through the relay and waits (real time) until the driver has handled it."""
        assert self.relay is not None
        ts = self.relay.target(pos, **kw)
        self.wait_real(lambda: self._was_handled("target", ts), what=f"cel {ts} u kierowcy")
        return ts

    def grip(self, action: str, **kw) -> int:
        assert self.relay is not None
        ts = self.relay.grip(action, **kw)
        self.wait_real(lambda: self._was_handled("gripper", ts), what=f"chwytak {ts} u kierowcy")
        return ts

    def follow(self, stream: int, seq: int, pos=None, **kw) -> int:
        """Sends a follow-stream sample and waits (real time) until the driver has handled it."""
        assert self.relay is not None
        ts = self.relay.follow(stream, seq, pos, **kw)
        self.wait_real(lambda: self._was_handled("follow", ts), what=f"punkt {stream}/{seq} u kierowcy")
        return ts

    def grab_object(self, pos, **kw) -> int:
        """Sends a grab and waits (real time) until the driver has handled it."""
        assert self.relay is not None
        ts = self.relay.grab_object(pos, **kw)
        self.wait_real(lambda: self._was_handled("grab", ts), what=f"chwyt {ts} u kierowcy")
        return ts

    def stream_states(self, stream: int) -> list[dict]:
        """targetState messages for a follow stream or a grab (`stream` = the grab's ts), in the order sent."""
        return [m for m in self._sent("targetState") if m.get("stream") == stream]

    def saw_stream(self, stream: int, states, reason: str | None = None) -> dict | None:
        hit = [m for m in self.stream_states(stream)
               if m["state"] in states and (reason is None or m.get("reason") == reason)]
        return hit[-1] if hit else None

    def wait_stream(self, stream: int, states, *, max_sim_s: float = 10.0, reason: str | None = None) -> dict:
        """Pumps until the driver sends a targetState for `stream` in `states` (and `reason`); returns it."""
        ok = self.pump(until=lambda: self.saw_stream(stream, states, reason), max_sim_s=max_sim_s)
        assert ok, f"brak stanu {set(states)} strumienia {stream} w {max_sim_s} s: " \
                   f"{[(m['state'], m.get('reason'), m.get('detail')) for m in self.stream_states(stream)]}"
        found = self.saw_stream(stream, states, reason)
        assert found is not None
        return found

    def _sent(self, kind: str) -> list[dict]:
        with self._lock:
            return [m for m in self.sent if m.get("type") == kind]

    def states_for(self, ts: int) -> list[str]:
        return [m["state"] for m in self._sent("targetState") if m.get("ts") == ts]

    def grips_for(self, ts: int) -> list[str]:
        return [m["state"] for m in self._sent("gripperState") if m.get("ts") == ts]

    def events(self) -> list[dict]:
        """targetState and gripperState messages in the order sent."""
        with self._lock:
            return [m for m in self.sent if m.get("type") in ("targetState", "gripperState")]

    def overlays(self) -> list[dict]:
        return self._sent("overlay")

    def tcp_overlay(self) -> dict:
        """Moves the clock one TCP-sphere period on and waits (real time) for the sphere that the driver's
        network thread then sends; returns its overlay object."""
        def spheres() -> list[dict]:
            return [o for m in self.overlays() for o in m["objects"] if o.get("id") == "tcp"]
        n = len(spheres())
        self.pump(sim_s=SpecsReach.tcp_period_s + DT)
        self.wait_real(lambda: len(spheres()) > n, what="kula TCP")
        return spheres()[-1]

    def saw(self, ts: int, states) -> dict | None:
        hit = [m for m in self._sent("targetState") if m.get("ts") == ts and m["state"] in states]
        return hit[-1] if hit else None

    def saw_grip(self, ts: int, states) -> dict | None:
        hit = [m for m in self._sent("gripperState") if m.get("ts") == ts and m["state"] in states]
        return hit[-1] if hit else None

    def wait_state(self, ts: int, states=END, *, max_sim_s: float = 10.0) -> dict:
        """Pumps until the driver sends a targetState for `ts` in `states`; returns it."""
        ok = self.pump(until=lambda: self.saw(ts, states), max_sim_s=max_sim_s)
        assert ok, f"brak stanu {set(states)} dla {ts} w {max_sim_s} s: " \
                   f"{[(m['ts'], m['state'], m.get('reason')) for m in self._sent('targetState')]}"
        found = self.saw(ts, states)
        assert found is not None
        return found

    def wait_grip(self, ts: int, states=GRIP_END, *, max_sim_s: float = 10.0) -> dict:
        ok = self.pump(until=lambda: self.saw_grip(ts, states), max_sim_s=max_sim_s)
        assert ok, f"brak stanu chwytaka {set(states)} dla {ts} w {max_sim_s} s: " \
                   f"{[(m['ts'], m['state'], m.get('reason')) for m in self._sent('gripperState')]}"
        found = self.saw_grip(ts, states)
        assert found is not None
        return found

    def assert_on_time(self) -> None:
        """Fails if a clock barrier timed out during this bench (a thread ran a tick late: see `VirtualClock`)."""
        late = self.clock.late - self._late0
        assert late == 0, (f"VirtualClock: {late} barrier(s) timed out after {self.clock.settle_s:.0f} s real - "
                           f"a thread did not park in time (see the VirtualClock warning in the log)")

    def close(self) -> None:
        """Stops the drivers, then fails if a clock barrier timed out (`assert_on_time`).

        The twin and its clock belong to whoever made them; close them after this, in that order. The drivers
        go first: their waits end on `clock.set`, while a clock closed before them releases every wait at once
        and the worker loops poll it until they see the stop flag."""
        for d in self.drivers:
            d.stop()
        self.drivers.clear()
        self.assert_on_time()


#: The user's A1X bench: a 1.2 x 0.8 m table with the base 0.45 m behind its centre (the table spans x from
#: -0.15 to 1.05 m and y +-0.40 m in the base frame).
USER_TABLE = {"size": [1.2, 0.8], "base_xy": [-0.45, 0.0]}


def ant_workspace(xy=None, table=USER_TABLE) -> Workspace:
    """An A1X workspace with the simulated ant (`ant.add_ant`: the far table end, or `xy`); no cameras."""
    ws = Workspace(robot="a1x")
    ws.table = {**ws.table, **table}
    data = add_ant(ws.to_dict(), xy)
    ws.ant = data["ant"]
    return ws

