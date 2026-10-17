"""Test bench: a twin stepped on a `VirtualClock`.

The twin runs without a loop thread (`connect(threaded=False)`) on a `VirtualClock`. `Bench.pump` moves
time one loop tick at a time, and `Bench.call` runs a blocking call (`Twin.move`) in a thread meanwhile.
Nothing waits in wall time, so ten simulated seconds of motion cost a fraction of a second and the
timings repeat to the tick.

    bench = Bench.sim("testbot")
    bench.call(lambda: bench.twin.move(goal, 2.0, owner="test"))
    bench.pump(until=lambda: bench.twin.owner is None, max_sim_s=10.0)
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

from machinekind_twin import runtime as rt
from machinekind_twin.clock import VirtualClock
from machinekind_twin.workspace import Workspace

DT = 0.02                                      # twin loop period, 50 Hz


class Bench:
    """A stepped twin on a `VirtualClock`."""

    #: A pump that takes longer than this in real time fails instead of hanging the suite.
    real_timeout_s = 60.0

    def __init__(self, twin: rt.Twin):
        assert isinstance(twin.clock, VirtualClock), "Bench needs a twin on a VirtualClock (connect(threaded=False))"
        self.twin = twin
        self.clock: VirtualClock = twin.clock
        # Barrier timeouts before this bench (a module-scoped clock outlives many benches).
        self._late0 = self.clock.late

    @classmethod
    def sim(cls, robot: str = "testbot", *, t0: float = 100.0, ws: Workspace | None = None) -> Bench:
        """A simulated `robot` (or workspace `ws`) on a fresh `VirtualClock` starting at `t0` s."""
        tw = rt.Twin(ws if ws is not None else Workspace(robot=robot), clock=VirtualClock(t0=t0))
        tw.connect("sim", threaded=False)
        return cls(tw)

    # ------------------------------------------------------------ time
    def step(self, dt: float = DT) -> None:
        """One loop tick of `dt` (the clock first, then the loop)."""
        self.twin.step(dt)

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

    def call(self, fn: Callable[[], object], *, max_sim_s: float = 30.0):
        """Runs a blocking call (`Twin.move`) in a thread while pumping; returns its result.

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

    def assert_on_time(self) -> None:
        """Fails if a clock barrier timed out during this bench (a thread ran a tick late: see `VirtualClock`)."""
        late = self.clock.late - self._late0
        assert late == 0, (f"VirtualClock: {late} barrier(s) timed out after {self.clock.settle_s:.0f} s real - "
                           f"a thread did not park in time (see the VirtualClock warning in the log)")
