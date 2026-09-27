"""One time axis for the twin's loop and the threads that drive it.

Production runs on `WallClock`: `time.monotonic`, real sleeps, real waits. Tests step the loop with
`Twin.step` on a `VirtualClock`, where time moves only in `advance(dt)`. The twin's loop, the
Spectacles driver's worker (ramps, arrival and gripper waits, the backoff grace) and the network
thread's timers then all read the same simulated seconds, and ten seconds of motion take a fraction
of a second of CPU.

Code that waits for time goes through the clock: `sleep`, `wait(event, timeout)`, `get(queue, timeout)`.
Code that wakes a waiter goes through it too: `set(event)`, `put(queue, item)`, so a virtual waiter
sees the change at once instead of at its next poll.

`VirtualClock.advance` returns only after every thread it woke has run up to its next wait. A thread
that joins with `member()` (the driver's worker) counts as running until it is back in a clock wait,
so its computation (IK, collision checks) takes no simulated time and a ramp tick always lands before
the loop tick that follows it, as it does at 50 Hz on the real loop.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import ContextManager, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Clock(Protocol):
    """What the twin and the driver need from time."""

    def now(self) -> float: ...

    def sleep(self, secs: float) -> None: ...

    def wait(self, event: threading.Event, timeout: float | None) -> bool:
        """`event.wait(timeout)`: True if the event is set."""
        ...

    def get(self, q: queue.Queue[T], timeout: float) -> T:
        """`q.get(timeout=timeout)`; raises `queue.Empty` on timeout (a virtual clock may also raise it early)."""
        ...

    def set(self, event: threading.Event) -> None:
        """`event.set()` and wake whoever waits for it."""
        ...

    def put(self, q: queue.Queue[T], item: T) -> None:
        """`q.put(item)` and wake whoever waits for it."""
        ...

    def member(self) -> ContextManager[None]:
        """The calling thread takes part in `VirtualClock.advance` for the duration of the block."""
        ...


class WallClock:
    """Real time: `time.monotonic`, `time.sleep`, `Event.wait`, `Queue.get`."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, secs: float) -> None:
        time.sleep(max(0.0, secs))

    def wait(self, event: threading.Event, timeout: float | None) -> bool:
        return event.wait(timeout)

    def get(self, q: queue.Queue[T], timeout: float) -> T:
        return q.get(timeout=timeout)

    def set(self, event: threading.Event) -> None:
        event.set()

    def put(self, q: queue.Queue[T], item: T) -> None:
        q.put(item)

    def member(self) -> ContextManager[None]:
        return nullcontext()


class VirtualClock:
    """Simulated time that moves only in `advance(dt)`, called by one thread (the test's pump).

    Waiters register a deadline. `advance` moves time, wakes every waiter whose deadline passed and
    waits (in real time, at most `settle_s`) until each has left its wait and every `member()` thread
    is parked in a new one. A barrier that times out is logged and counted in `late`: the thread
    then runs a tick late, so tests assert timings with a tolerance of a few ticks.

    `close()` releases every waiter at once (test teardown).
    """

    #: Real poll inside a wait: bounds the latency of an `Event.set()` that bypassed `set()`.
    poll_s = 0.005

    def __init__(self, t0: float = 100.0, *, settle_s: float = 10.0):
        self._t = float(t0)
        self.settle_s = float(settle_s)
        #: Barrier timeouts so far (a thread that did not park within `settle_s`).
        self.late = 0
        self._cond = threading.Condition()
        # Thread id -> (deadline, "about to wake" predicate).
        self._waiters: dict[int, tuple[float, Callable[[], bool]]] = {}
        # Thread id -> nesting depth of `member()`.
        self._members: dict[int, int] = {}
        # Bumped by `set`/`put`/`close`: a `get` waiter returns (queue.Empty) so its caller rechecks its flags.
        self._wakeups = 0
        self._advancer: int | None = None
        self._closed = False

    def now(self) -> float:
        return self._t

    # ------------------------------------------------------------ waiting
    def sleep(self, secs: float) -> None:
        self.wait(threading.Event(), secs)

    def wait(self, event: threading.Event, timeout: float | None) -> bool:
        with self._cond:
            deadline = math.inf if timeout is None else self._t + max(0.0, float(timeout))
            if event.is_set() or self._t >= deadline or self._closed:
                return event.is_set()
            with self._parked(deadline, event.is_set):
                while not event.is_set() and self._t < deadline and not self._closed:
                    self._cond.wait(self.poll_s)
            return event.is_set()

    def get(self, q: queue.Queue[T], timeout: float) -> T:
        with self._cond:
            deadline = self._t + max(0.0, float(timeout))
            wakeups = self._wakeups

            def ready() -> bool:
                return not q.empty() or self._wakeups != wakeups

            try:
                return q.get_nowait()
            except queue.Empty:
                if self._t >= deadline or self._closed:
                    raise
            with self._parked(deadline, ready):
                while True:
                    try:
                        return q.get_nowait()
                    except queue.Empty:
                        if self._t >= deadline or self._closed or self._wakeups != wakeups:
                            raise
                    self._cond.wait(self.poll_s)

    @contextmanager
    def _parked(self, deadline: float, ready: Callable[[], bool]) -> Iterator[None]:
        """Registers the calling thread as a waiter; the caller holds `_cond`."""
        me = threading.get_ident()
        if me == self._advancer:
            raise RuntimeError("VirtualClock: the thread that advances time would wait for it - "
                               "run this call in another thread and pump the clock")
        self._waiters[me] = (deadline, ready)
        self._cond.notify_all()
        try:
            yield
        finally:
            del self._waiters[me]
            self._cond.notify_all()

    # ------------------------------------------------------------ waking
    def set(self, event: threading.Event) -> None:
        event.set()
        with self._cond:
            self._wakeups += 1
            self._cond.notify_all()

    def put(self, q: queue.Queue[T], item: T) -> None:
        q.put(item)
        with self._cond:
            self._wakeups += 1
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._wakeups += 1
            self._cond.notify_all()

    # ------------------------------------------------------------ members and time
    @contextmanager
    def member(self) -> Iterator[None]:
        me = threading.get_ident()
        with self._cond:
            self._members[me] = self._members.get(me, 0) + 1
        try:
            yield
        finally:
            with self._cond:
                depth = self._members[me] - 1
                if depth:
                    self._members[me] = depth
                else:
                    del self._members[me]
                self._cond.notify_all()

    def waiters(self) -> int:
        """How many threads wait on this clock right now."""
        with self._cond:
            return len(self._waiters)

    def _settled(self) -> bool:
        if self._closed:
            return True
        for deadline, ready in self._waiters.values():
            if deadline <= self._t or ready():
                return False                            # woken, still has to leave its wait
        return all(m in self._waiters for m in self._members)

    def advance(self, dt: float) -> None:
        """Moves time by `dt` and returns once the threads it woke have parked again."""
        if dt < 0:
            raise ValueError(f"VirtualClock.advance: dt {dt} < 0")
        with self._cond:
            self._advancer = threading.get_ident()
            self._t += float(dt)
            self._cond.notify_all()
            if not self._cond.wait_for(self._settled, timeout=self.settle_s):
                self.late += 1
                woken = [f"{i} woken" for i, (d, ready) in self._waiters.items() if d <= self._t or ready()]
                running = [f"{m} running" for m in self._members if m not in self._waiters]
                busy = woken + running
                logger.warning("VirtualClock t=%.3f: threads not parked after %.1f s real: %s",
                               self._t, self.settle_s, ", ".join(busy) or "?")
