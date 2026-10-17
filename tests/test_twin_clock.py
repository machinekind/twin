"""`clock`: the wall clock of production and the virtual clock of the tests."""

from __future__ import annotations

import queue
import threading
import time
from contextlib import nullcontext

import pytest

from machinekind_twin.clock import VirtualClock, WallClock


def spawn(clock: VirtualClock, fn, *, member: bool = True) -> tuple[threading.Thread, list]:
    """Runs `fn()` in a thread (a clock member by default); its result or exception lands in the list."""
    out: list = []

    def run():
        with clock.member() if member else nullcontext():
            try:
                out.append(fn())
            except BaseException as exc:                        # noqa: BLE001 - reported to the test
                out.append(exc)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th, out


def parked(clock: VirtualClock, n: int = 1, timeout: float = 2.0) -> None:
    """Real wait until `n` threads wait on the clock."""
    deadline = time.monotonic() + timeout
    while clock.waiters() < n:
        assert time.monotonic() < deadline, "thread never started waiting"
        time.sleep(0.001)


def test_wait_returns_at_its_deadline_and_advance_waits_for_the_woken_member():
    clock = VirtualClock(t0=10.0)
    ev = threading.Event()
    th, out = spawn(clock, lambda: (clock.wait(ev, 0.5), clock.now()))
    parked(clock)
    clock.advance(0.3)
    assert out == [] and th.is_alive()
    clock.advance(0.2)
    # A member counts as running until it parks again or leaves: by now it has finished.
    assert out == [(False, pytest.approx(10.5))]
    th.join(1.0)


def test_set_wakes_a_waiter_without_time_moving():
    clock = VirtualClock()
    ev = threading.Event()
    th, out = spawn(clock, lambda: clock.wait(ev, None))
    parked(clock)
    t0 = clock.now()
    clock.set(ev)
    th.join(1.0)
    assert out == [True] and clock.now() == t0


def test_get_takes_an_item_put_by_another_thread_and_times_out_in_virtual_time():
    clock = VirtualClock()
    q: queue.Queue = queue.Queue()
    th, out = spawn(clock, lambda: clock.get(q, 1.0))
    parked(clock)
    q.put("bare")                                               # not through the clock: found by the poll
    th.join(1.0)
    assert out == ["bare"]

    th, out = spawn(clock, lambda: clock.get(q, 1.0))
    parked(clock)
    clock.put(q, "via clock")
    th.join(1.0)
    assert out == ["via clock"]

    th, out = spawn(clock, lambda: clock.get(q, 1.0))
    parked(clock)
    clock.advance(0.99)
    assert out == []
    clock.advance(0.01)
    assert len(out) == 1 and isinstance(out[0], queue.Empty)
    th.join(1.0)


def test_set_of_any_event_makes_a_waiting_get_return_so_its_caller_rechecks_its_flags():
    clock = VirtualClock()
    q: queue.Queue = queue.Queue()
    th, out = spawn(clock, lambda: clock.get(q, 100.0))
    parked(clock)
    clock.set(threading.Event())                                # e.g. `stop()` setting the driver's stop flag
    th.join(1.0)
    assert len(out) == 1 and isinstance(out[0], queue.Empty)


def test_close_releases_every_waiter():
    clock = VirtualClock()
    ev = threading.Event()
    q: queue.Queue = queue.Queue()
    a, out_a = spawn(clock, lambda: clock.wait(ev, None))
    b, out_b = spawn(clock, lambda: clock.get(q, 100.0), member=False)
    parked(clock, 2)
    clock.close()
    a.join(1.0)
    b.join(1.0)
    assert out_a == [False] and isinstance(out_b[0], queue.Empty)
    assert clock.wait(ev, 5.0) is False                         # closed: no wait blocks any more


def test_a_closed_clock_naps_before_returning_empty_handed():
    """A worker loop still running after a mis-ordered teardown polls a closed clock, it does not spin."""
    clock = VirtualClock()
    clock.close()
    q: queue.Queue = queue.Queue()
    t0 = time.monotonic()
    for _ in range(4):
        with pytest.raises(queue.Empty):
            clock.get(q, 1.0)
        assert clock.wait(threading.Event(), 1.0) is False
    assert time.monotonic() - t0 >= 8 * clock.closed_nap_s * 0.9
    q.put(1)
    ev = threading.Event()
    ev.set()
    assert clock.get(q, 1.0) == 1 and clock.wait(ev, 1.0) is True  # something there: returned as usual


def test_advance_waits_for_a_members_computation_before_returning():
    clock = VirtualClock()
    ev = threading.Event()
    marks: list[str] = []

    def worker():
        clock.wait(ev, 0.02)
        time.sleep(0.05)                                        # IK: real CPU time, no simulated time
        marks.append("computed")
        clock.wait(ev, 0.02)
        marks.append("second tick")
    th, _ = spawn(clock, worker)
    parked(clock)
    t0 = time.monotonic()
    clock.advance(0.02)
    assert marks == ["computed"]
    assert time.monotonic() - t0 >= 0.045
    clock.advance(0.02)
    assert marks == ["computed", "second tick"]
    th.join(1.0)


def test_a_member_woken_by_a_message_finishes_before_time_moves():
    clock = VirtualClock(t0=0.0)
    q: queue.Queue = queue.Queue()

    def worker():
        clock.get(q, 100.0)
        time.sleep(0.03)                                        # planning the target
        return clock.now()                                      # the moment it acts on it
    th, out = spawn(clock, worker)
    parked(clock)
    clock.put(q, "target")
    clock.advance(0.02)
    th.join(1.0)
    assert out == [0.0]


def test_a_member_that_does_not_park_in_time_is_counted_late():
    clock = VirtualClock(settle_s=0.05)
    ev = threading.Event()

    def slow():
        clock.wait(ev, 0.02)
        time.sleep(0.3)
    th, _ = spawn(clock, slow)
    parked(clock)
    t0 = time.monotonic()
    clock.advance(0.02)
    assert time.monotonic() - t0 < 0.25
    assert clock.late == 1
    th.join(1.0)


def test_the_thread_that_advances_time_cannot_wait_for_it():
    clock = VirtualClock()
    clock.advance(0.02)
    with pytest.raises(RuntimeError, match="advances time"):
        clock.wait(threading.Event(), 1.0)
    assert clock.wait(threading.Event(), 0.0) is False          # nothing to wait for: fine
    with pytest.raises(ValueError):
        clock.advance(-0.1)


def test_wall_clock_is_plain_real_time():
    clock = WallClock()
    t0 = clock.now()
    ev = threading.Event()
    assert clock.wait(ev, 0.01) is False
    clock.set(ev)
    assert clock.wait(ev, 0.0) is True
    q: queue.Queue = queue.Queue()
    clock.put(q, 1)
    assert clock.get(q, 0.01) == 1
    with pytest.raises(queue.Empty):
        clock.get(q, 0.01)
    with clock.member():
        clock.sleep(0.001)
    assert clock.now() - t0 >= 0.015
