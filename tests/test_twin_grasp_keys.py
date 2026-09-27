"""The one stdin reader: a pipe as stdin and an injected clock, no terminal needed."""

from __future__ import annotations

import io
import os
import sys
import threading
import time

import pytest

from lerobot_mp.twin.grasp.contract import CONTINUE_WORD, ENTER_GUARD_S
from lerobot_mp.twin.grasp.keys import Keys

WAIT_S = 5.0                               # generous: other test runs share the machine


class Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class Tty:
    """The read end of a pipe that says it is a terminal."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def readline(self) -> str:
        return self._stream.readline()

    def isatty(self) -> bool:
        return True


class Out(io.StringIO):
    """stdout that tells when a prompt was printed."""

    def __init__(self) -> None:
        super().__init__()
        self.printed = threading.Event()

    def write(self, s: str) -> int:
        n = super().write(s)
        if s:
            self.printed.set()
        return n


class Term:
    def __init__(self, keys: Keys, writer, clock: Clock, out: Out, monkeypatch) -> None:
        self.keys, self.writer, self.clock, self.out = keys, writer, clock, out
        self._mp = monkeypatch

    def type(self, text: str) -> None:
        self.writer.write(text)
        self.writer.flush()

    def queued(self, n: int) -> None:
        until(lambda: len(self.keys._lines) >= n)

    def call(self, fn, *args, **kwargs) -> Call:
        # Here and not in the fixture: pytest puts its own stdout back between setup and the test.
        self._mp.setattr(sys, "stdout", self.out)
        self.out.printed.clear()
        return Call(fn, *args, **kwargs)


class Call:
    """Runs a blocking Keys call on a thread, as the main thread of a run would."""

    def __init__(self, fn, *args, **kwargs) -> None:
        self.result = None
        self.thread = threading.Thread(target=self._run, args=(fn, args, kwargs), daemon=True)
        self.thread.start()

    def _run(self, fn, args, kwargs) -> None:
        self.result = fn(*args, **kwargs)

    def join(self):
        self.thread.join(WAIT_S)
        assert not self.thread.is_alive(), "the call did not return"
        return self.result


def until(cond, timeout_s: float = WAIT_S) -> None:
    end = time.monotonic() + timeout_s
    while not cond():
        assert time.monotonic() < end, "condition not reached"
        time.sleep(0.005)


@pytest.fixture
def term(monkeypatch):
    r_fd, w_fd = os.pipe()
    reader = os.fdopen(r_fd, "r", encoding="utf-8")
    writer = os.fdopen(w_fd, "w", encoding="utf-8")
    clock = Clock()
    keys = Keys(Tty(reader), clock=clock)
    keys.start()
    yield Term(keys, writer, clock, Out(), monkeypatch)
    if not writer.closed:
        writer.close()
    if keys._thread is not None:
        keys._thread.join(WAIT_S)
    reader.close()


def prompt_then(term: Term, fn, text: str, *args, **kwargs):
    call = term.call(fn, *args, **kwargs)
    assert term.out.printed.wait(WAIT_S)
    term.type(text)
    return call.join()


def test_before_arm_a_bare_enter_confirms(term):
    assert term.keys.interactive and not term.keys.armed
    assert prompt_then(term, term.keys.confirm, "\n", "Enter = start ") is True
    assert "Enter = start" in term.out.getvalue()
    assert not term.keys.stop.is_set()


def test_confirm_drops_lines_typed_before_the_prompt_and_refuses_other_text(term):
    term.type("\n")
    term.queued(1)
    assert prompt_then(term, term.keys.confirm, "no\n", "Enter = start ") is False
    assert prompt_then(term, term.keys.confirm, "\n", "Enter = start ") is True
    assert term.keys.confirm("Enter = start ", timeout_s=0.1) is False
    assert not term.keys.stop.is_set()


def test_confirm_after_arm_is_a_programming_error(term):
    term.keys.arm()
    with pytest.raises(RuntimeError):
        term.keys.confirm("Enter = start ")


def test_before_arm_a_bare_enter_does_not_stop(term):
    term.type("\n")
    term.queued(1)
    assert not term.keys.stop.is_set()


def test_after_arm_a_bare_enter_sets_stop_with_nothing_pending(term):
    term.keys.arm()
    term.type("\n")
    assert term.keys.stop.wait(WAIT_S)


def test_after_arm_a_bare_enter_sets_stop_while_a_readline_waits(term):
    term.keys.arm()
    call = term.call(term.keys.readline)
    time.sleep(0.05)
    term.type("   \n")
    assert call.join() == ""
    assert term.keys.stop.is_set()


def test_after_arm_a_bare_enter_sets_stop_while_other_lines_are_queued(term):
    term.keys.arm()
    term.type("x\ny\n")
    term.queued(2)
    assert not term.keys.stop.is_set()
    term.type("\n")
    assert term.keys.stop.wait(WAIT_S)


def test_an_enter_typed_after_the_confirm_and_before_arm_is_a_stop_at_arm(term):
    assert prompt_then(term, term.keys.confirm, "\n\n", "Enter = start ") is True
    term.queued(1)
    assert not term.keys.stop.is_set()
    term.keys.arm()
    assert term.keys.stop.is_set()


def test_proceed_goes_on_only_for_the_word_after_the_prompt_and_outside_the_guard_window(term):
    assert CONTINUE_WORD == "d" and ENTER_GUARD_S == 1.0
    term.keys.arm()
    term.keys.note_end()
    term.clock.t += ENTER_GUARD_S + 0.5
    assert prompt_then(term, term.keys.proceed, "d\n", "d + Enter = dalej ") is True
    assert not term.keys.stop.is_set()
    assert "d + Enter = dalej" in term.out.getvalue()


def test_an_enter_50_ms_after_an_ending_makes_proceed_stop(term):
    term.keys.arm()
    term.keys.note_end()
    term.clock.t += 0.05
    assert prompt_then(term, term.keys.proceed, "\n", "d + Enter = dalej ") is False
    assert term.keys.stop.is_set()


def test_the_word_inside_the_guard_window_makes_proceed_stop(term):
    term.keys.arm()
    term.keys.note_end()
    term.clock.t += ENTER_GUARD_S
    assert prompt_then(term, term.keys.proceed, "d\n", "d + Enter = dalej ") is False
    assert term.keys.stop.is_set()


def test_a_line_buffered_before_the_prompt_makes_proceed_stop(term):
    term.keys.arm()
    term.keys.note_end()
    term.clock.t += 5.0
    term.type("d\n")
    term.queued(1)
    assert term.call(term.keys.proceed, "d + Enter = dalej ").join() is False
    assert term.keys.stop.is_set()


@pytest.mark.parametrize("text", ["\n", "dalej\n", "D\n", "x\n"])
def test_anything_but_the_word_makes_proceed_stop(term, text):
    term.keys.arm()
    term.keys.note_end()
    term.clock.t += 5.0
    assert prompt_then(term, term.keys.proceed, text, "d + Enter = dalej ") is False
    assert term.keys.stop.is_set()


def test_after_a_stop_proceed_never_goes_on(term):
    term.keys.arm()
    term.type("\n")
    assert term.keys.stop.wait(WAIT_S)
    assert term.keys.readline(timeout_s=WAIT_S) == ""
    term.keys.note_end()
    term.clock.t += 5.0
    call = term.call(term.keys.proceed, "d + Enter = dalej ", timeout_s=WAIT_S)
    term.type("d\n")
    assert call.join() is False
    assert term.keys.stop.is_set()


def test_a_timeout_or_end_of_file_makes_proceed_stop(term):
    term.keys.arm()
    assert term.keys.proceed("d + Enter = dalej ", timeout_s=0.1) is False
    assert term.keys.stop.is_set()


def test_end_of_file_makes_proceed_stop(term):
    term.keys.arm()
    term.clock.t += 5.0
    call = term.call(term.keys.proceed, "d + Enter = dalej ")
    assert term.out.printed.wait(WAIT_S)
    term.writer.close()
    assert call.join() is False
    assert term.keys.stop.is_set()


def test_readline_returns_lines_in_order_and_none_on_timeout(term):
    term.type("up\n  down \n\nsave\n")
    assert [term.keys.readline(timeout_s=WAIT_S) for _ in range(4)] == ["up", "down", "", "save"]
    assert term.keys.readline(timeout_s=0.1) is None
    assert not term.keys.stop.is_set()                    # not armed: an empty line is only a line


def test_readline_returns_none_at_end_of_file_after_the_queued_lines(term):
    term.type("last\n")
    term.writer.close()
    assert term.keys.readline(timeout_s=WAIT_S) == "last"
    assert term.keys.readline(timeout_s=WAIT_S) is None


def test_end_of_file_makes_interactive_false(term):
    assert term.keys.interactive
    term.writer.close()
    until(lambda: not term.keys.interactive)
    assert term.keys.confirm("Enter = start ") is False
    assert not term.keys.stop.is_set()


def test_without_a_terminal_confirm_and_proceed_return_false_at_once(monkeypatch):
    r_fd, w_fd = os.pipe()
    reader = os.fdopen(r_fd, "r", encoding="utf-8")
    writer = os.fdopen(w_fd, "w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", Out())
    try:
        keys = Keys(reader, clock=Clock())
        keys.start()
        assert not keys.interactive
        assert Call(keys.confirm, "Enter = start ").join() is False
        assert not keys.stop.is_set()
        keys.arm()
        assert Call(keys.proceed, "d + Enter = dalej ").join() is False
        assert keys.stop.is_set()
        writer.write("scripted\n")
        writer.flush()
        assert keys.readline(timeout_s=WAIT_S) == "scripted"      # bench line commands still read
    finally:
        writer.close()
        if keys._thread is not None:
            keys._thread.join(WAIT_S)
        reader.close()


def test_the_default_stdin_is_the_one_in_place_when_keys_is_built(monkeypatch):
    r_fd, w_fd = os.pipe()
    reader = os.fdopen(r_fd, "r", encoding="utf-8")
    writer = os.fdopen(w_fd, "w", encoding="utf-8")
    try:
        monkeypatch.setattr(sys, "stdin", Tty(reader))
        assert Keys().interactive
        monkeypatch.setattr(sys, "stdin", None)
        assert not Keys().interactive
    finally:
        writer.close()
        reader.close()
