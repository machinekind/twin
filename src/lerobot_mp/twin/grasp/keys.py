"""The one stdin reader of the grasp commands.

    keys = Keys(); keys.start()                    # before connecting
    keys.confirm("Enter = ... ")                   # only before the first motion
    keys.arm()                                     # right before the first motion
    ...                                            # a bare Enter now sets `keys.stop`, always
    keys.note_end()                                # an episode has just ended
    keys.proceed("d + Enter = ... ")               # the only way to go on

Episodes end by themselves, so an Enter meant as STOP can arrive just after an ending. Once armed
there is therefore no state in which a bare Enter means "go on": going on needs a typed word,
typed after the prompt and more than `ENTER_GUARD_S` after the ending. Everything else stops.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import TextIO

from .contract import CONTINUE_WORD, ENTER_GUARD_S

logger = logging.getLogger(__name__)

_POLL_S = 0.1                                  # waits wake up this often, so Ctrl-C gets through


def _isatty(stream: TextIO | None) -> bool:
    try:
        return stream is not None and bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


class Keys:
    """One stdin reader for the whole command. Once armed, a bare Enter is always STOP."""

    def __init__(self, stdin: TextIO | None = None, *, clock: Callable[[], float] = time.monotonic) -> None:
        # None = `sys.stdin` as it is now, not as it was at import (tests and callers replace it).
        stdin = sys.stdin if stdin is None else stdin
        self._stdin = stdin
        self._clock = clock
        self._tty = _isatty(stdin)
        self._cond = threading.Condition()
        self._lines: deque[tuple[float, str]] = deque()      # (arrival time, stripped line)
        self._eof = stdin is None
        self._armed = False
        self._t_end: float | None = None
        self._thread: threading.Thread | None = None
        self.stop = threading.Event()

    @property
    def interactive(self) -> bool:
        """stdin is a TTY and has not reached end of file."""
        with self._cond:
            return self._tty and not self._eof

    @property
    def armed(self) -> bool:
        with self._cond:
            return self._armed

    def start(self) -> None:
        """Starts the reader thread (daemon); every line is queued with its arrival time."""
        with self._cond:
            if self._thread is not None or self._eof:
                return
            self._thread = threading.Thread(target=self._read, name="chwyt-klawiatura", daemon=True)
        self._thread.start()

    def arm(self) -> None:
        """From now on a bare Enter sets `stop`; call right before the first motion."""
        with self._cond:
            self._armed = True
            # An Enter typed after the last prompt was answered is already a STOP.
            if any(text == "" for _, text in self._lines):
                self.stop.set()

    def note_end(self) -> None:
        """An episode or a move has just ended: input within `ENTER_GUARD_S` from now is a STOP."""
        with self._cond:
            self._t_end = self._clock()

    def confirm(self, prompt: str, timeout_s: float | None = None) -> bool:
        """Before the first motion only: True on a bare Enter typed after the prompt."""
        if self.armed:
            raise RuntimeError(
                "Keys.confirm after arm(): only proceed() may continue once motion has started")
        if not self.interactive:
            return False
        self.start()
        with self._cond:
            self._lines.clear()
        self._print(prompt)
        t_prompt = self._clock()
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            got = self._next(deadline)
            if got is None:
                return False
            t, text = got
            if t < t_prompt:
                continue
            return text == ""

    def proceed(self, prompt: str, word: str = CONTINUE_WORD, timeout_s: float | None = None) -> bool:
        """True only for `word` + Enter typed after the prompt and outside the guard window.

        Anything else (a line already queued, a line inside the guard window, a bare Enter, other
        text, a timeout, end of file, no TTY) sets `stop` and returns False. So does a STOP given
        earlier: `stop` is never cleared, so nothing goes on after it.
        """
        if self.stop.is_set():
            return self._refuse("STOP already given")
        if not self.interactive:
            return self._refuse("stdin is not interactive")
        self.start()
        with self._cond:
            if self._lines:
                self._lines.clear()
                return self._refuse("input typed before the prompt")
        self._print(prompt)
        t_prompt = self._clock()
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        got = self._next(deadline)
        if got is None:
            return self._refuse("no answer")
        t, text = got
        with self._cond:
            t_end = self._t_end
        if t < t_prompt:
            return self._refuse("input typed before the prompt")
        if t_end is not None and t - t_end <= ENTER_GUARD_S:
            return self._refuse("input within the guard window after an ending")
        if text != word:
            return self._refuse(f"answer {text!r} is not {word!r}")
        return True

    def readline(self, timeout_s: float | None = None) -> str | None:
        """The next line, stripped; None on timeout or end of file. An empty line is a STOP when armed."""
        self.start()
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        got = self._next(deadline)
        if got is None:
            return None
        text = got[1]
        if text == "" and self.armed:
            self.stop.set()
        return text

    # ------------------------------------------------------------------ internals
    def _refuse(self, why: str) -> bool:
        logger.info("keys: STOP (%s)", why)
        self.stop.set()
        return False

    @staticmethod
    def _print(prompt: str) -> None:
        out = sys.stdout
        if out is not None:
            out.write(prompt)
            out.flush()

    def _next(self, deadline: float | None) -> tuple[float, str] | None:
        """The oldest queued line; None at the deadline, or at end of file with nothing queued."""
        with self._cond:
            while not self._lines:
                if self._eof:
                    return None
                wait = _POLL_S
                if deadline is not None:
                    left = deadline - time.monotonic()
                    if left <= 0.0:
                        return None
                    wait = min(wait, left)
                self._cond.wait(wait)
            return self._lines.popleft()

    def _read(self) -> None:
        stream = self._stdin
        while True:
            try:
                line = stream.readline() if stream is not None else ""
            except (OSError, ValueError, UnicodeDecodeError):
                line = ""
            if line == "":
                with self._cond:
                    self._eof = True
                    self._cond.notify_all()
                return
            text = line.strip()
            with self._cond:
                if text == "" and self._armed:
                    self.stop.set()
                self._lines.append((self._clock(), text))
                self._cond.notify_all()
