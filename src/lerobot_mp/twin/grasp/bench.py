"""Bench measurements for the grasp student: jaw, table, grasp point, latency, dot, and a summary.

    lerobot-twin grasp-bench gripper --backend a1x --i-know        # jaw stroke and effort at 200 Hz
    lerobot-twin grasp-bench gripper --set-freeze 100              # record the freeze decision
    lerobot-twin grasp-bench gripper --gap-mm 0:3 --gap-mm 100:103 # ruler readings of the tip gap
    lerobot-twin grasp-bench table --backend a1x --i-know --xy 0 0.33      # fingertips on the table
    lerobot-twin grasp-bench point --backend a1x --i-know --xy 0 0.33 --yaw-deg 0
    lerobot-twin grasp-bench latency --backend a1x --i-know        # arm and camera latency at the hover
    lerobot-twin grasp-bench dot --error-px 3.5                    # the dot error read from the dry run dump
    lerobot-twin grasp-bench show                                  # the per-rig file, what each mode misses

Every tool that moves the arm takes it through `motion.ArmSession`: one owner (`OWNER`), the
clutch, the tick listener, the impact guard, the driver's acceptance rule, pre-positioning and
the hold. There is no free drive on the A1X, so teaching is jogging under position control: small
Cartesian steps of a top-down tool, each one `ik_step`, ramped through `Twin.set_target`. The arm
does not yield on contact, so three rules watch every step (measured height, joint lag, effort)
and the first to fire holds the measured pose and backs off 2 mm.

Every wait, ramp and timeout goes through the session's clock and sleep. A bare Enter, once the
arm is armed, and Ctrl-C both STOP through `motion.stop_now`. Nothing here opens the CAN link or a
camera device directly, reads stdin except through `Keys`, or writes `workspace/twin.json`.
Values measured on the simulated arm reach a per-rig file only when `--rig` names it.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...paths import data_path
from .. import scene as sc
from ..kinematics import RobotKinematics
from ..runtime import Twin, smoothstep
from .camera import (
    CameraError,
    FrameSource,
    LiveWristSource,
    SimWristSource,
    WristCamera,
    wrist_intrinsics,
)
from .contract import (
    DOT_MAX_PX,
    FOVY_DEG,
    FREEZE_DISABLED,
    GAP_CLOSED,
    GAP_PER_UNIT,
    HOVER,
    IMG_H,
    IMG_W,
    JOG_BACKOFF,
    JOG_CONTACT_MIN_DROP,
    JOG_LAG_DEG,
    JOG_SLOW_BELOW_Z,
    LATENCY_MAX_S,
    OBJECT_HEIGHT_DEFAULT,
    OWNER,
    PREPOSITION_SPEED_DEG_S,
    SAG_LIMIT,
    T_GRIPPER_CAMERA,
    TABLE_TOUCH_FAR,
    TEACH_MAX_AGE_S,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
    TOUCH_TOLERANCE,
    grasp_R,
    wrap,
    yaw_of,
)
from .hover import Hover, HoverError, from_rig, from_scene, solve_hover
from .keys import Keys
from .motion import ArmSession, Ended, enable_refusal, load_workspace, stop_now
from .rig import OBJECT_HEIGHT_RANGE, Rig, RigError

logger = logging.getLogger(__name__)

#: Raw logs of the jaw measurement (`gripper_<stamp>.npz`), resolved with `data_path`.
BENCH_DIR = "workspace/grasp/bench"

TOOLS = ("gripper", "table", "point", "latency", "dot", "show")
STOP_REASON = "STOP z klawiatury (pomiar chwytu)"
CTRL_C_REASON = "Ctrl-C (pomiar chwytu)"

# ---- jaw
#: 99 first: on the A1X a first command within 0.5 of the reported opening (100) sends nothing.
GRIPPER_FREE_MOVES = (99.0, 100.0, 0.0, 100.0, 0.0, 100.0)
GRIPPER_LOADED_MOVES = (0.0, 100.0)
GRIPPER_SETTLE_S = 2.0                # after each command; the A1X stream needs 1.3 s for a full stroke
PRE_MOVE_S = 0.5                      # logged before each command: the noise band of its effort
MIN_RATE_HZ = 150.0
NOISE_SIGMAS = 3.0
EFFORT_NOISE_MIN = 0.05               # backend effort units; a flat trace would give a zero band
STROKE_MIN_UNITS = 50.0               # a move shorter than this is not a stroke
STROKE_EDGE_UNITS = 1.0               # start and end of a stroke: this far from its end values
SEPARABLE_RATIO = 1.5
GAP_WARN_MM = 3.0
OPERATOR_MIN_S = 0.05                 # an operator line sooner than this after a command was typed before it

# ---- jog
JOG_STEP_M = 0.005
JOG_SLOW_STEP_M = 0.001               # `d` below JOG_SLOW_BELOW_Z
JOG_STEP_YAW = math.radians(5.0)
JOG_SPEED_DEG_S = 10.0
JOG_RAMP_MIN_S = 0.2
JOG_RAMP_DT = 0.02
JOG_SETTLE_S = 0.3
JOG_POLL_S = 0.01
#: Below this commanded TCP z the collision checker would refuse the touch pose itself.
JOG_COLLISION_ABOVE_Z = 0.04
JOG_GUARD_LAG_DEG, JOG_GUARD_LAG_S = 3.0, 0.15
MOVE_GUARD_LAG_DEG = 4.0
GRIP_WAIT_S = 2.0
START_Z_DEFAULT = 0.12
START_XY_DEFAULT = (0.0, 0.33)        # training's region: the arm's left, about 0.33 m out

# ---- latency
LATENCY_STEPS = 5
LATENCY_STEP_DEG = 3.0
LATENCY_MOVED_DEG = 0.2
LATENCY_SIGMAS = 4.0
LATENCY_STILL_FRAMES = 8
LATENCY_STILL_S = 1.0
LATENCY_STEP_TIMEOUT_S = 1.5
LATENCY_SETTLE_S = 0.6
LATENCY_POLL_S = 0.005
LATENCY_MIN_GOOD = 3
#: Grey levels. Identical still renders (sim) give sigma 0; a larger floor hides a slow start of
#: the motion in a flat scene (the twin's: 3 deg of J6 change the mean by only 1.1 grey levels).
MAD_SIGMA_MIN = 0.01
FIRST_FRAME_S = 3.0

#: The simulated bench: the table of the sim evaluation (under training's whole region) and the
#: wrist camera. No cube: nothing here grasps.
SIM_TABLE = sc.Table(size=(1.2, 1.2), base_xy=(0.0, -0.20), base_yaw=0.0)
WRIST_CAMERA = "wrist"


def configure_sim(twin: Twin) -> None:
    """The simulated bench scene: a table under the grasp region and the hand camera."""
    twin.configure(table=SIM_TABLE,
                   hand_cameras=[sc.HandCamera(WRIST_CAMERA, T_GRIPPER_CAMERA, FOVY_DEG, IMG_W, IMG_H)])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _refuse(reason: str, hint: str = "") -> int:
    print(f"Odmowa: {reason}" + (f" - {hint}" if hint else ""), file=sys.stderr, flush=True)
    return 2


#: Endings of taking or pre-positioning the arm that are failures (exit 1), not refusals (exit 2).
FAILED = ("stop", "preempted", "not connected", "error", "non-finite")


def _not_started(why: str, detail: str, keys: Keys) -> int:
    """Exit code and message when `check`, `take` or pre-positioning did not hand over the arm."""
    if keys.stop.is_set() or why in FAILED or why.startswith("impact"):
        print(f"Koniec: {why}" + (f" ({detail})" if detail else "") + " - ramie trzyma poze.",
              file=sys.stderr, flush=True)
        return 1
    return _refuse(why, detail)


# ====================================================================== the connected bench
@dataclass
class Bench:
    """A connected twin and what every tool needs next to it."""

    twin: Twin
    rig: Rig
    kin: RobotKinematics
    simulated: bool
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] | None = None     # None: wait on the session's cancel event
    may_write: bool = True                           # sim: only into a file named with --rig
    rig_path: Path | None = None

    def session(self) -> ArmSession:
        return ArmSession(self.twin, self.kin, self.rig, clock=self.clock, sleep=self.sleep)


def _open(a: argparse.Namespace, *, moves: bool) -> Bench | int:
    """Flag checks, workspace, per-rig file and the refusals that come BEFORE connecting; then connect."""
    tool = getattr(a, "action", "?")
    backend = str(getattr(a, "backend", None) or "sim").lower()
    if moves and backend != "sim" and not getattr(a, "i_know", False):
        print(f"lerobot-twin grasp-bench {tool} --backend {backend} rusza prawdziwym ramieniem - "
              f"dodaj --i-know.", file=sys.stderr)
        return 2
    try:
        ws = load_workspace(getattr(a, "workspace", None))
    except (OSError, ValueError, TypeError) as exc:
        print(f"Nie wczytano pliku stanowiska: {exc}", file=sys.stderr)
        return 2
    if ws is None:
        return _refuse("not a1x", "plik stanowiska opisuje inne ramie; narzedzia chwytu dzialaja tylko "
                                  "na A1X")
    rig_arg = getattr(a, "rig", None)
    try:
        rig = Rig.load(rig_arg)
    except RigError as exc:
        print(f"Plik wartosci stanowiska chwytu jest bledny: {exc}", file=sys.stderr)
        return 2
    port = getattr(a, "port", None)
    why = enable_refusal(ws, backend, port)
    if why is not None:
        return _refuse(why, "a1x.enable_on_connect wlacza silniki przy polaczeniu (na chwile je puszcza); "
                            "wlacz ramie z panelu i ustaw enable_on_connect: false")
    twin = Twin(ws)
    try:
        twin.connect(backend, port=port, go_home=False, threaded=True)
        simulated = bool(twin.status.simulated)
        if simulated:
            configure_sim(twin)
    except KeyboardInterrupt:
        if twin.connected:
            stop_now(twin, CTRL_C_REASON)
        twin.close()
        raise
    except Exception as exc:
        print(f"Nie polaczono z ramieniem: {exc}", file=sys.stderr)
        twin.close()
        return 1
    st = twin.status
    print(f"Ramie: {st.backend} ({'symulacja' if simulated else 'prawdziwe'}), narzedzie {tool}", flush=True)
    return Bench(twin, rig, RobotKinematics(twin.spec), simulated,
                 may_write=not simulated or rig_arg is not None, rig_path=rig.path)


def _update_rig(path: Path | None, section: str, values: Mapping[str, Any], *,
                overrides: Mapping[str, Any] | None = None) -> Rig:
    """Re-reads the file (the operator may have edited it meanwhile), merges one section, saves."""
    rig = Rig.load(path)
    getattr(rig, section).update(values)
    if overrides:
        rig.overrides.update(overrides)
    rig.save()
    return rig


class _StopWatch:
    """Turns the keys' STOP (a bare Enter once armed, or a refused `proceed`) into `stop_now`."""

    def __init__(self, keys: Keys, twin: Twin) -> None:
        self.keys, self.twin = keys, twin
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._fired = False
        self._thread = threading.Thread(target=self._run, name="chwyt-pomiar-stop", daemon=True)

    def start(self) -> _StopWatch:
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._done.is_set():
            if self.keys.stop.wait(0.05):
                self.fire()
                return

    def fire(self) -> None:
        with self._lock:
            if self._fired:
                return
            self._fired = True
        print("\nSTOP.", flush=True)
        stop_now(self.twin, STOP_REASON)

    def close(self) -> None:
        """A STOP typed just before the tool ended is still carried out."""
        self._done.set()
        if self.keys.stop.is_set():
            self.fire()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(1.0)


def _release(session: ArmSession | None) -> None:
    """Holds (the jaw where the stream is) and releases; after a STOP the twin holds already.

    An impact that ended the tool holds the measured pose, so a blocked joint stops pushing.
    """
    if session is None:
        return
    ending = session.ending_of_cause() if session.cancel.is_set() else "stop"
    try:
        session.hold(ending)
    except Exception:
        logger.exception("hold at the end of a bench tool failed")
    try:
        session.close()
    except Exception:
        logger.exception("closing the arm session failed")


def _session_sleep(session: ArmSession) -> Callable[[float], None]:
    """The session's sleep (its injected one, else a wait on its cancel event), as a plain callable."""
    def sleep(dt: float) -> None:
        session.wait(dt)
    return sleep


def _arm_lost(session: ArmSession) -> str | None:
    """The session's ending when the arm must stop now (cancel, STOP, taken away, disconnected), else None."""
    try:
        session.check_arm()
    except Ended as exc:
        return exc.ending
    return None


# ====================================================================== jaw: log and stroke
@dataclass
class EffortLog:
    """One jaw move: samples of the backend's gripper state, `t` relative to the command (pre-move < 0)."""

    t: np.ndarray
    effort: np.ndarray
    streamed: np.ndarray
    commanded: np.ndarray
    frozen: np.ndarray
    rate_hz: float
    valid: bool                       # False when the freeze fired during the move
    target: float = math.nan          # the opening commanded
    t0: float = 0.0                   # clock time of the command
    operator_s: float | None = None   # when the operator saw the jaw stop, after the command


def _nan(v: float | None) -> float:
    return math.nan if v is None else float(v)


def _froze(t: np.ndarray, frozen: np.ndarray) -> bool:
    """Whether the freeze fired during the move: `frozen` turned true after the command.

    A flag left over from the previous close, which this move's open clears, does not count.
    """
    before = frozen[t < 0.0]
    seq = np.concatenate([[bool(before[-1]) if len(before) else False], frozen[t >= 0.0]])
    return bool(np.any(seq[1:] & ~seq[:-1]))


def log_gripper(twin: Twin, moves: Sequence[float], *, arm_joints: Mapping[str, float], settle_s: float = 2.0,
                poll_s: float = 0.001, stop: threading.Event | None = None,
                clock: Callable[[], float] = time.monotonic,
                sleep: Callable[[float], None] = time.sleep) -> list[EffortLog]:
    """Commands each opening in turn (arm held at `arm_joints`) and samples the gripper state.

    Per move: `PRE_MOVE_S` of samples before the command, then `settle_s` after it. A sample is
    kept only when the backend's stamp changed. The caller owns the arm as `OWNER`, engaged. A
    move cut short (`stop`, the arm taken away, a disconnect) is dropped and logging ends.
    """
    grip = twin.spec.gripper
    arm = {k: float(v) for k, v in arm_joints.items() if k != grip}
    if not all(math.isfinite(v) for v in arm.values()):
        raise ValueError(f"arm joints not finite: {arm}")
    logs: list[EffortLog] = []

    def sample(t_end: float, rows: list[tuple], last: list[float | None]) -> bool:
        while True:
            if stop is not None and stop.is_set():
                return False
            b = twin.backend
            if b is None or not twin.connected:
                return False
            gs = b.gripper_state()
            now = clock()
            if gs is not None and gs.t != last[0]:
                last[0] = gs.t
                rows.append((now, _nan(gs.effort), _nan(gs.streamed), _nan(gs.commanded), bool(gs.frozen)))
            if now >= t_end:
                return True
            sleep(poll_s)

    for target in moves:
        v = float(target)
        if not (math.isfinite(v) and 0.0 <= v <= 100.0):
            raise ValueError(f"gripper opening {v} outside 0..100")
        rows: list[tuple] = []
        last: list[float | None] = [None]
        t_pre = clock()
        if not sample(t_pre + PRE_MOVE_S, rows, last):
            break
        t0 = clock()
        try:
            twin.set_target({**arm, grip: v}, owner=OWNER)
        except RuntimeError as exc:
            logger.warning("jaw log ended: %s", exc)
            break
        if not sample(t0 + settle_s, rows, last):
            break
        t = np.array([r[0] for r in rows], float) - t0
        frozen = np.array([r[4] for r in rows], bool)
        span = max(float(t[-1] - t[0]), 1e-9) if len(t) > 1 else math.inf
        logs.append(EffortLog(
            t=t,
            effort=np.array([r[1] for r in rows], float),
            streamed=np.array([r[2] for r in rows], float),
            commanded=np.array([r[3] for r in rows], float),
            frozen=frozen,
            rate_hz=(len(t) - 1) / span if len(t) > 1 else 0.0,
            valid=not _froze(t, frozen),
            target=v,
            t0=t0,
        ))
    return logs


def stroke_from_log(log: EffortLog, *, noise: float | None = None) -> dict[str, float | None]:
    """Stream stroke, effort peak and mean, and the last effort above the noise band of one move.

    `stream_s`: time of a full 0..100 stroke of the backend's stream, scaled from the part of the
    move between `STROKE_EDGE_UNITS` from its start and from its end (None for a move under
    `STROKE_MIN_UNITS`). `effort_s`: the last time after the command at which the effort left the
    band of `NOISE_SIGMAS` sigma of the pre-move samples (or `noise` given). `delta`: units moved.
    """
    t = np.asarray(log.t, float)
    post = t >= 0.0
    eff = np.asarray(log.effort, float)
    pre_eff = eff[(~post) & np.isfinite(eff)]
    centre = float(np.mean(pre_eff)) if len(pre_eff) else 0.0
    if noise is None:
        noise = NOISE_SIGMAS * float(np.std(pre_eff)) if len(pre_eff) > 1 else 0.0
        noise = max(noise, EFFORT_NOISE_MIN)
    post_eff = eff[post]
    finite = np.isfinite(post_eff)
    peak = float(np.max(np.abs(post_eff[finite]))) if finite.any() else None
    mean = float(np.mean(np.abs(post_eff[finite]))) if finite.any() else None
    tp = t[post]
    above = finite & (np.abs(post_eff - centre) > noise)
    effort_s = float(tp[above][-1]) if above.any() else None

    s = np.asarray(log.streamed, float)[post]
    ok = np.isfinite(s)
    stream_s: float | None = None
    delta: float | None = None
    if ok.sum() >= 2:
        ts, ss = tp[ok], s[ok]
        s0, s1 = float(ss[0]), float(ss[-1])
        delta = s1 - s0
        if abs(delta) >= STROKE_MIN_UNITS:
            moved = np.abs(ss - s0) > STROKE_EDGE_UNITS
            arrived = np.abs(ss - s1) <= STROKE_EDGE_UNITS
            if moved.any() and arrived.any():
                i0 = int(np.argmax(moved))
                i1 = int(np.argmax(arrived))
                if i1 > i0:
                    stream_s = float(ts[i1] - ts[i0]) * 100.0 / (abs(delta) - 2.0 * STROKE_EDGE_UNITS)
    return {"stream_s": stream_s, "effort_s": effort_s, "peak": peak, "mean": mean, "noise": float(noise),
            "delta": delta}


def _is_stroke(stats: Mapping[str, float | None], closing: bool) -> bool:
    d = stats.get("delta")
    return d is not None and (d <= -STROKE_MIN_UNITS if closing else d >= STROKE_MIN_UNITS)


def gripper_summary(free: Sequence[EffortLog], loaded: Sequence[EffortLog]) -> dict[str, Any]:
    """Per-rig `gripper` values from VALID moves only; a key without a valid move is left out.

    A freeze cuts a close short, and that short time would become the jaw speed of the gap
    model, so nothing of a move in which the freeze fired counts. `freeze_threshold` is never
    here: the operator decides it and records it with `--set-freeze`.
    """
    out: dict[str, Any] = {}
    stats = [(log, stroke_from_log(log)) for log in free if log.valid]
    for closing, name in ((True, "close"), (False, "open")):
        strokes = [(log, s) for log, s in stats if _is_stroke(s, closing)]
        effort = [s["effort_s"] for _, s in strokes if s["effort_s"] is not None]
        operator = [log.operator_s for log, _ in strokes if log.operator_s is not None]
        if effort:
            out[f"stroke_{name}_s_effort"] = round(max(effort), 4)
        if operator:
            out[f"stroke_{name}_s_operator"] = round(max(operator), 4)
    free_peaks = [s["peak"] for _, s in stats if _is_stroke(s, True) and s["peak"] is not None]
    loaded_stats = [stroke_from_log(log) for log in loaded if log.valid]
    loaded_peaks = [s["peak"] for s in loaded_stats if _is_stroke(s, True) and s["peak"] is not None]
    if free_peaks:
        out["free_effort_peak"] = round(max(free_peaks), 4)
    if loaded_peaks:
        out["loaded_effort_peak"] = round(max(loaded_peaks), 4)
    if free_peaks and loaded_peaks:
        free_peak, loaded_peak = max(free_peaks), max(loaded_peaks)
        separable = loaded_peak > SEPARABLE_RATIO * free_peak
        out["separable"] = bool(separable)
        out["suggested_threshold"] = round(0.5 * (free_peak + loaded_peak), 4) if separable else None
    return out


def _parse_gap_mm(items: Sequence[str] | None) -> dict[str, float] | str:
    """`O:MM` ruler readings -> {opening: mm}; a malformed item gives the error text."""
    out: dict[str, float] = {}
    for item in items or ():
        try:
            o_text, mm_text = str(item).split(":")
            o, mm = float(o_text), float(mm_text)
        except ValueError:
            return f"--gap-mm {item!r}: oczekiwano ROZWARCIE:MM, np. 50:53"
        if not (math.isfinite(o) and 0.0 <= o <= 100.0 and math.isfinite(mm) and mm > 0.0):
            return f"--gap-mm {item!r}: rozwarcie 0..100 i dodatnie mm"
        out[f"{o:g}"] = mm
    return out


def _record_gripper_notes(a: argparse.Namespace, gap: Mapping[str, float], freeze: float | None) -> int:
    """`--gap-mm` and `--set-freeze`: written without moving anything."""
    values: dict[str, Any] = {}
    if freeze is not None:
        if not (math.isfinite(freeze) and freeze > 0.0):
            print(f"--set-freeze {freeze}: oczekiwano dodatniej liczby (100 = zamrozenie wylaczone)",
                  file=sys.stderr)
            return 2
        values["freeze_threshold"] = float(freeze)
    try:
        rig = Rig.load(getattr(a, "rig", None))
    except RigError as exc:
        print(f"Plik wartosci stanowiska chwytu jest bledny: {exc}", file=sys.stderr)
        return 2
    if gap:
        merged = dict(rig.gripper.get("gap_mm") or {})
        for key, mm in gap.items():
            model = (GAP_CLOSED + GAP_PER_UNIT * float(key)) * 1000.0
            if abs(mm - model) > GAP_WARN_MM:
                print(f"UWAGA: rozwarcie {key}: {mm:.1f} mm, model szczek zaklada {model:.1f} mm "
                      f"(roznica {mm - model:+.1f} mm, ponad {GAP_WARN_MM:g} mm)", flush=True)
            merged[key] = mm
        values["gap_mm"] = merged
    rig = _update_rig(rig.path, "gripper", values)
    if freeze is not None:
        print(f"Zapisano gripper.freeze_threshold = {freeze:g}"
              + (" (zamrozenie wylaczone)" if freeze >= FREEZE_DISABLED else "")
              + ". Ta sama wartosc musi byc w a1x.gripper_force konfiguracji YAML, "
                "inaczej tryb real odmowi (gripper freeze differs).", flush=True)
    if gap:
        print(f"Zapisano gripper.gap_mm = {values['gap_mm']}", flush=True)
    print(f"Plik: {rig.path}", flush=True)
    return 0


def _log_with_operator(twin: Twin, keys: Keys, moves: Sequence[float], arm_joints: Mapping[str, float],
                       stop: threading.Event, clock: Callable[[], float],
                       sleep: Callable[[float], None]) -> list[EffortLog]:
    """`log_gripper`, while a reader stamps the operator's lines ("the jaw has stopped")."""
    stamps: list[float] = []
    done = threading.Event()

    def read() -> None:
        while not done.is_set():
            text = keys.readline(timeout_s=0.1)
            if text is None:
                done.wait(0.02)
            elif text:
                stamps.append(clock())

    reader = threading.Thread(target=read, name="chwyt-pomiar-oko", daemon=True)
    reader.start()
    try:
        logs = log_gripper(twin, moves, arm_joints=arm_joints, settle_s=GRIPPER_SETTLE_S, stop=stop,
                           clock=clock, sleep=sleep)
    finally:
        done.set()
        reader.join(1.0)
    for log in logs:
        seen = [t - log.t0 for t in stamps if OPERATOR_MIN_S <= t - log.t0 <= GRIPPER_SETTLE_S]
        if seen:
            log.operator_s = min(seen)
    return logs


def _fmt(v: float | None, spec: str = ".2f") -> str:
    return "-" if v is None or not math.isfinite(v) else format(v, spec)


def _print_moves(logs: Sequence[tuple[str, EffortLog]]) -> None:
    print("\nruch  faza      cel  suw strumienia [s]  wysilek szczyt  srednio  ostatni ponad szum [s]  "
          "oko [s]  wazny", flush=True)
    for i, (part, log) in enumerate(logs, 1):
        s = stroke_from_log(log)
        print(f"{i:4d}  {part:<8s} {log.target:4.0f}  {_fmt(s['stream_s']):>18s}  {_fmt(s['peak']):>14s}  "
              f"{_fmt(s['mean']):>7s}  {_fmt(s['effort_s']):>22s}  {_fmt(log.operator_s):>7s}  "
              f"{'tak' if log.valid else 'NIE (zamrozenie)'}", flush=True)


def _save_logs(logs: Sequence[tuple[str, EffortLog]]) -> Path:
    out = data_path(BENCH_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"gripper_{_stamp()}.npz"
    arrays: dict[str, np.ndarray] = {
        "part": np.array([p for p, _ in logs]),
        "target": np.array([log.target for _, log in logs], float),
        "t0": np.array([log.t0 for _, log in logs], float),
        "rate_hz": np.array([log.rate_hz for _, log in logs], float),
        "valid": np.array([log.valid for _, log in logs], bool),
        "operator_s": np.array([_nan(log.operator_s) for _, log in logs], float),
    }
    for i, (_, log) in enumerate(logs):
        for key in ("t", "effort", "streamed", "commanded", "frozen"):
            arrays[f"{key}_{i}"] = np.asarray(getattr(log, key))
    np.savez(path, **arrays)
    return path


def _gripper_refusal(b: Bench) -> int | None:
    twin = b.twin
    backend = twin.backend
    gs = backend.gripper_state() if backend is not None else None
    if gs is None:
        return _refuse("not armed", "backend nie podaje stanu chwytaka")
    if b.simulated:
        return None
    measured, limits = twin.status.measured, twin.joint_limits()
    outside = [f"{k} {measured[k]:.1f} st. poza {limits[k][0]:.1f}..{limits[k][1]:.1f}"
               for k in twin.spec.arm_joints
               if k in measured and k in limits and not limits[k][0] <= measured[k] <= limits[k][1]]
    if outside:
        return _refuse("arm outside limits",
                       "sprzeglo sprowadziloby staw w granice ruchem (" + "; ".join(outside) + "). "
                       "Najpierw ustaw ramie w pozie domowej albo nad punktem (panel albo grasp-bench point)")
    if gs.force_threshold is not None and gs.force_threshold < FREEZE_DISABLED:
        return _refuse("freeze active",
                       f"prog zamrozenia szczek {gs.force_threshold:g} skrocilby swobodny ruch; "
                       f"na czas pomiaru ustaw a1x.gripper_force: {FREEZE_DISABLED:g} w konfiguracji YAML")
    return None


def bench_gripper(a: argparse.Namespace, keys: Keys) -> int:
    """Jaw stroke and effort: free open/close moves, then a close on a 3 cm cube."""
    gap = _parse_gap_mm(getattr(a, "gap_mm", None))
    if isinstance(gap, str):
        print(gap, file=sys.stderr)
        return 2
    freeze = getattr(a, "set_freeze", None)
    if freeze is not None or gap:
        return _record_gripper_notes(a, gap, None if freeze is None else float(freeze))
    b = _open(a, moves=True)
    if isinstance(b, int):
        return b
    twin = b.twin
    session: ArmSession | None = None
    watch: _StopWatch | None = None
    try:
        refused = _gripper_refusal(b)
        if refused is not None:
            return refused
        if not b.simulated and not keys.confirm(
                "Enter = pomiar szczek (99, 100, 0, 100, 0, 100, potem z kostka 0, 100); ramie stoi; "
                "Ctrl-C = koniec. "):
            return _refuse("not confirmed", "bez potwierdzenia Enter (potrzebny terminal)")
        keys.arm()
        watch = _StopWatch(keys, twin).start()
        session = b.session()
        why = session.check() or session.take()
        if why is not None:
            return _not_started(why, session.detail, keys)
        session.guard(lag_deg=MOVE_GUARD_LAG_DEG, lag_s=JOG_GUARD_LAG_S)
        clock, sleep = session.clock, _session_sleep(session)
        arm_joints = {k: float(twin.status.measured[k]) for k in twin.spec.arm_joints}
        print("Szczeki bez niczego miedzy nimi. Gdy szczeki stana, wpisz s i Enter (pomiar na oko); "
              "sam Enter = STOP.", flush=True)
        free = _log_with_operator(twin, keys, GRIPPER_FREE_MOVES, arm_joints, session.cancel, clock, sleep)
        complete = len(free) == len(GRIPPER_FREE_MOVES)
        loaded: list[EffortLog] = []
        if complete:
            keys.note_end()
            if keys.proceed("Poloz kostke 3 cm miedzy szczekami, potem d i Enter; sam Enter = STOP. "):
                loaded = _log_with_operator(twin, keys, GRIPPER_LOADED_MOVES, arm_joints, session.cancel,
                                            clock, sleep)
            complete = len(loaded) == len(GRIPPER_LOADED_MOVES)
        logs = [("wolny", log) for log in free] + [("kostka", log) for log in loaded]
        if not logs:
            print("Brak pomiaru (przerwany).", file=sys.stderr)
            return 1
        _print_moves(logs)
        path = _save_logs(logs)
        print(f"Zapis przebiegow: {path}", flush=True)
        samples = sum(len(log.t) for _, log in logs)
        span = sum(float(log.t[-1] - log.t[0]) for _, log in logs if len(log.t) > 1)
        rate = samples / span if span > 0 else 0.0
        print(f"Probkowanie: {rate:.0f} probek/s (wymagane >= {MIN_RATE_HZ:.0f})", flush=True)
        if rate < MIN_RATE_HZ:
            print(f"Za wolne probkowanie stanu chwytaka ({rate:.0f} < {MIN_RATE_HZ:.0f}/s) - "
                  f"nic nie zapisano do pliku stanowiska.", file=sys.stderr)
            return 1
        invalid = [i for i, (_, log) in enumerate(logs, 1) if not log.valid]
        if invalid:
            print(f"Ruchy {invalid}: zamrozenie zadzialalo - ich liczby NIE ida do pliku stanowiska.",
                  flush=True)
        values = gripper_summary(free, loaded)
        if values.get("separable") is False:
            print("Wysilek z kostka nie odroznia sie od swobodnego ruchu - progu nie da sie dobrac; "
                  f"zostaw zamrozenie wylaczone ({FREEZE_DISABLED:g}).", flush=True)
        elif values.get("suggested_threshold") is not None:
            print(f"Sugerowany prog zamrozenia: {values['suggested_threshold']:g} (swobodnie "
                  f"{values['free_effort_peak']:g}, z kostka {values['loaded_effort_peak']:g}). "
                  f"Ustaw go jako a1x.gripper_force w konfiguracji YAML i zapisz: "
                  f"grasp-bench gripper --set-freeze <wartosc>.", flush=True)
        if not values:
            print("Brak waznych ruchow - nic nie zapisano do pliku stanowiska.", flush=True)
        elif b.may_write:
            rig = _update_rig(b.rig_path, "gripper", values)
            print(f"Zapisano do {rig.path}: {values}", flush=True)
        else:
            print(f"Symulacja bez --rig: nic nie zapisano ({values}).", flush=True)
        return 0 if complete else 1
    except KeyboardInterrupt:
        if twin.connected:
            stop_now(twin, CTRL_C_REASON)
        print("\nPrzerwano (Ctrl-C).", file=sys.stderr)
        return 1
    finally:
        if watch is not None:
            watch.close()
        _release(session)
        twin.close()


# ====================================================================== jog and teaching
class Jog:
    """Small Cartesian steps of a top-down tool, by ik_step, through Twin.set_target."""

    def __init__(self, session: ArmSession, *, speed_deg_s: float = JOG_SPEED_DEG_S) -> None:
        self.session = session
        self.twin: Twin = session.twin
        # Own model: FK of measured poses on this thread, next to the session's IK model.
        self.kin = RobotKinematics(self.twin.spec)
        self.speed_deg_s = float(speed_deg_s)
        self.clock = session.clock
        self.arm = tuple(self.twin.spec.arm_joints)
        self.grip_name = self.twin.spec.gripper
        self.p: np.ndarray | None = None                 # commanded TCP, base frame
        self.yaw: float = 0.0                            # commanded closing-axis yaw
        self.q: np.ndarray | None = None                 # last accepted arm joints, rad
        self.grip = 100.0
        self.contact_z: float | None = None              # measured TCP z when the last contact fired
        #: Commanded minus measured TCP z in free air (servo sag), from the hover and every clean step.
        self.sag = 0.0
        self.note = ""                                   # what the last refusal or contact was
        self.clipped = False                             # the last step was cut to its limit
        self._d_hist: list[float] = []                   # measured z before and after the last `d` steps
        self._d_cmd: list[float] = []                    # commanded drop of each of those steps

    # ------------------------------------------------------------ public
    def start(self, hover: Hover) -> str | None:
        """Claim, engage and pre-position over `hover` at 20 deg/s; the reason on a refusal."""
        s = self.session
        why = s.check() or s.take()
        if why is not None:
            self.note = s.detail
            return why
        why, sag = s.preposition(hover, PREPOSITION_SPEED_DEG_S)
        if why is not None:
            self.note = s.detail
            return why
        self.sag = float(sag)
        self.p = np.asarray(hover.p, float).copy()
        self.yaw = float(hover.yaw)
        self.q = np.asarray(hover.q, float).copy()
        self.grip = s.jaw_command()
        self._d_hist.clear()
        self._d_cmd.clear()
        return None

    def step(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, dyaw: float = 0.0) -> str | None:
        """One step; None, a refusal (`ik`, `jump`, `collision`, `stop`, ...) or "contact"."""
        self.note = ""
        self.clipped = False
        if self.p is None or self.q is None:
            return "not started"
        if not all(math.isfinite(v) for v in (dx, dy, dz, dyaw)):
            return "non-finite"
        why = self._lost()
        if why == "contact":
            return self._contact()
        if why is not None:
            return why
        joints = self._newest()
        if joints is None:
            return "stale feedback"
        z_meas = float(self._tcp(joints)[2])
        slow = dz < 0.0 and dx == 0.0 and dy == 0.0 and dyaw == 0.0 and z_meas < JOG_SLOW_BELOW_Z
        want = (float(dx), float(dy), float(dz), float(dyaw))
        dx, dy = (float(np.clip(v, -JOG_STEP_M, JOG_STEP_M)) for v in (dx, dy))
        dz = float(np.clip(dz, -(JOG_SLOW_STEP_M if slow else JOG_STEP_M), JOG_STEP_M))
        dyaw = float(np.clip(dyaw, -JOG_STEP_YAW, JOG_STEP_YAW))
        self.clipped = (dx, dy, dz, dyaw) != want

        p = self.p + np.array([dx, dy, dz])
        yaw = float(wrap(self.yaw + dyaw))
        # Below 4 cm the checker would refuse the touch pose itself: the IK rule only.
        q, ik, why = self.session.accept(p, grasp_R(yaw), self.q,
                                         collision=bool(p[2] > JOG_COLLISION_ABOVE_Z))
        if q is None:
            self.note = (f"blad pozycji {ik.pos_err * 1000:.2f} mm, obrotu {ik.rot_err * 1000:.1f} mrad, "
                         f"skok {ik.jump_deg:.1f} st.")
            return why or "ik"

        if slow:
            if not self._d_hist:
                self._d_hist.append(z_meas)
            self._d_cmd.append(-dz)
        else:
            self._d_hist.clear()
            self._d_cmd.clear()

        why = self._rearm()
        if why is not None:
            return why
        start_deg, goal_deg = np.degrees(self.q), np.degrees(q)
        lag0 = self._lags(joints, start_deg)
        self.p, self.yaw, self.q = p, yaw, np.asarray(q, float)
        hit = self._ramp(start_deg, goal_deg)
        if hit is None:
            hit = self._settle(goal_deg, lag0)
        if hit is None and slow:
            hit = self._height_rule()
        if hit == "contact":
            return self._contact()
        # The guard is for moves; between steps the arm stands on its target.
        self.session.disarm()
        if hit is None and not slow:
            last = self._newest()
            if last is not None:
                self.sag = min(max(float(self.p[2]) - float(self._tcp(last)[2]), 0.0), SAG_LIMIT)
        return hit

    def gripper(self, opening: float) -> str | None:
        """Commands the jaw (arm held at the last accepted joints) and waits for the stream to arrive."""
        if self.q is None:
            return "not started"
        v = float(opening)
        if not math.isfinite(v):
            return "non-finite"
        v = min(max(v, 0.0), 100.0)
        why = self._lost() or self._send(np.degrees(self.q), v)
        if why is not None:
            return why
        self.grip = v
        t_end = self.clock() + GRIP_WAIT_S
        while self.clock() < t_end:
            why = self._lost()
            if why is not None:
                return why
            b = self.twin.backend
            gs = b.gripper_state() if b is not None else None
            if gs is not None and gs.streamed is not None and abs(gs.streamed - v) <= 1.0:
                break
            self.session.wait(JOG_POLL_S)
        return None

    def measured_tcp(self, average_s: float = 0.5) -> tuple[np.ndarray, float, dict[str, float]]:
        """Measured TCP position, closing-axis yaw and joints, averaged over `average_s` of fresh samples."""
        t0 = self.clock()
        t_end = t0 + average_s
        samples: dict[float, dict[str, float]] = {}
        while True:
            js = self.session.history.newest()
            if js is not None and js.t >= t0 - JOG_POLL_S:
                samples[js.t] = dict(js.joints)
            if self.clock() >= t_end:
                break
            self.session.wait(JOG_POLL_S)
        if not samples:
            raise RuntimeError("no fresh joint samples to average")
        joints = dict(samples[max(samples)])
        for k in self.arm:
            joints[k] = float(np.mean([s[k] for s in samples.values()]))
        T = self.kin.tcp(joints)
        return T[:3, 3].copy(), float(yaw_of(T[:3, :3])), joints

    def close(self) -> None:
        _release(self.session)

    # ------------------------------------------------------------ internals
    def _newest(self) -> dict[str, float] | None:
        js = self.session.history.newest()
        return None if js is None else dict(js.joints)

    def _tcp(self, joints: Mapping[str, float]) -> np.ndarray:
        return self.kin.tcp(joints)[:3, 3]

    def _lags(self, joints: Mapping[str, float], cmd_deg: np.ndarray) -> dict[str, float]:
        return {k: abs(float(cmd_deg[i]) - float(joints[k])) for i, k in enumerate(self.arm)}

    def _lost(self) -> str | None:
        """None while the step may go on; "contact" when the impact guard fired; else the ending."""
        why = _arm_lost(self.session)
        if why is not None and why.startswith("impact"):
            self.note = f"straznik uderzen: {why}"
            return "contact"
        return why

    def _rearm(self) -> str | None:
        """A fresh cancel and impact guard for the next move; a STOP or a lost arm ends instead."""
        s = self.session
        if s.cancel.is_set() and s.cause() != "impact":
            return _arm_lost(s) or "stop"
        s.begin()
        why = _arm_lost(s)
        if why is not None:
            return why
        s.guard(lag_deg=JOG_GUARD_LAG_DEG, lag_s=JOG_GUARD_LAG_S)
        return None

    def _send(self, arm_deg: np.ndarray, grip: float) -> str | None:
        target = {k: float(arm_deg[i]) for i, k in enumerate(self.arm)}
        if self.grip_name is not None:
            target[self.grip_name] = float(grip)
        try:
            self.session.command(target)
        except Ended as exc:
            return exc.ending
        return None

    def _ramp(self, q0_deg: np.ndarray, q1_deg: np.ndarray) -> str | None:
        """Smoothstep from `q0` to `q1` at `speed_deg_s`, a target every 20 ms; paused while the loop is."""
        largest = float(np.max(np.abs(q1_deg - q0_deg))) if len(q0_deg) else 0.0
        duration = max(JOG_RAMP_MIN_S, largest / self.speed_deg_s)
        elapsed = 0.0
        t_prev = self.clock()
        while True:
            why = self._lost()
            if why is not None:
                return why
            now = self.clock()
            if not self.twin._frozen():
                elapsed += now - t_prev
            t_prev = now
            u = min(1.0, elapsed / duration)
            why = self._send(q0_deg + (q1_deg - q0_deg) * smoothstep(u), self.grip)
            if why is not None:
                return why
            if u >= 1.0:
                return None
            self.session.wait(JOG_RAMP_DT)

    def _settle(self, goal_deg: np.ndarray, lag0: Mapping[str, float]) -> str | None:
        """0.3 s after the ramp: the guard at any moment; the lag rule when the lag stays grown throughout."""
        t_ramp = self.clock()
        t_end = t_ramp + JOG_SETTLE_S
        excess = dict.fromkeys(self.arm, math.inf)
        seen = 0
        last_t: float | None = None
        while True:
            why = self._lost()
            if why is not None:
                return why
            js = self.session.history.newest()
            if js is not None and js.t != last_t and js.t >= t_ramp - 1e-9:
                last_t = js.t
                seen += 1
                lags = self._lags(js.joints, goal_deg)
                for k in self.arm:
                    excess[k] = min(excess[k], lags[k] - lag0[k])
            if self.clock() >= t_end:
                break
            self.session.wait(JOG_POLL_S)
        if seen:
            joint = max(self.arm, key=lambda k: excess[k])
            if excess[joint] > JOG_LAG_DEG:
                self.note = (f"opoznienie: {joint} o {excess[joint]:.1f} st. dalej za rozkazem "
                             f"niz na poczatku kroku")
                return "contact"
        return None

    def _height_rule(self) -> str | None:
        """After a settled 1 mm `d` step below 6 cm: contact when two steps together dropped under 0.4 mm."""
        last = self._newest()
        if last is None:
            return None
        self._d_hist.append(float(self._tcp(last)[2]))
        if len(self._d_hist) < 3:
            return None
        drop = self._d_hist[-3] - self._d_hist[-1]
        commanded = sum(self._d_cmd[-2:])
        # Scaled by the commanded drop, so steps shorter than 1 mm in free air are not read as contact.
        if drop < JOG_CONTACT_MIN_DROP * commanded / (2.0 * JOG_SLOW_STEP_M):
            self.note = (f"wysokosc: TCP opadl o {drop * 1000:.2f} mm w dwoch krokach "
                         f"(rozkaz {commanded * 1000:.1f} mm)")
            return "contact"
        return None

    def _contact(self) -> str:
        """Hold the measured pose, back off `JOG_BACKOFF` straight up, report "contact".

        The backoff is commanded from the measured pose, so it adds the free-air sag: a servo
        that sags 4.5 mm (the twin's arm at 0.33 m) would otherwise leave the tips on the table.
        """
        s = self.session
        joints = self._hold_measured()
        self._d_hist.clear()
        self._d_cmd.clear()
        if joints is None:
            return "contact"
        p_m = self._tcp(joints).copy()
        self.contact_z = float(p_m[2])
        why = self._rearm()
        if why is not None:
            return why
        p_back = p_m + np.array([0.0, 0.0, JOG_BACKOFF + self.sag])
        q, _ik, why = s.accept(p_back, grasp_R(self.yaw), self.q, collision=False)
        if q is None:
            s.disarm()
            self.note += f"; cofniecie odrzucone ({why})"
            return "contact"
        start = np.degrees(self.q)
        self.p, self.q = p_back, np.asarray(q, float)
        why = self._ramp(start, np.degrees(q))
        t_end = self.clock() + JOG_SETTLE_S
        while why is None and self.clock() < t_end:
            why = self._lost()
            if why is None:
                s.wait(JOG_POLL_S)
        if why == "contact":
            # The backoff itself met something: stay where the arm is. That hit is handled, so the
            # next line command starts clean instead of backing off again.
            self._hold_measured()
            s.begin()
            self.note += "; cofniecie przerwane"
        elif why is not None:
            return why
        s.disarm()
        return "contact"

    def _hold_measured(self) -> dict[str, float] | None:
        """`hold("impact")` (the plain reading) and the jog's command moved onto the measured pose.

        The yaw too: a blocked rotation left the tool at its old yaw, and a backoff at the
        commanded one would push the blocked joint again.
        """
        self.session.hold("impact")
        joints = self._newest()
        if joints is not None:
            T = self.kin.tcp(joints)
            self.p = T[:3, 3].copy()
            self.yaw = float(yaw_of(T[:3, :3]))
            self.q = np.radians([float(joints[k]) for k in self.arm])
        return joints


def _default_xy(rig: Rig, what: str) -> tuple[float, float]:
    other = "grasp" if what == "table" else "table"
    tcp = getattr(rig, other).get("tcp")
    if isinstance(tcp, (list, tuple)) and len(tcp) == 3:
        return float(tcp[0]), float(tcp[1])
    return START_XY_DEFAULT


def _joint_offsets(twin: Twin) -> list[float]:
    offsets = getattr(getattr(twin.backend, "acfg", None), "joint_offsets_deg", None)
    if offsets is None:
        return [0.0] * 6
    return [float(v) for v in offsets]


def store_table(path: Path | None, pos: np.ndarray, joints: Mapping[str, float], *,
                force: bool = False) -> str | None:
    """Writes `table`; returns `table touch implausible` instead when the touch is more than 15 mm off."""
    touch = float(pos[2])
    diff = touch - TOUCH_EXPECTED_Z
    print(f"Dotyk stolu: TCP z {touch * 1000:.1f} mm, oczekiwane {TOUCH_EXPECTED_Z * 1000:.1f} mm "
          f"(roznica {diff * 1000:+.1f} mm)", flush=True)
    overrides = None
    if abs(diff) > TOUCH_TOLERANCE:
        if not force:
            print(f"Nie zapisano: roznica ponad {TOUCH_TOLERANCE * 1000:.0f} mm - problem z mocowaniem albo "
                  f"kinematyka. Sprawdz i naucz ponownie; --force zapisze mimo to (z wyjatkiem w pliku).",
                  file=sys.stderr, flush=True)
            return "table touch implausible"
        overrides = {"table": f"--force {_now_iso()}: dotyk {diff * 1000:+.1f} mm od oczekiwanego"}
    rig = _update_rig(path, "table", {
        "touch_tcp_z": round(touch, 6),
        "tcp": [round(float(v), 6) for v in pos],
        "joints": {k: round(float(v), 4) for k, v in joints.items()},
        "taught_at": _now_iso(),
    }, overrides=overrides)
    grasp = rig.grasp.get("tcp")
    if isinstance(grasp, (list, tuple)) and len(grasp) == 3:
        far = math.hypot(float(pos[0]) - float(grasp[0]), float(pos[1]) - float(grasp[1]))
        verdict = "w porzadku" if far <= TABLE_TOUCH_FAR else "ZA DALEKO - tryby odmowia (table touch far)"
        print(f"Od punktu chwytu: {far * 1000:.0f} mm (najwyzej {TABLE_TOUCH_FAR * 1000:.0f} mm) - {verdict}",
              flush=True)
    else:
        print(f"Punkt chwytu naucz najwyzej {TABLE_TOUCH_FAR * 1000:.0f} mm (xy) od tego dotyku.", flush=True)
    print(f"Zapisano table do {rig.path} (wazne {TEACH_MAX_AGE_S / 3600:.0f} h).", flush=True)
    return None


def store_point(path: Path | None, pos: np.ndarray, yaw: float, joints: Mapping[str, float], *,
                object_height: float, joint_offsets_deg: Sequence[float]) -> Rig:
    """Writes `grasp` (the taught TCP, yaw, object height and the joint offsets in force)."""
    rig = _update_rig(path, "grasp", {
        "tcp": [round(float(v), 6) for v in pos],
        "yaw": round(float(yaw), 6),
        "joints": {k: round(float(v), 4) for k, v in joints.items()},
        "taught_at": _now_iso(),
        "object_height": float(object_height),
        "joint_offsets_deg": [float(v) for v in joint_offsets_deg],
    })
    print(f"Zapisano grasp do {rig.path}: TCP ({pos[0] * 1000:.1f}, {pos[1] * 1000:.1f}, "
          f"{pos[2] * 1000:.1f}) mm, obrot osi zamykania {math.degrees(yaw):.1f} st., obiekt "
          f"{object_height * 1000:.0f} mm, przesuniecia stawow {list(joint_offsets_deg)} "
          f"(wazne {TEACH_MAX_AGE_S / 3600:.0f} h).", flush=True)
    return rig


def _explain_point(rig: Rig, kin: RobotKinematics, limits: Mapping[str, tuple[float, float]]) -> None:
    believed, yaw, touch = rig.believed(), rig.grasp_yaw(), rig.touch_tcp_z()
    if believed is None or yaw is None or touch is None:
        print("Stol nie jest nauczony: chwyt, w ktory wierzy polityka (xy punktu, z = dotyk + 6 mm), "
              "bedzie znany po grasp-bench table.", flush=True)
        return
    tcp = rig.grasp.get("tcp") or [math.nan] * 3
    print(f"Chwyt dla polityki: ({believed[0] * 1000:.1f}, {believed[1] * 1000:.1f}, "
          f"{believed[2] * 1000:.1f}) mm = xy punktu, z = dotyk stolu + {TIP_CLEAR * 1000:.0f} mm; "
          f"nauczony TCP z {float(tcp[2]) * 1000:.1f} mm (tylko do porownania).", flush=True)
    table = rig.table.get("tcp")
    if isinstance(table, (list, tuple)) and len(table) == 3:
        far = math.hypot(float(table[0]) - believed[0], float(table[1]) - believed[1])
        if far > TABLE_TOUCH_FAR:
            print(f"UWAGA: dotyk stolu {far * 1000:.0f} mm od punktu (najwyzej "
                  f"{TABLE_TOUCH_FAR * 1000:.0f}) - tryby odmowia (table touch far).", flush=True)
    try:
        hover = solve_hover(kin, believed, yaw, limits=limits, source="taught")
    except HoverError as exc:
        print(f"Zawis nad punktem nie do osiagniecia: {exc}", flush=True)
        return
    if hover.outside_training:
        print("Poza obszarem treningu: " + "; ".join(hover.outside_training)
              + " (tryby odmowia bez --outside-training).", flush=True)
    else:
        print("Zawis w obszarze treningu.", flush=True)


JOG_HELP = ("Polecenia (Enter po kazdym): d MM w dol, u MM w gore, x MM, y MM (ze znakiem), r ST obrot, "
            "g 0..100 szczeki, ok = zapisz, q = koniec bez zapisu; sam Enter = STOP. "
            f"Kroki najwyzej {JOG_STEP_M * 1000:.0f} mm ({JOG_SLOW_STEP_M * 1000:.0f} mm w dol ponizej "
            f"{JOG_SLOW_BELOW_Z * 1000:.0f} mm), obrot {math.degrees(JOG_STEP_YAW):.0f} st.")


def teach_lines(jog: Jog, keys: Keys, what: Literal["table", "point"], *, rig_path: Path | None,
                force: bool = False, object_height: float = OBJECT_HEIGHT_DEFAULT, may_write: bool = True,
                seconds: float | None = None) -> int:
    """The operator's line commands: 0 when `ok` saved, 1 on `q` or a STOP, 2 on `q` after a refused save."""
    clock = jog.clock
    t_end = None if seconds is None else clock() + float(seconds)
    refused = False
    print(JOG_HELP, flush=True)
    while True:
        if keys.stop.is_set():
            return 1
        if t_end is not None and clock() >= t_end:
            print("Koniec czasu (--seconds) - bez zapisu.", flush=True)
            return 2 if refused else 1
        if not jog.twin.connected:
            print(f"Ramie rozlaczone: {jog.twin.status.error}", file=sys.stderr)
            return 1
        line = keys.readline(timeout_s=0.2 if keys.interactive else None)
        if line is None:
            if keys.interactive:
                continue
            print("Koniec wejscia - bez zapisu.", flush=True)
            return 2 if refused else 1
        if line == "":
            keys.stop.set()                    # the stop watch carries it out through stop_now
            return 1
        parts = line.split()
        cmd = parts[0].lower()
        if cmd == "q":
            print("Koniec bez zapisu.", flush=True)
            return 2 if refused else 1
        if cmd == "ok":
            try:
                pos, yaw, joints = jog.measured_tcp()
            except RuntimeError as exc:
                print(f"Nie zmierzono pozy: {exc}", file=sys.stderr)
                continue
            if not may_write:
                print(f"Symulacja bez --rig: nic nie zapisano (TCP z {pos[2] * 1000:.1f} mm).", flush=True)
                return 0
            if what == "table":
                if store_table(rig_path, pos, joints, force=force) is not None:
                    refused = True
                    continue
                return 0
            rig = store_point(rig_path, pos, yaw, joints, object_height=object_height,
                              joint_offsets_deg=_joint_offsets(jog.twin))
            _explain_point(rig, jog.kin, jog.twin.joint_limits())
            return 0
        if cmd not in ("d", "u", "x", "y", "r", "g") or len(parts) != 2:
            print(JOG_HELP, flush=True)
            continue
        try:
            value = float(parts[1])
        except ValueError:
            print(f"{parts[1]!r} nie jest liczba.", flush=True)
            continue
        if not math.isfinite(value):
            print(f"{parts[1]!r} nie jest liczba.", flush=True)
            continue
        if cmd == "g":
            why = jog.gripper(value)
        else:
            mm = value / 1000.0
            why = jog.step(**{"d": {"dz": -abs(mm)}, "u": {"dz": abs(mm)}, "x": {"dx": mm}, "y": {"dy": mm},
                              "r": {"dyaw": math.radians(value)}}[cmd])
            if jog.clipped:
                print("Krok przyciety do limitu.", flush=True)
        if why == "contact":
            z = "-" if jog.contact_z is None else f"{jog.contact_z * 1000:.1f}"
            print(f"Kontakt prawdopodobny ({jog.note}); TCP przy kontakcie z {z} mm. Ramie trzyma pomiar "
                  f"i cofnelo sie o {JOG_BACKOFF * 1000:.0f} mm. Zejdz 'd 1' do dotyku i wpisz 'ok'.",
                  flush=True)
        elif why in ("ik", "jump", "collision", "non-finite", "stale feedback"):
            print(f"Krok odrzucony ({why}): {jog.note}", flush=True)
        elif why is not None:
            print(f"Koniec: {why}", file=sys.stderr)
            return 1
        _print_pose(jog)


def _print_pose(jog: Jog) -> None:
    joints = jog._newest()
    if joints is None or jog.p is None:
        return
    p = jog._tcp(joints)
    print(f"TCP zmierzony ({p[0] * 1000:.1f}, {p[1] * 1000:.1f}, {p[2] * 1000:.1f}) mm, rozkaz z "
          f"{jog.p[2] * 1000:.1f} mm, obrot {math.degrees(jog.yaw):.1f} st., szczeki {jog.grip:.0f}",
          flush=True)


def bench_teach(a: argparse.Namespace, keys: Keys, what: Literal["table", "point"]) -> int:
    """Teach the table touch (`table`) or the grasp point (`point`) by jogging the fingertips onto it."""
    height_arg = getattr(a, "object_height", None)
    object_height = OBJECT_HEIGHT_DEFAULT if height_arg is None else float(height_arg)
    if what == "point" and not OBJECT_HEIGHT_RANGE[0] <= object_height <= OBJECT_HEIGHT_RANGE[1]:
        lo, hi = OBJECT_HEIGHT_RANGE
        print(f"--object-height {object_height}: oczekiwano {lo}..{hi} m", file=sys.stderr)
        return 2
    z_arg = getattr(a, "start_z", None)
    start_z = START_Z_DEFAULT if z_arg is None else float(z_arg)
    if not math.isfinite(start_z) or start_z <= 0.0:
        print(f"--start-z {start_z}: oczekiwano wysokosci TCP nad stolem [m]", file=sys.stderr)
        return 2
    b = _open(a, moves=True)
    if isinstance(b, int):
        return b
    twin = b.twin
    jog: Jog | None = None
    watch: _StopWatch | None = None
    try:
        xy = getattr(a, "xy", None)
        x, y = (float(xy[0]), float(xy[1])) if xy else _default_xy(b.rig, what)
        yaw = math.radians(float(getattr(a, "yaw_deg", None) or 0.0))
        try:
            hover = solve_hover(b.kin, np.array([x, y, start_z - HOVER]), yaw, limits=twin.joint_limits(),
                                source="taught")
        except HoverError as exc:
            return _refuse(exc.reason, f"punkt startowy ({x:.3f}, {y:.3f}, {start_z:.3f}) m: {exc.detail}")
        print(f"Start: TCP ({x * 1000:.0f}, {y * 1000:.0f}, {start_z * 1000:.0f}) mm, narzedzie w dol, "
              f"obrot {math.degrees(hover.yaw):.0f} st.", flush=True)
        if what == "table":
            print(f"Opuszki na stol najwyzej {TABLE_TOUCH_FAR * 1000:.0f} mm od punktu chwytu.", flush=True)
        else:
            print(f"Plytki szczek po obu stronach obiektu ({object_height * 1000:.0f} mm), "
                  f"opuszki tuz nad stolem.", flush=True)
        if not b.simulated and not keys.confirm(
                "Enter = ramie jedzie nad punkt startowy (20 st./s); Ctrl-C = koniec. "):
            return _refuse("not confirmed", "bez potwierdzenia Enter (potrzebny terminal)")
        keys.arm()
        watch = _StopWatch(keys, twin).start()
        jog = Jog(b.session())
        why = jog.start(hover)
        if why is not None:
            return _not_started(why, jog.note, keys)
        _print_pose(jog)
        return teach_lines(jog, keys, what, rig_path=b.rig_path, force=bool(getattr(a, "force", False)),
                           object_height=object_height, may_write=b.may_write,
                           seconds=getattr(a, "seconds", None))
    except KeyboardInterrupt:
        if twin.connected:
            stop_now(twin, CTRL_C_REASON)
        print("\nPrzerwano (Ctrl-C).", file=sys.stderr)
        return 1
    finally:
        if watch is not None:
            watch.close()
        if jog is not None:
            jog.close()
        twin.close()


# ====================================================================== latency
def _mad(image: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(np.abs(image.astype(np.float32) - ref)))


def measure_latency(session: ArmSession, frames: FrameSource, hover: Hover, *, steps: int = LATENCY_STEPS,
                    step_deg: float = LATENCY_STEP_DEG) -> dict[str, Any]:
    """Steps of joint 6 by +step and -step at the hover; per step the actuation and the camera latency.

    `actuation_s`: from the target to the first fresh joint sample that moved more than 0.2 deg.
    `camera_s`: from that sample to the stamp of the first frame whose mean absolute difference
    from the still frames exceeds 4 sigma of theirs. Medians over the steps that saw both.
    """
    twin = session.twin
    clock, sleep = session.clock, _session_sleep(session)
    arm = tuple(twin.spec.arm_joints)
    j6 = arm[-1]
    grip = twin.spec.gripper
    opening = session.jaw_command()
    base = {k: float(hover.joints[k]) for k in arm}
    rows: list[dict[str, float | None]] = []
    ending: str | None = None

    def interrupted() -> str | None:
        return _arm_lost(session)

    twin.set_speed(None)                     # the step as the policy's loop would see it (90 deg/s)
    try:
        for k in range(steps):
            # Still frames at rest: their spread is the noise the motion has to rise above.
            still: list[np.ndarray] = []
            last_seq = None
            t_end = clock() + LATENCY_STILL_S
            while len(still) < LATENCY_STILL_FRAMES and clock() < t_end:
                ending = interrupted()
                if ending:
                    break
                fr = frames.latest()
                if fr is not None and fr.seq != last_seq:
                    last_seq = fr.seq
                    still.append(fr.image.astype(np.float32))
                sleep(LATENCY_POLL_S)
            if ending:
                break
            if len(still) < 2:
                ending = "camera"
                break
            ref = np.mean(still, axis=0)
            mads = np.array([_mad(f, ref) for f in still])
            threshold = float(np.mean(mads)) + LATENCY_SIGMAS * max(float(np.std(mads)), MAD_SIGMA_MIN)

            js0 = session.history.newest()
            if js0 is None:
                ending = "stale feedback"
                break
            q6_0 = float(js0.joints[j6])
            target = dict(base)
            target[j6] = base[j6] + (step_deg if k % 2 == 0 else 0.0)
            if grip is not None:
                target[grip] = opening
            t_cmd = clock()
            try:
                session.command(target)
            except Ended as exc:
                ending = exc.ending
                break
            t_act: float | None = None
            t_frame: float | None = None
            deadline = t_cmd + LATENCY_STEP_TIMEOUT_S
            while clock() < deadline and (t_act is None or t_frame is None):
                ending = interrupted()
                if ending:
                    break
                js = session.history.newest()
                if t_act is None and js is not None and js.t > t_cmd and \
                        abs(float(js.joints[j6]) - q6_0) > LATENCY_MOVED_DEG:
                    t_act = float(js.t)
                fr = frames.latest()
                if t_frame is None and fr is not None and fr.seq != last_seq:
                    last_seq = fr.seq
                    if fr.t > t_cmd and _mad(fr.image, ref) > threshold:
                        t_frame = float(fr.t)
                sleep(LATENCY_POLL_S)
            if ending:
                break
            row: dict[str, float | None] = {
                "actuation_s": None if t_act is None else t_act - t_cmd,
                "camera_s": None if t_act is None or t_frame is None else t_frame - t_act,
            }
            rows.append(row)
            print(f"krok {k + 1}: J6 {'+' if k % 2 == 0 else '-'}{step_deg:g} st.: ruch po "
                  f"{_fmt(row['actuation_s'], '.3f')} s, kamera {_fmt(row['camera_s'], '.3f')} s po nim",
                  flush=True)
            t_settle = clock() + LATENCY_SETTLE_S
            while ending is None and clock() < t_settle:
                ending = interrupted()
                sleep(LATENCY_POLL_S)
            if ending:
                break
        if ending is None and steps % 2 == 1:
            # Back to the hover after an odd number of steps (not measured), and settled there:
            # the hold at the end of the tool would otherwise stop joint 6 on its way.
            back = dict(base)
            if grip is not None:
                back[grip] = opening
            try:
                session.command(back)
            except Ended as exc:
                ending = exc.ending
            t_settle = clock() + LATENCY_SETTLE_S
            while ending is None and clock() < t_settle:
                ending = interrupted()
                sleep(LATENCY_POLL_S)
    finally:
        twin.set_speed(PREPOSITION_SPEED_DEG_S)
    good = [r for r in rows if r["actuation_s"] is not None and r["camera_s"] is not None]
    act = [float(r["actuation_s"]) for r in good]
    cam = [float(r["camera_s"]) for r in good]
    return {
        "steps": rows,
        "good": len(good),
        "ending": ending,
        "actuation_s": float(np.median(act)) if act else None,
        # A joint sample stamped later than the frame that already shows the motion gives a
        # negative value; pairing then falls back to the frame's own stamp.
        "camera_s": max(0.0, float(np.median(cam))) if cam else None,
        "camera_raw_s": float(np.median(cam)) if cam else None,
    }


def _checkpoint(a: argparse.Namespace) -> Path | None:
    """The checkpoint the run would use; None (after a message naming the fetch command) without one."""
    from .policy import find_checkpoint

    path = find_checkpoint(getattr(a, "checkpoint", None))
    if path is None:
        print("Brak modelu. Pobierz: lerobot-twin grasp-model fetch  (albo --checkpoint PATH, "
              "LEROBOT_GRASP_STUDENT)", file=sys.stderr)
    return path


def _inference_p95_s(a: argparse.Namespace) -> float | None:
    """p95 of one policy call on this machine, with the twin and the camera running [s]; None on failure."""
    from .policy import CheckpointError, GraspStudentPolicy

    path = _checkpoint(a)
    if path is None:
        return None
    try:
        policy = GraspStudentPolicy.load(path)
    except CheckpointError as exc:
        print(f"Model odrzucony: {exc}", file=sys.stderr)
        return None
    t = policy.time()
    print(f"Inferencja: mediana {t['median_ms']:.1f} ms, p95 {t['p95_ms']:.1f} ms", flush=True)
    return float(t["p95_ms"]) / 1000.0


def _open_frames(b: Bench, a: argparse.Namespace) -> FrameSource | int:
    if b.simulated:
        try:
            return SimWristSource(b.twin, name=WRIST_CAMERA, clock=b.clock)
        except CameraError as exc:
            print(f"Kamera na nadgarstku (symulacja): {exc}", file=sys.stderr)
            return 1
    source = b.rig.wrist_source() or getattr(a, "camera", None)
    if not source:
        return _refuse("wrist camera not set", "podaj wrist.source w pliku --rig albo --camera")
    try:
        return LiveWristSource(str(source), WristCamera(wrist_intrinsics(b.rig)))
    except CameraError as exc:
        print(f"Kamera na nadgarstku: {exc}", file=sys.stderr)
        return 1


def _first_frame(frames: FrameSource, session: ArmSession) -> bool:
    """Waits up to `FIRST_FRAME_S` for the wrist camera's first frame, before anything moves."""
    t_end = session.clock() + FIRST_FRAME_S
    while session.clock() < t_end:
        try:
            if frames.latest() is not None:
                return True
        except CameraError as exc:
            print(f"Kamera na nadgarstku: {exc}", file=sys.stderr)
            return False
        if session.wait(0.02):
            return False
    return False


def bench_latency(a: argparse.Namespace, keys: Keys) -> int:
    """Arm and camera latency at the hover, and the total a policy tick adds up to."""
    if _checkpoint(a) is None:
        return 2
    b = _open(a, moves=True)
    if isinstance(b, int):
        return b
    twin = b.twin
    session: ArmSession | None = None
    watch: _StopWatch | None = None
    frames: FrameSource | None = None
    try:
        limits = twin.joint_limits()
        try:
            if b.simulated:
                xy = getattr(a, "xy", None) or START_XY_DEFAULT
                yaw = math.radians(float(getattr(a, "yaw_deg", 0.0) or 0.0))
                hover = from_scene(b.kin, np.array(xy, float), yaw, touch_z=TOUCH_EXPECTED_Z, limits=limits)
            else:
                hover = from_rig(b.kin, b.rig, limits=limits)
        except HoverError as exc:
            return _refuse(exc.reason, exc.detail)
        opened = _open_frames(b, a)
        if isinstance(opened, int):
            return opened
        frames = opened
        session = b.session()
        # A silent camera is found before the arm moves.
        if not _first_frame(frames, session):
            print("Kamera na nadgarstku milczy (wrist camera silent).", file=sys.stderr)
            return 1
        # Timed with the twin loop and the camera running, as in a run.
        infer = _inference_p95_s(a)
        if infer is None:
            return 2
        if not b.simulated and not keys.confirm(
                "Enter = ramie jedzie nad punkt chwytu (20 st./s), potem 5 krokow J6 o 3 st.; "
                "Ctrl-C = koniec. "):
            return _refuse("not confirmed", "bez potwierdzenia Enter (potrzebny terminal)")
        keys.arm()
        watch = _StopWatch(keys, twin).start()
        why = session.check() or session.take()
        if why is None:
            why, _sag = session.preposition(hover, PREPOSITION_SPEED_DEG_S)
        if why is not None:
            return _not_started(why, session.detail, keys)
        session.guard(lag_deg=MOVE_GUARD_LAG_DEG, lag_s=JOG_GUARD_LAG_S)
        try:
            result = measure_latency(session, frames, hover)
        except CameraError as exc:
            print(f"Kamera na nadgarstku: {exc}", file=sys.stderr)
            return 1
        if result["ending"] is not None:
            print(f"Pomiar przerwany: {result['ending']}", file=sys.stderr)
            return 1
        if result["good"] < LATENCY_MIN_GOOD:
            print(f"Za malo udanych krokow ({result['good']} z {LATENCY_STEPS}) - nic nie zapisano.",
                  file=sys.stderr)
            return 1
        camera_s, actuation_s = float(result["camera_s"]), float(result["actuation_s"])
        total = camera_s + infer + actuation_s
        values = {"camera_s": round(camera_s, 4), "actuation_s": round(actuation_s, 4),
                  "infer_p95_s": round(infer, 4), "total_s": round(total, 4)}
        print(f"Opoznienie: kamera {camera_s * 1000:.0f} ms (surowo {result['camera_raw_s'] * 1000:.0f}) + "
              f"inferencja p95 {infer * 1000:.0f} ms + ramie {actuation_s * 1000:.0f} ms = "
              f"{total * 1000:.0f} ms ({'w granicy' if total <= LATENCY_MAX_S else 'PONAD'} "
              f"{LATENCY_MAX_S * 1000:.0f} ms)", flush=True)
        if b.may_write:
            rig = _update_rig(b.rig_path, "latency", values)
            print(f"Zapisano latency do {rig.path}.", flush=True)
        else:
            print("Symulacja bez --rig: nic nie zapisano.", flush=True)
        return 0
    except KeyboardInterrupt:
        if twin.connected:
            stop_now(twin, CTRL_C_REASON)
        print("\nPrzerwano (Ctrl-C).", file=sys.stderr)
        return 1
    finally:
        if watch is not None:
            watch.close()
        _release(session)
        if frames is not None:
            try:
                frames.close()
            except Exception:
                logger.exception("closing the wrist camera failed")
        twin.close()


# ====================================================================== dot and show
def bench_dot(a: argparse.Namespace) -> int:
    """Records `wrist.dot_error_px`, the dot error the operator read from the dry run dump."""
    value = getattr(a, "error_px", None)
    if value is None:
        print("Podaj --error-px V: odleglosc [px w obrazie 160x120] miedzy krzyzykiem (chwyt, w ktory wierzy "
              "polityka) a srodkiem gornej sciany kostki na zrzucie z lerobot-twin grasp --mode dry "
              f"(kolko ma promien {DOT_MAX_PX:g} px).", file=sys.stderr)
        return 2
    value = float(value)
    if not (math.isfinite(value) and value >= 0.0):
        print(f"--error-px {value}: oczekiwano liczby >= 0", file=sys.stderr)
        return 2
    try:
        rig = _update_rig(Rig.load(getattr(a, "rig", None)).path, "wrist", {"dot_error_px": value})
    except RigError as exc:
        print(f"Plik wartosci stanowiska chwytu jest bledny: {exc}", file=sys.stderr)
        return 2
    if value <= DOT_MAX_PX:
        print(f"Blad kropki {value:g} px - w granicy {DOT_MAX_PX:g} px. Zapisano do {rig.path}.", flush=True)
    else:
        print(f"Blad kropki {value:g} px - PONAD {DOT_MAX_PX:g} px (dot off): potrzebna ponowna kalibracja "
              f"kamery na nadgarstku; wyjatek tylko przez overrides.dot. Zapisano do {rig.path}.", flush=True)
    return 0


def bench_show(a: argparse.Namespace) -> int:
    """Prints the per-rig file and what each mode still misses on a real backend."""
    try:
        rig = Rig.load(getattr(a, "rig", None))
    except RigError as exc:
        print(f"Plik wartosci stanowiska chwytu jest bledny: {exc}", file=sys.stderr)
        return 2
    exists = rig.path is not None and Path(rig.path).is_file()
    print(f"Plik: {rig.path}" + ("" if exists else " (brak - nic nie zmierzono)"))
    print(json.dumps(rig.to_dict(), indent=2, ensure_ascii=False))
    now = time.time()
    for name, age in rig.taught_age_s(now).items():
        label = {"table": "stol", "grasp": "punkt chwytu"}[name]
        if age is None:
            print(f"{label}: nie nauczony")
        else:
            state = "wazne" if age <= TEACH_MAX_AGE_S else "PRZETERMINOWANE"
            print(f"{label}: nauczony {age / 3600:.1f} h temu ({state}, "
                  f"limit {TEACH_MAX_AGE_S / 3600:.0f} h)")
    print("Czego brakuje na prawdziwym ramieniu:")
    for mode in ("dry", "shadow", "real"):
        missing = rig.missing(mode, simulated=False, now=now)
        print(f"  {mode:<7s} " + (", ".join(missing) if missing else "nic - moze ruszac"))
    print("  sim     nic (wartosci symulacji z Rig.for_sim)")
    print("Zawsze przy uruchomieniu (z ramieniem i stanowiskiem): table not under grasp, "
          "joint offsets changed, joint offsets differ, gripper freeze differs.")
    return 0


def main_bench(a: argparse.Namespace) -> int:
    """`lerobot-twin grasp-bench gripper|table|point|latency|dot|show`."""
    what = getattr(a, "action", None)
    if what not in TOOLS:
        print(f"lerobot-twin grasp-bench: nieznane narzedzie {what!r} ({', '.join(TOOLS)})", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if what == "show":
        return bench_show(a)
    if what == "dot":
        return bench_dot(a)
    keys = Keys()
    keys.start()
    try:
        if what == "gripper":
            return bench_gripper(a, keys)
        if what == "latency":
            return bench_latency(a, keys)
        return bench_teach(a, keys, what)
    except KeyboardInterrupt:
        print("\nPrzerwano (Ctrl-C).", file=sys.stderr)
        return 1
