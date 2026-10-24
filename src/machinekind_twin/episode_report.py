"""What an episode log says about the arm: the latency of every joint and the gripper's drift against the cameras.

    log = EpisodeLog.load(path)
    print("\\n".join(report(log, gripper="chwytak").lines()))

Latency. A log holds three signals per joint on one clock: the intent (the strategy's target, per
step), the command (what the supervisor sent to the servos, per loop tick) and the readings. For
each pair `delay_fit` finds the time shift that best lays the earlier signal over the later one.
Intent to readings is the whole actuator latency, from a decision to the motion. Intent to
command is the supervisor's ramp at `max_vel`, so it grows with the length of the move. Command
to readings is the servo itself. Within one loop tick the twin reads before it sends, so one tick
of that ordering is part of every delay measured here, in simulation and on hardware.

Drift. A label the operator names marks the gripper among the objects the cameras see
(`--gripper ETYKIETA`). Where the cameras saw it (source "kamery"), the report compares its
position with the TCP that the forward kinematics give for the readings at the frames' capture
time: readings minus cameras, in the robot's base frame. A pose "w dloni" (in the gripper) is
the kinematics times a fixed offset, so it is no evidence and is skipped.

Without an intent, a command or a camera pose, the report says so in one sentence instead. The
latency needs only numpy; the drift needs the robot's package and MuJoCo for the kinematics.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .episode import EpisodeLog

__all__ = ["DecisionStats", "DelayFit", "EpisodeReport", "GripperDrift", "JointLatency", "delay_fit", "report"]

#: A reference that moved less than this [app units] says nothing about a delay.
MIN_MOTION = 1.0
#: A sample counts when the reference changed by more than this [app units] shortly before it.
ACTIVE_CHANGE = 0.05
#: Fewer active samples than this give no delay.
MIN_SAMPLES = 10
#: Shifts whose error is within this factor, or this many app units, of the best one form the band.
BAND_FACTOR = 1.05
BAND_UNITS = 0.05
#: A shift that explains less than half of the error is flagged.
RESIDUAL_FLAG = 0.5
#: A share of samples at a joint limit above this is flagged.
AT_LIMIT_FLAG = 0.05
#: The TCP slower than this [m/s] counts as at rest.
REST_SPEED = 0.010
#: Half the interval the TCP speed is measured over [s].
SPEED_HALF_S = 0.05

#: The source of a pose the cameras measured (`ObjectPose.source`).
CAMERAS = "kamery"
DEGREES = "st."
GRIPPER_UNITS = "jedn."

STILL = "staw nie ruszal sie"
NOT_NAMED = "strategia nie podawala celu temu stawowi"
NOTHING_SENT = "nic nie poszlo do serw"
NO_INTENT = "brak zamiaru w zapisie"

NO_LOOP = "Brak taktow petli w zapisie: opoznien nie licze."
NEITHER = "Opoznien napedow nie licze: w zapisie nie ma ani zamiaru, ani rozkazu do serw."
ONLY_ACTUATOR = "Brak zamiaru w zapisie (nikt nie podawal celu): licze tylko rozkaz->odczyt."
ONLY_TOTAL = "Rozkaz do serw nie przechodzi przez blizniaka: licze tylko zamiar->odczyt."

DRIFT = "Dryf chwytaka wzgledem kamer"


@dataclass(frozen=True)
class DelayFit:
    """The shift that best lays a reference over its samples."""

    #: The best shift [s].
    lag_s: float
    #: The shifts whose error is close to the best one's [s].
    band_s: tuple[float, float]
    #: RMS of the samples minus the reference without a shift [app units]: the error the delay makes.
    error_rms: float
    #: RMS left at the best shift [app units].
    residual_rms: float
    #: Median of the samples minus the shifted reference [app units]: a constant offset, not a delay.
    bias: float
    #: Samples taken while the reference moved.
    samples: int
    #: The best shift is the longest scanned: the true delay may be longer.
    at_bound: bool
    #: Share of those samples with the joint at a limit.
    at_limit_share: float = 0.0


@dataclass(frozen=True)
class JointLatency:
    """The three delays of one joint; each a `DelayFit`, or why there is none."""

    joint: str
    unit: str
    #: Intent to readings: the whole latency.
    total: DelayFit | str
    #: Intent to command: the supervisor's ramp.
    supervisor: DelayFit | str
    #: Command to readings: the servo.
    actuator: DelayFit | str


@dataclass(frozen=True)
class GripperDrift:
    """Readings minus cameras for the gripper: the TCP from the readings minus the camera-seen pose."""

    label: str
    samples: int
    #: Poses of the label that are no camera evidence, or could not be aligned, by why.
    skipped: dict[str, int]
    #: Mean difference [m], shape (3,), base frame.
    mean_m: np.ndarray
    rms_m: float
    max_m: float
    #: Median age of the camera poses [s].
    median_age_s: float
    #: Samples with the TCP slower than `REST_SPEED`: timing errors vanish there.
    at_rest_samples: int
    at_rest_rms_m: float | None
    #: Angle between the camera's rotation and the readings'; None when the cameras give positions only.
    angle_median_deg: float | None
    angle_max_deg: float | None


@dataclass(frozen=True)
class DecisionStats:
    """The model decisions the strategy logged."""

    count: int
    without_confidence: int
    response_median_s: float | None
    response_max_s: float | None
    confidence_mean: float | None


@dataclass(frozen=True, eq=False)
class EpisodeReport:
    """The report on one log; `lines` gives it as Polish text."""

    log: EpisodeLog
    latency: tuple[JointLatency, ...]
    #: One sentence when the log lacks an intent or a command; "" when it has both.
    latency_note: str
    drift: GripperDrift | None
    #: Why there is no drift; "" when there is.
    drift_note: str
    decisions: DecisionStats = field(default_factory=lambda: DecisionStats(0, 0, None, None, None))

    def lines(self) -> list[str]:
        return [*self._about(), *self._latency_lines(), *self._drift_lines(), self._decision_line()]

    def _about(self) -> list[str]:
        log, end = self.log, self.log.end
        header = log.header
        strategy = header.get("strategy")
        if isinstance(strategy, dict):
            mode = f"strategia {strategy.get('class', '?')} {_short(strategy.get('hz'))} Hz"
        else:
            mode = "obserwacja"
        where = "symulacja" if header.get("simulated") else "sprzet"
        length = "?" if end.t is None else f"{end.t:.2f}"

        if end.code == "unfinished":
            ending = "Koniec: zapis urwany [unfinished]"
        else:
            ending = f"Koniec: {end.reason} [{end.code}]"
        success = "-" if end.success is None else ("tak" if end.success else "nie")
        ending += f", sukces: {success}"
        if end.error is not None:
            ending += f", blad zadania {end.error:.3g}"

        return [
            f"Przebieg {header.get('name', '?')} ({log.path})",
            (f"{header.get('robot', '?')}, {where}, {mode}, {length} s, krokow: {len(log.steps)}, "
             f"taktow petli: {len(log.loop_t)}"),
            ending,
        ]

    def _latency_lines(self) -> list[str]:
        lines = []
        if self.latency_note:
            lines.append(self.latency_note)
        if not self.latency:
            return lines

        lines.append("Opoznienie (zamiar->odczyt = calosc; zamiar->rozkaz = rampa nadzoru, rosnie z dlugoscia "
                     "ruchu; rozkaz->odczyt = naped):")
        lines.append(f"  {'staw':<14}{'zamiar->odczyt':<20}{'zamiar->rozkaz':<20}{'rozkaz->odczyt':<20}"
                     f"{'blad RMS':<13}{'reszta':<13}stale")
        notes = []
        for joint in self.latency:
            main = next((fit for fit in (joint.total, joint.actuator) if isinstance(fit, DelayFit)), None)
            errors = "" if main is None else (f"{_units(main.error_rms, joint.unit):<13}"
                                              f"{_units(main.residual_rms, joint.unit):<13}"
                                              f"{_units(main.bias, joint.unit, signed=True)}")
            cells = "".join(f"{_cell(fit):<20}" for fit in (joint.total, joint.supervisor, joint.actuator))
            lines.append(f"  {joint.joint:<14}{cells}{errors}".rstrip())
            notes.extend(_joint_notes(joint, said={ONLY_ACTUATOR: NO_INTENT, ONLY_TOTAL: NOTHING_SENT}.get(
                self.latency_note, "")))

        return lines + notes

    def _drift_lines(self) -> list[str]:
        drift = self.drift
        if drift is None:
            return [self.drift_note]

        lines = [(f"{DRIFT} ('{drift.label}', pomiarow: {drift.samples}, wiek kamer mediana "
                  f"{drift.median_age_s * 1000:.0f} ms):")]
        mean = ", ".join(f"{round(value * 1000, 1) + 0.0:+.1f}" for value in drift.mean_m)
        if drift.at_rest_rms_m is None:
            rest = "w spoczynku: brak pomiarow"
        else:
            rest = f"w spoczynku: RMS {drift.at_rest_rms_m * 1000:.1f} mm (pomiarow: {drift.at_rest_samples})"
        lines.append(f"  odczyty - kamery: srednio [{mean}] mm, RMS {drift.rms_m * 1000:.1f} mm, "
                     f"max {drift.max_m * 1000:.1f} mm; {rest}")
        if drift.angle_median_deg is None or drift.angle_max_deg is None:
            lines.append("  obrot: kamery podaja tylko polozenie")
        else:
            lines.append(f"  obrot: mediana {drift.angle_median_deg:.1f} st., max {drift.angle_max_deg:.1f} st.")
        if drift.skipped:
            skipped = ", ".join(f"{why} {count}" for why, count in sorted(drift.skipped.items()))
            lines.append(f"  pominiete: {skipped}")
        return lines

    def _decision_line(self) -> str:
        stats = self.decisions
        line = f"Decyzje modelu: {stats.count}"
        if stats.response_median_s is not None and stats.response_max_s is not None:
            line += f", czas odpowiedzi mediana {stats.response_median_s:.2f} s, max {stats.response_max_s:.2f} s"
        if stats.confidence_mean is not None:
            line += f", pewnosc srednio {stats.confidence_mean:.2f}"
        if stats.without_confidence:
            line += f", bez pewnosci: {stats.without_confidence}"
        return line


def report(log: EpisodeLog, *, gripper: str | None = None, offset_m: Sequence[float] = (0.0, 0.0, 0.0),
           max_delay_s: float = 1.0) -> EpisodeReport:
    """The latency of every joint, the drift of the gripper's label and the model decisions.

    `gripper` is the label of the camera-seen object that marks the gripper, by default the one
    the run named (`header.gripper_label`); `offset_m` is where that marker sits in the TCP frame.
    """
    latency, latency_note = _latency(log, max_delay_s)
    label = gripper or log.header.get("gripper_label")
    drift, drift_note = _drift(log, str(label) if label else None, np.asarray(offset_m, float).reshape(3))

    return EpisodeReport(log, tuple(latency), latency_note, drift, drift_note, _decisions(log))


def delay_fit(ref_t: Any, ref_v: Any, prior: float, sample_t: Any, sample_v: Any, *, max_delay_s: float = 1.0,
              grid_s: float = 0.005, active_window_s: float = 0.25, at_limit: Any = None) -> DelayFit | str:
    """The shift of a held reference that best explains the samples, or why there is none.

    The reference holds each value from its time on (a zero-order hold) and `prior` before its
    first event; at a sample's time it holds the last value set strictly before. Only samples
    taken while the reference moved within `max_delay_s + active_window_s` before them count, the
    same set for every shift, so the errors compare.
    """
    ref_t, ref_v = np.asarray(ref_t, float), np.asarray(ref_v, float)
    usable = np.isfinite(ref_t) & np.isfinite(ref_v)
    ref_t, ref_v = ref_t[usable], ref_v[usable]
    if not len(ref_t):
        return STILL
    if not math.isfinite(prior):
        prior = float(ref_v[0])
    if np.ptp(np.r_[prior, ref_v]) < MIN_MOTION:
        return STILL

    sample_t, sample_v = np.asarray(sample_t, float), np.asarray(sample_v, float)
    limits = np.zeros(len(sample_t), bool) if at_limit is None else np.asarray(at_limit, bool)
    usable = np.isfinite(sample_t) & np.isfinite(sample_v) & (sample_t >= ref_t[0])
    sample_t, sample_v, limits = sample_t[usable], sample_v[usable], limits[usable]

    active = _active(ref_t, ref_v, prior, sample_t, max_delay_s + active_window_s)
    count = int(active.sum())
    if count < MIN_SAMPLES:
        return f"za malo odczytow w ruchu ({count})"

    times, values = sample_t[active], sample_v[active]
    shifts = np.arange(round(max_delay_s / grid_s) + 1) * grid_s
    errors = np.array([_rms(values - _held(ref_t, ref_v, prior, times - shift)) for shift in shifts])
    best = int(np.argmin(errors))
    lowest = float(errors[best])
    near = shifts[errors <= max(BAND_FACTOR * lowest, lowest + BAND_UNITS)]

    return DelayFit(
        lag_s=float(shifts[best]),
        band_s=(float(near.min()), float(near.max())),
        error_rms=float(errors[0]),
        residual_rms=lowest,
        bias=float(np.median(values - _held(ref_t, ref_v, prior, times - shifts[best]))),
        samples=count,
        at_bound=bool(shifts[best] >= max_delay_s - grid_s / 2),
        at_limit_share=float(limits[active].mean()),
    )


# --------------------------------------------------------------------------- latency


def _latency(log: EpisodeLog, max_delay_s: float) -> tuple[list[JointLatency], str]:
    if not len(log.loop_t):
        return [], NO_LOOP

    end_t = log.end.t if log.end.t is not None else float(np.nanmax(log.loop_t))

    reading_rows = log.reading_rows()
    reading_rows = reading_rows[log.measured_t[reading_rows] <= end_t + 1e-9]
    reading_t, readings = log.measured_t[reading_rows], log.measured[reading_rows]
    command_rows = log.command_rows()
    command_rows = command_rows[log.loop_t[command_rows] <= end_t + 1e-9]
    command_t, commands = log.loop_t[command_rows], log.sent[command_rows]
    intent_t, intents = log.intents()

    has_intent = bool(np.isfinite(intents).any())
    has_command = bool(len(command_rows))
    if not has_intent and not has_command:
        return [], NEITHER
    note = ONLY_ACTUATOR if not has_intent else ONLY_TOTAL if not has_command else ""

    def fit(ref_t, ref_v, prior, rows, sample_t, sample_v, column):
        return delay_fit(ref_t, ref_v, prior, sample_t, sample_v, max_delay_s=max_delay_s,
                         at_limit=log.at_limit[rows, column])

    latency = []
    for column, joint in enumerate(log.joints):
        unit = GRIPPER_UNITS if joint == log.header.get("gripper") else DEGREES

        named = np.isfinite(intents[:, column])
        if not has_intent:
            total = supervisor = NO_INTENT
        elif not named.any():
            total = supervisor = NOT_NAMED
        else:
            event_t, event_v = intent_t[named], intents[named, column]
            prior = _prior(event_t[0], command_t, commands[:, column], readings[:, column])
            total = fit(event_t, event_v, prior, reading_rows, reading_t, readings[:, column], column)
            supervisor = (fit(event_t, event_v, prior, command_rows, command_t, commands[:, column], column)
                          if has_command else NOTHING_SENT)

        sent = np.isfinite(commands[:, column])
        if sent.any():
            actuator = fit(command_t[sent], commands[sent, column], float(commands[sent, column][0]), reading_rows,
                           reading_t, readings[:, column], column)
        else:
            actuator = NOTHING_SENT

        latency.append(JointLatency(joint, unit, total, supervisor, actuator))

    return latency, note


def _prior(first_t: float, command_t: np.ndarray, command_v: np.ndarray, reading_v: np.ndarray) -> float:
    """Where the intent starts from: the last command before it (the twin ramps from there), else
    the first reading."""
    before = np.isfinite(command_v) & (command_t < first_t)
    if before.any():
        return float(command_v[before][-1])

    finite = reading_v[np.isfinite(reading_v)]
    return float(finite[0]) if len(finite) else math.nan


def _held(ref_t: np.ndarray, ref_v: np.ndarray, prior: float, at: np.ndarray) -> np.ndarray:
    index = np.searchsorted(ref_t, at, side="left") - 1
    return np.where(index >= 0, ref_v[np.clip(index, 0, None)], prior)


def _active(ref_t: np.ndarray, ref_v: np.ndarray, prior: float, sample_t: np.ndarray, span: float) -> np.ndarray:
    """Samples whose reference changed within `span` before them, counting the value held at the
    span's start."""
    starts = sample_t - span
    at_start = _held(ref_t, ref_v, prior, starts)
    first = np.searchsorted(ref_t, starts, side="right")
    last = np.searchsorted(ref_t, sample_t, side="right")

    active = np.zeros(len(sample_t), bool)
    for index, (low, high) in enumerate(zip(first, last, strict=True)):
        values = np.r_[at_start[index], ref_v[low:high]]
        active[index] = np.ptp(values) > ACTIVE_CHANGE
    return active


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2)))


def _cell(fit: DelayFit | str) -> str:
    if not isinstance(fit, DelayFit):
        return "-"
    if fit.at_bound:
        return f">= {fit.lag_s * 1000:.0f} ms (poza zakresem)"
    low, high = fit.band_s
    return f"{fit.lag_s * 1000:.0f} ms ({low * 1000:.0f}-{high * 1000:.0f})"


def _joint_notes(joint: JointLatency, said: str) -> list[str]:
    """Why a pair has no delay, and the flags of those that have one; `said` is a reason the
    latency note gave already."""
    notes = []
    for pair, fit in (("zamiar->odczyt", joint.total), ("zamiar->rozkaz", joint.supervisor),
                      ("rozkaz->odczyt", joint.actuator)):
        if not isinstance(fit, DelayFit):
            if fit != said:
                notes.append(f"  {joint.joint} {pair}: {fit}")
            continue
        if fit.error_rms > 0 and fit.residual_rms / fit.error_rms > RESIDUAL_FLAG:
            notes.append(f"  {joint.joint} {pair}: przesuniecie tlumaczy mniej niz polowe bledu - serwo "
                         "wolniejsze niz rozkaz, przeszkoda albo chwytak na obiekcie")
        if fit.at_limit_share > AT_LIMIT_FLAG:
            notes.append(f"  {joint.joint} {pair}: {fit.at_limit_share * 100:.0f}% odczytow na limicie")
    return notes


def _units(value: float, unit: str, signed: bool = False) -> str:
    return f"{value:+.1f} {unit}" if signed else f"{value:.1f} {unit}"


def _short(value: Any) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else "?"


# --------------------------------------------------------------------------- drift


def _drift(log: EpisodeLog, label: str | None, offset: np.ndarray) -> tuple[GripperDrift | None, str]:
    if not label:
        return None, f"{DRIFT}: nie licze - nie wskazano etykiety chwytaka (--gripper ETYKIETA)."

    skipped: collections.Counter[str] = collections.Counter()
    seen = []
    for step in log.steps:
        pose = (step.get("objects") or {}).get(label)
        if not isinstance(pose, dict):
            continue

        source = str(pose.get("source", ""))
        if source != CAMERAS:
            skipped[source or "bez zrodla"] += 1
            continue

        position = _array(pose.get("position"), (3,))
        rotation = _array(pose.get("rotation"), (3, 3))
        obs_t, age = _finite(step.get("obs_t")), _finite(pose.get("age_s"))
        if position is None or rotation is None or obs_t is None or age is None:
            skipped["niepelna poza"] += 1
            continue

        seen.append((obs_t - age, position, rotation, age))

    if not seen:
        return None, f"{DRIFT}: nie licze - kamery nie widzialy obiektu '{label}' (zrodlo '{CAMERAS}')."

    robot = str(log.header.get("robot") or "?")
    kinematics = _kinematics(log)
    if kinematics is None:
        return None, f"{DRIFT}: nie licze - brak opisu ramienia {robot}."

    reading_t, readings = log.readings()
    complete = np.isfinite(readings).all(axis=1)
    reading_t, readings = reading_t[complete], readings[complete]
    if len(reading_t) < 2:
        return None, f"{DRIFT}: nie licze - za malo odczytow ramienia."

    def tcp_at(t: float) -> np.ndarray | None:
        if not reading_t[0] <= t <= reading_t[-1]:
            return None
        joints = {joint: float(np.interp(t, reading_t, readings[:, column]))
                  for column, joint in enumerate(log.joints)}
        return kinematics.tcp(joints)

    differences, ages, angles, speeds = [], [], [], []
    for captured, position, rotation, age in seen:
        tcp = tcp_at(captured)
        if tcp is None:
            skipped["poza odczytami"] += 1
            continue

        differences.append(tcp[:3, :3] @ offset + tcp[:3, 3] - position)
        ages.append(age)
        if not np.array_equal(rotation, np.eye(3)):
            cosine = (np.trace(rotation.T @ tcp[:3, :3]) - 1.0) / 2.0
            angles.append(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))
        speeds.append(_speed(tcp_at, captured, offset))

    if not differences:
        return None, f"{DRIFT}: nie licze - zadna klatka nie wypada miedzy odczytami ramienia."

    difference = np.array(differences)
    distance = np.linalg.norm(difference, axis=1)
    at_rest = np.array([speed is not None and speed < REST_SPEED for speed in speeds])

    drift = GripperDrift(
        label=label,
        samples=len(difference),
        skipped=dict(skipped),
        mean_m=difference.mean(axis=0),
        rms_m=_rms(distance),
        max_m=float(distance.max()),
        median_age_s=float(np.median(ages)),
        at_rest_samples=int(at_rest.sum()),
        at_rest_rms_m=_rms(distance[at_rest]) if at_rest.any() else None,
        angle_median_deg=float(np.median(angles)) if angles else None,
        angle_max_deg=float(np.max(angles)) if angles else None,
    )
    return drift, ""


def _speed(tcp_at: Any, t: float, offset: np.ndarray) -> float | None:
    before, after = tcp_at(t - SPEED_HALF_S), tcp_at(t + SPEED_HALF_S)
    if before is None or after is None:
        return None
    moved = (after[:3, :3] @ offset + after[:3, 3]) - (before[:3, :3] @ offset + before[:3, 3])
    return float(np.linalg.norm(moved)) / (2 * SPEED_HALF_S)


def _kinematics(log: EpisodeLog) -> Any:
    """The robot's forward kinematics, built on this thread; None without its package or MuJoCo."""
    try:
        from .kinematics import RobotKinematics
        from .robots import get_spec

        spec = get_spec(str(log.header.get("robot") or ""))
        if tuple(spec.joints) != log.joints:
            return None
        return RobotKinematics(spec)
    except (ImportError, KeyError, ValueError, OSError):
        return None


def _array(value: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.shape == shape and np.isfinite(array).all() else None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


# --------------------------------------------------------------------------- decisions


def _decisions(log: EpisodeLog) -> DecisionStats:
    decisions = [decision for step in log.steps for decision in step.get("decisions") or ()
                 if isinstance(decision, dict)]
    responses = [value for value in (_finite(decision.get("response_s")) for decision in decisions)
                 if value is not None]
    confidences = [value for value in (_finite(decision.get("confidence")) for decision in decisions)
                   if value is not None]

    return DecisionStats(
        count=len(decisions),
        without_confidence=len(decisions) - len(confidences),
        response_median_s=float(np.median(responses)) if responses else None,
        response_max_s=float(np.max(responses)) if responses else None,
        confidence_mean=float(np.mean(confidences)) if confidences else None,
    )
