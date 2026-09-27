"""One grasp episode, tick by tick: the student policy on the wrist camera, driving the arm through the twin.

    drv = GraspDriver(twin, policy, frames, camera, rig, mode="sim")
    drv.check()                        # refusal reason or None; moves nothing
    drv.take()                         # claim and engage, once per session
    drv.preposition(hover)             # straight up when low, staged plan, jaw opened at the hover
    report = drv.episode(hover)        # blocks the caller until an ending
    drv.close()

Each 20 Hz tick reads the newest fresh joint sample, the newest wrist frame and the joint sample
nearest the frame's exposure time, raises the floor from the measured tool height, builds the
19-vector, asks the policy, moves the command by one bookkeeping step, accepts it through one
warm IK step and one collision question, and sends all seven joints. A soft guard (stale feedback,
camera, pairing, lag) skips a tick and sends the previous target again; five in a row (ten for
lag) end the episode. Every ending holds the arm in one call (`ArmSession.hold`).

Modes: `sim` moves only a simulated arm; `real` needs consent; `shadow` runs the policy on the
real arm with the jaw always open and a floor above the object; `dry` never takes the arm and
only scores ticks (`dry_tick`).
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import IO, TYPE_CHECKING, Any, Literal

import numpy as np

from ..kinematics import RobotKinematics
from ..runtime import Twin
from .camera import CameraError, Frame, FrameSource, WristCamera
from .command import GraspCommand
from .contract import (
    ARM_SPEED_DEG_S,
    ARRIVE_STILL_DEG_S,
    CEILING,
    CLOSE_BELOW,
    FB_AGE_END_S,
    FLOOR_MARGIN,
    FRAME_AGE_MAX_S,
    IK_MAX_JUMP_DEG,
    JOINT_AGE_MAX_S,
    LAG_HOLD_DEG,
    LIFT,
    MAX_TICKS,
    OBJECT_HEIGHT_DEFAULT,
    PAIR_GAP_MAX_S,
    PREPOSITION_SPEED_DEG_S,
    SAG_FILTER_S,
    SAG_LIMIT,
    SHADOW_CLEAR,
    SHADOW_FLOOR_ABOVE_BELIEVED,
    STREAM_UNITS_S,
    TABLE_END_BELOW,
    TABLE_END_TICKS,
    TICK_S,
    TIP_AHEAD,
    TOUCH_EXPECTED_Z,
    TRAINING_FLOOR_Z,
)
from .gripper import GripperEstimator
from .hover import Hover
from .motion import ArmSession, Ended
from .observer import GraspObserver
from .rig import Rig

if TYPE_CHECKING:
    from ...robot.base import GripperState
    from .policy import GraspStudentPolicy

logger = logging.getLogger(__name__)

Mode = Literal["dry", "sim", "shadow", "real"]
MODES = ("dry", "sim", "shadow", "real")

FRAME_END_S = 0.5                      # a frame older than this ends the episode at once
SKIPS_END = 5                          # soft skips in a row that end an episode ...
SKIPS_END_LAG = 10                     # ... or, when the last one is `lag`, this many (0.5 s)
RATE_LIMITED_DEG = 0.5
DRY_EFFORT_PERIOD_S = 1.0
GRIP_OPEN = 100.0
# The measured-side floor starts at the command floor's own margin, not at FLOOR_MEASURED_MIN (2 mm):
# at full lateral speed the tool dips for 3 to 4 ticks while the raised floor brings the command back.
MEASURED_FLOOR_START = FLOOR_MARGIN
_BOX_CORNERS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


@dataclass
class TickRecord:                     # one JSON line per tick
    tick: int
    t: float
    kind: str                         # "policy" | "skipped"
    skip: str                         # "" or the soft guard that skipped the tick
    action: list[float] | None
    uv: list[float] | None
    p_cmd: list[float]
    yaw_cmd: float
    g_cmd: float
    attempts: int
    ik_ok: bool | None
    ik_why: str                       # "" | "ik" | "jump" | "collision"
    ik_pos_err: float | None
    ik_rot_err: float | None
    ik_jump_deg: float | None
    box_hit: bool
    floor_hit: bool
    ceiling_hit: bool
    floor_z: float
    sag: float
    tcp: list[float]
    gap: float
    lag_deg: float
    rate_limited: bool
    frame_t: float | None             # the frame's stamp ...
    t_exp: float | None               # ... and the exposure time used for pairing
    frame_age_s: float | None
    joint_age_s: float | None
    fb_age_s: float | None
    pair_gap_s: float | None
    infer_ms: float | None
    tick_ms: float


@dataclass(frozen=True)
class DryInputs:                      # what a dry tick scored, for the dump
    frame: Frame | None
    pair_joints: dict[str, float] | None
    inputs: np.ndarray | None         # (4, 120, 160) float32, policy.inputs(frame.image, uv)
    obs: np.ndarray | None            # (19,) float32, not normalised


@dataclass
class EpisodeReport:
    ending: str
    detail: str
    ticks: int                        # every tick run, skipped ones included
    skipped: int
    ik_refused: int
    collision_refused: int
    rate_limited: int
    attempts: int
    box_hits: int
    floor_hits: int
    floor_raised: int
    max_lag_deg: float
    min_tcp_z: float
    floor_z: float
    floor_z_end: float
    sag: float
    no_effort_rule: list[str]
    tick_ms: dict[str, float]
    infer_ms: dict[str, float]
    frame_age_ms: dict[str, float]
    joint_age_ms: dict[str, float]
    fb_age_ms: dict[str, float]
    log: Path | None
    loop_gaps: int = 0                # ticks that started more than one period late
    jump_refused: int = 0             # of `ik_refused`: steps over IK_MAX_JUMP_DEG, which training took


class _Skip(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class _Seen:
    frame: Frame | None = None
    pair_joints: dict[str, float] | None = None
    inputs: np.ndarray | None = None
    obs: np.ndarray | None = None


@dataclass
class _Run:
    """The state of one episode (or of one dry tick) between ticks."""

    hover: Hover
    cmd: GraspCommand
    est: GripperEstimator
    touch: float
    object_height: float
    sag_used: float
    sag_est: float
    t0: float
    floor_z0: float
    grip_cmd: float = GRIP_OPEN       # the jaw command after this tick's slew
    target: dict[str, float] = field(default_factory=dict)       # the last target sent
    skipped: int = 0
    streak: int = 0
    table_ticks: int = 0
    ik_refused: int = 0
    jump_refused: int = 0             # of `ik_refused`: steps over IK_MAX_JUMP_DEG, which training took
    collision_refused: int = 0
    rate_limited: int = 0
    box_hits: int = 0
    floor_hits: int = 0
    floor_raised: int = 0
    loop_gaps: int = 0
    max_lag: float = 0.0
    min_tcp_z: float = math.inf
    gap: float = math.nan
    n: int = 0                        # records made
    pending: TickRecord | None = None
    pending_target: dict[str, float] | None = None
    tick_ms: list[float] = field(default_factory=list)
    infer_ms: list[float] = field(default_factory=list)
    frame_age_ms: list[float] = field(default_factory=list)
    joint_age_ms: list[float] = field(default_factory=list)
    fb_age_ms: list[float] = field(default_factory=list)
    log: IO[str] | None = None
    log_path: Path | None = None


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    v = np.asarray(values, float)
    return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "max": float(np.max(v))}


def _plain(value: Any) -> Any:
    """JSON numbers only: a non-finite float becomes null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _same_hover(a: Hover, b: Hover) -> bool:
    return a is b or (np.array_equal(np.asarray(a.p), np.asarray(b.p)) and a.yaw == b.yaw
                      and a.joints == b.joints)


class GraspDriver:
    """One grasp session on one twin: check, take, then pre-positioning and an episode per grasp, close."""

    def __init__(self, twin: Twin, policy: GraspStudentPolicy, frames: FrameSource, camera: WristCamera,
                 rig: Rig, *, mode: Mode = "dry", real_ok: bool = False,
                 floor: Literal["bench", "training"] = "bench",
                 gap: Literal["scene", "model"] = "model",
                 jaw_stroke_s: float | None = None,
                 camera_latency_s: float | None = None,
                 success: Callable[[], bool] | None = None,
                 max_ticks: int = MAX_TICKS, max_seconds: float = 12.0,
                 log_dir: Path | None = None,
                 on_tick: Callable[[TickRecord], None] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if floor not in ("bench", "training"):
            raise ValueError(f"unknown floor {floor!r}")
        # Training's floor puts the fingertips 13 mm into the table: only a simulated table forgives it.
        if floor == "training" and mode != "sim":
            raise ValueError("the training floor exists in sim mode only")
        if gap not in ("scene", "model"):
            raise ValueError(f"unknown gap source {gap!r}")
        if jaw_stroke_s is not None and not (math.isfinite(jaw_stroke_s) and jaw_stroke_s > 0.0):
            raise ValueError(f"jaw_stroke_s must be a positive number, got {jaw_stroke_s}")
        self.twin = twin
        self.policy = policy
        self.frames = frames
        self.camera = camera
        self.rig = rig
        self.mode: Mode = mode
        self.real_ok = bool(real_ok)
        self.floor = floor
        self.gap = gap
        self.jaw_stroke_s = None if jaw_stroke_s is None else float(jaw_stroke_s)
        self.camera_latency_s = (rig.camera_latency_s() if camera_latency_s is None
                                 else float(camera_latency_s))
        self.success = success
        self.max_ticks = int(max_ticks)
        self.max_seconds = float(max_seconds)
        self.log_dir = None if log_dir is None else Path(log_dir)
        self.on_tick = on_tick
        self.clock = clock
        # One kinematics for the driver's thread: IK steps, the observation and the believed pixel.
        self.kin = RobotKinematics(twin.spec)
        self.observer = GraspObserver(self.kin)
        self.arm = ArmSession(twin, self.kin, rig, clock=clock, sleep=sleep)
        self.names: tuple[str, ...] = self.arm.arm
        self.grip = twin.spec.gripper
        tips = twin.spec.fingertips
        self._tips = [self.kin.model.geom(self.kin.prefix + g).id for g in tips] if all(tips) else []
        #: Detail of the last refusal, for the operator.
        self.detail = ""
        self._ready: Hover | None = None
        self._sag = 0.0
        self._episodes = 0
        self._dry_ticks = 0
        self._effort_t = -math.inf

    # ------------------------------------------------------------------ checks
    def _mode_refusal(self) -> str | None:
        tw = self.twin
        if self.mode == "sim" and not tw.status.simulated:
            self.detail = "sim mode on a real backend"
            return "not armed"
        if self.mode in ("shadow", "real") and not self.real_ok:
            self.detail = f"{self.mode} mode without consent"
            return "not armed"
        if self.mode != "dry":
            b = tw.backend
            if b is None or b.gripper_state() is None:
                self.detail = "backend has no gripper state"
                return "not armed"
        return None

    def check(self) -> str | None:
        """Refusal reason or None; moves nothing. A real backend is listened to 0.3 s for another sender."""
        self.detail = ""
        self.arm.detail = ""
        why = self.arm.check_basic() or self._mode_refusal() or self.arm.check_other_sender()
        self.detail = self.arm.detail or self.detail
        if why is None and self.mode == "dry":
            self.arm.listen()
        return why

    def take(self) -> str | None:
        """Claim and engage; None on success."""
        if self.mode == "dry":
            return "dry"
        self.detail = ""
        self.arm.detail = ""
        why = self.arm.check_basic() or self._mode_refusal()
        if why is None:
            why = self.arm.take()
        self.detail = self.arm.detail or self.detail
        return why

    def preposition(self, hover: Hover, speed_deg_s: float = PREPOSITION_SPEED_DEG_S) -> str | None:
        if self.mode == "dry":
            return "dry"
        self._ready = None
        reason, sag = self.arm.preposition(hover, speed_deg_s)
        self.detail = self.arm.detail
        if reason is None:
            self._ready, self._sag = hover, sag
            logger.info("at the hover (%.3f, %.3f, %.3f) m, sag %.1f mm", *hover.p, sag * 1000.0)
        return reason

    def stop(self, reason: str = "stop") -> None:
        """Ends the running episode or move with `reason`. Any thread."""
        self.arm.interrupt(reason)

    def close(self) -> None:
        """An open episode or pre-positioning on another thread ends with its own hold first."""
        self.arm.close()

    def _hold(self, ending: str) -> None:
        self.arm.hold(ending)

    # ------------------------------------------------------------------ floors
    def _floor_of(self, touch: float, sag: float, hover: Hover, object_height: float) -> float:
        """The command-side floor on the commanded TCP height."""
        if self.floor == "training":
            return TRAINING_FLOOR_Z
        bench = touch + FLOOR_MARGIN + sag
        if self.mode == "shadow":
            # An open finger can stand over the object: its commanded tip stays SHADOW_CLEAR above the top.
            return max(bench, touch + object_height + SHADOW_CLEAR + sag,
                       float(hover.believed[2]) + SHADOW_FLOOR_ABOVE_BELIEVED)
        return bench

    def _new_run(self, hover: Hover, touch: float, sag: float, now: float) -> _Run:
        height = self.rig.object_height()
        object_height = OBJECT_HEIGHT_DEFAULT if height is None else height
        floor = self._floor_of(touch, sag, hover, object_height)
        cmd = GraspCommand(hover.p, hover.yaw, hover.q, floor_z=floor)
        if self.jaw_stroke_s is not None:
            units_s = 100.0 / self.jaw_stroke_s
        else:
            units_s = self.rig.jaw_units_s() or STREAM_UNITS_S
        est = GripperEstimator(units_s=units_s, source=self.gap)
        est.reset(GRIP_OPEN, now)
        return _Run(hover=hover, cmd=cmd, est=est, touch=touch, object_height=object_height,
                    sag_used=sag, sag_est=sag, t0=now, floor_z0=floor)

    def _record(self, r: _Run, now: float) -> TickRecord:
        s = r.cmd.state
        rec = TickRecord(
            tick=r.n, t=now, kind="skipped", skip="", action=None, uv=None,
            p_cmd=[float(v) for v in s.p_cmd], yaw_cmd=float(s.yaw_cmd), g_cmd=float(s.g_cmd),
            attempts=int(s.attempts), ik_ok=None, ik_why="", ik_pos_err=None, ik_rot_err=None,
            ik_jump_deg=None, box_hit=False, floor_hit=False, ceiling_hit=False, floor_z=float(r.cmd.floor_z),
            sag=float(r.sag_used),
            tcp=[math.nan] * 3, gap=float(r.gap), lag_deg=math.nan, rate_limited=False,
            frame_t=None, t_exp=None, frame_age_s=None, joint_age_s=None, fb_age_s=None, pair_gap_s=None,
            infer_ms=None, tick_ms=0.0)
        r.n += 1
        return rec

    # ------------------------------------------------------------------ steps 3 to 12
    @staticmethod
    def _guard_end(ending: str, detail: str, dry: bool) -> None:
        if dry:
            raise _Skip(ending)
        raise Ended(ending, detail)

    def _score(self, r: _Run, rec: TickRecord, seen: _Seen, *, dry: bool) -> np.ndarray:
        """Joints, frame, pairing, lag, floor, gap, observation, policy, bookkeeping, IK.

        Returns the measured TCP.

        Raises `_Skip` for a soft guard and `Ended` for a hard one (in dry mode every guard is a skip).
        """
        tw, arm, names = self.twin, self.arm, self.names
        now = self.clock()

        # 3. Joints: the newest fresh sample; the feedback age by the backend's own clock.
        js = arm.history.newest()
        b = tw.backend
        gs = b.gripper_state() if b is not None else None
        fb_age = None if gs is None else gs.age_s
        rec.fb_age_s = fb_age
        joint_age = math.inf if js is None else now - js.t
        tcp_m = np.full(3, math.nan)
        if js is not None:
            rec.joint_age_s = joint_age
            tcp_m = self.observer.tcp(js.joints)
            rec.tcp = [float(v) for v in tcp_m]
            q_deg = np.degrees(r.cmd.state.q_cmd)
            rec.lag_deg = max(abs(float(q_deg[i]) - float(js.joints[j])) for i, j in enumerate(names))
        if joint_age > FB_AGE_END_S or (fb_age is not None and fb_age > FB_AGE_END_S):
            self._guard_end("stale feedback", f"joint sample {joint_age:.2f} s, feedback {fb_age} s old", dry)
        frozen = getattr(tw, "_frozen", None)
        if (joint_age > JOINT_AGE_MAX_S or (fb_age is not None and fb_age > JOINT_AGE_MAX_S)
                or (frozen is not None and frozen())):
            raise _Skip("stale feedback")
        assert js is not None

        # 4. Frame.
        fr: Frame | None = None
        try:
            fr = self.frames.latest()
        except CameraError as exc:
            self._guard_end("camera", str(exc), dry)
        frame_age = math.inf
        if fr is not None:
            seen.frame = fr
            frame_age = self.clock() - float(fr.t)
            rec.frame_t, rec.frame_age_s = float(fr.t), frame_age
            if frame_age > FRAME_END_S:
                self._guard_end("camera", f"frame {frame_age:.2f} s old", dry)
        if fr is None or frame_age > FRAME_AGE_MAX_S:
            raise _Skip("camera")

        # 5. Pairing at the exposure time: the frame's stamp is taken after read and decode.
        t_exp = float(fr.t) - self.camera_latency_s
        rec.t_exp = t_exp
        jp = arm.history.nearest(t_exp)
        if jp is None:
            raise _Skip("pairing")
        rec.pair_gap_s = abs(jp.t - t_exp)
        seen.pair_joints = dict(jp.joints)
        if rec.pair_gap_s > PAIR_GAP_MAX_S:
            raise _Skip("pairing")

        # 6. Tracking lag. Nothing commands the arm in dry mode, so there is no lag to hold for.
        if not dry and rec.lag_deg > LAG_HOLD_DEG:
            raise _Skip("lag")

        # 7. Measured floor.
        if self.floor != "training":
            low = self._tool_low(js.joints, float(tcp_m[2]), self._widest_opening(r, js.joints, gs))
            self._measured_floor(r, tcp_m, low, dry)
        rec.floor_z, rec.sag = float(r.cmd.floor_z), float(r.sag_used)

        # 8. Gap: never from status.command or status.measured.
        if self.mode == "shadow":
            gap = r.est.update(self.clock(), 100.0 * r.cmd.state.g_cmd, None)
        else:
            gap = r.est.update(self.clock(), r.grip_cmd, gs)
        r.gap = rec.gap = float(gap)

        # 9. Observation and the believed pixel at the exposure time.
        obs = self.observer.build(js.joints, gap, r.cmd)
        seen.obs = obs
        uv = self.camera.believed_uv(self.kin, jp.joints, r.hover.believed)
        rec.uv = [float(v) for v in uv]
        if dry:
            seen.inputs = self.policy.inputs(fr.image, uv)
        if not np.all(np.isfinite(obs)):
            self._guard_end("non-finite", f"observation {obs.tolist()}", dry)

        # 10. Policy; nothing is sent.
        t_inf = perf_counter()
        a = np.asarray(self.policy.act(fr.image, obs, uv), np.float32)
        rec.infer_ms = (perf_counter() - t_inf) * 1000.0
        rec.action = [float(v) for v in a.ravel()]
        if a.shape != (5,) or not np.all(np.isfinite(a)):
            self._guard_end("non-finite", f"action {a.tolist()}", dry)

        # 11. Bookkeeping: clips to the floor of step 7.
        prop = r.cmd.propose(a)
        rec.box_hit, rec.floor_hit, rec.ceiling_hit = prop.box_hit, prop.floor_hit, prop.ceiling_hit
        r.box_hits += int(prop.box_hit)
        r.floor_hits += int(prop.floor_hit)

        # 12. IK and collision; a refusal holds the command, as in training.
        q, step, why = arm.accept(prop.p, prop.R, r.cmd.state.q_cmd, collision=self.floor != "training")
        r.cmd.commit(prop, q)
        rec.ik_ok, rec.ik_why = q is not None, why
        rec.ik_pos_err, rec.ik_rot_err = float(step.pos_err), float(step.rot_err)
        rec.ik_jump_deg = float(step.jump_deg)
        if why == "collision":
            r.collision_refused += 1
        elif why:
            r.ik_refused += 1
            r.jump_refused += int(why == "jump")
        s = r.cmd.state
        rec.p_cmd, rec.yaw_cmd, rec.g_cmd, rec.attempts = [float(v) for v in s.p_cmd], float(s.yaw_cmd), \
            float(s.g_cmd), int(s.attempts)
        rec.kind = "policy"
        return tcp_m

    def _widest_opening(self, r: _Run, joints: Mapping[str, float], gs: GripperState | None) -> float | None:
        """The widest jaw opening that can be true now: on the A1X the reported opening is the last
        command, which leads the jaw by up to 1.3 s."""
        if self.grip is None:
            return None
        values = [joints.get(self.grip), r.target.get(self.grip)]
        if gs is not None:
            values += [gs.streamed, gs.position]
        known = [float(v) for v in values if v is not None]
        return max(known) if known else None

    def _tool_low(self, joints: Mapping[str, float], tcp_z: float, opening: float | None = None) -> float:
        """The measured tool height as a TCP height: the lower of the TCP and the lowest fingertip
        corner plus `TIP_AHEAD`, with the jaw at `opening` when given.

        At full lateral speed the wrist lags and the tool tilts: a fingertip corner then sits 1 to
        2 mm lower than the TCP height says, and an open finger rolled 5 deg by joint 5 sits 5 mm lower."""
        if not self._tips:
            return tcp_z
        kin = self.kin
        pose = {j: float(joints[j]) for j in kin.spec.joints if j in joints}
        if opening is not None and self.grip is not None:
            pose[self.grip] = float(opening)
        kin.tcp(pose)
        m, d = kin.model, kin.data
        Ti = kin._base_inv()
        low = math.inf
        for g in self._tips:
            corners = d.geom_xpos[g] + (_BOX_CORNERS * m.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3).T
            low = min(low, float(np.min(corners @ Ti[2, :3])) + float(Ti[2, 3]))
        return min(tcp_z, low + TIP_AHEAD)

    def _measured_floor(self, r: _Run, tcp_m: np.ndarray, low: float, dry: bool) -> None:
        """Step 7. `low` is the measured tool height as a TCP height (`_tool_low`)."""
        # (a) Sag grows with reach; estimated only while the arm is nearly still (tracking lag is not sag).
        # Nothing commands the arm in dry mode: the difference would be its distance from the hover.
        speed = self.arm.history.speed_deg_s()
        if not dry and speed and all(abs(speed.get(j, math.inf)) < ARRIVE_STILL_DEG_S for j in self.names):
            q_cmd = {j: math.degrees(float(v)) for j, v in zip(self.names, r.cmd.state.q_cmd, strict=True)}
            s = float(self.kin.tcp(q_cmd)[2, 3]) - float(tcp_m[2])
            r.sag_est += (s - r.sag_est) * min(1.0, TICK_S / SAG_FILTER_S)
            r.sag_used = min(max(max(r.sag_used, r.sag_est), 0.0), SAG_LIMIT)
        # (b) The command-side floor with the new sag.
        floor = self._floor_of(r.touch, r.sag_used, r.hover, r.object_height)
        # (c) A measured tool under the start height raises the floor by the deficit, from the floor
        # itself: after a dive the command is already rising while the tool still overshoots, and a
        # floor lifted to that command would hold every later attempt 5 to 10 mm higher.
        d = r.touch + MEASURED_FLOOR_START - low
        if d > 0.0:
            floor = max(floor, min(max(float(r.cmd.floor_z), floor) + d, r.touch + FLOOR_MARGIN + SAG_LIMIT))
            r.floor_raised += 1
        # (d) The floor only rises within an episode, and never above the ceiling.
        r.cmd.floor_z = min(max(r.cmd.floor_z, floor), float(r.cmd.state.p0[2]) + CEILING)
        # (e) On the table.
        if low < r.touch + TABLE_END_BELOW:
            r.table_ticks += 1
            if r.table_ticks >= TABLE_END_TICKS and not dry:
                raise Ended("table", f"measured tool {low * 1000:.1f} mm "
                                     f"(TCP {float(tcp_m[2]) * 1000:.1f} mm), touch {r.touch * 1000:.1f} mm")
        else:
            r.table_ticks = 0

    # ------------------------------------------------------------------ the episode
    def _empty_report(self, ending: str, detail: str) -> EpisodeReport:
        return EpisodeReport(ending=ending, detail=detail, ticks=0, skipped=0, ik_refused=0,
                             collision_refused=0, rate_limited=0, attempts=0, box_hits=0, floor_hits=0,
                             floor_raised=0,
                             max_lag_deg=0.0, min_tcp_z=math.nan, floor_z=math.nan, floor_z_end=math.nan,
                             sag=0.0, no_effort_rule=list(self.arm.no_effort_rule), tick_ms={}, infer_ms={},
                             frame_age_ms={}, joint_age_ms={}, fb_age_ms={}, log=None)

    def episode(self, hover: Hover) -> EpisodeReport:
        """One episode from the hover, on the caller's thread, until an ending."""
        if self.mode == "dry":
            return self._empty_report("dry", "dry mode moves nothing")
        ready, self._ready = self._ready, None
        if ready is None or not _same_hover(ready, hover):
            self._hold("error")
            return self._empty_report("error", "pre-positioning has not succeeded for this hover")
        touch = self.rig.touch_tcp_z()
        if touch is None:
            self._hold("error")
            return self._empty_report("error", "table not taught")

        self._episodes += 1
        now = self.clock()
        try:
            r = self._new_run(hover, touch, self._sag, now)
        except Exception as exc:
            logger.exception("grasp episode: set-up failed")
            self._hold("error")
            return self._empty_report("error", f"{type(exc).__name__}: {exc}"[:200])
        r.target = {j: float(hover.joints[j]) for j in self.names}
        if self.grip is not None:
            r.target[self.grip] = GRIP_OPEN
        with self.arm.busy():
            try:
                ending, detail = self._start(r)
                if not ending:
                    ending, detail = self._loop(r)
            except Exception as exc:
                logger.exception("grasp episode: error in a tick")
                ending, detail = "error", f"{type(exc).__name__}: {exc}"[:200]
            except BaseException:
                self._finish(r, "stop", "interrupted")
                raise
            return self._finish(r, ending, detail)

    def _start(self, r: _Run) -> tuple[str, str]:
        """Guard, speed check and log; ("", "") when the ticks may run."""
        try:
            self.arm.check_arm()
            self.arm.guard(lag_deg=3.0, lag_s=0.15)
            self.twin.set_speed(None)
            limits = self.twin.speed_limits()
            slow = {j: limits.get(j, 0.0) for j in self.names if limits.get(j, 0.0) < ARM_SPEED_DEG_S - 1e-6}
            if slow:
                return "speed limited", ", ".join(f"{j} {v:.0f} deg/s" for j, v in slow.items())
            # The ticks run capped at exactly 90: the lag and tracking rules compare against this slew.
            self.twin.set_speed(ARM_SPEED_DEG_S)
        except Ended as e:
            return e.ending, e.detail
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            r.log_path = self.log_dir / f"episode_{self._episodes}.jsonl"
            r.log = r.log_path.open("w", encoding="utf-8")
        hover = r.hover
        logger.info("episode %d (%s): hover (%.3f, %.3f, %.3f) m, yaw %.1f deg, floor %.1f mm, sag %.1f mm",
                    self._episodes, self.mode, *hover.p, math.degrees(hover.yaw), r.cmd.floor_z * 1000.0,
                    r.sag_used * 1000.0)
        return "", ""

    def _loop(self, r: _Run) -> tuple[str, str]:
        next_t = self.clock() + TICK_S
        try:
            while True:
                self._tick(r)
                now = self.clock()
                if now - next_t > TICK_S:
                    r.loop_gaps += 1
                    logger.warning("grasp loop %.0f ms late", (now - next_t) * 1000.0)
                    next_t = now + TICK_S
                self.arm.wait(next_t - now)
                next_t += TICK_S
        except Ended as e:
            return e.ending, e.detail

    def _tick(self, r: _Run) -> None:
        t_perf = perf_counter()
        now = self.clock()
        prev = self._judge(r)

        # 1. Hard checks, 2. caps. An ending here holds before the previous record is written.
        try:
            self.arm.check_arm()
            if r.cmd.state.ticks + r.skipped >= self.max_ticks:
                raise Ended("tick cap")
            if now - r.t0 > self.max_seconds:
                raise Ended("time cap")
        except Ended:
            r.pending, r.pending_target = prev, None
            raise
        if prev is not None:
            self._emit(r, prev)

        rec = self._record(r, now)
        try:
            tcp_m = self._score(r, rec, _Seen(), dry=False)
        except _Skip as s:
            rec.kind, rec.skip = "skipped", s.reason
            r.skipped += 1
            r.streak += 1
            self._keep_ages(r, rec)
            self._send(r, rec, r.target, t_perf)
            limit = SKIPS_END_LAG if s.reason == "lag" else SKIPS_END
            if r.streak >= limit:
                raise Ended(s.reason, f"{r.streak} soft skips in a row") from None
            return
        except Exception as exc:
            # The tick that explains a hard ending goes to the log too.
            rec.kind, rec.skip = "skipped", exc.ending if isinstance(exc, Ended) else "error"
            self._keep_ages(r, rec)
            rec.tick_ms = (perf_counter() - t_perf) * 1000.0
            r.pending, r.pending_target = rec, None
            raise
        r.streak = 0
        r.pending, r.pending_target = rec, None
        self._keep_ages(r, rec)
        if math.isfinite(rec.lag_deg):
            r.max_lag = max(r.max_lag, rec.lag_deg)
        r.min_tcp_z = min(r.min_tcp_z, float(tcp_m[2]))
        if rec.infer_ms is not None:
            r.infer_ms.append(rec.infer_ms)

        # 13. Gripper command, slewed at the measured jaw speed when asked; the shadow jaw stays open.
        opening = 100.0 * r.cmd.state.g_cmd
        if self.jaw_stroke_s is not None:
            step = 100.0 / self.jaw_stroke_s * TICK_S
            r.grip_cmd += min(max(opening - r.grip_cmd, -step), step)
        else:
            r.grip_cmd = opening
        grip = GRIP_OPEN if self.mode == "shadow" else r.grip_cmd

        # 14. Send all seven joints.
        target = {j: math.degrees(float(v)) for j, v in zip(self.names, r.cmd.state.q_cmd, strict=True)}
        if self.grip is not None:
            target[self.grip] = float(grip)
        if not all(math.isfinite(v) for v in target.values()):
            raise Ended("non-finite", f"target {target}")
        self._send(r, rec, target, t_perf)

        # 15. Success (sim, training's condition) or lift (real arm).
        if self.mode == "shadow":
            return
        g = r.cmd.state.g_cmd
        if self.success is not None:
            if g < CLOSE_BELOW and self.success():
                raise Ended("success")
        elif g < CLOSE_BELOW and float(tcp_m[2]) >= r.min_tcp_z + LIFT:
            raise Ended("lifted", f"measured TCP {float(tcp_m[2]) * 1000:.1f} mm, "
                                  f"lowest {r.min_tcp_z * 1000:.1f} mm")

    def _send(self, r: _Run, rec: TickRecord, target: dict[str, float], t_perf: float) -> None:
        """Step 14's write, after one more look at the hard checks: an impact or a STOP that came while
        the tick computed ends it here, not one policy target further into whatever was hit."""
        r.pending, r.pending_target = rec, None
        rec.tick_ms = (perf_counter() - t_perf) * 1000.0
        self.arm.check_arm()
        self.arm.command(target)
        r.target = target
        rec.tick_ms = (perf_counter() - t_perf) * 1000.0
        r.tick_ms.append(rec.tick_ms)
        r.pending_target = dict(target)

    @staticmethod
    def _keep_ages(r: _Run, rec: TickRecord) -> None:
        for value, into in ((rec.frame_age_s, r.frame_age_ms), (rec.joint_age_s, r.joint_age_ms),
                            (rec.fb_age_s, r.fb_age_ms)):
            if value is not None and math.isfinite(value):
                into.append(value * 1000.0)

    def _judge(self, r: _Run) -> TickRecord | None:
        """Takes the previous tick's pending record and sets its rate-limit verdict from the twin's
        command now (step 16); `_emit` then writes it."""
        rec, target = r.pending, r.pending_target
        if rec is None:
            return None
        r.pending = r.pending_target = None
        command = self.twin.status.command
        if target and command:
            diffs = [abs(float(target[j]) - float(command[j]))
                     for j in self.names if j in target and j in command]
            rec.rate_limited = bool(diffs) and max(diffs) > RATE_LIMITED_DEG
            r.rate_limited += int(rec.rate_limited)
        return rec

    def _emit(self, r: _Run, rec: TickRecord) -> None:
        if r.log is not None:
            r.log.write(json.dumps(_plain(asdict(rec))) + "\n")
        if self.on_tick is not None:
            try:
                self.on_tick(rec)
            except Exception:
                logger.exception("on_tick raised")

    def _finish(self, r: _Run, ending: str, detail: str) -> EpisodeReport:
        # The hold comes first: the log write and the caller's `on_tick` must not keep the arm pushing.
        rec = self._judge(r)
        self._hold(ending)
        if rec is not None:
            try:
                self._emit(r, rec)
            except Exception:
                logger.exception("writing the last tick record failed")
        if r.log is not None:
            r.log.close()
            r.log = None
        report = EpisodeReport(
            ending=ending, detail=detail, ticks=r.cmd.state.ticks + r.skipped, skipped=r.skipped,
            ik_refused=r.ik_refused, collision_refused=r.collision_refused, rate_limited=r.rate_limited,
            attempts=int(r.cmd.state.attempts), box_hits=r.box_hits, floor_hits=r.floor_hits,
            floor_raised=r.floor_raised, max_lag_deg=r.max_lag,
            min_tcp_z=r.min_tcp_z if math.isfinite(r.min_tcp_z) else math.nan,
            floor_z=r.floor_z0, floor_z_end=float(r.cmd.floor_z), sag=float(r.sag_used),
            no_effort_rule=list(self.arm.no_effort_rule), tick_ms=_stats(r.tick_ms),
            infer_ms=_stats(r.infer_ms), frame_age_ms=_stats(r.frame_age_ms),
            joint_age_ms=_stats(r.joint_age_ms), fb_age_ms=_stats(r.fb_age_ms),
            log=r.log_path, loop_gaps=r.loop_gaps, jump_refused=r.jump_refused)
        level = logging.INFO if ending in ("success", "lifted", "tick cap", "time cap") else logging.WARNING
        logger.log(level, "episode %d ended: %s%s; %d ticks (%d skipped), %d attempts, IK refused %d "
                   "(%d over %.0f deg), collision refused %d, lowest TCP %.1f mm", self._episodes, ending,
                   f" ({detail})" if detail else "", report.ticks, report.skipped, report.attempts,
                   report.ik_refused, r.jump_refused, IK_MAX_JUMP_DEG, report.collision_refused,
                   report.min_tcp_z * 1000.0)
        return report

    # ------------------------------------------------------------------ dry mode
    def dry_tick(self, hover: Hover) -> tuple[TickRecord, DryInputs]:
        """Steps 3 to 12 on a fresh tick-0 command at the hover; sends nothing, ends nothing."""
        self.arm.listen()
        now = self.clock()
        if now - self._effort_t >= DRY_EFFORT_PERIOD_S:
            self._effort_t = now
            self.arm.log_efforts("dry")
        t_perf = perf_counter()
        touch = self.rig.touch_tcp_z()
        r = self._new_run(hover, TOUCH_EXPECTED_Z if touch is None else touch, 0.0, now)
        r.n = self._dry_ticks
        self._dry_ticks += 1
        rec = self._record(r, now)
        seen = _Seen()
        try:
            self._score(r, rec, seen, dry=True)
        except _Skip as s:
            rec.kind, rec.skip = "skipped", s.reason
        rec.tick_ms = (perf_counter() - t_perf) * 1000.0
        return rec, DryInputs(frame=seen.frame, pair_joints=seen.pair_joints, inputs=seen.inputs,
                              obs=seen.obs)
