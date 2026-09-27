"""Following a point streamed from the glasses: a bounded tracker and one warm-started IK step per loop tick.

    f = Follower(kin, checker, spec, limits=twin.joint_limits(), joints=twin.status.command, volume=vol)
    f.v_max = 0.2                                      # m/s; the acceleration limit is 4 * v_max per s
    out = f.step(0.02, f.setpoint(pos), yaw)           # once per loop tick, on the loop thread
    if out.joints is not None:
        twin.set_target(out.joints, owner="specs")
    else:
        ...                                            # out.fail is "ik" or "collision"; nothing to send

Pure stepping: no threads, no clock and no twin. `SpecsReach` feeds it from a tick listener with the
loop's own time (`TickSample.t`), so a test with `connect(threaded=False)` is deterministic.

* **Tracker.** The commanded TCP point starts at the forward kinematics of the arm's command and moves
  towards the setpoint with its speed bounded by `v_max` and its acceleration by `a_max`, slowing down to
  stop at the setpoint. A far setpoint (the cursor entering the box on the other side) is therefore a ramp,
  not a jump; the twin's supervisor and the A1X backend only limit speed, not acceleration.
* **Rotation.** The tool points down with the jaws closing along `yaw` (the angle of the closing axis in the
  base xy plane; None = across the line from the base, as `down_rotation`). The jaws are symmetric, so
  `yaw` and `yaw + pi` are the same grasp: the one nearer the current jaws wins. Best effort: each tick asks
  IK for a rotation at most `rot_rate * dt` from the one the tool has, in the nullspace of the position,
  and falls back to position only. The follower never refuses on tilt; `Step.tilt` says how far from
  straight down the tool ended (near the top of the box the A1X cannot point down).
* **Checks per tick.** IK within `ik_tol`, joints clipped to the limits in force, no joint step above
  `max_step_deg` (a jump to another elbow branch), `config_clear` (arm, table). A failed tick commits
  nothing: the tracker stays at the last good point with zero speed, and the caller sends nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import mujoco
import numpy as np

from .collision import CollisionChecker
from .kinematics import RobotKinematics
from .reach import ReachVolume

Joints = dict[str, float]


def down_rotation(spec, p: np.ndarray, yaw: float | None = None) -> np.ndarray:
    """TCP rotation in the base frame: approach axis pointing down, jaws closing horizontally along `yaw`
    (angle of the closing axis in the base xy plane, rad). `yaw=None`: across the line from the base
    to `p`."""
    a = np.asarray(spec.tcp_approach, float)
    c = np.asarray(spec.tcp_closing, float)
    local = np.column_stack([a, c, np.cross(a, c)])

    if yaw is None:
        yaw = math.atan2(float(p[1]), float(p[0])) + math.pi / 2
    down = np.array([0.0, 0.0, -1.0])
    side = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    world = np.column_stack([down, side, np.cross(down, side)])

    return world @ local.T


def wrap(angle: float) -> float:
    """Angle in [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def nearest_yaw(want: float, current: float) -> float:
    """`want` or `want + pi` (the same grasp for symmetric jaws), whichever is nearer `current`."""
    d = wrap(want - current)
    if d > math.pi / 2:
        d -= math.pi
    elif d < -math.pi / 2:
        d += math.pi
    return current + d


def rotate_toward(R: np.ndarray, target: np.ndarray, max_angle: float) -> np.ndarray:
    """`R` turned towards `target` along the shortest rotation, by at most `max_angle` rad."""
    delta = target @ R.T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, delta.ravel())
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, quat, 1.0)
    angle = float(np.linalg.norm(vel))
    if angle <= max_angle or angle < 1e-9:
        return target.copy()
    step = np.zeros(4)
    mujoco.mju_axisAngle2Quat(step, vel / angle, max_angle)
    turn = np.zeros(9)
    mujoco.mju_quat2Mat(turn, step)
    return turn.reshape(3, 3) @ R


def jaw_yaw(kin: RobotKinematics, spec, joints: Mapping[str, float]) -> float:
    """Angle of the jaws' closing axis in the base xy plane for `joints`, rad (the `yaw` convention)."""
    closing = kin.tcp(joints)[:3, :3] @ np.asarray(spec.tcp_closing, float)
    return math.atan2(float(closing[1]), float(closing[0]))


def tilt_of(approach: np.ndarray) -> float:
    """Angle between the approach axis and straight down, rad."""
    return float(np.arccos(np.clip(-float(approach[2]), -1.0, 1.0)))


class Tracker:
    """A point that follows a setpoint with bounded speed and acceleration and stops on it."""

    def __init__(self, p: np.ndarray, v_max: float, a_max: float | None = None):
        self.p = np.asarray(p, float).reshape(3).copy()
        self.v = np.zeros(3)
        self.v_max = float(v_max)
        self.a_max = float(a_max) if a_max is not None else 4.0 * self.v_max

    def propose(self, target: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """(point, velocity) after `dt` towards `target`; nothing changes until `commit`."""
        if dt <= 0.0:
            return self.p.copy(), self.v.copy()
        e = np.asarray(target, float).reshape(3) - self.p
        dist = float(np.linalg.norm(e))
        # The fastest speed from which the point can still stop at the target, never past it in one tick.
        speed = min(self.v_max, math.sqrt(2.0 * self.a_max * dist), dist / dt)
        want = e / dist * speed if dist > 1e-9 else np.zeros(3)
        dv = want - self.v
        n = float(np.linalg.norm(dv))
        if n > self.a_max * dt:
            dv *= self.a_max * dt / n
        v = self.v + dv
        return self.p + v * dt, v

    def commit(self, p: np.ndarray, v: np.ndarray) -> None:
        self.p, self.v = np.asarray(p, float), np.asarray(v, float)

    def halt(self) -> None:
        """Speed to zero where the point is (a failed tick, a lagging joint)."""
        self.v = np.zeros(3)


@dataclass
class Step:
    """One tick: the arm joints to send, or why not (`fail` "ik" / "collision", with `detail`)."""

    joints: Joints | None
    fail: str = ""
    detail: str = ""
    #: Achieved angle between the approach axis and straight down [rad].
    tilt: float = 0.0


class Follower:
    """One stream's stepping state: the tracker, the last good joints, the rotation."""

    def __init__(self, kin: RobotKinematics, checker: CollisionChecker, spec, *, limits: Mapping[str, tuple],
                 joints: Mapping[str, float], volume: ReachVolume, v_max: float = 0.2, z_floor: float = 0.01,
                 rot_rate: float = 1.0, ik_tol: float = 0.003, max_step_deg: float = 5.0, iters: int = 30):
        self.kin = kin
        self.checker = checker
        self.spec = spec
        self.arm = tuple(j for j in spec.joints if j != spec.gripper)
        self.limits = {k: (float(lo), float(hi)) for k, (lo, hi) in limits.items() if k in self.arm}
        self.volume = volume
        #: Setpoints stay this far above the box floor: tool down, the tips are then >= 27 mm above the table.
        self.z_floor = float(z_floor)
        self.rot_rate = float(rot_rate)
        self.ik_tol = float(ik_tol)
        self.max_step_deg = float(max_step_deg)
        self.iters = int(iters)
        # An arm with fewer than 6 joints (SO-101) can't reach an arbitrary rotation: position only.
        self.full_rotation = len(self.arm) >= 6
        self.joints: Joints = {k: float(joints[k]) for k in self.arm if k in joints}
        T = kin.tcp(self.joints)
        self.tracker = Tracker(T[:3, 3], v_max)
        self.tilt = tilt_of(T[:3, :3] @ np.asarray(spec.tcp_approach, float))

    @property
    def v_max(self) -> float:
        return self.tracker.v_max

    @v_max.setter
    def v_max(self, v: float) -> None:
        self.tracker.v_max = float(v)
        self.tracker.a_max = 4.0 * float(v)

    @property
    def point(self) -> np.ndarray:
        """The commanded TCP point (the tracker's)."""
        return self.tracker.p.copy()

    def setpoint(self, pos: np.ndarray) -> np.ndarray:
        """A received point clamped to the box and `z_floor` above its floor."""
        p = self.volume.clamp(pos)
        p[2] = max(float(p[2]), float(self.volume.lo[2]) + self.z_floor)
        return p

    def hold(self) -> None:
        """Stop advancing (a joint lags behind its command): the tracker's speed goes to zero."""
        self.tracker.halt()

    def step(self, dt: float, target: np.ndarray, yaw: float | None, advance: bool = True) -> Step:
        """One tick towards `target` (already a `setpoint`). `advance=False` keeps the joints where they are:
        the point and the rotation (a stalled joint must not see the wrist run on towards tool-down)."""
        if not advance:
            self.tracker.halt()
            return Step(dict(self.joints), tilt=self.tilt)
        p, v = self.tracker.propose(target, dt)

        rotation = self._rotation(p, yaw, dt) if self.full_rotation else None
        sol = self.kin.ik_track(p, rotation, seed=self.joints, iters=self.iters, tol=self.ik_tol)
        if not sol.ok and rotation is not None:
            sol = self.kin.ik_track(p, None, seed=self.joints, iters=self.iters, tol=self.ik_tol)
        if not sol.ok:
            self.tracker.halt()
            return Step(None, "ik", f"IK: blad {sol.pos_err * 1000:.1f} mm", self.tilt)

        free = (-math.inf, math.inf)
        joints = {k: float(np.clip(sol.joints[k], *self.limits.get(k, free))) for k in self.arm}
        jump = max(abs(joints[k] - self.joints[k]) for k in self.arm)
        if jump > self.max_step_deg:
            self.tracker.halt()
            return Step(None, "ik", f"skok {jump:.0f} st. w jednym takcie", self.tilt)

        collision = self.checker.config_clear(joints)
        if collision is not None:
            self.tracker.halt()
            return Step(None, "collision", collision, self.tilt)

        self.tracker.commit(p, v)
        self.joints = joints
        self.tilt = tilt_of(self.kin.tool_axes(joints)[0])
        return Step(dict(joints), tilt=self.tilt)

    def _rotation(self, p: np.ndarray, yaw: float | None, dt: float) -> np.ndarray:
        """Rotation to ask IK for: from the tool's current one towards tool-down at `yaw`, rate limited."""
        R = self.kin.tcp(self.joints)[:3, :3]
        current = jaw_yaw(self.kin, self.spec, self.joints)
        if yaw is None:
            yaw = math.atan2(float(p[1]), float(p[0])) + math.pi / 2
        want = down_rotation(self.spec, p, nearest_yaw(float(yaw), current))
        return rotate_toward(R, want, self.rot_rate * max(dt, 0.0))
