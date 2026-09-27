"""The hover an episode starts from, and the staged joint plan that gets the arm there.

The hover is the believed grasp plus `HOVER` in z, with the tool pointing down and the
closing axis at the taught yaw. It is solved once per episode with the twin's full `ik`
from the spec home pose, for the yaw and for the yaw + pi (the pinch is symmetric, the
joints are not: J6 differs by 180 deg), and the solution whose J6 is further from its
limits wins. Solving from home keeps J5 at zero, as in every training hover.

`stage_plan` is the staged joint-group order of the Specs driver's pre-positioning as a
free function; the collision question is the caller's `CollisionChecker.path_clear`.
The reach volume is not asked here (the run module asks it about the hover when the
workspace sets one).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from ..kinematics import RobotKinematics
from ..robots import RobotSpec
from .contract import (
    HOVER,
    HOVER_J5_MAX_DEG,
    HOVER_J6_MARGIN_DEG,
    IK_TOL_POS,
    IK_TOL_ROT,
    NOISE_XY,
    NOISE_YAW,
    NOISE_Z,
    TIP_CLEAR,
    TRAIN_HOVER_Z,
    TRAIN_J1_DEG,
    TRAIN_RADIUS,
    grasp_R,
    wrap,
)
from .rig import Rig

logger = logging.getLogger(__name__)

JOINT1 = "arm_joint1"


class HoverError(RuntimeError):
    """No usable hover: `reason` is a refusal code (`ik`, `grasp point not taught`)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Hover:
    """A solved hover: positions in the base frame (m), `joints` the six arm joints (deg), `q` in rad."""

    p: np.ndarray
    yaw: float                        # closing-axis yaw: the taught yaw or the taught yaw + pi, wrapped
    believed: np.ndarray              # == p - [0, 0, HOVER]
    joints: dict[str, float]
    q: np.ndarray
    j6_margin_deg: float
    outside_training: list[str]       # empty inside the training envelope, else the failed checks


@dataclass
class _Candidate:
    yaw: float
    joints: dict[str, float]
    pos_err: float
    rot_err: float
    j6_margin_deg: float
    failed: list[str]


def _rotation_angle(R: np.ndarray) -> float:
    """Angle of a rotation matrix; atan2 keeps it accurate near zero, where acos is not."""
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return float(math.atan2(0.5 * float(np.linalg.norm(w)), 0.5 * (float(np.trace(R)) - 1.0)))


def _joint_box(kin: RobotKinematics,
               limits: Mapping[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Arm joint limits in degrees: the caller's limits within the MJCF range, as `ik_step` has them."""
    box: dict[str, tuple[float, float]] = {}
    for k, name in enumerate(kin.spec.joints):
        if name == kin.spec.gripper:
            continue
        model_lo, model_hi = math.degrees(kin.lo[k]), math.degrees(kin.hi[k])
        lo, hi = limits.get(name, (model_lo, model_hi))
        box[name] = (max(float(lo), model_lo), min(float(hi), model_hi))
    return box


def _candidate(kin: RobotKinematics, p: np.ndarray, yaw: float,
               box: Mapping[str, tuple[float, float]]) -> _Candidate:
    arm = kin.spec.arm_joints
    R = grasp_R(yaw)
    sol = kin.ik(p, rotation=R, seed=kin.spec.home)
    joints = {n: float(sol.joints[n]) for n in arm}
    # `sol.ok` ignores rotation and `sol.rot_err` can be one update stale: judge by FK.
    T = kin.tcp(joints)
    pos_err = float(np.linalg.norm(p - T[:3, 3]))
    rot_err = _rotation_angle(R @ T[:3, :3].T)

    failed: list[str] = []
    if not pos_err < IK_TOL_POS:
        failed.append(f"pozycja {pos_err * 1000.0:.2f} mm")
    if not rot_err < IK_TOL_ROT:
        failed.append(f"obrot {rot_err * 1000.0:.2f} mrad")
    j5 = joints[arm[4]]
    if not abs(j5) <= HOVER_J5_MAX_DEG:
        failed.append(f"J5 {j5:.2f} st.")
    outside = [f"{n} {joints[n]:.1f} st." for n in arm if not box[n][0] <= joints[n] <= box[n][1]]
    if outside:
        failed.append("poza limitami " + ", ".join(outside))
    lo6, hi6 = box[arm[5]]
    j6 = joints[arm[5]]
    margin = min(j6 - lo6, hi6 - j6)
    if not margin >= HOVER_J6_MARGIN_DEG:
        failed.append(f"zapas J6 {margin:.1f} st.")
    return _Candidate(yaw, joints, pos_err, rot_err, float(margin), failed)


def solve_hover(kin: RobotKinematics, believed: np.ndarray, yaw: float, *,
                limits: Mapping[str, tuple[float, float]]) -> Hover:
    """Hover above `believed` (base frame, m) with the closing axis at `yaw` or `yaw + pi`.

    Raises `HoverError("ik", detail)` when neither candidate passes; the detail names the
    failed checks of both in Polish, for the operator. `outside_training` is filled but not acted on.
    """
    believed = np.asarray(believed, dtype=float).reshape(3)
    yaw = float(yaw)
    if not (np.all(np.isfinite(believed)) and math.isfinite(yaw)):
        raise HoverError("ik", f"nieskonczone: punkt {believed.tolist()}, yaw {yaw}")
    p = believed + np.array([0.0, 0.0, HOVER])
    box = _joint_box(kin, limits)
    candidates = [_candidate(kin, p, y, box) for y in (wrap(yaw), wrap(yaw + math.pi))]
    for c in candidates:
        logger.debug("hover yaw %.1f deg: %s, J6 margin %.1f deg, %s", math.degrees(c.yaw),
                     {k: round(v, 2) for k, v in c.joints.items()}, c.j6_margin_deg, c.failed or "ok")
    valid = [c for c in candidates if not c.failed]
    if not valid:
        raise HoverError("ik", "; ".join(
            f"yaw {math.degrees(c.yaw):.1f} st.: {', '.join(c.failed)}" for c in candidates))
    best = max(valid, key=lambda c: c.j6_margin_deg)
    arm = kin.spec.arm_joints
    return Hover(
        p=p,
        yaw=float(best.yaw),
        believed=p - np.array([0.0, 0.0, HOVER]),
        joints=dict(best.joints),
        q=np.radians([best.joints[n] for n in arm]),
        j6_margin_deg=best.j6_margin_deg,
        outside_training=training_envelope(best.joints, p),
    )


def from_rig(kin: RobotKinematics, rig: Rig, *, limits: Mapping[str, tuple[float, float]]) -> Hover:
    """Hover over the touch-taught grasp point of the per-rig file."""
    believed, yaw = rig.believed(), rig.grasp_yaw()
    if believed is None or yaw is None:
        raise HoverError("grasp point not taught")
    return solve_hover(kin, believed, yaw, limits=limits)


def from_scene(kin: RobotKinematics, grasp_xy: np.ndarray, grasp_yaw: float, *, touch_z: float,
               limits: Mapping[str, tuple[float, float]], rng: np.random.Generator | None = None) -> Hover:
    """Hover over a grasp read from the simulated scene, with training's hover noise when `rng` is given."""
    xy = np.asarray(grasp_xy, dtype=float).ravel()[:2]
    believed = np.array([xy[0], xy[1], float(touch_z) + TIP_CLEAR])
    yaw = float(grasp_yaw)
    if rng is not None:
        # Same draws in the same order as training's reset.
        dxy = rng.normal(0.0, NOISE_XY, 2)
        dz = rng.normal(0.0, NOISE_Z)
        dyaw = rng.normal(0.0, NOISE_YAW)
        believed = believed + np.array([dxy[0], dxy[1], dz])
        yaw += float(dyaw)
    return solve_hover(kin, believed, yaw, limits=limits)


def training_envelope(joints: Mapping[str, float], p: np.ndarray) -> list[str]:
    """Which checks of the training envelope a hover fails, in Polish; empty when inside."""
    out: list[str] = []
    j1 = float(joints[JOINT1])
    lo, hi = TRAIN_J1_DEG
    if not lo <= j1 <= hi:
        out.append(f"joint1 {j1:.1f} st. poza {lo:g}..{hi:g}")
    radius = float(math.hypot(p[0], p[1]))
    lo, hi = TRAIN_RADIUS
    if not lo <= radius <= hi:
        out.append(f"promien {radius:.2f} m poza {lo:g}..{hi:g}")
    z = float(p[2])
    lo, hi = TRAIN_HOVER_Z
    if not lo <= z <= hi:
        out.append(f"wysokosc zawisu {z:.3f} m poza {lo:g}..{hi:g}")
    return out


def _largest_step(arm: tuple[str, ...], start: Mapping[str, float], goal: Mapping[str, float]) -> float:
    return max((abs(v - start.get(k, v)) for k, v in goal.items() if k in arm), default=0.0)


def _walk_stages(arm: tuple[str, ...], start: Mapping[str, float], goal: Mapping[str, float],
                 stages: list[tuple[str, ...]],
                 clear: Callable[[Mapping[str, float], Mapping[str, float]], str | None],
                 ) -> tuple[list[dict[str, float]], str | None]:
    pose = {k: float(v) for k, v in start.items()}
    poses: list[dict[str, float]] = []
    for stage in stages:
        nxt = {**pose, **{j: float(goal[j]) for j in stage}}
        if _largest_step(arm, pose, nxt) < 1e-3:
            continue
        collision = clear(pose, nxt)
        if collision is not None:
            return poses, collision
        poses.append(nxt)
        pose = nxt
    return poses, None


def stage_plan(spec: RobotSpec, start: Mapping[str, float], goal: Mapping[str, float],
               clear: Callable[[Mapping[str, float], Mapping[str, float]], str | None],
               ) -> tuple[list[dict[str, float]], str | None]:
    """Intermediate poses from `start` to `goal`: the joint groups of `spec.home_stages` one
    after another, else in reverse, else one straight move. The first plan whose every
    segment `clear` passes wins. Returns (poses, None) or ([], the first collision)."""
    arm = spec.arm_joints
    stages = [tuple(j for j in stage if j in arm) for stage in spec.home_stages]
    stages = [stage for stage in stages if stage]
    orders = [stages, stages[::-1]] if stages else []

    first_collision: str | None = None
    for order in [*orders, []]:
        poses, collision = _walk_stages(arm, start, goal, [*order, arm], clear)
        if collision is None:
            return poses, None
        first_collision = first_collision or collision
    return [], first_collision
