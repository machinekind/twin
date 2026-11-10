"""Pulling another robot's copy to where the station's cameras see its base tag.

The anchor is where the robot's fixed frame stands in the driven arm's base frame: it starts at the
record's pose (`RobotRecord.T_base`), and the copy stands at anchor x odometry (`RobotBackend.base_pose`).
Odometry drifts, so each sighting of the tag moves the anchor. The pattern is the RealAnt copy's
(`docs/archive/NOTES_REMOTE_ROBOT.md`, "Pulling the pose from tags, and camera latency").
"""

from __future__ import annotations

import numpy as np

#: Share of the error one frame corrects: the archived note's g = 0.5 for a normal frame.
GAIN = 0.5
#: An error over this [m] is corrected all at once: the archived note's jump (lost, or moved by hand).
JUMP_M = 0.10
#: A heading error over this [deg] is corrected all at once: the archived note's jump.
JUMP_DEG = 30.0
#: A frame older than this [s] is not used: the archived note skipped frames older than 0.5 s.
FRAME_MAX_AGE = 0.5


def planar_error(stood: np.ndarray, seen: np.ndarray) -> tuple[np.ndarray, float]:
    """From pose `stood` to pose `seen` (4x4) in the plane: the shift in x and y [m] and the turn about
    the vertical [rad], in -pi..pi."""
    shift = seen[:2, 3] - stood[:2, 3]
    turn = np.arctan2(seen[1, 0], seen[0, 0]) - np.arctan2(stood[1, 0], stood[0, 0])
    return shift, float(np.arctan2(np.sin(turn), np.cos(turn)))


def corrected_anchor(anchor: np.ndarray, odometry: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """The anchor (4x4) that brings the copy closer to the base pose `measured` from the tag.

    `odometry` is the base pose when the frame was taken: compared with where the copy stood then, the
    robot's motion since the frame is left alone. The anchor turns about the vertical through the base
    by `GAIN` x the heading error and shifts by `GAIN` x the position error; past `JUMP_M` or `JUMP_DEG`
    by all of it. Height, pitch and roll stay."""
    stood = anchor @ odometry
    shift, turn = planar_error(stood, measured)
    jump = np.linalg.norm(shift) > JUMP_M or abs(turn) > np.radians(JUMP_DEG)
    gain = 1.0 if jump else GAIN

    c = np.cos(gain * turn)
    s = np.sin(gain * turn)
    correction = np.eye(4)
    correction[:2, :2] = [[c, -s], [s, c]]
    pivot = stood[:2, 3]
    correction[:2, 3] = pivot - correction[:2, :2] @ pivot + gain * shift
    return correction @ anchor
