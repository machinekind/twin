"""The move watch on a wobbly mount: synthetic frames with texture, no hardware and no browser.

A wobbly camera sways while the arm moves and tilts with its pose. Only a shift that lasts with
the arm still is a notice (`drifted`), and it never makes the camera `moved`.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("mujoco")  # watch.py imports mujoco

from machinekind_twin.ui.watch import CameraWatch, arm_still


def textured(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = cv2.GaussianBlur((rng.random((480, 640)) * 255).astype(np.uint8), (0, 0), 3)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    for _ in range(30):
        cv2.circle(g, (int(rng.integers(0, 640)), int(rng.integers(0, 480))), int(rng.integers(5, 40)),
                   int(rng.integers(0, 255)), -1)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def shifted(img: np.ndarray, tx: float) -> np.ndarray:
    """The picture moved sideways by `tx` px: every corner moves by `tx`."""
    M = np.array([[1.0, 0.0, tx], [0.0, 1.0, 0.0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


IMG = textured()


def a_watch(tolerance_px: float | None = 4.0) -> CameraWatch:
    w = CameraWatch()
    w.remember("k", IMG)
    w.set_wobbly("k", tolerance_px)
    return w


def run(w: CameraWatch, checks: list[tuple[float, bool]], t0: float = 0.0) -> float:
    """One check every 2 s, as in the panel: (shift [px], arm still). Returns the time after them."""
    t = t0
    for tx, still in checks:
        w.check("k", shifted(IMG, tx), still=still, now=t)
        t += 2.0
    return t


def test_shakes_that_come_and_go_never_set_a_wobbly_camera_moved():
    """Large shakes while the arm moves, pose tilts that return, a wander within the tolerance:
    a steady camera would be moved by any of the first two, the wobbly one gets not even a notice."""
    moving = [(15.0, False)] * 4                       # the same big shake in every check, arm moving
    tilts = [(9.0, True), (9.0, True), (0.0, True)] * 3     # a pose holds two checks, then the arm goes back
    wander = [(3.0, True)] * 5
    w = a_watch()
    t = run(w, moving + tilts + wander)
    assert not w.moved("k") and w.drifted("k") is None
    assert w.wander("k", now=t) == pytest.approx(15.0, abs=0.5)

    steady = a_watch(None)
    run(steady, moving[:2])
    assert steady.moved("k")
    steady = a_watch(None)
    run(steady, tilts[:2])
    assert steady.moved("k")


def test_a_lasting_shift_with_the_arm_still_is_a_notice_and_never_moved():
    w = a_watch()
    t = run(w, [(8.0, True), (8.0, True)])
    assert w.drifted("k") is None and w.detail["k"]["waiting"]
    t = run(w, [(8.0, True)], t)
    assert w.drifted("k") == pytest.approx(8.0, abs=0.5) and not w.moved("k")

    # The arm moves again: its shakes neither change the notice nor make the camera moved.
    t = run(w, [(25.0, False), (0.0, False)], t)
    assert w.drifted("k") == pytest.approx(8.0, abs=0.5) and not w.moved("k")
    # Back within the tolerance with the arm still: the notice goes.
    t = run(w, [(1.0, True)], t)
    assert w.drifted("k") is None
    # Over a minute later only the recent checks count as wander.
    assert w.wander("k", now=t + 60.0) is None


def test_only_a_shift_well_past_the_tolerance_counts_and_a_moving_arm_restarts_the_count():
    w = a_watch(4.0)
    run(w, [(5.5, True)] * 6)                            # past 4 px, but not past 1.5 x 4 px
    assert w.drifted("k") is None
    w = a_watch(4.0)
    run(w, [(8.0, True), (8.0, True), (8.0, False), (8.0, True), (8.0, True)])
    assert w.drifted("k") is None
    w = a_watch(4.0)
    run(w, [(8.0, True), (12.0, True), (8.0, True)])     # three checks, but not the same shift
    assert w.drifted("k") is None


def test_steady_cameras_are_judged_as_before_whatever_the_arm_does():
    w = a_watch(None)
    w.check("k", shifted(IMG, 8.0), still=False)
    assert not w.moved("k") and w.detail["k"]["waiting"]
    w.check("k", shifted(IMG, 8.0), still=False)
    assert w.moved("k") and w.drifted("k") is None and w.wander("k") is None
    w.check("k", shifted(IMG, 1.0))
    assert not w.moved("k")


def test_switching_the_mount_starts_the_judgement_afresh():
    """Ticked on a camera blinking red, the red goes at once; unticked, the steady rule applies again."""
    w = a_watch(None)
    run(w, [(8.0, True), (8.0, True)])
    assert w.moved("k")
    w.set_wobbly("k", 4.0)
    assert not w.moved("k")
    run(w, [(8.0, True)] * 3)
    assert w.drifted("k") is not None and not w.moved("k")
    w.set_wobbly("k", None)
    assert w.drifted("k") is None
    run(w, [(8.0, True), (8.0, True)])
    assert w.moved("k")


def test_the_arm_is_still_when_no_joint_moved_faster_than_the_still_speed():
    def history(values, gripper=None, dt=1 / 30):
        grip = gripper or [0.0] * len(values)
        return [(k * dt, {"j1": v, "gripper": grip[k]}) for k, v in enumerate(values)]

    rest = history([10.0 + (0.09 if k % 2 else 0.0) for k in range(60)])  # one servo tick of noise
    assert arm_still(rest, since=0.5, max_speed=10.0, skip=("gripper",))
    moving = history([10.0 + 0.5 * k for k in range(60)])                 # 15 units/s
    assert not arm_still(moving, since=0.5, max_speed=10.0)
    stopped = history([10.0 + 0.5 * min(k, 20) for k in range(60)])       # stops at 0.67 s
    assert not arm_still(stopped, since=0.5, max_speed=10.0)
    assert arm_still(stopped, since=1.0, max_speed=10.0)
    grip = history([10.0] * 60, gripper=[3.0 * k for k in range(60)])
    assert arm_still(grip, since=0.0, max_speed=10.0, skip=("gripper",))
    assert not arm_still(grip, since=0.0, max_speed=10.0)
    assert arm_still([], since=0.0, max_speed=10.0)                         # no readings: nothing moved
