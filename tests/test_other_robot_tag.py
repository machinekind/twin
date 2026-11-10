"""A walking robot's base tag: the station's cameras see it and pull the copy to where the robot really is.

The test walker's fixed frame stands on the table, 35 cm ahead of the driven arm and 15 cm to its left,
turned 90 degrees; its sticker lies flat on top of its trunk (`tests/robots/walker/robot.yaml`).
"""

from __future__ import annotations

import shutil
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

pytest.importorskip("mujoco")

from conftest import BOT, WALKER
from test_other_robot_base import in_arm_frame

from machinekind_twin import basefix, passive, tagreach
from machinekind_twin.backends.factory import BaseDriveArm
from machinekind_twin.calib.simulate import look_at
from machinekind_twin.kinematics import inverse, pose
from machinekind_twin.passive import PassiveReader
from machinekind_twin.robots import ROBOT_FILE, BaseTag, get_spec, load_robot
from machinekind_twin.runtime import Twin
from machinekind_twin.workspace import CameraRecord, RobotRecord, Workspace

ON_THE_TABLE = [0.35, 0.15, 0.0]
#: Intrinsics of the two simulated cameras, 1280 x 720.
K = [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]
#: The top of the trunk with the base at `ODOMETRY`: both cameras look down at it.
TRUNK_TOP = (0.35, 0.20, 0.30)
#: The walker's sticker as its `robot.yaml` gives it: flat on top of the trunk, facing up.
TAG = BaseTag(7, 0.08, ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.03),
                        (0.0, 0.0, 0.0, 1.0)))


def flat(x: float, y: float, yaw_deg: float, z: float = 0.0) -> np.ndarray:
    c = np.cos(np.radians(yaw_deg))
    s = np.sin(np.radians(yaw_deg))
    return pose(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([x, y, z]))


def nudged(T: np.ndarray, dx: float = 0.0, turn_deg: float = 0.0) -> np.ndarray:
    """`T` turned in place about the vertical by `turn_deg`, then moved `dx` along the arm's x."""
    out = T.copy()
    out[:3, :3] = flat(0.0, 0.0, turn_deg)[:3, :3] @ T[:3, :3]
    out[0, 3] += dx
    return out


#: Where the record puts the walker's fixed frame (`RobotRecord.T_base`).
ANCHOR = flat(ON_THE_TABLE[0], ON_THE_TABLE[1], 90.0)
#: The trunk standing 5 cm ahead of the fixed frame, turned 20 degrees.
ODOMETRY = flat(0.05, 0.0, 20.0, z=0.27)
STOOD = ANCHOR @ ODOMETRY


def station(walker: str) -> Workspace:
    ws = Workspace(robot=BOT)
    ws.other_robots.append(RobotRecord("walker", walker, xyz=list(ON_THE_TABLE), yaw_deg=90.0))
    for name, eye in (("a", (0.20, 0.05, 0.85)), ("b", (0.55, 0.35, 0.85))):
        T = look_at(eye, TRUNK_TOP).tolist()
        ws.add_camera(CameraRecord(name, "sim", 1280, 720, K=K, sim_pose=T, T_cam2base=T,
                                   intrinsics_from="symulacja", calibration={"trusted": True}))
    return ws


@pytest.fixture
def twin(walker):
    twin = Twin(station(walker))
    yield twin
    twin.close()


def seen_at(twin: Twin, monkeypatch, base: np.ndarray, taken: float) -> None:
    """The cameras' next frames, taken at `taken`, show the walker's base at `base` (arm base frame)."""
    T_tag = base @ np.asarray(TAG.pose, float)
    monkeypatch.setattr(twin.cameras, "frame_t", lambda name: (np.zeros((2, 2, 3), np.uint8), taken))
    monkeypatch.setattr(tagreach, "tag_from_frames",
                        lambda frames, cameras, tag_id, size: tagreach.TagFix(T_tag, {"a": T_tag}))


def trunk(twin: Twin) -> np.ndarray:
    with twin.lock:
        return in_arm_frame(twin.scene, "walker/trunk")


def test_the_walker_carries_tag_7_flat_on_top_of_its_trunk(walker):
    assert get_spec(walker).base_tag == TAG


@pytest.mark.parametrize("key, value, problem", [
    ("size", None, r"Brak kluczy w sekcji .*base_tag: \['size'\]"),
    ("size", 0.0, "`base_tag.size`: bok czarnego kwadratu"),
    ("pose", [[1.0, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
     "`base_tag.pose` musi byc przeksztalceniem sztywnym"),
])
def test_a_base_tag_missing_a_key_sized_zero_or_with_a_sheared_pose_is_refused(tmp_path, key, value, problem):
    folder = tmp_path / "walker"
    shutil.copytree(WALKER, folder)
    data = yaml.safe_load((folder / ROBOT_FILE).read_text(encoding="utf-8"))
    if value is None:
        del data["base_tag"][key]
    else:
        data["base_tag"][key] = value
    (folder / ROBOT_FILE).write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match=problem):
        load_robot(folder)


def test_no_error_leaves_the_anchor_alone():
    assert np.allclose(basefix.corrected_anchor(ANCHOR, ODOMETRY, STOOD), ANCHOR)


def test_a_small_position_error_moves_the_base_by_half_of_it():
    anchor = basefix.corrected_anchor(ANCHOR, ODOMETRY, nudged(STOOD, dx=0.04))

    assert np.allclose(anchor @ ODOMETRY, nudged(STOOD, dx=0.02))


def test_a_small_heading_error_turns_the_base_in_place_by_half_of_it():
    base = basefix.corrected_anchor(ANCHOR, ODOMETRY, nudged(STOOD, turn_deg=10.0)) @ ODOMETRY

    assert np.allclose(base[:3, :3], nudged(STOOD, turn_deg=5.0)[:3, :3])
    assert np.allclose(base[:3, 3], STOOD[:3, 3])


@pytest.mark.parametrize("dx, turn_deg", [(0.15, 0.0), (0.0, 40.0)])
def test_an_error_over_the_threshold_jumps_all_the_way(dx, turn_deg):
    seen = nudged(STOOD, dx=dx, turn_deg=turn_deg)

    assert np.allclose(basefix.corrected_anchor(ANCHOR, ODOMETRY, seen) @ ODOMETRY, seen)


def test_the_height_pitch_and_roll_of_the_anchor_stay():
    tilt, _ = cv2.Rodrigues(np.radians([4.0, -3.0, 0.0]))       # about a horizontal axis: pitch and roll
    tilted = ANCHOR @ pose(tilt, np.array([0.0, 0.0, 0.05]))
    corrected = basefix.corrected_anchor(tilted, ODOMETRY, nudged(tilted @ ODOMETRY, dx=0.04, turn_deg=10.0))

    assert corrected[2, 3] == pytest.approx(tilted[2, 3])
    assert np.allclose(corrected[2, :3], tilted[2, :3])          # the vertical's row: pitch and roll


def test_a_sighting_moves_the_copy_half_way_to_the_tag_and_shows_that_move_as_the_drift(twin, monkeypatch):
    now = time.monotonic()
    twin.set_passive_joints("walker", {}, (ODOMETRY, now))
    seen_at(twin, monkeypatch, nudged(STOOD, dx=0.04, turn_deg=10.0), now)
    twin.locate_other("walker")
    reported = twin.other_base_pose("walker")
    anchor = twin.tag_anchors["walker"]

    assert np.allclose(trunk(twin), nudged(STOOD, dx=0.02, turn_deg=5.0))
    assert reported is not None
    assert np.allclose(reported[0], nudged(STOOD, dx=0.02, turn_deg=5.0))
    assert anchor.drift_m == pytest.approx(0.02)
    assert anchor.drift_deg == pytest.approx(5.0)


def test_a_frame_no_newer_than_the_last_one_used_changes_nothing(twin, monkeypatch):
    now = time.monotonic()
    twin.set_passive_joints("walker", {}, (ODOMETRY, now))
    seen_at(twin, monkeypatch, nudged(STOOD, dx=0.04), now)
    twin.locate_other("walker")
    after_first = twin.tag_anchors["walker"].T.copy()
    seen_at(twin, monkeypatch, nudged(STOOD, dx=-0.04), now)
    twin.locate_other("walker")

    assert np.array_equal(twin.tag_anchors["walker"].T, after_first)


def test_a_robot_that_moved_on_after_the_frame_is_not_dragged_back(twin, monkeypatch):
    now = time.monotonic()
    twin.set_passive_joints("walker", {}, (ODOMETRY, now - 0.2))
    twin.set_passive_joints("walker", {}, (flat(0.15, 0.0, 20.0, z=0.27), now))
    seen_at(twin, monkeypatch, STOOD, now - 0.2)
    twin.locate_other("walker")

    assert np.allclose(twin.tag_anchors["walker"].T, ANCHOR)


def test_after_a_rebuild_the_copy_stands_where_it_stood(twin, monkeypatch):
    now = time.monotonic()
    twin.set_passive_joints("walker", {}, (ODOMETRY, now))
    seen_at(twin, monkeypatch, nudged(STOOD, dx=0.04, turn_deg=10.0), now)
    twin.locate_other("walker")
    before = trunk(twin)
    twin.rebuild()

    assert np.allclose(trunk(twin), before)


def test_the_exported_station_has_the_walker_without_its_sticker(walker, tmp_path):
    station(walker).export(tmp_path)
    xml = (tmp_path / "scene.xml").read_text(encoding="utf-8")

    assert "walker/trunk" in xml
    assert "base_tag" not in xml


def test_the_reader_looks_for_the_tag_after_the_velocity_at_most_five_times_a_second(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(passive, "time", SimpleNamespace(monotonic=lambda: clock.now))
    calls = []
    plugin = SimpleNamespace(connect=lambda: None, disconnect=lambda: None, read_joints=dict,
                             base_pose=lambda: (np.eye(4), 1.0),
                             send_velocity=lambda *velocity: calls.append("velocity"))
    reader = PassiveReader("walker", lambda: BaseDriveArm(plugin, "drivebus"), lambda joints, base: None,
                           lambda: calls.append("tag"))
    reader.drive(0.1, 0.0, 0.0, 10.0)
    reader.read_once()
    reader.read_once()
    clock.now += 0.2
    reader.read_once()

    assert calls == ["velocity", "tag", "velocity", "velocity", "tag"]


@pytest.mark.render
def test_the_base_measured_from_the_rendered_sticker_is_where_the_copy_stands(twin):
    twin.set_passive_joints("walker", {}, (ODOMETRY, time.monotonic()))
    frames = {name: twin.render(name) for name in ("a", "b")}
    stands = trunk(twin)
    fix = tagreach.tag_from_frames(frames, twin.workspace.cameras, TAG.id, TAG.size)
    assert fix is not None
    measured = fix.T_tag2base @ inverse(np.asarray(TAG.pose, float))
    off = measured[:3, :3].T @ stands[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(off) - 1) / 2, -1.0, 1.0)))

    assert set(fix.per_camera) == {"a", "b"}
    assert np.linalg.norm(measured[:3, 3] - stands[:3, 3]) < 0.01
    assert angle < 2.0
