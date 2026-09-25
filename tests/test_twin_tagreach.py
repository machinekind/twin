"""Tag na biurku: poza z kamer w ukladzie podstawy (tagreach)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lerobot_mp.twin import tagreach
from lerobot_mp.twin.calib import tags
from lerobot_mp.twin.workspace import CameraRecord


def _look_at(eye, target, up=(0, 0, 1)):
    """T_cam2base kamery OpenCV (z do przodu, y w dol) patrzacej z `eye` na `target`."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = eye
    return T


def _frame_with_tag(cam: CameraRecord, T_tag2base, tag_id, size, w=640, h=480):
    """Bialy kadr z tagiem wklejonym perspektywicznie tam, gdzie kamera go widzi."""
    K = np.asarray(cam.K, float)
    T_tag2cam = np.linalg.inv(np.asarray(cam.T_cam2base)) @ T_tag2base
    plate = size / tags.MARKER_FRACTION                       # caly kwadrat z marginesem
    hp = plate / 2
    plate_tag = np.array([[-hp, hp, 0], [hp, hp, 0], [hp, -hp, 0], [-hp, -hp, 0]], float)
    pts_cam = (T_tag2cam[:3, :3] @ plate_tag.T).T + T_tag2cam[:3, 3]
    px = tags.project(pts_cam, K).astype(np.float32)
    img_tag = tags.render(tag_id, px=800)
    src = np.array([[0, 0], [799, 0], [799, 799], [0, 799]], np.float32)
    H = cv2.getPerspectiveTransform(src, px)
    frame = cv2.warpPerspective(img_tag, H, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)


def _camera(name, eye, target):
    K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    return CameraRecord(name, "0", K=K, dist=[0.0] * 5, T_cam2base=_look_at(eye, target).tolist())


def _tag_pose(xy, yaw):
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]        # tag lezy na blacie, z do gory
    T[:3, 3] = [xy[0], xy[1], 0.0]
    return T


def test_tag_pose_from_two_calibrated_cameras_lands_within_a_millimetre():
    size, tag_id = 0.05, 7
    truth = _tag_pose((0.30, 0.05), 0.4)
    cams = [_camera("lewa", (0.2, 0.5, 0.45), (0.3, 0.0, 0.0)),
            _camera("prawa", (0.6, -0.4, 0.5), (0.3, 0.0, 0.0))]
    frames = {c.name: _frame_with_tag(c, truth, tag_id, size) for c in cams}
    fix = tagreach.tag_from_frames(frames, cams, tag_id, size)
    assert fix is not None and set(fix.per_camera) == {"lewa", "prawa"}
    assert np.linalg.norm(fix.position - truth[:3, 3]) < 1e-3
    assert all(r < 0.5 for r in fix.reproj_px.values())
    assert fix.spread_mm < 2.0
    p = tagreach.hover_point(fix.T_tag2base, 0.08)
    assert np.allclose(p[:2], truth[:2, 3], atol=1e-3) and abs(p[2] - 0.08) < 1e-3


def test_untrusted_disabled_and_blind_cameras_are_skipped():
    size, tag_id = 0.05, 7
    truth = _tag_pose((0.30, 0.05), 0.0)
    good = _camera("dobra", (0.2, 0.5, 0.45), (0.3, 0.0, 0.0))
    untrusted = _camera("bez_pozy", (0.6, -0.4, 0.5), (0.3, 0.0, 0.0))
    off = _camera("wylaczona", (0.6, -0.4, 0.5), (0.3, 0.0, 0.0))
    blind = _camera("slepa", (0.6, -0.4, 0.5), (0.3, 0.0, 0.0))
    frames = {"dobra": _frame_with_tag(good, truth, tag_id, size),
              "bez_pozy": _frame_with_tag(untrusted, truth, tag_id, size),
              "wylaczona": _frame_with_tag(off, truth, tag_id, size),
              "slepa": np.full((480, 640, 3), 255, np.uint8)}
    untrusted.T_cam2base = None
    off.enabled = False
    fix = tagreach.tag_from_frames(frames, [good, untrusted, off, blind], tag_id, size)
    assert fix is not None and list(fix.per_camera) == ["dobra"]
    assert fix.spread_mm == 0.0
    assert tagreach.tag_from_frames(frames, [good], 8, size) is None       # inny id


def test_no_camera_sees_the_tag():
    cam = _camera("c", (0.2, 0.5, 0.45), (0.3, 0.0, 0.0))
    frames = {"c": np.full((480, 640, 3), 255, np.uint8)}
    assert tagreach.tag_from_frames(frames, [cam], 0, 0.05) is None
    assert tagreach.tag_from_frames({}, [cam], 0, 0.05) is None
