"""Specs reach volume: defaults, helpers, workspace persistence, and every default grid point reachable
without colliding with the empty table."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from lerobot_mp.twin import scene as sc
from lerobot_mp.twin.collision import CollisionChecker
from lerobot_mp.twin.kinematics import RobotKinematics
from lerobot_mp.twin.reach import MIN_Z, ReachVolume
from lerobot_mp.twin.robots import get_spec
from lerobot_mp.twin.workspace import Workspace


def grid(vol: ReachVolume) -> list[np.ndarray]:
    """4x4x3 grid; linspace includes the endpoints, so all 8 corners are in it."""
    axes = [np.linspace(vol.lo[k], vol.hi[k], n) for k, n in enumerate((4, 4, 3))]
    return [np.array(p) for p in itertools.product(*axes)]


@pytest.mark.parametrize("robot", ["a1x", "so101"])
def test_every_grid_point_of_the_default_volume_is_reachable_and_collision_free(robot):
    spec = get_spec(robot)
    vol = ReachVolume.from_spec(spec)
    assert vol.lo[2] >= MIN_Z
    kin = RobotKinematics(spec)
    checker = CollisionChecker(sc.build(Workspace(robot=robot).scene_config()))
    points = grid(vol)
    corners = {tuple(np.round(c, 6)) for c in itertools.product(*zip(vol.lo, vol.hi))}
    assert corners <= {tuple(np.round(p, 6)) for p in points}
    bad = []
    for p in points:
        sol = kin.ik(p, seed=spec.home, tol=0.003, accept=lambda j: checker.config_clear(j) is None)
        why = checker.config_clear(sol.joints) if sol.ok else f"IK {sol.pos_err * 1000:.1f} mm"
        if why is not None:
            bad.append((p.round(3).tolist(), why))
    assert bad == []


def test_a1x_default_numbers():
    vol = ReachVolume.from_spec(get_spec("a1x"))
    assert vol.lo.tolist() == pytest.approx([0.20, -0.30, 0.05])
    assert vol.hi.tolist() == pytest.approx([0.50, 0.30, 0.30])


def test_contains_margin_and_clamp():
    vol = ReachVolume(np.array([0.2, -0.3, 0.05]), np.array([0.5, 0.3, 0.3]))
    assert vol.contains([0.35, 0.0, 0.1])
    assert vol.contains([0.2, -0.3, 0.05])                                       # the boundary counts as inside
    assert not vol.contains([0.19, 0.0, 0.1])
    assert not vol.contains([0.21, 0.0, 0.1], margin=0.02)
    assert vol.contains([0.19, 0.0, 0.1], margin=-0.02)
    assert not vol.contains([np.nan, 0.0, 0.1])
    assert vol.clamp([0.0, 1.0, 0.1]).tolist() == pytest.approx([0.2, 0.3, 0.1])
    with pytest.raises(ValueError):
        ReachVolume(np.array([0.5, 0, 0]), np.array([0.2, 1, 1]))


def test_overlay_object_is_a_volume_in_the_base_frame():
    vol = ReachVolume(np.array([0.2, -0.3, 0.05]), np.array([0.5, 0.3, 0.3]))
    obj = vol.overlay_object(alpha=0.5)
    assert obj["id"] == "reach" and obj["kind"] == "volume" and obj["label"] == "zasieg"
    assert obj["pos"] == pytest.approx([0.35, 0.0, 0.175])
    assert obj["size"] == pytest.approx([0.3, 0.6, 0.25])
    assert obj["alpha"] == 0.5 and len(obj["color"]) == 3
    assert json.loads(json.dumps(obj)) == obj
    assert vol.overlay_object()["alpha"] == 0.35


def test_workspace_override_survives_save_and_load_and_old_files_still_load(tmp_path):
    path = tmp_path / "twin.json"
    ws = Workspace(robot="a1x")
    ws.save(path)
    assert "reach" not in json.loads(path.read_text())                         # no override: file format unchanged
    assert Workspace.load(path).reach is None
    assert ReachVolume.from_workspace(Workspace.load(path)).lo.tolist() == pytest.approx([0.20, -0.30, 0.05])

    ws.reach = ReachVolume(np.array([0.25, -0.2, 0.06]), np.array([0.45, 0.2, 0.25])).to_dict()
    ws.save(path)
    back = Workspace.load(path)
    assert back.reach == {"lo": [0.25, -0.2, 0.06], "hi": [0.45, 0.2, 0.25]}
    vol = ReachVolume.from_workspace(back)
    assert vol.lo.tolist() == pytest.approx([0.25, -0.2, 0.06]) and vol.hi.tolist() == pytest.approx([0.45, 0.2, 0.25])
    with pytest.raises(ValueError):
        ReachVolume.from_dict({"lo": [0, 0, 0]})
