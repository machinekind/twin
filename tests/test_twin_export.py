"""The station as data: `scene.xml` + `station.json`, read by plain MuJoCo without the twin.

This is the boundary to the training package: the measured station reaches its simulator as
files, not through an import of `machinekind_twin`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from conftest import BOT  # noqa: E402

from machinekind_twin import scene as sc  # noqa: E402
from machinekind_twin.robots import resolve_asset  # noqa: E402
from machinekind_twin.workspace import Workspace  # noqa: E402


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    ws = Workspace.load(resolve_asset("examples/twin.sim.json"))
    folder = ws.export(tmp_path_factory.mktemp("station"))
    return ws, folder


def test_exported_scene_matches_the_live_one(exported):
    ws, folder = exported
    live = sc.build(ws.scene_config())
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    assert (model.nbody, model.njnt, model.ncam, model.nmesh) == \
        (live.model.nbody, live.model.njnt, live.model.ncam, live.model.nmesh)
    assert model.opt.timestep == live.model.opt.timestep
    # The cameras where the live scene has them, with the same K.
    live.set_joints(ws.spec().home)
    q = live.data.qpos.copy()
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    # The XML carries quaternions with six significant digits, hence a tolerance of micrometres.
    assert np.allclose(data.cam_xpos, live.data.cam_xpos, atol=1e-5)
    assert np.allclose(data.cam_xmat, live.data.cam_xmat, atol=1e-5)
    assert np.array_equal(model.cam_intrinsic, live.model.cam_intrinsic)
    assert np.array_equal(model.cam_resolution, live.model.cam_resolution)
    # The arm under the same prefix, with the same kinematics: the TCP at home in the same place.
    tcp = sc.PREFIX + ws.spec().tcp_site
    assert np.allclose(data.site(tcp).xpos, live.data.site(tcp).xpos, atol=1e-5)


def test_exported_scene_takes_task_objects(exported):
    """A training package adds its task's objects through MjSpec, without the twin's code."""
    _, folder = exported
    station = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    spec = mujoco.MjSpec.from_file(str(folder / "scene.xml"))
    body = spec.worldbody.add_body()
    body.name = "cube"
    body.pos = [0.0, 0.0, 0.8]
    body.add_freejoint()
    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [0.015, 0.015, 0.015]
    model = spec.compile()
    assert model.body("cube").id > 0 and model.nbody == station.nbody + 1
    assert model.nmesh == station.nmesh and model.ncam == station.ncam


def test_station_json_is_the_workspace(exported):
    ws, folder = exported
    data = json.loads((folder / "station.json").read_text(encoding="utf-8"))
    assert data == ws.to_dict()
    assert data["robot"] == BOT and "dynamics" in data and data["table"]["base_xy"] == [-0.35, 0.0]


def test_card_and_panels_do_not_export(tmp_path):
    ws = Workspace.load(resolve_asset("examples/twin.sim.json"))
    with pytest.raises(ValueError, match="karta"):
        sc.export(ws.scene_config(with_card=True), tmp_path)
