"""Stanowisko jako dane: `scene.xml` + `station.json`, czytane zwyklym MuJoCo bez blizniaka.

To jest granica dla treningu (so101): zmierzone stanowisko ma trafic do symulatora
jako plik, nie przez import `lerobot_mp`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.robots import resolve_asset  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402


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
    # Kamery dokladnie tam, gdzie w scenie na zywo, z tym samym K.
    live.set_joints(ws.spec().home)
    q = live.data.qpos.copy()
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    # XML niesie kwaterniony z szescioma cyframi - stad tolerancja mikrometrow.
    assert np.allclose(data.cam_xpos, live.data.cam_xpos, atol=1e-5)
    assert np.allclose(data.cam_xmat, live.data.cam_xmat, atol=1e-5)
    assert np.array_equal(model.cam_intrinsic, live.model.cam_intrinsic)
    assert np.array_equal(model.cam_resolution, live.model.cam_resolution)
    # Ramie pod tym samym prefiksem i z ta sama kinematyka: TCP w domu w tym samym miejscu.
    tcp = sc.PREFIX + ws.spec().tcp_site
    assert np.allclose(data.site(tcp).xpos, live.data.site(tcp).xpos, atol=1e-5)


def test_exported_scene_takes_task_objects(exported):
    """Pakiet treningowy doklada obiekty zadania przez MjSpec, bez kodu blizniaka."""
    _, folder = exported
    spec = mujoco.MjSpec.from_file(str(folder / "scene.xml"))
    body = spec.worldbody.add_body()
    body.name = "cube"
    body.pos = [0.0, 0.0, 0.8]
    body.add_freejoint()
    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [0.015, 0.015, 0.015]
    model = spec.compile()
    assert model.body("cube").id > 0 and model.nmesh == 18


def test_station_json_is_the_workspace(exported):
    ws, folder = exported
    data = json.loads((folder / "station.json").read_text(encoding="utf-8"))
    assert data == ws.to_dict()
    assert data["robot"] == "so101" and "dynamics" in data and data["table"]["base_xy"] == [-0.35, 0.0]


def test_card_and_panels_do_not_export(tmp_path):
    ws = Workspace.load(resolve_asset("examples/twin.sim.json"))
    with pytest.raises(ValueError, match="karta"):
        sc.export(ws.scene_config(with_card=True), tmp_path)
