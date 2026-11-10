"""Fixtures of the station: described by a robot package, placed in the scene, kept clear of, carrying cameras.

The test robot's pole (`examples/robots/testbot/testbot/robot/fixtures/pole`) stands 15 cm to the
driven arm's left: a foot with a mesh, a 20 cm segment and a head with a camera site.
"""

from __future__ import annotations

import json
import socket
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco
from conftest import BOT

from machinekind_twin import fixtures as fx
from machinekind_twin import scene as sc
from machinekind_twin import stand as st
from machinekind_twin.calib.card import Card
from machinekind_twin.calib.session import Session
from machinekind_twin.clock import VirtualClock
from machinekind_twin.collision import CollisionChecker
from machinekind_twin.kinematics import RobotKinematics
from machinekind_twin.robots import get_spec
from machinekind_twin.runtime import Twin
from machinekind_twin.workspace import CameraRecord, FixtureRecord, RobotRecord, Workspace

POLE = f"{BOT}/pole"
#: The driven arm turned left, through the pole; and turned half as far, beside it.
INTO_IT = {"j1": 90.0, "j2": 0.0, "gripper": 0.0}
BESIDE = {"j1": 30.0, "j2": 0.0, "gripper": 0.0}
#: Heights [m]: the foot, a segment, the head.
FOOT, SEGMENT, HEAD = 0.01, 0.2, 0.05
DT = 0.02


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def station(**kw) -> Workspace:
    ws = Workspace(robot=BOT)
    ws.fixtures.append(FixtureRecord("slup", POLE, **kw))
    return ws


def body_pos(scene: sc.Scene, name: str) -> np.ndarray:
    return scene.data.xpos[scene.model.body(name).id].copy()


# ------------------------------------------------------------------ description
def test_the_robot_package_brings_its_fixture():
    assert POLE in fx.fixture_folders()
    spec = fx.get_fixture(POLE)
    assert (spec.robot, spec.name, spec.stack) == (BOT, "pole", ("foot", "segment", "head"))
    assert set(spec.placements) == {"left", "right"}
    with pytest.raises(KeyError, match="nieznane mocowanie"):
        fx.get_fixture(f"{BOT}/nothing")


def test_a_broken_description_names_its_problems(tmp_path):
    folder = tmp_path / "pole"
    folder.mkdir()
    (folder / "fixture.yaml").write_text(
        "name: other\ntitle: x\nstack: [a, b]\nparts:\n  a: {mesh: missing.stl}\n", encoding="utf-8")
    with pytest.raises(ValueError) as err:
        fx.load_fixture(folder, BOT)
    for what in ("name: other", "nie ma czesci 'b'", "missing.stl"):
        assert what in str(err.value)


def test_the_stack_puts_each_part_on_top_of_the_one_below():
    spec = fx.get_fixture(POLE)
    heights = [T[2, 3] for _, T in fx.stack_poses(spec)]
    assert heights == pytest.approx([0.0, FOOT, FOOT + SEGMENT])
    # Another build of the same parts: two segments, the camera one segment higher.
    two = ["foot", "segment", "segment", "head"]
    assert [T[2, 3] for _, T in fx.stack_poses(spec, two)][-1] == pytest.approx(FOOT + 2 * SEGMENT)
    assert fx.camera_site(spec, two)[2, 3] - fx.camera_site(spec)[2, 3] == pytest.approx(SEGMENT)
    assert [p.name for p in fx.placed(spec, "slup", np.eye(4), two).parts] == ["foot", "segment", "segment2", "head"]
    with pytest.raises(ValueError, match="nie ma czesci"):
        fx.stack_poses(spec, ["foot", "roof"])


# ------------------------------------------------------------------ workspace
def test_a_workspace_without_fixtures_saves_as_before(tmp_path):
    ws = Workspace(robot=BOT)
    ws.add_camera(CameraRecord("kamera1"))
    data = ws.to_dict()
    assert "fixtures" not in data and "fixture" not in data["cameras"][0] and "mount" not in data["cameras"][0]
    path = ws.save(tmp_path / "twin.json")
    loaded = Workspace.load(path)
    assert loaded.fixtures == [] and loaded.to_dict() == data
    assert sc.compose(loaded.scene_config()).to_xml() == sc.compose(ws.scene_config()).to_xml()


def test_fixtures_round_trip_and_unknown_keys_fail(tmp_path):
    ws = station(placement="right", parts=["foot", "head"], rgba=[1.0, 0.5, 0.0, 1.0])
    ws.add_camera(CameraRecord("gora", fixture="slup"))
    path = ws.save(tmp_path / "twin.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["fixtures"][0]["fixture"] == POLE and data["cameras"][0]["fixture"] == "slup"
    assert "mount" not in data["cameras"][0]
    loaded = Workspace.load(path)
    assert loaded.fixtures == ws.fixtures and loaded.camera("gora").mount is not None

    data["fixtures"][0]["height"] = 1.0
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="height"):
        Workspace.load(path)


# ---------------------------------------------------------------------- scene
def test_the_fixture_stands_in_the_scene_with_its_mesh_and_colliders():
    alone = sc.build(Workspace(robot=BOT).scene_config())
    scene = sc.build(station().scene_config())
    m, m0 = scene.model, alone.model
    # Everything of the scene without it keeps its name and id.
    assert [m.body(b).name for b in range(m0.nbody)] == [m0.body(b).name for b in range(m0.nbody)]
    assert (m.njnt, m.nq, m.nu, m.ncam) == (m0.njnt, m0.nq, m0.nu, m0.ncam)

    names = {m.geom(g).name: g for g in np.flatnonzero(scene.fixture_geom)}
    assert set(names) == {"slup/foot", "slup/foot/plate", "slup/segment/pole", "slup/head/head"}
    assert scene.kept_geom[list(names.values())].all() and not scene.robot_geom[list(names.values())].any()
    mesh = names["slup/foot"]
    assert m.geom_type[mesh] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[mesh] == 0
    colliders = [names[n] for n in ("slup/foot/plate", "slup/segment/pole", "slup/head/head")]
    assert (m.geom_margin[colliders] == sc.PASSIVE_CLEARANCE).all() and (m.geom_contype[colliders] > 0).all()
    # A collider is drawn only where its part has no mesh.
    assert m.geom_group[names["slup/foot/plate"]] == 3 and m.geom_group[names["slup/segment/pole"]] == 1
    # Its foot 15 cm left of the driven arm's base, its parts stacked.
    base = scene.T_base2world[:3, 3]
    assert body_pos(scene, "slup/foot") - base == pytest.approx([0.0, 0.15, 0.0])
    assert body_pos(scene, "slup/head") - base == pytest.approx([0.0, 0.15, FOOT + SEGMENT])


def test_left_and_right_mirror_and_xyz_replaces_the_placement():
    def foot(**kw) -> np.ndarray:
        ws = station(**kw)
        return ws.fixture_pose(ws.fixtures[0])[:3, 3]

    assert foot() == pytest.approx([0.0, 0.15, 0.0])                    # the first placement
    assert foot(placement="right") == pytest.approx([0.0, -0.15, 0.0])
    assert foot(placement="left", xyz=[0.02, 0.18, 0.0]) == pytest.approx([0.02, 0.18, 0.0])
    ws = station(xyz=[0.1, 0.0, 0.0], yaw_deg=90.0)
    assert ws.fixture_pose(ws.fixtures[0])[:3, 0] == pytest.approx([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="nie ma polozenia"):
        foot(placement="behind")


def test_a_fixture_on_another_robot_or_on_the_table():
    ws = station(on="lewe")
    ws.other_robots.append(RobotRecord("lewe", BOT, xyz=[0.0, 0.4, 0.0], yaw_deg=180.0))
    # The other robot's left is the driven arm's right.
    assert ws.fixture_pose(ws.fixtures[0])[:3, 3] == pytest.approx([0.0, 0.25, 0.0])

    ws = station(on="table", xyz=[0.0, 0.0, 0.0])
    ws.table["base_xy"] = [-0.35, 0.1]
    assert ws.fixture_pose(ws.fixtures[0])[:3, 3] == pytest.approx([0.35, -0.1, 0.0])
    with pytest.raises(ValueError, match="podaj xyz"):
        station(on="table").placed_fixtures()
    with pytest.raises(ValueError, match="nie ma takiego robota"):
        station(on="prawe").placed_fixtures()


def test_the_fixture_exports_with_its_meshes(tmp_path):
    ws = station()
    path = sc.export(ws.scene_config(), tmp_path)
    xml = path.read_text(encoding="utf-8")
    assert 'file="slup/foot.stl"' in xml and str(tmp_path) not in xml
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    live = sc.build(ws.scene_config())
    assert np.allclose(data.xpos[model.body("slup/head").id], body_pos(live, "slup/head"), atol=1e-6)


# ------------------------------------------------------------------ collisions
@pytest.mark.parametrize("clearance", [0.0, 0.02, 0.05])
def test_a_pose_through_the_pole_collides_and_one_beside_it_is_clear(clearance):
    checker = CollisionChecker(sc.build(station().scene_config()), clearance=clearance)
    home = get_spec(BOT).home
    assert checker.config_clear(home) is None
    assert checker.config_clear(BESIDE) is None and checker.passive_gap(BESIDE) is None
    assert (checker.config_clear(INTO_IT) or "").startswith("kolizja") and "slup/" in checker.config_clear(INTO_IT)
    assert "slup/" in (checker.path_clear(home, INTO_IT) or "")
    gap, what = checker.passive_gap(INTO_IT)
    assert gap < 0 and "slup/" in what
    assert "za blisko elementu stanowiska" in (checker.closer(BESIDE, INTO_IT) or "")
    assert checker.closer(INTO_IT, BESIDE) is None


def test_without_the_pole_the_same_pose_is_clear():
    assert CollisionChecker(sc.build(Workspace(robot=BOT).scene_config())).config_clear(INTO_IT) is None


def test_the_loop_stops_the_arm_before_the_pole():
    twin = Twin(station(), clock=VirtualClock(t0=0.0))
    try:
        twin.connect("sim", threaded=False)
        twin.claim("test")
        twin.set_engaged(True, owner="test")
        twin.set_target(INTO_IT, owner="test")
        checker = CollisionChecker(twin.scene)
        closest = np.inf
        for _ in range(int(3.0 / DT)):
            twin.step(DT)
            gap = checker.passive_gap(twin.joints())
            closest = min(closest, np.inf if gap is None else gap[0])
        assert twin.owner is None and "za blisko elementu stanowiska" in twin.status.error
        assert closest > 0.0
        assert twin.joints()["j1"] < INTO_IT["j1"] - 10.0
    finally:
        twin.close()


# --------------------------------------------------------------------- cameras
def test_a_camera_on_the_fixture_stands_at_its_site_until_measured():
    ws = station(parts=["foot", "segment", "segment", "head"])
    ws.add_camera(CameraRecord("gora", source="sim", fixture="slup"))
    ws.add_camera(CameraRecord("real", source="1", fixture="slup/camera"))
    site = ws.fixture_pose(ws.fixtures[0]) @ fx.camera_site(fx.get_fixture(POLE), ws.fixtures[0].parts)
    assert site[:3, 3] == pytest.approx([0.05, 0.15, FOOT + 2 * SEGMENT])
    assert ws.camera("gora").true_pose() == pytest.approx(site)
    assert ws.camera("real").true_pose() == pytest.approx(site)
    scene = sc.build(ws.scene_config())
    assert scene.camera_pose("gora") == pytest.approx(site, abs=1e-6)
    assert scene.render("gora").shape == (480, 640, 3)

    # A measured pose wins; the simulated camera's own true pose does too.
    measured = np.eye(4)
    measured[:3, 3] = [0.5, 0.0, 0.4]
    ws.camera("real").T_cam2base = measured.tolist()
    ws.camera("gora").sim_pose = measured.tolist()
    assert ws.camera("real").true_pose() == pytest.approx(measured)
    assert ws.camera("gora").true_pose() == pytest.approx(measured)

    with pytest.raises(ValueError, match="nie ma go w stanowisku"):
        ws.add_camera(CameraRecord("zla", fixture="dach"))


class _Cameras:
    def __init__(self, mounts):
        self._mounts = mounts

    def grab(self):
        return {}

    def mounts(self):
        return self._mounts


def test_the_wave_starts_from_the_mount():
    ws = station()
    ws.add_camera(CameraRecord("gora", source="sim", fixture="slup"))
    spec = get_spec(BOT)
    kin = RobotKinematics(spec)
    card = Card()
    session = Session(None, _Cameras({"gora": ws.camera("gora").mount}), {"gora": (np.eye(3), None)},  # type: ignore[arg-type]
                      kin, card, np.eye(4))
    assert session.progress["gora"].rough == pytest.approx(np.asarray(ws.camera("gora").mount))
    assert session.status()["gora"].startswith("z mocowania")
    session._refit("gora")                                 # nothing agrees yet: the mount stays
    assert session.progress["gora"].rough == pytest.approx(np.asarray(ws.camera("gora").mount))


def test_the_wave_job_hands_the_wave_the_mounts_of_its_cameras():
    from machinekind_twin.ui.jobs import _CameraSubset

    ws = station()
    ws.add_camera(CameraRecord("gora", source="sim", fixture="slup"))
    ws.add_camera(CameraRecord("bok", source="sim"))
    mounts = _CameraSubset(SimpleNamespace(workspace=ws), ["gora", "bok"]).mounts()
    assert list(mounts) == ["gora"] and mounts["gora"] == pytest.approx(ws.camera("gora").true_pose())


# ----------------------------------------------------------------------- panel
class _SceneRecorder:
    """A viser scene API that records what the 3D view adds: (method, name)."""

    def __init__(self):
        self.added: list[tuple[str, str]] = []

    def __getattr__(self, method):
        def add(name, *args, **kw):
            self.added.append((method, name))
            return SimpleNamespace(remove=lambda: None)
        return add


def test_the_3d_view_draws_the_mesh_and_the_colliders_of_a_part_without_one():
    from machinekind_twin.ui.bridge import SceneMirror

    scene = sc.build(station().scene_config())
    recorder = _SceneRecorder()
    SceneMirror(SimpleNamespace(scene=recorder), scene.model)
    drawn = {scene.model.geom(int(name.rsplit("/g", 1)[1])).name: method
             for method, name in recorder.added if method != "add_frame"}
    assert drawn["slup/foot"] == "add_mesh_simple" and "slup/foot/plate" not in drawn
    assert drawn["slup/segment/pole"] == drawn["slup/head/head"] == "add_box"


@pytest.mark.render
def test_the_panel_starts_with_the_fixture_and_its_camera(tmp_path):
    from machinekind_twin.ui.app import TwinApp

    ws = station()
    ws.add_camera(CameraRecord("gora", source="sim", fixture="slup"))
    path = ws.save(tmp_path / "twin.json")
    app = TwinApp(path, host="127.0.0.1", port=_free_port())
    try:
        assert "slup/head" in [app.twin.scene.model.body(b).name for b in range(app.twin.scene.model.nbody)]
        frames = app._tick_slow()
        assert frames["gora"].shape == (480, 640, 3)
        assert app.frustums["gora"].visible
    finally:
        app.close()


# ---------------------------------------------------------------------- stands
def _stand(tmp_path, **extra):
    folder = tmp_path / "stanowisko"
    folder.mkdir()
    data = {"format": 1, "name": "test", "robots": [{"robot": BOT, "read_only": False}],
            "cameras": [{"name": "gora", "source": "sim", "width": 640, "height": 480, "fixture": "slup"}],
            "fixtures": [{"name": "slup", "fixture": POLE, "placement": "left", "parts": ["foot", "head"]}], **extra}
    (folder / st.STAND_FILE).write_text(json.dumps(data), encoding="utf-8")
    return folder


def test_a_stand_with_a_fixture_loads_and_writes_it_back(tmp_path):
    folder = _stand(tmp_path)
    ws = st.load(folder).workspace()
    assert ws.fixtures == [FixtureRecord("slup", POLE, placement="left", parts=["foot", "head"])]
    assert ws.camera("gora").fixture == "slup" and ws.camera("gora").mount is not None
    ws.save(tmp_path / "twin.json")
    data = json.loads((folder / st.STAND_FILE).read_text(encoding="utf-8"))
    assert data["fixtures"] == [{"name": "slup", "fixture": POLE, "on": "robot", "placement": "left",
                                 "parts": ["foot", "head"], "yaw_deg": 0.0}]
    assert data["cameras"][0]["fixture"] == "slup"
    assert list(data).index("fixtures") == list(data).index("cameras") + 1
    assert "slup/head" in (folder / "station" / "scene.xml").read_text(encoding="utf-8")


@pytest.mark.parametrize(("change", "message"), [
    ({"fixtures": [{"name": "robot", "fixture": POLE}]}, "fixtures[0].name"),
    ({"fixtures": [{"name": "slup", "fixture": "pole"}]}, "<robot>/<mocowanie>"),
    ({"fixtures": [{"name": "slup", "fixture": POLE, "on": "prawe"}]}, ".on"),
    ({"fixtures": [{"name": "slup", "fixture": POLE, "on": "table"}]}, "potrzebuje xyz"),
    ({"fixtures": [{"name": "slup", "fixture": POLE, "height": 1}]}, "nieznane pola"),
    ({"fixtures": []}, "nie ma mocowania 'slup'"),
])
def test_a_broken_fixture_entry_names_the_key(tmp_path, change, message):
    with pytest.raises(st.StandError, match=message.replace("[", r"\[").replace("]", r"\]")):
        st.load(_stand(tmp_path, **change))


def test_a_stand_without_fixtures_writes_none(tmp_path):
    folder = _stand(tmp_path, fixtures=[], cameras=[])
    ws = st.load(folder).workspace()
    ws.save(tmp_path / "twin.json")
    assert "fixtures" not in json.loads((folder / st.STAND_FILE).read_text(encoding="utf-8"))
