"""Mrowka RealAnt w blizniaku: tagi na grzbiecie (2) i maszcie (1), uklad z kadrow, montaz z blatu, poza z kamer.

Mrowka z prawdziwymi tagami stoi w scenie blizniaka (`Workspace.scene_config(with_ant=True)`, to samo,
co buduje runtime), trzy kamery symulowane patrza na nia z roznych stron. Sesja kalibracji widzi tylko
kadry i K kamer - uklad tagow, wysokosc i pochylenie taga na grzbiecie oraz poza mrowki wychodza z nich,
a prawda (sekcja `sim`) sluzy wylacznie do oceny.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.ant import (MAST_IN_BACK_TAG, LEG_POSES, Ant, level_mount, locate, mount_matrix,  # noqa: E402
                                 rest_height, snap_yaw, square_layout, zyx)
from lerobot_mp.twin.calib.simulate import look_at  # noqa: E402
from lerobot_mp.twin.kinematics import pose  # noqa: E402
from lerobot_mp.twin.tagbody import learn_layout, observe  # noqa: E402
from lerobot_mp.twin.workspace import CameraRecord, Workspace  # noqa: E402

TAG = 0.040
#: Prawdziwy montaz taga 2 na grzbiecie: troche obok srodka, na tekturze, osie wzdluz nog (135 st. od przodu),
#: lekko krzywo.
MOUNT = {"x": 0.012, "y": -0.006, "z": 0.031, "yaw": 135.0, "pitch": 2.5, "roll": -1.5}


def angle_deg(Ra, Rb) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def true_layout() -> dict[int, np.ndarray]:
    """Tag 1 na maszcie w ukladzie taga 2, jak na zdjeciu stanowiska: prostopadle, krawedzie rownolegle, nad noga."""
    return {1: pose(MAST_IN_BACK_TAG, np.array([0.095, -0.005, 0.075]))}


def ant_at(xy, heading_deg: float, joints=LEG_POSES["lezy"]) -> np.ndarray:
    return pose(rz(heading_deg), np.array([xy[0], xy[1], rest_height(joints)]))


def workspace(T_ant2base: np.ndarray) -> Workspace:
    W, H, f = 800, 600, 700.0
    K = np.array([[f, 0.0, (W - 1) / 2], [0.0, f, (H - 1) / 2], [0.0, 0.0, 1.0]])
    ws = Workspace()
    centre = np.array([0.30, 0.05, 0.05])
    for k, eye in enumerate(([0.75, 0.30, 0.40], [0.70, -0.30, 0.35], [0.25, 0.55, 0.45])):
        T = look_at(np.array(eye), centre)
        ws.add_camera(CameraRecord(f"cam{k}", "sim", W, H, K=K.tolist(), dist=None, T_cam2base=T.tolist(),
                                   sim_pose=T.tolist(), intrinsics_from="symulacja"))
    truth = Ant(sim={"T_ant2base": T_ant2base, "mount": MOUNT, "layout": true_layout()})
    truth.tags.tag_size = TAG
    ws.ant = truth.to_dict()
    return ws


def frames_of(ws: Workspace) -> dict[str, np.ndarray]:
    with sc.build(ws.scene_config(with_ant=True)) as scene:
        scene.set_joints({"shoulder_pan": 110.0, "shoulder_lift": -95.0, "elbow_flex": 90.0})
        return {c.name: scene.render(c.name) for c in ws.cameras}


@pytest.mark.render
def test_ant_layout_mount_and_pose_from_sim_cameras_match_the_truth():
    A, B = ant_at((0.30, 0.10), 15.0), ant_at((0.27, -0.03), -40.0)
    ws_a, ws_b = workspace(A), workspace(B)
    fa, fb = frames_of(ws_a), frames_of(ws_b)

    # Sesja: nic nie wie o tagach poza numerami i bokiem.
    ant = Ant()
    ant.tags.tag_size = TAG
    assert ant.tags.ids == (2, 1)
    obs = observe(fa, ws_a.cameras, ant.tags) + observe(fb, ws_b.cameras, ant.tags)
    assert sum(1 for o in obs if o.ids == [1, 2]) >= 3, [o.ids for o in obs]
    fit = learn_layout(obs, ant.tags)
    assert fit.missing(ant.tags.ids) == []
    T1 = true_layout()[1]
    assert np.linalg.norm(fit.layout[1][:3, 3] - T1[:3, 3]) < 0.0015
    assert angle_deg(fit.layout[1][:3, :3], T1[:3, :3]) < 1.0
    # Maszt prostopadly (panel: domyslnie): przyciagniecie wyuczonego obrotu daje dokladnie prawde.
    snapped, offs = square_layout({1: fit.layout[1]})
    assert np.allclose(snapped[1][:3, :3], T1[:3, :3], atol=1e-12) and offs[1] < 1.0
    ant.tags.layout = {2: np.eye(4), **snapped}

    # Montaz: x, y i obrot wokol pionu z linijki (tu z prawdy), wysokosc i pochylenie z blatu.
    ant.mount = {**ant.mount, "x": MOUNT["x"], "y": MOUNT["y"], "yaw": MOUNT["yaw"]}
    fix = locate(fa, ws_a.cameras, ant)
    assert fix is not None and set(fix.tags.per_camera) == {"cam0", "cam1", "cam2"}
    ant.mount = level_mount(fix.tags.T_body2base, ant)
    # Wysokosc to glebokosc pozy z kamer (mediana trzech, 2,6 mm zmierzone), pochylenie niosa oba tagi (0,2 st.).
    assert abs(ant.mount["z"] - MOUNT["z"]) < 0.0035
    assert abs(ant.mount["pitch"] - MOUNT["pitch"]) < 0.7 and abs(ant.mount["roll"] - MOUNT["roll"]) < 0.7

    # Poza mrowki w drugim miejscu - innymi kadrami niz te, z ktorych wypoziomowano montaz.
    fix = locate(fb, ws_b.cameras, ant)
    assert fix is not None
    # Tu tagi widza tylko dwie kamery, jedna z nich sam tag na grzbiecie: PnP jednego taga 40 mm z 0,5 m ma blad
    # glebokosci kilku mm (7 mm zmierzone) - tyle trzyma polozenie; kierunek i pochylenie ponizej stopnia.
    assert np.linalg.norm(fix.position - B[:3, 3]) < 0.008
    assert angle_deg(fix.T_ant2base[:3, :3], B[:3, :3]) < 1.0
    assert abs(fix.heading_deg - (-40.0)) < 1.0 and fix.tilt_deg < 1.0
    # Rozrzut miedzy kamerami (panel go pokazuje) jest wiekszy niz blad mediany: kamera z jednym tagiem
    # z ukosa rozjezdza sie w glebokosci o ~1 cm, mediana z trzech go tlumi.
    assert fix.tags.spread_mm < 15.0

    # Bez obrotu montazu (linijki nie przylozono) nogi w blizniaku sa obrocone dokladnie o tyle.
    wrong = Ant(tags=ant.tags, mount={**ant.mount, "yaw": 0.0})
    fix0 = locate(fb, ws_b.cameras, wrong)
    assert abs(np.degrees(np.angle(np.exp(1j * np.radians(fix0.heading_deg - (-40.0 + MOUNT["yaw"])))))) < 1.0

    assert snap_yaw(137.8) == pytest.approx((135.0, 2.8)) and snap_yaw(-47.0) == pytest.approx((-45.0, -2.0))
    assert snap_yaw(90.0)[0] == 90.0                        # 45 st. od siatki - bez przyciagania


def test_level_mount_refuses_a_tag_that_is_not_lying_on_the_back():
    ant = Ant()
    standing_tag = pose(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]), np.array([0.3, 0.0, 0.1]))
    with pytest.raises(ValueError, match="pochylony o 90"):
        level_mount(standing_tag, ant)
    # Tag lezacy plasko 5 mm nad gorna plyta, mrowka na brzuchu: z montazu = 24,4 + 5 mm.
    flat = pose(rz(70.0), np.array([0.3, 0.0, 0.0244 + 0.0244 + 0.005]))
    m = level_mount(flat, ant)
    assert abs(m["z"] - (0.0244 + 0.005)) < 1e-9 and abs(m["pitch"]) < 1e-9 and abs(m["roll"]) < 1e-9
    # Na palcach korpus jest wyzej - ten sam tag to montaz nizej.
    ant.joints = dict(LEG_POSES["stoi"])
    assert level_mount(flat, ant)["z"] < m["z"] - 0.04


def test_ant_round_trip_scene_only_in_the_twin_and_live_pose(tmp_path):
    from lerobot_mp.twin import runtime as rt

    T = ant_at((0.30, 0.10), 15.0)
    ant = Ant(mount=MOUNT, joints=LEG_POSES["stoi"], T_ant2base=T)
    ant.tags.layout = {2: np.eye(4), **true_layout()}
    ws = Workspace(path=tmp_path / "twin.json")
    ws.ant = ant.to_dict()
    ws.save()
    back = Workspace.load(tmp_path / "twin.json").ant_obj()
    assert back.tags.ids == (2, 1) and back.tags.known() == [2, 1] and not back.simulated
    assert np.allclose(back.T_ant2base, T) and back.mount == pytest.approx(MOUNT)
    assert np.allclose(back.tag_poses()[1], ant.tag_poses()[1])
    assert zyx(mount_matrix(MOUNT)[:3, :3]) == pytest.approx((MOUNT["yaw"], MOUNT["pitch"], MOUNT["roll"]))

    # Srodowiska RL, kolizje i Spectacles biora `scene_config()` - bez mrowki; runtime z mrowka.
    assert ws.scene_config().ant is None
    tw = rt.Twin(ws)
    try:
        assert np.allclose(tw.scene.ant_pose(), T, atol=1e-9)
        home = tw.joints()
        T2 = ant_at((0.20, -0.10), 90.0, LEG_POSES["stoi"])
        assert tw.place_ant(T2)
        assert np.allclose(tw.scene.ant_pose(), T2, atol=1e-9)
        # Przebudowa (np. nowa kamera) stawia mrowke tam, gdzie mowi stanowisko, ramie zostaje, jak bylo.
        ws.ant = {**ws.ant, "T_ant2base": T2.tolist()}
        tw.rebuild()
        assert np.allclose(tw.scene.ant_pose(), T2, atol=1e-9)
        assert tw.joints() == pytest.approx(home, abs=1e-6)
        m, d = tw.scene.model, tw.scene.data
        ankle = np.degrees(d.qpos[m.jnt_qposadr[m.joint("ant/ankle_3").id]])
        assert abs(ankle - LEG_POSES["stoi"]["ankle_3"]) < 0.5
    finally:
        tw.close()
