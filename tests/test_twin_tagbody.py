"""Bryla z tagami na obudowie: uklad tagow nauczony z kadrow i poza bryly z kamer, wzgledem prawdy.

Kostka 4 cm z tagiem 0 na sciance +z, 1 na +x i 2 na +y stoi w scenie; dwie kamery patrza z roznych
stron. Sesja widzi tylko kadry i K - uklad tagow i poza kostki wychodza z nich, a prawda sluzy
wylacznie do oceny. Do tego to, co przy tych samych numerach na karcie i obudowie musi zostac
odrzucone: tag widziany dwa razy w kadrze.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

from conftest import BOT  # noqa: E402

from machinekind_twin import scene as sc  # noqa: E402
from machinekind_twin.calib import tags  # noqa: E402
from machinekind_twin.calib.simulate import look_at  # noqa: E402
from machinekind_twin.kinematics import pose  # noqa: E402
from machinekind_twin.robots import get_spec  # noqa: E402
from machinekind_twin.tagbody import Observation, TagBody, learn_layout, locate, observe, optical_in_tag  # noqa: E402
from machinekind_twin.workspace import CameraRecord, Workspace  # noqa: E402

SPEC = get_spec(BOT)

TAG = 0.03
HALF = 0.02


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def true_layout() -> dict[int, np.ndarray]:
    """Tag 0 na sciance +z (uklad bryly), 1 na +x, 2 na +y kostki o srodku (0, 0, -HALF)."""
    return {0: np.eye(4),
            1: pose(_rot("y", 90.0), np.array([HALF, 0.0, -HALF])),
            2: pose(_rot("x", -90.0), np.array([0.0, HALF, -HALF]))}


def angle_deg(Ra, Rb) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


@pytest.fixture(scope="module")
def rig():
    T_body2base = pose(_rot("z", 25.0) @ _rot("x", 10.0), np.array([0.32, 0.10, 0.09]))
    layout = true_layout()
    plate = TAG / tags.MARKER_FRACTION
    panels = [sc.Panel(f"tag{i}", tags.render(i, 400), (plate, plate), T_body2base @ T) for i, T in layout.items()]
    centre = T_body2base[:3, 3]
    W, H, f = 800, 600, 700.0
    K = np.array([[f, 0.0, (W - 1) / 2], [0.0, f, (H - 1) / 2], [0.0, 0.0, 1.0]])
    # Kierunki w ukladzie bryly: cam0 z naroznika widzi trzy scianki, cam1 scianki +z i +x, cam2 +z i +y.
    R = T_body2base[:3, :3]
    eyes = [centre + 0.4 * R @ np.array(d) for d in ([0.55, 0.55, 0.65], [0.9, -0.35, 0.45], [-0.3, 0.9, 0.5])]
    views = [sc.CameraView(f"cam{k}", K, W, H, look_at(eye, centre)) for k, eye in enumerate(eyes)]
    cfg = sc.SceneConfig(SPEC, cameras=views, panels=panels)
    ws = Workspace(robot=BOT)
    for v in views:
        ws.add_camera(CameraRecord(v.name, "sim", W, H, K=K.tolist(), dist=None,
                                   T_cam2base=v.T_cam2base.tolist(), intrinsics_from="symulacja"))
    with sc.build(cfg) as scene:
        scene.set_joints({"j1": -90.0, "j2": -60.0})                    # ramie z dala od kostki
        frames = {v.name: scene.render(v.name) for v in views}
    return T_body2base, layout, ws, frames


@pytest.mark.render
def test_layout_learned_from_frames_and_pose_from_cameras_match_the_truth(rig):
    T_body2base, layout, ws, frames = rig
    body = TagBody("pmd", ids=(0, 1, 2), tag_size=TAG)

    obs = observe(frames, ws.cameras, body)
    assert [o.camera for o in obs] == ["cam0", "cam1", "cam2"]
    assert obs[0].ids == [0, 1, 2], "kamera z naroznika ma widziec wszystkie trzy scianki"
    # Sama mediana par: kazdy tag z osobna ma blad glebokosci kilku mm i uklad go dziedziczy.
    rough = learn_layout(obs, body, refine=False)
    assert rough.missing(body.ids) == []
    assert max(np.linalg.norm(rough.layout[i][:3, 3] - T[:3, 3]) for i, T in layout.items()) > 0.002
    fit = learn_layout(obs, body)
    assert fit.missing(body.ids) == [] and fit.pairs == {(0, 1): 2, (0, 2): 2, (1, 2): 1}
    for i, T in layout.items():
        got = fit.layout[i]
        assert np.linalg.norm(got[:3, 3] - T[:3, 3]) < 0.0015, i
        assert angle_deg(got[:3, :3], T[:3, :3]) < 1.0, i
    body.layout = fit.layout

    fix = locate(frames, ws.cameras, body)
    assert fix is not None and set(fix.per_camera) == {"cam0", "cam1", "cam2"}
    assert np.linalg.norm(fix.position - T_body2base[:3, 3]) < 0.0015
    assert angle_deg(fix.T_body2base[:3, :3], T_body2base[:3, :3]) < 1.0
    assert fix.spread_mm < 3.0
    assert max(fix.reproj_px.values()) < 0.6
    # Kazda kamera z osobna tez trafia - to, co pokazuje rozrzut miedzy kamerami w panelu.
    for T in fix.per_camera.values():
        assert np.linalg.norm(T[:3, 3] - T_body2base[:3, 3]) < 0.003

    # Bryla, z ktorej kamery widza tylko jeden tag, tez dostaje poze (PnP jednego taga jak w `tagreach`);
    # kamera patrzaca na ten tag z ukosa (cam0) ma blad glebokosci kilkunastu mm, frontalna (cam2) nie.
    only = TagBody("pmd", ids=(2,), tag_size=TAG, layout={2: layout[2]})
    fix2 = locate(frames, ws.cameras, only)
    assert fix2 is not None and set(fix2.per_camera) == {"cam0", "cam2"}
    assert fix2.tags_seen == {"cam0": [2], "cam2": [2]}
    assert np.linalg.norm(fix2.per_camera["cam2"][:3, 3] - T_body2base[:3, 3]) < 0.005


@pytest.mark.render
def test_layout_reaches_a_tag_only_ever_seen_next_to_another_non_reference_tag(rig):
    _, layout, ws, frames = rig
    body = TagBody("pmd", ids=(0, 1, 2), tag_size=TAG)
    obs = observe(frames, ws.cameras, body)
    cam0 = obs[0]

    def only(ids):
        return Observation(cam0.camera, cam0.K, cam0.dist, {i: cam0.corners[i] for i in ids},
                           {i: cam0.poses[i] for i in ids})

    # Tag 2 widziany wylacznie razem z tagiem 1 (nigdy z 0): poza przez lancuch 2 -> 1 -> 0.
    fit = learn_layout([only((0, 1)), only((1, 2))], body)
    assert fit.missing(body.ids) == [] and (0, 2) not in fit.pairs
    # Jedyny kadr z tagiem 2 patrzy na niego z ukosa: obrot wychodzi, glebokosc (kilkanascie mm) nie -
    # po to w pierwszym tescie jest trzecia kamera, a w panelu przycisk "dodaj kadr" z roznych stron.
    assert angle_deg(fit.layout[2][:3, :3], layout[2][:3, :3]) < 3.0
    assert np.linalg.norm(fit.layout[2][:3, 3] - layout[2][:3, 3]) < 0.025
    # Bez zadnego kadru laczacego tag 2 z reszta - zostaje nieznany i `locate` go nie uzywa.
    fit = learn_layout([only((0, 1))], body)
    assert fit.missing(body.ids) == [2]


def test_round_trip_through_json_and_optical_axis():
    layout = true_layout()
    body = TagBody("pmd", ids=(0, 1, 2), tag_size=0.02, layout=layout, optical=(0, "-z"))
    # A saved pose next to the body (`T_body2base`) is not part of it - `from_dict` skips it.
    data = json.loads(json.dumps({**body.to_dict(), "T_body2base": np.eye(4).tolist()}))
    b2 = TagBody.from_dict(data)
    assert b2.ids == (0, 1, 2) and b2.optical == (0, "-z") and abs(b2.tag_size - 0.02) < 1e-12
    for i in layout:
        assert np.allclose(b2.layout[i], layout[i])
    # Kamera patrzy od tylnej scianki (tag 0, os -z): z optyczne = -z taga, obraz "w dol" = -y taga.
    T = b2.T_optical2body()
    assert T is not None and np.allclose(T[:3, 2], [0, 0, -1]) and np.allclose(T[:3, 1], [0, -1, 0])
    for axis in ("-z", "+x", "-x", "+y", "-y"):
        R = optical_in_tag(axis)[:3, :3]
        assert abs(np.linalg.det(R) - 1) < 1e-9 and np.allclose(R.T @ R, np.eye(3))


def test_a_tag_id_seen_twice_in_one_frame_is_dropped_not_picked_at_random():
    img = np.full((400, 900, 3), 255, np.uint8)
    t = tags.render(0, 200)
    img[100:300, 50:250] = t[:, :, None]
    img[100:300, 650:850] = t[:, :, None]
    img[100:300, 350:550] = tags.render(7, 200)[:, :, None]
    seen = tags.detect(img)
    assert 0 not in seen and 7 in seen
    assert 0 in tags.detect(np.ascontiguousarray(img[:, :600]))
