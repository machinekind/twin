"""The calibration wave on the SO-101: finding cameras wherever they stand, and the stand's limits.

The stand of 2026-11-03: a Logitech C920 high above the table in front of the arm looking down steeply, a
second camera low from the front-right, a 100 mm tag, and a second SO-101 on the table to the left of the
working arm (+y), kept out with a box. The old wave saw the card in 5 of 90 poses there and saved nothing.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from conftest import BOT


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


so101 = pytest.mark.skipif(not _installed("soarm"), reason="SO-101 przychodzi z pakietu soarm")

#: The second arm next to the working one on the stand, in the working arm's base frame [m].
SECOND_ARM = {"x": (-0.12, 0.20), "y": (0.09, 0.30), "z": (0.0, 0.40), "name": "drugie ramie"}


class _StillRobot:
    def __init__(self, q):
        self.q = dict(q)

    def joints(self):
        return dict(self.q)

    def move(self, joints, duration):
        self.q.update(joints)


class _BlindCameras:
    def grab(self):
        return {}


def _blind_session(spec_name="so101", seed=0, cfg=None, objects=(), clearance=0.0, cameras=("c0",)):
    """A session on a still robot with cameras that see nothing: the wave plans and moves, nothing more."""
    from machinekind_twin import scene as sc
    from machinekind_twin.calib.card import Card, pinch_point
    from machinekind_twin.calib.session import Session, WaveConfig
    from machinekind_twin.collision import CollisionChecker
    from machinekind_twin.kinematics import RobotKinematics
    from machinekind_twin.robots import get_spec

    spec = get_spec(spec_name)
    card = Card.sized(0.1, 0.125)
    nominal = card.nominal(pinch_point(RobotKinematics(spec)), spec)
    check = sc.build(sc.SceneConfig(spec, card=card, card_pose=nominal, card_collider=0.012, objects=list(objects)))
    home = dict(spec.home)
    if spec.gripper:
        home[spec.gripper] = 0.0
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    s = Session(_StillRobot(home), _BlindCameras(), {c: (K, None) for c in cameras}, check.kin, card, nominal,
                CollisionChecker(check, clearance), cfg or WaveConfig.for_spec(spec), seed=seed)
    return s, check


def _along(s, start, end, step=0.01):
    """Joints every `step` rad of the largest joint's travel on the straight move start -> end."""
    q0, q1 = s.kin.to_q(start), s.kin.to_q(end)
    n = max(1, int(np.ceil(np.max(np.abs(q1 - q0)) / step)))
    return [s.kin.from_q(q0 + t * (q1 - q0)) for t in np.linspace(0.0, 1.0, n + 1)]


def _gripper_and_card(check, joints):
    """From the scene's own geometry, base frame: the corners of the card and the centres of every geom of
    the hand and its jaws."""
    import mujoco

    from machinekind_twin.kinematics import inverse
    from machinekind_twin.scene import PREFIX

    m, d, kin = check.model, check.data, check.kin
    kin.write_qpos(d.qpos, kin.to_q(joints))
    mujoco.mj_kinematics(m, d)
    hand = m.body(PREFIX + check.cfg.robot.hand_body).id
    in_hand = [g for g in range(m.ngeom) if hand in (m.geom_bodyid[g], m.body_parentid[m.geom_bodyid[g]])]
    pad = m.geom(PREFIX + "calib_card").id
    R, size = d.geom_xmat[pad].reshape(3, 3), m.geom_size[pad]
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    pts = np.vstack([d.geom_xpos[pad] + (signs * size) @ R.T, d.geom_xpos[in_hand]])
    B = inverse(check.T_base2world)
    return pts @ B[:3, :3].T + B[:3, 3]


@so101
def test_the_wave_keeps_the_card_and_gripper_out_of_the_second_arm_and_off_the_table():
    """With the stand's box around the second arm and the base's turn limited, every pose and every move
    in between keeps the card and the gripper out of the box and above the table, and the base inside
    its range - while still exploring right up to both. (The box reaches 9 cm from the base's axis: the
    arm's own links keep the base from turning far toward it.)"""
    from machinekind_twin.calib.session import KeepOut, WaveConfig
    from machinekind_twin.robots import get_spec

    box = KeepOut(**SECOND_ARM)
    cfg = replace(WaveConfig.for_spec(get_spec("so101")), keepout=(box,), base_turn=(-90.0, 60.0))
    near, pans = np.inf, []
    for seed in range(2):
        s, check = _blind_session(seed=seed, cfg=cfg)
        try:
            for _ in range(60):
                s.step()
            for a, b in s.moves:
                pans.append(b[s._pan])
                for j in _along(s, a, b):
                    pts = _gripper_and_card(check, j)
                    assert not box.hit(pts), f"seed {seed}: {j}"
                    assert pts[:, 2].min() >= cfg.table_margin - 0.002            # the card is 2 mm thick
                    lo, hi = np.array([box.x[0], box.y[0], box.z[0]]), np.array([box.x[1], box.y[1], box.z[1]])
                    near = min(near, float(np.linalg.norm(np.maximum(lo - pts, 0) + np.maximum(pts - hi, 0),
                                                          axis=1).min()))
        finally:
            check.close()
    assert all(-90.0 - 1e-6 <= v <= 60.0 + 1e-6 for v in pans)
    assert min(pans) < -10 and max(pans) > 45, "the wave did not turn toward both ends"
    assert near < 0.10, f"the wave never came near the box ({near:.2f} m) - the test proves nothing"


@so101
def test_the_wave_keeps_clear_of_any_body_in_the_scene():
    """A body in the twin's scene (here a column standing in for the second arm) is kept `clearance` away from
    every geom of the arm and the card, along every move - with no keep-out box for it."""
    import mujoco

    from machinekind_twin import scene as sc
    from machinekind_twin.scene import PREFIX

    column = sc.Box("drugie_ramie", (0.05, 0.05, 0.15), (0.12, 0.20), free=False)
    clearance = 0.02
    s, check = _blind_session(seed=1, objects=[column], clearance=clearance)
    try:
        for _ in range(60):
            s.step()
        m, d = check.model, check.data
        other = m.geom("drugie_ramie").id
        base = m.body(PREFIX + check.cfg.robot.base_body).id
        arm = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
               and _under(m, m.geom_bodyid[g], base)]
        nearest = np.inf
        for a, b in s.moves:
            for j in _along(s, a, b, step=0.02):
                check.kin.write_qpos(d.qpos, check.kin.to_q(j))
                mujoco.mj_fwdPosition(m, d)
                nearest = min(nearest, min(mujoco.mj_geomDistance(m, d, g, other, 0.2, None) for g in arm))
        # The wave checks every 0.02 rad of a move; in between the arm may come a few mm nearer.
        assert clearance - 0.008 <= nearest < 0.08, nearest
    finally:
        check.close()


def _under(m, body, root) -> bool:
    while body > 0 and body != root:
        body = m.body_parentid[body]
    return body == root


@so101
def test_the_collision_checker_tells_too_near_from_clear():
    import mujoco

    from machinekind_twin import scene as sc
    from machinekind_twin.collision import CollisionChecker
    from machinekind_twin.robots import get_spec
    from machinekind_twin.scene import PREFIX

    spec = get_spec("so101")
    # Clear of the arm at rest: a pair touching at rest stays allowed.
    scene = sc.build(sc.SceneConfig(spec, objects=[sc.Box("slup", (0.04, 0.04, 0.12), (0.2, 0.2), free=False)]))
    checker = CollisionChecker(scene, clearance=0.02)
    try:
        m = checker.model
        other = m.geom("slup").id
        base = m.body(PREFIX + spec.base_body).id
        arm = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and _under(m, m.geom_bodyid[g], base)]
        m0, d0 = scene.model, scene.data                     # the real sizes: the checker's copy is grown
        rng = np.random.default_rng(0)
        near = clear = 0
        for _ in range(3000):
            j = dict(spec.home, **{n: float(rng.uniform(lo, hi)) for n, (lo, hi) in spec.wave_ranges.items()})
            checker.kin.write_qpos(d0.qpos, checker.kin.to_q(j))
            mujoco.mj_fwdPosition(m0, d0)
            gaps = {g: mujoco.mj_geomDistance(m0, d0, g, other, 0.2, None) for g in arm}
            seen = {pair: dist for pair, dist in checker._contacts(j).items() if other in pair}
            for g, gap in gaps.items():
                pair = (min(g, other), max(g, other))
                # Nearer than the clearance: reported, never farther than it is (a grown box's corners
                # reach further than the clearance, so a pair near one is reported nearer).
                if 0.0 < gap < 0.017:
                    assert pair in seen and seen[pair] <= gap + 0.004, (gap, seen)
                    near += 1
                elif gap > 0.036:
                    assert pair not in seen
            if min(gaps.values()) > 0.036 and checker.config_clear(j) is None:
                clear += 1
        assert near > 20 and clear > 20
        def only_near_the_column(j):
            found = checker._contacts(j)
            return found and all(other in p and 0 < dist < 0.015 for p, dist in found.items())

        j = next(j for j in (dict(spec.home, **{n: float(rng.uniform(lo, hi)) for n, (lo, hi) in spec.wave_ranges.items()})
                             for _ in range(5000)) if only_near_the_column(j))
        assert (checker.config_clear(j) or "").startswith("za blisko") and "slup" in checker.config_clear(j)
    finally:
        scene.close()


def _stand_views(scale: float = 1.0):
    """The stand's cameras: the C920 about 0.95 m above the table in front of the arm looking down, and a
    second one low from the front-right; the real lenses' K."""
    from machinekind_twin import scene as sc
    from machinekind_twin.calib.simulate import look_at

    K1 = np.array([[931.2, 0, 617.5], [0, 933.4, 375.2], [0, 0, 1]])
    K2 = np.array([[1007.3, 0, 640.0], [0, 1033.5, 360.0], [0, 0, 1]])
    S = np.diag([scale, scale, 1.0])
    eyes = (("kamera1", (0.80, 0.05, 0.95), (0.22, 0.0, 0.0), K1), ("kamera2", (0.65, -0.60, 0.35), (0.2, 0.0, 0.08), K2))
    return [sc.CameraView(n, S @ K, int(1280 * scale), int(720 * scale), look_at(e, t)) for n, e, t, K in eyes]


@so101
@pytest.mark.render
def test_the_wave_finds_the_stand_cameras_and_the_servo_zeros_around_the_second_arm():
    """The stand that failed: both cameras trusted, the servo zero offsets found, and the second arm's box
    never entered by the card or the gripper on the way."""
    from machinekind_twin import scene as sc
    from machinekind_twin.calib import simulate
    from machinekind_twin.calib.card import Card, pinch_point
    from machinekind_twin.calib.session import KeepOut, WaveConfig
    from machinekind_twin.kinematics import RobotKinematics
    from machinekind_twin.robots import get_spec

    spec = get_spec("so101")
    box = KeepOut(**SECOND_ARM)
    cfg = replace(WaveConfig.for_spec(spec), keepout=(box,))
    card = Card.sized(0.1, 0.125)
    zeros = {"shoulder_lift": 1.8, "elbow_flex": -2.2, "wrist_flex": 2.5}
    res = simulate.run(3, 0, spec, card=card, cfg=cfg, views=_stand_views(), servo_offsets=zeros)
    assert all(res.trusted.values()), res.reasons
    assert max(res.err_mm.values()) < 5.0 and max(res.err_deg.values()) < 0.5, res.err_mm
    assert res.joint_offsets == pytest.approx(zeros, abs=0.3), res.offsets_note
    check = sc.build(sc.SceneConfig(spec, card=card, card_pose=card.nominal(pinch_point(RobotKinematics(spec)), spec),
                                    card_collider=0.012))
    try:
        s = type("S", (), {"kin": check.kin})()
        for a, b in res.moves:
            for j in _along(s, a, b, step=0.02):
                assert not box.hit(_gripper_and_card(check, j))
    finally:
        check.close()


@so101
@pytest.mark.render
def test_the_wave_finds_cameras_behind_beside_high_and_far():
    """Cameras no search spot covered: behind the arm, beside it, and one a metre above the table 1.3 m
    in front of it (the C920 hangs higher than the old search's 0.5 m)."""
    from machinekind_twin import scene as sc
    from machinekind_twin.calib import simulate
    from machinekind_twin.calib.card import Card
    from machinekind_twin.calib.simulate import look_at
    from machinekind_twin.robots import get_spec

    K = np.array([[931.2, 0, 640.0], [0, 933.4, 360.0], [0, 0, 1]])
    eyes = (("tyl", (-0.45, 0.35, 0.55), (0.2, 0.0, 0.05)), ("bok", (0.15, -0.75, 0.30), (0.2, 0.0, 0.08)),
            ("wysoko", (1.55, 0.3, 1.0), (0.25, 0.0, 0.1)))
    views = [sc.CameraView(n, K, 1280, 720, look_at(e, t)) for n, e, t in eyes]
    res = simulate.run(5, 0, get_spec("so101"), card=Card.sized(0.1, 0.125), views=views)
    assert all(res.trusted.values()), res.reasons
    assert max(res.err_mm.values()) < 5.0, res.err_mm


def test_a_wave_that_cannot_solve_says_which_camera_saw_the_card_how_often_and_from_where(tmp_path):
    """The stand's record had an empty `blad` after 90 poses and nothing saved. Now the error names every
    camera with what it saw, and the record carries the map and a reason in `blad`."""
    from machinekind_twin.calib.handeye import CameraFit, Fit
    from machinekind_twin.calib.session import Sighting, WaveConfig
    from machinekind_twin.robots import get_spec
    from machinekind_twin.ui import jobs
    from machinekind_twin.workspace import Workspace

    cfg = replace(WaveConfig.for_spec(get_spec(BOT)), max_poses=8)
    s, check = _blind_session(BOT, cfg=cfg, cameras=("lewa", "prawa"))
    try:
        with pytest.raises(RuntimeError, match="zadna kamera nie zobaczyla karty w 8 pozach") as err:
            s.run()
        assert "lewa: nie widziala karty ani razu w 8 pozach" in str(err.value) and "prawa:" in str(err.value)
        p = s.progress["prawa"]
        p.sightings = [Sighting(k, np.eye(4), 0, np.zeros((4, 2)), {}) for k in (2, 5)]
        p.poses = 2
        p.seen_dirs.update({"przod skosnie w gore": 2})
        p.seen_places.update({"podstawa -15..+15 st., daleko, nisko": 2})
        text = s.explain("prawa")
        assert "widziala karte 2 razy w 2 z 8 poz" in text and "przod skosnie w gore (2)" in text
        assert "za malo detekcji" in text

        ws = Workspace(robot=BOT, path=tmp_path / "twin.json")
        twin = type("Twin", (), {"workspace": ws})()
        fit = Fit({"prawa": CameraFit(np.eye(4), 9.0, 12.0, 4, 2.0, trusted=False,
                                      reason="tylko 4 obserwacji, potrzeba 30")})
        jobs._dump_wave(twin, s, fit)
        rec = json.loads((tmp_path / "ostatnia_fala.json").read_text())
        assert "prawa: tylko 4 obserwacji" in rec["blad"] and "lewa: nie widziala karty" in rec["blad"]
        assert rec["mapa"]["kamery"]["prawa"]["widziala"] == 2 and rec["mapa"]["pozy"] == 8
        assert rec["mapa"]["pokazane"]["kierunki"] and rec["pozy"][0]["faza"] == "szukanie"
        jobs._dump_wave(twin, s, None, error="zadna kamera nie zobaczyla karty")
        rec = json.loads((tmp_path / "ostatnia_fala.json").read_text())
        assert rec["blad"] == "zadna kamera nie zobaczyla karty" and "lewa" in rec["mapa"]["kamery"]
    finally:
        check.close()


def test_the_stand_keeps_its_keepout_boxes_and_turn_range_in_twin_json(tmp_path):
    from machinekind_twin.calib.session import KeepOut, WaveConfig
    from machinekind_twin.robots import get_spec
    from machinekind_twin.workspace import Workspace

    ws = Workspace(robot=BOT, path=tmp_path / "twin.json")
    assert ws.to_dict()["wave"] == {"keepout": [], "base_turn": None}
    data = ws.to_dict()
    data["wave"] = {"keepout": [{"name": "drugie ramie", "x": [-0.12, 0.2], "y": [0.09, 0.3], "z": [0, 0.4]}],
                    "base_turn": [-60, 45]}
    (tmp_path / "twin.json").write_text(json.dumps(data))
    again = Workspace.load(tmp_path / "twin.json")
    assert again.wave.keepout == (KeepOut((-0.12, 0.2), (0.09, 0.3), (0.0, 0.4), "drugie ramie"),)
    assert again.wave.base_turn == (-60.0, 45.0)
    assert "drugie ramie (x -0.12..+0.20" in again.wave.describe()
    cfg = WaveConfig.for_spec(get_spec(BOT), again.wave)
    assert cfg.keepout == again.wave.keepout and cfg.base_turn == (-60.0, 45.0)
    again.save()
    assert json.loads((tmp_path / "twin.json").read_text())["wave"] == data["wave"] | {
        "keepout": [{"name": "drugie ramie", "x": [-0.12, 0.2], "y": [0.09, 0.3], "z": [0.0, 0.4]}],
        "base_turn": [-60.0, 45.0]}
    for bad in ({"keepout": [{"x": [0.2, 0.1], "y": [0, 1], "z": [0, 1]}]}, {"base_turn": [30, -30]},
                {"keepout": [{"x": [0, 1], "y": [0, 1]}]}, {"strefy": []}):
        (tmp_path / "twin.json").write_text(json.dumps(data | {"wave": bad}))
        with pytest.raises(ValueError):
            Workspace.load(tmp_path / "twin.json")
