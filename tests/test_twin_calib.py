"""Kalibracja kamer z karty w chwytaku: planowanie fali, zestawy polozen, dopasowanie.

Pelnej sesji w symulacji tu nie ma: robot testowy (dwa stawy, bez nadgarstka) nie obraca
karty, wiec fala nie zbiera roznych katow.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from conftest import BOT  # noqa: E402


# ------------------------------------------------------------ planowanie fali
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


def _session(spec_name: str, seed: int, cameras=("c0", "c1")):
    import numpy as np
    from lerobot_mp.twin import scene as sc
    from lerobot_mp.twin.calib.card import Card, pinch_point
    from lerobot_mp.twin.calib.session import Session, WaveConfig
    from lerobot_mp.twin.collision import CollisionChecker
    from lerobot_mp.twin.kinematics import RobotKinematics
    from lerobot_mp.twin.robots import get_spec

    spec = get_spec(spec_name)
    card = Card(tag_size=0.037)
    nominal = card.nominal(pinch_point(RobotKinematics(spec)), spec)
    world = sc.build(sc.SceneConfig(spec, card=card, card_pose=nominal, card_collider=0.012))
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    home = dict(spec.home)
    if spec.gripper:
        home[spec.gripper] = 0.0
    s = Session(_StillRobot(home), _BlindCameras(), {c: (K, None) for c in cameras}, world.kin, card, nominal,
                CollisionChecker(world), WaveConfig.for_spec(spec), seed=seed)
    return s, world


def test_wave_search_sweeps_both_sides_of_the_table_in_every_run():
    """Na stanowisku A1X fala zwiedzila tylko prawa strone stolu: szukanie bylo losowym
    spacerem w stawach z limitem przejazdu, a nie zamiataniem. Teraz kazde 16 poz bez
    detekcji odwiedza wszystkie sektory stawu podstawy, po obu stronach."""
    import numpy as np

    spec_name = BOT
    for seed in range(6):
        s, world = _session(spec_name, seed)
        try:
            pan = s._pan
            lo, hi = s.kin.spec.wave_ranges[pan]
            edges = np.linspace(lo, hi, 6)[1:-1]
            j = np.array([s.step().joints[pan] for _ in range(16)])
            sectors = set(np.digitize(j, edges).tolist())
            assert sectors == {0, 1, 2, 3, 4}, f"{spec_name} ziarno {seed}: sektory {sorted(sectors)}, {j.round()}"
            assert 0.3 <= np.mean(j < (lo + hi) / 2) <= 0.7
        finally:
            world.close()


def test_wave_aims_at_a_camera_only_after_two_detections_and_never_goes_home_for_it():
    """Jedna detekcja malego taga z daleka dawala lustrzana poze kamery, w ktora celowala
    co druga poza; gdy kadru nie dalo sie trafic, fala wracala do domu (stracony przejazd).
    Kamera z jedna detekcja nie jest namierzona; namierzona bez osiagalnego kadru dostaje
    poze z zamiatania, nie dom."""
    import numpy as np

    s, world = _session(BOT, 3)
    try:
        p = s.progress["c0"]
        T = np.eye(4)
        T[:3, 3] = [0.9, 0.0, 0.5]
        p.estimates.append(T)
        assert not s._located(p) and s._target_camera() is None
        p.estimates.append(T)
        assert s._located(p) and s._target_camera() == "c0"
        # Kamera "namierzona" 30 m nad stolem: zaden kadr nie do trafienia - poza mimo to jest.
        p.estimates[:] = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 30.0], [0, 0, 0, 1.0]])] * 2
        s.index = 2                                       # parzysta = celowanie, nie zamiatanie
        rep = s.step()
        assert rep.note == "" and s._far_from(rep.joints, s._home)
    finally:
        world.close()


# ----------------------------------------------------------- zestawy polozen
def test_step2_results_are_kept_as_presets_and_newest_loads_by_default(tmp_path):
    """Kazdy zapis kroku 2 to nowy plik; `load` bierze najnowszy, wybrany starszy zostaje."""
    from lerobot_mp.twin.workspace import CameraRecord, Workspace

    def pose(x):
        return [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1]]

    ws = Workspace(robot=BOT, path=tmp_path / "twin.json")
    ws.add_camera(CameraRecord("front", T_cam2base=pose(0.1),
                               calibration={"trusted": True, "time": "2026-09-26T10:00:00"}))
    ws.save()                                           # stare stanowisko: jego poza staje sie pierwszym zestawem
    assert ws.presets() == ["20260926-100000"] and ws.calib_preset == "20260926-100000"

    ws.camera("front").T_cam2base = pose(0.2)
    ws.camera("front").calibration = {"trusted": True, "time": "2026-09-26T11:00:00"}
    ws.save_preset()
    ws.save_preset()                                    # ten sam wynik drugi raz - bez nowego pliku
    ws.save()
    assert ws.presets() == ["20260926-110000", "20260926-100000"]
    assert (tmp_path / "calib" / "20260926-100000.json").exists(), "starszy zapis nie znika"

    again = Workspace.load(tmp_path / "twin.json")
    assert again.calib_preset == "20260926-110000" and again.camera("front").T_cam2base[0][3] == 0.2
    assert again.load_preset("20260926-100000") == ["front"]
    assert again.camera("front").T_cam2base[0][3] == 0.1
    again.save()
    assert Workspace.load(tmp_path / "twin.json").calib_preset == "20260926-100000", "wybor starszego zostaje"

    (tmp_path / "twin.json").write_text(
        (tmp_path / "twin.json").read_text().replace('"calib_preset": "20260926-100000"', '"calib_preset": null'))
    assert Workspace.load(tmp_path / "twin.json").camera("front").T_cam2base[0][3] == 0.2, "bez wyboru - najnowszy"


def test_a_move_that_does_not_arrive_costs_one_pose_not_the_session():
    """Fala z 21 i 11 obserwacjami padla na jednym "ruch nie dojechal do celu" (nadgarstek pod
    ograniczeniem predkosci). Taki przejazd to pominieta poza; dopiero trzy z rzedu koncza sesje."""
    s, world = _session(BOT, 5)
    try:
        robot = s.robot
        fails = {"left": 0}
        real_move = robot.move

        def move(joints, duration):
            if fails["left"] > 0:
                fails["left"] -= 1
                raise RuntimeError("ruch nie dojechal do celu w 1.9 s: rozkaz jeszcze jedzie (j2)")
            real_move(joints, duration)

        robot.move = move
        s.step()                                          # do domu i pierwsza poza - bez bledu
        fails["left"] = 1
        rep = s.step()
        assert rep.note.startswith("przejazd pominiety") and rep.index == 2
        rep = s.step()
        assert rep.note == "" and s._skipped == 0
        fails["left"] = 3
        with pytest.raises(RuntimeError, match="trzy przejazdy"):
            for _ in range(3):
                s.step()
        with pytest.raises(RuntimeError, match="odebrane"):
            robot.move = lambda j, d: (_ for _ in ()).throw(RuntimeError("fala przerwana - ramie odebrane"))
            s.step()
    finally:
        world.close()


def test_solver_drops_outlier_frames_and_reports_them():
    """Fala z dobrymi kadrami i kilkoma zlapanymi w ruchu wychodzila 6,9 px dla calej kamery.
    Odstajace obserwacje (3 x mediana) odpadaja przed koncowym dopasowaniem, a werdykt mowi ile."""
    import cv2
    import numpy as np
    from lerobot_mp.twin.calib.handeye import Observation, judge, solve
    from lerobot_mp.twin.calib.tags import corners_in_tag

    rng = np.random.default_rng(3)
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    tag = 0.037
    T_cam = np.eye(4)
    T_cam[:3, :3], _ = cv2.Rodrigues(np.array([2.2, 0.0, 0.0]))       # patrzy z gory-przodu
    T_cam[:3, 3] = [0.5, 0.0, 0.6]
    T_mount = np.eye(4)
    T_mount[:3, 3] = [0.03, 0.0, 0.0]
    X = np.c_[corners_in_tag(tag), np.ones(4)].T
    obs = []
    for i in range(24):
        T = np.eye(4)
        T[:3, :3], _ = cv2.Rodrigues(rng.uniform(-0.8, 0.8, 3))
        T[:3, 3] = [0.35 + rng.uniform(-0.15, 0.15), rng.uniform(-0.25, 0.25), rng.uniform(0.1, 0.35)]
        pts = (np.linalg.inv(T_cam) @ T @ T_mount @ X)[:3].T
        px, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, np.zeros(5))
        px = px.reshape(4, 2) + rng.normal(0, 0.3, (4, 2))
        if i in (5, 11, 17):
            px += np.array([25.0, -18.0])                              # kadr zlapany w ruchu
        obs.append(Observation(T, np.eye(4), px, 0, "card", "cam", pose=i))
    nominal = T_mount.copy()
    nominal[:3, 3] += [0.01, -0.005, 0.004]
    fit = judge(solve(obs, {"cam": (K, None)}, tag, {"card": nominal}))
    cam = fit.cameras["cam"]
    assert cam.dropped == 3 and cam.n_obs == 21 and cam.rms_px < 1.0, (cam.dropped, cam.rms_px, cam.reason)
    assert cam.trusted and "odrzucono 3" in cam.reason
    assert np.linalg.norm(cam.T_cam2base[:3, 3] - T_cam[:3, 3]) < 0.01
    assert len(cam.per_obs_px) == 21 and all(p not in (5, 11, 17) for p, _ in cam.obs_ids)


def test_wave_uses_only_frames_captured_after_the_arm_stopped():
    """Kadr do obserwacji byl najnowszym kadrem do 1 s wstecz - nierzadko z karta jeszcze w ruchu.
    Teraz kamera musi oddac kadr zlapany po zatrzymaniu (+ margines); bez niego jest pomijana."""
    import time
    import numpy as np
    from lerobot_mp.twin.calib import session as sm

    t0 = time.monotonic()
    img = np.zeros((8, 8, 3), np.uint8)

    class Cams:
        def __init__(self):
            self.calls = 0

        def grab(self):
            raise AssertionError("z grab_t nie wolno brac starych kadrow")

        def grab_t(self):
            self.calls += 1
            fresh = time.monotonic() if self.calls >= 3 else t0 - 0.5
            return {"a": (img, fresh), "b": (img, t0 - 0.5)}        # b nigdy nie daje swiezego

    cams = Cams()
    sm.FRESH_TIMEOUT_S, old = 0.4, sm.FRESH_TIMEOUT_S
    try:
        out = sm.fresh_frames(cams, t0)
    finally:
        sm.FRESH_TIMEOUT_S = old
    assert set(out) == {"a"} and cams.calls >= 3

    class Plain:
        def grab(self):
            return {"a": img}

    assert set(sm.fresh_frames(Plain(), t0)) == {"a"}                 # symulacja: bez znacznikow czasu
