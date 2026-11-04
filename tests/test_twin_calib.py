"""Kalibracja kamer z karty w chwytaku: planowanie fali, zestawy polozen, dopasowanie.

Pelnej sesji w symulacji tu nie ma: robot testowy (dwa stawy, bez nadgarstka) nie obraca
karty, wiec fala nie zbiera roznych katow.
"""

from __future__ import annotations

import importlib.util

import pytest

pytest.importorskip("mujoco")

from conftest import BOT  # noqa: E402


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


#: The wave tests come from the A1X station (six joints, a base that must sweep both sides of
#: the table); with galaxeo installed next to the twin they run on the A1X too.
WAVE_ROBOTS = [
    BOT,
    pytest.param("arm6", marks=pytest.mark.skipif(not _installed("arm6"),
                                                  reason="ramie przykladowe instaluje uv sync")),
    pytest.param("a1x", marks=pytest.mark.skipif(not _installed("galaxeo.robot"),
                                                 reason="A1X przychodzi z pakietu galaxeo")),
    pytest.param("so101", marks=pytest.mark.skipif(not _installed("soarm"), reason="SO-101 przychodzi z pakietu soarm")),
]


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


def _session(spec_name: str, seed: int, cameras=("c0", "c1"), cfg=None, objects=(), clearance=0.0):
    import numpy as np

    from machinekind_twin import scene as sc
    from machinekind_twin.calib.card import Card, pinch_point
    from machinekind_twin.calib.session import Session, WaveConfig
    from machinekind_twin.collision import CollisionChecker
    from machinekind_twin.kinematics import RobotKinematics
    from machinekind_twin.robots import get_spec

    spec = get_spec(spec_name)
    card = Card(tag_size=0.037)
    nominal = card.nominal(pinch_point(RobotKinematics(spec)), spec)
    world = sc.build(sc.SceneConfig(spec, card=card, card_pose=nominal, card_collider=0.012, objects=list(objects)))
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    home = dict(spec.home)
    if spec.gripper:
        home[spec.gripper] = 0.0
    s = Session(_StillRobot(home), _BlindCameras(), {c: (K, None) for c in cameras}, world.kin, card, nominal,
                CollisionChecker(world, clearance), cfg or WaveConfig.for_spec(spec), seed=seed)
    return s, world


@pytest.mark.parametrize("spec_name", WAVE_ROBOTS)
def test_wave_explores_the_whole_turn_and_turns_the_card_every_way(spec_name):
    """Replaces the sweep of `wave_ranges` in five sectors toward where cameras "usually stand": on the SO-101
    stand that search missed a camera looking down from 0.9 m. With no camera seeing anything, 30 poses
    sweep every sector of the base's turn (the description's range widened, both sides) and, with a
    roll joint, turn the card's face to every exploration direction."""
    import numpy as np

    from machinekind_twin.calib.session import DIRECTIONS

    for seed in range(3):
        s, world = _session(spec_name, seed)
        try:
            reps = [s.step() for _ in range(30)]
            lo, hi = s.kin.spec.wave_ranges[s._pan]
            assert s._ranges[s._pan][0] < lo and s._ranges[s._pan][1] > hi
            pan = np.array([r.joints[s._pan] for r in reps])
            assert {s._sector(v) for v in pan} == set(range(len(s._sectors))), f"{spec_name} {seed}: {pan.round()}"
            assert all(r.phase == "szukanie" and r.aim is None for r in reps)
            if s._roll:
                assert (s.visits.sum(axis=0) > 0).all() and len(DIRECTIONS) == s.visits.shape[1]
        finally:
            world.close()


def _sightings(s, T_cam, K, joints_list, mount=None):
    """Tags a camera at `T_cam` (cam2base) sees at these poses, as the session records them; `mount` is where
    the card truly sits (default: nominal)."""
    import numpy as np

    from machinekind_twin.calib.session import Sighting
    from machinekind_twin.calib.tags import corners_in_tag, project

    mount = s.card_nominal if mount is None else mount
    X = np.c_[corners_in_tag(s.card.tag_size), np.ones(4)].T
    out = []
    for k, j in enumerate(joints_list):
        T = s.kin.tcp(j)
        for tid, T_tag in s.tags.items():
            P = (np.linalg.inv(T_cam) @ T @ mount @ T_tag @ X)[:3].T
            normal = (T @ mount @ T_tag)[:3, 2]
            if (P[:, 2] < 0.1).any() or normal @ (T_cam[:3, 3] - (T @ mount @ T_tag)[:3, 3]) <= 0:
                continue
            px = project(P, K)
            if ((px < 0) | (px > [1280, 720])).any():
                continue
            out.append(Sighting(100 + k, T, tid, px, dict(j)))
    return out


def _poses_seen_by(s, T_cam, K, n, seed=0, mount=None):
    """`n` poses from the wave's own sampling at which the camera sees a tag, with those sightings."""
    import numpy as np

    rng, found = np.random.default_rng(seed), []
    joints = dict(s._home)
    while len(found) < n:
        j = s._sample(joints)
        for name in s._ranges:
            j[name] = float(rng.uniform(*s._ranges[name]))
        if s._roll:
            j[s._roll] = float(rng.uniform(-150, 150))
        seen = _sightings(s, T_cam, K, [j], mount)
        if seen:
            for x in seen:
                x.pose = 100 + len(found)
            found.append(seen[0])
    return found


@pytest.mark.parametrize("spec_name", [p for p in WAVE_ROBOTS if p != BOT])
def test_wave_aims_only_where_several_sightings_agree_and_drops_what_later_disagrees(spec_name):
    """Replaces "aims after two detections": two sightings of a small tag from far away gave a mirrored pose and the
    wave aimed there for 80 poses. Now a camera is aimed at only when `locate_min` sightings from different poses
    agree by reprojection (both solutions of each PnP tried); a mixed bag does not locate it, later sightings at
    odds with the pose, or aimed poses that keep missing with little support, drop it. A camera located where no
    frame can be reached gets exploration, not the home pose."""
    import numpy as np

    from machinekind_twin.calib.card import perturb
    from machinekind_twin.calib.simulate import look_at

    s, world = _session(spec_name, 3, cameras=("c0",))
    try:
        K = s.intrinsics["c0"][0]
        cam = look_at((0.75, 0.1, 0.9), (0.22, 0.0, 0.0))
        p = s.progress["c0"]
        # A card in the jaws up to 10 deg and 12 mm off nominal: still agreeing.
        mount = perturb(s.card_nominal, np.random.default_rng(1))
        seen = _poses_seen_by(s, cam, K, 6, mount=mount)
        p.sightings[:] = seen[:2]
        assert s._locate("c0") is None                                    # two are not enough
        p.sightings[:] = seen[:3]
        found = s._locate("c0")
        assert found is not None and np.linalg.norm(found[0][:3, 3] - cam[:3, 3]) < 0.15
        # Three sightings, each from a camera somewhere else (what mirrored poses look like): no consensus.
        others = [look_at(e, (0.22, 0.0, 0.0)) for e in ((0.75, 0.1, 0.9), (0.2, -0.7, 0.4), (-0.3, 0.6, 0.6))]
        mixed = [_poses_seen_by(s, T, K, 1, seed=k)[0] for k, T in enumerate(others)]
        for k, x in enumerate(mixed):
            x.pose = 200 + k
        p.sightings[:] = mixed
        assert s._locate("c0") is None
        # Located from good sightings, then three that a camera elsewhere would have seen: dropped.
        p.sightings[:] = list(seen)
        s._adopt("c0", s._locate("c0"))
        assert p.rough is not None
        elsewhere = _poses_seen_by(s, others[1], K, 3, seed=5)
        n0 = len(p.sightings)
        p.sightings.extend(elsewhere)
        notes = s._weigh({"c0": list(range(n0, len(p.sightings)))}, None, {"c0"})
        assert p.rough is None and p.drops == 1 and any("odrzucone" in n for n in notes)
        # Aimed poses that miss: a pose few sightings back is dropped, one many back is kept.
        p.sightings[:] = list(seen[:3])
        p.drops = 0
        s._adopt("c0", s._locate("c0"))
        for _ in range(s.cfg.max_misses):
            s._weigh({}, "c0", {"c0"})
        assert p.rough is None
        p.sightings[:] = list(seen) + _poses_seen_by(s, cam, K, 6, seed=7, mount=mount)
        p.drops = 0
        s._adopt("c0", s._locate("c0"))
        assert len(p.inliers) >= 2 * s.cfg.locate_min
        for _ in range(s.cfg.max_misses + 2):
            s._weigh({}, "c0", {"c0"})
        assert p.rough is not None
        # Located 30 m above the table: no frame to aim at - an exploration pose, not home.
        p.rough = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 30.0], [0, 0, 0, 1.0]])
        s.explored, s.index, p.misses = s.cfg.explore_min, 2, 0
        rep = s.step()
        assert rep.phase == "szukanie" and "wracam" not in rep.note and s._far_from(rep.joints, s._home)
    finally:
        world.close()


# ----------------------------------------------------------- zestawy polozen
def test_step2_results_are_kept_as_presets_and_newest_loads_by_default(tmp_path):
    """Kazdy zapis kroku 2 to nowy plik; `load` bierze najnowszy, wybrany starszy zostaje."""
    from machinekind_twin.workspace import CameraRecord, Workspace

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


@pytest.mark.parametrize("spec_name", WAVE_ROBOTS)
def test_a_move_that_does_not_arrive_costs_one_pose_not_the_session(spec_name):
    """Fala z 21 i 11 obserwacjami padla na jednym "ruch nie dojechal do celu" (nadgarstek pod
    ograniczeniem predkosci). Taki przejazd to pominieta poza; dopiero trzy z rzedu koncza sesje."""
    s, world = _session(spec_name, 5)
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


def test_card_size_follows_the_tag_and_how_far_it_sticks_out():
    from machinekind_twin.calib.card import Card

    small, default = Card.sized(0.050, 0.070), Card()
    assert (small.width, small.length) == (pytest.approx(default.width), pytest.approx(default.length))
    big = Card.sized(0.100, 0.110)
    assert big.width == pytest.approx(0.100 / 0.8 + 0.0075) and big.length == pytest.approx(0.150)
    assert big.white_margin == pytest.approx(0.005)


def test_solver_drops_outlier_frames_and_reports_them():
    """Fala z dobrymi kadrami i kilkoma zlapanymi w ruchu wychodzila 6,9 px dla calej kamery.
    Odstajace obserwacje (3 x mediana) odpadaja przed koncowym dopasowaniem, a werdykt mowi ile."""
    import cv2
    import numpy as np

    from machinekind_twin.calib.handeye import Observation, judge, solve
    from machinekind_twin.calib.tags import corners_in_tag

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


def test_wave_uses_only_frames_taken_after_the_settle_with_the_card_standing_still():
    """Was: the newest frame up to 1 s back, then a frame 0.15 s after the stop - the SO-101 still swayed and
    gave 8-16 px residuals (2026-11-03). Now a frame counts `settle` s after the stop, and a camera's tags only
    when two consecutive new frames agree within `still_px`; a camera whose tag keeps moving gives none, one
    without a new frame is left out."""
    import time

    import numpy as np

    from machinekind_twin.calib import session as sm

    t0 = time.monotonic()
    img = np.zeros((8, 8, 3), np.uint8)
    corners = np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]])
    # Tag shift [px] in each new frame: camera a settles after three frames, camera c keeps swaying.
    sway = {0: [2.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0], 100: [3.0, -3.0] * 30}

    class Cams:
        def __init__(self):
            self.n = {"a": 0, "c": 0}

        def grab(self):
            raise AssertionError("z grab_t nie wolno brac starych kadrow")

        def grab_t(self):
            out = {"b": (img, t0 - 0.5)}                                 # b nigdy nie daje swiezego
            for name, base in (("a", 0), ("c", 100)):
                self.n[name] += 1
                out[name] = (np.full((8, 8, 3), base + min(self.n[name], 60), np.uint8), time.monotonic())
            return out

    def tags_in(frame, ids):
        v = int(frame[0, 0, 0])
        base = 100 if v >= 100 else 0
        return {0: corners + sum(sway[base][:v - base])}

    orig = sm._tags_in
    sm._tags_in = tags_in
    try:
        frames, found, moving = sm.still_frames(Cams(), t0, 0.05, 0.3, 0.4, {0: None})
    finally:
        sm._tags_in = orig
    assert time.monotonic() - t0 >= 0.05                                  # the settle went by first
    assert set(frames) == {"a", "c"} and "b" not in found
    assert set(found["a"]) == {0} and moving == {"c"} and found["c"] == {}

    class Plain:
        def grab(self):
            return {"a": img}

    frames, found, moving = sm.still_frames(Plain(), t0, 0.8, 0.3, 2.5, {0: None})  # symulacja: bez czasow
    assert set(frames) == {"a"} and found == {"a": {}} and not moving
