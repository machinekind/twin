"""Kalibracja kamer z karty w chwytaku: detekcja, solver, karta, pelna sesja.

Solver sprawdzamy na obserwacjach syntetycznych (szybko, bez renderu), a cala
sesje - ramie, fala, render, detekcja, dopasowanie, bramki - w symulacji,
wzgledem prawdziwych poz kamer i karty, ktorych sesja nie widzi.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.calib.card import Card, perturb, pinch_point  # noqa: E402
from lerobot_mp.twin.calib.handeye import (  # noqa: E402
    Observation,
    _median_T,
    judge,
    pose_error,
    rotation_spread_R,
    solve,
)
from lerobot_mp.twin.calib.simulate import run, sample_cameras  # noqa: E402
from lerobot_mp.twin.calib.tags import corners_in_tag, detect, project, render  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics, inverse  # noqa: E402
from lerobot_mp.twin.robots import SO101  # noqa: E402


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics(SO101)


def synthetic(kin, cameras, truth, card, rng, n_poses=48, noise=0.2, only_roll=False):
    """Obserwacje takie, jakie dalaby prawdziwa fala: rogi taga rzutowane przez K z szumem."""
    obs = []
    X = np.c_[corners_in_tag(card.tag_size), np.ones(4)]
    for _ in range(n_poses):
        joints = dict(SO101.home, gripper=0.0)
        if only_roll:
            joints["wrist_roll"] = float(rng.uniform(-150, 150))
        else:
            for name, (lo, hi) in SO101.wave_ranges.items():
                joints[name] = float(rng.uniform(lo, hi))
            joints["wrist_roll"] = float(rng.uniform(-150, 150))
        T_tcp = kin.tcp(joints)
        for view in cameras:
            for tid, T_tag in card.tag_poses().items():
                T = inverse(view.T_cam2base) @ T_tcp @ truth @ T_tag
                if T[2, 3] <= 0.05 or T[:3, 2] @ -T[:3, 3] <= 0.0:     # za kamera albo tylem
                    continue
                px = project((T @ X.T).T[:, :3], view.K)
                if not ((px >= 0) & (px < [view.width, view.height])).all():
                    continue
                obs.append(Observation(T_tcp, T_tag, px + rng.normal(0, noise, px.shape), tid,
                                       "card", view.name))
    return obs


# ----------------------------------------------------------------- tagi
def test_detected_corners_follow_the_pinhole_pixel_convention():
    """Krawedzie czarnego kwadratu (piksele 140..459) leza na 139,5 i 459,5.

    Doszlifowanie APRILTAG oddaje rogi o pol piksela dalej; bez poprawki solver
    kompensowal to obrotem kamery - i blad obrotu w symulacji byl dwa razy wiekszy.
    """
    img = np.full((600, 600), 255, np.uint8)
    img[100:500, 100:500] = render(0, 400)
    corners = detect(img)[0]
    assert np.abs(corners - [[139.5, 139.5], [459.5, 139.5], [459.5, 459.5], [139.5, 459.5]]).max() < 0.05


# ----------------------------------------------------------------- karta
def test_card_faces_are_back_to_back_with_opposite_normals():
    card = Card()
    poses = card.tag_poses()
    front, back = poses[card.ids[0]], poses[card.ids[1]]
    assert np.allclose(front[:2, 3], back[:2, 3])                      # ten sam srodek w plaszczyznie
    assert front[2, 3] - back[2, 3] == pytest.approx(card.thickness)
    assert front[:3, 2] @ back[:3, 2] == pytest.approx(-1.0)


def test_printable_sheet_carries_both_tags():
    card = Card()
    assert sorted(detect(card.sheet(dpi=150))) == sorted(card.ids)


def test_nominal_card_sticks_out_past_the_jaws(kin):
    """Srodek taga ma lezec przed czubkami szczek, na linii ich uchwytu."""
    pinch = pinch_point(kin)
    nominal = Card().nominal(pinch)
    assert nominal[0, 3] > pinch[0] + 0.02
    assert nominal[2, 3] == pytest.approx(pinch[2])


def test_misplacement_stays_within_its_limits():
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = perturb(np.eye(4), rng)
        dt, dr = pose_error(np.eye(4), d)
        assert dt <= np.sqrt(3) * 0.012 + 1e-9
        assert dr <= np.sqrt(3) * np.radians(10.0) + 1e-9


# ----------------------------------------------------------------- solver
@pytest.mark.parametrize("n_cameras", [1, 2])
def test_solver_recovers_cameras_and_card_from_noisy_corners(kin, n_cameras):
    rng = np.random.default_rng(10 + n_cameras)
    card = Card()
    nominal = card.nominal(pinch_point(kin))
    truth = perturb(nominal, rng)
    cameras = sample_cameras(rng, n_cameras)
    obs = synthetic(kin, cameras, truth, card, rng)
    fit = judge(solve(obs, {v.name: (v.K, None) for v in cameras}, card.tag_size, {"card": nominal}))

    for view in cameras:
        dt, dr = pose_error(view.T_cam2base, fit.cameras[view.name].T_cam2base)
        assert dt < 1.0e-3, f"{view.name}: {dt * 1000:.2f} mm"
        assert np.degrees(dr) < 0.15, f"{view.name}: {np.degrees(dr):.3f} st"
        assert fit.cameras[view.name].trusted, fit.cameras[view.name].reason
    ct, cr = pose_error(truth, fit.mounts["card"])
    assert ct < 1.0e-3 and np.degrees(cr) < 0.2


def test_a_wave_that_only_rolls_the_wrist_is_refused(kin):
    """Obrot wokol jednej osi zostawia kamere wolna wzdluz tej osi - dopasowanie
    wychodzi pewne siebie i bledne. Bramka ma to odrzucic z podaniem powodu."""
    rng = np.random.default_rng(3)
    card = Card()
    nominal = card.nominal(pinch_point(kin))
    cameras = sample_cameras(rng, 1)
    obs = synthetic(kin, cameras, perturb(nominal, rng), card, rng, only_roll=True)
    assert rotation_spread_R([o.T_frame2base[:3, :3] for o in obs]) < 1.0
    fit = judge(solve(obs, {cameras[0].name: (cameras[0].K, None)}, card.tag_size, {"card": nominal}))
    cam = fit.cameras[cameras[0].name]
    assert not cam.trusted
    assert "rozrzut" in cam.reason


# ----------------------------------------------------------------- sesja
@pytest.mark.render
def test_full_session_in_simulation_calibrates_the_camera():
    """Cala sesja: fala, render, detekcja, dopasowanie i werdykt - wzgledem prawdy."""
    res = run(seed=0, n_cameras=1)
    cam = next(iter(res.trusted))
    assert res.trusted[cam], res.reasons[cam]
    # Zaufane sesje na ziarnach 0..5 daja 0,3-1,7 mm i 0,006-0,19 st. (srednio ok. 0,5 mm);
    # ziarno 0 jest najgorsze z nich. Prog 1 mm przy jednym ziarnie lamal sie od samej zmiany
    # drogi fali (inny kat szczeki przy chwytaku 0) - stary uklad tez mial 1,15-1,17 mm na
    # ziarnach 8 i 12. Do chwytania 25-milimetrowej kostki 2 mm to wciaz zapas.
    assert res.err_mm[cam] < 2.0
    assert res.err_deg[cam] < 0.25
    assert res.poses < 40, "fala ma sie zmiescic w kilkudziesieciu pozach, nie w maksimum"


@pytest.mark.render
def test_calib_session_converges_on_the_a1x():
    """Ta sama sesja na A1X: fala z `A1X.wave_ranges`, kamery w skali `A1X.desk`.

    Osiagniete na ziarnie 0: 16 poz, 0,20 mm i 0,017 st.
    """
    from lerobot_mp.twin.robots import A1X

    res = run(seed=0, n_cameras=1, spec=A1X)
    for cam, ok in res.trusted.items():
        assert ok, res.reasons[cam]
        assert res.err_mm[cam] < 5.0


@pytest.mark.render
def test_calib_session_trusts_a_side_camera_on_the_a1x():
    """Ziarno 1, dwie kamery: cam1 stoi z boku, obrot ~180 st. wzgledem podstawy.

    Przy medianie po globalnych wektorach Rodriguesa przyblizenia po obu stronach
    zawiniecia +-pi dawaly poze zgrubna odwrocona o ~180 st., fala nie znajdowala
    zadnej pozy w kadrze i stala 90 poz w domu (0/2 zaufanych).
    Osiagniete: 20 poz, 0,54 mm i 0,17 mm.
    """
    from lerobot_mp.twin.robots import A1X

    res = run(seed=1, n_cameras=2, spec=A1X)
    for cam, ok in res.trusted.items():
        assert ok, f"{cam}: {res.reasons[cam]}"
        assert res.err_mm[cam] < 5.0
    assert res.poses < 60


def _rot(axis, deg) -> np.ndarray:
    import cv2

    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.radians(deg) * np.asarray(axis, float) / np.linalg.norm(axis))[0]
    return T


def _angle_deg(A, B) -> float:
    c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_median_transform_is_continuous_across_the_rotation_wrap(n):
    """Obroty 178 i -178 st. (4 st. od siebie) - mediana ma lezec miedzy nimi, nie 180 st. dalej."""
    rng = np.random.default_rng(n)
    axis = np.array([0.2, -0.3, 1.0])
    Ts = [_rot(axis, 178.0 if k % 2 == 0 else -178.0) for k in range(n)]
    for T in Ts:
        T[:3, 3] = rng.normal(0.0, 0.01, 3) + [0.4, 0.1, 0.3]
    M = _median_T(Ts)
    for T in Ts:
        assert _angle_deg(M, T) < 5.0
    assert np.allclose(M[:3, :3] @ M[:3, :3].T, np.eye(3), atol=1e-9)
    assert M[:3, 3] == pytest.approx(np.median([T[:3, 3] for T in Ts], axis=0))


def test_median_transform_ignores_one_outlier_even_when_it_comes_first():
    rng = np.random.default_rng(0)
    axis = np.array([1.0, 1.0, 0.0])
    good = [_rot(axis, 175.0 + d) for d in rng.normal(0.0, 1.0, 6)]
    bad = _rot([0.0, 0.0, 1.0], 60.0)
    M = _median_T([bad, *good])
    assert _angle_deg(M, _rot(axis, 175.0)) < 3.0


def test_card_nominal_is_the_identity_for_so101():
    """Obrot karty z osi opisu - dla SO-101 dokladnie ta sama macierz co wczesniejsza stala."""
    card = Card()
    pinch = np.array([0.0123, -0.0045, 0.0067])
    old = np.eye(4)
    old[:3, 3] = pinch + np.array([card.out / 2, 0.0, 0.0])
    assert np.array_equal(card.nominal(pinch), old)
    assert np.array_equal(card.nominal(pinch, SO101), old)


def test_card_normal_is_the_closing_axis_for_every_robot(spec, toy_spec):
    for s in (spec, toy_spec):
        T = Card().nominal(np.zeros(3), s)
        a = np.asarray(s.tcp_approach, float)
        c = np.asarray(s.tcp_closing, float)
        assert T[:3, 0] == pytest.approx(a / np.linalg.norm(a))
        assert T[:3, 2] == pytest.approx(c / np.linalg.norm(c))
        assert T[:3, :3] @ T[:3, :3].T == pytest.approx(np.eye(3))
        assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
        assert T[:3, 3] == pytest.approx(a / np.linalg.norm(a) * Card().out / 2)


def test_wave_config_for_so101_equals_the_defaults():
    from lerobot_mp.twin.calib.session import WaveConfig

    assert WaveConfig.for_spec(SO101) == WaveConfig()


def test_sample_cameras_for_so101_are_the_default_ones():
    a = sample_cameras(np.random.default_rng(7), 2)
    b = sample_cameras(np.random.default_rng(7), 2, SO101)
    for va, vb in zip(a, b):
        assert np.array_equal(va.K, vb.K) and np.array_equal(va.T_cam2base, vb.T_cam2base)


def test_calib_sim_cli_runs_the_requested_robot(monkeypatch, capsys):
    """`lerobot-twin calib-sim --robot a1x` przekazuje opis A1X do sesji (bez renderu - sesja podmieniona)."""
    from lerobot_mp.twin.calib import simulate
    from lerobot_mp.twin.cli import main as cli_main
    from lerobot_mp.twin.robots import A1X

    seen = []

    def fake_run(seed, n_cameras=1, spec=SO101, **kw):
        seen.append((seed, n_cameras, spec))
        cams = [f"cam{k}" for k in range(n_cameras)]
        return simulate.Result(seed, 12, {c: True for c in cams}, {c: "" for c in cams},
                               {c: 0.3 for c in cams}, {c: 0.5 for c in cams}, {c: 0.02 for c in cams}, 0.1, 0.01)

    monkeypatch.setattr(simulate, "run", fake_run)
    assert cli_main(["calib-sim", "--robot", "a1x", "--n", "2", "--cameras", "2", "--seed", "3"]) == 0
    assert seen == [(3, 2, A1X), (4, 2, A1X)]
    out = capsys.readouterr().out
    assert "Galaxea A1X" in out and "zaufane kamery: 4/4" in out
    seen.clear()
    assert cli_main(["calib-sim", "--n", "1"]) == 0
    assert seen == [(0, 2, SO101)]
