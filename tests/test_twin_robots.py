"""Opis ramienia i kinematyka blizniaka.

Najwazniejszy test jest pierwszy: MJCF z MuJoCo Menagerie ma sie zgadzac
z niezaleznym modelem Articulusa co do polozenia kazdego czlonu w dowolnej
pozie. Articulus jest tym, na czym stoi podglad 3D aplikacji i przeliczenia
backendu `feetech`, wiec zgodnosc znaczy, ze katy z prawdziwego ramienia
wpisane w symulacje daja w niej te sama poze. Bez tego blizniak klamie.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

# Przed importem blizniaka: kinematics importuje mujoco, a bez niego (instalacja
# bez [twin]) caly plik ma sie pominac, a nie przerwac zbieranie testow.
mujoco = pytest.importorskip("mujoco")

from lerobot_mp.config import JOINT_NAMES, load_config  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics, inverse  # noqa: E402
from lerobot_mp.twin.robots import REGISTRY, SO101, get_spec, joint_configs, robot_config  # noqa: E402
from lerobot_mp.paths import REPO_ROOT  # noqa: E402


#: Cialo nadgarstka kazdego ramienia - tylko dla testu kierunku podejscia (bez pola w opisie).
WRIST = {"so101": "wrist", "a1x": "arm_link6"}


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics(SO101)


@pytest.fixture
def spec_kin(spec) -> RobotKinematics:
    return RobotKinematics(spec)


def random_joints(kin: RobotKinematics, rng: np.random.Generator) -> dict[str, float]:
    spec = kin.spec
    joints = {
        name: float(rng.uniform(np.degrees(lo), np.degrees(hi)))
        for name, lo, hi in zip(spec.joints, kin.lo, kin.hi)
    }
    if spec.gripper:
        joints[spec.gripper] = float(rng.uniform(0.0, 100.0))
    return joints


def test_mjcf_matches_the_articulus_reference_in_every_pose(kin):
    preview = pytest.importorskip("lerobot_mp.preview.model")
    art = preview.load_model()
    if art is None:
        pytest.skip("brak modelu podgladu 3D w repozytorium")

    base = art.link_names.index("base")
    links = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        joints = random_joints(kin, rng)
        Ta = art.link_transforms(art.from_lerobot(joints))
        for name in links:
            ref = (inverse(Ta[base]) @ Ta[art.link_names.index(name)])[:3, 3]
            worst = max(worst, float(np.linalg.norm(ref - kin.body(name, joints)[:3, 3])))
    assert worst < 1e-5, f"czlony rozjezdzaja sie o {worst * 1000:.3f} mm"


def test_units_round_trip(spec_kin):
    kin = spec_kin
    rng = np.random.default_rng(1)
    for _ in range(20):
        joints = random_joints(kin, rng)
        back = kin.from_q(kin.to_q(joints))
        for name in kin.spec.joints:
            assert back[name] == pytest.approx(joints[name], abs=1e-9)


def test_gripper_angle_is_the_servo_angle_of_the_feetech_backend(kin):
    """0..100 chwytaka ma byc w blizniaku TYM SAMYM katem szczeki, co na serwie.

    Wczesniej 0..100 szlo liniowo na caly zakres MJCF (-10..100 st.), a backend
    `feetech` rozciaga je na tiki 1986..2670 (60 st. skoku serwa) - jednostka
    chwytaka byla w symulacji 1,8 raza wieksza niz na ramieniu.
    """
    from lerobot_mp.config import load_config
    from lerobot_mp.robot.feetech import FeetechArm

    cfg = load_config(overrides={"robot": {"backend": "feetech", "port": "COM_TEST"}})
    arm = FeetechArm(cfg, bus=object())                   # tylko przeliczenia, bez portu
    rc = cfg.robot
    k = SO101.joints.index("gripper")
    for ticks in (rc.gripper_closed_ticks, 2100, 2300, rc.gripper_open_ticks):
        servo_deg = (ticks - rc.center_ticks) * 360.0 / 4096
        units = arm._to_units("gripper", ticks)
        assert np.degrees(kin.to_q({"gripper": units})[k]) == pytest.approx(servo_deg, abs=1e-6)
        assert kin.from_q(kin.to_q({"gripper": units}))["gripper"] == pytest.approx(units, abs=1e-6)
    # Poza zakresem stawu MJCF kat jest przyciety, a nie zawiniety.
    assert kin.to_q({"gripper": 1000.0})[k] == pytest.approx(kin.hi[k])
    assert kin.to_q({"gripper": -1000.0})[k] == pytest.approx(kin.lo[k])


def test_approach_axis_points_out_of_the_jaws(spec_kin):
    """Krok wzdluz podejscia ma oddalac TCP od nadgarstka - inaczej karta
    kalibracyjna sterczalaby w strone ramienia."""
    kin = spec_kin
    joints = dict(kin.spec.home)
    tcp = kin.tcp(joints)
    wrist = kin.body(WRIST[kin.spec.name], joints)[:3, 3]
    approach, _ = kin.tool_axes(joints)
    before = np.linalg.norm(tcp[:3, 3] - wrist)
    after = np.linalg.norm(tcp[:3, 3] + 0.05 * approach - wrist)
    assert after > before + 0.045


def test_closing_axis_separates_the_jaw_tips(spec_kin):
    """Os zamykania to kierunek, w ktorym rozchodza sie czubki szczek."""
    kin = spec_kin
    spec = kin.spec
    joints = dict(spec.home, **{spec.gripper: 50.0})
    _, closing = kin.tool_axes(joints)
    d = kin.data
    fixed = d.geom_xpos[kin.model.geom(spec.fingertips[0]).id]
    moving = d.geom_xpos[kin.model.geom(spec.fingertips[1]).id]
    gap = (moving - fixed) / np.linalg.norm(moving - fixed)
    assert float(gap @ closing) > 0.9


def test_ik_reaches_poses_the_arm_can_take(kin):
    """Cel wziety z kinematyki prostej jest osiagalny z definicji - IK ma w niego trafic."""
    rng = np.random.default_rng(2)
    misses = []
    for _ in range(25):
        joints = random_joints(kin, rng)
        target = kin.tcp(joints)[:3, 3]
        sol = kin.ik(target, seed=SO101.home, rng=np.random.default_rng(3))
        reached = kin.tcp(sol.joints)[:3, 3]
        if not sol.ok:
            misses.append(sol.pos_err)
        else:
            assert np.linalg.norm(reached - target) < 1.5e-3
    assert len(misses) <= 2, f"IK nie trafil w {len(misses)}/25 osiagalnych celow: {misses}"


def test_ik_keeps_the_gripper_as_it_was(kin):
    target = kin.tcp(SO101.home)[:3, 3] + np.array([0.0, 0.03, 0.02])
    sol = kin.ik(target, seed=dict(SO101.home, gripper=80.0))
    assert sol.joints["gripper"] == pytest.approx(80.0, abs=1e-6)


def test_ik_treats_orientation_as_soft_for_a_five_joint_arm(kin):
    """Pelnej orientacji SO-101 nie osiagnie - pozycja ma byc trafiona mimo to."""
    home_tcp = kin.tcp(SO101.home)
    target = home_tcp[:3, 3] + np.array([0.02, 0.05, 0.0])
    # Obrot o 90 stopni wokol pionu: tego piec stawow w tej pozycji nie zrobi.
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ home_tcp[:3, :3]
    sol = kin.ik(target, turn, seed=SO101.home)
    assert sol.ok
    assert np.linalg.norm(kin.tcp(sol.joints)[:3, 3] - target) < 1.5e-3


def test_ik_matches_orientation_when_the_arm_can_take_it(kin):
    """Pierwszenstwo pozycji nie moze znaczyc ignorowania obrotu: poza wzieta
    z kinematyki prostej jest osiagalna w calosci i ma byc trafiona w calosci."""
    rng = np.random.default_rng(4)
    rot_errors = []
    for _ in range(15):
        joints = random_joints(kin, rng)
        T = kin.tcp(joints)
        sol = kin.ik(T[:3, 3], T[:3, :3], seed=SO101.home, rng=np.random.default_rng(5))
        if sol.ok:
            rot_errors.append(sol.rot_err)
    assert len(rot_errors) >= 13
    assert float(np.median(rot_errors)) < 0.02


def test_unknown_robot_is_reported_with_the_known_ones():
    with pytest.raises(KeyError, match="so101"):
        get_spec("ur5")


def test_so101_spec_limits_equal_the_hand_app_defaults():
    """Limity SO-101 sa w dwoch miejscach (opis blizniaka, aplikacja dloni) - maja byc te same."""
    hand = load_config()
    twin = joint_configs(SO101)
    assert tuple(twin) == JOINT_NAMES
    for n in JOINT_NAMES:
        assert (twin[n].min, twin[n].max, twin[n].max_vel) == (
            hand.joint(n).min, hand.joint(n).max, hand.joint(n).max_vel)


def test_robot_config_uses_spec_joints_and_home():
    cfg = robot_config(SO101)
    assert tuple(cfg.joints) == SO101.joints
    assert cfg.safety.home == SO101.home == load_config().safety.home
    over = robot_config(SO101, overrides={"joints": {"gripper": {"max_vel": 42}}})
    assert over.joint("gripper").max_vel == 42
    assert robot_config(SO101).joint("gripper").max_vel == 300.0     # opis nietkniety
    with pytest.raises(ValueError, match="Nieznany staw"):
        robot_config(SO101, overrides={"joints": {"knee": {"min": 0}}})


def _without_joints(cfg) -> dict:
    d = dataclasses.asdict(cfg)
    d.pop("joints")
    return d


def test_robot_config_equals_load_config_for_so101_with_a_partial_home_in_the_file(tmp_path, monkeypatch):
    """Czesciowa `safety.home` z pliku zastepuje poze domowa w calosci - jak w aplikacji dloni.

    Opis nie moze uzupelniac brakujacych stawow: ten sam plik dawalby blizniakowi
    inna poze startowa (rampa po polaczeniu) niz `lerobot-mp`.
    """
    from lerobot_mp.control.safety import SafetySupervisor

    path = tmp_path / "local.yaml"
    path.write_text("safety:\n  home:\n    shoulder_pan: 10.0\n", encoding="utf-8")
    assert robot_config(SO101, path).safety.home == {"shoulder_pan": 10.0}
    assert _without_joints(robot_config(SO101, path)) == _without_joints(load_config(path))
    monkeypatch.setenv("LEROBOT_MP_CONFIG", str(path))
    assert SafetySupervisor(robot_config(SO101)).home == SafetySupervisor(load_config()).home
    assert SafetySupervisor(robot_config(SO101)).home["elbow_flex"] == 0.0
    monkeypatch.delenv("LEROBOT_MP_CONFIG")
    over = {"safety": {"home": {"gripper": 50.0}}}
    assert robot_config(SO101, overrides=over).safety.home == {"gripper": 50.0}
    assert _without_joints(robot_config(SO101, path, over)) == _without_joints(load_config(path, over))
    default_yaml = REPO_ROOT / "configs" / "default.yaml"
    assert _without_joints(robot_config(SO101, default_yaml)) == _without_joints(load_config(default_yaml))


def test_robot_config_skips_the_so101_tables_of_the_file_for_another_robot(toy_spec):
    """`configs/default.yaml` (kopiowany do `local.yaml`) nie moze wywrocic ani przestawic innego ramienia."""
    default_yaml = REPO_ROOT / "configs" / "default.yaml"
    cfg = robot_config(toy_spec, default_yaml)
    assert tuple(cfg.joints) == toy_spec.joints
    assert cfg.safety.home == toy_spec.home                     # nie gripper 35 z tabeli SO-101
    assert cfg.joint("gripper").max_vel == toy_spec.max_vel["gripper"]
    hand = load_config(default_yaml)
    assert cfg.safety.startup_ramp_s == hand.safety.startup_ramp_s    # reszta pliku dziala
    assert cfg.robot == hand.robot
    over = robot_config(toy_spec, default_yaml, {"joints": {"j1": {"max_vel": 7.0}},
                                                  "safety": {"home": {"j1": 5.0}}})
    assert over.joint("j1").max_vel == 7.0
    assert over.safety.home == {"j1": 5.0}
    with pytest.raises(ValueError, match="Nieznany staw"):
        robot_config(toy_spec, overrides={"joints": {"shoulder_pan": {"min": 0}}})


def test_scene_set_joints_leaves_the_mirror_joint_at_rest(toy_spec):
    """Poza wpisana bez dynamiki stoi - takze staw sprzezony, nie tylko stawy opisu."""
    from lerobot_mp.twin import scene as sc

    scene = sc.build(sc.SceneConfig(toy_spec))
    dadr_m = scene.model.jnt_dofadr[scene.model.joint(sc.PREFIX + "gripper_m").id]
    scene.set_joints({"j1": 0.0, "j2": 20.0, "gripper": 0.0})
    scene.command({"j1": 0.0, "j2": 20.0, "gripper": 100.0})
    scene.step(0.05)
    assert abs(scene.data.qvel[dadr_m]) > 1e-3                   # szczeka w ruchu
    scene.set_joints({"j1": 0.0, "j2": 20.0, "gripper": 0.0})
    assert scene.data.qvel[dadr_m] == 0.0
    assert np.all(scene.data.qvel[scene.kin.dadr] == 0.0)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registry_entry_declares_limits_and_max_vel_for_all_joints(name):
    spec = REGISTRY[name]
    assert set(spec.limits) == set(spec.joints)
    assert set(spec.max_vel) == set(spec.joints)
    for n in spec.joints:
        lo, hi = spec.limits[n]
        assert lo < hi and spec.max_vel[n] > 0
    assert set(spec.home) == set(spec.joints)
    for n, v in spec.home.items():
        assert spec.limits[n][0] <= v <= spec.limits[n][1]


def test_linear_gripper_spec_maps_0_100_onto_its_declared_range(monkeypatch):
    from dataclasses import replace

    from lerobot_mp.twin import kinematics

    def no_ticks():
        raise AssertionError("opis liniowy nie moze czytac tikow serwa")

    monkeypatch.setattr(kinematics, "backend_gripper_ticks", no_ticks)
    lin = RobotKinematics(replace(SO101, gripper_range_q=(-0.1, 0.9)))
    k = SO101.joints.index("gripper")
    assert lin.gripper_ticks is None
    assert lin.to_q({"gripper": 0.0})[k] == pytest.approx(max(-0.1, lin.lo[k]))
    assert lin.to_q({"gripper": 100.0})[k] == pytest.approx(min(0.9, lin.hi[k]))
    assert lin.to_q({"gripper": 50.0})[k] == pytest.approx(0.4)
    assert lin.to_q({"gripper": 1000.0})[k] == pytest.approx(min(0.9, lin.hi[k]))
    for v in (5.0, 37.5, 50.0, 90.0):
        assert lin.from_q(lin.to_q({"gripper": v}))["gripper"] == pytest.approx(v, abs=1e-9)


def test_mirror_joint_is_written_by_kinematics_scene_and_collision_checker(toy_spec):
    from lerobot_mp.twin import scene as sc
    from lerobot_mp.twin.collision import CollisionChecker

    kin = RobotKinematics(toy_spec)
    k = toy_spec.joints.index("gripper")
    m = kin.model
    adr_m = m.jnt_qposadr[m.joint("gripper_m").id]
    joints = {"j1": 10.0, "j2": 15.0, "gripper": 70.0}
    q = kin.to_q(joints)
    assert q[k] == pytest.approx(0.014)
    kin.tcp(joints)
    assert kin.data.qpos[adr_m] == -q[k]

    scene = sc.build(sc.SceneConfig(toy_spec))
    adr_s = scene.model.jnt_qposadr[scene.model.joint(sc.PREFIX + "gripper_m").id]
    scene.set_joints(joints)
    assert scene.data.qpos[adr_s] == -q[k]
    assert scene.joints()["gripper"] == pytest.approx(70.0)

    checker = CollisionChecker(scene)
    checker._contacts(dict(joints, gripper=30.0))
    assert checker.data.qpos[adr_s] == -kin.to_q({"gripper": 30.0})[k]
