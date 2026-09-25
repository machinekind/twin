"""Galaxea A1X w blizniaku: model, opis, kinematyka, scena, srodowisko RL i blizniak w symulacji.

Model jest generowany z URDF producenta (`scripts/import_a1x_model.py`); kazda regula
przetworzenia ma tu swoj test, zeby opis ramienia i MJCF nie mogly sie rozjechac.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from lerobot_mp.paths import REPO_ROOT  # noqa: E402
from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics  # noqa: E402
from lerobot_mp.twin.robots import A1X, get_spec  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

#: TCP podejsciem w dol (x site'u na -z podstawy), szczeki zamykaja sie wzdluz y podstawy.
DOWN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(A1X.mjcf_path))


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics(A1X)


def _import_script():
    path = REPO_ROOT / "scripts" / "import_a1x_model.py"
    spec = importlib.util.spec_from_file_location("import_a1x_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ model
def test_a1x_mjcf_compiles_under_the_pinned_mujoco(model):
    m = model
    assert m.nbody == 11                                     # swiat + 10 czlonow z URDF
    joints = [m.joint(i).name for i in range(m.njnt)]
    assert joints == [*A1X.joints, "gripper_finger_joint2"]  # regula 1: gripper_finger_joint1 -> gripper
    assert tuple(m.actuator(i).name for i in range(m.nu)) == A1X.joints          # regula 2
    for i in range(m.nu):
        assert m.joint(m.actuator_trnid[i, 0]).name == m.actuator(i).name
    for tip in ("finger1_tip", "finger2_tip"):                                   # regula 3
        g = m.geom(tip)
        assert m.body(m.geom_bodyid[g.id]).name == ("gripper_finger_link" + tip[6])
        assert m.geom_type[g.id] == mujoco.mjtGeom.mjGEOM_BOX
    assert m.opt.timestep == 0.002                                                # regula 4
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert m.opt.impratio == pytest.approx(10.0)
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert m.site("tcp").pos == pytest.approx([0.045, 0.0, 0.0])                  # regula 5
    assert m.site_quat[m.site("tcp").id] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    key = m.key("home")
    assert key.qpos == pytest.approx([0.0, 1.0, -1.6, 0.6, 0.0, 0.0, 0.03, -0.03])
    assert key.ctrl == pytest.approx([0.0, 1.0, -1.6, 0.6, 0.0, 0.0, 0.03])
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key.id)
    mujoco.mj_forward(m, d)
    assert np.isfinite(d.qacc).all()
    assert m.ntendon == 0
    eq = [i for i in range(m.neq) if m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT]
    assert len(eq) == m.neq == 1
    pair = {m.joint(m.eq_obj1id[eq[0]]).name, m.joint(m.eq_obj2id[eq[0]]).name}
    assert pair == {"gripper_finger_joint2", "gripper"}
    assert m.eq_solref[eq[0]] == pytest.approx([0.01, 1.0])                        # regula 6
    assert m.eq_solimp[eq[0]] == pytest.approx([0.99, 0.999, 0.001, 0.5, 2.0])


def test_a1x_import_script_is_idempotent(tmp_path):
    """Skrypt z zaimportowanego URDF daje dokladnie zatwierdzony plik - nikt nie poprawial go recznie."""
    script = _import_script()
    out = tmp_path / "a1x.xml"
    assert script.main(["--out", str(out)]) == 0
    assert out.read_bytes() == A1X.mjcf_path.read_bytes()
    meshes = sorted(p.name for p in (tmp_path / "assets").iterdir())
    assert meshes == sorted(p.name for p in (A1X.mjcf_path.parent / "assets").glob("*.STL"))
    assert len(meshes) == 10


def test_a1x_mirror_spec_matches_the_mjcf_equality(model):
    """`A1X.mirror_joints` to ta sama rownosc, co w MJCF - opis i model nie moga sie rozjechac."""
    m = model
    assert len(A1X.mirror_joints) == m.neq == 1
    (slaved, (src, gain)), = A1X.mirror_joints.items()
    assert m.joint(m.eq_obj1id[0]).name == slaved
    assert m.joint(m.eq_obj2id[0]).name == src
    # polycoef: q1 = c0 + c1 * q2 + ... -> 0 - 1 * q2
    assert m.eq_data[0, :5] == pytest.approx([0.0, gain, 0.0, 0.0, 0.0])


# ------------------------------------------------------------------ opis
def test_registry_lists_a1x():
    assert get_spec("A1X") is A1X
    assert get_spec("a1x").joints[-1] == A1X.gripper == "gripper"
    with pytest.raises(KeyError, match="so101"):
        get_spec("ur5")


def test_a1x_home_is_above_the_table_and_reachable(kin):
    """Poza domowa (keyframe `home` modelu) nad blatem i przed podstawa.

    Plan zakladal x > 0,20 m; keyframe galaxeo ma TCP w x = 0,169 m, z = 0,549 m
    (ramie uniesione, szczeki nad podstawa). Poza domowa zostaje ta z modelu
    i z mostka galaxeo, wiec prog x to 0,15 m.
    """
    T = kin.tcp(A1X.home)
    assert T[2, 3] > 0.10
    assert T[0, 3] > 0.15
    assert abs(T[1, 3]) < 0.01
    for n, v in A1X.home.items():
        lo, hi = A1X.limits[n]
        assert lo <= v <= hi
    sol = kin.ik(T[:3, 3], seed=dict(A1X.home, arm_joint2=80.0, arm_joint3=-70.0))
    assert sol.ok


def test_a1x_units_round_trip_and_gripper_is_linear_in_metres(kin):
    k = A1X.joints.index("gripper")
    assert kin.gripper_ticks is None                         # tiki serwa SO-101 nic tu nie znacza
    assert kin.to_q({"gripper": 100.0})[k] == pytest.approx(0.05)
    assert kin.to_q({"gripper": 60.0})[k] == pytest.approx(0.03)
    assert kin.to_q({"gripper": 0.0})[k] == pytest.approx(0.0)
    assert kin.to_q({"arm_joint2": 57.3})[1] == pytest.approx(np.radians(57.3))
    back = kin.from_q(kin.to_q(A1X.home))
    for n in A1X.joints:
        assert back[n] == pytest.approx(A1X.home[n], abs=1e-9)


def test_a1x_wave_ranges_keep_the_card_over_the_table(kin):
    """Fala kalibracyjna losuje stawy z `wave_ranges` - wiekszosc poz ma trzymac karte nad stolem.

    Osiagniete: 170/200 = 85 % (zakresy z planu: 57 %).
    """
    rng = np.random.default_rng(0)
    lo_r, hi_r = A1X.desk.wave_radius
    good = 0
    for _ in range(200):
        joints = dict(A1X.home, gripper=0.0)
        for name, (lo, hi) in A1X.wave_ranges.items():
            joints[name] = float(rng.uniform(lo, hi))
        p = kin.tcp(joints)[:3, 3]
        good += lo_r <= float(np.hypot(p[0], p[1])) <= hi_r and p[2] > 0.03
    assert good >= 0.6 * 200, f"tylko {good}/200 poz nad stolem"


def test_a1x_ik_reaches_sampled_goals(kin):
    """Cele `reach` w skali A1X sa osiagalne dla IK.

    Osiagniete: 20/20 celow (100 %), najgorszy blad 0,33 mm.
    """
    from lerobot_mp.twin.rl import task as tk

    rng = np.random.default_rng(0)
    goals = tk.sample_goals(kin, tk.make_task("reach", A1X), rng, 20)
    hits = 0
    for g in goals:
        sol = kin.ik(g, seed=A1X.home, rng=np.random.default_rng(1))
        hits += float(np.linalg.norm(kin.tcp(sol.joints)[:3, 3] - g)) < 0.005
    assert hits >= 16, f"IK trafil tylko {hits}/20 celow"


# ------------------------------------------------------------------ scena i RL
def _mirror_err(kin, qpos) -> float:
    (k, adr, gain), = kin.mirror
    return abs(float(qpos[adr]) - gain * float(qpos[kin.qadr[k]]))


def test_a1x_mirror_finger_follows_in_scene_and_env_reset():
    from lerobot_mp.twin.rl.env import TwinEnv

    with sc.build(sc.SceneConfig(A1X)) as scene:
        for g in (0.0, 35.0, 100.0):
            scene.set_joints(dict(A1X.home, gripper=g))
            assert _mirror_err(scene.kin, scene.data.qpos) == 0.0
            assert scene.joints()["gripper"] == pytest.approx(g)

    env = TwinEnv("lift", workspace=Workspace(robot="a1x"))
    try:
        d = env.scene.data
        for seed in range(3):
            env.reset(seed=seed)
            assert _mirror_err(env.kin, d.qpos) == 0.0
            for _ in range(5):
                env.step(np.zeros(env.task.act_dim))
            # Rownosc jest miekka (regula 6 ja usztywnia, ale nie do zera): przy zamykaniu
            # (serwo tylko na szczece 1) druga szczeka moze zostac kilka mm z tylu, a gdy
            # szczeka stoi - dochodzi (< 0,2 mm po 25 taktach).
            assert _mirror_err(env.kin, d.qpos) < 0.004
            for _ in range(20):
                env.step(np.zeros(env.task.act_dim))
            assert _mirror_err(env.kin, d.qpos) < 2e-4
    finally:
        env.close()


def test_a1x_jaws_stay_symmetric_under_violent_arm_actions():
    """Regula 6: akcje ramienia +-1 co takt przy kostce nie rozsuwaja szczek.

    Epizody startuja ze szczekami wokol kostki (IK), wiec szczeki trzaskaja o blat
    i kostke. Zmierzone (ziarna 200-202): najgorzej 5,8 mm; z domyslna rownoscia
    (solref 0.02, solimp 0.9/0.95) 36 mm.
    """
    from lerobot_mp.twin.rl.env import TwinEnv

    env = TwinEnv("lift", workspace=Workspace(robot="a1x"))
    d = env.scene.data
    worst = 0.0
    try:
        for ep in range(3):
            env.reset(seed=200 + ep)
            cube = env.state()["cube_pos"]
            sol = env.kin.ik(cube + np.array([0.0, 0.0, 0.04]), DOWN, seed=dict(A1X.home))
            assert sol.ok
            q = env.kin.to_q(dict(sol.joints, gripper=float(np.random.default_rng(ep).uniform(0.0, 100.0))))
            env.reset(seed=200 + ep, options={"q": q, "cube": (cube, np.array([1.0, 0.0, 0.0, 0.0]))})
            rng = np.random.default_rng(ep)
            for _ in range(env.task.episode_steps):
                a = rng.choice([-1.0, 1.0], size=env.task.act_dim).astype(np.float32)
                _, _, term, trunc, _ = env.step(a)
                assert np.isfinite(d.qpos).all()
                worst = max(worst, _mirror_err(env.kin, d.qpos))
                if term or trunc:
                    break
    finally:
        env.close()
    assert worst < 0.015, f"szczeki rozjechaly sie o {worst * 1000:.1f} mm"


def test_a1x_twin_in_sim_moves_a_joint():
    from lerobot_mp.twin.runtime import Twin

    tw = Twin(Workspace(robot="a1x"))
    try:
        assert tw.spec is A1X
        tw.connect("sim", threaded=False)
        tw.set_engaged(True)
        tw.set_target({"arm_joint2": 70.0})
        for _ in range(150):
            tw.step(0.02)
        assert tw.status.measured["arm_joint2"] == pytest.approx(70.0, abs=2.0)
        assert set(tw.status.command) == set(A1X.joints) and len(tw.status.command) == 7
        assert tw.joint_limits()["arm_joint2"] == A1X.limits["arm_joint2"]
    finally:
        tw.close()


def test_a1x_lift_env_passes_the_gymnasium_checker():
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    from lerobot_mp.twin.rl.env import TwinEnv

    env = TwinEnv("lift", workspace=Workspace(robot="a1x"))
    try:
        assert env.task.act_dim == 7 and env.task.n_joints == 7
        assert env.task.obs_dim == 36
        assert env.task.cube_radius == A1X.desk.cube_radius
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_a1x_random_actions_keep_the_physics_stable():
    """Krok RL 5 ms na modelu strojonym pod 2 ms: losowe akcje przy kostce, bez NaN i wystrzalow.

    Epizody startuja ze szczekami wokol kostki (IK), zeby losowe akcje naprawde
    wciskaly szczeki w kostke i blat. Zmierzone (ziarna 100-111, 2400 taktow, 227
    z kontaktem szczeki): kostka najwyzej 2,0 m/s, najwyzej 23 mm od blatu, zero
    NaN (prog 3 m/s) - krok 5 ms wystarcza, `Desk.timestep` niepotrzebny.
    """
    from lerobot_mp.twin.rl.env import TwinEnv
    from lerobot_mp.twin.rl.randomize import Randomization

    env = TwinEnv("lift", workspace=Workspace(robot="a1x"), randomization=Randomization())
    assert env.scene.model.opt.timestep == pytest.approx(0.005)
    d = env.scene.data
    env.action_space.seed(0)
    touched = 0
    try:
        for ep in range(12):
            env.reset(seed=100 + ep)
            cube = env.state()["cube_pos"]
            # Czubki szczek sa 33 mm za TCP wzdluz podejscia: TCP 4 cm nad srodkiem kostki.
            sol = env.kin.ik(cube + np.array([0.0, 0.0, 0.04]), DOWN, seed=dict(A1X.home))
            assert sol.ok
            q = env.kin.to_q(dict(sol.joints, gripper=float(np.random.default_rng(ep).uniform(0.0, 100.0))))
            env.reset(seed=100 + ep, options={"q": q, "cube": (cube, np.array([1.0, 0.0, 0.0, 0.0]))})
            for _ in range(env.task.episode_steps):
                obs, _, term, trunc, info = env.step(env.action_space.sample())
                assert np.isfinite(d.qpos).all() and np.isfinite(obs).all()
                v = float(np.linalg.norm(d.qvel[env.cube_dadr:env.cube_dadr + 3]))
                assert v < 3.0 and abs(info["height"]) < 0.5, f"epizod {ep}: kostka {v:.1f} m/s"
                touched += bool(env.state()["jaws"].max() > 0.5)
                if term or trunc:
                    break
    finally:
        env.close()
    assert touched > 50, "szczeki nie dotknely kostki - test nie sprawdza kontaktow"
