"""Srodowiska RL blizniaka: poprawnosc API, wykonalnosc zadan, zgodnosc CPU/GPU, stabilnosc fizyki."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")

import lerobot_mp.twin.rl  # noqa: E402,F401
from lerobot_mp.twin.rl import task as tk  # noqa: E402
from lerobot_mp.twin.rl.env import TwinEnv  # noqa: E402
from lerobot_mp.twin.rl.policy import Policy, PolicyMeta  # noqa: E402
from lerobot_mp.twin.rl.randomize import Dynamics, Randomization  # noqa: E402

_no_warp = importlib.util.find_spec("mujoco_warp") is None or importlib.util.find_spec("warp") is None


def cuda(fn):
    """Test na GPU: potrzebuje CUDA w torchu ORAZ MuJoCo Warp (extra [train]) - bez nich pominiety."""
    fn = pytest.mark.skipif(_no_warp, reason="brak mujoco_warp/warp - pip install -e \".[train]\"")(fn)
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo Warp potrzebuje GPU z CUDA")(fn)


@pytest.mark.parametrize("name", ["LeRobotMP/TwinReach-v0", "LeRobotMP/TwinLift-v0", "LeRobotMP/TwinPickPlace-v0"])
def test_env_passes_the_gymnasium_checker(name):
    from gymnasium.utils.env_checker import check_env

    env = gym.make(name)
    try:
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def test_scripted_ik_expert_solves_reach():
    """Dowod, ze zadanie jest wykonalne w granicach akcji - zanim cokolwiek sie uczy."""
    env = TwinEnv("reach", randomization=Randomization.none())
    try:
        for ep in range(5):
            env.reset(seed=ep)
            q_goal = env.kin.to_q(env.kin.ik(env.goal, seed=env.kin.from_q(env.q_cmd)).joints)
            for _ in range(env.task.episode_steps):
                a = np.zeros(env.task.act_dim)
                a[:5] = np.clip((q_goal[:5] - env.q_cmd[:5]) / env.task.arm_step, -1, 1)
                _, _, term, trunc, info = env.step(a)
                if term or trunc:
                    break
            assert info["success"], f"epizod {ep}: {info['distance'] * 1000:.1f} mm od celu"
    finally:
        env.close()


# ------------------------------------------------------------------ pickplace
def _workspace(robot: str):
    from lerobot_mp.twin.workspace import Workspace

    return Workspace(robot=robot)


def _top_down(spec, yaw: float) -> np.ndarray:
    """Obrot site'u TCP (uklad podstawy): podejscie pionowo w dol, szczeki zamykaja sie wzdluz `yaw`."""
    a, c = np.asarray(spec.tcp_approach, float), np.asarray(spec.tcp_closing, float)
    down, closing = np.array([0.0, 0.0, -1.0]), np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.column_stack([down, closing, np.cross(down, closing)]) @ np.column_stack([a, c, np.cross(a, c)]).T


def _grasp_height(env) -> float:
    """Wysokosc TCP nad blatem przy chwycie: kostka w polowie, ale czubki szczek 6 mm nad blatem.

    Site `tcp` A1X lezy 30 mm przed czubkami (wzdluz podejscia), SO-101 - miedzy nimi.
    """
    kin, spec = env.kin, env.kin.spec
    joints = dict(spec.home, **{spec.gripper: 0.0})
    T = kin.tcp(joints)                                   # liczy tez kinematyke `kin.data` dla tej pozy
    m, d = kin.model, kin.data
    base = m.body(kin.prefix + spec.base_body).id
    Rb, tb = d.xmat[base].reshape(3, 3), d.xpos[base]
    reach = 0.0
    for tip in spec.fingertips:
        g = m.geom(kin.prefix + tip).id
        p_base = Rb.T @ (d.geom_xpos[g] - tb)
        along = float(np.asarray(spec.tcp_approach) @ (T[:3, :3].T @ (p_base - T[:3, 3])))
        reach = max(reach, along + float(m.geom_size[g].max()))
    return max(env.task.cube_half, reach + 0.006)


def scripted_pickplace(env, seed: int, actions=None):
    """Ekspert IK: nad kostke -> w dol -> zamknij -> w gore ponad `lift_height` -> nad cel -> w dol ->
    otworz -> odjazd. Kazdy takt to zwykla akcja w [-1, 1] (najwyzej `arm_step` na staw).

    Zwraca (ostatnie info, liczba taktow, takt pierwszego sukcesu albo None, akcje).
    `actions` - odtwarza podane akcje zamiast planowac (test zgodnosci GPU/CPU).
    """
    env.reset(seed=seed)
    t, lim, kin = env.task, env.limits, env.kin
    g = lim.grip
    log = {"ticks": 0, "first": None, "actions": []}

    def tick(a):
        _, _, term, trunc, info = env.step(a)
        log["ticks"] += 1
        log["actions"].append(np.asarray(a, float).copy())
        if info["success"] and log["first"] is None:
            log["first"] = log["ticks"]
        return info, term or trunc

    if actions is not None:
        for a in actions:
            info, end = tick(a)
            if end:
                break
        return info, log["ticks"], log["first"], log["actions"]

    s = env.state()
    cube, R = s["cube_pos"], s["cube_rot"]
    yaw = (np.arctan2(R[1, 0], R[0, 0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4
    z = _grasp_height(env)
    up, hover = z + t.lift_height + 0.02, z + 0.05

    def solve(p, seed_q):
        """Obrot szczek z czterech rownowaznych (kostka ma symetrie 90 st.) - trafiony, w limitach, najblizej."""
        best = None
        for y in (yaw, yaw + np.pi / 2, yaw - np.pi / 2, yaw + np.pi):
            sol = kin.ik(p, _top_down(kin.spec, y), seed=kin.from_q(seed_q))
            q = kin.to_q(sol.joints)
            inside = bool(np.all(q[:g] >= lim.lo[:g]) and np.all(q[:g] <= lim.hi[:g]))
            key = (not sol.ok, not inside, sol.rot_err > 0.1, float(np.abs(q[:g] - seed_q[:g]).sum()))
            if best is None or key < best[0]:
                best = (key, q)
        return np.clip(best[1], lim.lo, lim.hi)

    tgt = env.target
    plan = [(np.r_[cube[:2], hover], 1.0), (np.r_[cube[:2], z], 1.0), (12, -1.0), (np.r_[cube[:2], up], -1.0),
            (np.r_[tgt[:2], up], -1.0), (np.r_[tgt[:2], z + 0.004], -1.0), (10, 1.0), (np.r_[tgt[:2], hover], 1.0),
            (40, 1.0)]
    q_ref = env.q_cmd.copy()
    for step, grip in plan:
        if not isinstance(step, int):
            q_ref = solve(step, q_ref)
        for k in range(1, 121):
            a = np.zeros(t.act_dim)
            a[:g] = np.clip((q_ref[:g] - env.q_cmd[:g]) / t.arm_step, -1.0, 1.0)
            a[g] = grip
            info, end = tick(a)
            if end:
                return info, log["ticks"], log["first"], log["actions"]
            if isinstance(step, int):
                if k >= step:
                    break
            elif np.abs(q_ref[:g] - env.q_cmd[:g]).max() < 1e-6 and np.abs(env.state()["q"][:g] - q_ref[:g]).max() < 0.02:
                break
    return info, log["ticks"], log["first"], log["actions"]


@pytest.mark.parametrize("robot", ["so101", "a1x"])
def test_scripted_expert_solves_pickplace(robot):
    """Dowod wykonalnosci pickplace w granicach akcji - na obu ramionach, zanim pojdzie czas GPU.

    Zmierzone (Randomization.none(), seedy 0-4; pierwszy takt sukcesu / koniec po serii 10):
    so101 105/123, 106/126, 89/98, 90/108, 99/108; a1x 82/91, 85/94, 86/95, 79/88, 86/95
    - wszystko ponizej `episode_steps` = 300.
    """
    env = TwinEnv("pickplace", workspace=_workspace(robot), randomization=Randomization.none())
    try:
        for seed in range(5):
            info, ticks, first, _ = scripted_pickplace(env, seed)
            where = (f"{robot} seed {seed}: takt {ticks}, kostka {info['dist_target'] * 1000:.1f} mm od celu, "
                     f"{info['height'] * 1000:.1f} mm nad blatem, niesiona {info['carried']}")
            assert first is not None and first < env.task.episode_steps, where
            assert info["finished"] and info["carried"], where
    finally:
        env.close()


def test_pickplace_obs_dim_is_39_for_so101_and_42_for_a1x():
    from lerobot_mp.twin.robots import A1X, SO101

    assert tk.make_task("pickplace", SO101).obs_dim == 39 == tk.OBS_DIMS["pickplace"]
    assert tk.make_task("pickplace", A1X).obs_dim == 42
    task = tk.make_task("pickplace", A1X)
    assert (task.episode_steps, task.end_on_success, task.cube, task.target) == (300, 10, "cube", "target")
    assert task.target_radius == A1X.desk.cube_radius and task.target_bearing == A1X.desk.cube_bearing
    env = TwinEnv("pickplace", workspace=_workspace("a1x"), randomization=Randomization.none())
    try:
        obs, _ = env.reset(seed=0, options={"target": [0.3, -0.1, 0.5]})
        assert obs.shape == (42,)
        n = task.n_joints
        cube = env.state()["cube_pos"]
        k = 2 * n + 3 + 12                                   # po q, q_cmd, TCP, kostce, kostka-TCP, obrocie
        assert np.allclose(obs[k:k + 3], [0.3, -0.1, 0.0], atol=1e-6)       # z celu zawsze 0 (blat)
        assert np.allclose(obs[k + 3:k + 6], np.array([0.3, -0.1, 0.0]) - cube, atol=1e-6)
    finally:
        env.close()


def test_pickplace_is_not_solved_by_pushing():
    """Kostka przesunieta po blacie na cel, szczeki otwarte: lezy na celu, ale to nie sukces."""
    env = TwinEnv("pickplace", randomization=Randomization.none())
    try:
        env.reset(seed=0, options={"cube": ([0.2, -0.08, 0.015], [1.0, 0, 0, 0]), "target": [0.2, 0.08, 0.0]})
        env.set_cube(np.array([0.2, 0.08, 0.015]), np.array([1.0, 0, 0, 0]))
        mujoco.mj_forward(env.scene.model, env.scene.data)
        a = np.zeros(env.task.act_dim)
        a[env.limits.grip] = 1.0                              # chwytak otwarty
        for _ in range(20):
            _, r_push, _, _, info = env.step(a)
        assert info["placed"] and info["released"] and not info["carried"]
        assert not info["success"]
        # Ta sama poza, ale po niesieniu - sukces i wyraznie wieksza nagroda.
        env._carried = True
        _, r_carried, _, _, info = env.step(a)
        assert info["success"] and r_carried - r_push > 8.0
    finally:
        env.close()


def test_pickplace_target_is_at_least_min_sep_from_the_cube():
    env = TwinEnv("pickplace", randomization=Randomization())
    try:
        t = env.task
        for seed in range(200):
            env.reset(seed=seed)
            s = env.state()
            sep = float(np.linalg.norm(s["target"][:2] - s["cube_pos"][:2]))
            r, b = np.hypot(*s["target"][:2]), np.arctan2(s["target"][1], s["target"][0])
            assert sep > t.target_min_sep and s["target"][2] == 0.0, f"seed {seed}: {sep * 1000:.0f} mm"
            assert t.target_radius[0] <= r <= t.target_radius[1] and t.target_bearing[0] <= b <= t.target_bearing[1]
            pad = env.scene.data.mocap_pos[env.target_mocap]
            T = env.scene.T_base2world
            assert np.allclose(pad, T[:3, :3] @ np.r_[s["target"][:2], 0.002] + T[:3, 3])
    finally:
        env.close()
    # Pierscien za waski na odstep: po `TARGET_TRIES` probach cel po drugiej stronie pierscienia.
    tight = tk.make_task("pickplace", target_radius=(0.2, 0.2), target_bearing=(-0.3, 0.3), target_min_sep=0.13)
    cubes = np.array([[0.2 * np.cos(0.25), 0.2 * np.sin(0.25), 0.015], [0.2 * np.cos(-0.25), 0.2 * np.sin(-0.25), 0.015]])
    got = tk.sample_targets(tight, np.random.default_rng(0), cubes)
    assert np.allclose(np.arctan2(got[:, 1], got[:, 0]), [-0.3, 0.3])
    assert np.all(np.hypot(*(got[:, :2] - cubes[:, :2]).T) > 0.1)


def test_target_pad_does_not_collide():
    """Kostka upuszczona na podkladke celu lezy na blacie (z = polowa boku) - `height` znaczy to co w lift."""
    env = TwinEnv("pickplace", randomization=Randomization.none())
    try:
        env.reset(seed=0, options={"target": [0.2, 0.1, 0.0]})
        env.set_cube(np.array([0.2, 0.1, 0.06]), np.array([1.0, 0, 0, 0]))
        mujoco.mj_forward(env.scene.model, env.scene.data)
        for _ in range(30):
            _, _, _, _, info = env.step(np.zeros(env.task.act_dim))
        cube = env.state()["cube_pos"]
        assert cube[2] == pytest.approx(env.task.cube_half, abs=5e-4)
        assert info["dist_target"] < 1e-3 and abs(info["height"]) < 5e-4
        m = env.scene.model
        g = m.geom("target").id
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and m.body_mocapid[m.body("target").id] >= 0
    finally:
        env.close()


def test_randomized_cube_stays_on_the_table_under_random_actions():
    """Losowe akcje wciskaja kostke w blat; kiedys wylatywala z predkoscia 10-180 m/s,
    bo zmiana masy nie przeliczala stalych kontaktu (`mj_setConst`)."""
    env = TwinEnv("lift", randomization=Randomization())
    d = env.scene.data
    try:
        for ep in range(12):
            env.reset(seed=100 + ep)
            for _ in range(env.task.episode_steps):
                _, _, term, trunc, info = env.step(env.action_space.sample())
                v = float(np.linalg.norm(d.qvel[env.cube_dadr:env.cube_dadr + 3]))
                assert v < 3.0 and abs(info["height"]) < 0.5, f"epizod {ep}: kostka {v:.1f} m/s"
                if term or trunc:
                    break
    finally:
        env.close()


def test_randomization_is_relative_to_the_nominal_model():
    env = TwinEnv("lift", randomization=Randomization())
    try:
        m = env.scene.model
        kp0 = env.fields.kp.copy()
        for seed in range(5):
            env.reset(seed=seed)
        env.fields.apply(m, env.scene.data, {k: 1.0 for k in ("kp", "damping", "armature", "frictionloss",
                                                              "cube_mass", "cube_friction")})
        assert np.allclose(m.actuator_gainprm[env.scene.act_ids, 0], kp0)
        assert m.body_mass[env.fields.cube_body] == pytest.approx(env.fields.cube_mass)
    finally:
        env.close()


@pytest.mark.parametrize("name", ["reach", "lift", "pickplace"])
def test_observation_and_reward_are_the_same_in_numpy_and_torch(name):
    """Jedna definicja zadania dla CPU, GPU i ramienia - wzory nie moga sie rozjechac."""
    task = tk.make_task(name)
    env = TwinEnv(task, randomization=Randomization.none())
    lim = env.limits
    env.close()
    rng = np.random.default_rng(0)
    n = 16
    q = rng.uniform(lim.lo, lim.hi, (n, task.act_dim))
    q_cmd = rng.uniform(lim.lo, lim.hi, (n, task.act_dim))
    tcp, goal, cube = rng.normal(0, 0.2, (n, 3)), rng.normal(0, 0.2, (n, 3)), rng.normal(0, 0.2, (n, 3))
    rot = np.array([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(n)])
    act, prev = rng.uniform(-1, 1, (n, task.act_dim)), rng.uniform(-1, 1, (n, task.act_dim))
    jaws = rng.integers(0, 2, (n, 2)).astype(float)

    task = tk.make_task(name, limit_penalty=0.5, hold_still=0.5, end_on_success=2, roll_penalty=0.3)
    cube[: n // 2, 2] = 0.09                                   # polowa "podniesiona" - sukces i kara za ruch
    streak = rng.integers(0, 3, n)
    # pickplace: cel, predkosc kostki, zatrzask i otwarty chwytak - tak, zeby czesc swiatow miala sukces.
    # Dane reach/lift zostaja takie jak przed pickplace.
    target = np.c_[rng.normal(0, 0.2, (n, 2)), np.zeros(n)]
    vel = rng.normal(0, 0.02, (n, 3))
    carried = rng.integers(0, 2, n).astype(bool)
    opened = rng.integers(0, 2, n).astype(bool)
    if name == "pickplace":
        cube[n // 2:, 2] = task.cube_half + rng.uniform(-0.002, 0.004, n - n // 2)
        cube[n // 2::2, :2] = target[n // 2::2, :2] + rng.normal(0, 0.01, (len(cube[n // 2::2]), 2))
        ok = slice(n // 2, n, 4)                               # tu wszystkie warunki sukcesu naraz
        vel[ok], jaws[ok], carried[ok], opened[ok] = 0.001, 0.0, True, True

    def run(xp, conv):
        b = (lambda x: x) if xp is np else torch.as_tensor
        obs = tk.observe(xp, task, lim, conv(q), conv(q_cmd), conv(tcp), conv(prev), goal=conv(goal),
                         cube_pos=conv(cube), cube_rot=conv(rot), target=conv(target))
        r, s, f = tk.reward(xp, task, conv(tcp), conv(act), conv(prev), goal=conv(goal), cube_pos=conv(cube),
                            jaw_contacts=conv(jaws), q_cmd=conv(q_cmd), limits=lim, target=conv(target),
                            cube_vel=conv(vel), carried=b(carried), grip_open=b(opened))
        nxt = tk.apply_action(xp, task, lim, conv(q_cmd), conv(act))
        st, done = tk.success_streak(xp, task, streak if xp is np else torch.as_tensor(streak), s)
        latch = tk.carry_latch(xp, task, b(carried), conv(jaws), conv(cube))
        grip = tk.grip_is_open(xp, lim, conv(q_cmd))
        return [np.asarray(x) for x in (obs, r, s, f, nxt, st, done, latch, grip)]

    a = run(np, lambda x: x)
    b = run(torch, lambda x: torch.as_tensor(x, dtype=torch.float64))
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-9)
    assert a[0].shape == (n, task.obs_dim)
    if name == "pickplace":
        assert 0 < a[2].sum() < n                              # sukces w czesci swiatow - obie galezie liczone


def test_actions_respect_the_supervisor_limits():
    env = TwinEnv("reach", randomization=Randomization.none())
    try:
        env.reset(seed=0)
        for _ in range(60):
            env.step(np.ones(env.task.act_dim))
        assert np.all(env.q_cmd <= env.limits.hi + 1e-9)
        # chwytak: najwyzej `grip_step` zakresu na takt
        env.reset(seed=0)
        g0 = env.q_cmd[5]
        env.step(-np.ones(env.task.act_dim))
        span = env.limits.hi[5] - env.limits.lo[5]
        assert g0 - env.q_cmd[5] == pytest.approx(min(env.task.grip_step * span, g0 - env.limits.lo[5]))
    finally:
        env.close()


def test_policy_file_roundtrip(tmp_path):
    task = tk.make_task("reach")
    from dataclasses import asdict

    pol = Policy(PolicyMeta(task=asdict(task), hidden=[32, 32], obs_dim=task.obs_dim, act_dim=task.act_dim))
    pol.norm.update(torch.randn(100, task.obs_dim) * 3 + 1)
    obs = np.random.default_rng(0).normal(size=(4, task.obs_dim)).astype(np.float32)
    path = pol.save(tmp_path / "p" / "policy.pt")
    back = Policy.load(path)
    assert np.allclose(pol.act(obs), back.act(obs))
    assert back.task == task
    assert (tmp_path / "p" / "meta.json").is_file()


def test_randomization_around_measured_dynamics_centres_on_it():
    dyn = Dynamics(kp=1.3, damping=0.7, delay=1.2, source="pomiar")
    r = Randomization.around(dyn, spread=0.5)
    s = r.sample(np.random.default_rng(0), 2000)
    assert np.median(s["kp"]) == pytest.approx(1.3, rel=0.02)
    assert r.max_delay == 3
    assert s["kp"].min() >= 1.3 * r.kp[0] - 1e-9


def test_randomization_widens_what_identification_did_not_fit():
    """kp i tarcie z modelu (nie z pomiaru) - po identyfikacji szeroko, bo reszta wyniku jest
    dobra tylko razem z ich PRAWDZIWA wartoscia. Stare kp 0,85-1,15 nie zawieralo kp 0,7-1,3."""
    from lerobot_mp.twin.rl.randomize import UNFITTED_RANGES

    default = Randomization()
    menagerie = Randomization.around(Dynamics())               # nic nie mierzono - zakresy domyslne
    assert menagerie.kp == default.kp and menagerie.frictionloss == default.frictionloss
    fitted = Dynamics(damping=1.3, armature=1.1, delay=1.1, source="identyfikacja",
                      fitted=("damping", "armature", "delay"))
    r = Randomization.around(fitted)
    assert r.kp == UNFITTED_RANGES["kp"] and r.frictionloss == UNFITTED_RANGES["frictionloss"]
    assert r.kp[0] <= 0.7 and r.kp[1] >= 1.3 and r.frictionloss[0] <= 0.6 and r.frictionloss[1] >= 1.4
    assert r.damping == default.damping and r.armature == default.armature
    half = Randomization.around(fitted, spread=0.5)            # `spread` skaluje tez szerokie zakresy
    assert half.kp == pytest.approx((0.8, 1.25))
    assert not Randomization.around(fitted, spread=0.0).randomized
    both = Randomization.around(Dynamics(kp=0.9, fitted=("kp", "damping")))   # kp zmierzone - zwykly zakres
    assert both.kp == default.kp and both.frictionloss == UNFITTED_RANGES["frictionloss"]


def test_randomization_widens_fitted_ranges_to_the_measured_band():
    """Pasmo z identyfikacji (`Dynamics.band`) poszerza zakres dopasowanego parametru.

    Zmierzone przed poprawka: `around` nie czytal pasma - tlumienie z pasmem +-35 % dostawalo
    stale 0,75-1,3, a armatura x0,45 z pasmem +-inf (s6 weryfikatora) zwykle +-25 %.
    """
    from lerobot_mp.twin.rl.randomize import MAX_BAND, MAX_DELAY_BAND

    default = Randomization()
    dyn = Dynamics(damping=1.2, armature=0.45, delay=1.0, source="identyfikacja",
                   fitted=("damping", "armature", "delay"),
                   band={"damping": 0.35, "armature": float("inf"), "delay": 1.4, "kp": 0.9})
    r = Randomization.around(dyn)
    assert r.damping == pytest.approx((0.65, 1.35))
    assert r.armature == pytest.approx((1 - MAX_BAND, 1 + MAX_BAND))       # inf - przyciete, nie <= 0
    assert r.kp == (0.6, 1.5)                                               # niedopasowane: pasmo nie liczy sie
    assert r.max_delay == 3 and r.min_delay == 0                            # ceil(1,0 + 1,4)
    narrow = Randomization.around(Dynamics(damping=1.1, fitted=("damping", "delay"), band={"damping": 0.05}))
    assert narrow.damping == default.damping                                # waskie pasmo nie zweza zakresu
    half = Randomization.around(dyn, spread=0.5)
    assert half.damping == pytest.approx((1 - 0.35 / 2, 1 + 0.35 / 2))
    assert not Randomization.around(dyn, spread=0.0).randomized
    huge = Randomization.around(Dynamics(delay=1.0, fitted=("delay",), band={"delay": float("inf")}))
    assert huge.max_delay == int(np.ceil(1.0 + MAX_DELAY_BAND))
    assert Randomization.around(Dynamics(delay=1.0, fitted=("delay",), band={"delay": 0.1})).max_delay == 2


def test_dynamics_band_survives_strict_json():
    """Pasmo z inf przezywa zapis workspace'u - takze scisly JSON (bez Infinity)."""
    import json

    dyn = Dynamics(damping=1.2, fitted=("damping", "armature"), band={"damping": 0.2, "armature": float("inf")})
    text = json.dumps(dyn.to_dict(), allow_nan=False)
    back = Dynamics.from_dict(json.loads(text))
    assert back == dyn and np.isinf(back.band["armature"])
    old = dyn.to_dict()
    old.pop("band")
    assert Dynamics.from_dict(old).band == {}


def test_no_randomization_keeps_the_measured_arm_and_the_camera_model():
    """--no-rand = zmierzona dynamika bez rozrzutu (a nie Menagerie), percepcja kostki jak z kamer."""
    dyn = Dynamics(kp=0.8, damping=1.4, delay=1.6, source="identyfikacja")
    r = Randomization.around(dyn, spread=0.0)
    s = r.sample(np.random.default_rng(0), 500)
    assert np.allclose(s["kp"], 0.8) and np.allclose(s["damping"], 1.4) and np.all(s["delay"] == 2)
    assert np.allclose(s["cube_mass"], 1.0)
    assert r.centre.source == "identyfikacja" and not r.randomized
    assert r.fold_yaw and r.cube_delay > 0 and r.cube_period > 1
    nom = Randomization.nominal(dyn)
    assert not nom.randomized and nom.centre is dyn and nom.min_delay == nom.max_delay == 2
    assert not nom.fold_yaw and nom.cube_delay == 0 and nom.obs_noise_deg == 0.0
    assert Randomization.around(dyn, 1.0).randomized and Randomization().randomized
    assert not Randomization.none().randomized


def test_train_plan_does_not_label_a_nominal_run_as_randomized():
    from lerobot_mp.twin.cli import _train_plan, parser

    dyn = Dynamics(kp=0.8, delay=1.2, source="identyfikacja")
    rand, evals = _train_plan(dyn, no_rand=True, spread=1.0)
    assert rand.centre is dyn and [e[1] for e in evals] == ["cpu_nominal"]
    assert evals[0][2].centre is dyn
    rand, evals = _train_plan(dyn, no_rand=False, spread=1.0)
    assert [e[1] for e in evals] == ["cpu_nominal", "cpu_rand"] and evals[1][2] is rand and rand.randomized
    assert parser().parse_args(["train", "--task", "pickplace"]).task == "pickplace"
    # panel ruszajacy ramieniem nie wystawia sie domyslnie na siec
    assert parser().parse_args(["ui"]).host == "127.0.0.1"
    assert parser().parse_args(["ui", "--host", "0.0.0.0"]).host == "0.0.0.0"


def test_nominal_evaluation_runs_on_the_identified_arm():
    from dataclasses import asdict

    from lerobot_mp.twin.rl.evaluate import evaluate
    from lerobot_mp.twin.workspace import Workspace

    task = tk.make_task("reach", episode_steps=5)
    pol = Policy(PolicyMeta(task=asdict(task), hidden=[8], obs_dim=task.obs_dim, act_dim=task.act_dim))
    ws = Workspace()
    ws.dynamics = Dynamics(kp=0.8, delay=1.0, source="identyfikacja").to_dict()
    res = evaluate(pol, 1, workspace=ws)
    assert res["centre"] == "identyfikacja" and res["randomized"] is False
    res = evaluate(pol, 1, randomization=Randomization.none(), workspace=ws)
    assert res["centre"] == "menagerie" and res["randomized"] is False


def test_lift_episode_ends_after_the_success_streak(monkeypatch):
    """Po sukcesie lift-v2 wymachiwal kostka do limitow przez 170 taktow - teraz epizod sie konczy."""
    real = tk.reward

    def always_success(xp, task, *a, **kw):
        r, s, f = real(xp, task, *a, **kw)
        return r, s | True, f

    monkeypatch.setattr(tk, "reward", always_success)
    task = tk.make_task("lift")
    assert task.end_on_success == 10 and tk.make_task("reach").end_on_success == 0
    env = TwinEnv(task, randomization=Randomization.none())
    try:
        env.reset(seed=0)
        for k in range(1, 11):
            _, _, term, trunc, info = env.step(np.zeros(task.act_dim))
            assert not term and trunc == (k == 10) and info["finished"] == (k == 10)
    finally:
        env.close()
    env = TwinEnv(tk.make_task("reach", episode_steps=15), randomization=Randomization.none())
    try:
        env.reset(seed=0)
        ends = [env.step(np.zeros(env.task.act_dim))[3] for _ in range(15)]
        assert ends == [False] * 14 + [True]                  # reach: tylko limit czasu
    finally:
        env.close()


def test_old_lift_policy_file_stops_on_success_too():
    from dataclasses import asdict

    old = asdict(tk.make_task("lift"))
    for k in ("end_on_success", "limit_penalty", "limit_margin", "hold_still"):
        old.pop(k)
    meta = PolicyMeta(task=old, hidden=[8], obs_dim=tk.OBS_DIMS["lift"], act_dim=6)
    assert meta.task_config().end_on_success == 10


def test_reach_reward_keeps_the_wrist_roll_near_home():
    """Obrot nadgarstka nie zmienia TCP - bez kary reach-v2 krecila nim do limitu +150 st."""
    task = tk.make_task("reach")
    env = TwinEnv(task, randomization=Randomization.none())
    lim = env.limits
    env.close()
    tcp = goal = np.zeros((2, 3))
    act = prev = np.zeros((2, task.act_dim))
    q = np.repeat(lim.home[None], 2, axis=0)
    q[1, 4] = np.radians(150.0)
    r, s, _ = tk.reward(np, task, tcp, act, prev, goal=goal, q_cmd=q, limits=lim)
    assert s.all() and r[0] - r[1] > 0.1                      # 150 st. kosztuje wyraznie
    q[1, 4] = lim.home[4] + np.radians(20.0)
    r, _, _ = tk.reward(np, task, tcp, act, prev, goal=goal, q_cmd=q, limits=lim)
    assert r[0] - r[1] < 0.01                                 # 20 st. prawie nic


def test_lift_reward_pushes_away_from_joint_limits_and_holds_still_after_success():
    task = tk.make_task("lift")
    env = TwinEnv("reach", randomization=Randomization.none())
    lim = env.limits
    env.close()
    tcp = np.array([[0.2, 0.0, 0.1]])
    cube = tcp.copy()
    jaws = np.ones((1, 2))
    a0 = np.zeros((1, task.act_dim))

    def r(q_cmd, action=a0, cube_pos=cube):
        return tk.reward(np, task, tcp, action, action, cube_pos=cube_pos, jaw_contacts=jaws,
                         q_cmd=q_cmd[None], limits=lim)[0][0]

    home = lim.home.copy()
    at_limits = home.copy()
    at_limits[[0, 2, 3, 4]] = [lim.lo[0], lim.lo[2], lim.hi[3], lim.lo[4]]     # poza z wymachu lift-v2
    assert r(home) - r(at_limits) == pytest.approx(4 * task.limit_penalty)
    near = home.copy()
    near[0] = lim.lo[0] + task.limit_margin / 2
    assert 0 < r(home) - r(near) < task.limit_penalty
    moving = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]])
    assert r(home) - r(home, moving) == pytest.approx(2 * task.hold_still)           # kostka w gorze
    low = cube.copy()
    low[0, 2] = 0.03
    assert r(home, moving, low) == pytest.approx(r(home, a0, low))                   # jeszcze nie sukces


def test_observed_tcp_matches_the_joint_angles():
    """Runner liczy TCP z FK katow; w treningu TCP bylo sprzed ostatniego kroku fizyki (4-8 mm w ruchu)."""
    env = TwinEnv("reach", randomization=Randomization.none())
    try:
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(40):
            env.step(np.sign(rng.uniform(-1, 1, env.task.act_dim)))
            s = env.state()
            fk = env.kin.tcp(env.kin.from_q(s["q"]))[:3, 3]
            worst = max(worst, float(np.linalg.norm(fk - s["tcp"])))
        assert worst < 1e-4, f"TCP {worst * 1000:.2f} mm od FK(q)"
    finally:
        env.close()


def test_policy_sees_the_cube_as_the_cameras_would():
    """Obserwacja kostki spozniona i ze zlozonym obrotem - nagroda dalej z prawdy."""
    from dataclasses import replace

    rand = replace(Randomization.none(), cube_delay=2, cube_period=1, fold_yaw=True)
    env = TwinEnv("lift", randomization=rand)
    try:
        env.reset(seed=0)
        env._cube_delay = 2                          # losowane z {0..2}; tu wymuszamy najgorsze
        start = env.state()["cube_pos"].copy()
        moved = start + np.array([0.03, 0.0, 0.0])
        env.set_cube(moved, np.array([np.cos(0.8), 0, 0, np.sin(0.8)]))   # obrot 91,7 st.
        mujoco.mj_forward(env.scene.model, env.scene.data)
        seen = []
        for _ in range(3):
            obs, *_ = env.step(np.zeros(env.task.act_dim))
            seen.append(obs[15:18].copy())
        assert np.allclose(seen[0], start, atol=2e-3)           # jeszcze stara poza (opoznienie)
        assert np.allclose(seen[2], env.state()["cube_pos"], atol=2e-3)
        yaw = np.arctan2(obs[22], obs[21])
        assert abs(np.degrees(yaw)) <= 45.0 + 1e-6               # 91,7 st. widziane jak 1,7 st.
    finally:
        env.close()


@cuda
def test_gpu_batch_env_matches_the_cpu_env():
    """Ta sama scena, start i akcje: MuJoCo Warp (float32) i MuJoCo CPU daja te same obserwacje."""
    from lerobot_mp.twin.rl.batch import BatchEnv

    cpu = TwinEnv("reach", randomization=Randomization.none())
    cpu.reset(seed=3)
    gpu = BatchEnv("reach", num_envs=2, randomization=Randomization.none())
    gpu.reset()
    dev = gpu.device
    gpu.qpos[:] = torch.as_tensor(cpu.scene.data.qpos, dtype=torch.float32, device=dev)
    gpu.qvel[:] = 0
    q_cmd = torch.as_tensor(cpu.q_cmd, dtype=torch.float32, device=dev)
    gpu.ctrl[:, gpu.act] = q_cmd
    gpu.q_cmd[:] = q_cmd
    gpu.goal[:] = torch.as_tensor(cpu.goal, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(-1, 1, cpu.task.act_dim)
        oc, rc, *_ = cpu.step(a)
        og, rg, _, _ = gpu.step(torch.as_tensor(a, device=dev).repeat(2, 1))
        assert np.abs(og[0].cpu().numpy() - oc).max() < 1e-3
        assert float(rg[0]) == pytest.approx(rc, abs=1e-3)
    cpu.close()


@cuda
def test_gpu_pickplace_matches_the_cpu_env():
    """Ten sam start, kostka, cel i akcje (ekspert ze skryptu): te same obserwacje - razem z celem
    odlozenia - i ten sam zatrzask "niesiona" w MuJoCo Warp (float32) i MuJoCo CPU."""
    from lerobot_mp.twin.rl.batch import BatchEnv

    rec = TwinEnv("pickplace", randomization=Randomization.none())
    try:
        _, _, first, actions = scripted_pickplace(rec, 0)
    finally:
        rec.close()
    assert first is not None
    cpu = TwinEnv("pickplace", randomization=Randomization.none())
    cpu.reset(seed=0)
    gpu = BatchEnv("pickplace", num_envs=2, randomization=Randomization.none())
    gpu.reset()
    dev, f32 = gpu.device, torch.float32
    gpu.qpos[:] = torch.as_tensor(cpu.scene.data.qpos, dtype=f32, device=dev)
    gpu.qvel[:] = 0
    q_cmd = torch.as_tensor(cpu.q_cmd, dtype=f32, device=dev)
    gpu.ctrl[:, gpu.act] = q_cmd
    gpu.q_cmd[:] = q_cmd
    gpu.set_targets(torch.arange(2, device=dev), torch.as_tensor(cpu.target, dtype=f32, device=dev).repeat(2, 1))
    pos, rot = (torch.as_tensor(x, dtype=f32, device=dev) for x in cpu._cube_seen)
    gpu.hist_pos[:], gpu.hist_rot[:], gpu.seen_pos[:], gpu.seen_rot[:] = pos, rot, pos, rot
    gpu.carried[:] = False
    torch.cuda.synchronize(dev)
    gpu.mjw.kinematics(gpu.m, gpu.d)
    gpu.wp.synchronize_device()
    k = 2 * cpu.task.n_joints + 3 + 12                        # poczatek celu w obserwacji
    carried = []
    for a in actions:
        oc, _, term, trunc, ic = cpu.step(a)
        og, _, done, ig = gpu.step(torch.as_tensor(a, dtype=f32, device=dev).repeat(2, 1))
        if bool(done[0]) or term or trunc:                   # GPU zresetowal swiat - dalej inny epizod
            break
        og = og[0].cpu().numpy()
        assert np.abs(og - oc).max() < 5e-3
        assert np.allclose(og[k:k + 3], cpu.target, atol=1e-6) and cpu.target[2] == 0.0
        assert bool(ig["carried"][0]) == ic["carried"]
        carried.append(ic["carried"])
    assert any(carried)                                       # zatrzask naprawde sie ustawil


@cuda
def test_fine_tuning_waits_for_the_critic_before_moving_the_actor(tmp_path):
    """Douczanie z krytykiem od zera psulo reach-v1 (98% -> 9%): aktor ma stac, dopoki
    krytyk sie nie rozgrzeje, a polityka zapisana z krytykiem rozgrzewki nie potrzebuje."""
    from dataclasses import replace

    from lerobot_mp.twin.rl.ppo import PPOConfig, train

    cfg = PPOConfig(num_envs=64, horizon=8, iterations=2, hidden=(32, 32))
    train("reach", cfg, randomization=Randomization.none(), out_dir=tmp_path / "a")
    base = Policy.load(tmp_path / "a" / "policy.pt")
    assert base.critic_state is not None                  # krytyk jedzie w pliku do douczania

    obs = np.random.default_rng(0).normal(size=(16, base.meta.obs_dim)).astype(np.float32)

    def actor_moved(init):
        before = init.act(obs)
        tuned = train("reach", replace(cfg, critic_warmup=2), randomization=Randomization.none(), init=init)
        return not np.array_equal(before, tuned.to("cpu").act(obs))

    assert actor_moved(base)                              # krytyk z pliku - aktor uczy sie od razu
    base.critic_state = None                              # jak polityka sprzed zapisu krytyka
    assert not actor_moved(base)


@cuda
def test_gpu_applies_the_same_multi_tick_action_delay_as_the_cpu():
    """Opoznienie z identyfikacji (np. 2 takty) - GPU scinalo je do 1 taktu."""
    from dataclasses import replace

    from lerobot_mp.twin.rl.batch import BatchEnv

    rand = replace(Randomization.none(), min_delay=2, max_delay=2)
    cpu = TwinEnv("reach", randomization=rand)
    cpu.reset(seed=3)
    assert len(cpu._delay) == 2
    gpu = BatchEnv("reach", num_envs=2, randomization=rand)
    gpu.reset()
    assert gpu.delay_n.tolist() == [2, 2]
    dev = gpu.device
    gpu.qpos[:] = torch.as_tensor(cpu.scene.data.qpos, dtype=torch.float32, device=dev)
    gpu.qvel[:] = 0
    q_cmd = torch.as_tensor(cpu.q_cmd, dtype=torch.float32, device=dev)
    gpu.ctrl[:, gpu.act] = q_cmd
    gpu.q_cmd[:] = q_cmd
    gpu.goal[:] = torch.as_tensor(cpu.goal, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(1)
    for _ in range(12):
        a = rng.uniform(-1, 1, cpu.task.act_dim)
        oc, *_ = cpu.step(a)
        og, *_ = gpu.step(torch.as_tensor(a, device=dev).repeat(2, 1))
        assert np.abs(gpu.q_cmd[0].cpu().numpy() - cpu.q_cmd).max() < 1e-5
        assert np.abs(og[0].cpu().numpy() - oc).max() < 1e-3
    cpu.close()


@cuda
def test_gpu_ends_episodes_after_the_success_streak(monkeypatch):
    from lerobot_mp.twin.rl.batch import BatchEnv

    real = tk.reward

    def always_success(xp, task, *a, **kw):
        r, s, f = real(xp, task, *a, **kw)
        return r, s | True, f

    monkeypatch.setattr(tk, "reward", always_success)
    gpu = BatchEnv(tk.make_task("reach", end_on_success=3), num_envs=4, randomization=Randomization.none())
    gpu.reset()
    for k in range(1, 4):
        _, _, done, info = gpu.step(torch.zeros(4, gpu.task.act_dim, device=gpu.device))
        assert bool(done.all()) == (k == 3) and bool(info["finished"].all()) == (k == 3)
    assert bool(info["time_outs"].all())                     # PPO dolicza wartosc stanu
    assert gpu.t.tolist() == [0] * 4 and gpu.streak.tolist() == [0] * 4


def test_runner_drives_the_simulated_arm_through_the_supervisor():
    """Polityka na zywym blizniaku (sim): ruch idzie przez nadzor, a epizod konczy sie limitem."""
    import time
    from dataclasses import asdict

    from lerobot_mp.twin.rl.runner import PolicyRunner
    from lerobot_mp.twin.runtime import Twin
    from lerobot_mp.twin.workspace import Workspace

    task = tk.make_task("reach", episode_steps=12)
    pol = Policy(PolicyMeta(task=asdict(task), hidden=[8], obs_dim=task.obs_dim, act_dim=task.act_dim))
    with torch.no_grad():
        for p in pol.actor.parameters():
            p.zero_()
        pol.actor[-1].bias[0] = 1.0                     # stale "obracaj podstawe w lewo"
    twin = Twin(Workspace())
    try:
        twin.connect("sim")
        start = twin.joints()["shoulder_pan"]
        runner = PolicyRunner(twin, pol)
        runner.start()
        deadline = time.monotonic() + 5.0
        while runner.status.running and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not runner.status.running
        assert runner.status.stopped_because == "koniec epizodu"
        assert twin.joints()["shoulder_pan"] > start + 5.0
    finally:
        twin.close()


def test_policy_metadata_stays_strict_json_with_an_infinite_band():
    """Dopasowanie bez niepewnosci (band inf) szlo do meta.json jako Infinity - panel go nie czytal."""
    import json

    from lerobot_mp.twin.rl.ppo import _rand_dict

    dyn = Dynamics(damping=1.2, fitted=("damping",), band={"damping": float("inf")}, source="identyfikacja")
    json.dumps(_rand_dict(Randomization.around(dyn)), allow_nan=False)


def test_obs_and_act_dims_are_unchanged_for_so101():
    from lerobot_mp.twin.robots import SO101

    reach, lift = tk.make_task("reach"), tk.make_task("lift")
    assert (reach.obs_dim, lift.obs_dim, reach.act_dim, lift.act_dim) == (27, 33, 6, 6)
    assert tk.OBS_DIMS == {"reach": 27, "lift": 33, "pickplace": 39}
    assert tk.make_task("lift", SO101).obs_dim == 33
    assert tk.obs_dim("lift", 7) == 36 and tk.obs_dim("reach", 7) == 30


def test_make_task_with_so101_spec_equals_the_defaults():
    from lerobot_mp.twin.robots import SO101

    for name in tk.TASKS:
        assert tk.make_task(name, SO101) == tk.make_task(name)
        assert tk.make_task(name, SO101, episode_steps=7) == tk.make_task(name, episode_steps=7)


def test_make_task_takes_the_joint_count_and_desk_from_the_spec(toy_spec):
    from dataclasses import replace

    from lerobot_mp.twin.robots import Desk

    toy = replace(toy_spec, desk=Desk(goal_radius=(0.1, 0.2), cube_radius=(0.12, 0.18)))
    task = tk.make_task("lift", toy)
    assert task.n_joints == 3 and task.act_dim == 3 and task.obs_dim == tk.obs_dim("lift", 3)
    assert task.goal_radius == (0.1, 0.2) and task.cube_radius == (0.12, 0.18)
    assert tk.make_task("lift", toy, cube_radius=(0.1, 0.11)).cube_radius == (0.1, 0.11)   # nadpisanie wygrywa


def test_limits_expose_arm_count_gripper_and_roll_indices(toy_spec):
    from lerobot_mp.twin.kinematics import RobotKinematics
    from lerobot_mp.twin.robots import SO101

    lim = tk.Limits.of(RobotKinematics(SO101))
    assert (lim.n_arm, lim.grip, lim.roll) == (5, 5, 4)
    toy = tk.Limits.of(RobotKinematics(toy_spec))
    assert (toy.n_arm, toy.grip, toy.roll) == (2, 2, None)
    assert toy.hi[2] == pytest.approx(0.02) and toy.lo[2] == pytest.approx(0.0)
    task = tk.make_task("reach", toy_spec, roll_penalty=0.5)
    q = toy.home[None].copy()
    r, _, _ = tk.reward(np, task, np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), goal=np.zeros((1, 3)),
                        q_cmd=q, limits=toy)                      # bez stawu obrotu - bez kary za obrot
    nxt = tk.apply_action(np, task, toy, q, np.ones((1, 3)))
    assert nxt.shape == (1, 3) and nxt[0, 0] == pytest.approx(min(q[0, 0] + task.arm_step, toy.hi[0]))
    assert np.isfinite(r).all()


def test_old_policy_meta_without_n_joints_and_robot_loads(tmp_path):
    """Pliki polityk sprzed `n_joints`/`robot` to SO-101: 6 stawow, obserwacja jak dotad."""
    from dataclasses import asdict

    old_task = asdict(tk.make_task("lift"))
    old_task.pop("n_joints")
    blob = {"task": old_task, "hidden": [8], "obs_dim": 33, "act_dim": 6}
    meta = PolicyMeta(**blob)
    assert meta.robot == "so101"
    assert meta.task_config().n_joints == 6 and meta.task_config() == tk.make_task("lift")
    pol = Policy(meta)
    path = pol.save(tmp_path / "old" / "policy.pt")
    raw = torch.load(path, weights_only=False)
    raw["meta"].pop("robot")
    torch.save(raw, path)
    back = Policy.load(path)
    assert back.meta.robot == "so101" and back.task.act_dim == 6 and back.task.obs_dim == 33
