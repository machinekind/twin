"""Mrowka jezdzi na wlasnej polityce, blizniak ma jej kopie: WSAD przez przemianowanie nog, kopia z serw i tagow.

Mrowka udajaca prawdziwa (`antprawda/`) chodzi w fizyce sceny blizniaka na polityce chodu realant-rl - jak
`robot_teleop.py` na prawdziwej. WSAD nie zmienia polityki - te same wagi jada w czterech kierunkach, bo nogi sa
przemianowane, a obserwacja obrocona. Kopia blizniaka (`ant/`, schowana przed kamerami) dostaje tylko to, co
dostalaby od prawdziwej: odczyty serw (zaokraglone do jednostek AX-12A) i kadry kamer z tagami - i z tego
dochodzi do tej samej pozy, a montaz taga kalibruje sie sam. Lacze ZeroMQ: tylko nasluch, na udawanym antproxy.
"""

from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin import antwalk as aw  # noqa: E402
from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin.ant import (DEMO_MOUNT, LEG_POSES, Ant, add_demo_ant, demo_layout, rest_height,  # noqa: E402
                                 square_layout)
from lerobot_mp.twin.antdrive import (DEADMAN, FINE_HIP_AMP, FINE_STEPS, POSE_RAMP_STEPS, RAMP_STEPS,  # noqa: E402
                                      AntDrive, wave_actions)
from lerobot_mp.twin.antmirror import AntMirror, StandInServos, mirror_pose_error  # noqa: E402
from lerobot_mp.twin.scene import STANDIN_PREFIX as STANDIN  # noqa: E402
from lerobot_mp.twin.kinematics import pose  # noqa: E402
from lerobot_mp.twin.tagbody import learn_layout, observe  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402


def sim_workspace(xy=(0.45, 0.0)) -> Workspace:
    """SO-101 w symulacji, blat 1,8 x 1,4 m, mrowka stojaca na palcach, bez kamer."""
    ws = Workspace()
    ws.table = {**ws.table, "size": [1.8, 1.4]}
    joints = dict(LEG_POSES["stoi"])
    T = pose(np.eye(3), np.array([xy[0], xy[1], rest_height(joints) + 0.002]))
    ws.ant = Ant(joints=joints, sim={"T_ant2base": T, "mount": {}, "layout": {}}).to_dict()
    return ws


class Pad:
    """Pulpit w tescie: klawisz trzymany przez `seconds` (odswiezany co krok, jak strona co 0,1 s)."""

    def __init__(self, drive: AntDrive):
        self.drive, self.t = drive, 0.0
        drive.clock = lambda: self.t

    def hold(self, key: str, seconds: float, refresh: bool = True) -> None:
        for _ in range(int(round(seconds / aw.DT))):
            if refresh:
                self.drive.keys(key)
            self.drive.tick()
            self.t += aw.DT


def test_policy_file_units_and_same_weights_in_every_direction():
    pol = aw.load_policy()
    assert pol.obs_size == 184 and pol.cfg.get("mode", "walk") == "walk" and len(pol.layers) == 4
    a = pol.act(np.zeros(184))
    assert a.shape == (8,) and np.all(np.abs(a) <= 1)
    # Odczyt serw jak rollout_server.get_obs: s1..s8 -> stawy polityki (noga 1 = s3, s4), biodro 510 = 0,
    # kostka 511 = poziomo (0), 288 = w dol (zakres serw modelu).
    q = aw.units_to_joints([510, 511, 510, 288, 510, 511, 510, 511])
    assert np.allclose(q, [0.0, aw.ANKLE_CTRL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    for _ in range(20):                                    # w jednostke AX-12A (0,29 st.) i z powrotem
        q = aw.action_to_ctrl(rng.uniform(-1, 1, 8))
        assert np.allclose(aw.units_to_joints(aw.joints_to_units(q)), q, atol=0.006)
    # Kierunki: permutacje nog o 90 st., obrot obserwacji o ten sam kat.
    assert list(aw.leg_index("tyl")) == [4, 5, 6, 7, 0, 1, 2, 3]
    assert {d: round(np.degrees(t)) for d, (_, t) in aw.DIRECTIONS.items()} == {"przod": 0, "lewo": 90, "prawo": -90,
                                                                             "tyl": 180}


def test_wsad_walks_four_ways_with_one_policy_and_turns_in_place():
    tw = rt.Twin(sim_workspace())
    try:
        drive = AntDrive(tw, lambda: None, camera="prawda")
        pad = Pad(drive)
        pad.hold("", 1.5)                                  # wstaje do pozy startowej polityki
        assert drive.status.mode == "stoi"
        for key, want in (("w", (1, 0)), ("d", (0, -1)), ("s", (-1, 0)), ("a", (0, 1))):
            p0, h0 = tw.scene.ant_pose(STANDIN), np.degrees(aw.heading(tw.scene.ant_pose(STANDIN)))
            pad.hold(key, 2.5)
            assert drive.status.mode == "jazda" and drive.status.steps > 40
            pad.hold("", 1.0)
            d = tw.scene.ant_pose(STANDIN)[:3, 3] - p0[:3, 3]
            c, s = np.cos(np.radians(h0)), np.sin(np.radians(h0))
            local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])   # w ukladzie mrowki sprzed jazdy
            along, across = local @ np.array(want), local @ np.array([-want[1], want[0]])
            assert along > 0.17, (key, local)                   # 10 cm/s z treningu: ~0,23 m w 2,5 s
            assert abs(across) < 0.05, (key, local)
        assert drive.status.mode == "stoi" and tw.scene.ant_pose(STANDIN)[2, 2] > 0.95
        h0 = aw.heading(tw.scene.ant_pose(STANDIN))
        pad.hold("q", 2.0)
        turned = np.degrees(np.angle(np.exp(1j * (aw.heading(tw.scene.ant_pose(STANDIN)) - h0))))
        assert turned > 60.0, turned                        # obrot w lewo (otwarta petla z robot_route.py)
    finally:
        tw.close()


class FakeAnt:
    """Mrowka bez fizyki: pamieta rozkazy, stawy = ostatni rozkaz, poza = stoi prosto (do testow przejsc)."""

    kind = "sim"

    def __init__(self, start: np.ndarray):
        self.cmds: list[np.ndarray] = []
        self.q = aw.action_to_ctrl(start)
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def ready(self) -> str:
        return ""

    def startup(self) -> None:
        pass

    def command(self, a: np.ndarray) -> None:
        self.cmds.append(np.array(a, float))
        self.q = aw.action_to_ctrl(a)

    def advance(self) -> None:
        self.t += aw.DT

    def joints(self):
        return self.q.copy(), np.zeros(8)

    def truth(self) -> np.ndarray:
        return np.eye(4)

    def close(self) -> None:
        pass


def fake_drive():
    pol = aw.load_policy()
    io = FakeAnt(aw.start_action(pol.cfg))
    drive = AntDrive(None, lambda: None, io=io, policy=pol, camera="prawda")
    pad = Pad(drive)
    pad.hold("", 1.5)                                      # rampa 20 krokow do pozy startowej
    assert drive.status.mode == "stoi" and np.allclose(io.cmds[-1], drive.start_pose)
    io.cmds.clear()
    return drive, io, pad


def cycle_steps(cmds, cycle) -> int:
    return sum(any(np.allclose(c, row) for row in cycle) for c in cmds)


def test_a_turn_tap_completes_one_cycle_and_a_fine_turn_half_of_it():
    """robot_teleop_v2.py: tapniecie Q = dokladnie jeden cykl obrotu (20 krokow), Z = FINE_STEPS krokow z biodrami
    o amplitudzie FINE_HIP_AMP; zmiana trybu w trakcie czeka na granice cyklu, spacja przerywa od razu."""
    drive, io, pad = fake_drive()
    pad.hold("q", 0.2)                                     # tapniecie: 4 takty
    pad.hold("", 2.0)
    full = drive.cycles["obrot_lewo"]
    assert cycle_steps(io.cmds, full) == len(full) == 20
    assert drive.status.mode == "stoi" and np.allclose(io.cmds[-1], drive.start_pose)
    io.cmds.clear()
    pad.hold("z", 0.1)                                     # maly obrot: pol cyklu
    pad.hold("", 1.5)
    fine = drive.cycles["obrot_lewo_maly"]
    assert cycle_steps(io.cmds, fine) == FINE_STEPS == 10
    assert max(abs(c[0::2]).max() for c in io.cmds) == pytest.approx(FINE_HIP_AMP, abs=1e-6)
    io.cmds.clear()
    pad.hold("e", 1.7)                                     # trzymany: rampa 0,5 s, potem cykle leca dalej...
    pad.hold("", 2.0)
    right = drive.cycles["obrot_prawo"]
    assert cycle_steps(io.cmds, right) == 2 * len(right)   # ...puszczony w drugim cyklu konczy go
    io.cmds.clear()
    pad.hold("q", 0.8)                                     # rampa 0,5 s + 6 krokow cyklu...
    drive.keys("!")                                        # ...spacja w pulpicie: stop natychmiast
    pad.hold("", 1.0)
    assert 0 < cycle_steps(io.cmds, full) < len(full)


def test_pose_keys_ramp_in_and_hold_then_walking_goes_through_the_start_pose():
    drive, io, pad = fake_drive()
    pad.hold("f", 0.1)                                     # klawisz pozy: raz, przy pojawieniu sie
    pad.hold("", 1.0)
    assert np.allclose(io.cmds[-1], aw.FLAT_POSE) and drive.status.mode == "stoi"
    assert drive.status.direction == "plasko"
    assert len([c for c in io.cmds if not np.allclose(c, aw.FLAT_POSE)]) == POSE_RAMP_STEPS - 1
    io.cmds.clear()
    n_ramp = max(RAMP_STEPS, POSE_RAMP_STEPS)
    pad.hold("w", n_ramp * aw.DT)                          # z pozy plasko do jazdy: rampa przez poze startowa
    ramp = io.cmds[:n_ramp]
    assert np.allclose(ramp[-1], drive.start_pose) and not np.allclose(ramp[0], drive.start_pose)
    pad.hold("w", 1.0)
    assert drive.status.mode == "jazda"
    pad.hold("wt", 0.1)                                    # poza w trakcie jazdy: przerywa ruch, W trzymany nie wznawia
    pad.hold("wt", 1.0)
    assert drive.status.mode == "stoi" and np.allclose(io.cmds[-1], aw.TIPTOE_POSE)
    pad.hold("", 0.2)
    pad.hold("w", 1.0)                                     # po puszczeniu i ponownym wcisnieciu - jedzie
    assert drive.status.mode == "jazda"
    pad.hold("n", 0.1)                                     # machanie noga w gore: kostka ponad +1, konczy w postawie
    pad.hold("", 4.0)
    assert max(c[3] for c in io.cmds) == pytest.approx(1.5) and np.allclose(io.cmds[-1], aw.SETTLE_POSE)
    assert len(wave_actions(2, 3, "gora", aw.SETTLE_POSE, aw.SETTLE_POSE)) == 10 + 48 + 5 + 10
    pad.hold("1", 0.1)                                     # noga 1 prosto, reszta zostaje w postawie
    pad.hold("", 1.0)
    assert np.allclose(io.cmds[-1], [0, 1.0, 0, -0.25, 0, -0.25, 0, -0.25])
    drive.stop("STOP")                                     # STOP w pozie: trzyma ja (bez ruchu), klawisze ignorowane
    pad.hold("w", 1.0)
    assert np.allclose(io.cmds[-1], [0, 1.0, 0, -0.25, 0, -0.25, 0, -0.25]) and drive.status.mode == "STOP"


def test_deadman_stop_and_the_arm_untouched():
    tw = rt.Twin(sim_workspace())
    try:
        arm = tw.joints()
        drive = AntDrive(tw, lambda: None, camera="prawda")
        pad = Pad(drive)
        pad.hold("", 1.5)
        pad.hold("w", 1.0)
        assert drive.status.mode == "jazda"
        # Pulpit milknie (zamknieta karta, zerwana siec): po DEADMAN mrowka schodzi do pozy startowej.
        pad.hold("", DEADMAN + 0.2, refresh=False)
        pad.hold("", 0.6, refresh=False)
        assert drive.status.mode == "stoi", drive.status
        drive.stop("STOP z panelu")
        p0 = tw.scene.ant_pose(STANDIN)
        pad.hold("w", 2.0)
        assert drive.status.mode == "STOP" and np.linalg.norm(tw.scene.ant_pose(STANDIN)[:3, 3] - p0[:3, 3]) < 0.02
        drive.clear()
        pad.hold("w", 1.5)
        assert drive.status.mode == "jazda"
        assert tw.joints() == pytest.approx(arm, abs=0.5)   # ramie stoi - mrowka ma wlasna petle
    finally:
        tw.close()


def demo_twin(tmp_path, ant_ids=None):
    from lerobot_mp.twin.robots import resolve_asset

    data = add_demo_ant(json.loads(resolve_asset("examples/twin.sim.json").read_text(encoding="utf-8")))
    (tmp_path / "twin.json").write_text(json.dumps(data), encoding="utf-8")
    ws = Workspace.load(tmp_path / "twin.json")
    return ws, rt.Twin(ws)


def drive_both(drive, mirror, pad, key: str, seconds: float) -> None:
    """Krok mrowki udajacej prawdziwa (ona kroczy fizyke), potem kopii - jak dwie petle w panelu."""
    for _ in range(int(round(seconds / aw.DT))):
        drive.keys(key)
        drive.tick()
        if mirror is not None:
            mirror.tick()
        pad.t += aw.DT


@pytest.mark.render
def test_the_copy_follows_on_servos_and_tags_and_calibrates_the_mount_itself(tmp_path):
    """Kopia wie tylko tyle, co o prawdziwej: odczyty serw i kadry. Montaz startuje zly (obrot 45 zamiast 135,
    wysokosc domyslna) - po kilku odcinkach jazdy jest dobry, a kopia stoi tam, gdzie mrowka."""
    ws, tw = demo_twin(tmp_path)
    try:
        ant = Ant(sim=ws.ant_obj().sim)                   # sesja: numery, bok, uklad z kadrow
        frames = {c.name: tw.render(c.name) for c in ws.cameras}
        fit = learn_layout(observe(frames, ws.cameras, ant.tags), ant.tags)
        snapped, offs = square_layout({1: fit.layout[1]})
        assert offs[1] < 1.5 and np.allclose(snapped[1][:3, :3], demo_layout()[1][:3, :3])
        ant.tags.layout = {2: np.eye(4), **snapped}
        assert ant.mount["yaw"] == 45.0                   # osie wzdluz nog, ale nie ta noga (prawda: 135)

        drive = AntDrive(tw, lambda: ant, camera="nominal")        # jej wlasna polityka, bez kamery
        mirror = AntMirror(tw, lambda: ant, StandInServos(tw))
        pad = Pad(drive)
        drive_both(drive, mirror, pad, "", 1.5)
        for key in ("w", "d", "s"):
            drive_both(drive, mirror, pad, key, 3.0)
            drive_both(drive, mirror, pad, "", 1.0)
        assert ant.mount["yaw"] == DEMO_MOUNT["yaw"], mirror.status.calib
        assert abs(ant.mount["z"] - DEMO_MOUNT["z"]) < 0.006, ant.mount
        err_m, err_deg = mirror_pose_error(tw)
        assert err_m < 0.02 and err_deg < 5.0, (err_m, err_deg)
        assert mirror.status.mode == "sledzi" and mirror.status.fixes > 50
    finally:
        tw.close()


def test_without_tags_the_copy_walks_on_servo_readings_alone(tmp_path):
    """Kamery nie widza tagow (tu: zly numer) - kopia idzie sama, bo nogi ma z odczytow serw."""
    ws, tw = demo_twin(tmp_path)
    try:
        blind = Ant(sim=ws.ant_obj().sim)
        blind.tags = type(blind.tags)("mrowka", (99,), 0.04)
        drive = AntDrive(tw, lambda: blind, camera="nominal")
        mirror = AntMirror(tw, lambda: blind, StandInServos(tw))
        pad = Pad(drive)
        drive_both(drive, mirror, pad, "", 1.5)
        a0, b0 = tw.scene.ant_pose(STANDIN)[:3, 3], tw.scene.ant_pose()[:3, 3]
        drive_both(drive, mirror, pad, "w", 2.5)
        da = tw.scene.ant_pose(STANDIN)[:3, 3] - a0
        db = tw.scene.ant_pose()[:3, 3] - b0
        assert np.hypot(*da[:2]) > 0.08 and mirror.status.mode == "bez tagow"
        assert np.hypot(*(db[:2] - da[:2])) < 0.05, (da, db)       # ten sam przejazd z samych serw
    finally:
        tw.close()


def test_zmq_only_listens_to_ant_server():
    zmq = pytest.importorskip("zmq")
    ctx = zmq.Context()
    meas_out = ctx.socket(zmq.PUB)                         # antproxy: XPUB na 3001 (pomiary JSON z ant_server)
    port = meas_out.bind_to_random_port("tcp://127.0.0.1")
    ant = aw.ZmqAnt(aw.ZmqAntConfig("127.0.0.1", sub_port=port), context=ctx)
    try:
        assert not hasattr(ant, "pub") and not hasattr(ant, "command")    # blizniak niczego mrowce nie wysyla
        q_true = aw.action_to_ctrl(aw.start_action(aw.load_policy().cfg))
        units = aw.joints_to_units(q_true)
        deadline, k = time.monotonic() + 5.0, 0
        while ant.joints() is None and time.monotonic() < deadline:
            msg = {"id": "serial", "ant_time": 1000 + 20 * k, **{f"s{i + 1}_angle": units[i] for i in range(8)}}
            meas_out.send_multipart([json.dumps(msg).encode()])
            k += 1
            time.sleep(0.02)
        q, qd, age = ant.joints()
        assert np.allclose(q, q_true, atol=0.006) and age < 1.0
    finally:
        ant.close()
        meas_out.close(0)
        ctx.term()


def test_pad_forwards_keys_and_reports_state():
    from lerobot_mp.twin.antdrive import AntStatus
    from lerobot_mp.twin.antpad import AntPad

    class Drive:
        def __init__(self):
            self.status, self.got = AntStatus(mode="jazda", direction="przod"), []

        def keys(self, k):
            self.got.append(k)

    d = Drive()
    pad = AntPad(d, port=0, host="127.0.0.1").start()
    try:
        base = f"http://127.0.0.1:{pad.port}"
        req = urllib.request.Request(base + "/keys", data=b"k=wdx", method="POST")
        urllib.request.urlopen(req, timeout=2).read()
        assert d.got == ["wd"]                             # tylko klawisze pulpitu
        req = urllib.request.Request(base + "/keys", data=b"k=zf!", method="POST")
        urllib.request.urlopen(req, timeout=2).read()
        assert d.got[-1] == "zf!"                          # maly obrot, poza, stop natychmiast (spacja)
        st = json.loads(urllib.request.urlopen(base + "/state", timeout=2).read())
        assert st == {"text": "jazda - przod", "level": "ok"}
        page = urllib.request.urlopen(base + "/", timeout=2).read().decode()
        assert str(pad.port) in page and "keydown" in page
        assert 'srcdoc="' in pad.html_embed()
    finally:
        pad.close()


def test_wsad_relative_to_the_3d_view_latched_while_held():
    tw = rt.Twin(sim_workspace())
    try:
        view = [np.array([-1.0, 0.0])]                     # patrzacy stoi przed mrowka i patrzy w strone -x
        drive = AntDrive(tw, lambda: None, camera="prawda", view_dir=lambda: view[0])
        pad = Pad(drive)
        pad.hold("", 1.5)
        assert abs(drive.status.heading_deg) < 10         # mrowka przodem do +x - do patrzacego
        pad.hold("w", 0.5)
        assert drive.status.direction == "tyl"            # W = w glab ekranu = od patrzacego = tyl mrowki
        view[0] = np.array([0.0, 1.0])                     # obrot widoku w trakcie jazdy nie przelacza nog
        pad.hold("w", 0.5)
        assert drive.status.direction == "tyl"
        pad.hold("", 1.0)
        view[0] = np.array([-1.0, 0.0])
        pad.hold("a", 0.5)
        assert drive.status.direction == "prawo"          # lewo ekranu (-y) = prawo mrowki
        drive.frame = "mrowka"
        pad.hold("", 1.0)
        pad.hold("w", 0.5)
        assert drive.status.direction == "przod"
    finally:
        tw.close()
