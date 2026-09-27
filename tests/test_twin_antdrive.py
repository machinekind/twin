"""Mrowka jezdzi: polityka chodu realant-rl w bliznaku, WSAD przez przemianowanie nog, poza z tagow, kalibracja kierunku.

Mrowka chodzi w fizyce sceny blizniaka (ta sama scena, co w panelu; `SimAnt` dokracza ja, gdy ramie nie jest
polaczone). WSAD nie zmienia polityki - te same wagi jada w czterech kierunkach, bo nogi sa przemianowane,
a obserwacja obrocona. Poza dla polityki: prawda z fizyki (szybko) albo tagi z renderowanych kamer (tak jak
na prawdziwej mrowce). Prawdziwa mrowka: lacze ZeroMQ sprawdzone na udawanym antproxy.
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
from lerobot_mp.twin.ant import (DEMO_MOUNT, LEG_POSES, Ant, add_demo_ant, demo_layout, level_mount, locate,  # noqa: E402
                                 rest_height, snap_yaw, square_layout)
from lerobot_mp.twin.antdrive import DEADMAN, AntDrive, heading_from_tracks  # noqa: E402
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
    # Nastawy serw jak rollout_server.apply_controls: akcja 0 -> biodra 510 (nogi 1-2) / 512, kostki 399;
    # s1..s8 w kolejnosci serw (noga 1 polityki = s3, s4).
    assert aw.action_to_units(np.zeros(8)) == [510, 399, 510, 399, 512, 399, 512, 399]
    assert aw.action_to_units(np.array([1.0, -1.0, 0, 0, 0, 0, 0, 0]))[2:4] == [663, 287]
    # Rozkaz -> jednostki -> odczyt wraca do kata z dokladnoscia 0,015 rad: rollout_server rozkazuje biodrom nog 3-4
    # od srodka 512, a odczyt liczy od 510 (2 jednostki = 0,6 st.) - przeniesione wiernie, tak jedzie robot.
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(-1, 1, 8)
        assert np.allclose(aw.units_to_joints(aw.action_to_units(a)), aw.action_to_ctrl(a), atol=0.015)
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
            p0, h0 = tw.scene.ant_pose(), np.degrees(aw.heading(tw.scene.ant_pose()))
            pad.hold(key, 2.5)
            assert drive.status.mode == "jazda" and drive.status.steps > 40
            pad.hold("", 1.0)
            d = tw.scene.ant_pose()[:3, 3] - p0[:3, 3]
            c, s = np.cos(np.radians(h0)), np.sin(np.radians(h0))
            local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])   # w ukladzie mrowki sprzed jazdy
            along, across = local @ np.array(want), local @ np.array([-want[1], want[0]])
            assert along > 0.17, (key, local)                   # 10 cm/s z treningu: ~0,23 m w 2,5 s
            assert abs(across) < 0.05, (key, local)
        assert drive.status.mode == "stoi" and tw.scene.ant_pose()[2, 2] > 0.95
        h0 = aw.heading(tw.scene.ant_pose())
        pad.hold("q", 2.0)
        turned = np.degrees(np.angle(np.exp(1j * (aw.heading(tw.scene.ant_pose()) - h0))))
        assert turned > 60.0, turned                        # obrot w lewo (otwarta petla z robot_route.py)
    finally:
        tw.close()


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
        p0 = tw.scene.ant_pose()
        pad.hold("w", 2.0)
        assert drive.status.mode == "STOP" and np.linalg.norm(tw.scene.ant_pose()[:3, 3] - p0[:3, 3]) < 0.02
        drive.clear()
        pad.hold("w", 1.5)
        assert drive.status.mode == "jazda"
        assert tw.joints() == pytest.approx(arm, abs=0.5)   # ramie stoi - mrowka ma wlasna petle
    finally:
        tw.close()


@pytest.mark.render
def test_calibrated_against_the_table_from_frames_then_walks_on_tag_poses(tmp_path):
    """Demo z mrowka: uklad tagow z kadrow, poziom z blatu, kierunek z przejazdu W/S - potem W na pozie z tagow."""
    from lerobot_mp.twin.robots import resolve_asset

    data = add_demo_ant(json.loads(resolve_asset("examples/twin.sim.json").read_text(encoding="utf-8")))
    (tmp_path / "twin.json").write_text(json.dumps(data), encoding="utf-8")
    ws = Workspace.load(tmp_path / "twin.json")
    tw = rt.Twin(ws)
    try:
        ant = Ant()                                       # sesja: tylko numery i bok taga
        frames = {c.name: tw.render(c.name) for c in ws.cameras}
        fit = learn_layout(observe(frames, ws.cameras, ant.tags), ant.tags)
        assert fit.missing(ant.tags.ids) == []
        snapped, offs = square_layout({1: fit.layout[1]})
        assert offs[1] < 1.5 and np.allclose(snapped[1][:3, :3], demo_layout()[1][:3, :3])
        ant.tags.layout = {2: np.eye(4), **snapped}
        ant.joints = dict(LEG_POSES["stoi"])
        fix = locate(frames, ws.cameras, ant)
        ant.mount = level_mount(fix.tags.T_body2base, ant)
        assert abs(ant.mount["z"] - DEMO_MOUNT["z"]) < 0.004
        assert ant.mount["yaw"] == 45.0                   # osie wzdluz nog, ale ktora noga - jeszcze nie wiadomo

        drive = AntDrive(tw, lambda: ant, camera="tagi")
        pad = Pad(drive)
        pad.hold("", 1.5)
        drive.calibrate_heading(2.5)
        pad.hold("", 9.0)
        res = drive.last_calibration
        assert res is not None and not res["error"], res
        assert abs(res["yaw_deg"] - DEMO_MOUNT["yaw"]) < 6.0, res
        assert snap_yaw(res["yaw_deg"])[0] == DEMO_MOUNT["yaw"]
        assert min(res["dists"]) > 0.15 and res["spread_deg"] < 8.0, res
        assert drive.camera == "tagi"
        ant.mount = {**ant.mount, "yaw": snap_yaw(res["yaw_deg"])[0]}

        # Jazda do przodu na pozie z tagow: prosto wzdluz przodu mrowki.
        p0 = tw.scene.ant_pose()
        pad.hold("w", 2.0)
        assert drive.status.mode == "jazda" and drive.status.pose_age is not None and drive.status.pose_age < 0.2
        d = tw.scene.ant_pose()[:3, 3] - p0[:3, 3]
        fwd = p0[:3, 0]
        assert d @ fwd > 0.12 and abs(d @ np.cross([0, 0, 1], fwd)) < 0.04, d
    finally:
        tw.close()


def test_heading_from_tracks_needs_a_real_move():
    T = pose(np.eye(3), np.zeros(3))
    res = heading_from_tracks([[T] * 20, [T] * 20])
    assert res["yaw_deg"] is None and "nie przejechala" in res["error"]


def test_zmq_real_ant_against_a_fake_antproxy():
    zmq = pytest.importorskip("zmq")
    ctx = zmq.Context()
    cmd_in = ctx.socket(zmq.SUB)                           # antproxy: XSUB na 3002 (tu SUB na cmd)
    cmd_port = cmd_in.bind_to_random_port("tcp://127.0.0.1")
    cmd_in.setsockopt(zmq.SUBSCRIBE, b"cmd")
    meas_out = ctx.socket(zmq.PUB)                         # antproxy: XPUB na 3001 (pomiary JSON)
    meas_port = meas_out.bind_to_random_port("tcp://127.0.0.1")
    ant = aw.ZmqAnt(aw.ZmqAntConfig("127.0.0.1", pub_port=cmd_port, sub_port=meas_port), context=ctx)
    try:
        units = aw.action_to_units(aw.start_action(aw.load_policy().cfg))
        deadline = time.monotonic() + 5.0
        k = 0
        while ant.joints() is None and time.monotonic() < deadline:
            msg = {"id": "serial", "ant_time": 1000 + 20 * k, **{f"s{i + 1}_angle": units[i] for i in range(8)}}
            meas_out.send_multipart([json.dumps(msg).encode()])
            k += 1
            time.sleep(0.02)
        q, qd, age = ant.joints()
        assert np.allclose(q, aw.action_to_ctrl(aw.start_action(aw.load_policy().cfg)), atol=0.012)
        assert age < 1.0
        got = []
        cmd_in.setsockopt(zmq.RCVTIMEO, 2000)
        for _ in range(20):                                # PUB/SUB: pierwsze wiadomosci moga zginac przy laczeniu
            ant.command(np.zeros(8))
            try:
                got.append(cmd_in.recv_multipart())
                break
            except zmq.Again:
                continue
        assert got and got[0][0] == b"cmd" and got[0][1] == b"s1 510 s2 399 s3 510 s4 399 s5 512 s6 399 s7 512 s8 399\n"
    finally:
        ant.close()
        cmd_in.close(0)
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
