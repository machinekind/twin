"""Backend `a1x`: Galaxea A1X po CAN (protokol i lacze z pakietu galaxeo) bez sprzetu.

`FakeA1XBus` udaje ramie na wirtualnym zegarze: trzyma 7 grup (6 stawow + chwytak),
jedzie do p_des ze skonczona predkoscia, na kazde `tick()` wystawia jedna ramke 0x052
i zapisuje kazda wyslana ramke (id, dane, fd). Pokretla: `silent`, `frozen`, `deaf`,
`grip_effort`, `state`.
"""

from __future__ import annotations

import math
import struct
import threading
import time
from collections import deque

import pytest

galaxeo = pytest.importorskip("galaxeo")

from galaxeo import protocol as P  # noqa: E402
from galaxeo.bus import Frame  # noqa: E402

from lerobot_mp.config import JointConfig, load_config  # noqa: E402
from lerobot_mp.robot.a1x import A1XArm  # noqa: E402

ARM = tuple(f"arm_joint{i}" for i in range(1, 7))
#: Stawy A1X w jednostkach aplikacji - lokalnie, bo opis A1X (robots.A1X) przychodzi z innej galezi.
A1X_JOINTS = {
    "arm_joint1": JointConfig(min=-160.0, max=160.0, max_vel=90.0),
    "arm_joint2": JointConfig(min=-2.0, max=175.0, max_vel=90.0),
    "arm_joint3": JointConfig(min=-185.0, max=2.0, max_vel=90.0),
    "arm_joint4": JointConfig(min=-88.0, max=88.0, max_vel=120.0),
    "arm_joint5": JointConfig(min=-88.0, max=88.0, max_vel=120.0),
    "arm_joint6": JointConfig(min=-160.0, max=160.0, max_vel=150.0),
    "gripper": JointConfig(min=0.0, max=100.0, max_vel=300.0),
}
START_DEG = (10.0, 30.0, -60.0, 5.0, 0.0, -20.0)
STEP = 0.005                                   # jeden takt strumienia (200 Hz)
LSB_DEG = math.degrees(1.0 / P.POS_SCALE)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FakeA1XBus:
    """Ramie A1X na magistrali: odpowiada 0x052, slucha 0x050/0x051, zapisuje nadawanie."""

    def __init__(self, start_deg=START_DEG, grip_q=0.0):
        self.q = [math.radians(x) for x in start_deg] + [grip_q]
        self.v = [0.0] * 7
        self.eff = [0.0] * 7
        self.rate = math.radians(120.0)        # stawy jada do p_des z ta predkoscia [rad/s]
        self.grip_rate = 2.0
        self.p_des: list[float] | None = None
        self.g_des: float | None = None
        self.silent = self.frozen = self.deaf = False
        self.grip_effort = 0.0
        self.state = "ok"
        self.rx: deque = deque()
        self.tx: list[tuple[int, bytes, bool]] = []
        self.closed = False
        self._n = 0
        self._frozen_payload: bytes | None = None

    # --- CanBus
    def recv(self, timeout: float = 0.0):
        return self.rx.popleft() if self.rx else None

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None:
        data = bytes(data)
        self.tx.append((can_id, data, fd))
        if can_id == P.CMD_ID and not self.deaf and not self.frozen:
            self.p_des = [struct.unpack_from(">h", data, j * 10)[0] / P.POS_SCALE for j in range(6)]
        elif can_id == P.GRIP_ID and not self.frozen:
            self.g_des = struct.unpack_from(">h", data, 0)[0] / P.POS_SCALE

    def close(self) -> None:
        self.closed = True

    # --- fizyka
    def tick(self, dt: float = STEP, n: int = 1) -> None:
        for _ in range(n):
            if self.p_des is not None:
                for j in range(6):
                    d = max(-self.rate * dt, min(self.rate * dt, self.p_des[j] - self.q[j]))
                    self.q[j] += d
                    self.v[j] = d / dt
            else:
                self.v[:6] = [0.0] * 6
            if self.g_des is not None and not self.grip_effort:
                self.q[6] += max(-self.grip_rate * dt, min(self.grip_rate * dt, self.g_des - self.q[6]))
            self.eff[6] = self.grip_effort
            self._n += 1
            if self.silent:
                continue
            if self.frozen:
                if self._frozen_payload is None:
                    self._frozen_payload = P.encode_feedback(self.q, self.v, self.eff)
                payload = self._frozen_payload
            else:
                self._frozen_payload = None
                eff = list(self.eff)
                eff[0] += 0.002 * (self._n % 2)          # szum pomiaru: zywa telemetria nie powtarza sie
                payload = P.encode_feedback(self.q, self.v, eff)
            self.rx.append(Frame(P.FB_ID, payload))

    # --- odczyt nadawania
    def frames(self, can_id: int) -> list[bytes]:
        return [d for i, d, _ in self.tx if i == can_id]

    def pdes_deg(self, data: bytes) -> list[float]:
        return [math.degrees(struct.unpack_from(">h", data, j * 10)[0] / P.POS_SCALE) for j in range(6)]


def make_arm(bus=None, clock=None, joints=None, **a1x):
    clock = clock or Clock()
    bus = bus or FakeA1XBus()

    def sleep(secs):
        n = max(1, round(secs / STEP))
        for _ in range(n):
            clock.t += secs / n
            bus.tick(secs / n)

    cfg = load_config(overrides={"a1x": a1x} if a1x else None, joints=A1X_JOINTS if joints is None else joints)
    arm = A1XArm(cfg, bus=bus, clock=clock, threaded=False, sleep=sleep, connect_timeout_s=0.05)
    return arm, bus, clock


def connected(**kw):
    arm, bus, clock = make_arm(**kw)
    bus.tick(n=60)
    arm.connect()
    return arm, bus, clock


def run(arm, bus, clock, secs: float, read: bool = True) -> None:
    for _ in range(int(round(secs / STEP))):
        clock.t += STEP
        bus.tick(STEP)
        arm.io_step(clock.t)
        if read:
            arm.read_joints()


# ------------------------------------------------------------------ polaczenie
def test_nothing_is_transmitted_before_the_first_command():
    arm, bus, clock = connected()
    run(arm, bus, clock, 1.0)
    assert arm.read_joints()["arm_joint1"] == pytest.approx(10.0, abs=LSB_DEG)
    assert arm.goal_positions() == {}
    assert bus.tx == []
    assert arm.faults() == []


def test_connect_refuses_without_feedback():
    arm, bus, clock = make_arm()
    bus.silent = True
    bus.tick(n=10)
    with pytest.raises(RuntimeError, match="brak ramek 0x052"):
        arm.connect()
    assert not arm.is_connected and bus.tx == []


def test_connect_refuses_frozen_telemetry():
    arm, bus, clock = make_arm()
    bus.frozen = True
    bus.tick(n=60)
    with pytest.raises(RuntimeError, match="zamrozona"):
        arm.connect()
    assert bus.tx == []


def test_backend_refuses_a_so101_joint_table():
    with pytest.raises(ValueError, match="6 stawow ramienia"):
        A1XArm(load_config(), bus=FakeA1XBus(), threaded=False)


def test_first_frame_equals_the_measured_pose_and_slews_from_there():
    arm, bus, clock = connected()
    arm.send_joints({"arm_joint1": 40.0})
    run(arm, bus, clock, 0.6)
    frames = [bus.pdes_deg(d) for d in bus.frames(P.CMD_ID)]
    assert len(frames) == 120
    measured = START_DEG
    first = frames[0]
    assert first[0] == pytest.approx(measured[0], abs=90.0 * STEP + 2 * LSB_DEG)   # jeden krok od pomiaru
    assert first[1:] == pytest.approx(list(measured[1:]), abs=2 * LSB_DEG)         # reszta = pomiar
    for a, b in zip(frames, frames[1:]):
        assert abs(b[0] - a[0]) <= 90.0 * STEP + 2 * LSB_DEG                     # <= 90 st./s
    assert frames[-1][0] == pytest.approx(40.0, abs=2 * LSB_DEG)
    assert all(len(d) == P.CMD_LEN for d in bus.frames(P.CMD_ID))
    assert all(fd for i, _, fd in bus.tx if i == P.CMD_ID)
    # kp i kd z konfiguracji, nigdy 0
    d = bus.frames(P.CMD_ID)[0]
    assert all(struct.unpack_from(">h", d, j * 10 + 4)[0] == int(20.0 * 60) for j in range(6))


def test_stream_repeats_the_last_target_between_send_joints_calls():
    arm, bus, clock = connected()
    arm.send_joints({"arm_joint2": 35.0})
    run(arm, bus, clock, 0.5)
    n0 = len(bus.frames(P.CMD_ID))
    run(arm, bus, clock, 0.5)
    frames = bus.frames(P.CMD_ID)[n0:]
    assert len(frames) == 100 and len(set(frames)) == 1                          # ten sam cel, 200 Hz
    assert arm.read_joints()["arm_joint2"] == pytest.approx(35.0, abs=0.1)
    assert arm.goal_positions()["arm_joint2"] == 35.0


def test_stale_feedback_sets_link_silent_stops_transmit_and_reports_a_fault():
    arm, bus, clock = connected()
    arm.send_joints({"arm_joint1": 20.0})
    run(arm, bus, clock, 0.2)
    bus.silent = True
    run(arm, bus, clock, 0.2)
    assert arm.link_silent and set(arm._stale) == set(ARM)
    faults = arm.faults()
    assert len(faults) == 1 and faults[0].startswith("brak odpowiedzi ramienia od")
    n = len(bus.tx)
    run(arm, bus, clock, 0.3)
    assert len(bus.tx) == n                                                      # wysylka stoi
    assert arm.send_joints({"arm_joint1": 25.0}) == {}
    # powrot: nastepna ramka konczy cisze, strumien startuje od nowego pomiaru
    bus.silent = False
    run(arm, bus, clock, 0.05)
    assert not arm.link_silent and arm.faults() == []
    assert len(bus.tx) > n
    first = bus.pdes_deg(bus.tx[n][1])
    assert abs(first[0] - math.degrees(bus.q[0])) <= 90.0 * STEP * 2 + 2 * LSB_DEG


def test_frozen_payloads_are_reported_and_stop_transmit():
    arm, bus, clock = connected()
    arm.send_joints({"arm_joint1": 12.0})
    run(arm, bus, clock, 0.1)
    bus.frozen = True
    run(arm, bus, clock, 0.4)                                                    # 80 identycznych ramek
    assert any("telemetria zamrozona" in f for f in arm.faults())
    assert arm.link_silent
    n = len(bus.tx)
    run(arm, bus, clock, 0.2)
    assert len(bus.tx) == n


def test_a_deaf_arm_is_reported_from_the_command_measured_gap():
    from lerobot_mp.twin.runtime import _fault_joint

    arm, bus, clock = connected()
    bus.deaf = True
    arm.send_joints({"arm_joint2": 40.0})
    run(arm, bus, clock, 0.6)
    assert arm.faults() == []                                                    # jeszcze nie deaf_s
    run(arm, bus, clock, 1.0)
    faults = arm.faults()
    assert len(faults) == 1 and "gluche" in faults[0] and "zablokowane" in faults[0]
    assert _fault_joint(faults[0], ARM + ("gripper",)) == "arm_joint2"          # blizniak: staw ramienia = STOP
    bus.deaf = False                                                             # FF 1-5-6 pomoglo: ramie jedzie
    run(arm, bus, clock, 0.3)
    assert arm.faults() == []


def test_stale_and_frozen_faults_are_arm_faults_for_the_twin():
    from lerobot_mp.twin.runtime import _fault_joint

    arm, bus, clock = connected()
    bus.silent = True
    run(arm, bus, clock, 0.3)
    [f] = arm.faults()
    assert "brak odpowiedzi" in f and _fault_joint(f, ARM + ("gripper",)) is None
    bus.state = "bus-off"
    assert "magistrala CAN: bus-off" in arm.faults()


def test_read_joints_are_degrees_with_the_joint3_offset_applied_symmetrically():
    start = (0.0, 20.0, 1.49, 0.0, 0.0, 0.0)
    arm, bus, clock = make_arm(bus=FakeA1XBus(start), joint_offsets_deg=[0.0, 0.0, 1.49, 0.0, 0.0, 0.0])
    bus.tick(n=60)
    arm.connect()
    assert arm.read_joints()["arm_joint3"] == pytest.approx(0.0, abs=2 * LSB_DEG)
    assert arm.read_joints()["arm_joint2"] == pytest.approx(20.0, abs=LSB_DEG)
    arm.send_joints({"arm_joint3": -10.0})
    run(arm, bus, clock, 0.5)
    assert bus.pdes_deg(bus.frames(P.CMD_ID)[-1])[2] == pytest.approx(-10.0 + 1.49, abs=2 * LSB_DEG)
    assert arm.read_joints()["arm_joint3"] == pytest.approx(-10.0, abs=2 * LSB_DEG)
    lo, hi = arm.joint_limits()["arm_joint3"]
    assert hi == pytest.approx(0.0, abs=0.02) and lo == pytest.approx(math.degrees(-3.316) - 1.49, abs=1e-6)


def test_limits_are_urdf_widened_to_the_start_pose():
    arm, bus, clock = make_arm(bus=FakeA1XBus((0.0, 20.0, 1.49, 0.0, 0.0, 0.0)))
    bus.tick(n=60)
    arm.connect()
    lim = arm.joint_limits()
    assert set(lim) == set(ARM)                                                  # bez chwytaka
    assert lim["arm_joint3"][1] == pytest.approx(1.49, abs=0.02)
    assert lim["arm_joint3"][0] == pytest.approx(math.degrees(-3.316))
    assert lim["arm_joint1"] == pytest.approx((math.degrees(-2.880), math.degrees(2.880)))
    assert arm.send_joints({"arm_joint2": -30.0})["arm_joint2"] == pytest.approx(0.0)    # URDF 0 < cfg -2
    assert arm.send_joints({"arm_joint4": 95.0})["arm_joint4"] == pytest.approx(88.0)


def test_send_joints_clamps_to_the_mutable_joint_config():
    arm, bus, clock = connected()
    arm.cfg.joint("arm_joint1").max = 10.0                                       # _sync_backend_limits blizniaka
    assert arm.send_joints({"arm_joint1": 20.0}) == {"arm_joint1": 10.0}
    arm.cfg.joint("arm_joint1").max = 160.0
    assert arm.send_joints({"arm_joint1": 20.0}) == {"arm_joint1": 20.0}
    assert arm.send_joints({"arm_joint6": -170.0}) == {"arm_joint6": -160.0}


# --------------------------------------------------------------------- chwytak
def test_gripper_0_100_maps_linearly_to_pdes_endpoints_in_a_ten_byte_frame():
    arm, bus, clock = connected()
    for v, want in ((0.0, 0.6), (100.0, -2.0), (50.0, -0.7)):
        assert arm.send_joints({"gripper": v}) == {"gripper": v}
        run(arm, bus, clock, 1.5)
        data = bus.frames(P.GRIP_ID)[-1]
        assert len(data) == 10
        assert struct.unpack_from(">h", data, 0)[0] == int(want * P.POS_SCALE)
        assert struct.unpack_from(">h", data, 4)[0] == int(25.0 * 60)              # gripper_kp
    assert all(fd for i, _, fd in bus.tx if i == P.GRIP_ID)
    # slew p_des chwytaka: najwyzej 2/s
    ps = [struct.unpack_from(">h", d, 0)[0] / P.POS_SCALE for d in bus.frames(P.GRIP_ID)]
    assert max(abs(b - a) for a, b in zip(ps, ps[1:])) <= 2.0 * STEP + 2 / P.POS_SCALE


def test_gripper_stream_starts_only_after_a_real_command():
    arm, bus, clock = connected()
    held = arm.read_joints()
    assert held["gripper"] == 100.0                                              # gripper_start, bez sprzezenia
    arm.send_joints(held)                                                        # utrzymanie pozy po Polacz
    run(arm, bus, clock, 0.3)
    assert bus.frames(P.CMD_ID) and not bus.frames(P.GRIP_ID)
    arm.send_joints({"gripper": 99.7})
    run(arm, bus, clock, 0.1)
    assert not bus.frames(P.GRIP_ID)
    arm.send_joints({"gripper": 90.0})
    run(arm, bus, clock, 0.1)
    first = struct.unpack_from(">h", bus.frames(P.GRIP_ID)[0], 0)[0] / P.POS_SCALE
    assert first == pytest.approx(-2.0 + 2.0 * STEP, abs=2 / P.POS_SCALE)       # od zgloszonego otwarcia


def test_gripper_target_freezes_on_contact_but_send_joints_returns_the_command():
    arm, bus, clock = connected()
    assert arm.send_joints({"gripper": 0.0}) == {"gripper": 0.0}
    run(arm, bus, clock, 0.4)
    bus.grip_effort = 2.0                                                        # szczeki na kostce
    run(arm, bus, clock, 0.05)
    held = bus.frames(P.GRIP_ID)[-1]
    run(arm, bus, clock, 0.5)
    assert set(bus.frames(P.GRIP_ID)[-100:]) == {held}                           # cel stoi - chwyta, nie miazdzy
    p = struct.unpack_from(">h", held, 0)[0] / P.POS_SCALE
    assert -2.0 < p < 0.6
    assert arm.send_joints({"gripper": 0.0}) == {"gripper": 0.0}                 # rozkaz, nie zamrozone p_des
    assert arm.goal_positions()["gripper"] == 0.0
    run(arm, bus, clock, 0.2)
    assert bus.frames(P.GRIP_ID)[-1] == held
    arm.send_joints({"gripper": 100.0})                                          # otwieranie zdejmuje zamrozenie
    run(arm, bus, clock, 0.2)
    assert struct.unpack_from(">h", bus.frames(P.GRIP_ID)[-1], 0)[0] / P.POS_SCALE < p - 0.3


def test_gripper_feedback_two_point_calibration_and_open_loop_fallback():
    arm, bus, clock = make_arm(bus=FakeA1XBus(grip_q=-0.4), gripper_fb_closed=0.2, gripper_fb_open=-1.0)
    bus.tick(n=60)
    arm.connect()
    assert arm.read_joints()["gripper"] == pytest.approx(50.0, abs=0.1)
    assert arm.calibration_warnings() == []
    arm, bus, clock = connected()
    assert arm.read_joints()["gripper"] == 100.0
    arm.send_joints({"gripper": 30.0})
    assert arm.read_joints()["gripper"] == 30.0                                  # petla otwarta: ostatni rozkaz
    [w] = arm.calibration_warnings()
    assert "gripper_fb_closed/open" in w
    assert arm.gripper_ticks() is None


# ------------------------------------------------------ wlaczenie i rozlaczenie
def test_enable_sequence_streams_before_between_and_after_each_code():
    arm, bus, clock = connected()
    measured = [math.degrees(x) for x in P.decode_feedback(bus.rx[-1].data if bus.rx else
                                                             P.encode_feedback(bus.q)).pos[:6]]
    arm.enable()
    ff = [(k, d, fd) for k, (i, d, fd) in enumerate(bus.tx) if i == P.FF_ID]
    assert [d for _, d, _ in ff] == [b"\x01", b"\x05", b"\x06"]
    assert all(fd is False for _, _, fd in ff)                                   # 1 bajt, klasyczna CAN
    idx = [k for k, _, _ in ff]
    bounds = [-1] + idx + [len(bus.tx)]
    for a, b in zip(bounds, bounds[1:]):                                         # przed, miedzy i po
        arm_frames = [d for i, d, _ in bus.tx[a + 1:b] if i == P.CMD_ID]
        assert len(arm_frames) >= 50
    assert len(set(bus.frames(P.CMD_ID))) == 1                                   # p_des = pomiar, caly czas
    assert bus.pdes_deg(bus.frames(P.CMD_ID)[0]) == pytest.approx(measured, abs=LSB_DEG + 1e-9)
    assert not bus.frames(P.GRIP_ID)


def test_send_joints_is_refused_while_enabling_and_enable_refuses_a_silent_link():
    arm, bus, clock = connected()
    seen = []
    orig = arm._wait

    def wait(secs):
        seen.append(arm.send_joints({"arm_joint1": 50.0}))
        orig(secs)
    arm._wait = wait
    arm.enable()
    assert seen and all(s == {} for s in seen)
    assert arm.send_joints({"arm_joint1": 11.0}) == {"arm_joint1": 11.0}
    bus.silent = True
    run(arm, bus, clock, 0.3)
    with pytest.raises(RuntimeError, match="brak swiezych"):
        arm.enable()


def test_enable_on_connect_is_opt_in():
    arm, bus, clock = connected()
    assert not bus.frames(P.FF_ID)
    arm, bus, clock = connected(enable_on_connect=True)
    assert [d for d in bus.frames(P.FF_ID)] == [b"\x01", b"\x05", b"\x06"]


def test_kp_zero_never_reaches_the_bus():
    with pytest.raises(ValueError, match="kp"):
        make_arm(kp=0.0)
    with pytest.raises(ValueError, match="kp"):
        make_arm(gripper_kp=0.0)


def test_disconnect_never_sends_a_release_frame(caplog):
    arm, bus, clock = make_arm()
    arm.cfg.robot.torque_off_on_exit = True
    bus.tick(n=60)
    arm.connect()
    arm.send_joints({"arm_joint1": 15.0, "gripper": 20.0})
    run(arm, bus, clock, 0.3)
    arm.enable()
    run(arm, bus, clock, 0.1)
    with caplog.at_level("WARNING"):
        arm.disconnect()
    assert "nie ma hamulcow" in caplog.text
    assert bus.closed and not arm.is_connected
    released = [d for i, d, _ in bus.tx if i == P.FF_ID and d[0] in P.RELEASE_CODES]
    assert released == []
    n = len(bus.tx)
    run(arm, bus, clock, 0.1, read=False)
    assert len(bus.tx) == n


def test_no_heartbeat_ever():
    arm, bus, clock = connected()
    arm.send_joints({"arm_joint1": 15.0, "gripper": 20.0})
    run(arm, bus, clock, 1.0)
    arm.enable()
    arm.disconnect()
    ids = {i for i, _, _ in bus.tx}
    assert ids <= {P.CMD_ID, P.GRIP_ID, P.FF_ID}
    assert P.HEARTBEAT_ID not in ids


def test_threaded_io_streams_at_the_configured_rate_and_stops_on_disconnect():
    """Watek IO na prawdziwym zegarze: ramki 0x052 przychodza same, strumien idzie ~200 Hz."""

    class LiveBus(FakeA1XBus):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()
            self.t_last = time.monotonic()

        def recv(self, timeout=0.0):
            with self.lock:
                now = time.monotonic()
                while now - self.t_last >= STEP:
                    self.t_last += STEP
                    self.tick(STEP)
                return super().recv(timeout)

        def send(self, can_id, data, *, fd=True):
            with self.lock:
                super().send(can_id, data, fd=fd)

    bus = LiveBus()
    bus.tick(n=60)
    cfg = load_config(joints=A1X_JOINTS)
    arm = A1XArm(cfg, bus=bus)
    arm.connect()
    try:
        time.sleep(0.1)
        assert bus.tx == []
        arm.send_joints({"arm_joint1": 12.0})
        time.sleep(0.4)
        n = len(bus.frames(P.CMD_ID))
        assert 30 <= n <= 120
        assert arm.read_joints()["arm_joint1"] == pytest.approx(12.0, abs=0.2)
        assert not arm.link_silent and arm.faults() == []
    finally:
        arm.disconnect()
    n = len(bus.tx)
    time.sleep(0.05)
    assert len(bus.tx) == n and bus.closed


# ------------------------------------------------------------------ blizniak
def test_twin_connects_to_a_fake_a1x_holds_pose_and_follows_a_target(monkeypatch):
    pytest.importorskip("mujoco")
    from lerobot_mp.twin import robots
    from lerobot_mp.twin import runtime as rt
    from lerobot_mp.twin.runtime import Twin
    from lerobot_mp.twin.workspace import Workspace

    if "a1x" not in robots.REGISTRY:
        pytest.skip("opis ramienia a1x (robots.A1X) jeszcze nie jest w rejestrze")
    spec = robots.REGISTRY["a1x"]
    home = [spec.home.get(n, 0.0) for n in ARM]
    clock = Clock()
    bus = FakeA1XBus(home)
    bus.tick(n=60)
    made = {}

    def sleep(secs):
        clock.t += secs
        bus.tick(secs)

    def create(cfg):
        made["arm"] = A1XArm(cfg, bus=bus, clock=clock, threaded=False, sleep=sleep, connect_timeout_s=0.05)
        return made["arm"]

    monkeypatch.setattr(rt, "create_backend", create)
    twin = Twin(Workspace(robot="a1x"))
    dt = 0.02

    def run_twin(secs):
        for _ in range(int(round(secs / dt))):
            for _ in range(int(round(dt / STEP))):
                clock.t += STEP
                bus.tick(STEP)
                made["arm"].io_step(clock.t)
            twin.step(dt)

    try:
        twin.connect("a1x", port="fake", threaded=False)
        run_twin(1.0)
        assert set(twin.status.measured) == set(spec.joints)
        assert bus.tx == []                                                      # nic nie jechalo
        twin.set_engaged(True)
        j1 = twin.status.measured["arm_joint1"]
        twin.set_target({"arm_joint1": j1 + 5.0})
        run_twin(1.5)
        frames = [bus.pdes_deg(d)[0] for d in bus.frames(P.CMD_ID)]
        assert frames
        assert max(abs(b - a) for a, b in zip(frames, frames[1:])) <= 90.0 * STEP + 2 * LSB_DEG
        assert twin.status.measured["arm_joint1"] == pytest.approx(j1 + 5.0, abs=2.0)
        bus.silent = True
        run_twin(0.5)
        assert twin.status.faults
        n = len(bus.tx)
        run_twin(0.3)
        assert len(bus.tx) == n
    finally:
        twin.close()
