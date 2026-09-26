"""Kierowca `SpecsReach` przez backend `a1x` na atrapie magistrali (`FakeA1XBus`), bez sprzetu.

Blizniak z `threaded=False`: petle i magistrale kreci test (`Rig.pump`), zegar wirtualny
idzie za zegarem sciennym, bo watek wykonawcy kierowcy jest prawdziwy (rampa, czekanie
na dojechanie). Przekaznik to ta sama atrapa co w `test_twin_specsreach.py`.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("websockets")
pytest.importorskip("galaxeo")

from galaxeo import protocol as P  # noqa: E402
from test_a1x import LSB_DEG, STEP, Clock, FakeA1XBus  # noqa: E402
from test_twin_specsreach import FakeRelay  # noqa: E402

from lerobot_mp.robot.a1x import A1XArm  # noqa: E402
from lerobot_mp.twin import robots  # noqa: E402
from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics  # noqa: E402
from lerobot_mp.twin.reach import ReachVolume  # noqa: E402
from lerobot_mp.twin.specsreach import SpecsReach  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

ARM = tuple(f"arm_joint{i}" for i in range(1, 7))
DT = 0.02                                      # takt petli blizniaka (4 takty magistrali)


class WallBus(FakeA1XBus):
    """Atrapa ze sciana: staw `wall[0]` nie przejdzie za `wall[1]` [rad] w strone rozkazu."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.wall: tuple[int, float, float] | None = None       # (staw, granica, znak kierunku)

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None:
        super().send(can_id, data, fd=fd)
        if self.wall is not None and self.p_des is not None:
            j, lim, sign = self.wall
            if sign * (self.p_des[j] - lim) > 0:
                self.p_des[j] = lim


class Rig:
    """Blizniak na atrapie A1X, kierowca i przekaznik; `pump` kreci petle w tempie zegara."""

    def __init__(self, monkeypatch, relay: FakeRelay, offset: dict[str, float] | None = None):
        spec = robots.REGISTRY["a1x"]
        self.kin = RobotKinematics(spec)
        self.vol = ReachVolume.from_workspace(Workspace(robot="a1x"))
        sol = self.kin.ik(self.vol.center, seed=spec.home, tol=0.003)
        assert sol.ok
        self.start = {k: float(sol.joints[k]) for k in ARM}  # z pozy domowej kazdy cel to skok > 45 st.
        self.clock = Clock()
        # `offset` [st.] - ramie startuje tyle od pozy nad srodkiem pudelka (dla `start_center`).
        self.bus = WallBus([self.start[k] + (offset or {}).get(k, 0.0) for k in ARM])
        self.bus.tick(n=60)
        self.arm: A1XArm | None = None

        def sleep(secs):
            self.clock.t += secs
            self.bus.tick(secs)

        def create(cfg):
            self.arm = A1XArm(cfg, bus=self.bus, clock=self.clock, threaded=False, sleep=sleep,
                              connect_timeout_s=0.05)
            return self.arm

        monkeypatch.setattr(rt, "create_backend", create)
        self.twin = rt.Twin(Workspace(robot="a1x"))
        self.twin.connect("a1x", port="fake", threaded=False)
        self.relay = relay
        self.faults: list[str] = []
        self.driver: SpecsReach | None = None
        self.pump(0.5)

    def tick(self) -> None:
        assert self.arm is not None
        for _ in range(int(round(DT / STEP))):
            self.clock.t += STEP
            self.bus.tick(STEP)
            self.arm.io_step(self.clock.t)
        self.twin.step(DT)
        self.faults.extend(self.twin.status.faults)

    def pump(self, secs: float, until=None) -> bool:
        """Takty petli przez `secs` s zegara sciennego (albo do `until()`); True - `until` spelnione."""
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < secs:
            if until is not None and until():
                return True
            self.tick()
            n += 1
            ahead = t0 + n * DT - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)
        return until is not None and bool(until())

    def connect_driver(self, **kw) -> SpecsReach:
        kw.setdefault("mode", "real")
        kw.setdefault("real_ok", True)
        kw.setdefault("hold_off_s", 0.0)
        d = SpecsReach(self.twin, relay_url=self.relay.url, **kw)
        d.start()
        self.driver = d
        self.relay.wait_client()
        self.pump(0.3)
        return d

    def state(self, ts: int, states: set[str]) -> dict | None:
        with self.relay.cond:
            hit = [s for s in self.relay.states if s["ts"] == ts and s["state"] in states]
        return hit[-1] if hit else None

    def wait_state(self, ts: int, states: set[str], secs: float) -> dict:
        assert self.pump(secs, lambda: self.state(ts, states) is not None), \
            f"brak stanu {states}: {[(s['state'], s.get('reason')) for s in self.relay.states]}"
        st = self.state(ts, states)
        assert st is not None
        return st

    def cmd_frames(self, since: int = 0) -> list[list[float]]:
        return [self.bus.pdes_deg(d) for i, d, _ in self.bus.tx[since:] if i == P.CMD_ID]

    def close(self) -> None:
        if self.driver is not None:
            # Watek wykonawcy moze czekac na petle (dojechanie) - kreci ja, az kierowca skonczy.
            import threading
            t = threading.Thread(target=self.driver.stop)
            t.start()
            self.pump(6.0, lambda: not t.is_alive())
            t.join(1.0)
        self.twin.close()


@pytest.fixture
def relay():
    r = FakeRelay()
    yield r
    r.close()


@pytest.fixture
def rig(monkeypatch, relay):
    r = Rig(monkeypatch, relay)
    yield r
    r.close()


def far_target(rig: Rig) -> np.ndarray:
    """Cel dloni ~20 st. na J1 od srodka pudelka: rampa trwa ponad sekunde."""
    return rig.vol.center + np.array([0.0, 0.2, -0.05])


@pytest.mark.parametrize("mode,real_ok", [("sim", True), ("real", False)])
def test_sim_mode_or_real_without_consent_on_the_a1x_is_not_armed(rig, mode, real_ok):
    rig.connect_driver(mode=mode, real_ok=real_ok)
    ts = rig.relay.target(rig.vol.center + np.array([0.02, 0.02, 0.0]))
    st = rig.wait_state(ts, {"rejected", "moving", "reached"}, 5.0)
    assert (st["state"], st.get("reason")) == ("rejected", "not armed")
    rig.pump(0.3)
    assert rig.cmd_frames() == []                                   # zadnej ramki 0x050
    assert rig.twin.owner is None


def test_effort_spike_mid_ramp_backs_off_slowly_toward_the_earlier_pose(rig):
    d = rig.connect_driver()
    trip: dict = {}
    real_backoff = d._backoff

    def spy(t):
        rig.bus.eff[:6] = [0.0] * 6                                # sila minela - wycofanie bez drugiego uderzenia
        trip["hit"] = d._impact
        trip["tx"] = len(rig.bus.tx)
        return real_backoff(t)
    d._backoff = spy
    ts = rig.relay.target(far_target(rig))
    rig.wait_state(ts, {"moving"}, 5.0)
    rig.pump(0.6)
    before = dict(rig.twin.status.measured)
    rig.bus.eff[0] = 60.0                                           # > progu 20 w skali ramki 0x052
    st = rig.wait_state(ts, {"reached", "hold", "backoff", "rejected"}, 15.0)
    assert (st["state"], st.get("reason")) == ("backoff", "impact effort arm_joint1"), rig.relay.states_for(ts)
    hit = trip["hit"]
    assert hit is not None and hit.backoff_pose
    frames = rig.cmd_frames(trip["tx"])
    assert frames
    j1 = [f[0] for f in frames]
    # Wstecz: J1 wraca w strone pozy sprzed uderzenia i konczy w pozie wycofania.
    assert abs(j1[-1] - hit.backoff_pose["arm_joint1"]) < 0.5
    ahead = np.sign(before["arm_joint1"] - rig.start["arm_joint1"])              # kierunek jazdy do celu
    assert ahead != 0 and (j1[-1] - j1[0]) * ahead < -2.0                           # co najmniej 2 st. wstecz
    # Wolno: szczyt smoothstep 1,5 x 8 = 12 st./s (+ drgania zegara testu) w oknach 0,2 s,
    # nigdy szybciej niz granica nadzoru 15 st./s.
    win = int(round(0.2 / STEP))
    rates = [abs(j1[i + win] - j1[i]) / 0.2 for i in range(len(j1) - win)]
    assert rates and max(rates) <= 13.0, max(rates)
    # Strumien backendu goni rozkaz petli skokami (90 st./s, co 4 ramki) - predkosc tylko w oknach.
    win = int(round(0.1 / STEP))
    fast = [abs(j1[i + win] - j1[i]) / 0.1 for i in range(len(j1) - win)]
    assert max(fast) <= 15.0 + 2 * LSB_DEG / 0.1, max(fast)
    assert rig.twin.safety_state.value != "estop"
    assert rig.twin.owner is None


def test_stall_at_a_wall_trips_the_guard_before_the_backend_deaf_fault(rig):
    d = rig.connect_driver()
    tripped: list[float] = []
    real_backoff = d._backoff

    def spy(t):
        tripped.append(rig.clock.t)
        return real_backoff(t)
    d._backoff = spy
    target = far_target(rig)
    ts = rig.relay.target(target)
    rig.wait_state(ts, {"moving"}, 5.0)
    rig.pump(0.3)
    # J1 staje 2 st. dalej niz teraz, w kierunku jazdy.
    q = rig.bus.q[0]
    sign = 1.0 if rig.bus.p_des[0] >= q else -1.0
    rig.bus.wall = (0, q + sign * np.radians(2.0), sign)
    st = rig.wait_state(ts, {"reached", "hold", "backoff", "rejected"}, 15.0)
    rig.bus.wall = None
    assert (st["state"], st.get("reason")) == ("backoff", "impact lag arm_joint1"), rig.relay.states_for(ts)
    assert tripped
    assert not any("gluche" in f for f in rig.faults), rig.faults
    rig.pump(1.5)                                                   # po wycofaniu tez bez bledu serwa
    assert not any("gluche" in f for f in rig.faults), rig.faults
    assert rig.twin.safety_state.value != "estop"


def test_stop_mid_ramp_holds_releases_and_streams_only_the_held_pose(rig):
    d = rig.connect_driver()
    ts = rig.relay.target(far_target(rig))
    rig.wait_state(ts, {"moving"}, 5.0)
    rig.pump(0.6)
    rig.twin.estop("test")
    st = rig.wait_state(ts, {"reached", "hold", "backoff", "rejected"}, 5.0)
    assert (st["state"], st.get("reason")) == ("hold", "stop")
    assert rig.twin.owner is None
    rig.pump(0.2)
    n = len(rig.bus.tx)
    rig.pump(1.0)
    after = rig.cmd_frames(n)
    held = after[0] if after else None
    assert all(max(abs(a - b) for a, b in zip(f, held)) <= 2 * LSB_DEG for f in after)   # zadnej jazdy
    assert d._worker is not None and d._worker.is_alive()


def test_relay_loss_mid_move_in_real_mode_holds(rig):
    d = rig.connect_driver()
    ts = rig.relay.target(far_target(rig))
    rig.wait_state(ts, {"moving"}, 5.0)
    rig.pump(0.4)
    rig.relay.drop_clients()
    assert rig.pump(3.0, lambda: (d.last_state or {}).get("state") == "hold")
    assert d.last_state["reason"] == "relay"
    assert rig.twin.owner is None
    rig.pump(0.3)
    cmd = dict(rig.twin.status.command)
    rig.pump(0.8)
    assert max(abs(rig.twin.status.command[k] - cmd[k]) for k in ARM) < 0.2          # rozkaz stoi


@pytest.mark.parametrize("left_deg", [6.0, 7.0, 8.0])
def test_start_center_stall_near_the_end_holds_before_the_backend_deaf_fault(monkeypatch, relay, left_deg):
    """`start_center` 10 st./s, J1 staje 2 st. dalej, gdy do konca rampy zostalo `left_deg`:
    uchyb konczy sie na ok. 4-6 st. - do starego progu 6 st., nad "gluchym" 3 st. backendu.
    Z 6 st. / 0,3 s (jak zwykly ruch) backend liczyl juz do 0,74 s z 1 s; z 4 st. / 0,2 s - do 0,34 s."""
    rig = Rig(monkeypatch, relay, offset={"arm_joint1": -15.0})
    try:
        d = SpecsReach(rig.twin, relay_url=relay.url, mode="real", real_ok=True)
        goal = rig.start["arm_joint1"]
        trip: dict = {}
        real_interrupt = d._interrupt

        def spy(cause):
            if cause == "impact" and "deaf_age" not in trip:
                # Jak dlugo backend liczy juz "gluchy" staw w chwili wyzwolenia straznika.
                trip["deaf_age"] = max((rig.clock.t - t0 for t0 in rig.arm._deaf_since.values()), default=0.0)
            real_interrupt(cause)
        d._interrupt = spy
        out: dict = {}
        th = threading.Thread(target=lambda: out.setdefault("r", d.start_center(10.0)))
        th.start()
        assert rig.pump(5.0, lambda: goal - rig.twin.status.measured["arm_joint1"] < left_deg)
        rig.bus.wall = (0, rig.bus.q[0] + np.radians(2.0), 1.0)
        assert rig.pump(6.0, lambda: not th.is_alive()), "start_center nie skonczyl"
        th.join(1.0)
        assert out["r"] == "impact lag arm_joint1", (out, rig.faults)
        assert not any("gluche" in f for f in rig.faults), rig.faults
        assert trip["deaf_age"] <= 1.0 - 0.3, trip                     # co najmniej 0,3 s przed bledem backendu
        rig.pump(1.5)                                                   # trzyma - bez wycofania i bez bledu
        rig.bus.wall = None
        assert not any("gluche" in f for f in rig.faults), rig.faults
        meas, cmd = rig.twin.status.measured, rig.twin.status.command
        assert max(abs(cmd[k] - meas[k]) for k in ARM) < 1.0, (cmd, meas)
        assert goal - meas["arm_joint1"] > 2.0                         # zostal przy scianie
        assert rig.twin.safety_state.value != "estop"
        assert rig.twin.owner is None
    finally:
        rig.close()
