"""Petla blizniaka na prawdziwym (tu: udawanym) ramieniu - wlasnosc ramienia, STOP, bledy serw.

Wiekszosc testow idzie bez watku petli (`connect(..., threaded=False)` + `step`),
w czasie symulowanym: wynik nie zalezy od obciazenia procesora.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("mujoco")

from conftest import BOT  # noqa: E402

from machinekind_twin import runtime as rt  # noqa: E402
from machinekind_twin.backends import backend_context  # noqa: E402
from machinekind_twin.backends.base import RobotBackend, RobotInfo  # noqa: E402
from machinekind_twin.clock import VirtualClock  # noqa: E402
from machinekind_twin.robots import get_spec  # noqa: E402
from machinekind_twin.runtime import Twin  # noqa: E402
from machinekind_twin.workspace import Workspace  # noqa: E402

SPEC = get_spec(BOT)

DT = 0.02


class FakeArm(RobotBackend):
    """Ramie "prawdziwe" (simulated=False): stawy ida za rozkazem, chyba ze sa zablokowane.

    `vmax` - serwo jedzie do celu najwyzej tyle st./s (w `step`), zamiast byc tam od razu.
    `silent` - ile kolejnych odczytow bez odpowiedzi: jak `FeetechArm`, oddaje wtedy stare
    pozycje, ustawia `_link_silent` i nic nie wysyla. `glitch` - nadpisania kolejnych
    odczytow (przeklamana ramka z poprawna suma kontrolna).

    `delay` - odczyt pokazuje poze sprzed tylu sekund (wiek pomiaru wzgledem najnowszego rozkazu:
    okres odczytu + most; pesymistycznie, bo prawdziwa probka jest mlodsza niz chwila zapytania
    w petli), `amax` - serwo
    rozpedza sie i hamuje najwyzej tyle st./s^2 (jak STS3215: P do celu, limit predkosci
    i przyspieszenia, kroki 1 ms). `vmax_of` - inna predkosc dla wybranych stawow.
    `late` - nastepny swiezy odczyt zwraca te poze (spozniona odpowiedz sprzed przestoju).
    """

    def __init__(self, start: dict[str, float] | None = None, vmax: float | None = None,
                 delay: float = 0.0, amax: float | None = None, vmax_of: dict[str, float] | None = None):
        self.info = RobotInfo(name="udawane", simulated=False)
        self.pos = dict(SPEC.home, **(start or {}))
        self.goal = dict(self.pos)
        self.vmax = vmax
        self.vmax_of = dict(vmax_of or {})
        self.delay, self.amax = delay, amax
        self.vel = {k: 0.0 for k in self.pos}
        self.t = 0.0
        self.history: list[tuple[float, dict[str, float]]] = [(0.0, dict(self.pos))]
        self.late: dict[str, float] | None = None
        self.blocked: dict[str, float] = {}
        #: Twarde oparcie stawu (lo, hi): serwo dalej nie pojedzie, ale do niego tak.
        self.wall: dict[str, tuple[float, float]] = {}
        self.sent: list[dict[str, float]] = []
        self.fault_list: list[str] = []
        self.limits: dict[str, tuple[float, float]] = {}
        self.clamp: dict[str, tuple[float, float]] = {}
        #: Like `FeetechArm`: clamps commands to the limits it got (`BackendContext`, `set_joint_limits`).
        self.clamp_to_limits = False
        self.backend_limits: dict[str, tuple[float, float]] = {}
        self.silent = 0
        self._link_silent = False
        self._last_read = dict(self.pos)
        self.glitch: list[dict[str, float]] = []
        self.grip_range: tuple[float, float] | None = None
        #: Jak A1X bez `gripper_fb_*`: `joints()` oddaje rozkaz, a przed nim zgadywanke (`start`).
        self.grip_measured = True
        self._grip_cmd: float | None = None     # jak A1X: rozkaz rowny zglaszanemu nic nie ustala
        self._grip_sent = False                 # jak A1X `_g_target`: strumien chwytaka ruszyl
        self._on = False

    def connect(self):
        self._on = True

    def disconnect(self):
        self._on = False

    @property
    def is_connected(self):
        return self._on

    def read_joints(self):
        if self.silent > 0:
            self.silent -= 1
            self._link_silent = True
            return dict(self._last_read)
        self._link_silent = False
        out = dict(self.pos)
        if self.delay > 0:
            old = [p for t, p in self.history if t <= self.t - self.delay]
            out = dict(old[-1] if old else self.history[0][1])
        if self.late is not None:
            out, self.late = dict(self.late), None
        if self.glitch:
            out.update(self.glitch.pop(0))
        self._last_read = out
        return dict(out)

    def send_joints(self, targets):
        if self._link_silent:
            return {}
        out = {}
        for k, v in targets.items():
            lo, hi = self.clamp.get(k, (-1e9, 1e9))
            if self.clamp_to_limits:
                lo, hi = max(lo, self.backend_limits[k][0]), min(hi, self.backend_limits[k][1])
            out[k] = min(max(v, lo), hi)
        self.sent.append(out)
        if "gripper" in out and (self._grip_sent or abs(out["gripper"] - self.pos["gripper"]) > 0.5):
            self._grip_cmd, self._grip_sent = out["gripper"], True
        for k, v in out.items():
            self.goal[k] = v
            if self.vmax is None:
                self.pos[k] = self._walled(k, self.blocked.get(k, v))
        return out

    def step(self, dt):
        self.t += dt
        if self.vmax is None:
            self._record()
            return
        if self.amax is None:
            for k, g in self.goal.items():
                vm = self.vmax_of.get(k, self.vmax)
                p = self.pos[k] + float(np.clip(g - self.pos[k], -vm * dt, vm * dt))
                self.pos[k] = self._walled(k, self.blocked.get(k, p))
            self._record()
            return
        n = max(1, int(round(dt / 0.001)))
        h = dt / n
        da = self.amax * h
        for _ in range(n):
            for k, g in self.goal.items():
                vm = self.vmax_of.get(k, self.vmax)
                vd = min(max(25.0 * (g - self.pos[k]), -vm), vm)
                self.vel[k] += min(max(vd - self.vel[k], -da), da)
                p = self.pos[k] + self.vel[k] * h
                if k in self.blocked:
                    p, self.vel[k] = self.blocked[k], 0.0
                if self._walled(k, p) != p:
                    p, self.vel[k] = self._walled(k, p), 0.0
                self.pos[k] = p
        self._record()

    def _walled(self, k, v):
        lo, hi = self.wall.get(k, (-1e9, 1e9))
        return min(max(v, lo), hi)

    def _record(self):
        if self.delay > 0:
            self.history.append((self.t, dict(self.pos)))
            self.history = [h for h in self.history if h[0] >= self.t - self.delay - 0.1] or self.history[-1:]

    def faults(self):
        return list(self.fault_list)

    def joint_limits(self):
        return dict(self.limits)

    def gripper_range_q(self):
        return self.grip_range

    def set_joint_limits(self, limits):
        self.backend_limits.update(limits)

    def gripper_measured(self):
        return self.grip_measured

    def gripper_known(self):
        return self.grip_measured or self._grip_cmd is not None

    def assume_gripper(self, v):
        """Like galaxeo's A1X backend: only without gripper feedback and before the gripper stream started."""
        if self.grip_measured or self._grip_sent:
            return False
        self._grip_cmd = v
        self.pos["gripper"] = self.goal["gripper"] = self._last_read["gripper"] = float(v)
        return True


@pytest.fixture
def twin():
    # Petla bez watku na zegarze wirtualnym: `step(dt)` go przesuwa (`connect(threaded=False)` bez
    # `clock` tez by go zalozyl - tu jawnie).
    tw = Twin(Workspace(robot=BOT), clock=VirtualClock(t0=0.0))
    try:
        yield tw
    finally:
        tw.close()
    assert_on_time(tw)


def assert_on_time(tw) -> None:
    """Zadna bariera zegara nie przekroczyla czasu (watek spozniony o takt psuje pozniejsze asercje)."""
    assert tw.clock.late == 0, f"VirtualClock: {tw.clock.late} barier(y) po czasie - watek nie zaparkowal (ostrzezenie w logu)"


def connect_fake(twin, monkeypatch, arm: FakeArm, **kw):
    def make(cfg, spec):
        arm.backend_limits = dict(backend_context(cfg, spec).limits)
        return arm
    monkeypatch.setattr(rt, "create_backend", make)
    twin.connect("feetech", port="COM_TEST", threaded=False, **kw)


def run(twin, seconds):
    for _ in range(int(round(seconds / DT))):
        twin.step(DT)


def move_stepped(twin, joints, max_s: float = 15.0, **kw) -> None:
    """`Twin.move` w watku pomocniczym, a petla krokami `step` na zegarze wirtualnym: przejazd trwa tyle,
    ile obliczenia, a nie sekundy sciany. Wyjatek z `move` idzie dalej."""
    out: dict = {}

    def go():
        with twin.clock.member():                          # kazdy krok rampy przed taktem petli
            try:
                twin.move(joints, **kw)
            except Exception as exc:                       # noqa: BLE001 - rzucony dalej ponizej
                out["exc"] = exc
    th = threading.Thread(target=go, daemon=True)
    th.start()
    for _ in range(int(round(max_s / DT))):
        if not th.is_alive():
            break
        twin.step(DT)
    th.join(5.0)
    assert not th.is_alive(), "move nie skonczyl sie"
    if "exc" in out:
        raise out["exc"]


# ------------------------------------------------------------ wlasnosc ramienia
def test_home_preempts_the_owner_and_the_arm_does_not_snap_back(twin, monkeypatch):
    """Pozycja domowa w trakcie polityki: po rampie ramie zostaje w domu, a nie wraca do celu polityki.

    Zmierzone przed poprawka: 57,8 -> 1,3 (rampa), potem 6 -> 45 -> 60 st. w pol sekundy.
    """
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    called = []
    twin.claim("polityka", preempt=lambda: called.append(True))
    twin.set_engaged(True)
    twin.set_target({"j1": 60.0}, owner="polityka")
    run(twin, 1.0)
    assert arm.pos["j1"] > 55
    twin.home()
    assert called == [True] and twin.owner is None
    with pytest.raises(RuntimeError, match="odebrane"):
        twin.set_target({"j1": 60.0}, owner="polityka")   # spozniony takt starej polityki
    twin.set_target({"j1": 60.0})                          # albo ktos bez wlasnosci
    run(twin, 3.5)
    tail = []
    for _ in range(50):
        twin.step(DT)
        tail.append(arm.pos["j1"])
    assert max(abs(v - SPEC.home["j1"]) for v in tail) < 0.5


def test_move_belongs_to_the_calibration_and_refuses_a_foreign_owner():
    """Fala (`move`) i polityka nie moga pisac celu na zmiane: fala bierze ramie, polityka
    wtedy nie rusza, a fala nie ruszy, gdy ramie ma polityka."""
    tw = Twin(Workspace(robot=BOT))
    try:
        tw.connect("sim")
        tw.move({"j1": 5.0}, duration=0.2, settle=0.05)
        assert tw.owner == "kalibracja"
        with pytest.raises(RuntimeError, match="ramie zajete: kalibracja"):
            tw.claim("polityka")
        tw.home()                                         # koniec fali oddaje ramie
        assert tw.owner is None
        tw.claim("polityka")
        with pytest.raises(RuntimeError, match="ramie zajete: polityka"):
            tw.move({"j1": 10.0}, duration=0.2)
        tw.release("polityka")
        # Fala, ktorej ramie odebrano miedzy przejazdami, nie bierze wolnego ramienia na nowo.
        with pytest.raises(RuntimeError, match="ramie odebrane"):
            tw.move({"j1": 10.0}, duration=0.2, take=False)
        assert tw.owner is None
    finally:
        tw.close()


def test_estop_holds_the_measured_pose_not_the_last_command(twin, monkeypatch):
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    arm.blocked["j2"] = 30.0
    twin.set_engaged(True)
    twin.set_target({"j2": 55.0})
    run(twin, 1.0)
    twin.estop()
    run(twin, 0.3)
    assert twin.safety_state.value == "ESTOP"
    assert arm.sent[-1]["j2"] == pytest.approx(30.0, abs=0.5)


def test_connect_without_moving_sends_nothing_to_a_joint_outside_the_limits(twin, monkeypatch):
    """Ramie spoczywa z j2 -66 st. przy limicie -60: polaczenie bez ruchu nie moze
    skoczyc do limitu, a pierwszy ruch wraca w zakres powoli - takze przez backend, ktory
    przycina rozkaz do limitow konfiguracji."""
    arm = FakeArm({"j2": -66.0})
    arm.clamp_to_limits = True
    connect_fake(twin, monkeypatch, arm)
    run(twin, 1.0)
    assert arm.sent == []
    twin.set_engaged(True)
    twin.set_target({"j1": 5.0})
    twin.step(DT)
    assert arm.sent[0]["j2"] == pytest.approx(-66.0, abs=0.5)
    run(twin, 0.2)
    steps = np.diff([s["j2"] for s in arm.sent])
    assert np.abs(steps).max() <= 15.0 * DT * 1.5 + 1e-6
    run(twin, 1.0)
    assert arm.sent[-1]["j2"] >= -60.0 - 1e-6
    # Poszerzenie limitow backendu bylo chwilowe - staw w zakresie, limity znow z konfiguracji.
    assert arm.backend_limits["j2"][0] == pytest.approx(-60.0)
    assert twin.joint_limits()["j2"][0] == pytest.approx(-60.0)


def test_a_backend_no_plugin_provides_refuses_instead_of_simulating(twin):
    """A request that reaches `create_backend` wants hardware: a backend plugin or an error naming the
    installed ones, never a simulation."""
    with pytest.raises(ValueError, match=f"nieznany_backend.*{BOT}"):
        twin.connect("nieznany_backend", port="", threaded=False)
    assert not twin.connected
    assert "nieznany_backend" in twin.status.error


def test_config_file_keeps_foreign_sections_for_backends_but_fails_on_unknown_keys(tmp_path):
    """Top-level sections the twin does not read stay raw for backend plugins (a file copied from
    the old `configs/default.yaml` still loads), but a typo inside a section the twin reads must
    still fail instead of silently dropping a safety setting."""
    from machinekind_twin.config import TwinConfig, load_config

    old = tmp_path / "old.yaml"
    old.write_text(
        "loop_hz: 30.0\ncamera: {source: 0, mirror: true}\ntracker: {backend: auto}\n"
        "filters: {gripper_ema: 0.5}\nmapping: {mode: direct}\nclutch: {mode: gesture}\n"
        "ui: {show: true}\nsomething_else: [1, 2]\nsafety: {velocity_scale: 0.5}\n"
    )
    cfg = load_config(old)
    assert isinstance(cfg, TwinConfig)
    assert cfg.safety.velocity_scale == 0.5
    assert cfg.sections["camera"] == {"source": 0, "mirror": True}
    assert "loop_hz" not in cfg.sections and "something_else" not in cfg.sections

    bad = tmp_path / "bad.yaml"
    bad.write_text("safety: {bogus: 1}\n")
    with pytest.raises(ValueError, match="bogus"):
        load_config(bad)


# ------------------------------------------------------------ bledy serw, przerwy, limity
def test_servo_fault_triggers_estop_and_preempts_the_owner(twin, monkeypatch):
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    called = []
    twin.claim("identyfikacja", preempt=lambda: called.append(True))
    twin.set_engaged(True)
    run(twin, 0.2)
    arm.fault_list = ["j2: przeciazenie"]
    twin.step(DT)
    assert twin.safety_state.value == "ESTOP"
    assert "przeciazenie" in twin.status.error
    assert twin.owner is None
    deadline = time.monotonic() + 2.0                      # wywolanie zwrotne z watku petli idzie osobno
    while not called and time.monotonic() < deadline:
        time.sleep(0.002)
    assert called == [True]


def test_link_loss_stops_sending(twin, monkeypatch):
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    twin.set_engaged(True)
    twin.set_target({"j1": 20.0})
    run(twin, 0.2)
    n = len(arm.sent)
    arm.fault_list = ["brak odpowiedzi serw od 300 ms"]
    run(twin, 0.5)
    assert len(arm.sent) == n                              # nic nie czeka w buforze gniazda


def test_a_long_loop_gap_is_not_turned_into_a_command_jump(twin, monkeypatch):
    """Po przestoju petli (render, odczyt z zerwanego lacza) rozkaz nie skacze o max_vel * 0,2 s."""
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    twin.claim("polityka")
    twin.set_engaged(True)
    twin.set_target({"j1": 80.0}, owner="polityka")
    run(twin, 0.1)
    before = arm.sent[-1]["j1"]
    twin.step(0.9)                                         # 0,9 s bez taktu
    after = arm.sent[-1]["j1"]
    assert after - before <= SPEC.max_vel["j1"] * 1.5 * DT + 1e-6
    assert twin.owner is None and "przerwa" in twin.status.error


def test_backend_limits_narrow_the_config_limits(twin, monkeypatch):
    arm = FakeArm()
    arm.limits = {"j2": (-200.0, 50.0)}
    connect_fake(twin, monkeypatch, arm)
    assert twin.joint_limits()["j2"] == (-60.0, 50.0)
    twin.set_engaged(True)
    twin.set_target({"j2": 60.0})
    run(twin, 2.0)
    assert arm.sent[-1]["j2"] == pytest.approx(50.0)


def test_loop_death_preempts_the_owner(twin, monkeypatch):
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    called = []
    twin.claim("identyfikacja", preempt=lambda: called.append(True))

    def boom():
        raise OSError("kabel wypadl")
    arm.read_joints = boom
    twin.step(DT)
    assert not twin.connected and "kabel" in twin.status.error
    assert called == [True] and twin.owner is None


# ------------------------------------------------------------ krotkie milczenie lacza
def stall(twin, arm, cycles: int, block: float = 0.27):
    """`cycles` odczytow bez odpowiedzi; kazdy trzyma petle `block` s (timeout odczytu przez most)."""
    arm.silent = cycles
    for _ in range(cycles):
        twin.step(block)


def test_a_short_link_stall_does_not_send_a_jump_when_the_link_returns(twin, monkeypatch):
    """Przestoj ponizej progu utraty lacza: nadzor stoi, a pierwszy rozkaz po powrocie nie skacze.

    Zmierzone przed poprawka (most, przestoj 0,9 s): rozkaz 10,5 -> 27,2 st. w jednej paczce.
    """
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    twin.claim("kalibracja")
    twin.set_engaged(True)
    twin.set_target({"j1": 80.0}, owner="kalibracja")
    run(twin, 0.1)
    n, last = len(arm.sent), arm.sent[-1]["j1"]
    t_meas = twin.status.measured_t
    stall(twin, arm, 3)
    assert len(arm.sent) == n                              # nic nie poszlo w trakcie
    # Panel i identyfikacja widza to, co naprawde poszlo, i ze pomiaru nie bylo.
    assert twin.status.command["j1"] == pytest.approx(last)
    assert twin.status.measured_t == t_meas
    run(twin, 0.5)
    goals = [last] + [s["j1"] for s in arm.sent[n:]]
    assert np.abs(np.diff(goals)).max() <= SPEC.max_vel["j1"] * 1.5 * DT + 1e-6
    assert goals[-1] > last + 20                           # ruch jedzie dalej
    assert twin.safety_state.value == "ACTIVE" and twin.owner == "kalibracja"


def squeeze(twin, monkeypatch, arm, owner="polityka"):
    """Chwytak zablokowany na kostce (30), rozkaz 5 - sciska."""
    connect_fake(twin, monkeypatch, arm)
    arm.blocked["gripper"] = 30.0
    twin.claim(owner)
    twin.set_engaged(True)
    twin.set_target({"gripper": 5.0}, owner=owner)
    run(twin, 1.0)
    assert arm.sent[-1]["gripper"] == pytest.approx(5.0)


def test_stop_home_and_preempt_keep_the_gripper_squeeze(twin, monkeypatch):
    """STOP/Dom/odebranie trzymaly zmierzony kat zablokowanej szczeki (30) = zerowa sila:
    kostka niesiona nad blatem wypadala. Stawy ramienia dalej trzymaja pomiar."""
    arm = FakeArm()
    squeeze(twin, monkeypatch, arm)
    n = len(arm.sent)
    twin.estop()
    run(twin, 3.5)
    assert max(s["gripper"] for s in arm.sent[n:]) == pytest.approx(5.0)
    assert twin.status.command["gripper"] == pytest.approx(5.0)


def test_gripper_overload_eases_the_squeeze_instead_of_estop(twin, monkeypatch):
    """Przeciazenie TYLKO chwytaka przy mocnym chwycie: bez STOP-u, szczeki luzniej, ostrzezenie."""
    arm = FakeArm()
    squeeze(twin, monkeypatch, arm)
    arm.fault_list = ["gripper (serwo 3): przeciazenie"]
    run(twin, 1.0)
    assert twin.safety_state.value == "ACTIVE" and twin.owner == "polityka"
    assert not twin.status.error
    ease = 30.0 - Twin.grip_ease
    assert arm.sent[-1]["gripper"] == pytest.approx(ease)  # wlasciciel dalej chce 5
    assert 30.0 - arm.sent[-1]["gripper"] > Twin.grip_squeeze_margin   # chwyt zostaje
    assert any("chwytak" in w and "przeciazenie" in w for w in twin.status.warnings)
    twin.set_target({"j1": 10.0}, owner="polityka")   # reszta ramienia jedzie dalej
    run(twin, 0.5)
    assert arm.sent[-1]["j1"] == pytest.approx(10.0)
    # Blad trwa mimo odciazenia - STOP, a chwytak staje na zmierzonym rozwarciu: to docisk
    # podtrzymywal przeciazenie (patrz test_gripper_fault_stop_releases_the_squeeze_...).
    run(twin, 1.0)
    assert twin.safety_state.value == "ESTOP" and "gripper" in twin.status.error
    run(twin, 0.5)
    assert arm.sent[-1]["gripper"] == pytest.approx(30.0)


def test_arm_fault_with_a_gripper_fault_stops_at_once(twin, monkeypatch):
    arm = FakeArm()
    squeeze(twin, monkeypatch, arm)
    arm.fault_list = ["gripper (serwo 3): przeciazenie", "j2 (serwo 2): przegrzanie"]
    twin.step(DT)
    assert twin.safety_state.value == "ESTOP" and "j2" in twin.status.error
    run(twin, 0.3)
    assert arm.sent[-1]["gripper"] == pytest.approx(30.0 - Twin.grip_ease)


# ------------------------------------------------------------ przeklamany odczyt przy Polacz
def test_connect_ignores_a_single_corrupt_read(twin, monkeypatch):
    """Jedna przeklamana ramka (0 tikow = -180 st.) tuz po polaczeniu nie zostaje poza startowa."""
    arm = FakeArm()
    arm.glitch = [{"j1": -180.0}]
    connect_fake(twin, monkeypatch, arm)
    assert twin.status.command["j1"] == pytest.approx(SPEC.home["j1"])
    twin.set_engaged(True)
    run(twin, 0.2)
    assert all(abs(s["j1"] - SPEC.home["j1"]) < 1.0 for s in arm.sent)


def test_set_engaged_with_an_owner_only_touches_its_own_clutch(twin, monkeypatch):
    connect_fake(twin, monkeypatch, FakeArm())
    twin.claim("panel")
    assert twin.set_engaged(True, owner="panel")
    assert not twin.set_engaged(False, owner="polityka")   # spozniony runner
    twin.step(DT)
    assert twin.status.engaged


def pan_trace(twin, arm, seconds):
    """Takty petli; (pozycje serwa pan po kazdym takcie, rozkazy pan wyslane w tym czasie)."""
    n = len(arm.sent)
    pos = []
    for _ in range(int(round(seconds / DT))):
        twin.step(DT)
        pos.append(arm.pos["j1"])
    return pos, [s["j1"] for s in arm.sent[n:]]


def test_a_late_reply_after_a_stall_does_not_become_the_reseed_pose(twin, monkeypatch):
    """Przestoj lacza w szybkim ruchu; pierwsza "swieza" odpowiedz po nim to spozniona ramka sprzed
    przestoju (TCP oddaje ja po powrocie, SYNC READ dopasowuje tylko po ID serwa).

    Zmierzone (emulator STS3215 za mostem, pan 140 st./s): reseed do niej cofal rozkaz o 6-7 st.
    i serwo zawracalo z ~90 st./s - takze po STOP-ie wcisnietym w przestoju i po utracie lacza.
    """
    arm = FakeArm(vmax=140.0)
    connect_fake(twin, monkeypatch, arm)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    twin.set_target({"j1": 90.0}, owner="panel")
    run(twin, 0.2)
    late = dict(arm.pos)                                   # poza, o ktora pytal odczyt sprzed przestoju
    run(twin, 0.12)
    arm.silent, arm.late = 8, late
    for k in range(16):                                    # 8 odczytow bez odpowiedzi (co 40 ms)
        if k == 6:
            twin.estop()
        twin.step(DT)
    at_release = arm.pos["j1"]
    assert at_release - late["j1"] > 10           # serwo dojechalo do ostatniego rozkazu
    pos, sent = pan_trace(twin, arm, 0.6)
    assert sent, "po powrocie lacza nic nie poszlo"
    assert min(sent) >= at_release - 1.0
    assert min(pos) >= at_release - 1.0


def test_reads_that_never_agree_after_a_stall_stop_the_arm(twin, monkeypatch):
    """Po przestoju odczyty co takt inne (np. przeklamane ramki) - petla dalej stoi, a po
    `confirm_timeout_s` STOP z powodem zamiast cichego "ACTIVE", w ktorym nic sie nie rusza."""
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    run(twin, 0.1)
    n = len(arm.sent)
    arm.silent = 2
    arm.glitch = [{"j1": v} for v in (10.0, -10.0) * 30]
    run(twin, 1.5)
    assert len(arm.sent) == n
    assert twin.safety_state.value == "ESTOP" and "nie zgadzaja" in twin.status.error


def test_stop_or_home_while_the_gripper_closes_does_not_keep_closing(twin, monkeypatch):
    """STOP w trakcie zamykania szczek (palec miedzy nimi): regula "sciska" brala zamykajaca sie
    szczeke za sciskajaca i trzymala pelny rozkaz zamkniecia - szczeki zamykaly sie dalej
    o 15-27 jednostek. Dom zostawial chwytak w przypadkowym rozwarciu zamiast w domu."""
    arm = FakeArm(vmax=300.0, delay=0.03, vmax_of={"gripper": 150.0})
    connect_fake(twin, monkeypatch, arm)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    twin.set_target({"gripper": 100.0}, owner="panel")
    run(twin, 1.0)
    twin.set_target({"gripper": 0.0}, owner="panel")
    run(twin, 0.12)
    at = arm.pos["gripper"]
    twin.estop()
    low = at
    for _ in range(int(3.5 / DT)):
        twin.step(DT)
        low = min(low, arm.pos["gripper"])
    assert at - low < 3.0                                  # najwyzej droga hamowania
    assert arm.pos["gripper"] >= at - 1.0


# ------------------------------------------------------------ runda 3: straznik rozjazdu
def test_tracking_guard_still_stops_a_blocked_joint_with_stale_reads(twin, monkeypatch):
    """Ten sam wolny serwo i stary odczyt: staw zatrzymany przeszkoda w pelnym biegu albo bez
    momentu od poczatku - STOP w ciagu ~1 s, ramie trzyma zmierzona poze (bez docisku)."""
    arm = FakeArm(vmax=150.0, delay=0.07, amax=2232.0)
    connect_fake(twin, monkeypatch, arm)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    arm.blocked["j1"] = arm.pos["j1"]   # bez momentu od poczatku
    twin.set_target({"j1": 80.0}, owner="panel")
    t = 0.0
    while twin.safety_state.value != "ESTOP" and t < 3.0:
        twin.step(DT)
        t += DT
    assert twin.safety_state.value == "ESTOP" and "j1" in twin.status.error
    assert t <= 1.0
    run(twin, 0.3)
    assert arm.sent[-1]["j1"] == pytest.approx(arm.blocked["j1"], abs=1.0)


# ------------------------------------------------------------ runda 3: trzymanie pozy przy STOP/Dom
def test_stop_at_speed_does_not_snap_back_after_the_stopping_distance(twin, monkeypatch):
    """STOP w pelnym biegu j1 (90 st./s): serwo hamuje najwyzej ~2230 st./s^2, czyli ~1,8 st.
    - trzymanie punktu przed ta droga hamowania cofalo je o tyle (przy 220 st./s o 6-14 st.)."""
    arm = FakeArm(vmax=250.0, amax=2232.0)
    connect_fake(twin, monkeypatch, arm)
    twin.set_engaged(True)
    twin.set_target({"j1": -80.0})
    run(twin, 2.0)
    twin.set_target({"j1": 80.0})
    run(twin, 0.52)
    twin.estop()
    trace = []
    for _ in range(40):
        twin.step(DT)
        trace.append(arm.pos["j1"])
    # P hamuje od v/kp = 3,6 st., dalej niz droga hamowania (1,8 st.): serwo nie przeregulowuje celu.
    assert max(trace) - trace[-1] < 0.5
    assert trace[-1] <= twin.status.command["j1"] + 0.5


def test_move_raises_when_the_arm_stops_short_of_the_command(monkeypatch):
    """Staw oparty 15 st. przed celem (ponizej progu straznika 25 st.) - `move` nie konczy sie
    "normalnie", tylko mowi, ktory staw nie dojechal."""
    arm = FakeArm()
    arm.wall["j1"] = (-1e9, 65.0)
    monkeypatch.setattr(rt, "create_backend", lambda cfg, spec: arm)
    tw = Twin(Workspace(robot=BOT), clock=VirtualClock())
    try:
        tw.connect("feetech", port="COM_TEST", threaded=False)
        with pytest.raises(RuntimeError, match="j1"):
            move_stepped(tw, {"j1": 80.0}, duration=0.5, settle=0.2)
    finally:
        tw.close()
    assert_on_time(tw)


# ------------------------------------------------------------ runda 3: sprzeglo z wlascicielem
def test_claim_and_clutch_are_refused_under_estop(twin, monkeypatch):
    """Polityka ladowana 0,1 s po Uruchom brala ramie POD STOP-em (verify2, s15): `claim` i sprzeglo
    tego nie sprawdzaly, a po Skasuj STOP ramie jechalo 83,7 st. pod zatrzymana polityka."""
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    twin.estop("STOP z panelu")
    with pytest.raises(RuntimeError, match="aktywny STOP"):
        twin.claim("polityka")
    assert twin.owner is None
    assert not twin.set_engaged(True, owner="polityka")
    assert not twin.set_engaged(True)
    assert not twin._engaged
    with pytest.raises(RuntimeError, match="aktywny STOP"):
        twin.move({"j1": 20.0}, duration=0.2)
    assert twin.owner is None and not twin._engaged
    twin.clear_estop()
    twin.claim("polityka")
    assert twin.set_engaged(True, owner="polityka")


def test_claim_after_a_preempt_since_the_mark_is_refused_and_home_keeps_ramping(twin, monkeypatch):
    """Dom wcisniety w trakcie startu polityki: jej `claim` przerywal rampe do domu (verify2, s15c)."""
    arm = FakeArm({"j1": 50.0})
    connect_fake(twin, monkeypatch, arm)
    mark = twin.preempt_gen
    twin.home()
    with pytest.raises(RuntimeError, match="odebrane w trakcie startu.*pozycja domowa"):
        twin.claim("polityka", gen=mark)
    with pytest.raises(RuntimeError, match="w trakcie startu"):
        twin.claim("polityka", guard=lambda: False)
    assert twin.owner is None
    run(twin, 3.5)
    assert arm.pos["j1"] == pytest.approx(SPEC.home["j1"], abs=0.5)
    twin.claim("polityka", gen=twin.preempt_gen, guard=lambda: True)
    assert twin.owner == "polityka"


# ------------------------------------------------------------ runda 3b: Polacz z kostka w szczekach
class GoalArm(FakeArm):
    """Backend, ktory zna cel serw (jak Goal_Position) - `Twin._squeeze_at_connect`."""

    def goal_positions(self):
        return dict(self.goal)


def test_reconnect_keeps_a_squeezing_gripper_closed_when_the_backend_knows_the_goal(twin, monkeypatch):
    """Polacz w trakcie lift: nadzor startowal od zmierzonego kata szczeki (30), a pierwsze sprzeglo
    wysylalo go jako cel - zerowa sila, kostka zsuwala sie 3,8 -> 2,3 cm (verify2, s1)."""
    arm = GoalArm()
    squeeze(twin, monkeypatch, arm)
    connect_fake(twin, monkeypatch, arm, go_home=False)
    assert twin.status.command["gripper"] == pytest.approx(5.0)
    n = len(arm.sent)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    twin.set_target({"j1": 5.0}, owner="panel")
    run(twin, 1.0)
    assert arm.sent[n:] and max(s["gripper"] for s in arm.sent[n:]) == pytest.approx(5.0)


def test_unmeasured_gripper_takes_its_first_command_straight_not_ramped_from_the_guess(twin, monkeypatch):
    """A1X bez pomiaru chwytaka zglasza po polaczeniu domyslne 100 (otwarty), choc szczeki
    trzymaly karte. Rampa i ogranicznik predkosci szly od tej zgadywanki do 0 - szczeki
    OTWIERALY sie i karta wypadala z chwytaka na starcie fali. Pierwszy rozkaz chwytaka idzie
    wprost do celu; nastepne jak zwykle."""
    arm = FakeArm(start={"gripper": 100.0}, vmax=300.0)
    arm.grip_measured = False
    connect_fake(twin, monkeypatch, arm)
    assert twin._ramp_start({"gripper": 0.0})["gripper"] == 0.0        # rampa `move`/`home` od celu
    n = len(arm.sent)
    twin.claim("kalibracja")
    twin.set_engaged(True)
    twin.set_target({"gripper": 0.0}, owner="kalibracja")
    run(twin, 0.3)
    grips = [s["gripper"] for s in arm.sent[n:] if "gripper" in s]
    assert grips and max(grips) == pytest.approx(0.0)                   # nigdy 100 -> 0
    # Od tej chwili pozycja jest znana: kolejny cel jedzie rampa (ogranicznik 300/s), nie skokiem.
    twin.set_target({"gripper": 60.0}, owner="kalibracja")
    twin.step(DT)
    assert 0.0 < arm.sent[-1]["gripper"] < 60.0
    assert twin._ramp_start({"gripper": 0.0})["gripper"] == pytest.approx(arm.sent[-1]["gripper"], abs=1.0)


def test_measured_gripper_keeps_its_ramp(twin, monkeypatch):
    """Chwytak z pomiarem (Feetech, sim) jedzie od pomiaru jak dotad."""
    arm = FakeArm(start={"gripper": 100.0}, vmax=300.0)
    connect_fake(twin, monkeypatch, arm)
    assert twin._ramp_start({"gripper": 0.0})["gripper"] == pytest.approx(100.0)
    n = len(arm.sent)
    twin.claim("kalibracja")
    twin.set_engaged(True)
    twin.set_target({"gripper": 0.0}, owner="kalibracja")
    twin.step(DT)
    assert 0.0 < arm.sent[-1]["gripper"] < 100.0


def test_a_target_equal_to_the_gripper_guess_does_not_make_the_position_known(twin, monkeypatch):
    """Druga fala 2026-09-26: przed pierwszym przejazdem poszedl cel z chwytakiem rownym zgadywance
    (100 - trzymanie zmierzonej pozy); lokalna flaga uznala pozycje za znana, choc backend niczego
    nie wyslal, i przejazd fali jechal rampa 100 -> 0 - szczeki znow sie otworzyly."""
    arm = FakeArm(start={"gripper": 100.0}, vmax=300.0)
    arm.grip_measured = False
    connect_fake(twin, monkeypatch, arm)
    twin.claim("kalibracja")
    twin.set_engaged(True)
    twin.set_target({"gripper": 100.0}, owner="kalibracja")     # = zgadywanka: nic nie ustala
    run(twin, 0.2)
    assert not twin._gripper_known()
    assert twin._ramp_start({"gripper": 0.0})["gripper"] == 0.0
    n = len(arm.sent)
    twin.set_target({"gripper": 0.0}, owner="kalibracja")
    run(twin, 0.3)
    grips = [s["gripper"] for s in arm.sent[n:] if "gripper" in s]
    assert grips and max(grips) == pytest.approx(0.0)
    assert twin._gripper_known()


# ------------------------------------------------------ Ustaw dom = tu: chwytak zamkniety
@pytest.fixture
def home_twin(tmp_path):
    tw = Twin(Workspace(robot=BOT, path=tmp_path / "twin.json"))
    try:
        yield tw
    finally:
        tw.close()


def test_set_home_here_closes_the_unmeasured_gripper_and_the_next_open_ramps_from_zero(home_twin, monkeypatch):
    """Dom = szczeki zamkniete. A1X bez pomiaru zglaszal zgadywanke "otwarty" (100), `set_home_here`
    zapisywal ja do domu, a Dom otwieral szczeki na koncu. Teraz dom ma chwytak 0, backend dostaje
    zalozenie, nadzor rusza od 0: nastepne "otworz" to rampa 0 -> 100 bez drgniecia w druga strone."""
    twin = home_twin
    arm = FakeArm(start={"gripper": 100.0}, vmax=300.0)
    arm.grip_measured = False
    connect_fake(twin, monkeypatch, arm)
    run(twin, 0.2)
    assert arm.sent == []                                                  # Polacz niczym nie rusza
    home = twin.set_home_here()
    assert home["gripper"] == 0.0
    assert Workspace.load(twin.workspace.path).home["gripper"] == 0.0
    assert twin._supervisor.home["gripper"] == 0.0
    assert twin._supervisor.command["gripper"] == 0.0
    assert twin._latest["gripper"] == 0.0 and twin._gripper_known()
    run(twin, 0.2)
    assert arm.sent == []                                                  # przycisk niczym nie rusza
    assert twin.status.measured["gripper"] == 0.0 and twin.status.command["gripper"] == 0.0
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    twin.set_target({"gripper": 100.0}, owner="panel")
    run(twin, 0.6)
    grips = [x["gripper"] for x in arm.sent if "gripper" in x]
    assert grips[0] < 10.0                                                 # rampa od 0, nie skok
    assert all(b >= a for a, b in zip(grips, grips[1:])) and max(grips) <= 100.0
    assert grips[-1] == pytest.approx(100.0)


def test_set_home_here_after_the_gripper_was_commanded_keeps_its_state(home_twin, monkeypatch, caplog):
    """Strumien chwytaka juz ruszyl: prawda jest ostatni rozkaz. Przestawienie nadzoru na 0 zamknelo
    by szczeki takt po kliknieciu przycisku, ktory ma tylko zapisac dom."""
    twin = home_twin
    arm = FakeArm(start={"gripper": 100.0}, vmax=300.0)
    arm.grip_measured = False
    connect_fake(twin, monkeypatch, arm)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    twin.set_target({"gripper": 50.0}, owner="panel")
    run(twin, 0.5)
    assert arm.sent[-1]["gripper"] == pytest.approx(50.0)
    n = len(arm.sent)
    with caplog.at_level("WARNING"):
        home = twin.set_home_here()
    assert "chwytak juz sterowany" in caplog.text
    assert home["gripper"] == 0.0 and twin.workspace.home["gripper"] == 0.0
    assert twin._supervisor.command["gripper"] == pytest.approx(50.0)
    run(twin, 0.4)
    assert [x["gripper"] for x in arm.sent[n:]] == pytest.approx([50.0] * (len(arm.sent) - n))


def test_set_home_here_with_a_measured_gripper_only_records_it_closed(home_twin, monkeypatch):
    twin = home_twin
    arm = FakeArm(start={"gripper": 40.0})
    connect_fake(twin, monkeypatch, arm)
    assert twin.set_home_here()["gripper"] == 0.0
    assert twin._supervisor.command["gripper"] == pytest.approx(40.0)     # pomiar jest prawda
    assert twin._supervisor.home["gripper"] == 0.0


def test_move_waits_for_a_speed_capped_command_instead_of_failing_arrival():
    """Suwak Predkosc 20 st./s ogranicza rozkaz nadzoru; obrot j1 o 60 st. potrzebuje 3 s,
    a staly termin 0,5 + 0,4 + 1,5 s konczyl przejazd bledem "rozkaz jeszcze jedzie" - fala
    z 21 i 11 obserwacjami przepadla na jednym takim przejezdzie."""
    tw = Twin(Workspace(robot=BOT), clock=VirtualClock())
    try:
        tw.connect("sim", threaded=False)
        tw.set_speed(20.0)
        pan = tw.joints()["j1"]
        move_stepped(tw, {"j1": pan + 60.0}, duration=0.5, settle=0.2)
        assert abs(tw.status.command["j1"] - (pan + 60.0)) < 1.0
        tw.home()
    finally:
        tw.close()
    assert_on_time(tw)


def test_a_partial_target_of_the_same_owner_merges_over_its_pending_target(twin, monkeypatch):
    """Sledzenie z okularow podaje co takt same stawy ramienia: rampa chwytaka z wczesniejszego celu
    tego samego wlasciciela jedzie dalej (nie staje na ostatnim wyslanym rozkazie)."""
    arm = FakeArm(start={"gripper": 100.0}, vmax=300.0)
    connect_fake(twin, monkeypatch, arm)
    pan = twin.status.command["j1"]
    twin.claim("specs")
    twin.set_engaged(True, owner="specs")
    twin.set_target({"gripper": 0.0}, owner="specs")
    twin.step(DT)
    assert 0.0 < twin.status.command["gripper"] < 100.0
    for i in range(1, 60):
        twin.set_target({"j1": pan + 0.1 * i}, owner="specs")
        twin.step(DT)
    assert twin._target["gripper"] == 0.0
    assert twin.status.command["gripper"] == pytest.approx(0.0)
    assert twin.status.command["j1"] == pytest.approx(pan + 5.9, abs=0.2)


def test_a_partial_target_does_not_merge_over_another_owners_target(twin, monkeypatch):
    arm = FakeArm(start={"gripper": 100.0}, vmax=300.0)
    connect_fake(twin, monkeypatch, arm)
    pan = twin.status.command["j1"]
    twin.claim("specs")
    twin.set_engaged(True, owner="specs")
    twin.set_target({"gripper": 0.0}, owner="specs")
    twin.step(DT)
    twin.release("specs")
    twin.claim("panel")
    twin.set_target({"j1": pan + 2.0}, owner="panel")
    g = twin.status.command["gripper"]
    assert 0.0 < g < 100.0 and twin._target["gripper"] == pytest.approx(g)     # from the ramp start
    # Without an owner: never merged either.
    twin.set_target({"gripper": 0.0}, owner="panel")
    twin.set_target({"j1": pan + 3.0})
    assert twin._target["gripper"] == pytest.approx(twin.status.command["gripper"])


def test_hold_command_stops_the_arm_where_its_command_is_and_keeps_the_gripper_goal(twin, monkeypatch):
    arm = FakeArm(start={"gripper": 100.0}, vmax=300.0)
    connect_fake(twin, monkeypatch, arm)
    pan = twin.status.command["j1"]
    twin.claim("specs")
    twin.set_engaged(True, owner="specs")
    twin.set_target({"j1": pan + 30.0, "gripper": 0.0}, owner="specs")
    run(twin, 0.1)
    held = twin.hold_command(owner="specs")
    assert held is not None and held["gripper"] == 0.0
    at = twin.status.command["j1"]
    assert pan < at < pan + 30.0 and held["j1"] == pytest.approx(at)
    run(twin, 1.0)
    assert twin.status.command["j1"] == pytest.approx(at)
    assert twin.status.command["gripper"] == pytest.approx(0.0)
    with pytest.raises(RuntimeError):
        twin.hold_command(owner="panel")
