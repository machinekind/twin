"""Identyfikacja dynamiki serw: nagranie z symulacji o ZNANEJ dynamice ma dac ja z powrotem.

Prawdy sa celowo "krzywe": poza siatka startowa dopasowania i z kp oraz tarciem
innymi niz w modelu. Dopasowanie ma oddac to, co z tego ruchu da sie wyznaczyc
(tlumienie, armatura, opoznienie), a kp i tarcia NIE udawac - stary test
sprawdzal tylko tlumienie i opoznienie na prawdzie z siatki startowej i nie
zauwazyl kp 0,72 przy prawdzie 1,0. Potem test jechal na skroconym pobudzeniu
(2 stawy), a na PELNYM - tym, ktore puszcza panel - pierwsza z jego prawd
dawala armature -13 % i opoznienie +5 ms przy "pasmie" +-5 % i +-5 ms.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.rl.randomize import Dynamics, Randomization  # noqa: E402
from lerobot_mp.twin.rl.sysid import (  # noqa: E402
    IDENTIFIED, JOINTS, OWNER, Recording, Replayer, excitation, identify, plan_target, record,
)
from lerobot_mp.twin.robots import SO101  # noqa: E402


def synthetic_recording(truth: Dynamics, delay: float, joints=None, hold=0.7):
    """Nagranie jak z `record`: `joints=None` - pelne pobudzenie panelu (`excitation(home)`)."""
    plan = excitation(SO101.home) if joints is None else excitation(SO101.home, joints=joints, hold=hold)
    hz = 50.0
    t = np.arange(0, plan[-1][0] + 0.5, 1 / hz)
    cur = np.array([SO101.home[j] for j in JOINTS])
    cmd = []
    for tk in t:                                   # jak nadzor: cel z planu, najwyzej 120 st/s
        want = np.array([plan_target(plan, tk)[j] for j in JOINTS])
        cur = cur + np.clip(want - cur, -120 / hz, 120 / hz)
        cmd.append(cur.copy())
    cmd = np.array(cmd)
    q = Replayer().run(Recording(t, cmd, np.full_like(cmd, np.nan)), truth, delay)
    meas = q + np.random.default_rng(0).normal(0, 0.05, q.shape)
    meas[1::2] = np.nan                             # serwa czytane co drugi takt
    return Recording(t, cmd, meas, {"backend": "test", "time": "test"})


def test_identification_on_the_panel_excitation_is_as_good_as_it_says():
    truth, delay = Dynamics(kp=0.8, damping=1.4, armature=1.3, frictionloss=0.7), 0.055  # z przegladu: armatura -13 %
    rec = synthetic_recording(truth, delay)
    out = identify(rec)
    dyn = out.dyn
    assert dyn.fit_deg < 0.1 * out.base              # symulacja po dopasowaniu wyraznie blizej nagrania
    assert dyn.fit_deg < 0.15                        # blisko szumu (0,05 st.) mimo kp i tarcia z modelu
    # Dokladnosc zmierzona na 13 prawdach (kp 0,7-1,3, tarcie 0,55-1,8): tlumienie do 10,5 %,
    # armatura do 8,2 %, opoznienie do 1,9 ms. Tu z zapasem, ale ciasniej niz stary wynik (13 %, 5 ms).
    assert dyn.damping == pytest.approx(truth.damping, rel=0.12)
    assert dyn.armature == pytest.approx(truth.armature, rel=0.1)
    assert dyn.delay / 20 == pytest.approx(delay, abs=0.003)
    # Niepewnosc mowi prawde: blad kazdego dopasowanego parametru miesci sie w jego pasmie,
    # a pasmo nie jest "nieskonczone na wszelki wypadek".
    for name in ("damping", "armature"):
        err = abs(getattr(dyn, name) / getattr(truth, name) - 1)
        assert err <= out.band[name] <= 0.4, (name, err, out.band[name])
    assert abs(dyn.delay / 20 - delay) <= out.band["delay"] <= 0.01
    # Pasmo idzie z dynamika do workspace'u (opoznienie w taktach polityki, jak `dyn.delay`).
    assert dyn.band == {"damping": out.band["damping"], "armature": out.band["armature"],
                        "delay": pytest.approx(20 * out.band["delay"])}
    print("idealne", {k: round(v, 3) for k, v in out.band.items()}, out.reasons)
    # kp i tarcie nie sa wyznaczalne z tego ruchu - zostaja z modelu, wynik i opis to mowia.
    assert dyn.kp == 1.0 and dyn.frictionloss == 1.0
    assert dyn.fitted == IDENTIFIED == ("damping", "armature", "delay")
    assert "z modelu" in dyn.source and "kp" in dyn.source and "frictionloss" in dyn.source
    assert np.isinf(out.band["kp"]) and np.isinf(out.band["frictionloss"])
    # Prawdziwe ramie lezy w swiatach treningu wokol wyniku - WSZYSTKIE parametry, takze kp
    # i tarcie (stare kp 0,85-1,15 wokol 1,0 nie zawieralo kp 0,7 / 0,8 / 1,3).
    rand = Randomization.around(dyn)
    for name in ("kp", "damping", "armature", "frictionloss"):
        lo, hi = getattr(rand, name)
        assert lo * getattr(dyn, name) <= getattr(truth, name) <= hi * getattr(dyn, name), name
    assert rand.min_delay <= delay * 20 <= rand.max_delay


# ------------------------------------------------------------ nagranie na blizniaku
@pytest.fixture
def twin():
    from lerobot_mp.twin.runtime import Twin
    from lerobot_mp.twin.workspace import Workspace

    tw = Twin(Workspace())
    try:
        yield tw
    finally:
        tw.close()


def short_plan():
    home = dict(SO101.home)
    return [(0.0, home), (0.3, dict(home, shoulder_pan=5.0)), (0.6, home)]


def test_record_does_not_take_back_an_arm_freed_by_home(twin):
    """Dom wcisniety tuz po "Identyfikuj", zanim watek zadania doszedl do `record`.

    Zmierzone przed poprawka: `record` bral zwolnione ramie z powrotem, `claim`
    przerywal rampe do domu, a dojazd jechal jako "identyfikacja". Flage przerwania
    zadania czyscil jeszcze `Job.start` - wiec bez niej tez nie wolno ruszyc.
    """
    with twin.lock:
        twin.scene.set_joints(dict(SO101.home, shoulder_pan=40.0))
    twin.connect("sim")
    cancel = threading.Event()
    twin.claim(OWNER, preempt=cancel.set)            # panel bierze ramie dla identyfikacji
    twin.home()                                      # operator: Dom
    assert cancel.is_set() and twin.owner is None
    with pytest.raises(RuntimeError, match="przerwana"):
        record(twin, short_plan(), settle=0.2, should_stop=cancel.is_set)
    cancel.clear()                                   # jak po `Job.start`
    for kw in ({}, {"take": False}):
        with pytest.raises(RuntimeError, match="przerwana"):
            record(twin, short_plan(), settle=0.2, should_stop=cancel.is_set, **kw)
        assert twin.owner is None
    assert twin.safety_state.value == "HOMING"       # rampy do domu nikt nie przerwal


# ------------------------------------------------------------ nagranie z czasem prawdziwego ramienia
class JitterArm:
    """Petla ramienia z czasem jak na prawdziwym ramieniu (s17 weryfikatora), w zegarze wirtualnym.

    30 Hz z jitterem 0-8 ms i 3 % przestojow 30 ms, serwa czytane co drugi takt, odczyt
    odpowiada pozycji 0-4 ms po poczatku taktu, status widac dopiero po koncu taktu.
    Rozkazy ida za planem pobudzenia jak nadzor (najwyzej 120 st./s) - odpowiedz liczy
    `Replayer` ze znana dynamika. `record` jedzie na tym przez podmieniony `sysid.time`:
    `sleep` przesuwa zegar (z narzutem rejestratora 0,5-2 ms) i puszcza takty, ktore minely.
    """

    def __init__(self, truth: Dynamics, delay: float, hz: float = 30.0, jit: float = 0.008,
                 settle: float = 0.5, seed: int = 3):
        rng = np.random.default_rng(seed)
        self.rng = rng
        plan = excitation(SO101.home)
        span = plan[-1][0] + settle + 1.0
        tau = [0.0]
        while tau[-1] < span:
            tau.append(tau[-1] + 1 / hz + rng.uniform(0, jit) + (0.03 if rng.random() < 0.03 else 0.0))
        self.tau = np.array(tau)
        cur, prev, cmd = np.array([SO101.home[j] for j in JOINTS]), 0.0, []
        for tk in self.tau:
            want = np.array([plan_target(plan, tk)[j] for j in JOINTS])
            step = 120 * max(tk - prev, 1e-3)
            prev = tk
            cur = cur + np.clip(want - cur, -step, step)
            cmd.append(cur.copy())
        cmd = np.array(cmd)
        tf = np.arange(0, span, 0.001)
        kf = np.searchsorted(self.tau, tf, side="right") - 1
        q = Replayer().run(Recording(tf, cmd[kf], np.full((len(tf), 6), np.nan)), truth, delay)
        lag = rng.uniform(0.0, 0.004, len(self.tau))           # serwo podaje pozycje chwile po starcie taktu
        meas = np.array([np.interp(self.tau + lag, tf, q[:, j]) for j in range(6)]).T
        meas += rng.normal(0, 0.05, meas.shape)
        self.shown = self.tau + lag + 0.001                     # status po koncu taktu
        self.samples, self.statuses = [], []
        m_last, mt_last = {}, 0.0
        for k, tk in enumerate(self.tau):
            fresh = k % 2 == 0
            if fresh:
                m_last, mt_last = dict(zip(JOINTS, map(float, meas[k]))), float(tk)
            sent = dict(zip(JOINTS, map(float, cmd[k])))
            self.samples.append(SimpleNamespace(t=float(tk), sent=sent, measured=m_last, measured_t=mt_last,
                                                fresh=fresh))
            self.statuses.append(SimpleNamespace(connected=True, error="", backend="jitter", measured=m_last,
                                                 measured_t=mt_last, command=sent))
        self.now, self.fired = -0.2, 0
        self.listeners: dict[int, object] = {}
        self.add_tick_listener = self._add
        self.remove_tick_listener = self.listeners.pop
        self.connected, self.loop_hz, self.owner, self.preempt_reason = True, hz, None, ""
        self.safety_state = SimpleNamespace(value="IDLE")
        self.time = SimpleNamespace(monotonic=lambda: self.now, sleep=self._sleep, strftime=lambda f: "test")

    def _add(self, fn):
        self.listeners[id(fn)] = fn
        return id(fn)

    def _sleep(self, d: float) -> None:
        self.now += d + self.rng.uniform(0.0005, 0.002)
        while self.fired < len(self.tau) and self.shown[self.fired] <= self.now:
            for fn in list(self.listeners.values()):
                fn(self.samples[self.fired])
            self.fired += 1

    @property
    def status(self):
        k = int(np.searchsorted(self.shown, self.now, side="right")) - 1
        return self.statuses[k] if k >= 0 else SimpleNamespace(
            connected=True, error="", backend="jitter", measured={}, measured_t=0.0, command=dict(SO101.home))

    def joints(self):
        return dict(SO101.home)

    def claim(self, owner, preempt=None):
        self.owner = owner

    def release(self, owner):
        self.owner = None

    def move(self, *a, **kw):
        self._sleep(0.2)

    def set_target(self, joints, owner=None):
        pass

    def set_engaged(self, on, owner=None):
        pass


def test_record_under_real_arm_timing_keeps_the_documented_accuracy(monkeypatch):
    """Petla z jitterem i rzadszym odczytem serw: nagranie przez `record` ma dac te sama dynamike.

    Zmierzone przed poprawka (s17 weryfikatora, ta sama prawda): `record` stemplowal wiersze
    zegarem rejestratora - tlumienie +62 % (pasmo 35 %), armatura -11 %, opoznienie -7,5 ms,
    blad 0,35 st. Nagranie w taktach petli: tlumienie i armatura w kilku procentach.
    """
    from lerobot_mp.twin.rl import sysid

    truth, delay = Dynamics(kp=0.85, damping=1.1, armature=0.9, frictionloss=1.2), 0.03
    arm = JitterArm(truth, delay)
    monkeypatch.setattr(sysid, "time", arm.time)
    rec = record(arm, excitation(SO101.home), settle=0.5)
    assert rec.meta["stamps"] == "tick"
    assert not arm.listeners                                   # sluchacz zdjety po nagraniu
    out = identify(rec)
    dyn = out.dyn
    errs = {"damping": dyn.damping / truth.damping - 1, "armature": dyn.armature / truth.armature - 1,
            "delay_ms": 1000 * (dyn.delay / 20 - delay)}
    print("jitter", {k: round(v, 3) for k, v in errs.items()},
          {k: round(v, 3) for k, v in out.band.items()}, round(dyn.fit_deg, 3), round(out.base, 3))
    assert dyn.fit_deg < 0.2 * out.base and dyn.fit_deg < 0.2
    assert abs(errs["damping"]) <= 0.15 and abs(errs["armature"]) <= 0.12 and abs(errs["delay_ms"]) <= 3.0
    for name in ("damping", "armature"):
        assert abs(errs[name]) <= out.band[name], (name, errs[name], out.band[name])
    assert abs(errs["delay_ms"]) / 1000 <= out.band["delay"]
    assert out.useful, out.reasons
