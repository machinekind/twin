"""Identyfikacja dynamiki serw: nagrywanie na blizniaku.

Dokladnosc dopasowania (`identify`) zalezy od dynamiki modelu ramienia. Na robocie testowym
(lekkie czlony, armatura ponad ich bezwladnoscia, bez tarcia) armatury nie da sie wyznaczyc
i `identify` sam to mowi, wiec tu zostaje nagrywanie: kto ma ramie i kiedy.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("mujoco")

from conftest import BOT  # noqa: E402

from machinekind_twin.dynamics import Dynamics  # noqa: E402
from machinekind_twin.robots import get_spec  # noqa: E402
from machinekind_twin.sysid import (  # noqa: E402
    OWNER, Recording, Replayer, excitation, identify, plan_target, record,
)

SPEC = get_spec(BOT)
JOINTS = SPEC.joints


def test_identification_says_what_this_motion_cannot_determine():
    """Nagranie z symulacji o znanej dynamice, z pelnym pobudzeniem panelu: dopasowanie zbliza
    symulacje do nagrania, kp i tarcie zostaja z modelu, a armature, ktorej ten robot nie pokazuje,
    `identify` oznacza jako niewyznaczalna - wynik nie jest "przydatny"."""
    truth, delay = Dynamics(kp=0.8, damping=1.4, armature=1.3, frictionloss=0.7), 0.055
    plan = excitation(SPEC.home, joints=SPEC.arm_joints)
    hz = 50.0
    t = np.arange(0, plan[-1][0] + 0.5, 1 / hz)
    cur = np.array([SPEC.home[j] for j in JOINTS])
    cmd = []
    for tk in t:                                   # jak nadzor: cel z planu, najwyzej 120 st/s
        want = np.array([plan_target(plan, tk)[j] for j in JOINTS])
        cur = cur + np.clip(want - cur, -120 / hz, 120 / hz)
        cmd.append(cur.copy())
    cmd = np.array(cmd)
    q = Replayer(SPEC).run(Recording(t, cmd, np.full_like(cmd, np.nan), joints=JOINTS), truth, delay)
    meas = q + np.random.default_rng(0).normal(0, 0.05, q.shape)
    meas[1::2] = np.nan                             # serwa czytane co drugi takt

    out = identify(Recording(t, cmd, meas, {"backend": "test", "time": "test"}, JOINTS), SPEC)

    assert out.dyn.fit_deg < 0.1 * out.base         # symulacja po dopasowaniu wyraznie blizej nagrania
    assert out.dyn.kp == 1.0 and out.dyn.frictionloss == 1.0
    assert np.isinf(out.band["kp"]) and np.isinf(out.band["frictionloss"])
    assert np.isinf(out.band["armature"]) and any("armature" in r for r in out.reasons)
    assert not out.useful


# ------------------------------------------------------------ nagranie na blizniaku
@pytest.fixture
def twin():
    from machinekind_twin.runtime import Twin
    from machinekind_twin.workspace import Workspace

    tw = Twin(Workspace(robot=BOT))
    try:
        yield tw
    finally:
        tw.close()


def short_plan():
    home = dict(SPEC.home)
    return [(0.0, home), (0.3, dict(home, j1=5.0)), (0.6, home)]


def test_record_does_not_take_back_an_arm_freed_by_home(twin):
    """Dom wcisniety tuz po "Identyfikuj", zanim watek zadania doszedl do `record`.

    Zmierzone przed poprawka: `record` bral zwolnione ramie z powrotem, `claim`
    przerywal rampe do domu, a dojazd jechal jako "identyfikacja". Flage przerwania
    zadania czyscil jeszcze `Job.start` - wiec bez niej tez nie wolno ruszyc.
    """
    with twin.lock:
        twin.scene.set_joints(dict(SPEC.home, j1=40.0))
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
