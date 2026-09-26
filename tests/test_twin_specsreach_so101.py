"""Kierowca `SpecsReach` na SO-101 w symulacji: zwykly ruch bez falszywego uderzenia, sciana tak.

Naped SO-101 (kp 998 N m/rad, forcerange 2,94 N m) nasyca sie przy 0,17 st. uchybu - w zwyklym
ruchu sila stoi na forcerange, wiec regula wysilku 0,8 x forcerange wyzwalala przy kazdym
ruchu (tez przy `start_center`). W symulacji SO-101 zostaje sama regula opoznienia.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("websockets")

from test_twin_specsreach import FakeRelay  # noqa: E402

from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics  # noqa: E402
from lerobot_mp.twin.reach import ReachVolume  # noqa: E402
from lerobot_mp.twin.scene import Box  # noqa: E402
from lerobot_mp.twin.specsreach import SpecsReach  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

END = {"reached", "rejected", "hold", "backoff"}


@pytest.fixture
def twin():
    tw = rt.Twin(Workspace(robot="so101"))
    tw.connect("sim", threaded=True)
    yield tw
    tw.close()


@pytest.fixture
def relay():
    r = FakeRelay()
    yield r
    r.close()


def centred_driver(twin, relay) -> SpecsReach:
    """Kierowca `sim` po `start_center` (z pozy domowej nad srodek pudelka, 10 st./s)."""
    d = SpecsReach(twin, relay_url=relay.url, mode="sim")
    assert d.start_center(10.0) is None                    # z regula 0,8 x forcerange: "impact effort shoulder_lift"
    d.start()
    relay.wait_client()
    time.sleep(0.2)
    return d


def test_so101_sim_hand_and_object_moves_complete_without_a_false_trip(twin, relay):
    vol = ReachVolume.from_workspace(twin.workspace)
    kin = RobotKinematics(twin.spec)
    d = centred_driver(twin, relay)
    try:
        assert d._effort_limits(False) == {}               # w symulacji SO-101 tylko regula opoznienia
        c = vol.center
        hand = c + np.array([0.015, -0.06, -0.07])
        ts = relay.target(hand)
        st = relay.wait_state(ts, END, 20.0)
        assert st["state"] == "reached", (st, relay.states_for(ts))
        assert np.linalg.norm(kin.tcp(twin.status.measured)[:3, 3] - hand) < 0.008
        obj = c + np.array([0.027, 0.007, -0.032])
        ts = relay.target(obj, source="object", label="cup", depthMeasured=True, age=0.1)
        st = relay.wait_state(ts, END, 20.0)
        assert st["state"] == "reached", (st, relay.states_for(ts))
        assert not [s for s in relay.states if s["state"] in ("backoff", "hold")], relay.states
        assert twin.safety_state.value != "estop"
    finally:
        d.stop()


def test_so101_sim_static_wall_trips_the_lag_rule_and_backs_off(twin, relay):
    vol = ReachVolume.from_workspace(twin.workspace)
    d = centred_driver(twin, relay)
    try:
        d._tools()                                         # sprawdzacz zbudowany bez sciany
        c = vol.center
        wall = Box("wall", size=(0.08, 0.005, 0.15), xy=(float(c[0]), float(c[1]) + 0.08),
                   rgba=(0.5, 0.5, 0.5, 1.0), free=False)
        twin.configure(objects=[wall])
        time.sleep(1.0)                                    # po przebudowie sceny ramie sie uspokaja
        ts = relay.target(c + np.array([0.0, 0.12, 0.0]))
        st = relay.wait_state(ts, END, 20.0)
        assert (st["state"], st.get("reason")) == ("backoff", "impact lag shoulder_pan"), relay.states_for(ts)
        assert twin.safety_state.value != "estop"
        assert twin.owner is None
    finally:
        d.stop()
        twin.configure(objects=None)
