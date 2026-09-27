"""`SpecsReach` on the SO-101 in simulation: normal moves don't trip the impact guard, a wall does.

The SO-101 actuator (kp 998 N m/rad, forcerange 2.94 N m) saturates at 0.17 deg of error, so during any
move the force sits at forcerange and an effort rule of 0.8 x forcerange tripped on every move (including
`start_center`). In sim the SO-101 therefore uses only the lag rule.

The twin is stepped on a `VirtualClock` shared with the driver (`twinbench.Bench`).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("websockets")

from twinbench import END, Bench, FakeRelay  # noqa: E402

from lerobot_mp.twin.kinematics import RobotKinematics  # noqa: E402
from lerobot_mp.twin.reach import ReachVolume  # noqa: E402
from lerobot_mp.twin.scene import Box  # noqa: E402
from lerobot_mp.twin.specsreach import SpecsReach  # noqa: E402


@pytest.fixture
def relay():
    r = FakeRelay()
    yield r
    r.close()


@pytest.fixture
def bench(relay):
    b = Bench.sim("so101", relay)
    yield b
    try:
        b.close()
    finally:
        b.twin.close()
        b.clock.close()


def centred_driver(bench: Bench) -> SpecsReach:
    """`sim` driver after `start_center` (from home to above the volume centre at 10 deg/s)."""
    d = bench.attach(SpecsReach(bench.twin, relay_url=bench.relay.url, mode="sim"))
    # The effort rule made this "impact effort shoulder_lift".
    assert bench.call(lambda: d.start_center(10.0)) is None
    return bench.start_driver(d)


def test_so101_sim_hand_and_object_moves_complete_without_a_false_trip(bench):
    twin = bench.twin
    vol = ReachVolume.from_workspace(twin.workspace)
    kin = RobotKinematics(twin.spec)
    d = centred_driver(bench)
    assert d._effort_limits(False) == {}
    c = vol.center
    hand = c + np.array([0.015, -0.06, -0.07])
    ts = bench.target(hand)
    st = bench.wait_state(ts, END, max_sim_s=20.0)
    assert st["state"] == "reached", (st, bench.states_for(ts))
    assert np.linalg.norm(kin.tcp(twin.status.measured)[:3, 3] - hand) < 0.008
    obj = c + np.array([0.027, 0.007, -0.032])
    ts = bench.target(obj, source="object", label="cup", depthMeasured=True, age=0.1)
    st = bench.wait_state(ts, END, max_sim_s=20.0)
    assert st["state"] == "reached", (st, bench.states_for(ts))
    states = bench._sent("targetState")
    assert not [s for s in states if s["state"] in ("backoff", "hold")], states
    assert twin.safety_state.value != "estop"


def test_so101_sim_static_wall_trips_the_lag_rule_and_backs_off(bench):
    twin = bench.twin
    vol = ReachVolume.from_workspace(twin.workspace)
    d = centred_driver(bench)
    try:
        d._tools()                                         # build the collision checker before the wall exists
        c = vol.center
        wall = Box("wall", size=(0.08, 0.005, 0.15), xy=(float(c[0]), float(c[1]) + 0.08),
                   rgba=(0.5, 0.5, 0.5, 1.0), free=False)
        twin.configure(objects=[wall])
        bench.pump(sim_s=1.0)                              # let the arm settle after the scene rebuild
        ts = bench.target(c + np.array([0.0, 0.12, 0.0]))
        st = bench.wait_state(ts, END, max_sim_s=20.0)
        assert (st["state"], st.get("reason")) == ("backoff", "impact lag shoulder_pan"), bench.states_for(ts)
        assert twin.safety_state.value != "estop"
        assert twin.owner is None
    finally:
        twin.configure(objects=None)
