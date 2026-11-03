"""Objects on the table do not sink: into the table, into each other, or out of a grasp.

The scene checks run on the test robot. The physics checks run on the SO-101 from `soarm` in the twin
runtime, driven through the supervisor as the panel drives it; without soarm installed next to the
twin they skip.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco
from conftest import BOT

from machinekind_twin import scene as sc
from machinekind_twin.clock import VirtualClock
from machinekind_twin.robots import get_spec
from machinekind_twin.runtime import Twin
from machinekind_twin.workspace import Workspace

needs_so101 = pytest.mark.skipif(importlib.util.find_spec("soarm") is None,
                                 reason="the SO-101 comes from soarm, not installed next to the twin")

#: Half the edge of a 29 mm block [m].
HALF = 0.0145
#: Where the blocks lie, in the base frame [m].
XY = (0.2, 0.0)
#: The tool pointing down, the jaws closing along the base's y.
DOWN = np.column_stack([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
#: One loop tick [s], as the panel runs the twin.
DT = 0.02


def test_the_table_and_every_colliding_object_get_stiff_contacts(tmp_path):
    objects = [sc.Box("free", (HALF,) * 3, (0.2, 0.0)),
               sc.Box("fixed", (HALF,) * 3, (0.2, 0.1), free=False),
               sc.Box("marker", (HALF,) * 3, (0.2, -0.1), free=False, mocap=True)]
    cfg = sc.SceneConfig(robot=get_spec(BOT), objects=objects)
    model = sc.build(cfg).model
    for name in ("table", "free", "fixed"):
        g = model.geom(name).id
        assert model.geom_solref[g] == pytest.approx(sc.OBJECT_SOLREF)
        assert model.geom_solimp[g] == pytest.approx(sc.OBJECT_SOLIMP)
    g = model.geom("marker").id
    assert model.geom_contype[g] == model.geom_conaffinity[g] == 0
    # Training reads the exported scene: its table is as stiff.
    exported = mujoco.MjModel.from_xml_path(str(sc.export(cfg, tmp_path)))
    g = exported.geom("table").id
    assert exported.geom_solref[g] == pytest.approx(sc.OBJECT_SOLREF)
    assert exported.geom_solimp[g] == pytest.approx(sc.OBJECT_SOLIMP)


# ---------------------------------------------------------------- the SO-101 in the twin runtime

def so101(tmp_path, *names: str, sensors: bool = False) -> Twin:
    """The SO-101 connected in simulation, with blocks at `XY` stacked in the order of `names`."""
    twin = Twin(Workspace(robot="so101", path=tmp_path / "twin.json"), clock=VirtualClock(t0=0.0))
    twin.configure(objects=[sc.Box(n, (HALF,) * 3, XY) for n in names],
                   grasp_sensors=list(names) if sensors else None)
    twin.connect("sim", threaded=False)
    with twin.lock:
        m, d = twin.scene.model, twin.scene.data
        for k, name in enumerate(names):
            adr = m.jnt_qposadr[m.body_jntadr[m.body(name).id]]
            d.qpos[adr + 2] += 2 * HALF * k
        mujoco.mj_forward(m, d)
    twin.set_engaged(True)
    return twin


def run(twin: Twin, seconds: float, watch=None) -> None:
    for _ in range(round(seconds / DT)):
        twin.step(DT)
        if watch is not None:
            watch()


def tool_down(twin: Twin, z: float, gripper: float, seed: dict[str, float] | None = None) -> dict[str, float]:
    """Joints with the tool pointing down at height `z` above `XY` [m, base frame].

    Without `seed` a full IK from home; with it one IK step, which also gives a target below a
    surface the tool cannot reach (a press)."""
    kin = twin.scene.kin
    target = np.array([XY[0], XY[1], z])
    if seed is None:
        solution = kin.ik(target, DOWN, seed=dict(kin.spec.home, gripper=gripper))
        assert solution.ok
    else:
        solution = kin.ik_step(target, DOWN, seed=seed)
    return dict(solution.joints, gripper=gripper)


def straight(twin: Twin, z0: float, z1: float, gripper: float, joints: dict[str, float] | None = None,
             watch=None) -> dict[str, float]:
    """Move the tool vertically from `z0` to `z1` in 2 mm steps; returns the last target.

    Without `joints` the tool first goes to `z0` from wherever it is."""
    if joints is None:
        joints = tool_down(twin, z0, gripper)
        twin.set_target(joints)
        run(twin, 2.0, watch)
    step = 0.002 if z1 > z0 else -0.002
    for z in np.append(np.arange(z0, z1, step), z1):
        joints = tool_down(twin, float(z), gripper, joints)
        twin.set_target(joints)
        run(twin, 0.1, watch)
    return joints


def depth(twin: Twin, a: str, b: str) -> float:
    """The deepest penetration [m] between geoms `a` and `b`, 0 without contact."""
    m, d = twin.scene.model, twin.scene.data
    pair = {m.geom(a).id, m.geom(b).id}
    return max([-c.dist for c in d.contact[:d.ncon] if {c.geom1, c.geom2} == pair] + [0.0])


def jaw_depth(twin: Twin, block: str) -> float:
    """The deepest penetration [m] of the arm's geoms into `block`."""
    m, d = twin.scene.model, twin.scene.data
    g = m.geom(block).id
    arm = {i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.startswith(sc.PREFIX)}
    return max([-c.dist for c in d.contact[:d.ncon]
                if g in (c.geom1, c.geom2) and ({c.geom1, c.geom2} - {g}) <= arm] + [0.0])


def in_hand(twin: Twin, block: str) -> np.ndarray:
    """The block's centre in the hand's frame [m]."""
    m, d = twin.scene.model, twin.scene.data
    hand = m.body(sc.PREFIX + twin.spec.hand_body).id
    b = m.body(block).id
    return d.xmat[hand].reshape(3, 3).T @ (d.xpos[b] - d.xpos[hand])


def at_torque_limit(twin: Twin) -> bool:
    """The shoulder or the elbow pushes at its actuator's force limit."""
    s = twin.scene
    act = s.act_ids[1:3]
    return bool(np.isclose(np.abs(s.data.actuator_force[act]), s.model.actuator_forcerange[act, 1]).any())


@needs_so101
def test_a_block_pressed_by_the_closed_gripper_stays_on_the_table_top(tmp_path):
    twin = so101(tmp_path, "block")
    worst = {"table": 0.0, "jaws": 0.0}

    def watch():
        worst["table"] = max(worst["table"], depth(twin, "block", "table"))
        worst["jaws"] = max(worst["jaws"], jaw_depth(twin, "block"))

    try:
        straight(twin, 2 * HALF + 0.04, 2 * HALF - 0.02, gripper=0.0, watch=watch)
        run(twin, 1.0, watch)
        assert at_torque_limit(twin)
        assert worst["table"] < 0.002
        assert worst["jaws"] < 0.0015
    finally:
        twin.close()


@needs_so101
def test_a_block_resting_on_another_does_not_sink_into_it_even_when_pressed(tmp_path):
    twin = so101(tmp_path, "lower", "upper")
    try:
        run(twin, 1.0)
        assert depth(twin, "upper", "lower") < 1e-4
        assert depth(twin, "lower", "table") < 1e-4
        worst = {"blocks": 0.0, "table": 0.0}

        def watch():
            worst["blocks"] = max(worst["blocks"], depth(twin, "upper", "lower"))
            worst["table"] = max(worst["table"], depth(twin, "lower", "table"))

        straight(twin, 4 * HALF + 0.02, 4 * HALF - 0.02, gripper=0.0, watch=watch)
        run(twin, 1.0, watch)
        assert at_torque_limit(twin)
        assert worst["blocks"] < 0.002
        assert worst["table"] < 0.002
    finally:
        twin.close()


@needs_so101
def test_a_held_block_lifts_without_slipping(tmp_path):
    twin = so101(tmp_path, "block", sensors=True)
    m, d = twin.scene.model, twin.scene.data
    block, hand = m.body("block").id, m.body(sc.PREFIX + twin.spec.hand_body).id
    try:
        grip_z = HALF + 0.003
        joints = dict(straight(twin, grip_z + 0.06, grip_z, gripper=70.0), gripper=0.0)
        twin.set_target(joints)
        run(twin, 1.0)
        held = in_hand(twin, "block")
        z0 = d.xpos[[block, hand], 2].copy()
        straight(twin, grip_z, grip_z + 0.06, gripper=0.0, joints=joints)
        run(twin, 2.0)
        rise = d.xpos[[block, hand], 2] - z0
        assert rise[1] > 0.055
        assert rise[0] == pytest.approx(rise[1], abs=0.0005)
        assert np.linalg.norm(in_hand(twin, "block") - held) < 0.0025
        assert all(d.sensordata[m.sensor(f"block_jaw{k}").adr[0]] > 0 for k in range(2))
        assert jaw_depth(twin, "block") < 0.0015
    finally:
        twin.close()
