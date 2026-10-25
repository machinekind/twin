"""The scene's own account of the evaluation object: seeded placement, its true pose and jaw contacts,
and the touch and lift rules, on the test robot."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from conftest import BOT

from machinekind_twin.clock import VirtualClock
from machinekind_twin.evaluation import settle
from machinekind_twin.robots import get_spec
from machinekind_twin.runtime import Twin
from machinekind_twin.scene_truth import (
    OBJECT_BODY,
    SIM_SOURCE,
    Area,
    ObjectSpec,
    Placement,
    SceneSample,
    SimulationPoses,
    Watch,
    check_jaws,
    check_on_table,
    configure_object,
    draw_placement,
    judge,
    place_object,
    robot_contacts,
    sample,
)
from machinekind_twin.workspace import Workspace

CUBE = ObjectSpec()
RING = Area((0.15, 0.25), (-0.8, 0.8))


@pytest.fixture
def stand():
    """The test robot with the evaluation cube in its scene, connected on a virtual clock."""
    twin = Twin(Workspace(robot=BOT), clock=VirtualClock(t0=0.0))
    try:
        configure_object(twin, CUBE)
        twin.connect("sim", threaded=False)
        yield twin
    finally:
        twin.close()


def placed(x: float, y: float, yaw_deg: float = 0.0) -> Placement:
    return Placement(0, x, y, math.hypot(x, y), math.atan2(y, x), math.radians(yaw_deg))


def raise_object(twin: Twin, height_m: float) -> None:
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        address = model.jnt_qposadr[model.body_jntadr[model.body(OBJECT_BODY).id]]
        data.qpos[address + 2] += height_m
        mujoco.mj_forward(model, data)


def look(x: float = 0.1, z: float = 0.015, tilt_deg: float = 0.0, contacts: tuple[float, float] = (0.0, 0.0)
         ) -> SceneSample:
    """The cube as a sample would give it, tilted about the base frame's x axis."""
    c, s = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return SceneSample(np.array([x, 0.0, z]), rotation, contacts)


# --------------------------------------------------------------------------- placement


def test_a_draw_depends_on_its_seed_alone_in_the_order_of_soarm_lab_s_sampler():
    first = draw_placement(RING, 50003)
    np.random.seed(1)
    np.random.uniform()

    rng = np.random.default_rng(50003)
    radius, bearing, yaw = rng.uniform(0.15, 0.25), rng.uniform(-0.8, 0.8), rng.uniform(-math.pi, math.pi)

    assert draw_placement(RING, 50003) == first
    assert first == Placement(50003, radius * math.cos(bearing), radius * math.sin(bearing), radius, bearing, yaw)
    assert draw_placement(RING, 50004) != first


def test_every_draw_lies_in_the_area_and_a_range_of_one_value_gives_that_value():
    draws = [draw_placement(RING, seed) for seed in range(300)]

    assert all(0.15 <= draw.radius_m <= 0.25 for draw in draws)
    assert all(-0.8 <= draw.bearing_rad <= 0.8 for draw in draws)
    assert all(-math.pi <= draw.yaw_rad <= math.pi for draw in draws)
    assert draw_placement(Area((0.1, 0.1), (0.0, 0.0), (0.0, 0.0)), 7) == Placement(7, 0.1, 0.0, 0.1, 0.0, 0.0)


@pytest.mark.parametrize("make", [
    lambda: Area((0.2, 0.1), (0.0, 0.0)),
    lambda: Area((-0.1, 0.1), (0.0, 0.0)),
    lambda: Area((0.1, 0.2), (0.5, -0.5)),
    lambda: Area((0.1, math.nan), (0.0, 0.0)),
    lambda: Area((0.1, 0.2), (0.0, 0.0), (1.0, -1.0)),
    lambda: ObjectSpec(size_m=(0.03, 0.0, 0.03)),
    lambda: ObjectSpec(mass_kg=-0.01),
    lambda: ObjectSpec(shape="sphere"),
])
def test_an_inverted_or_negative_range_and_a_bad_object_are_refused(make):
    with pytest.raises(ValueError):
        make()


def test_an_area_off_the_table_is_refused(stand):
    desk = get_spec(BOT).desk

    with pytest.raises(ValueError, match="poza blat"):
        check_on_table(Area((0.6, 0.7), desk.cube_bearing), CUBE, stand.scene)
    check_on_table(Area.from_desk(desk), CUBE, stand.scene)


def test_the_object_rests_where_placed_and_reads_back_in_the_base_frame(stand):
    place_object(stand, CUBE, placed(0.10, 0.02, yaw_deg=30.0))
    now = sample(stand)
    clash = robot_contacts(stand)
    pose = SimulationPoses(stand, "cube").objects()["cube"]
    settled = settle(stand)

    assert now.position_m == pytest.approx([0.10, 0.02, 0.015], abs=1e-6)
    assert math.degrees(math.atan2(now.rotation[1, 0], now.rotation[0, 0])) == pytest.approx(30.0)
    assert now.contacts == (0.0, 0.0)
    assert clash == 0
    assert settled is not None
    assert settled.position_m == pytest.approx(now.position_m, abs=2e-4)
    assert settled.contacts == (0.0, 0.0)

    assert (pose.source, pose.age_s) == (SIM_SOURCE, 0.0) == ("symulacja", 0.0)
    assert pose.position == pytest.approx(now.position_m)


def test_settling_runs_the_physics_so_a_raised_object_falls_back_to_the_table(stand):
    place_object(stand, CUBE, placed(0.10, 0.0))
    raise_object(stand, 0.03)
    raised = sample(stand)
    settled = settle(stand)

    assert raised.position_m[2] == pytest.approx(0.045, abs=1e-6)
    assert settled is not None
    assert settled.position_m[2] == pytest.approx(0.015, abs=1e-3)


def test_an_object_placed_on_the_robot_base_is_a_clash(stand):
    place_object(stand, CUBE, placed(0.0, 0.0))

    assert robot_contacts(stand) > 0


def test_a_robot_without_jaw_bodies_is_refused():
    spec = get_spec(BOT)
    check_jaws(spec)

    with pytest.raises(ValueError, match="jaw_bodies"):
        check_jaws(dataclasses.replace(spec, jaw_bodies=("", "")))


# --------------------------------------------------------------------------- the rules


def verdict(rule: str, end: SceneSample, settled: SceneSample | None, *, start: SceneSample | None = None,
            clash: bool = False):
    start = start if start is not None else look()
    watch = Watch(start)
    watch.observe(end, 1.0)
    return judge(rule, start, clash, watch, end, settled)


def test_touch_needs_a_jaw_on_the_object_at_the_end_and_after_settling_without_pushing_or_tilting_it():
    touched = look(x=0.104, tilt_deg=2.0, contacts=(2.0, 0.0))
    ok = verdict("touch", touched, touched)

    assert ok.ok and ok.reason == ""
    assert ok.push_m == pytest.approx(0.004)
    assert ok.tilt_deg == pytest.approx(2.0)
    assert ok.most_contacts == (2.0, 0.0)
    assert ok.first_contact_s == 1.0

    on = (1.0, 0.0)
    assert verdict("touch", look(x=0.01, contacts=on), look(x=0.01, contacts=on), start=look(x=0.0)).reason \
        == "obiekt przesuniety o 1.0 cm"
    assert verdict("touch", look(contacts=on), look(tilt_deg=25.0, contacts=on)).reason == "obiekt przechylony o 25 st"
    assert verdict("touch", look(contacts=on), look()).reason == "brak kontaktu po 0.5 s"
    assert verdict("touch", look(), look()).reason == "brak kontaktu na koncu"
    assert verdict("touch", look(contacts=on), look(contacts=on), clash=True).reason \
        == "obiekt dotyka ramienia na starcie"
    assert verdict("touch", look(contacts=on), None).reason == "brak pomiaru po uspokojeniu"


def test_lift_needs_both_jaws_and_four_centimetres_at_the_end_and_after_settling():
    held = (3.0, 3.0)
    ok = verdict("lift", look(z=0.071, contacts=held), look(z=0.071, contacts=held))

    assert ok.ok
    assert ok.rise_end_m == ok.rise_settled_m == pytest.approx(0.056)
    assert ok.max_rise_m == pytest.approx(0.056)

    assert verdict("lift", look(z=0.054, contacts=held), look(z=0.054, contacts=held)).reason \
        == "uniesiony o 3.9 cm na koncu (potrzeba 4 cm)"
    assert verdict("lift", look(z=0.071, contacts=(3.0, 0.0)), look(z=0.071, contacts=held)).reason \
        == "nie obie szczeki na koncu"
    assert verdict("lift", look(z=0.071, contacts=held), look(z=0.071, contacts=(3.0, 0.0))).reason \
        == "nie obie szczeki po 0.5 s"
    assert verdict("lift", look(z=0.071, contacts=held), look(z=0.015, contacts=held)).reason \
        == "uniesiony o 0.0 cm po 0.5 s (potrzeba 4 cm)"
    assert not verdict("lift", look(z=0.071, contacts=held), look(z=0.071, contacts=held), clash=True).ok
