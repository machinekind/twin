"""Staged home in the twin, on the test robot in simulation: stage by stage, and a stop on a collision.

`stagebot` is the test robot with `home_stages` (base, arm, gripper) and a post on its base that
stands between the side and home: with the arm raised it passes over the post, with the arm low it
runs into it. So the order of the stages decides whether a home from the side arrives. The plain
test robot keeps the supervisor's one-ramp home the rest of the tests rely on.

The twin runs on a virtual clock, stepped by the test (`twinbench.Bench`): the home thread (`dom`)
and `move` drive in the same simulated time.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import pytest
import yaml

pytest.importorskip("mujoco")

from conftest import TESTBOT  # noqa: E402
from twinbench import Bench  # noqa: E402

from lerobot_mp.twin.collision import CollisionChecker  # noqa: E402
from lerobot_mp.twin.robots import get_spec  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

ROBOT = "stagebot"
#: 10 cm from the j1 axis at -40 degrees, 20 cm tall. From the side with the arm raised (j1 -80,
#: j2 -50) one ramp of both joints, or j2 before j1, runs the arm into it; base first, arm up,
#: passes over it.
POST = '<geom name="post" type="cylinder" fromto="0.0766 -0.0643 0.002 0.0766 -0.0643 0.2" size="0.01" mass="0.05"/>'
STAGES = [["j1"], ["j2"], ["gripper"]]

#: The arm raised above the post, jaws open.
RAISED = {"j1": 0.0, "j2": -50.0, "gripper": 60.0}
#: Swung to the side with the arm raised: the post lies between here and home.
SIDE_RAISED = {"j1": -80.0, "j2": -50.0, "gripper": 60.0}
#: Swung to the side with the arm low: every way home sweeps the arm through the post.
SIDE_LOW = {"j1": -80.0, "j2": 20.0, "gripper": 60.0}


@dataclass
class MoveSpy:
    """`Twin.move` that records every target it is given, rounded to 0.1."""

    move: Callable
    targets: list[dict[str, float]] = field(default_factory=list)

    def __call__(self, joints, duration, *args, **kwargs):
        self.targets.append({name: round(value, 1) for name, value in joints.items()})
        return self.move(joints, duration, *args, **kwargs)


@pytest.fixture
def stagebot(install_dist, tmp_path):
    """A package registering a copy of the test robot's folder with the post and the stages."""
    folder = tmp_path / ROBOT
    shutil.copytree(TESTBOT / "testbot" / "robot", folder)
    data = yaml.safe_load((folder / "robot.yaml").read_text(encoding="utf-8"))
    data.update(name=ROBOT, home_stages=STAGES)
    (folder / "robot.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    mjcf = folder / data["mjcf"]
    base = '<geom name="base" type="cylinder" size="0.04 0.02" pos="0 0 0.02" mass="1"/>'
    text = mjcf.read_text(encoding="utf-8")
    assert base in text
    mjcf.write_text(text.replace(base, f"{base}\n      {POST}"), encoding="utf-8")
    install_dist("stagebot", {"lerobot_mp.robots": {ROBOT: "stagebot_dir:DIR"}},
                 {"stagebot_dir": f"DIR = {str(folder)!r}\n"})
    return ROBOT


@pytest.fixture
def bench(stagebot, tmp_path):
    bench = Bench.sim(ws=Workspace(robot=stagebot, path=tmp_path / "twin.json"))
    yield bench
    bench.twin.disconnect()
    bench.clock.close()


def drive(bench: Bench, *poses: dict[str, float]) -> None:
    """Moves through `poses` one after another, each a clean straight line, then lets go of the arm."""
    for pose in poses:
        bench.call(partial(bench.twin.move, pose, 2.0, owner="test"))

    bench.twin.release("test")


def test_one_ramp_home_from_the_side_runs_into_the_post(bench):
    """Why the stages: every joint at once from the side drives the arm through the post."""
    checker = CollisionChecker(bench.twin.scene)

    why = checker.path_clear(SIDE_RAISED, get_spec(ROBOT).home)

    assert why is not None
    assert "post" in why


def test_staged_home_drives_the_base_first_over_the_post_and_arrives(bench, monkeypatch):
    twin = bench.twin
    home = get_spec(ROBOT).home
    drive(bench, RAISED, SIDE_RAISED)
    spy = MoveSpy(twin.move)
    monkeypatch.setattr(twin, "move", spy)

    twin.home()
    arrived = bench.pump(until=lambda: twin.owner != "dom" and len(spy.targets) >= 3, max_sim_s=40.0)
    bench.pump(sim_s=0.5)

    assert arrived, spy.targets
    assert twin.home_note == ""
    first = spy.targets[0]
    assert [name for name in first if abs(first[name] - SIDE_RAISED[name]) > 0.5] == ["j1"]   # the rest holds
    assert first["j1"] == pytest.approx(home["j1"], abs=0.6)
    assert twin.status.measured["j1"] == pytest.approx(home["j1"], abs=2.0)
    assert twin.status.measured["j2"] == pytest.approx(home["j2"], abs=2.0)
    bench.assert_on_time()


def test_a_home_whose_every_stage_order_runs_into_the_post_stops_where_the_arm_stands(bench, monkeypatch):
    twin = bench.twin
    drive(bench, RAISED, SIDE_RAISED, SIDE_LOW)
    spy = MoveSpy(twin.move)
    monkeypatch.setattr(twin, "move", spy)

    twin.home()
    stopped = bench.pump(until=lambda: twin.home_note, max_sim_s=5.0)

    assert stopped
    assert "post" in twin.home_note
    assert spy.targets == []
    assert twin.status.measured["j1"] == pytest.approx(SIDE_LOW["j1"], abs=2.0)
    bench.assert_on_time()


def test_sim_starts_closed_and_set_home_here_closes_the_scene_gripper(bench):
    """"Ustaw dom = tu" closes the scene gripper at once and the supervisor starts from 0, as on an arm
    without gripper feedback: the station's home has the jaws closed."""
    twin = bench.twin
    assert twin.joints()["gripper"] == pytest.approx(0.0, abs=0.5)                # the description's home

    twin.claim("test")
    twin.set_engaged(True, owner="test")
    twin.set_target({"gripper": 60.0}, owner="test")
    bench.pump(sim_s=1.0)
    assert twin.joints()["gripper"] == pytest.approx(60.0, abs=2.0)

    arm_before = {name: value for name, value in twin.joints().items() if name != "gripper"}
    home = twin.set_home_here()
    assert home["gripper"] == 0.0
    assert twin.workspace.home["gripper"] == 0.0
    assert twin.joints()["gripper"] == pytest.approx(0.0, abs=0.5)                 # closed at once
    assert twin._supervisor.command["gripper"] == 0.0

    bench.pump(sim_s=0.5)
    assert twin.joints()["gripper"] == pytest.approx(0.0, abs=0.5)                 # the servo holds 0
    assert twin.status.command["gripper"] == pytest.approx(0.0, abs=0.5)
    for name, value in arm_before.items():
        assert twin.joints()[name] == pytest.approx(value, abs=0.5), name

    twin.set_target({"gripper": 100.0}, owner="test")
    bench.step()
    assert 0.0 < twin.status.command["gripper"] < 10.0                            # the next open ramps from 0
