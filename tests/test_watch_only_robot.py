"""A robot without a tool point (a walking robot, a cart): it stands in the scene, the twin never drives it.

The test walker stands on the floor 40 cm to the driven arm's left, the table top 0.75 m above the floor.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import yaml

pytest.importorskip("mujoco")

import mujoco
from conftest import BOT, WALKER

from machinekind_twin import scene as sc
from machinekind_twin.cli import main
from machinekind_twin.robots import ROBOT_FILE, default_robot, get_spec, load_robot, robot_folders
from machinekind_twin.workspace import RobotRecord, Workspace

ON_THE_FLOOR = [0.1, 0.4, -0.75]
#: Length of the walker's thigh [m], hip to knee.
THIGH = 0.12


def station(walker: str) -> Workspace:
    ws = Workspace(robot=BOT)
    ws.other_robots.append(RobotRecord("walker", walker, xyz=list(ON_THE_FLOOR)))
    return ws


def body_pos(scene: sc.Scene, name: str) -> np.ndarray:
    return scene.data.xpos[scene.model.body(name).id].copy()


def test_a_description_without_tool_point_and_hand_is_watch_only(walker):
    assert get_spec(walker).watch_only


@pytest.mark.parametrize("field, value", [("tcp_site", "foot"), ("hand_body", "shin")])
def test_a_description_with_only_one_of_tool_point_and_hand_fails(tmp_path, field, value):
    folder = tmp_path / "walker"
    shutil.copytree(WALKER, folder)
    data = yaml.safe_load((folder / ROBOT_FILE).read_text(encoding="utf-8"))
    data[field] = value
    (folder / ROBOT_FILE).write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="`tcp_site` i `hand_body` podaj oba albo zaden"):
        load_robot(folder)


def test_the_workspace_refuses_it_as_the_driven_arm(walker):
    with pytest.raises(ValueError, match="nie ma punktu narzedzia.*`other_robots`"):
        Workspace(robot=walker).spec()


@pytest.mark.parametrize("command", [["demo", "--path", "twin.json"], ["calib-sim"]])
def test_a_command_refuses_it_as_the_robot(walker, command, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert main([*command, "--robot", walker]) == 1
    assert "nie ma punktu narzedzia" in capsys.readouterr().err
    assert not (tmp_path / "twin.json").exists()


def test_a_new_workspace_does_not_start_with_it(walker):
    assert next(iter(robot_folders())) == walker
    assert default_robot() != walker


def test_its_bodies_are_in_the_scene_under_its_prefix(walker):
    scene = sc.build(station(walker).scene_config())
    names = {scene.model.body(b).name for b in range(scene.model.nbody)}

    assert {"walker/trunk", "walker/thigh", "walker/shin"} <= names


def test_its_leg_follows_new_joints(walker):
    scene = sc.build(station(walker).scene_config())
    scene.set_passive_joints("walker", {"hip": 90.0, "knee": 0.0})
    thigh = body_pos(scene, "walker/shin") - body_pos(scene, "walker/thigh")

    assert np.allclose(thigh, [-THIGH, 0.0, 0.0], atol=1e-6)            # from straight down to straight back


def test_its_floating_root_stays_where_the_record_puts_it_under_gravity(walker):
    ws = station(walker)
    scene = sc.build(ws.scene_config())
    mount = scene.T_base2world @ ws.other_robots[0].T_base()
    trunk = mujoco.MjModel.from_xml_path(str(get_spec(walker).mjcf_path)).body("trunk").pos
    scene.step(1.0)

    assert np.allclose(body_pos(scene, "walker/trunk"), mount[:3, :3] @ trunk + mount[:3, 3])


def test_the_exported_station_holds_it(walker, tmp_path):
    folder = station(walker).export(tmp_path / "station")
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    names = {model.body(b).name for b in range(model.nbody)}

    assert "walker/shin" in names
