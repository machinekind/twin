"""The scene's look: lights and headlight as a workspace setting.

The default is the lighting the twin always had, so no render changes. Another look, such as the
grasp student's training scene, has to compile to exactly its own values: a policy trained on
those pixels lost 12 of 50 grasps under the twin's look.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from conftest import BOT  # noqa: E402
from machinekind_twin.robots import get_spec  # noqa: E402
from machinekind_twin.scene import CameraView, Headlight, Light, Look, SceneConfig, Table, build  # noqa: E402
from machinekind_twin.workspace import Workspace  # noqa: E402

#: The training scene's look (a directional light from straight above, a spot from the side).
TRAINING_LOOK = Look(
    lights=(Light(pos=(0.0, 0.0, 1.8), dir=(0.0, 0.0, -1.0), directional=True),
            Light(pos=(1.0, -1.0, 1.3), dir=(-0.4, 0.4, -1.0), diffuse=0.5)),
    headlight=Headlight(ambient=0.3, diffuse=0.6, specular=0.0),
)

#: The lights the twin hard-coded before the look became a setting: position above the table top
#: and strength, aimed at the table top's centre, shadows from the first one only.
FORMER_LIGHTS = (((0.4, -0.6, 1.6), 0.55), ((-0.5, 0.3, 1.4), 0.35))

#: Above the base, looking straight down at the arm (OpenCV axes: z down, y along the base's -y).
TOP_DOWN = np.array([
    [1.0, 0.0, 0.0, 0.15],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.6],
    [0.0, 0.0, 0.0, 1.0],
])

SPOT = mujoco.mjtLightType.mjLIGHT_SPOT
DIRECTIONAL = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL


def grey(*values: float) -> np.ndarray:
    return np.array([[v] * 3 for v in values], np.float32)


def render_top_down(robot: str, look: Look) -> np.ndarray:
    view = CameraView.from_fov("top", 160, 120, 60.0, TOP_DOWN)

    with build(SceneConfig(get_spec(robot), cameras=[view], look=look)) as scene:
        return scene.render("top")


def write_workspace(path, look: dict) -> None:
    path.write_text(json.dumps({"robot": BOT, "look": look}), encoding="utf-8")


# ------------------------------------------------------------------ the workspace file
def test_the_look_round_trips_through_the_workspace_file(tmp_path):
    path = Workspace(look=TRAINING_LOOK).save(tmp_path / "twin.json")

    assert Workspace.load(path).look == TRAINING_LOOK


def test_a_workspace_file_without_a_look_has_the_default_one(tmp_path):
    path = tmp_path / "twin.json"
    path.write_text(json.dumps({"robot": BOT}), encoding="utf-8")

    assert Workspace.load(path).look == Look()


def test_an_unknown_key_in_a_light_is_refused(tmp_path):
    light = {"pos": [0.0, 0.0, 1.8], "dir": [0.0, 0.0, -1.0], "colour": [1.0, 1.0, 1.0]}
    write_workspace(tmp_path / "twin.json", {"lights": [light]})

    with pytest.raises(ValueError, match="Nieznane klucze.*look.lights.*colour"):
        Workspace.load(tmp_path / "twin.json")


def test_an_unknown_key_in_the_headlight_is_refused(tmp_path):
    write_workspace(tmp_path / "twin.json", {"headlight": {"ambient": 0.3, "glare": 0.1}})

    with pytest.raises(ValueError, match="Nieznane klucze.*look.headlight.*glare"):
        Workspace.load(tmp_path / "twin.json")


def test_an_unknown_key_in_the_look_is_refused(tmp_path):
    write_workspace(tmp_path / "twin.json", {"fog": 0.2})

    with pytest.raises(ValueError, match="Nieznane klucze.*look.*fog"):
        Workspace.load(tmp_path / "twin.json")


def test_a_light_without_a_direction_is_refused(tmp_path):
    write_workspace(tmp_path / "twin.json", {"lights": [{"pos": [0.0, 0.0, 1.8]}]})

    with pytest.raises(ValueError, match="Brak kluczy.*look.lights.*dir"):
        Workspace.load(tmp_path / "twin.json")


def test_the_scene_takes_the_workspaces_look():
    assert Workspace(look=TRAINING_LOOK).scene_config().look == TRAINING_LOOK


# ------------------------------------------------------------------ what the scene compiles
def test_the_default_look_compiles_to_the_lighting_the_twin_always_had(testbot):
    top = Table().height
    positions = np.array([[x, y, top + z] for (x, y, z), _ in FORMER_LIGHTS])
    directions = np.array([-np.asarray(pos) / np.linalg.norm(pos) for pos, _ in FORMER_LIGHTS])

    with build(SceneConfig(get_spec(testbot))) as scene:
        model = scene.model

    assert np.array_equal(model.light_pos, positions)
    assert np.allclose(model.light_dir, directions, rtol=0.0, atol=1e-12)
    assert np.array_equal(model.light_diffuse, grey(0.55, 0.35))
    assert list(model.light_castshadow) == [1, 0]
    assert list(model.light_type) == [SPOT, SPOT]
    assert np.array_equal(model.vis.headlight.ambient, grey(0.35)[0])
    assert np.array_equal(model.vis.headlight.diffuse, grey(0.35)[0])
    assert np.array_equal(model.vis.headlight.specular, grey(0.5)[0])


def test_a_training_look_compiles_to_exactly_its_values(testbot):
    with build(SceneConfig(get_spec(testbot), look=TRAINING_LOOK)) as scene:
        model = scene.model

    assert list(model.light_type) == [DIRECTIONAL, SPOT]
    assert np.array_equal(model.light_diffuse, grey(0.7, 0.5))
    assert list(model.light_castshadow) == [1, 1]
    assert np.array_equal(model.vis.headlight.ambient, grey(0.3)[0])
    assert np.array_equal(model.vis.headlight.diffuse, grey(0.6)[0])
    assert np.array_equal(model.vis.headlight.specular, grey(0.0)[0])


def test_a_training_look_changes_the_pixels(testbot):
    own = render_top_down(testbot, Look())
    training = render_top_down(testbot, TRAINING_LOOK)

    assert not np.array_equal(own, training)
