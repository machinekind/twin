"""Cameras on the arm: declared in `robot.yaml`, rendered by the scene, served by the camera hub.

The test robot carries a wrist camera on its hand (160 x 120). For a strategy (`arm_frame_t`) the
hub renders it from the scene in simulation and serves the workspace camera of the same name, the
physical one, with a real arm. The panel and calibration (`frame_t`) always see that physical one.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from machinekind_twin import cameras, runtime  # noqa: E402
from machinekind_twin.clock import VirtualClock  # noqa: E402
from machinekind_twin.robots import get_spec  # noqa: E402
from machinekind_twin.runtime import Twin  # noqa: E402
from machinekind_twin.scene import CameraView, SceneConfig, build  # noqa: E402
from machinekind_twin.strategy import StrategyRunner  # noqa: E402
from machinekind_twin.workspace import CameraRecord, Workspace  # noqa: E402

from test_strategy import Plugin, drive  # noqa: E402

WRIST = "wrist"
#: Where a calibration wave would have put the physical wrist camera: meaningless for a camera on the arm.
WAVED_POSE = [[1.0, 0.0, 0.0, 0.5], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.6], [0.0, 0.0, 0.0, 1.0]]
AHEAD = {"j1": 0.0, "j2": 20.0, "gripper": 0.0}
TURNED = {"j1": 45.0, "j2": 20.0, "gripper": 0.0}


@dataclass
class FakeStream:
    """A real camera's stream whose latest frame is always `image`, captured just now."""

    image: np.ndarray
    error: str | None = None
    is_running: bool = True

    def rgb_t(self) -> tuple[np.ndarray, float, str | None]:
        return self.image, time.monotonic(), None

    def close(self) -> None:
        pass


@pytest.fixture
def twin(testbot):
    twin = Twin(Workspace(robot=testbot), clock=VirtualClock(t0=0.0))
    try:
        yield twin
    finally:
        twin.close()


def a_real_wrist_camera(twin: Twin, monkeypatch) -> FakeStream:
    """The operator added the physical wrist camera to the workspace under the camera's name."""
    stream = FakeStream(np.full((480, 640, 3), 7, np.uint8))
    monkeypatch.setattr(cameras, "open_live", lambda record: stream)

    twin.workspace.add_camera(CameraRecord(WRIST, source="5"))
    twin.cameras.sync()
    return stream


# ------------------------------------------------------------------ the scene
def test_the_camera_on_the_arm_renders_at_its_own_size(testbot):
    with build(SceneConfig(get_spec(testbot))) as scene:
        image = scene.render(WRIST)

    assert image.shape == (120, 160, 3)


def test_its_image_follows_the_hand(testbot):
    with build(SceneConfig(get_spec(testbot))) as scene:
        scene.set_joints(AHEAD)
        ahead = scene.render(WRIST)

        scene.set_joints(TURNED)
        turned = scene.render(WRIST)

    assert not np.array_equal(ahead, turned)


def test_after_a_physics_step_it_sees_the_hand_where_the_joints_are(testbot):
    with build(SceneConfig(get_spec(testbot))) as scene:
        scene.set_joints(AHEAD)
        scene.command(TURNED)
        scene.step(0.05)
        stepped = scene.render(WRIST)

        mujoco.mj_forward(scene.model, scene.data)
        settled = scene.render(WRIST)

    assert np.array_equal(stepped, settled)


def test_a_camera_larger_than_the_default_offscreen_buffer_renders_whole(testbot):
    spec = get_spec(testbot)
    large = dataclasses.replace(spec.cameras[WRIST], width=800, height=600)
    robot = dataclasses.replace(spec, cameras={WRIST: large})

    with build(SceneConfig(robot)) as scene:
        image = scene.render(WRIST)

    assert image.shape == (600, 800, 3)


def test_a_fixed_camera_named_like_a_camera_on_the_arm_is_refused_by_the_scene(testbot):
    view = CameraView.from_fov(WRIST, 160, 120, 60.0, np.eye(4))

    with pytest.raises(ValueError, match="nazywaja sie jak kamery na ramieniu"):
        build(SceneConfig(get_spec(testbot), cameras=[view]))


# ------------------------------------------------------------------ the name rule in the workspace
def test_a_simulated_workspace_camera_may_not_take_the_name_of_a_camera_on_the_arm(testbot):
    workspace = Workspace(robot=testbot)

    with pytest.raises(ValueError, match="nazywaja sie jak kamery na ramieniu"):
        workspace.add_camera(CameraRecord(WRIST, source="sim"))


def test_a_workspace_holding_such_a_camera_builds_no_scene(testbot):
    workspace = Workspace(robot=testbot, cameras=[CameraRecord(WRIST, source="sim", enabled=False)])

    with pytest.raises(ValueError, match="nazywaja sie jak kamery na ramieniu"):
        workspace.scene_config()


def test_a_real_workspace_camera_may_carry_that_name(testbot):
    workspace = Workspace(robot=testbot)

    workspace.add_camera(CameraRecord(WRIST, source="5"))

    assert workspace.camera(WRIST).source == "5"


def test_a_real_camera_of_that_name_with_a_pose_stays_out_of_the_scene(testbot):
    workspace = Workspace(robot=testbot, cameras=[CameraRecord(WRIST, source="5", T_cam2base=WAVED_POSE)])

    cfg = workspace.scene_config()

    assert cfg.cameras == []


def test_the_scene_of_such_a_workspace_still_renders_the_camera_on_the_arm(testbot):
    workspace = Workspace(robot=testbot, cameras=[CameraRecord(WRIST, source="5", T_cam2base=WAVED_POSE)])

    with build(workspace.scene_config()) as scene:
        image = scene.render(WRIST)

    assert image.shape == (120, 160, 3)


# ------------------------------------------------------------------ the camera hub
def test_with_nothing_connected_the_scenes_arm_is_the_arm(twin):
    assert twin.arm_simulated()


def test_while_a_real_arm_connects_the_scenes_arm_is_not_the_arm(twin, monkeypatch):
    seen = []
    create_backend = runtime.create_backend

    def create_and_look(cfg, spec):
        seen.append(twin.arm_simulated())
        return create_backend(cfg, spec)

    monkeypatch.setattr(runtime, "create_backend", create_and_look)
    twin.connect("testbot", threaded=False)

    assert seen == [False]


def test_the_hub_renders_the_camera_on_the_simulated_arm_fresh(twin):
    twin.connect("sim", threaded=False)

    image, captured = twin.cameras.arm_frame_t(WRIST)

    assert image.shape == (120, 160, 3)
    assert time.monotonic() - captured < 1.0


def test_a_real_arm_without_a_workspace_camera_of_that_name_has_no_frame(twin):
    twin.connect("testbot", threaded=False)

    image, captured = twin.cameras.arm_frame_t(WRIST)

    assert image is None
    assert captured == 0.0


def test_a_real_arm_gets_the_frames_of_the_workspace_camera_of_that_name(twin, monkeypatch):
    stream = a_real_wrist_camera(twin, monkeypatch)
    twin.connect("testbot", threaded=False)

    image, _ = twin.cameras.arm_frame_t(WRIST)

    assert image is stream.image


def test_a_simulated_arm_renders_even_with_a_real_camera_of_that_name(twin, monkeypatch):
    a_real_wrist_camera(twin, monkeypatch)
    twin.connect("sim", threaded=False)

    image, _ = twin.cameras.arm_frame_t(WRIST)

    assert image.shape == (120, 160, 3)


def test_the_panel_and_calibration_see_the_physical_camera_even_on_a_simulated_arm(twin, monkeypatch):
    stream = a_real_wrist_camera(twin, monkeypatch)
    twin.connect("sim", threaded=False)

    image, _ = twin.cameras.frame_t(WRIST)

    assert image is stream.image


# ------------------------------------------------------------------ strategies
def test_a_strategy_sees_the_camera_on_the_arm_in_its_observation(twin):
    twin.connect("sim", threaded=False)
    seen = []

    def look(obs):
        seen.append(obs.frames[WRIST])
        return {}

    runner = StrategyRunner(twin, Plugin(look), cameras=(WRIST,))
    runner.start(threaded=False)
    drive(twin, runner, 0.1)

    assert seen[-1].image.shape == (120, 160, 3)
    assert 0.0 <= seen[-1].age_s < 1.0


@pytest.fixture
def panel(testbot, tmp_path, monkeypatch):
    pytest.importorskip("viser")
    from machinekind_twin.ui.app import TwinApp

    from test_twin_ui import free_port

    monkeypatch.setattr(cameras, "open_live", lambda record: FakeStream(np.zeros((480, 640, 3), np.uint8)))
    path = Workspace(robot=testbot).save(tmp_path / "twin.json")
    app = TwinApp(path, host="127.0.0.1", port=free_port())
    try:
        yield app
    finally:
        app.close()


def test_the_strategies_tab_offers_the_camera_on_the_arm(panel):
    assert WRIST in panel.strategies._cameras()


def test_the_strategies_tab_offers_it_once_beside_a_real_camera_of_that_name(panel):
    panel.ws.add_camera(CameraRecord(WRIST, source="5"))

    assert panel.strategies._cameras().count(WRIST) == 1


def test_with_a_real_arm_and_no_workspace_camera_of_that_name_the_tab_leaves_it_out(panel):
    panel.twin.connect("testbot", threaded=False)

    assert WRIST not in panel.strategies._cameras()
