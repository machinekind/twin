"""The example arm (`examples/robots/arm6`): the robot `uv sync` installs and the example scene opens.

It is built from primitives, so these tests also guard the twin's handling of an arm without meshes.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco

from machinekind_twin.kinematics import RobotKinematics
from machinekind_twin.robots import get_spec, robot_folders

ARM = "arm6"
pytestmark = pytest.mark.skipif(ARM not in robot_folders(), reason="arm6 nie jest zainstalowane (uv sync)")

#: The jaws pointing straight down, closing along the base's y: the columns are the TCP axes.
JAWS_DOWN = np.column_stack([(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])


@pytest.fixture(scope="module")
def kin():
    return RobotKinematics(get_spec(ARM))


def test_at_home_the_jaws_point_down_above_the_table(kin):
    T = kin.tcp(kin.spec.home)
    approach, _ = kin.tool_axes(kin.spec.home)

    assert 0.10 < T[2, 3] < 0.20 and 0.15 < T[0, 3] < 0.25
    assert approach @ (0.0, 0.0, -1.0) > 0.99


def test_the_declared_tool_axes_match_the_jaw_geometry(kin):
    """The approach runs from the hand through the jaw tips; the jaws close along the closing axis."""
    home = kin.spec.home
    approach, closing = kin.tool_axes(home)
    tips = [kin.body(body, home)[:3, 3] for body in kin.spec.jaw_bodies]
    hand = kin.body(kin.spec.hand_body, home)[:3, 3]

    reach = (tips[0] + tips[1]) / 2 - hand
    assert approach @ (reach / np.linalg.norm(reach)) > 0.99
    across = kin.site(kin.spec.tcp_site, home)[:3, :3] @ np.asarray(kin.spec.tcp_closing)
    assert abs(closing @ across) > 0.99


@pytest.mark.parametrize("bearing", [-0.6, 0.0, 0.6])
@pytest.mark.parametrize("radius", [0.15, 0.25])
def test_ik_reaches_the_desk_with_the_jaws_down(kin, bearing, radius):
    target = np.array([radius * np.cos(bearing), radius * np.sin(bearing), 0.05])
    R = np.array([[np.cos(bearing), -np.sin(bearing), 0.0], [np.sin(bearing), np.cos(bearing), 0.0],
                  [0.0, 0.0, 1.0]]) @ JAWS_DOWN

    sol = kin.ik(target, R)

    assert sol.ok and sol.pos_err < 1e-3
    assert kin.tool_axes(sol.joints)[0] @ (0.0, 0.0, -1.0) > 0.95


def test_the_arm_holds_its_home_pose_under_physics():
    spec = get_spec(ARM)
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    target = {name: np.radians(spec.home[name]) for name in spec.joints if name != spec.gripper}
    for name, q in target.items():
        data.qpos[model.joint(name).qposadr[0]] = q
        data.ctrl[model.actuator(name).id] = q

    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    drift = max(abs(data.qpos[model.joint(name).qposadr[0]] - q) for name, q in target.items())
    assert np.degrees(drift) < 0.5


def test_the_3d_view_draws_every_geom_of_the_arm():
    from machinekind_twin.ui.bridge import SceneMirror

    model = mujoco.MjModel.from_xml_path(str(get_spec(ARM).mjcf_path))
    added: list[str] = []

    class Scene:
        def __getattr__(self, method):
            def add(name, *args, **kw):
                added.append(method)
                return SimpleNamespace(remove=lambda: None)
            return add

    SceneMirror(SimpleNamespace(scene=Scene()), model)

    assert len([m for m in added if m != "add_frame"]) == model.ngeom
