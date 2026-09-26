"""Synthetic images and placeholder calibrations cannot authorize physical motion."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.workspace import CameraRecord, require_physical_cameras


def camera(source="0"):
    return CameraRecord("scene", source, K=np.eye(3).tolist(), T_cam2base=np.eye(4).tolist(),
                        intrinsics_from="szachownica", intrinsics_info={"trusted": True},
                        calibration={"trusted": True})


def test_physical_camera_requires_both_real_source_and_trusted_calibration():
    real = camera()
    require_physical_cameras([real], [real.name])
    for bad in (camera("sim"), CameraRecord("scene"), camera()):
        if bad.source == "0" and bad.K is not None:
            bad.intrinsics_from = "symulacja"
        with pytest.raises(RuntimeError, match="physical"):
            require_physical_cameras([bad], [bad.name])
    with pytest.raises(RuntimeError, match="physical"):
        require_physical_cameras([real], [])


def test_physical_policy_provider_rechecks_sources_before_using_cached_detection():
    pytest.importorskip("viser")
    from lerobot_mp.twin.ui.app import TwinApp

    app = TwinApp.__new__(TwinApp)
    app.twin = SimpleNamespace(status=SimpleNamespace(simulated=False))
    app.ws = SimpleNamespace(cameras=[camera("sim")])
    app._mapper_now = lambda: SimpleNamespace(cameras={"scene": None})
    with pytest.raises(RuntimeError, match="physical"):
        app._vision_cube()


def test_tag_detector_does_not_use_untrusted_or_simulated_cameras_for_physical_motion(monkeypatch):
    from lerobot_mp.twin import tagreach

    calls = []
    monkeypatch.setattr(tagreach.tags, "detect", lambda img: calls.append(1) or {})
    frames = {"scene": np.zeros((10, 10, 3), dtype=np.uint8)}
    untrusted = camera()
    untrusted.calibration = {}
    for bad in (camera("sim"), untrusted):
        assert tagreach.tag_from_frames(frames, [bad], 0, 0.05, physical=True) is None
    assert not calls
    tagreach.tag_from_frames(frames, [camera()], 0, 0.05, physical=True)
    assert calls == [1]


@pytest.mark.parametrize("simulated,source,publishes", [
    (False, "kamery", True), (False, "w dloni", True),
    (False, "ostatnie widziane", False), (True, "kamery", False)])
def test_vision_updates_physical_collision_geometry_only_from_current_estimates(simulated, source, publishes):
    pytest.importorskip("viser")
    from lerobot_mp.twin.kinematics import RobotKinematics
    from lerobot_mp.twin.robots import A1X
    from lerobot_mp.twin.ui.app import TwinApp

    app = TwinApp.__new__(TwinApp)
    sent = []
    app.twin = SimpleNamespace(status=SimpleNamespace(simulated=simulated, measured=dict(A1X.home),
                                                       command=dict(A1X.home)),
                               observe_object=lambda *args: sent.append(args))
    app.ws = SimpleNamespace(cameras=[camera()])
    app._mapper_now = lambda: SimpleNamespace(cameras={"scene": None})
    app._kin_vision = RobotKinematics(A1X)
    app._grip_closed_q = lambda: 0.0
    app.last_cube, app._cube_used_t = None, 0.0
    observed = (np.array([.35, .01, .015]), np.eye(3))
    app.cube_tracker = SimpleNamespace(source=source, update=lambda *args: observed)
    assert app._vision_cube() is observed
    assert len(sent) == int(publishes)
    if publishes:
        assert sent[0][0] == "cube"
        np.testing.assert_array_equal(sent[0][1], observed[0])


def test_adding_the_policy_cube_preserves_other_obstacles_and_uses_saved_dimensions():
    pytest.importorskip("viser")
    from lerobot_mp.twin import scene as sc
    from lerobot_mp.twin.rl.task import make_task
    from lerobot_mp.twin.robots import A1X
    from lerobot_mp.twin.runtime import Twin
    from lerobot_mp.twin.ui.app import TwinApp
    from lerobot_mp.twin.workspace import Workspace

    app = TwinApp.__new__(TwinApp)
    app.ws = Workspace(robot="a1x")
    app.twin = Twin(app.ws)
    try:
        app.twin.configure(objects=[sc.Box("obstacle", (.04, .04, .04), (.5, .4))])
        app._configure_objects(cube=True, pad=False, task=make_task("lift", A1X, cube_half=.022))
        assert {o.name for o in app.twin.scene.cfg.objects} == {"obstacle", "cube"}
        np.testing.assert_allclose(app.twin.scene.model.geom("cube").size, [.022] * 3)
    finally:
        app.twin.close()
