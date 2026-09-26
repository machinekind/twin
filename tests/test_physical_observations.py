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
