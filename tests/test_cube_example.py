"""The example cube detector (`examples/detectors/cube`) through the twin's perception, on the
`cube_scene` stand: its mask is the old Map tab detector's, and the pose the twin fits to its masks
lands within the old path's measured error of the physics (`test_perception.py`), plus 1 mm and 1 deg.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_perception import LEGACY_ERROR_DEG, LEGACY_ERROR_M, POSES, yaw_apart_deg

from machinekind_twin.detection import Perception
from machinekind_twin.detectors import load_detector
from machinekind_twin.perception import CubeDetector

pytestmark = pytest.mark.render

#: What the example may add to the old path's error against the physics.
EXTRA_M = 0.001
EXTRA_DEG = 1.0


def _perception(scene) -> Perception:
    """Headless: frames from the twin's cameras, arm masks as the panel computes them, no hand."""
    return Perception(scene.twin, scene.ws, occluders=lambda times: scene.occluders(), hand=lambda: None)


@pytest.mark.parametrize("x, y, yaw", POSES)
def test_the_twin_places_the_example_cube_where_the_physics_has_it(cube_scene, cube_example, x, y, yaw):
    cube_scene.place(x, y, yaw)
    pos, R = cube_scene.truth()
    perception = _perception(cube_scene)
    try:
        runner = perception.start_detector(load_detector(cube_example), {}, threaded=False)
        runner.step_once()
        objects = perception.objects()
        views = {v.label: v for v in perception.snapshot()}
    finally:
        perception.stop_all("koniec testu")

    cube = objects["cube"]
    assert cube.source == "kamery"
    assert views["cube"].n_cameras == 2
    assert views["cube"].method == "sylwetka"
    assert np.linalg.norm(np.asarray(cube.position) - pos) <= LEGACY_ERROR_M + EXTRA_M
    assert yaw_apart_deg(np.asarray(cube.rotation), R) <= LEGACY_ERROR_DEG + EXTRA_DEG


def test_the_example_mask_is_the_old_detector_s_mask_pixel_for_pixel(cube_scene, cube_example):
    from cube_detector import ColourCube, CubeSettings

    cube_scene.place(0.2, 0.05, 30.0)
    example, old = ColourCube(CubeSettings()), CubeDetector()

    for name, image in cube_scene.frames().items():
        mine = example.mask(image) > 0
        assert mine.any(), name
        assert np.array_equal(mine, old.mask(image) > 0), name


def test_one_camera_alone_still_places_the_cube(cube_scene, cube_example):
    cube_scene.place(0.22, -0.03, 15.0)
    cube_scene.ws.camera("b").enabled = False
    perception = _perception(cube_scene)
    try:
        runner = perception.start_detector(load_detector(cube_example), {}, threaded=False)
        runner.step_once()
        perception.objects()
        views = perception.snapshot()
    finally:
        perception.stop_all("koniec testu")

    assert [v.label for v in views] == ["cube"]
    assert views[0].n_cameras == 1
