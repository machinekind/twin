"""The box fit on a 50x25x25 mm block on the `CubeScene` stand, against the physics.

The block lies on a long side, red like the cube, and the example detector's colour mask marks it.
The fit gets its three sides, so it places the block with its yaw: `rotation[:, 0]` along the 50 mm
side, folded to a half turn. Through `Perception` an inline detector sends the masks with the
block's `size_m`, as a detector that knows the block does.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import CubeScene
from cube_detector import ColourCube, CubeSettings
from machinekind_twin_api.detector import Detection
from test_detection import plugin
from test_perception import LEGACY_ERROR_DEG, LEGACY_ERROR_M, POSES, turn_apart_deg, yaw_deg

from machinekind_twin.detection import Perception
from machinekind_twin.perception import CubeDetector, TableMapper

pytestmark = pytest.mark.render

#: Half the block's sides [m]: 50 mm along its x, 25 mm across and 25 mm high.
BLOCK_HALF = (0.025, 0.0125, 0.0125)
#: What the block may add to the cube's measured error against the physics. Measured at `POSES`
#: and four more poses (yaw up to 160 deg): at most 1.26 mm and 1.0 deg, IoU 0.94-0.96.
EXTRA_M = 0.001
EXTRA_DEG = 1.0


@pytest.fixture
def block_scene():
    """The `CubeScene` stand with the block instead of the cube (render tests: MuJoCo and OpenGL)."""
    pytest.importorskip("mujoco")
    scene = CubeScene(half=BLOCK_HALF)
    try:
        yield scene
    finally:
        scene.close()


def _masks(scene: CubeScene) -> dict[str, np.ndarray]:
    colour = ColourCube(CubeSettings())
    return {name: colour.mask(image) > 0 for name, image in scene.frames().items()}


class ColourBlock:
    """The example's colour mask sent with the block's size, as a detector that knows the block does."""

    settings = None
    interval_s = 0.5
    pixel_error_px = 4.0

    def __init__(self):
        self._colour = ColourCube(CubeSettings())

    def start(self, info) -> None:
        pass

    def stop(self, reason: str) -> None:
        pass

    def detect(self, request) -> list[Detection]:
        found = []
        for name in request.frames:
            frame = request.frames.get(name)
            if frame is None:
                continue
            mask = self._colour.mask(frame.image) > 0
            v, u = np.nonzero(mask)
            if len(u) >= 20:
                found.append(Detection("block", name, (float(u.mean()), float(v.mean())), mask=mask,
                                       size_m=tuple(2 * h for h in BLOCK_HALF)))
        return found


@pytest.mark.parametrize("x, y, yaw", POSES)
def test_the_twin_fits_a_cuboid_with_its_yaw_where_the_physics_has_it(block_scene, x, y, yaw):
    block_scene.place(x, y, yaw)
    pos, R = block_scene.truth()

    got = CubeDetector(half_size=block_scene.half).fit_masks(
        _masks(block_scene), TableMapper.from_workspace(block_scene.ws), block_scene.occluders())

    assert got is not None and got.n_cameras == 2
    assert np.linalg.norm(got.pos - pos) <= LEGACY_ERROR_M + EXTRA_M
    assert turn_apart_deg(yaw_deg(got.rot), yaw_deg(R), 180.0) <= LEGACY_ERROR_DEG + EXTRA_DEG


def test_the_twin_places_a_cuboid_through_perception_with_method_sylwetka(block_scene):
    block_scene.place(0.22, 0.03, 35.0)
    _, R = block_scene.truth()
    perception = Perception(block_scene.twin, block_scene.ws, occluders=lambda times: block_scene.occluders(),
                            hand=lambda: None)
    try:
        runner = perception.start_detector(plugin(ColourBlock(), "block"), {}, threaded=False)
        runner.step_once()
        objects = perception.objects()
        views = {v.label: v for v in perception.snapshot()}
    finally:
        perception.stop_all("koniec testu")

    block = objects["block"]
    assert block.source == "kamery"
    assert views["block"].method == "sylwetka" and views["block"].n_cameras == 2
    along = np.asarray(block.rotation)[:, 0]
    assert turn_apart_deg(np.degrees(np.arctan2(along[1], along[0])), yaw_deg(R), 180.0) <= 5.0
