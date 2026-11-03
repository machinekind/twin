"""Blocks resting on blocks, rendered on the `CubeScene` stand and placed by `Perception`.

The scene builds the stack with `scene.Box(z=...)` and lets the physics settle it. An inline
detector masks each block by its colour word's ranges (`machinekind_twin_seeded`) and sends the
masks with the block's size, as the seeded detector does. The twin must place every block, the
lower ones partly hidden by the upper ones, where the physics has it, and never a block that is
not there.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from conftest import BOT, CubeScene
from machinekind_twin_api.detector import Detection
from machinekind_twin_seeded.grow import colour_mask
from machinekind_twin_seeded.settings import COLOURS, SHAPES, parse_hsv
from test_detection import plugin
from test_perception import turn_apart_deg, yaw_deg

from machinekind_twin import scene as sc
from machinekind_twin.detection import Perception

pytestmark = pytest.mark.render

#: Paints the colour words' ranges take, RGBA.
PAINT = {"red": (0.85, 0.25, 0.2, 1.0), "light blue": (0.45, 0.72, 0.92, 1.0),
         "yellow": (0.92, 0.78, 0.2, 1.0), "green": (0.25, 0.65, 0.3, 1.0)}
#: How far a placed block may lie from the physics [m] and turn from it [deg].
ERROR_M = 0.003
ERROR_DEG = 5.0
#: How far the settled physics may move a block from where the scene built it [m].
SETTLE_M = 0.0005


def size(label: str) -> tuple[float, float, float]:
    return SHAPES[label.split()[-1]]


#: Stacks, bottom first: (label, x, y, yaw [deg]); each rests on the one before.
STACKS = {
    "cube on cube": [("red cube", 0.22, 0.02, 20.0), ("light blue cube", 0.22, 0.02, 35.0)],
    "cube on cuboid": [("green cuboid", 0.24, -0.03, -30.0), ("yellow cube", 0.243, -0.028, 10.0)],
    "cuboid on cube": [("red cube", 0.20, 0.05, 0.0), ("light blue cuboid", 0.20, 0.05, 60.0)],
    "three high": [("red cube", 0.21, 0.0, 15.0), ("green cuboid", 0.212, 0.002, 40.0),
                   ("yellow cube", 0.214, 0.0, -10.0)],
}


class BlocksScene(CubeScene):
    """The `CubeScene` stand with these boxes instead of the cube."""

    def __init__(self, boxes: list[sc.Box]):
        from machinekind_twin.calib.simulate import look_at
        from machinekind_twin.clock import VirtualClock
        from machinekind_twin.runtime import Twin
        from machinekind_twin.workspace import CameraRecord, Workspace

        self.ws = Workspace(robot=BOT)
        for name, eye in self.EYES:
            T = look_at(eye, (0.2, 0.0, 0.0)).tolist()
            self.ws.add_camera(CameraRecord(name, "sim", 640, 480, K=[list(r) for r in self.K], sim_pose=T,
                                            T_cam2base=T, intrinsics_from="symulacja",
                                            calibration={"trusted": True}))
        self.cameras = tuple(name for name, _ in self.EYES)
        self.twin = Twin(self.ws, clock=VirtualClock(t0=0.0))
        self.twin.configure(objects=boxes)
        self.twin.connect("sim", threaded=False)

    def settle(self, seconds: float) -> None:
        with self.twin.lock:
            self.twin.scene.step(seconds)

    def body_pose(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """A body's centre and rotation in the base frame, from the physics."""
        from machinekind_twin.kinematics import inverse

        with self.twin.lock:
            s = self.twin.scene
            b = s.model.body(body).id
            Ti = inverse(s.T_base2world)
            return (Ti[:3, :3] @ s.data.xpos[b] + Ti[:3, 3],
                    Ti[:3, :3] @ s.data.xmat[b].reshape(3, 3))


class StackScene(BlocksScene):
    """The `CubeScene` stand with a stack of blocks instead of the cube."""

    def __init__(self, stack: list[tuple[str, float, float, float]]):
        self.labels = [label for label, *_ in stack]
        self.built: dict[str, np.ndarray] = {}
        boxes, z = [], 0.0
        for label, x, y, yaw in stack:
            half = tuple(s / 2 for s in size(label))
            colour = label.rsplit(" ", 1)[0]
            boxes.append(sc.Box(self.body(label), half, (x, y), rgba=PAINT[colour], z=z,
                                yaw=math.radians(yaw)))
            self.built[label] = np.array([x, y, z + half[2]])
            z += 2 * half[2]
        super().__init__(boxes)

    @staticmethod
    def body(label: str) -> str:
        return label.replace(" ", "_")

    def pose(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        """A block's centre and rotation in the base frame, from the physics."""
        return self.body_pose(self.body(label))


@pytest.fixture(params=list(STACKS))
def stack(request):
    pytest.importorskip("mujoco")
    scene = StackScene(STACKS[request.param])
    try:
        scene.settle(1.0)
        yield scene
    finally:
        scene.close()


class ColourBlocks:
    """Each label's colour mask with its size, per camera; `only` limits the cameras."""

    settings = None
    interval_s = 0.2
    pixel_error_px = 6.0

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.only: tuple[str, ...] | None = None

    def start(self, info) -> None:
        pass

    def stop(self, reason: str) -> None:
        pass

    def detect(self, request) -> list[Detection]:
        found = []
        for name in request.frames:
            frame = request.frames.get(name)
            if frame is None or (self.only is not None and name not in self.only):
                continue
            hsv = cv2.cvtColor(frame.image, cv2.COLOR_RGB2HSV)
            for label in self.labels:
                colour = label.rsplit(" ", 1)[0]
                mask = colour_mask(hsv, parse_hsv(f"{colour}={COLOURS[colour]}")[colour]) > 0
                v, u = np.nonzero(mask)
                if len(u) >= 20:
                    found.append(Detection(label, name, (float(u.mean()), float(v.mean())), mask=mask,
                                           size_m=size(label)))
        return found


class OnePerCamera:
    """One label for two same-coloured blocks: camera a gets one block's mask, camera b the other's,
    as when a detector follows a different block of the label in each camera."""

    settings = None
    interval_s = 0.2
    pixel_error_px = 6.0

    def __init__(self, label: str, pick: dict[str, int]):
        self.label, self.pick = label, pick

    def start(self, info) -> None:
        pass

    def stop(self, reason: str) -> None:
        pass

    def detect(self, request) -> list[Detection]:
        colour = self.label.rsplit(" ", 1)[0]
        found = []
        for name in request.frames:
            hsv = cv2.cvtColor(request.frames.get(name).image, cv2.COLOR_RGB2HSV)
            mask = colour_mask(hsv, parse_hsv(f"{colour}={COLOURS[colour]}")[colour]) > 0
            n, parts, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            blobs = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:2]
            assert len(blobs) == 2, name
            blobs.sort(key=lambda i: centres[i][0])
            one = parts == blobs[self.pick[name]]
            v, u = np.nonzero(one)
            found.append(Detection(self.label, name, (float(u.mean()), float(v.mean())), mask=one,
                                   size_m=size(self.label)))
        return found


def perceive(scene: BlocksScene, detector, calls: int = 1):
    perception = Perception(scene.twin, scene.ws, occluders=lambda times: scene.occluders(),
                            hand=lambda: None)
    try:
        runner = perception.start_detector(plugin(detector, "blocks"), {}, threaded=False)
        for _ in range(calls):
            runner.step_once()
        return perception.objects(), {v.label: v for v in perception.snapshot()}, perception, runner
    finally:
        perception.stop_all("koniec testu")


def assert_where_the_physics_has_it(scene: StackScene, objects, label: str) -> None:
    pos, R = scene.pose(label)
    got = objects[label]
    assert np.linalg.norm(np.asarray(got.position) - pos) <= ERROR_M, (label, got.position, pos)
    period = 90.0 if size(label)[0] == size(label)[1] else 180.0
    assert turn_apart_deg(yaw_deg(np.asarray(got.rotation)), yaw_deg(R), period) <= ERROR_DEG, label


def test_a_stack_settles_where_the_scene_built_it(stack: StackScene):
    for label in stack.labels:
        pos, _ = stack.pose(label)
        assert np.linalg.norm(pos - stack.built[label]) <= SETTLE_M, (label, pos, stack.built[label])


def test_every_block_of_a_stack_is_placed_where_the_physics_has_it(stack: StackScene):
    objects, views, *_ = perceive(stack, ColourBlocks(stack.labels))

    for label in stack.labels:
        assert objects[label].source == "kamery", label
        assert views[label].method == "sylwetka", label
        assert_where_the_physics_has_it(stack, objects, label)
    top = stack.labels[-1]
    assert views[top].n_cameras == 2


def test_a_stacked_block_seen_by_one_camera_keeps_its_height():
    pytest.importorskip("mujoco")
    scene = StackScene(STACKS["three high"])
    try:
        scene.settle(1.0)
        detector = ColourBlocks(scene.labels)
        perception = Perception(scene.twin, scene.ws, occluders=lambda times: scene.occluders(),
                                hand=lambda: None)
        try:
            runner = perception.start_detector(plugin(detector, "blocks"), {}, threaded=False)
            runner.step_once()
            detector.only = ("a",)
            runner.step_once()
            objects = perception.objects()
            views = {v.label: v for v in perception.snapshot()}
        finally:
            perception.stop_all("koniec testu")

        for label in scene.labels[1:]:
            assert views[label].n_cameras == 1, label
            assert_where_the_physics_has_it(scene, objects, label)
    finally:
        scene.close()


@pytest.mark.parametrize("apart", [((0.2, 0.0), (0.2, 0.1)), ((0.2, -0.05), (0.2, 0.05))])
@pytest.mark.parametrize("pick", [{"a": 0, "b": 1}, {"a": 1, "b": 0}])
def test_two_blocks_of_one_label_one_per_camera_make_no_block_in_the_air(apart, pick):
    pytest.importorskip("mujoco")
    half = tuple(s / 2 for s in size("red cube"))
    scene = BlocksScene([sc.Box(f"red_{i}", half, xy, rgba=PAINT["red"]) for i, xy in enumerate(apart)])
    try:
        scene.settle(0.5)
        objects, *_ = perceive(scene, OnePerCamera("red cube", pick))

        got = objects.get("red cube")
        if got is not None:
            real = [scene.body_pose(f"red_{i}")[0] for i in range(2)]
            assert min(np.linalg.norm(np.asarray(got.position) - p) for p in real) <= ERROR_M, got.position
    finally:
        scene.close()


@pytest.mark.parametrize("camera", ["a", "b"])
def test_a_stack_first_seen_by_one_camera_waits_for_the_second(stack: StackScene, camera: str):
    detector = ColourBlocks(stack.labels)
    detector.only = (camera,)
    perception = Perception(stack.twin, stack.ws, occluders=lambda times: stack.occluders(),
                            hand=lambda: None)
    try:
        runner = perception.start_detector(plugin(detector, "blocks"), {}, threaded=False)
        runner.step_once()
        first = perception.objects()
        detector.only = None
        runner.step_once()
        objects = perception.objects()
    finally:
        perception.stop_all("koniec testu")

    for label in stack.labels[1:]:
        assert label not in first, (label, first[label].position)
    for label in first:
        assert_where_the_physics_has_it(stack, first, label)
    for label in stack.labels:
        assert objects[label].source == "kamery", label
        assert_where_the_physics_has_it(stack, objects, label)
