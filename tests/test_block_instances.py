"""Several blocks of one label, rendered on the `CubeScene` stand and followed by the seeded detector.

The blocks have the real sizes and paint. The seeded detector runs as in the panel, except that its
question goes to `Model`, which reads each block's pose from the physics and answers like the LLM
detector: a pixel and a box per block per camera, in an order the test sets. The twin must give each
block its own key, place it where the physics has it, and keep the key on the block while blocks
move and whatever order the model lists them in.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest
from machinekind_twin_api.detector import Detection
from machinekind_twin_seeded import SeededDetector, SeededSettings
from test_detection import plugin
from test_perception import turn_apart_deg, yaw_deg
from test_stacked_blocks import ERROR_DEG, ERROR_M, PAINT, SETTLE_M, BlocksScene, size

from machinekind_twin import scene as sc
from machinekind_twin.detection import DetectorRunner, Perception

pytestmark = pytest.mark.render

CUBE = tuple(s / 2 for s in size("red cube"))
CUBOID = tuple(s / 2 for s in size("red cuboid"))


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class Model:
    """The seed question's answer from the physics: every block's centre pixel and box in every
    camera, blocks in `order`."""

    def __init__(self, scene: BlocksScene, blocks: list[tuple[str, str]]) -> None:
        self.scene, self.blocks = scene, blocks
        self.order = list(range(len(blocks)))

    def start(self, info) -> None:
        pass

    def stop(self, reason: str) -> None:
        pass

    def detect(self, request) -> list[Detection]:
        out = []
        for i in self.order:
            label, body = self.blocks[i]
            pos, R = self.scene.body_pose(body)
            half = np.array(size(label)) / 2
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half @ R.T + pos
            for name in request.frames:
                cam = next(c for c in self.scene.ws.cameras if c.name == name)
                Ti = np.linalg.inv(np.asarray(cam.T_cam2base, float))
                uv, _ = cv2.projectPoints(np.vstack([pos, corners]), cv2.Rodrigues(Ti[:3, :3])[0], Ti[:3, 3],
                                          np.asarray(cam.K, float), None)
                uv = uv.reshape(-1, 2)
                box = (float(uv[1:, 0].min()), float(uv[1:, 1].min()), float(uv[1:, 0].max()), float(uv[1:, 1].max()))
                out.append(Detection(label, name, (float(uv[0, 0]), float(uv[0, 1])), box=box, confidence=0.9,
                                     pixel_error_px=50.0))
        return out


class Run:
    """The scene, the seeded detector asking `Model`, and the twin's perception, stepped by hand."""

    def __init__(self, boxes: list[sc.Box], blocks: list[tuple[str, str]]) -> None:
        self.scene = BlocksScene(boxes)
        self.blocks = blocks
        built = {body: self.scene.body_pose(body)[0] for _, body in blocks}
        self.scene.settle(0.5)
        for body, where in built.items():   # the physics leaves the blocks where they were built
            assert np.linalg.norm(self.scene.body_pose(body)[0] - where) <= SETTLE_M, body
        self.clock = Clock()
        self.detector = SeededDetector(SeededSettings(cli="codex", seed_interval_s=2.0), clock=self.clock)
        self.model = Model(self.scene, blocks)
        self.detector._llm = self.model  # type: ignore[assignment]
        #: What the seeded detector returned on the last call.
        self.found: list[Detection] = []
        detect = self.detector.detect

        def recorded(request):
            self.found = list(detect(request))
            return self.found

        self.detector.detect = recorded  # type: ignore[method-assign]
        self.perception = Perception(self.scene.twin, self.scene.ws, occluders=lambda times: self.scene.occluders(),
                                     hand=lambda: None)
        self.runner: DetectorRunner = self.perception.start_detector(plugin(self.detector, "seeded"), {},
                                                                     threaded=False)
        self.ask()

    def ask(self) -> None:
        """A call that asks the model, its answer, and the call 0.2 s later that takes it."""
        self.answer(self.question())
        self.step()

    def question(self):
        """A call that asks the model; its answer, held back as if the model were still at it."""
        self.clock.t += 2.0
        self.runner.step_once()
        deadline = time.monotonic() + 5.0
        while self.detector._busy:
            assert time.monotonic() < deadline, "the model gave no answer"
            time.sleep(0.01)
        with self.detector._lock:
            held, self.detector._answer = self.detector._answer, None
        assert held is not None
        return held

    def answer(self, held) -> None:
        """The held answer, for the next call to take."""
        with self.detector._lock:
            self.detector._answer = held

    def step(self) -> None:
        self.clock.t += 0.2
        self.runner.step_once()

    def objects(self):
        return self.perception.objects()

    def key_of(self, body: str) -> str:
        """The key of the object nearest the body; it must lie where the physics has the body."""
        label = next(label for label, b in self.blocks if b == body)
        pos, R = self.scene.body_pose(body)
        key, got = min(self.objects().items(), key=lambda kv: float(np.linalg.norm(kv[1].position - pos)))
        assert np.linalg.norm(got.position - pos) <= ERROR_M, (body, key, got.position, pos)
        period = 90.0 if size(label)[0] == size(label)[1] else 180.0
        assert turn_apart_deg(yaw_deg(np.asarray(got.rotation)), yaw_deg(R), period) <= ERROR_DEG, (body, key)
        assert got.source == "kamery", (body, key)
        return key

    def move(self, body: str, x: float, y: float, z: float | None = None) -> None:
        """The body's centre put at (x, y) and height z (default: where it is), its yaw kept."""
        import mujoco

        twin = self.scene.twin
        with twin.lock:
            s = twin.scene
            m, d = s.model, s.data
            b = m.body(body).id
            a = m.jnt_qposadr[m.body_jntadr[b]]
            T = s.T_base2world
            if z is None:
                z = float(np.linalg.inv(T)[2, :3] @ d.qpos[a:a + 3] + np.linalg.inv(T)[2, 3])
            d.qpos[a:a + 3] = T[:3, :3] @ np.array([x, y, z]) + T[:3, 3]
            d.qvel[m.jnt_dofadr[m.body_jntadr[b]]:m.jnt_dofadr[m.body_jntadr[b]] + 6] = 0.0
            mujoco.mj_forward(m, d)

    def close(self) -> None:
        self.perception.stop_all("koniec testu")
        thread = self.detector._thread
        if thread is not None:
            thread.join(5.0)
        self.scene.close()


def red(body: str, xy: tuple[float, float], yaw_deg: float = 0.0, *, z: float = 0.0,
        half: tuple[float, ...] = CUBE) -> sc.Box:
    return sc.Box(body, half, xy, rgba=PAINT["red"], z=z, yaw=math.radians(yaw_deg))


def two_cubes(a: tuple[float, float], b: tuple[float, float], yaw_deg: float = 0.0) -> Run:
    return Run([red("red_0", a, yaw_deg), red("red_1", b, yaw_deg)], [("red cube", "red_0"), ("red cube", "red_1")])


@pytest.fixture
def run_of():
    pytest.importorskip("mujoco")
    made: list[Run] = []

    def make(build) -> Run:
        made.append(build())
        return made[-1]

    yield make
    for run in made:
        run.close()


def test_two_red_cubes_apart_get_a_key_each(run_of):
    run = run_of(lambda: two_cubes((0.2, -0.04), (0.22, 0.06), 20.0))

    assert {run.key_of("red_0"), run.key_of("red_1")} == {"red cube", "red cube 2"}
    assert len(run.objects()) == 2


# Touching along y, along x, and turned 30 deg along their own side.
TOUCHING = {
    "along y": ((0.2, -0.0146), (0.2, 0.0146), 0.0),
    "along x": ((0.1854, 0.02), (0.2146, 0.02), 0.0),
    "turned": ((0.22 - 0.0126, 0.04 - 0.0073), (0.22 + 0.0126, 0.04 + 0.0073), 30.0),
}


@pytest.mark.parametrize("where", list(TOUCHING))
def test_two_red_cubes_touching_are_two_blocks_where_the_physics_has_them(run_of, where: str):
    run = run_of(lambda: two_cubes(*TOUCHING[where]))

    assert {run.key_of("red_0"), run.key_of("red_1")} == {"red cube", "red cube 2"}
    assert len(run.objects()) == 2


@pytest.mark.parametrize("yaws", [(0.0, 10.0), (10.0, 50.0)])
def test_a_red_cube_on_a_red_cube_is_placed_on_it(run_of, yaws: tuple[float, float]):
    run = run_of(lambda: Run([red("red_0", (0.2, 0.0), yaws[0]), red("red_1", (0.2, 0.0), yaws[1], z=0.029)],
                             [("red cube", "red_0"), ("red cube", "red_1")]))

    assert {run.key_of("red_0"), run.key_of("red_1")} == {"red cube", "red cube 2"}
    views = {v.label: v for v in run.perception.snapshot()}
    assert all(v.method == "sylwetka" and v.n_cameras == 2 for v in views.values())


@pytest.mark.parametrize("turn", [0.0, 90.0])
def test_a_red_cuboid_flat_on_a_red_cuboid_is_placed_on_it(run_of, turn: float):
    run = run_of(lambda: Run([red("under", (0.2, 0.0), 0.0, half=CUBOID),
                              red("over", (0.2, 0.0), turn, z=2 * CUBOID[2], half=CUBOID)],
                             [("red cuboid", "under"), ("red cuboid", "over")]))

    assert {run.key_of("under"), run.key_of("over")} == {"red cuboid", "red cuboid 2"}
    assert len(run.objects()) == 2


def test_a_red_cube_touching_a_red_cuboid_are_both_placed(run_of):
    run = run_of(lambda: Run([red("cube", (0.23, -0.0451)), red("cuboid", (0.23, -0.0149), half=CUBOID)],
                             [("red cube", "cube"), ("red cuboid", "cuboid")]))

    assert run.key_of("cube") == "red cube" and run.key_of("cuboid") == "red cuboid"


def test_the_keys_stay_on_their_blocks_when_the_model_lists_them_the_other_way(run_of):
    run = run_of(lambda: two_cubes((0.2, -0.0146), (0.2, 0.0146)))
    keys = {body: run.key_of(body) for body in ("red_0", "red_1")}

    run.model.order = [1, 0]
    run.ask()

    assert {body: run.key_of(body) for body in ("red_0", "red_1")} == keys
    assert len(run.objects()) == 2


@pytest.mark.parametrize("jump", [False, True])
def test_one_of_two_red_cubes_moved_keeps_its_key_and_the_other_is_untouched(run_of, jump: bool):
    run = run_of(lambda: two_cubes((0.2, -0.0146), (0.2, 0.0146)))
    keys = {body: run.key_of(body) for body in ("red_0", "red_1")}
    still = run.objects()[keys["red_1"]].position.copy()

    if jump:
        # Lifted away by hand: the colour track loses it, the next answer seeds it again.
        run.move("red_0", 0.26, -0.08)
        run.ask()
    else:
        # Pushed away 1 cm per call, as the arm moves it.
        for k in range(1, 7):
            run.move("red_0", 0.2 + 0.01 * k, -0.0146 - 0.005 * k)
            run.step()

    assert {body: run.key_of(body) for body in ("red_0", "red_1")} == keys
    np.testing.assert_allclose(run.objects()[keys["red_1"]].position, still, atol=0.002)
    assert len(run.objects()) == 2


@pytest.mark.parametrize("lost_first", [False, True])
def test_an_answer_about_a_block_since_moved_away_makes_no_block_where_it_was(run_of, lost_first: bool):
    run = run_of(lambda: two_cubes((0.2, -0.0146), (0.2, 0.0146)))
    keys = {body: run.key_of(body) for body in ("red_0", "red_1")}
    still = run.objects()[keys["red_1"]].position.copy()

    held = run.question()                   # asked with both cubes where they are
    run.move("red_0", 0.26, -0.08)          # lifted away meanwhile: its colour tracks lose it
    if lost_first:
        run.step()
    run.answer(held)                        # the answer still names red_0 where it was
    for _ in range(3):
        run.step()

    assert sorted(d.camera for d in run.found) == ["a", "b"]     # red_1 alone, whole
    assert run.key_of("red_1") == keys["red_1"]
    np.testing.assert_allclose(run.objects()[keys["red_1"]].position, still, atol=0.002)
    run.ask()
    assert {body: run.key_of(body) for body in ("red_0", "red_1")} == keys


@pytest.mark.parametrize("lowered", [True, False], ids=["lowered", "dropped"])
@pytest.mark.parametrize("label", ["red cube", "red cuboid"])
def test_a_block_carried_onto_another_of_its_label_keeps_its_key(run_of, label: str, lowered: bool):
    half = CUBE if label == "red cube" else CUBOID
    run = run_of(lambda: Run([red("under", (0.2, 0.0), 0.0, half=half), red("over", (0.2, 0.08), 0.0, half=half)],
                             [(label, "under"), (label, "over")]))
    keys = {body: run.key_of(body) for body in ("under", "over")}

    # Lifted 6 cm and carried 1 cm per call; then lowered 1 cm per call, or dropped from there,
    # which loses its colour tracks until the model's next answer.
    h = 2 * half[2]
    path = [(0.2, 0.08 - 0.01 * k, 0.06) for k in range(9)]
    if lowered:
        path += [(0.2, 0.0, z) for z in np.arange(0.05, 1.5 * h + 0.005, -0.01)]
    path += [(0.2, 0.0, 1.5 * h + 0.003), (0.2, 0.0, 1.5 * h)]
    for x, y, z in path:
        run.move("over", x, y, z)
        run.step()
    for _ in range(3):
        run.step()

    if lowered:
        assert {body: run.key_of(body) for body in ("under", "over")} == keys
    run.ask()
    assert {body: run.key_of(body) for body in ("under", "over")} == keys
    assert len(run.objects()) == 2
