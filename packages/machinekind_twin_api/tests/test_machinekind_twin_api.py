"""The plugin contracts on their own: only numpy, plain dataclasses, structural protocols."""

from __future__ import annotations

import copy
import dataclasses
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import machinekind_twin_api
import numpy as np
import pytest
from machinekind_twin_api import (
    BACKENDS,
    DETECTORS,
    ROBOTS,
    STRATEGIES,
    Action,
    BackendContext,
    BackendPlugin,
    Detection,
    Detector,
    DetectorInfo,
    DetectRequest,
    Frame,
    ObjectPose,
    Observation,
    RobotBackend,
    RobotInfo,
    RunInfo,
    Strategy,
    backend,
    detector,
    entry_points,
    strategy,
)

PACKAGE = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- the package


def test_it_imports_nothing_but_numpy_and_the_standard_library():
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(PACKAGE)!r})\n"
        "before = set(sys.modules)\n"
        "import machinekind_twin_api\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "print(sorted(new - set(sys.stdlib_module_names)))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "['machinekind_twin_api', 'numpy']"


def test_it_exports_every_public_name_of_its_modules():
    for module in (strategy, backend, detector, entry_points):
        for name in module.__all__:
            assert getattr(machinekind_twin_api, name) is getattr(module, name)
            assert name in machinekind_twin_api.__all__


def test_its_version_is_the_distributions():
    pyproject = (PACKAGE / "pyproject.toml").read_text(encoding="utf-8")

    assert f'version = "{machinekind_twin_api.__version__}"' in pyproject


# --------------------------------------------------------------------------- entry points


def test_the_entry_point_groups_are_the_ones_the_twin_reads():
    assert ROBOTS == "machinekind_twin.robots"
    assert BACKENDS == "machinekind_twin.backends"
    assert STRATEGIES == "machinekind_twin.strategies"
    assert DETECTORS == "machinekind_twin.detectors"


# --------------------------------------------------------------------------- strategy


def test_the_observation_takes_only_what_the_twin_measures():
    obs = Observation(t=0.5, step=10, joints={"j1": 1.0}, command={"j1": 2.0})

    assert obs.goal is None
    assert obs.objects == {}
    assert obs.frames == {}


def test_the_observation_carries_objects_and_frames():
    cube = ObjectPose(position=np.zeros(3), rotation=np.eye(3), source="symulacja")
    frame = Frame(image=np.zeros((4, 6, 3), np.uint8), age_s=0.02)

    obs = Observation(t=0.0, step=0, joints={}, command={}, goal=np.ones(3),
                      objects={"cube": cube}, frames={"front": frame})

    assert obs.objects["cube"].age_s == 0.0
    assert obs.frames["front"].image.shape == (4, 6, 3)


def test_an_action_without_a_target_keeps_driving_to_the_previous_one():
    action = Action()

    assert action.target == {}
    assert not action.done
    assert action.success is None


def test_run_info_holds_the_robot_once():
    info = RunInfo(robot="testbot", joints=("j1", "gripper"), gripper="gripper",
                   limits={"j1": (-90.0, 90.0)}, max_vel={"j1": 90.0}, folder=Path("."), simulated=True)

    assert info.joints == ("j1", "gripper")


@pytest.mark.parametrize("cls", [ObjectPose, Frame, RunInfo, Observation, Action, BackendContext,
                                 Detection, DetectRequest, DetectorInfo])
def test_the_twins_data_is_frozen(cls):
    assert cls.__dataclass_params__.frozen


@dataclass
class SwingSettings:
    joint: str
    amplitude: float = 10.0


@dataclass
class Move:
    """A strategy's own answer object: `ActionLike`."""

    target: dict[str, float] = field(default_factory=dict)


class Swing:
    """A strategy without the base class: the twin runs it on its attributes alone."""

    hz = 20.0
    settings = SwingSettings

    def __init__(self, settings: SwingSettings):
        self.config = settings

    def act(self, obs: Observation) -> Move:
        return Move(target={self.config.joint: self.config.amplitude})


class Hold(Strategy[SwingSettings]):
    """A strategy as a package writes it: the base class gives the defaults."""

    hz = 20.0
    settings = SwingSettings

    def __init__(self, settings: SwingSettings):
        self.config = settings

    def act(self, obs: Observation) -> Action:
        return Action(target={self.config.joint: obs.joints[self.config.joint]})


class Nothing(Strategy[None]):
    """A strategy with no settings."""

    hz = 10.0

    def act(self, obs: Observation) -> Mapping[str, float]:
        return {}


def test_a_subclass_gets_the_defaults_of_the_protocol():
    hold = Hold(SwingSettings(joint="j1"))
    obs = Observation(t=0.0, step=0, joints={"j1": 3.0}, command={})
    run = RunInfo(robot="r", joints=("j1",), gripper=None, limits={"j1": (-90.0, 90.0)}, max_vel={"j1": 90.0},
                  folder=Path("."), simulated=True)

    assert Hold.settings is SwingSettings
    assert Nothing.settings is None
    assert hold.time_limit_s is None
    assert hold.start(run, obs) is None
    assert hold.act(obs).target == {"j1": 3.0}
    assert hold.stop("done") is None
    assert Nothing().act(obs) == {}


def test_a_strategy_follows_the_protocol_by_its_attributes():
    swing = Swing(SwingSettings(joint="j1"))
    answer = swing.act(Observation(t=0.0, step=0, joints={"j1": 0.0}, command={}))

    assert swing.hz > 0
    assert swing.settings is SwingSettings
    assert answer.target == {"j1": 10.0}
    assert Strategy[SwingSettings] is not None


def test_the_strategy_protocol_is_not_checked_with_isinstance():
    """Not runtime-checkable: a strategy needs only `hz` and `act`, which the twin checks by attribute."""
    with pytest.raises(TypeError):
        isinstance(Swing(SwingSettings(joint="j1")), Strategy)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- detector


class Seen(Detector[None]):
    """A detector as a package writes it: the base class gives the defaults."""

    def detect(self, request: DetectRequest) -> Sequence[Detection]:
        return [Detection(label="cube", camera=name, pixel=(1.0, 2.0)) for name in request.frames]


@dataclass
class ColourSettings:
    colour: str = "red"


class Colour:
    """A detector without the base class: the twin loads it by its attributes alone."""

    settings = ColourSettings
    interval_s = 0.2
    pixel_error_px = 2.0

    def __init__(self, settings: ColourSettings):
        self.config = settings

    def start(self, info: DetectorInfo) -> None:
        self.cameras = info.cameras

    def stop(self, reason: str) -> None:
        self.reason = reason

    def detect(self, request: DetectRequest) -> list[Detection]:
        return [Detection(label=self.config.colour, camera=name, pixel=(0.0, 0.0)) for name in request.frames]


def frame() -> Frame:
    return Frame(image=np.zeros((4, 6, 3), np.uint8), age_s=0.0)


def test_a_detector_subclass_gets_the_defaults():
    seen = Seen()
    info = DetectorInfo(cameras=("front",), robot="testbot", simulated=True)

    assert Seen.settings is None
    assert seen.interval_s == 1.0
    assert seen.pixel_error_px == 3.0
    assert seen.start(info) is None
    assert [d.camera for d in seen.detect(DetectRequest(frames={"front": frame()}, step=0))] == ["front"]
    assert seen.stop("done") is None


def test_a_class_without_the_detector_base_class_works_by_its_attributes():
    """The checks the twin's loader makes: `detect` callable, `settings` a dataclass or None."""
    colour = Colour(ColourSettings(colour="blue"))
    colour.start(DetectorInfo(cameras=("front", "side"), robot="testbot", simulated=False))
    found = colour.detect(DetectRequest(frames={"front": frame(), "side": frame()}, step=3))
    colour.stop("done")

    assert callable(getattr(Colour, "detect", None))
    assert Colour.settings is None or dataclasses.is_dataclass(Colour.settings)
    assert colour.interval_s > 0 and colour.pixel_error_px > 0
    assert [(d.label, d.camera) for d in found] == [("blue", "front"), ("blue", "side")]
    assert colour.cameras == ("front", "side") and colour.reason == "done"


def test_the_detector_protocol_is_not_checked_with_isinstance():
    """Not runtime-checkable: the twin checks a detector by its attributes."""
    with pytest.raises(TypeError):
        isinstance(Colour(ColourSettings()), Detector)  # type: ignore[arg-type]


def test_a_detection_has_only_pixels_by_default():
    d = Detection(label="cube", camera="front", pixel=(320.0, 240.0))

    assert d.box is None
    assert d.mask is None
    assert d.height_m is None
    assert d.size_m is None
    assert d.pixel_error_px is None
    assert d.confidence == 1.0


def test_a_detection_with_a_mask_compares_by_identity():
    d = Detection(label="cube", camera="front", pixel=(1.0, 1.0), mask=np.ones((4, 6), bool))

    assert d == d  # noqa: PLR0124 - the point: equal to itself even with an array inside
    assert d != copy.copy(d)
    assert hash(d) == hash(d)


# --------------------------------------------------------------------------- backend


def context(**settings) -> BackendContext:
    return BackendContext(robot="testbot", joints=("j1", "gripper"), gripper="gripper",
                          limits={"j1": (-90.0, 90.0), "gripper": (0.0, 100.0)},
                          max_vel={"j1": 90.0, "gripper": 300.0}, port=None, settings=settings)


class MemoryArm:
    """A backend plugin: no base class, the five required members."""

    def __init__(self, ctx: BackendContext):
        self.positions = {name: 0.0 for name in ctx.joints}
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def read_joints(self) -> dict[str, float]:
        return dict(self.positions)

    def send_joints(self, targets: Mapping[str, float]) -> dict[str, float]:
        self.positions.update(targets)
        return dict(targets)

    @property
    def is_connected(self) -> bool:
        return self._connected


def test_the_backend_context_is_plain_data():
    ctx = context(start={"j1": 10.0})

    assert ctx.settings == {"start": {"j1": 10.0}}
    assert BackendContext(**{f.name: getattr(ctx, f.name) for f in dataclasses.fields(ctx)}) == ctx


def test_a_plugin_with_the_required_members_is_a_backend_plugin():
    assert isinstance(MemoryArm(context()), BackendPlugin)


def test_an_object_missing_a_required_member_is_not():
    class NoSend:
        def connect(self) -> None: ...
        def disconnect(self) -> None: ...
        def read_joints(self) -> dict[str, float]: ...
        is_connected = True

    assert not isinstance(NoSend(), BackendPlugin)


class Subclassed(RobotBackend):
    """A backend that subclasses the ABC instead: the optional methods have their defaults."""

    info = RobotInfo(name="testbot", description="ramie w pamieci", simulated=False)

    def __init__(self):
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_joints(self) -> dict[str, float]:
        return {}

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        return dict(targets)

    @property
    def is_connected(self) -> bool:
        return self.connected


def test_a_robot_backend_subclass_gets_the_optional_defaults():
    with Subclassed() as arm:
        assert arm.is_connected
        assert not arm.link_silent
        assert arm.faults() == []
        assert arm.joint_limits() == {}
        assert arm.gripper_range_q() is None
        assert arm.efforts() is None and arm.velocities() is None

    assert not arm.is_connected
    assert isinstance(arm, BackendPlugin)


def test_the_robot_backend_abc_needs_the_required_methods():
    with pytest.raises(TypeError):
        RobotBackend()  # type: ignore[abstract]
