"""The plugin contracts on their own: only numpy, plain dataclasses, structural protocols."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import lerobot_mp_api
import numpy as np
import pytest
from lerobot_mp_api import (
    BACKENDS,
    ROBOTS,
    STRATEGIES,
    Action,
    BackendContext,
    BackendPlugin,
    Frame,
    ObjectPose,
    Observation,
    RobotBackend,
    RobotInfo,
    RunInfo,
    Strategy,
    backend,
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
        "import lerobot_mp_api\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "print(sorted(new - set(sys.stdlib_module_names)))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "['lerobot_mp_api', 'numpy']"


def test_it_exports_every_public_name_of_its_modules():
    for module in (strategy, backend, entry_points):
        for name in module.__all__:
            assert getattr(lerobot_mp_api, name) is getattr(module, name)
            assert name in lerobot_mp_api.__all__


def test_its_version_is_the_distributions():
    pyproject = (PACKAGE / "pyproject.toml").read_text(encoding="utf-8")

    assert f'version = "{lerobot_mp_api.__version__}"' in pyproject


# --------------------------------------------------------------------------- entry points


def test_the_entry_point_groups_are_the_ones_the_twin_reads():
    assert ROBOTS == "lerobot_mp.robots"
    assert BACKENDS == "lerobot_mp.backends"
    assert STRATEGIES == "lerobot_mp.strategies"


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


@pytest.mark.parametrize("cls", [ObjectPose, Frame, RunInfo, Observation, Action, BackendContext])
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
