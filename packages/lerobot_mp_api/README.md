# lerobot-mp-api

The contracts between the digital twin (`lerobot-mp`) and the packages that plug into it: a
robot's own repository (its robot folder and hardware backend) and a control strategy (an RL
policy, a planner, a VLA model). A plugin package depends on this package, never on the twin. It
needs only numpy: no MuJoCo, viser, torch or OpenCV.

| Module | What is in it |
|---|---|
| `lerobot_mp_api.strategy` | `Observation` (what a strategy sees in one tick), `Action` (what it wants), `ActionLike`, `RunInfo`, `ObjectPose`, `Frame`, the `Strategy` protocol and its settings type `S` |
| `lerobot_mp_api.backend` | `BackendContext` (what a backend factory gets), the `BackendPlugin` protocol, the `RobotBackend` base class, `RobotInfo` |
| `lerobot_mp_api.entry_points` | `ROBOTS`, `BACKENDS`, `STRATEGIES`: the entry point groups the twin reads |

Everything is also importable from `lerobot_mp_api` itself.

## How the twin uses them

The twin finds plugins through entry points in the installed packages' metadata, so a package
registers nothing until it is installed:

```toml
[project.entry-points."lerobot_mp.robots"]       # lerobot_mp_api.ROBOTS
myarm = "myarm:ROBOT_DIR"                        # a path to the folder with robot.yaml and the model

[project.entry-points."lerobot_mp.backends"]     # lerobot_mp_api.BACKENDS
myarm = "myarm.backend:create"                   # create(ctx: BackendContext) -> the backend

[project.entry-points."lerobot_mp.strategies"]   # lerobot_mp_api.STRATEGIES
swing = "swing:Swing"                            # a class following Strategy, built as cls(settings)
```

* **Robot**: the entry point resolves to the path of a folder holding `robot.yaml` and the MJCF
  model it names; `robot.yaml`'s `name` equals the entry's name.
* **Backend**: the twin calls `create(ctx)` with a `BackendContext` (joints, limits, port and the
  config section named like the backend, raw) and talks to the result through `RobotBackend`. A
  plugin that does not subclass `RobotBackend` is wrapped: the required calls go straight to it,
  the optional ones fall back to `RobotBackend`'s defaults.
* **Strategy**: the twin builds the class's `settings` dataclass strictly from the panel's form,
  which the config section named like the strategy prefills (`str`, `int`, `float` or `bool`
  fields), creates the strategy as `cls(settings)`, or `cls()` without `settings`, and runs it:
  an `Observation` in every tick at `hz`, an `Action` (or any object with its attributes, or a
  plain mapping of joint targets) out, every target through the twin's safety supervisor.

The twin re-exports the same classes (`lerobot_mp.twin.strategy`, `lerobot_mp.robot.base`), so
`lerobot_mp.twin.strategy.Action is lerobot_mp_api.Action`.

## A strategy

```python
from dataclasses import dataclass

from lerobot_mp_api import Action, Observation, RunInfo, Strategy


@dataclass
class HoldSettings:
    joint: str


class Hold(Strategy[HoldSettings]):
    """Holds one joint where it starts."""

    hz = 20.0
    settings = HoldSettings

    def __init__(self, settings: HoldSettings):
        self.joint = settings.joint

    def start(self, run: RunInfo, obs: Observation) -> None:
        if self.joint not in run.joints:
            raise ValueError(f"{run.robot} has no joint {self.joint!r}")
        self.where = obs.joints[self.joint]

    def act(self, obs: Observation) -> Action:
        if obs.t >= 5.0:
            return Action(target={self.joint: self.where}, done=True, success=True, note="held for 5 s")
        return Action(target={self.joint: self.where})
```

`Strategy` is a protocol with defaults, meant to be subclassed: `settings` None, `start` and
`stop` doing nothing, `time_limit_s` None. `settings` is a plain class attribute, so a subclass
sets it as `Hold` does. Subclassing also lets a type checker (pyright, mypy) check `act`, `start`
and `stop` against the contract. At runtime the twin does not need the base class: it requires
`hz` and `act`, and checks `settings` when the class has one.

## A backend

```python
from collections.abc import Mapping

from lerobot_mp_api import BackendContext, RobotInfo


def create(ctx: BackendContext) -> "MemoryArm":
    return MemoryArm(ctx)


class MemoryArm:
    """A `BackendPlugin`: connect, disconnect, read_joints, send_joints, is_connected."""

    def __init__(self, ctx: BackendContext):
        self.info = RobotInfo(name=ctx.robot, description="an arm in memory", simulated=False)
        self.positions = {name: 0.0 for name in ctx.joints}
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected

    def read_joints(self) -> dict[str, float]:
        return dict(self.positions)

    def send_joints(self, targets: Mapping[str, float]) -> dict[str, float]:
        self.positions.update(targets)
        return dict(targets)
```

`BackendPlugin` lists the optional methods (`faults`, `joint_limits`, `set_joint_limits`,
`efforts`, ...); `RobotBackend` documents what each one means. Complete, tested examples live in
the twin's repository: `examples/robots/testbot` and `examples/strategies/swing`.

## Installing

The package is not on PyPI; it lives in the twin's repository, `packages/lerobot_mp_api`. The twin
installs it from its clone, in the same command:

```bash
pip install -e packages/lerobot_mp_api -e ".[twin,dev]" -c constraints.txt
```

A plugin installed into the twin's environment finds it there. A plugin's own environment
without the twin takes it from git:

```bash
pip install "lerobot-mp-api @ git+https://github.com/machinekind/digital_twin_training.git@main#subdirectory=packages/lerobot_mp_api"
```

A plugin package names it by version, `lerobot-mp-api>=0.1,<0.2`, not by that git URL: pip
replaces an installed editable copy with the one from a direct URL, and refuses to install both
in one command. A uv project can point `[tool.uv.sources]` at the git URL instead; pip ignores it.

Tests: `pytest packages/lerobot_mp_api/tests` from the twin's root, or `pytest` here.

## Licence

Apache-2.0, [LICENSE](LICENSE).
