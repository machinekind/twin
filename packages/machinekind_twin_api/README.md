# machinekind-twin-api

The contracts between the digital twin (`machinekind-twin`) and the packages that plug into it: a
robot's own repository (its robot folder and hardware backend), a control strategy (an RL
policy, a planner, a VLA model) and an object detector (a colour threshold, a neural network, a
vision-language model). A plugin package depends on this package, never on the twin. It needs
only numpy: no MuJoCo, viser, torch or OpenCV.

| Module | What is in it |
|---|---|
| `machinekind_twin_api.strategy` | `Observation` (what a strategy sees in one tick), `Action` (what it wants), `ActionLike`, `RunInfo`, `ObjectPose`, `Frame`, the `Strategy` protocol and its settings type `S` |
| `machinekind_twin_api.backend` | `BackendContext` (what a backend factory gets), the `BackendPlugin` protocol, the `RobotBackend` base class, `RobotInfo` |
| `machinekind_twin_api.detector` | `DetectRequest` (one call's frames), `Detection` (one object seen in one frame, in pixels), `DetectorInfo`, the `Detector` protocol |
| `machinekind_twin_api.entry_points` | `ROBOTS`, `BACKENDS`, `STRATEGIES`, `DETECTORS`: the entry point groups the twin reads |

Everything is also importable from `machinekind_twin_api` itself.

## How the twin uses them

The twin finds plugins through entry points in the installed packages' metadata, so a package
registers nothing until it is installed:

```toml
[project.entry-points."machinekind_twin.robots"]       # machinekind_twin_api.ROBOTS
myarm = "myarm:ROBOT_DIR"                        # a path to the folder with robot.yaml and the model

[project.entry-points."machinekind_twin.backends"]     # machinekind_twin_api.BACKENDS
myarm = "myarm.backend:create"                   # create(ctx: BackendContext) -> the backend

[project.entry-points."machinekind_twin.strategies"]   # machinekind_twin_api.STRATEGIES
swing = "swing:Swing"                            # a class following Strategy, built as cls(settings)

[project.entry-points."machinekind_twin.detectors"]    # machinekind_twin_api.DETECTORS
cube = "cube_detector:ColourCube"                # a class following Detector, built as cls(settings)
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
* **Detector**: the twin builds its settings like a strategy's, from the config section named
  like the detector. It calls `detect` on a dedicated thread with a `DetectRequest`, which holds
  the frames by camera name. The detector returns `Detection`s, each giving what it sees in one
  image and where, in pixels. The twin places and tracks the objects, and strategies read them as
  `Observation.objects[label]`.

String settings can declare dropdown choices without importing the twin:

```python
from dataclasses import dataclass, field

@dataclass
class Settings:
    provider: str = field(default="local", metadata={"choices": ("local", "remote")})
    model: str = field(default="small", metadata={
        "choices_from": "provider",
        "choices": {"local": ("small", "large"), "remote": ("fast", "accurate")},
    })
```

`choices_from` names an earlier dropdown field, and its mapping covers every choice of that
field. Changing it refreshes the dependent dropdown and selects its first option if the old
value no longer applies. Choices guide the panel; the plugin still validates settings. Explicit
config values remain available in the dropdown for their configured provider. An empty string
appears as `(domyslne)` and is passed back as `""`; required dropdowns start with `(wybierz)`.

The twin re-exports the same classes (`machinekind_twin.strategy`, `machinekind_twin.backends.base`,
`machinekind_twin.detection`), so `machinekind_twin.strategy.Action is machinekind_twin_api.Action`.

## A strategy

```python
from dataclasses import dataclass

from machinekind_twin_api import Action, Observation, RunInfo, Strategy


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

from machinekind_twin_api import BackendContext, RobotInfo


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
the twin's repository: `examples/robots/arm6`, `examples/robots/testbot`, `examples/strategies/swing`
and `examples/detectors/cube`.

## A detector

```python
from dataclasses import dataclass

import numpy as np

from machinekind_twin_api import DetectRequest, Detection, Detector


@dataclass
class RedSettings:
    #: How high above the table the red thing's centre is [m].
    height_m: float = 0.0


class Red(Detector[RedSettings]):
    """Finds the red pixels in each frame and reports their centre as one object, "red"."""

    settings = RedSettings
    interval_s = 0.5

    def __init__(self, settings: RedSettings):
        self.height_m = settings.height_m

    def detect(self, request: DetectRequest) -> list[Detection]:
        found: list[Detection] = []
        for name in request.frames:
            frame = request.frames.get(name)
            if frame is None:
                continue
            rgb = frame.image.astype(np.int16)
            red = (rgb[..., 0] > 150) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80)
            if red.sum() < 20:
                continue
            v, u = np.nonzero(red)
            found.append(Detection("red", name, (float(u.mean()), float(v.mean())),
                                   height_m=self.height_m))
        return found
```

A detector reports what it sees in the image and where. Each detection has a label, the camera
and a pixel. When it can, the detector adds a box, a silhouette `mask`, the object's `height_m`
above the table or its `size_m`. The detector needs no intrinsics, camera poses or clock. The twin
turns each pixel into a ray through the calibrated camera and places the object in one of three
ways:

* It fits a cube of `size_m` to the masks from every camera.
* It takes the point nearest to the rays of two or more cameras.
* It takes the point where one ray meets the plane at the object's height.

The twin then tracks each label and publishes it with `age_s`, the seconds since the frames were
captured.

A detection's own `age_s` is the age in seconds of the frames it was made from, relative to this
call's frames. The default 0 means this call's frames. A detector that repeats an earlier answer
sets it to the time between that answer's frame and this call's frame from the same camera. The twin
adds it to the published `age_s`. The twin also counts it as latency, so it holds the pose until the
next answer.

`request.frames` lists the cameras offered in this call. A frame is fetched only when read.
`frames.get(name)` and `name in frames` tell whether a camera has a frame younger than a second.
`frames[name]` raises `KeyError` when it has none. Pixels follow the OpenCV convention in the
frame's own resolution. Labels are compared after stripping whitespace and lower-casing, and each
label holds one object.

`Detector` supplies these defaults:

* `settings` is None.
* `interval_s` is 1.0, the shortest time between the starts of two calls.
* `pixel_error_px` is 3.0, how far off the pixels may be. The twin pairs the rays of two cameras
  for one label only when they pass this close.
* `start` and `stop` do nothing.

The twin reads `interval_s` and `pixel_error_px` from the instance, so a detector can set them
from its settings. `start(info)` may raise to refuse the start. `stop(reason)` may arrive from
another thread while `detect` runs. A detector that blocks on a subprocess should then make
`detect` return soon. When `detect` raises, the panel shows the error and the twin calls it again
later, with a longer wait after each failure. The twin requires only `detect` at runtime, so
subclassing `Detector` is optional.

## Evolving the contract

Plugins pin `machinekind-twin-api>=0.1,<0.2`. Four rules keep every 0.1.x release working for them:

* A new field goes at the end of a frozen dataclass, with a default.
* Everything `detect` gets lives in `DetectRequest`, so its signature never changes.
* What a detector learns once, at start, lives in `DetectorInfo`.
* Nothing is removed or renamed within 0.x.

## Installing

The package is not on PyPI; it lives in the twin's repository, `packages/machinekind_twin_api`.
`uv sync` in the twin's clone installs it from there, editable.

A plugin installed into the twin's environment finds it there. A plugin's own environment
without the twin takes it from git:

```bash
pip install "machinekind-twin-api @ git+https://github.com/machinekind/twin.git@main#subdirectory=packages/machinekind_twin_api"
```

A plugin package names it by version, `machinekind-twin-api>=0.1,<0.2`, not by that git URL: pip
replaces an installed editable copy with the one from a direct URL, and refuses to install both
in one command. A uv project can point `[tool.uv.sources]` at the git URL instead; pip ignores it.

Tests: `uv run pytest packages/machinekind_twin_api/tests` from the twin's root.

## Licence

Copyright 2026 Machinekind and contributors.

Licensed under the [Apache License, Version 2.0](LICENSE).
See [NOTICE](NOTICE) for attribution.
