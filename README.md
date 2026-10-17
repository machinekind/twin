# Robot arm digital twin

A MuJoCo simulation that knows the same things about the desk as the real
workstation: where the arm stands, where the cameras stand, what optics they
have and how the servos really respond. A control strategy runs in it and on a
real arm through the same code path, through the safety supervisor and with the
same camera perception. The twin holds no robot of its own: robots and their
backends come from installed packages. The SO-101 robot and its backends are the
`soarm` package (the so-arm101 repository). Training and the policies are in
`soarm-lab` (the same repository), and a trained policy runs as the `policy`
strategy in the panel's Strategies tab.

```
  real workstation                            twin (MuJoCo)
 ┌──────────────────────┐   calibration     ┌──────────────────────────┐
 │ arm via its backend  │ ────────────────► │ arm from its robot.yaml  │
 │ USB cameras around   │  card in gripper, │ cameras in the same      │
 │ desk, cube           │  ChArUco, sysid   │ poses, with the same K,  │
 └──────────▲───────────┘                   │ servos as measured       │
            │                               └────────────┬─────────────┘
            │  `policy` strategy through the supervisor  │  twin export:
            │  (the same code path as in simulation)     ▼  scene.xml + station.json
            └────────────────────────────────── training in soarm-lab
```

## The problem it solves

A robot trained in an ordinary simulation usually fails on the real desk. The
simulation does not know where the cameras really stand and what optics they
have, how the particular servos respond, or how delayed and noisy the
perception is, so the policy works in sim and not on the arm. Every new skill
then costs hours of trial and error on the hardware, with a risk of damage.

## Approach

The twin is **measured from the workstation**, not modelled by eye:

- **Cameras** enter the simulation in their measured pose and with their own K
  matrix: intrinsics from a ChArUco board, position from a card with tags held
  in the gripper, all cameras at once.
- **Servos**: dynamics are identified from a recorded motion (damping,
  armature, delay) and leave with the station (`twin export`), so
  training randomizes conditions *around the measured* arm.
- **One code path for simulation and arm**: the same strategy runner, the
  same safety supervisor, the same perception; going from sim to the arm is a
  backend change (`sim` -> the arm's backend: `feetech` for the SO-101 from
  `soarm`, `a1x` for the Galaxea A1X from `galaxeo`).

## What it enables

- **Running control strategies** (a trained policy, a VLA, scripted motion) on
  a real arm through the safety supervisor, the same way as in simulation.
- **Locating objects seen only by the cameras**: no markers on the object; the
  cube is located to within 1-3 mm.
- **Arbitrary placement of many cameras**: a moved camera is detected by
  itself (zoom by 1 %, rotation by 1°), and one camera can be relocalized
  quickly.
- **A measured station as data**: `twin export` writes the scene
  (`scene.xml`, `assets/`) and the station record (`station.json`: cameras,
  identified dynamics) for packages that do not import the twin.
- **Live Sim-Real comparison**: a frame from the real camera with the edges of
  the twin's render from the same pose drawn over it.
- **Safe work with the arm**: one owner of motion at a time; STOP and Home take
  the arm from any owner; reaction to servo errors, link loss and collisions; a
  gripper that is squeezing an object does not let go on STOP.
- **A second arm, the Galaxea A1X**, from its own package (galaxeo, installed
  next to the twin): the same twin, panel and supervisor; in simulation the
  model and camera calibration, on hardware the `a1x` backend over CAN-FD
  (Linux, macOS, Windows) — galaxeo's
  [galaxeo/robot/README.md](https://github.com/machinekind/galaxeo-manipulators/blob/main/galaxeo/robot/README.md).
- **Remote work**: a browser panel with a step-by-step guide and a hint on
  every control.

## What it speeds up

Measured on an RTX A4500:

| | time |
|---|---|
| Calibration of all cameras | one wave, about 20 poses; in simulation 0.16 mm / 0.013° error |
| Servo dynamics identification | about 20 s of motion + a few seconds of fitting |
| From a fresh clone to the panel | `uv sync` -> `uv run twin ui` |

Most iterations (calibration, perception tuning, trying a strategy on the whole
chain) happen in the twin; on the hardware only the check is left.

## Status

The tests are integration tests only: chains through several modules and
behaviours dangerous to the hardware. The Galaxea A1X (from galaxeo) works in
simulation (camera calibration 6/6 trusted, IK, wave); its CAN backend is
tested on a simulated bus. **The whole thing has not yet been tested on a real
arm (neither the SO-101 nor the A1X)**: that is the next step, with a
step-by-step procedure in [docs/TWIN.md](docs/TWIN.md).
Decisions and pitfalls: [HANDOFF.md](HANDOFF.md).

---

## Quick start on a new computer

The repository holds everything needed to run it, including models and data:

| Directory | What is in it | In the repository? |
|---|---|---|
| `examples/robots/testbot/` | the example robot: a robot folder and a backend that keeps the arm in memory | yes |
| `examples/twin.sim.json` | example workstation in simulation: two cameras with a trusted calibration; the first `twin ui` and `twin demo` write the robot into its copy | yes |
| `assets/calib/*.txt` | example lens (Logitech Brio 4K 1280x720): K and distortion to load in Calibration, step 1 | yes |
| `workspace/twin.json` | **your** workstation: cameras and their calibration, port, table, measured dynamics | no: per desk, created by the first `twin ui` |
| `workspace/calib/*.json` | every save of the camera poses (Calibration, step 2) as a separate file; the panel loads the newest, older ones can be picked from a list | yes: example sets from the A1X waves |
| `configs/local.yaml` | your configuration (a copy of `configs/default.yaml`), e.g. a backend section | no (`.gitignore`) |

When run from a clone, the default paths (`workspace/…`) are counted from the
repository root, so `twin ui` started from another directory lands in
the same workstation. Paths given in flags are counted from the current
directory as usual.

**Requirements:** [uv](https://docs.astral.sh/uv/getting-started/installation/), Git and
OpenGL (any graphics card, including an integrated one). uv installs Python 3.12 and the package
versions in `uv.lock`.

### 1. Installation

```bash
git clone https://github.com/machinekind/twin.git
cd twin
uv sync
```

`uv sync` creates `.venv` with the twin and its panel, the plugin contracts from
`packages/machinekind_twin_api`, the example robot and strategy from `examples/`, and pytest,
pyright and ruff. `uv run <command>` runs a command in it, so nothing needs activating.

uv itself: `curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS and Linux,
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows.

Linux (x86_64, glibc ≥ 2.28, e.g. Ubuntu 22.04/24.04) needs the OpenGL libraries:
`sudo apt install libgl1 libglib2.0-0 libegl1`. On a server without a display:
`export MUJOCO_GL=egl` (NVIDIA), or without a GPU `sudo apt install libosmesa6` and
`export MUJOCO_GL=osmesa`.

Short on space on `C:`? Put uv's cache on another drive before `uv sync`:
`$env:UV_CACHE_DIR='D:\uv-cache'` (cmd.exe: `set UV_CACHE_DIR=D:\uv-cache`).

`packages/machinekind_twin_api` is `machinekind-twin-api`, the contracts the twin shares with robot
and strategy packages ([below](#control-strategies)). It is not on PyPI, so `uv sync` installs it
from the clone.

The twin holds no robot of its own. Robots are packages installed next to it with
`uv pip install --no-sources` (sections 5 and 6). `--no-sources` keeps the clone's copy of the
contracts: a robot repository may name its own copy for its development. `uv run` keeps them, a plain `uv sync` removes them, and
`uv sync --inexact` keeps them.

### 2. Check

```bash
uv run twin check
```

It prints the versions (mujoco, viser, cv2, numpy), the installed arm models
and the USB-serial ports: `OK` for every ready item, `BLAD` (error) for a
missing one. With no robot installed it names what to install.

### 3. The twin's panel

```bash
uv run twin ui       # http://localhost:8080
```

On the first start, with no `workspace/twin.json` yet, `twin ui` writes the example
workstation: `examples/twin.sim.json` with the first installed robot (after a bare `uv sync`, the
example robot) and two simulated cameras
with a trusted calibration. `twin demo` writes it again: `--force` overwrites your workstation,
and `--robot NAME` picks another installed robot.

Port 8080 taken (another server, a proxy)? `uv run twin ui --port 8765` and the address
`http://localhost:8765`. At startup the panel prints in the console the full path of the
workstation file and its state: `istnieje` (exists), `nowe: przykladowe stanowisko` (the example
was just written) or `nowe, puste` (new and empty, when no robot is installed). It shows that the
`workspace/twin.json` from the clone is used, also when the panel was started from another
directory. `twin workspace` shows the same.

In the panel: the **Ramię** tab -> `sim` -> **Połącz**; the **Mapa** tab shows
the table stitched from the two cameras, and the **Strategie** tab runs the
strategies of installed packages ([below](#control-strategies)). Cameras are added in the
Kamery tab (`Dodaj symulowana przed ramieniem`, then calibration, or in the Kalibracja tab
`Symulacja: pomin - wpisz prawdziwe polozenie kamer`).
The tab with the panel must be visible: viser draws nothing in a background
tab.

### 4. Tests

```bash
uv run pytest -q
uv run pytest -q -m "not render"     # a computer without OpenGL
uv run pyright                       # the contracts, the strategies and the example packages
```

The tests of the A1X in the twin (`tests/test_a1x_in_twin.py`) skip themselves without the
`galaxeo` package. With galaxeo installed too (section 6) the result is all `passed`.

### 5. A real SO-101

The SO-101 robot and its backends (`feetech`, `lerobot`, the COM<->TCP bridge)
are the `soarm` package from the so-arm101 repository. Install it next to the twin, with
`soarm-lab` for the `policy` and `ik` strategies:

```bash
uv pip install --no-sources "soarm[feetech] @ git+https://github.com/machinekind/so-arm101.git#subdirectory=packages/soarm" \
                            "soarm-lab @ git+https://github.com/machinekind/so-arm101.git#subdirectory=packages/soarm_lab"
```

Its `docs/first-contact.md` covers the port and the gripper ticks. In the
panel, **Ramię** -> `SO-101 przez port (feetech)`: with one USB-serial device plugged in, the
Port field fills itself. The `feetech` backend reads its `feetech:` section from your copy of
the configuration:

```bash
cp configs/default.yaml configs/local.yaml      # Windows: copy configs\default.yaml configs\local.yaml
uv run twin --config configs/local.yaml ui
```

Next: [docs/TWIN.md](docs/TWIN.md), "Pierwszy test na prawdziwym stanowisku"
(first test on a real workstation), from step 0.

### 6. Galaxea A1X (CAN)

The Galaxea A1X is a robot package of its own, in galaxeo (the
galaxeo-manipulators repository): the robot folder, an example workstation and
the `a1x` backend over CAN-FD. Install it next to the twin, which finds it
through its entry points:

```bash
uv pip install --no-sources "galaxeo @ git+https://github.com/machinekind/galaxeo-manipulators.git@main"
# macOS (XCAN dongle): "galaxeo[xcan] @ git+..." and brew install libusb; Windows (PCAN): "galaxeo[pcan] @ git+..."
uv run twin demo --robot a1x      # an A1X workstation in simulation (does not overwrite an existing one without --force)
```

The hardware, the `a1x` section of the configuration and the first session are
in galaxeo's
[galaxeo/robot/README.md](https://github.com/machinekind/galaxeo-manipulators/blob/main/galaxeo/robot/README.md).
**The A1X has no brakes**: read it before the first session.

---

## When the system does not see a USB device

A camera or a CAN adapter plugged into the physical computer **does not**
reach a virtual machine or a remote desktop by itself. It has to be passed
through explicitly:

| Environment | Where to enable it |
|---|---|
| VirtualBox | Devices -> USB -> pick the device (plus the Extension Pack) |
| VMware | VM -> Removable Devices -> the device -> Connect |
| Hyper-V | no USB redirection: use the enhanced session mode or `usbipd-win` |
| WSL2 | `usbipd bind --busid <id>` and `usbipd attach --wsl --busid <id>` |
| Remote desktop (RDP) | in the client: Local Resources -> More -> tick the device |

If a camera is visible in the guest system but returns no frame (OpenCV opens
it and right away gets `ERROR_OPERATION_ABORTED`), it is usually not the
driver's fault: it is the same USB redirection that has to be enabled.

---

## Control strategies

Every way of driving the arm goes through one contract, `machinekind_twin_api.strategy` in the
`machinekind-twin-api` package ([packages/machinekind_twin_api](packages/machinekind_twin_api)); the twin re-exports
it from `machinekind_twin.strategy`. A strategy has a rate `hz` and `act(observation)`. The
observation holds the time, the measured joints, the supervisor's command, the panel's goal, the
objects the perception knows (pose, source, age) and the camera frames with their age, fetched
only when read. `act` returns joint targets and, when the run is over, `done` with the task's
`success` and `error`.

`StrategyRunner` claims the arm and sends every target through the safety supervisor, the
same way in simulation and on hardware. It stops the run on a target that names another
robot's joint or is not a finite number, on Home, STOP, reconnect or a disconnected arm, and
at the time limit. A strategy from another package needs nothing from the twin: it returns an
`Action`, an object with its attributes, or a plain mapping of joint targets. The test strategy
`waypoints.Waypoints` drives through given poses.

Strategies come from installed packages, the way robots do. A package registers its strategy
class through an entry point and declares what the operator sets for one run (a VLA's
instruction, a policy file) as a dataclass in the class's `settings` attribute:

```toml
[project.entry-points."machinekind_twin.strategies"]
pi05 = "vla_strategies.pi05:Pi05"
```

The panel's Strategies tab lists them and draws a form from the settings dataclass: a text
field, a number or a switch per field, prefilled from the config file's section of the same
name (`pi05:`). Start builds the dataclass strictly, so an unknown key, a missing one or a
wrong type is refused before the arm moves, and creates the strategy as `Pi05(settings)`.
The contract types live in `machinekind-twin-api`; a strategy package depends on that, never on
`machinekind-twin` (MuJoCo, viser, OpenCV). A strategy subclasses `Strategy[S]`, `S` its settings
dataclass, which gives it the defaults (no settings, no time limit, empty `start` and `stop`)
and lets a type checker check its methods against the contract. [examples/strategies/swing](examples/strategies/swing) is a complete example, and the
tests load it.

## Configuration

The twin reads a configuration file only when one is given:
`twin --config configs/local.yaml ui`. The model
is `configs/default.yaml`, described by comments; a copy `configs/local.yaml` is
in `.gitignore`. The twin reads the `safety` and `robot` (backend, port)
sections; every other section belongs to the backend of that name (`feetech`
from `soarm`, `a1x` from galaxeo), which checks it itself, or to the strategy of that name, whose
settings form it prefills. The file can be shortened: missing
keys take their default values, and an unknown key inside a section ends with a
clear error. The arm's backend and port are chosen by the panel, not by the
file. The keys are described in [docs/TWIN.md](docs/TWIN.md), the paragraph
"Konfiguracja ramienia"; a robot package documents its own backend's section (the
`a1x` section: galaxeo's [galaxeo/robot/README.md](https://github.com/machinekind/galaxeo-manipulators/blob/main/galaxeo/robot/README.md)).

## Adding a robot

A robot is a folder: `robot.yaml` and the MJCF model it names. `robot.yaml` holds
what the model does not: the joints in app order, the gripper and its range in
model units, the tool point, joint limits and speeds, the home pose, the desk the
arm works on, and how the panel labels the joints and connects the arm. The
loader (`machinekind_twin.robots`) checks every field and every joint name. An
optional `cameras:` mapping mounts cameras on bodies of the arm (body, 4x4 pose
in OpenCV axes, `fovy_deg`, size): a strategy gets their frames in
`Observation.frames`, rendered in simulation and, with a real arm, from the
workspace camera of the same name. The folder may also ship `demo.json`, an
example workstation in simulation that `twin demo --robot NAME` copies.

The twin holds no robot: it finds them only in installed packages. A robot's
package registers its folder and its hardware backend through entry points and
never imports the twin: the contract types live in `machinekind-twin-api`
([packages/machinekind_twin_api](packages/machinekind_twin_api)), and a plugin depends on that,
never on `machinekind-twin`. The SO-101 is such a package: `soarm`, from the
so-arm101 repository; the Galaxea A1X is another: `galaxeo`, from the
galaxeo-manipulators repository. A package registers both in its `pyproject.toml`:

```toml
[project.entry-points."machinekind_twin.robots"]
myarm = "myarm:ROBOT_DIR"              # a path to the folder with robot.yaml

[project.entry-points."machinekind_twin.backends"]
myarm = "myarm.backend:create"         # create(context) -> the backend
```

The backend is any object with `connect`, `disconnect`, `read_joints`,
`send_joints` and `is_connected`; the optional methods are listed on
`machinekind_twin_api.BackendPlugin`. `create` gets a `machinekind_twin_api.BackendContext`: the
robot's joints and limits, the port and the backend's own config section.
[examples/robots/testbot](examples/robots/testbot) is a complete example, and the
tests load it. Entry points live in the installed package's metadata: after
pulling a version that changes them, run
`uv sync` again, or the panel finds stale backends.

## How it is built

```
packages/machinekind_twin_api/   machinekind-twin-api: the plugin contracts (strategy, backend, entry point groups), numpy only
src/machinekind_twin/   the twin: MuJoCo scene, the arm loop with the supervisor, cameras, perception, the twin command
├── calib/         intrinsics from ChArUco, camera poses from the card in the gripper
├── sysid.py       servo dynamics identification; dynamics.py is its result
├── strategy.py    the control-strategy runner (the contract re-exported); waypoints.py is the test strategy
├── strategies.py  strategies from installed packages and their settings
├── ui/            browser panel (viser)
├── backends/      the backend contract (re-exported) and plugin registry
└── paths.py       default data paths counted from the clone's root
```

The arm in simulation is MuJoCo physics in the twin's scene; a real arm is a
backend from the robot's package, and the scene only mirrors it. The rest of the twin sees
no difference: the same strategy runner, the same supervisor and the same
perception.

## Licence

Apache-2.0 ([NOTICE](NOTICE)). The repository holds no third-party files: robot
models and meshes ship with the robot packages.
