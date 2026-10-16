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
            │  `policy` strategy through the supervisor  │  lerobot-twin export:
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
  armature, delay) and leave with the station (`lerobot-twin export`), so
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
- **A measured station as data**: `lerobot-twin export` writes the scene
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
| From a fresh clone to the panel | about 6 min of installation -> `lerobot-twin demo` -> `lerobot-twin ui` |

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
| `examples/twin.sim.json` | example workstation in simulation: two cameras with a trusted calibration; `lerobot-twin demo` writes the chosen robot into its copy | yes |
| `assets/calib/*.txt` | example lens (Logitech Brio 4K 1280x720): K and distortion to load in Calibration, step 1 | yes |
| `workspace/twin.json` | **your** workstation: cameras and their calibration, port, table, measured dynamics | no: per desk, created on the first save in the panel or by `lerobot-twin demo` |
| `workspace/calib/*.json` | every save of the camera poses (Calibration, step 2) as a separate file; the panel loads the newest, older ones can be picked from a list | yes: example sets from the A1X waves |
| `configs/local.yaml` | your configuration (a copy of `configs/default.yaml`), e.g. a backend section | no (`.gitignore`) |

When run from a clone, the default paths (`workspace/…`) are counted from the
repository root, so `lerobot-twin ui` started from another directory lands in
the same workstation. Paths given in flags are counted from the current
directory as usual.

**Requirements:** Python **3.12** (tested with 3.12.10), Git, OpenGL (any
graphics card, including an integrated one). The tested package versions are in
[`constraints.txt`](constraints.txt).

### 1. Installation

**Windows** (PowerShell):

```powershell
git clone https://github.com/machinekind/digital_twin_training.git
cd digital_twin_training
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass   # this window only; without it Activate.ps1 is blocked
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e packages/lerobot_mp_api -e ".[twin,dev]" -c constraints.txt
```

A fresh Windows blocks PowerShell scripts ("running scripts is disabled on this
system"), hence `Set-ExecutionPolicy` before activation: in every new window
before `.venv\Scripts\activate`, or once for good:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. In `cmd.exe` activation
is `.venv\Scripts\activate.bat` and nothing more is needed. You can also skip
activation and write `.venv\Scripts\python -m pip …`,
`.venv\Scripts\lerobot-twin …`.

Short on space on `C:`? Before the installation (in the same PowerShell window)
move the temporary files and the pip cache to another drive:

```powershell
$env:TMP='D:\tmp'; $env:PIP_CACHE_DIR='D:\tmp\pipcache'   # cmd.exe: set TMP=D:\tmp & set PIP_CACHE_DIR=D:\tmp\pipcache
```

**Linux** (x86_64, glibc ≥ 2.28, e.g. Ubuntu 22.04/24.04) **and macOS** (Apple Silicon):

```bash
sudo apt install python3.12 python3.12-venv libgl1 libglib2.0-0 libegl1   # Linux only; Ubuntu 22.04: python3.12 from the deadsnakes PPA
git clone https://github.com/machinekind/digital_twin_training.git && cd digital_twin_training
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e packages/lerobot_mp_api -e ".[twin,dev]" -c constraints.txt
```

On a Linux server without a display: `export MUJOCO_GL=egl` (NVIDIA), or
without a GPU `sudo apt install libosmesa6` and `export MUJOCO_GL=osmesa`.

`packages/lerobot_mp_api` is `lerobot-mp-api`, the contracts the twin shares with robot and
strategy packages ([below](#control-strategies)). It is not on PyPI, so it is installed from the
clone in the same command as the twin; `pip install -e ".[…]"` alone stops at "No matching
distribution found for lerobot-mp-api". uv reads the same path from `[tool.uv.sources]` in
`pyproject.toml` (`uv sync`, `uv run`).

The twin holds no robot. Install one next to it: `pip install soarm` for the
SO-101 (the so-arm101 repository), or `pip install -e examples/robots/testbot`
for the example robot.

### 2. Check

```bash
lerobot-twin check
```

It prints the versions (mujoco, viser, cv2, numpy), the installed arm models
and the USB-serial ports: `OK` for every ready item, `BLAD` (error) for a
missing one. With no robot installed it names what to install.

### 3. The twin's panel with the example workstation

```bash
lerobot-twin demo     # examples/twin.sim.json with the first installed robot -> workspace/twin.json (does not overwrite an existing one without --force)
lerobot-twin ui       # http://localhost:8080
```

`lerobot-twin demo --robot NAME` picks another installed robot. With no robot
installed, `demo` stops and names what to install (`pip install soarm`, or
`pip install -e examples/robots/testbot`).

Port 8080 taken (another server, a proxy)? `lerobot-twin ui --port 8765` and
the address `http://localhost:8765`. At startup the panel prints in the console
the full path of the workstation file (`istnieje` or `nowe, puste`: exists or
new, empty), so you can see that the `workspace/twin.json` from the clone is
used, also when the panel was started from another directory. `lerobot-twin workspace` shows the same.

In the panel: the **Ramię** tab -> `sim` -> **Połącz**; the **Mapa** tab shows
the table stitched from the two cameras, and the **Strategie** tab runs the
strategies of installed packages ([below](#control-strategies)). Without `demo`
the twin starts with an empty workstation (the first installed robot, backend
`sim`, no cameras); cameras are
added in the Kamery tab (`Dodaj symulowana przed ramieniem`, then calibration,
or in the Kalibracja tab `Symulacja: pomin - wpisz prawdziwe polozenie kamer`).
The tab with the panel must be visible: viser draws nothing in a background
tab.

### 4. Tests

```bash
pytest -q
pytest -q -m "not render"     # a computer without OpenGL
pyright                       # the contracts, the strategies and the example packages
```

The twin's tests skip themselves without `mujoco` or `viser`, and the tests
of the A1X in the twin (`tests/test_a1x_in_twin.py`) without the `galaxeo`
package. With `.[twin,dev]` and galaxeo installed next to the twin (section 6)
the result is all `passed`.

### 5. A real SO-101

The SO-101 robot and its backends (`feetech`, `lerobot`, the COM<->TCP bridge)
are the `soarm` package (the so-arm101 repository), installed next to the twin:
`pip install "soarm[feetech]"`. Its `docs/first-contact.md` covers the port and
the gripper ticks. The `feetech` backend reads its `feetech:` section from your
copy of the configuration:

```bash
cp configs/default.yaml configs/local.yaml      # Windows: copy configs\default.yaml configs\local.yaml
lerobot-twin --config configs/local.yaml ui
```

Next: [docs/TWIN.md](docs/TWIN.md), "Pierwszy test na prawdziwym stanowisku"
(first test on a real workstation), from step 0.

### 6. Galaxea A1X (CAN)

The Galaxea A1X is a robot package of its own, in galaxeo (the
galaxeo-manipulators repository): the robot folder, an example workstation and
the `a1x` backend over CAN-FD. Install it next to the twin, which finds it
through its entry points:

```bash
pip install "galaxeo @ git+https://github.com/machinekind/galaxeo-manipulators.git@main"
# macOS (XCAN dongle): "galaxeo[xcan] @ git+..." and brew install libusb; Windows (PCAN): "galaxeo[pcan] @ git+..."
lerobot-twin demo --robot a1x      # an A1X workstation in simulation (does not overwrite an existing one without --force)
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

Every way of driving the arm goes through one contract, `lerobot_mp_api.strategy` in the
`lerobot-mp-api` package ([packages/lerobot_mp_api](packages/lerobot_mp_api)); the twin re-exports
it from `lerobot_mp.twin.strategy`. A strategy has a rate `hz` and `act(observation)`. The
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
[project.entry-points."lerobot_mp.strategies"]
pi05 = "vla_strategies.pi05:Pi05"
```

The panel's Strategies tab lists them and draws a form from the settings dataclass: a text
field, a number or a switch per field, prefilled from the config file's section of the same
name (`pi05:`). Start builds the dataclass strictly, so an unknown key, a missing one or a
wrong type is refused before the arm moves, and creates the strategy as `Pi05(settings)`.
The contract types live in `lerobot-mp-api`; a strategy package depends on that, never on
`lerobot-mp` (MuJoCo, viser, OpenCV). A strategy subclasses `Strategy[S]`, `S` its settings
dataclass, which gives it the defaults (no settings, no time limit, empty `start` and `stop`)
and lets a type checker check its methods against the contract. [examples/strategies/swing](examples/strategies/swing) is a complete example, and the
tests load it.

## Configuration

The twin reads a configuration file only when one is given:
`lerobot-twin --config configs/local.yaml ui`. The model
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
loader (`lerobot_mp.twin.robots`) checks every field and every joint name. The
folder may also ship `demo.json`, an example workstation in simulation that
`lerobot-twin demo --robot NAME` copies.

The twin holds no robot: it finds them only in installed packages. A robot's
package registers its folder and its hardware backend through entry points and
never imports the twin: the contract types live in `lerobot-mp-api`
([packages/lerobot_mp_api](packages/lerobot_mp_api)), and a plugin depends on that,
never on `lerobot-mp`. The SO-101 is such a package: `soarm`, from the
so-arm101 repository; the Galaxea A1X is another: `galaxeo`, from the
galaxeo-manipulators repository. A package registers both in its `pyproject.toml`:

```toml
[project.entry-points."lerobot_mp.robots"]
myarm = "myarm:ROBOT_DIR"              # a path to the folder with robot.yaml

[project.entry-points."lerobot_mp.backends"]
myarm = "myarm.backend:create"         # create(context) -> the backend
```

The backend is any object with `connect`, `disconnect`, `read_joints`,
`send_joints` and `is_connected`; the optional methods are listed on
`lerobot_mp_api.BackendPlugin`. `create` gets a `lerobot_mp_api.BackendContext`: the
robot's joints and limits, the port and the backend's own config section.
[examples/robots/testbot](examples/robots/testbot) is a complete example, and the
tests load it. Entry points live in the installed package's metadata: after
pulling a version that changes them, run
`pip install -e packages/lerobot_mp_api -e .` again, or the panel finds stale backends.

## How it is built

```
packages/lerobot_mp_api/   lerobot-mp-api: the plugin contracts (strategy, backend, entry point groups), numpy only
src/lerobot_mp/
├── twin/          the twin: MuJoCo scene, the arm loop with the supervisor, cameras, perception, lerobot-twin
│   ├── calib/     intrinsics from ChArUco, camera poses from the card in the gripper
│   ├── sysid.py   servo dynamics identification; dynamics.py is its result
│   ├── strategy.py  the control-strategy runner (the contract re-exported); waypoints.py is the test strategy
│   ├── strategies.py  strategies from installed packages and their settings
│   └── ui/        browser panel (viser)
├── robot/         the backend contract (re-exported) and plugin registry
└── paths.py       default data paths counted from the clone's root
```

The arm in simulation is MuJoCo physics in the twin's scene; a real arm is a
backend from the robot's package, and the scene only mirrors it. The rest of the twin sees
no difference: the same strategy runner, the same supervisor and the same
perception.

## Licence

Apache-2.0 ([NOTICE](NOTICE)). The repository holds no third-party files: robot
models and meshes ship with the robot packages.
