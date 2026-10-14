# SO-101 digital twin

A MuJoCo simulation that knows the same things about the desk as the real
workstation: where the arm stands, where the cameras stand, what optics they
have and how the servos really respond. Policies trained in it on a GPU then run
on a real **SO-101** through the same code path, through the safety supervisor
and with the same camera perception.

```
  real workstation                            twin (MuJoCo)
 ┌──────────────────────┐   calibration     ┌──────────────────────────┐
 │ SO-101 (COM/socket)  │ ────────────────► │ arm from MuJoCo Menagerie│
 │ USB cameras around   │  card in gripper, │ cameras in the same      │
 │ desk, cube           │  ChArUco, sysid   │ poses, with the same K,  │
 └──────────▲───────────┘                   │ servos as measured       │
            │                               └────────────┬─────────────┘
            │  policy through the supervisor             │  thousands of MuJoCo Warp worlds
            │  (same observation layouts)                ▼  PPO on the GPU
            └────────────────────────────────── reach / lift policy
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
  armature, delay); training randomizes conditions *around the measured* arm.
- **Perception**: training sees the cube the way the cameras see it (delayed,
  refreshed rarely, noisy), not the way the physics knows it.
- **One code path for simulation and arm**: the same observation layouts, the
  same safety supervisor, the same perception; going from sim to the arm is a
  backend change (`sim` -> `feetech`, and `sim` -> `a1x` for the Galaxea A1X).

## What it enables

- **Training policies on a GPU** (`reach`, `lift`) and running them on a real
  SO-101 through the safety supervisor.
- **Grasping objects seen only by the cameras**: no markers on the object; the
  cube is located to within 1-3 mm.
- **Arbitrary placement of many cameras**: a moved camera is detected by
  itself (zoom by 1 %, rotation by 1°), and one camera can be relocalized
  quickly.
- **Fine-tuning ready-made policies** on the measured dynamics of your own arm
  instead of training from scratch.
- **Live Sim-Real comparison**: a frame from the real camera with the edges of
  the twin's render from the same pose drawn over it.
- **Safe work with the arm**: one owner of motion at a time; STOP and Home take
  the arm from any owner; reaction to servo errors, link loss and collisions; a
  gripper that is squeezing an object does not let go on STOP.
- **A second arm, the Galaxea A1X**: the same twin, panel and supervisor; in
  simulation the model, camera calibration and RL environments, on hardware the
  `a1x` backend over CAN-FD (Linux, macOS, Windows) —
  [docs/TWIN.md](docs/TWIN.md#galaxea-a1x).
- **Remote work**: the arm through a network bridge (`socket://`), and a
  browser panel with a step-by-step guide and a hint on every control.

## What it speeds up

Measured on an RTX A4500:

| | time |
|---|---|
| GPU physics (MuJoCo Warp) | **1.1 M steps/s**, thousands of worlds at once |
| `reach` training | **about 2 min** |
| `lift` training from scratch / fine-tuning | about 28 min / about 16 min |
| Calibration of all cameras | one wave, about 20 poses; in simulation 0.16 mm / 0.013° error |
| Servo dynamics identification | about 20 s of motion + a few seconds of fitting |
| From a fresh clone to the panel | about 6 min of installation -> `lerobot-twin demo` -> `lerobot-twin ui` |

Most iterations (task design, training, perception tuning, calibration, trying
the whole chain) happen in the twin; on the hardware only the check is left.

## Status

In simulation: the `reach-v3` and `lift-v3` policies have 100 % success on the
CPU (a different engine than in training), and `lift` with the cube seen **only
by the cameras** picks it up 8/8. The tests are integration tests only: chains
through several modules and behaviours dangerous to the hardware. The Galaxea
A1X works in simulation (camera calibration 6/6 trusted, IK, wave, stable RL
physics); its CAN backend is tested on a simulated bus, and there are no
policies for the A1X yet. **The whole thing has not yet been tested on a real
arm (neither the SO-101 nor the A1X)**: that is the next step, with a
step-by-step procedure in [docs/TWIN.md](docs/TWIN.md).
Decisions and pitfalls: [HANDOFF.md](HANDOFF.md).

---

## Quick start on a new computer

The repository holds everything needed to run it, including models and data:

| Directory | What is in it | In the repository? |
|---|---|---|
| `src/lerobot_mp/robot/so101/robot/` | arm model from MuJoCo Menagerie with STL meshes | yes |
| `src/lerobot_mp/robot/a1x/robot/` | Galaxea A1X: the manufacturer's URDF and meshes (the manufacturer's licence, not Apache-2.0, see `NOTICE`) and `a1x.xml` from a script | yes |
| `assets/so101_preview.npz` | SO-101 reference model from Articulus: a test compares the twin's MJCF with it (rebuilt by `scripts/import_articulus_model.py`) | yes |
| `assets/policies/` | base policies `reach-v3` and `lift-v3` (a start for fine-tuning) | yes |
| `examples/twin.sim.json` | example workstation in simulation: two cameras with a trusted calibration | yes |
| `assets/calib/*.txt` | example lens (Logitech Brio 4K 1280x720): K and distortion to load in Calibration, step 1 | yes |
| `examples/twin.a1x.sim.json` | the same for the Galaxea A1X (`lerobot-twin demo --robot a1x`) | yes |
| `workspace/twin.json` | **your** workstation: cameras and their calibration, port, table, measured dynamics | no: per desk, created on the first save in the panel or by `lerobot-twin demo` |
| `workspace/calib/*.json` | every save of the camera poses (Calibration, step 2) as a separate file; the panel loads the newest, older ones can be picked from a list | yes: example sets from the A1X waves |
| `workspace/policies/` | policies you train yourself | no |
| `configs/local.yaml` | your configuration (a copy of `configs/default.yaml`), e.g. gripper ticks | no (`.gitignore`) |

When run from a clone, the default paths (`workspace/…`) are counted from the
repository root, so `lerobot-twin ui` started from another directory lands in
the same workstation. Paths given in flags are counted from the current
directory as usual. Only things created on a given machine stay outside the
repository: the LeRobot calibration (only the `lerobot` backend) and the MuJoCo
Warp kernel cache (about 100 s of compilation at the first training).

**Requirements:** Python **3.12** (tested with 3.12.10), Git, OpenGL (any
graphics card, including an integrated one). Policy training needs an **NVIDIA
card with CUDA** (a driver that supports CUDA 12). Without it everything works
except training: the panel, calibration, perception, dynamics identification
and policy evaluation on the CPU. Disk space: about 5 GB for the `.venv` with
CUDA torch (torch alone is about 3.9 GB), about 1 GB without NVIDIA (CPU torch);
a CUDA install adds about 3 GB of pip cache (`pip install --no-cache-dir …`
does not create it). The tested package versions are in
[`constraints.txt`](constraints.txt).

### 1. Installation

**Windows + NVIDIA card** (PowerShell):

```powershell
git clone https://github.com/machinekind/digital_twin_training.git
cd digital_twin_training
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass   # this window only; without it Activate.ps1 is blocked
.venv\Scripts\activate
python -m pip install --upgrade pip
# CUDA torch FIRST - torch from PyPI on Windows has no CUDA
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e packages/lerobot_mp_api -e ".[twin,train,feetech,dev]" -c constraints.txt
```

A fresh Windows blocks PowerShell scripts ("running scripts is disabled on this
system"), hence `Set-ExecutionPolicy` before activation: in every new window
before `.venv\Scripts\activate`, or once for good:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. In `cmd.exe` activation
is `.venv\Scripts\activate.bat` and nothing more is needed. You can also skip
activation and write `.venv\Scripts\python -m pip …`,
`.venv\Scripts\lerobot-twin …`.

RTX 50xx cards: `cu128` instead of `cu126`. Short on space on `C:`? Before the
installation (in the same PowerShell window) move the temporary files and the
pip cache to another drive:

```powershell
$env:TMP='D:\tmp'; $env:PIP_CACHE_DIR='D:\tmp\pipcache'   # cmd.exe: set TMP=D:\tmp & set PIP_CACHE_DIR=D:\tmp\pipcache
```

**Linux + NVIDIA card** (x86_64, glibc ≥ 2.28, e.g. Ubuntu 22.04/24.04):

```bash
sudo apt install python3.12 python3.12-venv libgl1 libglib2.0-0 libegl1   # Ubuntu 22.04: python3.12 from the deadsnakes PPA
git clone https://github.com/machinekind/digital_twin_training.git && cd digital_twin_training
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e packages/lerobot_mp_api -e ".[twin,train,feetech,dev]" -c constraints.txt
sudo usermod -aG dialout $USER    # access to the arm's port (log out and in after this)
```

Torch from PyPI on Linux is built with CUDA 13 and needs a driver ≥ 580, which
is why the `cu126` index is used here too. On a server without a display:
`export MUJOCO_GL=egl`.

**Without an NVIDIA card**: the same steps, only torch from the `cpu` index and
without `train`:

```powershell
# Windows (PowerShell)
git clone https://github.com/machinekind/digital_twin_training.git
cd digital_twin_training
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e packages/lerobot_mp_api -e ".[twin,feetech,dev]" -c constraints.txt
```

```bash
# Linux (system packages as above) and macOS on Apple Silicon
git clone https://github.com/machinekind/digital_twin_training.git && cd digital_twin_training
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu   # macOS: pip install torch==2.11.0
pip install -e packages/lerobot_mp_api -e ".[twin,feetech,dev]" -c constraints.txt
```

Linux without a display and without a GPU: additionally
`sudo apt install libosmesa6` and `export MUJOCO_GL=osmesa`.

`packages/lerobot_mp_api` is `lerobot-mp-api`, the contracts the twin shares with robot and
strategy packages ([below](#control-strategies)). It is not on PyPI, so it is installed from the
clone in the same command as the twin; `pip install -e ".[…]"` alone stops at "No matching
distribution found for lerobot-mp-api". uv reads the same path from `[tool.uv.sources]` in
`pyproject.toml` (`uv sync`, `uv run`).

### 2. Check

```bash
lerobot-twin check
```

It prints the versions (mujoco, viser, torch, cv2…), the SO-101 model and the
USB-serial ports, and separately the readiness for training: torch with CUDA
(`cuda.is_available`, the card), warp and mujoco_warp: `OK` for every ready
item, `--` for an unavailable one. "Trening na GPU: niedostepny" (GPU training:
unavailable) on a computer without NVIDIA is not an error.

### 3. The twin's panel with the example workstation

```bash
lerobot-twin demo     # examples/twin.sim.json -> workspace/twin.json (does not overwrite an existing one without --force)
lerobot-twin ui       # http://localhost:8080
```

Port 8080 taken (another server, a proxy)? `lerobot-twin ui --port 8765` and
the address `http://localhost:8765`. At startup the panel prints in the console
the full path of the workstation file (`istnieje` or `nowe, puste`: exists or
new, empty) and the policies directory, so you can see that the
`workspace/twin.json` from the clone is used, also when the panel was started
from another directory. `lerobot-twin workspace` shows the same.

In the panel: the **Ramię** tab -> `sim` -> **Połącz**; the **Polityki** tab ->
`lift-v3`, `lift: skad polozenie kostki` = **kamery**, and the cube is placed by
the button `lift (sim): poloz kostke losowo`. Without `demo` the twin starts
with an empty workstation (arm `sim`, no cameras); cameras are added in the
Kamery tab (`Dodaj symulowana przed ramieniem`, then calibration, or in the
Kalibracja tab `Symulacja: pomin - wpisz prawdziwe polozenie kamer`).
The tab with the panel must be visible: viser draws nothing in a background
tab.

### 4. Tests

```bash
pytest -q
pytest -q -m "not render"     # a computer without OpenGL
```

The twin's tests skip themselves without `mujoco`, `torch` or `viser`, and the
`a1x` backend tests without the `galaxeo` package (the `a1x` extra). With the
full installation (`.[twin,a1x,dev]`) the result is all `passed`.

### 5. Training (NVIDIA only)

```bash
lerobot-twin train --task reach --iters 160
lerobot-twin train --init assets/policies/reach-v3/policy.pt --iters 150    # fine-tuning a base policy
lerobot-twin eval workspace/policies/<name>/policy.pt --rand                # evaluation, runs on the CPU
```

The results go to `workspace/policies/<name>/`. On a card with less than 20 GB
of memory add `--envs 2048` or `1024`.

### 6. A real SO-101

The twin uses the `feetech` backend (`pyserial` alone, without LeRobot and its
calibration). The port is shown by `lerobot-twin check` or "Wykryj porty" in the
panel (Windows `COMx`, Linux `/dev/ttyACM0`). The gripper ticks in the
`feetech:` section of the configuration (`gripper_closed_ticks` 1986 /
`gripper_open_ticks` 2670) are the values of the example arm: check yours, put
them in your copy of the configuration and match `gripper_range_q` in
`src/lerobot_mp/robot/so101/robot/robot.yaml` (the panel warns when the two disagree):

```bash
cp configs/default.yaml configs/local.yaml      # Windows: copy configs\default.yaml configs\local.yaml
lerobot-twin --config configs/local.yaml ui
```

Next: [docs/TWIN.md](docs/TWIN.md), "Pierwszy test na prawdziwym stanowisku"
(first test on a real workstation), from step 0.

### 7. Galaxea A1X (CAN)

The `a1x` backend talks to the arm over CAN-FD through the `galaxeo` package
(protocol and link, the galaxeo-manipulators repository):

```bash
pip install -e packages/lerobot_mp_api -e ".[twin,a1x,dev]" -c constraints.txt   # needs the merged `pkg` branch in galaxeo-manipulators
# until it is merged: ".[twin,dev]" and pip install -e ../galaxeo-manipulators   (a clone on the `pkg` branch)
# macOS (XCAN dongle): additionally pip install pyusb and brew install libusb
# Windows (PCAN): additionally pip install python-can
lerobot-twin demo --robot a1x      # an A1X workstation in simulation (does not overwrite an existing one without --force)
```

In the panel: Ramię = `a1x`, Port = the CAN interface (`can0` on Linux after
`./can_up.sh`, `xcan` or `xcan:<usb address>` on macOS, `PCAN_USBBUS1` on
Windows). The parameters (joint zeros, gripper coupling, thresholds) are in the
`a1x` section of the configuration; its keys are described in
[docs/TWIN.md](docs/TWIN.md#galaxea-a1x). After "Połącz" nothing goes onto the
bus until the first move; the FF 1-5-6 enable sequence is manual only ("Włącz
silniki A1X"). **The A1X has no brakes**: before the first session read
[docs/TWIN.md](docs/TWIN.md#galaxea-a1x), "Galaxea A1X".

---

## When the system does not see the arm

The arm is not a USB device by itself: it shows up as a **USB-serial adapter**
on the controller board (most often a CH340, more rarely a CP210x or FTDI), and
only that creates the COM port. The twin has no way to "search" for it: if the
port is not in the system, there is nothing to open.

Check what the system sees:

```bash
python -c "import serial.tools.list_ports as l; print([p.device for p in l.comports()] or 'BRAK')"
```

An empty list means one of three things, in this order of likelihood:

1. **The cable**: many USB cables carry only the power wires. The arm lights
   up and there is no port. Swap the cable for one you know carries data.
2. **The adapter's driver**: the chip reached the system, but without a driver
   it sits in Device Manager as an unknown device instead of a COM port. Look in
   Device Manager for an entry with a yellow exclamation mark, check its
   `VID`/`PID` and install the driver: `1A86:7523` is a CH340, `10C4:EA60` is a
   CP210x, `0403:6001` is an FTDI.
3. **A virtual machine or a remote desktop**: a device plugged into the
   physical computer **does not** reach the guest system by itself. It has to be
   passed through explicitly:

| Environment | Where to enable it |
|---|---|
| VirtualBox | Devices -> USB -> pick the device (plus the Extension Pack) |
| VMware | VM -> Removable Devices -> the device -> Connect |
| Hyper-V | no USB redirection: use the enhanced session mode or `usbipd-win` |
| WSL2 | `usbipd bind --busid <id>` and `usbipd attach --wsl --busid <id>` |
| Remote desktop (RDP) | in the client: Local Resources -> More -> tick the device |

The same rule applies to a camera. If the camera is visible in the guest system
but returns no frame (OpenCV opens it and right away gets
`ERROR_OPERATION_ABORTED`), it is usually not the driver's fault: it is the same
USB redirection that has to be enabled.

---

## The arm on one machine, the twin on another

Redirecting the whole USB over the network works well on a LAN and on a local
hypervisor, and badly over the internet, and not because "it is slower". The
Feetech protocol is request-response: the twin sends a frame and **waits** for
the servo's answer before it sends the next one. Locally one such transaction
costs **0.29 ms**. With USB redirection it costs as much as the link's RTT,
times the number of URB round trips the driver needs for one frame.
`feetech-servo-sdk` takes about **34 ms** for an answer at 1 Mbaud, so this is a
race not worth starting on a link with an RTT of about 20 ms.

The bridge carries **only the byte stream** over the network, not USB
transactions, so one frame is one round trip instead of several. On the machine
with the arm:

```bash
pip install -e packages/lerobot_mp_api -e ".[bridge]"
lerobot-mp-bridge --port COM5 --listen 0.0.0.0:5555 --allow <IP-of-the-remote-machine>
```

On the machine with the twin, in the panel: Ramię = "SO-101 przez port
(feetech)", Port = `socket://<address of the machine with the arm>:5555`. The
`feetech` backend connects to the bridge directly, and the arm is read with one
SYNC READ packet: one round trip per read over the network instead of six. A
virtual serial port over TCP also works (**HW VSP3** or **com0com + com2tcp** on
Windows, `socat` on Linux) entered in the Port field like an ordinary `COM3`.

Measured: a ping of all six servos through the bridge over loopback is
**0.40 ms** against 0.29 ms directly, so the bridge itself costs **0.11 ms** and
the whole rest of the budget is the link's RTT. `--stats 5` shows the traffic
live when you need to check whether anything flows at all.

Two things worth remembering:

* `--allow` is not decoration. An open TCP port on the other side moves a
  physical arm: without an address list the bridge warns in the log and
  listens to everyone.
* A network disconnect **does not drop the arm**: the servos hold the last
  commanded position. The bridge then closes the port so that the next client
  does not find it busy.

What is left is the camera: it has to be on the machine where the vision runs.
The network carries the serial port here, not the image.

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
`lerobot-mp` (MuJoCo, viser, torch). The generic `Strategy[S]` protocol is there for its type
checker. [examples/strategies/swing](examples/strategies/swing) is a complete example, and the
tests load it.

## Configuration

The twin reads a configuration file only when one is given:
`lerobot-twin --config configs/local.yaml ui` (also `train`, `eval`). The model
is `configs/default.yaml`, described by comments; a copy `configs/local.yaml` is
in `.gitignore`. The twin reads the `safety` and `robot` (backend, port)
sections; every other section belongs to the backend of that name (`feetech`,
`lerobot`, `a1x`), which checks it itself, or to the strategy of that name, whose
settings form it prefills. The file can be shortened: missing
keys take their default values, and an unknown key inside a section ends with a
clear error. The arm's backend and port are chosen by the panel, not by the
file. The keys are described in [docs/TWIN.md](docs/TWIN.md): the paragraph
"Konfiguracja ramienia" and the chapter [Galaxea A1X](docs/TWIN.md#galaxea-a1x),
which has the `a1x` section.

## Adding a robot

A robot is a folder: `robot.yaml` and the MJCF model it names. `robot.yaml` holds
what the model does not: the joints in app order, the gripper and its range in
model units, the tool point, joint limits and speeds, the home pose, the desk the
arm works on, and how the panel labels the joints and connects the arm. The
loader (`lerobot_mp.twin.robots`) checks every field and every joint name.

The twin holds no robot: it finds them only in installed packages. A robot's
package registers its folder and its hardware backend through entry points and
never imports the twin: the contract types live in `lerobot-mp-api`
([packages/lerobot_mp_api](packages/lerobot_mp_api)), and a plugin depends on that,
never on `lerobot-mp`. SO-101 and A1X still live here as such packages,
`src/lerobot_mp/robot/so101/` and `src/lerobot_mp/robot/a1x/`, registered in this
repository's `pyproject.toml` until they move to the so101 repository and to
galaxeo:

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
pulling a version that changes them (this one registers the backends), run
`pip install -e packages/lerobot_mp_api -e .` again, or the panel finds no backends.

## How it is built

```
packages/lerobot_mp_api/   lerobot-mp-api: the plugin contracts (strategy, backend, entry point groups), numpy only
src/lerobot_mp/
├── twin/          the twin: MuJoCo scene, the arm loop with the supervisor, cameras, perception, lerobot-twin
│   ├── calib/     intrinsics from ChArUco, camera poses from the card in the gripper
│   ├── rl/        RL environments, PPO on the GPU, policies, servo dynamics identification
│   ├── strategy.py  the control-strategy runner (the contract re-exported); waypoints.py is the test strategy
│   ├── strategies.py  strategies from installed packages and their settings
│   └── ui/        browser panel (viser)
├── robot/         the backend contract (re-exported) and plugin registry
│   ├── so101/     SO-101 package, leaving for its repo: robot folder, backends feetech and lerobot, COM<->TCP bridge
│   └── a1x/       A1X package, leaving for galaxeo: robot folder, backend a1x (CAN-FD)
└── paths.py       default data paths counted from the clone's root
```

The arm in simulation is MuJoCo physics in the twin's scene; a real arm is a
backend from `robot/`, and the scene only mirrors it. The rest of the twin sees
no difference: the same observation layouts, the same supervisor and the same
perception.

## Licence

Apache-2.0. Third-party files (the SO-101 model from MuJoCo Menagerie, the
reference model from Articulus, the Galaxea A1X URDF and meshes under the
manufacturer's licence) and their licences: [NOTICE](NOTICE).
