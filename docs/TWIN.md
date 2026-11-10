# Workstation guide

Machinekind Twin builds digital twins of physical workstations, starting with
robot arms. This guide covers the currently implemented arm workflows: simulation,
cameras, calibration, perception, object detectors, dynamics identification and
control strategies.

## Status and package boundaries

The workflows are tested in simulation and with simulated hardware interfaces.
End-to-end validation on physical SO-101 and Galaxea A1X arms is still pending.
Simulation results are not hardware accuracy guarantees.

| Package | Responsibility |
|---|---|
| This repository (`machinekind-twin`) | Scene, calibration, perception, object tracking, safety supervisor, strategy and detector runners, UI and station export |
| [`machinekind-twin-api`](../packages/machinekind_twin_api) | Plugin contracts for robots, backends, strategies and detectors; depends only on NumPy |
| [`machinekind-twin-llm`](../packages/machinekind_twin_llm) | Detector that asks a vision model through the `claude` or `codex` CLI; depends on the API package, NumPy and OpenCV |
| [`soarm`](https://github.com/machinekind/so-arm101/tree/main/packages/soarm) | SO-101 model and hardware backends |
| `soarm-lab` in [so-arm101](https://github.com/machinekind/so-arm101) | Training and policy/IK strategies |
| [`galaxeo`](https://github.com/machinekind/galaxeo-manipulators) | A1X model, example workstation and CAN-FD backend |

Training, policy checkpoints and robot-specific drivers are maintained in their
respective packages. This repository does not provide a training CLI or bundle
trained policies. It passes measured station data to external tools through
`twin export`; installed strategies bring trained policies back to the panel.

## Installation

Requires Git, [uv](https://docs.astral.sh/uv/) and OpenGL. The clone pins Python
through `.python-version` and dependencies through `uv.lock`.

```bash
git clone https://github.com/machinekind/twin.git
cd twin
uv sync
uv run twin check
uv run twin ui
```

`uv sync` installs the twin, browser panel, local plugin API, example `arm6`
robot, `swing` strategy, `cube` detector and development tools. Open
**http://localhost:8080**.
`twin check` reports runtime dependencies, installed robot models and USB-serial
ports; it does not check a training environment.

On Linux, install the system OpenGL libraries if needed:

```bash
sudo apt install libgl1 libglib2.0-0 libegl1
```

On a Linux server without a display, use `MUJOCO_GL=egl` with a compatible GPU,
or install `libosmesa6` and use `MUJOCO_GL=osmesa` for software rendering.

### Install a hardware package

Install hardware plugins into the same environment. For SO-101, with its policy
and IK strategies:

```bash
uv pip install --no-sources "soarm[feetech] @ git+https://github.com/machinekind/so-arm101.git#subdirectory=packages/soarm" \
  "soarm-lab @ git+https://github.com/machinekind/so-arm101.git#subdirectory=packages/soarm_lab"
```

For A1X:

```bash
uv pip install --no-sources "galaxeo @ git+https://github.com/machinekind/galaxeo-manipulators.git@main"
```

`--no-sources` avoids applying a plugin repository's development source overrides
and keeps the API package installed from this clone. `uv run` retains these extra
packages; use `uv sync --inexact` when syncing to keep them. A plain `uv sync`
removes packages outside the project's dependency set. Reinstall a plugin after
changing its entry points so its installed metadata is refreshed.

Read the robot package's hardware instructions before connecting it. For A1X,
see [the first-session procedure](#galaxea-a1x-first-session) below.

## Workstation and configuration

| Path | Contents |
|---|---|
| `workspace/twin.json` | Your local robot selection, backend/port, table, other robots, scene look, cameras, calibration and measured dynamics; ignored by Git |
| `workspace/calib/*.json` | Saved camera-pose presets; example presets are tracked |
| `workspace/episodes/` | Opt-in episode recordings and optional camera images; ignored by Git |
| `workspace/episodes/eval-<strategy>-<time>/` | One folder per `twin eval-strategy` campaign: a recording per run and `summary.json`; ignored by Git |
| `<stands repo>/<stand>/` | A physical stand and its measurement history, outside this repository; see [Stands](#stands) |
| `configs/local.yaml` | Your backend, strategy, detector and safety settings; ignored by Git |
| `examples/twin.sim.json` | Example workstation with two calibrated simulated cameras |
| `assets/calib/` | Example lens intrinsics and distortion files |

On first launch, if `workspace/twin.json` does not exist, the panel creates an
example using the first installed robot. A robot package can supply its own
`demo.json`; otherwise the clone's example is used.

```bash
uv run twin demo --robot arm6 --path workspace/arm6.json
uv run twin ui --workspace workspace/arm6.json
```

An existing workstation is preserved unless `twin demo --force` is explicitly
used. Back up your workstation before replacing it. In a source checkout, default
`workspace/` paths resolve from the repository root even when launched elsewhere.
Explicit relative paths in flags resolve from the current directory.

Copy [configs/default.yaml](../configs/default.yaml) to `configs/local.yaml` and
launch with:

```bash
uv run twin --config configs/local.yaml ui
```

The twin reads `safety`, `robot`, `wave` and optional `joints` overrides. Robot descriptions
supply the joints, limits, gripper range and home pose. Other YAML sections belong
to the backend, strategy or detector with that name (`feetech:`, `a1x:`, `swing:`,
`cube:`, etc.). A strategy and a detector with the same name share a section, and
the panel warns about it. Backends validate their settings. Strategy and detector
sections prefill the panel's settings form and are validated before a start.
`twin eval-strategy` reads the same sections, and `--set KEY=VALUE` overrides one strategy key.
The panel selects the connection backend
and port; `--arm` reconnects using the workstation's saved selection at startup.
With one USB-serial adapter plugged in, its current port takes precedence over
a saved port. With multiple adapters, choose the correct one explicitly.

The panel's **Prędkość** (Speed) slider caps the speed of every arm joint,
strategy runs included. It starts at `safety.panel_speed_deg_s` (default 20
deg/s), clipped to the slider's range; a real arm that moves in jerks at that
speed may need a higher start value. Moving the slider changes the cap at once.

Saving camera positions creates a new preset in `workspace/calib/`. The panel
loads the selected preset, or the newest when none is selected. Trust is
reassessed against the current lens calibration. Older presets remain available
in the panel. A workspace loaded from a stand folder keeps its presets in that
folder instead ([Stands](#stands)).

## Browser panel

The UI currently uses Polish labels. Keep the browser tab visible: viser renders
through the browser's animation loop, which can pause in a background tab.

| Tab | Purpose |
|---|---|
| **Ramię** (Arm) | Backend/port, **Połącz** (Connect), **UZBRÓJ** (Arm manual controls), joint sliders, 3D IK handle, **Dom** (Home), **Ustaw dom = tu** (Set home here) and **Rozłącz** (Disconnect) |
| **Kamery** (Cameras) | Add USB or simulated cameras, preview streams, enable/remove cameras, move simulated cameras in 3D, inspect their status and trust a lens or pose by hand ([Trust by hand](#trust-by-hand)) |
| **Kalibracja** (Calibration) | Step 1: lens intrinsics; step 2: camera poses from a tagged card in the gripper; save/discard results and relocalize moved cameras |
| **Mapa** (Map) | Stitched tabletop view, the objects that running detectors track and a tag-on-table reach action |
| **Dynamika** (Dynamics) | Record supervised excitation, fit servo dynamics and save the result to the workstation |
| **Detektory** (Detectors) | Choose an installed detector, edit settings, start it and stop running ones. Tracked objects show as text and 3D markers |
| **Strategie** (Strategies) | Choose an installed strategy, edit settings, provide an optional 3D goal, start/stop the run. The strategy gets the tracked objects in `Observation.objects`. **Record episode** enables logging; **Save camera frames** also saves sampled images. See [episode recording](EPISODES.md) |
| **Sim-Real** | Real image with simulated edges overlaid using the same camera pose and intrinsics |

The guide (**Przewodnik**, **Teraz**) reports the next action from the current
state. Tooltips explain units and whether a control moves the arm.

### Motion ownership and stopping

All arm motion passes through `runtime.Twin` and `SafetySupervisor`. Only one
owner can drive it at a time: manual panel controls, a strategy, calibration or
dynamics recording. STOP, Home, Connect and Disconnect preempt the current owner.
A task interrupted during startup cannot resume simply by clearing STOP.
Hardware tasks require a fresh confirmation for each run.

STOP holds the measured pose, accounting for recent motion so a moving joint
does not reverse toward an old measurement. A blocked joint is not driven further
toward its old target. A gripper already squeezing an object retains its grip on
STOP or Home; a gripper fault can release that pressure. Open the gripper manually
when appropriate. Reconnecting or disconnecting a hardware backend can release a
held object (notably `feetech`); put the object down or support it first.

The runtime stops on arm servo faults, sustained loss of feedback or a stalled
control loop. Its tracking guard detects an arm joint more than 25 degrees from
its command for over 0.5 s **without sufficient progress toward the command**.
A gripper-only fault first eases grip pressure; a persistent fault triggers STOP
after 2 s. These are current defaults in `runtime.py`, not evidence of physical
hardware validation.

Connection starts from confirmed measurements without an automatic Home move.
A joint initially outside configured limits is left in place and brought back
under supervision when commanded. Hardware-reported limits constrain configured
limits where the backend provides them.

### Gripper strength

A cardboard card needs a firm grip and a hard plastic block a gentle one. A task
sets its own grip strength for as long as it has the arm. The calibration wave
reads it from the config file's `wave:` section. A strategy that has the setting
reads it from its own section, as the SO-101's block strategies do. The twin hands
it to the backend in its control loop. The stand's own grip comes back when the
task ends, stops, fails or is replaced, and when STOP, Home, Connect or Disconnect
takes the arm. The panel's manual gripper always uses the stand's own grip.

The scale is the backend's. On the SO-101 it is the gripper servo's P gain, Feetech
register 21, from 1 to 254. The squeeze on a held object is about P times how far
the goal lies past it. The stand's own grip is the backend's setting,
`feetech: p_gain: {gripper: ...}`. Set it to the softest grip any task needs. Tasks
then only raise it, so nothing left in the jaws is squeezed harder after a task
ends. A backend or robot that cannot change the grip ignores the setting, and the
log says so.

### Remote access and camera streams

```bash
uv run twin ui --port 8765
uv run twin ui --host 0.0.0.0
uv run twin ui --stream-port 8766 --stream-host 127.0.0.1
```

The control panel defaults to localhost and has no authentication. Binding it
to the network lets reachable clients operate the same controls; use a trusted
network. The optional stream server is read-only and defaults to `0.0.0.0`
unless `--stream-host` is supplied. It serves `/`, `/cameras.json`,
`/<camera>.jpg` and `/<camera>.mjpg` from frames already opened by the panel.

For a VM, WSL or remote desktop, explicitly forward physical USB cameras and
adapters to the machine running the twin. Device visibility and frame delivery
are separate checks; an opened camera with no frames can indicate a forwarding
problem. Check the host's device permissions and camera access too.

## First hardware session

Rehearse in `sim` first. Keep the arm secured, its workspace clear and STOP
within reach. The robot package's connection and power instructions take
precedence for that hardware.

1. **Connect the arm.** Install its package, configure its backend section in
   `configs/local.yaml`, then select the backend/port in **Ramię** and press
   **Połącz**. Check that measured joints match the real pose. Enable manual
   controls with **UZBRÓJ** and try a small motion of one joint. Check gripper
   zero and range: the panel's 0–100 maps linearly to `gripper_range_q` from
   `robot.yaml`; 0 should close empty jaws. SO-101 tick calibration belongs in
   the `soarm` backend's `feetech:` section.
2. **Add cameras.** Use **Szukaj kamer USB** (Find USB cameras) and **Dodaj kamerę
   USB** (Add USB camera). Confirm each camera delivers live, unmirrored frames.
3. **Calibrate each lens.** In **Kalibracja**, step 1, download the ChArUco board,
   print at 100%, measure the square size and enter it. A print of OpenCV's sample
   board (7 by 5 squares, 6x6 markers) works too: pick it in the **Tablica**
   dropdown and measure its square. Show the board across the
   image, at different distances and tilts. The session requires coverage and
   rotation diversity; a parallel board alone cannot determine focal length
   reliably. A frame is taken only while the board stands still: set it, hold it,
   move on (the dots turn orange while it moves). Views that disagree with the rest
   are left out of the fit. Each session's frames and per-view errors are kept in
   `workspace/intrinsics/<camera>-<time>/`. The step locks the focus before the
   first view, because a lens that refocuses changes its focal length: it turns
   autofocus off and sets the focus from the camera's settings, or, when they hold
   none, steps the focus through its range and keeps the sharpest value. Lay the
   board flat in the middle of the image for that part (up to about 30 s). The
   chosen focus goes into the camera's settings and into the lens result; the twin
   sets it whenever it opens the camera ([Camera controls](#camera-controls)). A
   camera with a fixed focus skips the lock. Compute/save the result,
   then repeat for the next camera. A saved
   lens file can be loaded only at its matching resolution. Untrusted intrinsics
   block camera-pose calibration.
4. **Locate the fixed cameras.** Print the tagged card, measure the **black tag
   square** (excluding its white border), fold it and hold the free end in the
   gripper with about 70 mm protruding. In calibration step 2, enter the black
   square and how far the card sticks out beyond the jaw tips: the card's width and
   length follow from them, and the wave keeps that card off the table. Leave at least
   10 mm of white between the tag and the jaw tips (about 125 mm out for a 100 mm
   tag): in simulation the SO-101's fixed jaw covers the border of a tag that sits
   only 5 mm away. Start the wave and confirm the card is in place and the table
   clear. A second click within the confirmation window starts hardware motion.
   The wave explores first, then aims at the cameras it has found, as described in
   [Calibration wave](#calibration-wave). Review each camera's trust verdict and
   reason before saving. Changing tag size after the measurement prevents saving
   that result. Remove the card afterwards.
   The fit also estimates the servos' zero offsets for the joints the cameras can
   tell apart (on the SO-101: shoulder lift, elbow, wrist flex). It keeps them
   only when they cut the residual clearly and stay within 4 degrees. Saving the
   result turns them on: the twin's model then stands where the real arm stands,
   while sliders, limits and home stay in the servos' own units. They are saved
   with the camera poses, and **Wyzeruj poprawki zer serw** goes back to the
   servos' zeros.
5. **Check Sim-Real alignment.** For every fixed camera, compare rendered arm and
   table edges with the real image. A median mismatch above about 3 px is a cue
   to recheck tag size, lens calibration, table geometry and camera poses.
6. **Identify dynamics.** In **Dynamika**, record the excitation on the connected
   arm (about 20 s of motion), inspect fit quality and uncertainty, then save as
   workstation dynamics. STOP, Home or reconnect interrupts recording. A
   completed recording can continue fitting after motion stops. Stale feedback
   or a dead control loop invalidates the recording.
7. **Run a strategy.** In **Strategie**, choose an installed strategy and check
   its settings. **Cel z uchwytu 3D** supplies `Observation.goal` in the robot's
   base frame, including while the run is active; only the strategy determines
   what that goal means. Confirm hardware motion, then **Uruchom** (Run).
   **Zatrzymaj** stops the strategy; STOP preempts all motion owners.

### Dynamics and training export

The excitation identifies damping, armature and command delay. Servo gain (`kp`)
and dry friction remain from the model because this motion does not distinguish
them reliably. The result records that distinction and uncertainty; do not treat
all model parameters as measured. The panel blocks saving an unhelpful fit unless
explicitly overridden, and such a fit does not complete the guide's dynamics step.
Dynamics measured in simulation do not count as a hardware measurement.

```bash
uv run twin export workspace/export
```

The export contains `scene.xml`, `assets/` and `station.json`, including camera
calibration and dynamics. External training tools can read those files without
importing the twin. Training commands and training randomization belong to the
training package; use its documentation for those workflows. `twin eval-strategy`
evaluates any installed strategy, a trained policy included, in the twin's own
simulation (see [Strategy evaluation](#strategy-evaluation)).

## Galaxea A1X first session

The A1X integration is provided by `galaxeo`. Read its
[robot guide](https://github.com/machinekind/galaxeo-manipulators/blob/main/galaxeo/robot/README.md)
for the CAN adapter, platform extras, `a1x:` configuration and current device
procedures. **A1X has no brakes:** support it before removing power or releasing
motors. Backend behavior has been exercised on simulated interfaces; the physical
procedure remains to be validated.

To create a separate simulated A1X workstation after installing the package:

```bash
uv run twin demo --robot a1x --path workspace/a1x.json
uv run twin ui --workspace workspace/a1x.json
```

For hardware:

1. Start with the arm folded and supported, space below it clear, power cutoff
   and panel STOP accessible, and no competing CAN controller. Configure the
   adapter/port using the robot package's instructions.
2. Connect through **Ramię** without Home. Verify telemetry and pose before
   requesting motion. The backend rejects missing or frozen telemetry; a released
   arm may need supported power cycling. FF 1-5-6 does not restore trustworthy
   pose feedback after FF 2.
3. Check offsets and gripper feedback calibration from the robot package. A folded
   arm can sit outside the twin's configured joint limits; connection must not
   jump it into range. Enable manual controls and try one small joint motion.
   The backend starts commands from the measured pose and limits their speed.
4. If motor enabling is required, **Włącz silniki A1X (FF 1-5-6)** is available
   only for a connected A1X, with confirmation, manual controls disarmed and no
   task owning the arm. It holds the measured pose during the sequence. Consult
   the hardware guide before using it; a freshly powered arm may not need it.
5. Treat the first **Dom** as a substantial move: the model's default Home is
   raised, not folded. **Ustaw dom = tu** records the current pose with the home
   gripper set to 0 (closed). Close the empty jaws physically first. If gripper
   feedback has not become trustworthy, this action seeds an assumed closed
   state; it is not a measurement. If the gripper was already commanded in this
   connection, close it with the slider and record Home again.
6. Before power-off, fold and support the arm using the robot package's procedure.
   The panel's disconnect action does not release motors or fold the arm.
   Continue with cameras, calibration and dynamics only after these checks.

## Commands and development checks

| Command (prefix with `uv run`) | Purpose |
|---|---|
| `twin check` | Runtime dependencies, robot models and USB-serial ports; no rendering |
| `twin demo` | Example workstation; supports `--robot`, `--path` and explicit `--force` |
| `twin ui` | Browser panel; supports `--workspace`, `--port`, `--arm` and `--detector NAME` |
| `twin card --tag-mm 50` | Printable calibration card |
| `twin board --square-mm 28` | Printable ChArUco board; `--layout 7x5` for OpenCV's sample board, `--layout 3x5 --square-mm 45` for the `charuco` detector's board |
| `twin calib-sim --n 5 --cameras 2` | Calibration against simulated ground truth; optional `--robot` |
| `twin workspace` | Inspect workstation state; optional `--path` |
| `twin export workspace/export` | Export scene, assets and station data; optional `--path` |
| `twin episode [PATH]` | Report the latest recording, report a specified file or list a folder; see [episode recording](EPISODES.md) |
| `twin stand save/load/export/list/obsolete` | Stand folders; see [Stands](#stands) |
| `twin eval-strategy NAME --episodes 50 --seed 50000` | Run a strategy many times in simulation with a seeded object, record every run and let the scene judge success; see [strategy evaluation](#strategy-evaluation) |
| `twin eval-strategy NAME --seed S+i --episodes 1` | Replay episode i of a campaign with seed S |

```bash
uv run pytest -q
uv run pytest -q -m "not cli"
uv run pyright
```

`-m "not render"` excludes tests explicitly marked `render`, but some camera and
UI tests also create renderers without that marker. It is not currently a
complete way to run the suite without OpenGL. UI and stream tests also need
permission to bind localhost sockets. A1X-specific tests skip when `galaxeo`
is not installed, and the SO-101 contact tests skip when `soarm` is not
installed. `pytest-xdist` is not part of the default dependencies.

Control-loop tests use [VirtualClock](../src/machinekind_twin/clock.py) and
[the test bench](../tests/twinbench.py) to advance simulation deterministically.
Tests can connect with `threaded=False`, advance `Twin.step(dt)` and call
`StrategyRunner.step_once` directly. The application uses `WallClock`.
Episode recording is optional. See [episode recording](EPISODES.md) for configuration,
file format and latency and drift reports.

## Strategy evaluation

`twin eval-strategy` runs one installed strategy many times in simulation,
without the panel, and counts what the scene judges a success. Each episode
places an object on the table, drawn from the seed, and every run is recorded.

```bash
uv run twin eval-strategy swing --robot arm6 --set joint=base_yaw --episodes 2
uv run twin --config configs/local.yaml eval-strategy touch --robot so101 --episodes 50 --seed 50000
```

The second line runs the `touch` strategy from so-arm101, in that repository's
environment. The command needs nothing from the twin's clone: it finds the
strategy by its entry point, the config file through `--config` (or
`MACHINEKIND_TWIN_CONFIG`) and the arm through `--robot` or a workstation file
(`--path`, by default `workspace/twin.json`). `--robot` builds a workstation
with no file and no cameras.

### Settings

The strategy's settings are built as in the Strategie tab. The config section
named like the strategy comes first, then each `--set KEY=VALUE`, then the
default of every field neither gives. A value given with `--set` is converted
by its field's type, and a switch takes `true` or `false`. A broken section is
refused even when `--set` would override the broken key. A required setting
left blank counts as missing, as in the panel. Every episode creates a new
instance of the strategy.

### One episode

Each episode builds a fresh twin on a `VirtualClock`. The arm starts at the
workstation's home pose. The twin connects to `sim` without a thread, so the
`safety` section of the config applies and the episode gets a new
`SafetySupervisor`. The strategy runs through the same `StrategyRunner` as in
the panel. The twin's loop steps at 50 Hz and the strategy ticks at exactly its
own rate on the same simulated clock. An 8 s `swing` episode on arm6 takes
about 0.33 s of real time; 0.25 s of that is the wait for the recorder to close
its file.

- `--time-limit-s` (120 s of simulated time by default) caps each run. A
  strategy's own shorter limit stays.
- `--speed-deg-s` sets the panel's joint speed cap. The panel starts at
  20 deg/s, while the command has no cap by default.
- `--wall-limit-s` stops a run after that many real seconds, checked between
  ticks. It is off by default, because it makes results depend on the machine.

### Placement

The object is a box: `--size-mm` gives one edge for a cube or three edges,
30 mm by default, and `--mass-g` its mass, 30 g by default. It is red, so the
cube detector sees it. `--shape` takes only `box` for now. Episode i draws its
pose from `np.random.default_rng(seed + i)`: the radius from the base, the
bearing from the base's x axis and the yaw, in that order, each uniformly in
`--radius-m`, `--bearing-deg` and `--yaw-deg`. This is the draw of soarm_lab's
`sample_cubes`. The area defaults to the robot's `desk.cube_radius` and
`desk.cube_bearing` with any yaw. Episode i's pose does not depend on
`--episodes`, so `--seed S+i --episodes 1` replays it.

An area that puts the object off the table top is refused before the first
episode. Placements are never redrawn: an object placed against the robot fails
the rule.

### Object poses

By default the strategy finds the true pose in `Observation.objects[label]`,
with source `symulacja` (the simulation) and age 0. The label is `--label`,
else the strategy's own `label` setting, else `cube`. A `--label` that differs
from the strategy's own is refused.

`--detector NAME` gives the strategy the poses a detector finds through
`Perception`, as in the panel, with its settings from its config section. The
frames are renders of the workstation's trusted simulated cameras, taken on
the simulated clock. A workstation without such a camera is refused, and so is
a detector that refuses to start. Rendering needs OpenGL (`MUJOCO_GL=egl`
without a display). A detector call on two 640x480 cameras costs about 32 ms.

### Rules

The scene judges each run from the physics, never from the strategy's answer.
Both rules read the object's true pose and MuJoCo contact sensors between the
object and each of the robot's `jaw_bodies`. A robot without `jaw_bodies` is
refused. The scene looks at the object as placed, after every physics step,
right after the run ends and once more after 0.5 s with the arm holding.

| Rule | Success when |
|---|---|
| `touch` | A jaw touches the object at the end and after settling, and the object moved less than 1 cm on the table and tilted less than 20 deg |
| `lift` | Both jaws touch the object at the end and after settling, and its centre is at least 4 cm above where it was placed at both looks |

Both fail when the object touches the robot as placed. The thresholds are fixed
constants in [`scene_truth.py`](../src/machinekind_twin/scene_truth.py), so one
rule name always means one judge, and `summary.json` repeats them. The first
failing check gives the reason, for example `brak kontaktu na koncu` (no
contact at the end).

### The strategy's claim

The strategy's own claim is kept next to the scene's verdict and never replaces
it. The claim is true for end code `done`, false for `gave_up` and null for any
other end. An episode agrees when a true claim meets a success of the scene, or
anything else meets a failure. The totals count `both_success`,
`both_failure`, `false_claim` (the strategy claims success and the scene says
no) and `unclaimed_success`.

### Recordings

Every run is recorded with the [episode recorder](EPISODES.md), whatever the
panel's **Record episode** says. Recordings hold telemetry and camera metadata,
no images. The header's `meta` holds the panel's keys (`strategy`, `source`,
`settings`), `run`, `seed` and `evaluation`: the rule, the label, the pose
source, the object and the placement. Every step holds the object as the
strategy saw it.

Each episode waits up to 5 s for its recording to finish, then reads it back
with `EpisodeLog.summary`. A recorder error, a log without its end record or
with another end code, or lost telemetry makes the log incomplete. The row then
shows `[zapis: ...]` with the reason, and the totals list the episode. The
scene's verdict does not depend on the log. A recording still unfinished after
the wait stops the campaign after that episode, because the next ones would
wait on the same storage.

### Output

The campaign folder is `--out`, which must be new or empty. The default is
`episodes/eval-<strategy>-<time>/` next to the workstation file.

- `000.jsonl.zst`, `001.jsonl.zst`, ...: one recording per run.
- `summary.json`: `conditions` (the robot, workstation, config file, seed, rule
  and its constants, object, area, pose source, limits, and the strategy with
  its settings), `totals` and one entry per episode. An entry holds the
  placement, the pose as placed, the strategy's end code, reason, claim, steps
  and attempts, the scene's verdict with every measure, and `log`: the file,
  `complete`, `error`, `pending` and the recording's own collection accounting.
  The totals give the scene's successes with a 95% Wilson interval, the claims,
  the agreement, the end codes, complete and incomplete logs, and medians. The
  file exists before the first episode and is rewritten after each one.

`twin episode <folder>` lists the runs. The terminal shows one row per episode
and the totals, here for the first example above:

```text
  000 done          strategia tak scena nie x  +19.9 y   -7.7 cm  obrot -165 st  czas    8.04 s  krokow 162   prob -   zadanie wykonane  [scena: brak kontaktu na koncu]  NIEZGODNE
  001 done          strategia tak scena nie x  +15.1 y  +13.3 cm  obrot -128 st  czas    8.04 s  krokow 162   prob -   zadanie wykonane  [scena: brak kontaktu na koncu]  NIEZGODNE
Scena (touch): 0/2 (0%, 95%: 0-66%)
Strategia: sukces 2, porazka 0, bez zdania 0; niezgodne ze scena: 2 (000, 001)
Konce: done 2
```

The swing strategy never reaches for the object, so it claims success and the
scene says no: `NIEZGODNE` (disagree).

The command exits with 0 when the campaign ran, whatever the successes and
however complete the logs. It exits with 1 when a check refused the campaign,
before anything is written, or when an episode's arm or detector could not
start or its recording did not finish, which marks the summary `aborted`.
Ctrl-C ends the current run with code `stopped`, waits for its recording's end
record, marks the summary `interrupted` and exits with 130.

### Limits

- Identified dynamics (`Workspace.dynamics`) are not applied to the simulated
  scene; it runs the model's own.
- On the SO-101, `jaw_bodies[0]` is the whole gripper body, so a contact of the
  palm counts as a touch. On the test robot and arm6 only the jaw tips count.
- Real time does not pass on the virtual clock. A strategy that asks a model on
  its own thread is not measured faithfully.
- The wall limit stops a run only between ticks, and its results depend on the
  machine.
- The default area is out of the test robot's reach.
- Placements are not checked against the arm's path.
- `attempts` is filled only for a strategy whose `status` has an `attempt`.

## Calibration and perception notes

Fixed-camera calibration jointly estimates camera poses relative to the robot
base and the card's mount in the gripper. Each real camera needs trusted
intrinsics first. The card uses AprilTag 36h11; the twin's ChArUco board uses ArUco
5x5, and the 7 by 5 layout of OpenCV's sample board uses ArUco 6x6. So does the
3 by 5 board of the `charuco` detector ([Detector guide](DETECTORS.md#charuco-detector)).
Board geometry, corner detection and the intrinsics fit live in
`machinekind_twin_api.boards`, so a robot package can use them without the twin.
The solver rejects outlier observations and refits while retaining enough
observations per camera. See [the calibration tests](../tests/test_twin_calib.py).

### Trust by hand

Fixed gates decide whether a result is trusted, such as a residual under 0.6 px
for a lens and under 1.5 px for a wave. Some stands never pass them. For the
selected camera, the Kamery tab shows each result's residual and time, with
**Zaufaj obiektywowi** and **Zaufaj położeniu** and the buttons that undo them.
A trust by hand sets `trusted` in `intrinsics_info` or `calibration` and keeps
the measured numbers. `reason` records that a person decided, when, and the
reason that was there before. The workspace file and the stand keep it like any
other result.

A camera on the arm has no pose in the base frame, so it gets the lens button
only. A nominal lens can be trusted too. The twin then writes the nominal K into
the camera. The pose button waits for a trusted lens.

A trust by hand goes away when its basis changes: a new lens, new servo zero
offsets or a moved camera. A pose trust also goes when the lens trust is
withdrawn. The reason then says why. A focus that does not hold in this session
([Camera controls](#camera-controls)) still makes the lens and the pose
untrusted for the session.

### Calibration wave

The wave assumes nothing about where the cameras stand. It first explores the
arm's reachable workspace. The card goes through the base's whole turn range,
near and far, low and high. At each place the card faces a different direction
of the half-space above the table, from flat to straight up. The wave keeps a
coarse map of which camera saw the card, from where and facing which way. A
camera gets aimed poses only after at least three sightings from different poses
agree on its position by reprojection. A single or mirrored PnP no longer steers
the wave. Until a camera is located, the arm keeps exploring and repeats poses
near the ones where that camera saw the card. The wave drops a rough position
when later sightings contradict it. It also drops one with little support when
the poses aimed at it keep missing. The camera is then searched for again.

A frame counts only 0.8 s after the arm stops, and only once the tag stays
still across two new frames (within 0.5 px). A cheap arm still sways right after a
move, which gave 8 to 16 px residuals on the SO-101 stand. Each camera
collects at least 36 observations from 16 poses with 18 degrees of rotation
spread, and the fit needs 30 of them after dropping outliers. The wave stops
after 200 poses. In simulation an SO-101 wave with two or three cameras takes
40 to 140 poses, about 2 to 7 minutes on the arm's speed limits.

When the wave cannot solve, the panel says which camera saw the card how often,
facing which way and from where, or that it never saw it.
`workspace/ostatnia_fala.json` keeps the whole record: every pose with its phase
and place, the map (`mapa`), and the reason in `blad`.

The gripper holds the card with `wave.grip_strength` from the config file, 16 by
default ([Gripper strength](#gripper-strength)). The wave reads it at every start,
so an edit needs no restart. 0 keeps the stand's own grip. The wave closes the jaw
to 0 on the card. A goal just past the card would not hold it better. The squeeze
is P times how far the goal lies past the card, and the servo's overload
protection trips on that same squeeze, whatever mix of P and distance makes it.
Closing to 0 also keeps the squeeze steadier when the cardboard gives a little.

The `wave` section of `twin.json` tells the wave what to keep out of. Keep-out
boxes are in the arm's base frame in metres: x forward, y left, z up from the
table top. `base_turn` limits the base joint, in the panel's degrees. Both
default to none:

```json
"wave": {
  "keepout": [
    {"name": "drugie ramie", "x": [-0.12, 0.20], "y": [0.09, 0.30], "z": [0.0, 0.40]}
  ],
  "base_turn": [-90.0, 60.0]
}
```

The example box is a second SO-101 standing on the working arm's left. The wave
keeps its card, gripper and links 4 cm outside every box, at every pose and
along every move. Only the wave reads this section; teleoperation and strategies
do not. The calibration tab lists the active limits, and a malformed section
fails the workstation load. On the twin's SO-101 model a positive `shoulder_pan`
turns the arm toward -y, to its right.

Besides the boxes, the wave keeps 2 cm from every body in the twin's scene that
is not the arm itself: the table, objects, a second robot model or a fixture. The collision
check grows those bodies' primitive geoms by that clearance, so a body added to
the scene needs no box. Mesh-only geoms get a best-effort margin. The card and
the gripper also stay 4 cm above the table top.

After calibration, the camera watcher compares live frames with references,
masking the moving arm. It detects translation, rotation and scale changes and
requires consecutive agreeing checks before declaring a moved camera. Relocalize
that camera from the calibration tab. Old frames do not count as live input.

Some cameras stand on a mount that sways while the arm moves, such as a pole on
the arm's base plate. Tick **Chwiejny statyw** for such a camera in the Kamery
tab. The watcher then judges it only after the arm has stood still for 1.5 s.
It never shows the red "moved" alarm for it and never withdraws a trust by hand.
A shift above 1.5 times the camera's tolerance (default 4 px) in three agreeing
checks gives a calm notice instead. The notice gives the shift in pixels and in
millimetres on the table, and points to **Kamera poruszona** for a quick
re-measure. The camera's details show its largest shift over the last minute.
`twin.json` and `stand.json` keep the setting as `wobbly` and `wobble_px`.

Camera rendering uses the full intrinsic matrix, including the principal point.
The AprilTag detector corrects its half-pixel corner convention before solving
poses. Use segmentation geometry to check projection accuracy; colored blob
centres can be biased by lighting and shadows.

The tabletop map warps frames from trusted cameras onto the table plane. It
draws the objects that the running detectors find, as described in
[the detector guide](DETECTORS.md). For a detector that reports
silhouettes and the object's size, the twin fits a box of that size across
camera views, masks arm occlusions as unknown and rejects poor fits. Near the
gripper, a single view can confuse a held object with one lying on the table,
so the tracker applies extra checks. When grasp evidence is present, the
tracker follows the object with the TCP using its last relative pose.
Detections retain their capture timestamps, and stale or repeated frames do not
extend tracking lifetime.

## Detectors and objects

[The detector guide](DETECTORS.md) covers the detector plugin contract, how the
twin places and tracks objects, the example cube detector, the LLM detector and
the seeded detector.

## Cameras on the arm

A camera that rides on the arm, such as a wrist camera, belongs to the arm's
description. `robot.yaml` has an optional `cameras:` section. It maps each
camera name to the body the camera rides on and to what the camera sees. The
test robot declares this one:

```yaml
cameras:
  wrist:
    body: hand            # the MJCF body the camera rides on
    pose:                 # T_cam2body, 4x4, in OpenCV axes: z forward, y down, x right
      - [0.0, 0.0, 1.0, 0.0]
      - [-1.0, 0.0, 0.0, 0.0]
      - [0.0, -1.0, 0.0, 0.03]
      - [0.0, 0.0, 0.0, 1.0]
    fovy_deg: 60.0        # vertical field of view
    width: 160
    height: 120
```

The camera is an ideal pinhole. Its pixels are square and its principal point
is the image centre. So `f = (height / 2) / tan(fovy_deg / 2)`,
`cx = (width − 1) / 2` and `cy = (height − 1) / 2`, as in
`scene.CameraView.from_fov`. The description fails to load if a camera has an
unknown or missing key, a body that is not in the model, or a name the model
already gives a camera. It also fails if `pose` is not a rigid transform or a
size is not a positive integer.

The scene mounts each such camera on its body. Its name in the model is
`robot/<name>`. The scene renders it at its own size, after `mj_camlight`.
After a physics step the twin runs only `mj_kinematics`. Without
`mj_camlight` the camera would see the hand from before the last substep.
A strategy gets the camera's frame from `CameraHub.arm_frame_t(name)`, the
default frame source of `StrategyRunner`. The source depends on the arm:

- **Simulated arm** (nothing connected, or `sim`). The hub renders the frame
  from the scene at the moment of the read, like a simulated workspace camera.
- **Real arm** (also one still connecting, or one whose loop died). The scene
  only mirrors the arm. The frame comes from the workspace camera **of the same
  name**. That camera is the physical wrist camera, added like any other. The
  panel names a new camera `kameraN`. You rename it in `twin.json`. Without
  such a camera there is no frame.

The panel and calibration read workspace cameras through `CameraHub.frame_t`.
In both, the physical wrist camera always shows its own stream. The scene
leaves it out. The camera rides on the arm and is not fixed to the base. Its
`T_cam2base` therefore means nothing, and the scene ignores it. That holds
whether the pose came from a wave or from a saved preset of that name.
The fixed-camera calibration procedure does not estimate its body-relative
mount. That transform comes from `robot.yaml`; calibrating its intrinsics in
step 1 still makes sense. The calibration wave and the guide skip it: step 2
reads `gotowe` without its pose, and the guide neither lists it nor asks for
its lens.

A simulated workspace camera may not take the name of a camera on the arm,
because the scene renders both by name. Adding such a camera fails. Building
the scene from a workspace that holds one fails too. A real camera with that
name is fine. The Strategies tab gives a strategy the enabled workspace cameras
in `Observation.frames[name]`. With a simulated arm it adds the cameras on the
arm. It lists each name once. With a real arm, a camera on the arm has frames
only as the enabled workspace camera of its name. The Kamery tab and the 3D
view do not show cameras on the arm and do not calibrate them. Their pose comes
from the description.

## Other robots in the scene

A second arm next to the working one is part of the station. The twin does not
drive it, but it stands in the scene: in the 3D view, in simulated camera
frames, in every collision check and in the arm masks, so detectors never
report it as an object. List it under `other_robots` in `twin.json`, or in the
panel: **Ramie**, folder **Inne roboty (tylko odczyt)**. The working arm stays
`robot`, `backend` and `port`. A workspace without the key is the working arm
alone, as before. This one has a second SO-101 19 cm to the working arm's left,
read from its own bus:

```json
"other_robots": [
  {"name": "lewe", "robot": "so101", "xyz": [0.0, 0.19, 0.0], "yaw_deg": 0.0,
   "backend": "feetech", "port": "/dev/ttyACM1", "joints": {}}
]
```

- `name`: its prefix in the scene (`lewe/<body>`): letters, digits, `_`, `-`.
- `robot`: an installed robot description, as for the working arm.
- `xyz` [m] and `yaw_deg` [deg]: its base relative to the working arm's base;
  x forward, y left, z up.
- `backend` and `port`: a backend of that robot's package, opened read-only.
  `null` keeps the fixed pose in `joints`.
- `joints` [app units]: the fixed pose, or the last pose read. Joints left out
  take the robot's home pose.

A robot description without `tcp_site` and `hand_body`, such as a walking
robot or a cart, can only be listed here. The twin refuses it as the working
arm, in `twin.json` and in `--robot`. A floating top body (a `<freejoint>`)
stays where the model puts it relative to `xyz` and `yaw_deg`; gravity does
not move it.

Read-only is part of the backend contract. The twin opens the backend only when
its factory is marked `read_only_capable`. It sets `BackendContext.read_only`,
and the backend then writes nothing to the arm: no torque, no goals, no
settings. The twin holds such a backend as a `ReadOnlyArm`, which only
connects, reads and disconnects. A backend without the mark is refused, and the
robot keeps its fixed pose. When the bus is missing or a read fails, the robot
keeps its last pose, the panel says why, and the twin retries every few seconds.

The copy is welded in place, so physics never moves it, and contacts with it
are real. Its collision geoms reach 2 cm further out (`scene.PASSIVE_CLEARANCE`).
Every collision check therefore keeps 2 cm from it, the calibration wave's
included. The wave's own clearance does not add to that gap: the larger of the
two counts, and a robot listed here needs no keep-out box. The control loop
refuses a command that brings the arm within that distance and closer than it
already is. The arm then holds where it stands, its owner loses it, and the
panel shows the reason. Moves away are allowed. A block held in the jaws is not
in that check; it meets the other arm only in physics.

## Fixtures in the scene

A fixture is a static part of the stand, such as a camera mount. A robot package
brings it, and it stands in the scene like another robot. The calibration wave,
teleoperation, staged homing and the control loop's guard all keep the same 2 cm
from it. The guard refuses a command that brings the arm closer. The 3D view and
simulated camera frames show the fixture. List it under `fixtures` in
`twin.json` or `stand.json`; a workspace without the key has none. This one is the
SO-101 overhead camera mount from the `soarm` package, two pole parts on the
arm's left, with `kamera3` on it:

```json
"fixtures": [
  {"name": "slup", "fixture": "so101/overhead_cam_32", "placement": "left", "parts": ["bottom", "top"]}
]
```

- `name`: its prefix in the scene (`slup/<part>`): letters, digits, `_`, `-`,
  unique among robots and fixtures.
- `fixture`: `<robot>/<name>`, a fixture folder of an installed robot package.
- `on`: `robot` (the working arm, the default), the name of another robot, or
  `table`.
- `placement`: a named pose from the fixture's description, relative to the
  robot it stands on, for example `left` or `right`. Default: its first.
- `parts`: the parts from the bottom up. Default: the description's `stack`.
- `xyz` [m] and `yaw_deg` [deg]: an explicit pose that replaces `placement`, in
  the frame of what the fixture stands on. On the table, from the table top's
  centre; required there.
- `rgba`: the colour of all its parts.

A camera record names the fixture it sits on with `"fixture": "slup"`, or
`"slup/<site>"` when the fixture has several camera sites. Until the wave
measures the camera, the twin shows it at that site, and the wave starts from
that pose instead of searching for the camera. A measured pose always wins;
trust rules do not change. A simulated camera without `sim_pose` stands at the
site.

A robot package brings fixtures in its robot folder, one folder each:
`fixtures/<name>/fixture.yaml` and the meshes it names. Lengths are in metres.

```yaml
name: pole
title: Slup
parts:
  foot:
    mesh: foot.stl                 # optional; without it the colliders are drawn
    top: [0.0, 0.0, 0.01]          # where the next part sits
    colliders:
      plate: {pos: [0.0, 0.0, 0.005], size: [0.03, 0.03, 0.005]}   # box centre and half sizes
  head:
    colliders:
      head: {pos: [0.02, 0.0, 0.025], size: [0.04, 0.02, 0.025]}
    cameras:
      camera: [[0, -1, 0, 0.05], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]   # T_cam2part, OpenCV axes
stack: [foot, head]
placements:                        # in the base frame of the robot whose folder holds it
  left: {xyz: [0.0, 0.15, 0.0], yaw_deg: 0.0}
```

Each part's frame sits at the `top` of the part below it, so a stand can list
another stack of the same parts. Only the colliders collide: collision checks
grow boxes by their clearance, and a mesh cannot grow.

## Scene look

The `look` section in `twin.json` sets the scene's lights and the headlight.
The headlight is MuJoCo's light at the camera's eye. A workspace without the
section has the twin's default look, the lighting the twin always had:

```json
"look": {
  "lights": [
    {"pos": [0.4, -0.6, 1.6], "dir": [-0.4, 0.6, -1.6], "diffuse": 0.55, "directional": false, "castshadow": true},
    {"pos": [-0.5, 0.3, 1.4], "dir": [0.5, -0.3, -1.4], "diffuse": 0.35, "directional": false, "castshadow": false}
  ],
  "headlight": {"ambient": 0.35, "diffuse": 0.35, "specular": 0.5}
}
```

`pos` is relative to the centre of the table top. `dir` need not be a unit
vector. `diffuse` and the headlight's values are grey levels, the same on all
three channels. `directional: true` makes a directional light, like the sun.
Otherwise the light is a spot light. `pos` and `dir` are required. The other
keys default to `diffuse` 0.7, `directional` false and `castshadow` true. An
unknown key fails the load. The table's colour is in `table.rgba`.

Match the scene look to the conditions used by an external policy when comparing
results. For example, the following lighting can be combined with
`"rgba": [0.55, 0.45, 0.35, 1.0]` in the table settings.

```json
"look": {
  "lights": [
    {"pos": [0.0, 0.0, 1.8], "dir": [0.0, 0.0, -1.0], "directional": true},
    {"pos": [1.0, -1.0, 1.3], "dir": [-0.4, 0.4, -1.0], "diffuse": 0.5}
  ],
  "headlight": {"ambient": 0.3, "diffuse": 0.6, "specular": 0.0}
}
```

## Stands

A stand folder describes one physical stand and everything measured on it.
Stand folders live in their own repository, one folder per stand. The twin
knows only the format and the commands.

```text
<stand>/
  stand.json                        the setup
  calib/poses/<time>.json           camera poses and servo zero offsets (the pose preset format)
  calib/lens/<camera>/<time>.json   one camera's K and distortion, with the resolution they belong to
                                    and the focus they were measured at
  calib/dynamics/<time>.json        identified servo dynamics
  station/                          `twin export` of the stand, for training
```

Calibration entries are never overwritten. A new result becomes a new dated
file, and `calib` in `stand.json` names the current entry of each kind. An
entry identical to an existing one is not written again.

### `stand.json`

```json
{
  "format": 1,
  "name": "dgx-biurko",
  "description": "SO-101 on the DGX desk, two Logitech C920 cameras",
  "obsolete": false,
  "robots": [
    {"robot": "so101", "read_only": false, "backend": "feetech", "port": "/dev/ttyACM0",
     "home": {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 95.0}},
    {"name": "lewe", "robot": "so101", "read_only": true, "xyz": [0.0, 0.19, 0.0], "yaw_deg": 0.0,
     "backend": "feetech", "port": "/dev/ttyACM1"}
  ],
  "cameras": [
    {"name": "kamera1", "source": "/dev/v4l/by-id/usb-046d_C920_A1B2C3-video-index0",
     "width": 1280, "height": 720, "fps": 30, "enabled": true,
     "controls": {"autofocus": false, "focus_absolute": 30}}
  ],
  "table": {"size": [0.9, 0.6], "height": 0.75, "base_xy": [-0.35, 0.0], "base_yaw": 0.0,
            "rgba": [0.55, 0.5, 0.44, 1.0]},
  "card": {"tag_size": 0.0492, "out": 0.07, "width": 0.069, "length": 0.11, "thickness": 0.002, "ids": [0, 1]},
  "look": {"lights": [{"pos": [0.0, 0.0, 1.8], "dir": [0.0, 0.0, -1.0], "directional": true}],
           "headlight": {"ambient": 0.3, "diffuse": 0.6, "specular": 0.0}},
  "wave": {"keepout": [{"name": "statyw", "x": [0.25, 0.35], "y": [-0.3, -0.2], "z": [0.0, 0.5]}],
           "base_turn": [-90.0, 60.0]},
  "calib": {"poses": "20261028-140211", "lens": {"kamera1": "20261028-135000"},
            "dynamics": "20261029-101500"}
}
```

| Key | Meaning |
|---|---|
| `format` | Format version, currently 1. A newer version than the twin knows is refused |
| `name`, `description` | What the stand is; `name` is required |
| `obsolete` | `false`, or `{"reason": "...", "date": "YYYY-MM-DD"}` for a stand that no longer matches reality (both keys optional; `true` also works) |
| `robots` | Exactly one driven arm (`read_only: false`): `robot`, `backend`, `port`, optional `home`. It is the origin of the stand and has no name or pose. Other robots (`read_only: true`) have `name`, `robot`, `xyz` [m], `yaw_deg`, `backend`, `port` and, without a backend, a fixed pose in `joints`, as in [Other robots](#other-robots-in-the-scene) |
| `cameras` | `name`, `source` (prefer the stable `/dev/v4l/by-id/...` path), `width`, `height`, `fps`, `enabled`, optional `controls`, `sim_pose` for a simulated camera, and `fixture` for a camera on a fixture |
| `fixtures` | Static parts of the stand, such as a camera mount, as in [Fixtures](#fixtures-in-the-scene). Optional |
| `table`, `card`, `look` | As in `twin.json` |
| `wave` | The calibration wave's keep-out boxes and base turn range, as in [Calibration wave](#calibration-wave). Optional; without it the wave keeps only the arm's own limits |
| `calib` | The current entries: `poses`, `lens` per camera, `dynamics`. A missing entry means none |

Loading checks every key and the entries `calib` names. A broken file fails with
the file, the key and the reason, for example
`stand.json: cameras[0] (kamera1).width: oczekiwano liczby calkowitej >= 1, jest 0`.
A lens entry must have the camera's resolution. Unknown keys are errors.

### Commands

```bash
uv run twin stand save ~/stands/dgx-biurko --name dgx-biurko --description "..."
uv run twin stand load ~/stands/dgx-biurko
uv run twin ui --stand ~/stands/dgx-biurko
uv run twin stand export ~/stands/dgx-biurko
uv run twin stand list ~/stands
uv run twin stand obsolete ~/stands/dgx-biurko --reason "kamera2 przestawiona"
```

- `save` writes the workspace (`--path`, default `workspace/twin.json`) into
  the folder, creating it if needed. Current calibration becomes new entries
  only where it differs from existing ones. A workspace not yet bound to that
  stand brings its pose preset history along, except presets for none of its
  cameras; `--no-history` takes only the current poses. `station/` is exported
  too.
- `load` writes the stand into the workspace file. A workspace file with a
  different setup is first copied to `twin.json.<time>.bak`.
- `ui --stand` does `load`, then starts the panel.
- `export` exports `station/` again, for example after a twin update.
- `list` prints the stands in a folder. Obsolete stands come last, marked
  `NIEAKTUALNE`. Broken ones are listed with their error.
- `obsolete` marks a stand obsolete (today, unless `--date`). `--clear` makes
  it current again.

Loading an obsolete stand prints a warning, and the panel shows it in red above
the tabs.

After `save` or `load`, the workspace is bound to the stand (`"stand"` in
`twin.json`). Its pose presets are the stand's `calib/poses/`, and every save
writes the stand back: `stand.json`, new calibration entries and `station/`.
Saves from the panel do this too: a calibration result, a lens, dynamics, a
camera setting or a table change. A connection in `sim` does not replace the
stand's backend and port, so a rehearsal leaves the real arm's settings alone.
A live robot's last joint reading is not saved to the stand. A workspace
without `"stand"` works as before. To experiment without touching a stand,
work on a copy of its folder.

### Camera controls

A camera record can carry `controls`. The twin sets them whenever it opens the
camera, and again on **Zastosuj i zapisz** in the panel, then reads each one back.
A script that sets focus before the twin starts is no longer needed.

| Control | Value |
|---|---|
| `autofocus` | `true` or `false` |
| `focus_absolute` | Fixed focus in the camera's units, often 0-255; turns autofocus off |
| `auto_exposure` | `true` (aperture priority, the UVC auto mode) or `false` (manual) |
| `exposure_absolute` | Exposure time in 100 µs units; turns auto exposure off |

A control left out keeps the camera's own setting. How the twin reaches the
camera depends on the OS; none needs an extra package or root:

| OS | Source | How |
|---|---|---|
| Linux | a `/dev/...` path or a camera index | V4L2 ioctls on the device node |
| macOS | a camera index (OpenCV's AVFoundation order) | UVC requests to the camera's USB device through IOKit |
| Windows | a camera index | OpenCV's DirectShow properties; focus only, no exposure |

On macOS the index leads to the camera's USB port, so two identical cameras keep
their own settings. Built-in and network cameras have no UVC controls and are
left alone. A control the camera refuses, or one that reads back another value,
is logged, and the panel shows it under the camera's settings.

A lens result measured at a locked focus carries it (`intrinsics_info.focus`).
On open the twin sets that focus when the camera's `controls` set none. If the
camera then does not report autofocus off and that focus, the panel says so and
the lens and the camera's pose are not trusted for the session: step 2 waits and
the table map leaves the camera out until the focus holds again. A lens result
without a focus is used as before.

## Historical notes

Earlier research and removed prototypes are kept in [docs/archive](archive/README.md).
They describe past experiments and possible directions, not supported workflows.
