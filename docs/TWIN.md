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
| `workspace/twin.json` | Your local robot selection, backend/port, table, scene look, cameras, calibration and measured dynamics; ignored by Git |
| `workspace/calib/*.json` | Saved camera-pose presets; example presets are tracked |
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

The twin reads `safety`, `robot` and optional `joints` overrides. Robot descriptions
supply the joints, limits, gripper range and home pose. Other YAML sections belong
to the backend, strategy or detector with that name (`feetech:`, `a1x:`, `swing:`,
`cube:`, etc.). A strategy and a detector with the same name share a section, and
the panel warns about it. Backends validate their settings. Strategy and detector
sections prefill the panel's settings form and are validated before a start.
The panel selects the connection backend
and port; `--arm` reconnects using the workstation's saved selection at startup.
With one USB-serial adapter plugged in, its current port takes precedence over
a saved port. With multiple adapters, choose the correct one explicitly.

Saving camera positions creates a new preset in `workspace/calib/`. The panel
loads the selected preset, or the newest when none is selected. Trust is
reassessed against the current lens calibration. Older presets remain available
in the panel.

## Browser panel

The UI currently uses Polish labels. Keep the browser tab visible: viser renders
through the browser's animation loop, which can pause in a background tab.

| Tab | Purpose |
|---|---|
| **Ramię** (Arm) | Backend/port, **Połącz** (Connect), **UZBRÓJ** (Arm manual controls), joint sliders, 3D IK handle, **Dom** (Home), **Ustaw dom = tu** (Set home here) and **Rozłącz** (Disconnect) |
| **Kamery** (Cameras) | Add USB or simulated cameras, preview streams, enable/remove cameras, move simulated cameras in 3D and inspect their status |
| **Kalibracja** (Calibration) | Step 1: lens intrinsics; step 2: camera poses from a tagged card in the gripper; save/discard results and relocalize moved cameras |
| **Mapa** (Map) | Stitched tabletop view, the objects that running detectors track and a tag-on-table reach action |
| **Dynamika** (Dynamics) | Record supervised excitation, fit servo dynamics and save the result to the workstation |
| **Detektory** (Detectors) | Choose an installed detector, edit settings, start it and stop running ones. Tracked objects show as text and 3D markers |
| **Strategie** (Strategies) | Choose an installed strategy, edit settings, provide an optional 3D goal, start/stop the run. The strategy gets the tracked objects in `Observation.objects` |
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
   print at 100%, measure the square size and enter it. Show the board across the
   image, at different distances and tilts. The session requires coverage and
   rotation diversity; a parallel board alone cannot determine focal length
   reliably. Compute/save the result, then repeat for the next camera. A saved
   lens file can be loaded only at its matching resolution. Untrusted intrinsics
   block camera-pose calibration.
4. **Locate the fixed cameras.** Print the tagged card, measure the **black tag
   square** (excluding its white border), fold it and hold the free end in the
   gripper with about 70 mm protruding. In calibration step 2, start the wave and
   confirm the card is in place and the table clear. A second click within the
   confirmation window starts hardware motion. The arm searches both sides of
   the table, then targets cameras needing more observations. Review each
   camera's trust verdict and reason before saving. Changing tag size after the
   measurement prevents saving that result. Remove the card afterwards.
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
importing the twin. Training commands, randomization and policy evaluation belong
to the training package; use its documentation for those workflows.

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
| `twin board --square-mm 28` | Printable ChArUco board |
| `twin calib-sim --n 5 --cameras 2` | Calibration against simulated ground truth; optional `--robot` |
| `twin workspace` | Inspect workstation state; optional `--path` |
| `twin export workspace/export` | Export scene, assets and station data; optional `--path` |

```bash
uv run pytest -q
uv run pytest -q -m "not cli"
uv run pyright
```

`-m "not render"` excludes tests explicitly marked `render`, but some camera and
UI tests also create renderers without that marker. It is not currently a
complete way to run the suite without OpenGL. UI and stream tests also need
permission to bind localhost sockets. A1X-specific tests skip when `galaxeo`
is not installed. `pytest-xdist` is not part of the default dependencies.

Control-loop tests use [VirtualClock](../src/machinekind_twin/clock.py) and
[the test bench](../tests/twinbench.py) to advance simulation deterministically.
Tests can connect with `threaded=False`, advance `Twin.step(dt)` and call
`StrategyRunner.step_once` directly. The application uses `WallClock`.

## Calibration and perception notes

Fixed-camera calibration jointly estimates camera poses relative to the robot
base and the card's mount in the gripper. Each real camera needs trusted
intrinsics first. The card uses AprilTag 36h11; the ChArUco board uses ArUco 5x5.
The solver rejects outlier observations and refits while retaining enough
observations per camera. See [the calibration tests](../tests/test_twin_calib.py).

After calibration, the camera watcher compares live frames with references,
masking the moving arm. It detects translation, rotation and scale changes and
requires consecutive agreeing checks before declaring a moved camera. Relocalize
that camera from the calibration tab. Old frames do not count as live input.

Camera rendering uses the full intrinsic matrix, including the principal point.
The AprilTag detector corrects its half-pixel corner convention before solving
poses. Use segmentation geometry to check projection accuracy; colored blob
centres can be biased by lighting and shadows.

The tabletop map warps frames from trusted cameras onto the table plane. It
draws the objects that the running detectors find, as described in
[Detectors and objects](#detectors-and-objects). For a detector that reports
cube silhouettes, the twin fits the cube across camera views, masks arm
occlusions as unknown and rejects poor fits. Near the gripper, a single view can
confuse a held object with one lying on the table, so the tracker applies extra
checks. When grasp evidence is present, the tracker follows the object with the
TCP using its last relative pose. Detections retain their capture timestamps, and
stale or repeated frames do not extend tracking lifetime.

## Detectors and objects

A detector is the fourth plugin kind, next to robots, backends and strategies.
It is a class in an installed package, registered in the
`machinekind_twin.detectors` entry point group. It reads camera frames and
reports what it sees **in the image** and where. The twin turns these image
positions into poses and tracks them. Strategies read the result as
`Observation.objects[label]`, an `ObjectPose` in the robot's base frame with
`source` and `age_s`. The contract lives in
[`machinekind_twin_api.detector`](../packages/machinekind_twin_api), and that
package's README has a minimal detector.

### The contract

The twin calls `detect(request)` on a dedicated thread, one call at a time. A
call starts at least `interval_s` after the previous one started. Nothing waits
for a detector, so a slow detector only makes its poses older.

`request.frames` maps camera names to frames. It offers the cameras that the
twin can place detections from at that moment. Such a camera is enabled and
fixed to the base, and its calibrated pose is trusted or the camera is
simulated. The twin builds the set again for every call, so a camera calibrated
during a run is offered from the next call on. A frame is fetched only when
read. Iterating and `len` list the offered cameras. `name in frames` and
`frames.get(name)` fetch the frame and tell whether the camera has one younger
than a second. `frames[name]` raises `KeyError` when it has none. Images are
raw RGB and never undistorted.

`request.arm_masks` maps camera names to arm masks; it was added in API 0.1.3.
A mask is a bool image of the frame's shape, True where the robot arm hides
the scene. The twin makes a mask only when it is read, for the capture time of
that camera's frame in the same request, so reading it reads the frame.
`arm_masks.get(name)` is None when the twin has no mask for that camera.
Iterating and `len` list the offered cameras, like `frames`, and
`arm_masks[name]` raises `KeyError` for a camera without a mask. Read masks one
camera at a time with `get`: `items()` or `dict(arm_masks)` fail when one
camera has none, and render every camera's mask. A request built without masks
holds an empty mapping, so a detector that ignores the arm needs no change.

Each `Detection` carries a label, a camera and a pixel. The pixel follows the
OpenCV convention in the frame's own resolution. A detection may also carry a
box, a confidence, a boolean silhouette `mask`, `height_m`, `size_m` and its own
pixel error. `height_m` is the height above the table of the point the pixel
marks, and `size_m` is the object's size. Labels are compared after stripping
whitespace and lower-casing. The twin drops a detection that names an unknown
camera, has a pixel outside the frame or has a mask of the wrong shape. The
Detectors tab shows how many it dropped.

`start(info)` runs once before the first call and may raise to refuse the
start. `stop(reason)` runs once after a successful start. It may run while
`detect` is still running on the detector thread, and the twin discards the
late answer. When `detect` raises, the panel shows the error and the twin
retries after a wait. The wait doubles with each consecutive failure, up to
300 s.

### Where an object is

For each label, the twin keeps the most confident detection from each camera.
It places the object in one of three ways, and the Detectors tab shows the
Polish name of the way it used:

| Path | When | Result |
|---|---|---|
| silhouette (`sylwetka`) | Every camera gave a mask of a full-size frame, and `size_m` is a cube | The cube silhouette fit across all masks. It gives the position, the yaw and the fit's IoU as confidence |
| triangulation (`triangulacja`) | Two or more cameras whose rays pass within the detectors' pixel error, scaled by depth | The point nearest to all rays. It must lie in front of every camera and between 3 cm below and 60 cm above the table |
| one ray (`promien`) | One camera, or rays that disagree | The most confident detection's ray meets the plane at `height_m`. Without `height_m` the plane is at half of `size_m`, and without both it is the table top |

The geometry uses the calibrated camera poses with lens distortion, also in
simulation. Simulated perception therefore tests the calibration. A placed
object's `t` is the capture time of the oldest frame used to place it.

### The example cube detector

The example detector in `examples/detectors/cube` replaces the Map tab's old
built-in cube search. Its package is `machinekind-twin-cube`, and `uv sync`
installs it. It thresholds one colour in HSV with the ranges and the opening
that the old `CubeDetector.mask` used. For each camera it reports the largest
blob's centroid, box and mask, and the cube's size. The twin runs the old
multi-view silhouette fit on those masks. The one change is the fit's input.
The masks come from the raw frames and are then warped to the table, where the
old pipeline thresholded the warped colour map. In simulation both pipelines
place the cube within 2 mm and 7 degrees of its true pose.

Its settings are `color` (red, green, blue or yellow), `size_m`, `label`,
`min_area_px` and `interval_s`. `label` defaults to `cube`, the key that
soarm-lab's policy reads. `interval_s` defaults to 0.33 s, the old Map tab's
rate.

### Tracking, `source` and `age_s`

Each label gets its own copy of the twin's cube tracker,
`perception.ObjectTracker`. The tracker also follows an object held in the
gripper. Each measurement is fed to it once, and only when its confidence is
above 0.4. A track is dropped after `hold_s` without a measurement. `hold_s` is
the larger of 6 s and three times the sum of the detector's interval and
latency, so a slow detector's object survives between answers.

`age_s` is the time since the capture of the frames the pose comes from. It is
0 while the object follows the jaws. `source` says what the pose is based on,
and it takes these Polish values:

- `kamery` (cameras) means the cameras saw the object recently. The value holds
  while the detector's latest answer contains the label and
  `age_s <= fresh_s`. `fresh_s` is twice the detector's interval, kept between
  0.5 and 1.0 s. The cube detector at 0.33 s keeps `kamery` for 0.66 s after
  capture.
- `ostatnie widziane` (last seen) means the pose is the last known one. The
  source switches to it when `age_s` exceeds `fresh_s` or when the detector's
  latest answer lacks the label. A measurement that is already older than
  `fresh_s` when it arrives gets this value at once. A detector that answers in
  tens of seconds, such as a language model, therefore never publishes
  `kamery` from a fresh answer. The LLM detector's repeat of an answer for
  frames that did not change is the one exception (see "LLM detector",
  "Freshness"). A strategy reads `age_s` to decide whether a pose is too old.
- `w dloni` (in hand) means the gripper holds the object and the pose follows
  the TCP. On a connected arm, the closed jaw position comes from the
  hardware's limit, so an empty closed gripper does not count as a grasp.
- The tracker can annotate a value, as in `ostatnie widziane (upuszczona)`
  (last seen, dropped). Annotated values pass through unchanged.

The tracker's jaw thresholds, such as the 0.02 rad `still` test, apply per
update. Their effect therefore depends on how often the tracks are stepped.

### Threads and who steps the tracks

| Thread | Rate | Does |
|---|---|---|
| `detektor:<name>`, one per running detector | `interval_s`, longer after failures | Reads frames, calls `detect`, places the answer and feeds it to the tracks |
| `detektor-start:<name>`, short-lived | Once per start | Creates the detector, calls its `start` and starts its thread |
| strategy runner | The strategy's `hz` | Reads `Observation.objects`, which steps the tracks |
| panel loop | 5 Hz | Steps the tracks when nothing else did for 0.15 s, then draws them |

The tracks have no thread of their own. Each read of the objects steps them, at
most once per 10 ms. A running strategy steps them at its `hz`. Without one,
the panel steps them at 5 Hz. The Detectors tab, the 3D markers and `w dloni`
therefore keep working while the operator moves the arm from the panel. Reading
the objects never waits for a detector.

### Running detectors

Start a detector in the **Detektory** (Detectors) tab. The config section with
the detector's name prefills its form. Starting a running detector again
restarts it with the new settings. Several detectors can run at once. When two
of them report the same label, the measurement with the newer capture time
wins. Detectors never move the arm, so STOP leaves them running. Closing the
panel stops them.

Starting shows `Startuje...` immediately. During each detection call, the panel
shows `Wywolanie w toku` with elapsed seconds beside Start and in the detector's
row. Previous results stay visible while the next answer is pending.

`twin ui --detector cube` starts the named detector when the panel starts. It
uses the settings from the detector's config section, like the **Uruchom
detektor** (Start detector) button. Repeat the flag to start several
detectors. When a detector fails to start, the panel logs the error, shows it
in the detector's row and keeps running.

Loader messages and panel labels are in Polish, like the rest of the panel. The
panel shows a plugin's exception text verbatim after a Polish label such as
`blad:` (error:).

### Limits

- Detectors only get cameras fixed to the base. A camera on the arm is never
  offered.
- An object placed by triangulation or by one ray has no orientation, so its
  rotation is the identity. Only the cube silhouette fit gives yaw.
- Each label holds one object. Two red pens need two labels.

## LLM detector

[`machinekind-twin-llm`](../packages/machinekind_twin_llm), installed by
`uv sync`, registers the detector `llm`. Each call sends the cameras' frames to
a vision model and reads back what is where in each frame. The model sits
behind a CLI installed on the machine, `codex` (Codex CLI) or `claude` (Claude
Code), and answers on the subscription that CLI is logged in with. The package
holds no API key, uses no model SDK and opens no network connection itself:
every question is one headless run of the CLI as a subprocess. Its README has
the exact command lines and a smoke check to run after a CLI update.

The panel never starts it by itself. Start it in the **Detektory** tab or with
`twin ui --detector llm`, after filling an `llm:` section in the config
(`configs/default.yaml` has a commented example).

### Subscription and login check

`start` finds the CLI (`executable`, else `cli` on PATH) and asks it how it is
logged in (`claude auth status`, `codex login status`), with a 15 s timeout. It
refuses a CLI that is not logged in and one that uses an API key. Any other
login passes and is named in the log: a claude.ai login, a `claude setup-token`
token (`CLAUDE_CODE_OAUTH_TOKEN`), a ChatGPT login for codex. The refusal shows
in the detector's row and the detector does not start. The check runs on the
start thread, so the panel never waits for it.

Every question runs the CLI with API keys, base-URL overrides and the variables
of a calling Claude Code or cmux session removed from its environment, so it
cannot fall back to a key or route through a proxy. When `cli` on PATH is a
terminal wrapper (a cmux shim), point `executable` at the real binary, e.g.
`~/.local/bin/codex`.

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `cli` | `codex` | `codex` or `claude` |
| `model` | `gpt-6-astra` | the model as the CLI names it (`sonnet`, `claude-opus-5-5`, ...); `""` = the CLI's default. A `gpt-` model needs codex, a Claude model needs claude |
| `effort` | `low` | `low`, `medium`, `high`, `xhigh` or `max`; `""` = the CLI's default |
| `objects` | `""` | labels to look for, comma-separated (`red pen, cube`); `""` = every movable object, labelled by the model in short lowercase English |
| `cameras` | `""` | cameras to send, comma-separated; `""` = every camera offered in each call |
| `prompt` | `""` | extra instructions appended to the built-in prompt |
| `interval_s` | 10 | shortest time between the starts of two questions [s], at least 2 |
| `timeout_s` | 120 | the CLI's process group is killed after this [s], 5 to 900 |
| `max_side_px` | 1280 | long edge of the images sent [px]; answers are scaled back to each frame |
| `pixel_error_px` | 25 | how far off the model's pixels may be, in the sent image's pixels |
| `reuse_if_still` | true | repeat the last answer without asking while no camera's image changed since the frames it was given, or for up to two calls while the scene still moves; the arm moving over the table does not count, see "Cost and rate" |
| `reuse_max_age_s` | 60 | ask again anyway once the last answer is this old [s], 0 to 3600 |
| `max_calls_per_hour` | 0 | cap on questions per hour; 0 = no cap |
| `min_confidence` | 0.3 | answers the model is less sure of are dropped |
| `executable` | `""` | the CLI binary; `""` = `cli` on PATH |
| `claude_images` | `stream` | claude only: `stream` sends images inline, `read` has the model open temp files with its Read tool (slower) |

A value the detector cannot work with refuses it when it is created, with a
message naming the setting. With `objects` set, an answer is kept only when its
label is one of them or ends with one ("the red pen" counts as `red pen`), so
label drift does not create stray tracks.

### Cost and rate

Every question counts against the CLI subscription's usage limits. At the
default 10 s, a scene that keeps changing costs up to 360 questions an hour and
a still one 60. Five things bound it:

- one question in flight at a time;
- `interval_s`, at least 2 s;
- `reuse_if_still`, on by default: a call whose frames did not change since the
  last question repeats that question's answer instead of asking. Each camera's
  frame is compared as a grey thumbnail 256 px long, blurred 3x3, with the mean
  brightness matched so auto exposure does not count. A pixel counts as changed
  when it moved by 16 grey levels plus a fifth of its distance from the mean, so
  a gain change does not count. Two changed pixels belong to one blob when
  neither their rows nor their columns lie more than 5 px apart, about 2.3 cm at
  1 m. The image counts as changed when its largest blob exceeds 24 thumbnail
  pixels. Changed pixels more than 5 px from any other stay blobs of one. A 3 cm
  cube 80 grey levels darker than the table, at 1 m through a 60 degree lens,
  makes a blob of at least 39 when it moves by a centimetre and at least 49 when
  it leaves. A blinking 2x2 pixel LED makes at most 14. Sensor noise of ±20
  levels, a +12 exposure step and a 1.15 gain make none. A cube needs 39 to 64
  grey levels of contrast to the table, depending on the frame width and on
  where its edges fall on the thumbnail's pixels, before every centimetre it
  moves along an axis counts. With less it can move a centimetre unseen. Frames
  that changed since the last question and also since the previous call show a
  scene that still moves. Such a call repeats the answer too, for at most two
  calls in a row, and the detector asks once two calls agree. An answer starts
  the count again. A failed question does not, so the backoff below still
  applies while the scene moves. So does a call whose frames match the last
  question's. Pixels the twin's arm masks (`DetectRequest.arm_masks`) hide are
  left out of the comparison. Each mask is widened by 5 thumbnail pixels, about
  2.3 cm at 1 m, over the arm's edge and shadow. The arm entering, hovering or
  moving over a table the last question saw therefore asks nothing. A place the
  arm hid when the question was asked counts as changed once it shows, whatever
  lies there: the answer was given without seeing it. Once the last answer is
  `reuse_max_age_s` old, the next call asks whatever the frames show;
- `max_calls_per_hour`: a call over the cap fails with "hourly cap of N
  questions reached";
- backoff: a failed question (a timeout, a non-zero exit, a subscription limit,
  an answer that is not the JSON asked for) is retried after a wait that
  doubles with each consecutive failure, up to 300 s.

A subscription limit shows in the row as
`blad: CliError: subscription limit: ...`, with the time of the next try.

### Latency and accuracy

In the smoke run (two 640x480 images, effort `low`) one question took 4 s with
claude and 9 s with codex. More cameras, larger images and higher effort take
longer. Strategies never wait for an answer; the poses only get older.

The answer has a pixel and a box per camera, but no mask, size or height. The
twin therefore places an object by triangulation when two cameras see it and
on the table top otherwise, with no orientation. Each detection's pixel error
is `pixel_error_px` scaled from the sent image to its frame, which is what lets
two cameras' rays pair. Expect pixel errors of tens of pixels: at 0.5 m with a
focal length of 600 to 900 px that is 1 to 3 cm. The positions are good for
"which object and roughly where", not for grasping a pen.

### Freshness: `ostatnie widziane`

An answer arrives seconds after its frames were captured, more than the 1 s
`fresh_s` allows. The LLM's objects therefore publish `ostatnie widziane`, not
`kamery`, with `age_s` counting from the capture. A track survives between
answers, because `hold_s` is three times the interval plus the latency: 57 s
at a 10 s interval and a 9 s answer. soarm-lab's policy, which waits for
`kamery`, never takes such a pose for a live one. A strategy that wants the
LLM's objects reads `age_s` and decides how old is too old. `w dloni` still
works: the tracker follows an object the gripper closes on.

A repeated answer (`reuse_if_still`) is the exception. It returns at once, and
the twin stamps it with the capture time of the frames just read, so the object
publishes `kamery` for `fresh_s` after each repeat. That is what the frames
say: they match the ones the model answered on to within the check's limits
above, so the object stands where the model saw it. The pose is as good as the
model's answer was, no better; `reuse_max_age_s` bounds how long a wrong answer
can be repeated. A repeat while the scene still moves is stamped the same way
although its frames differ, for at most two calls. Turn `reuse_if_still` off
when a strategy must only ever act on a pose the model has just confirmed.

### Privacy

The frames leave the machine. They go to OpenAI (codex) or Anthropic (claude)
under the terms of the subscription the CLI is logged in with, as JPEGs of
at most `max_side_px`. Each question's files live in a temporary directory
that is removed afterwards. With `MACHINEKIND_TWIN_LLM_KEEP=<dir>` set, a copy
of the images, the prompt and the CLI's output goes to `<dir>` first.

The tests never ask a model. One real question runs only on request:

```bash
MACHINEKIND_TWIN_LLM=codex uv run pytest -q packages/machinekind_twin_llm/tests/test_llm_real.py
```

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
step 1 still makes sense.

A simulated workspace camera may not take the name of a camera on the arm,
because the scene renders both by name. Adding such a camera fails. Building
the scene from a workspace that holds one fails too. A real camera with that
name is fine. The Strategies tab gives a strategy the enabled workspace cameras
in `Observation.frames[name]`. With a simulated arm it adds the cameras on the
arm. It lists each name once. With a real arm, a camera on the arm has frames
only as the enabled workspace camera of its name. The Kamery tab and the 3D
view do not show cameras on the arm and do not calibrate them. Their pose comes
from the description.

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

## Historical notes

Earlier research and removed prototypes are kept in [docs/archive](archive/README.md).
They describe past experiments and possible directions, not supported workflows.
