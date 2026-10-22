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
`cube:`, etc.). The namespace is flat, so a strategy and a detector of one name
share a section; the panel warns about it. Backends validate their settings;
strategy and detector sections prefill the panel's settings form and are
validated before a start. The panel selects the connection backend
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
| **Mapa** (Map) | Stitched tabletop view, the objects tracked from running detectors and a tag-on-table reach action |
| **Dynamika** (Dynamics) | Record supervised excitation, fit servo dynamics and save the result to the workstation |
| **Detektory** (Detectors) | Choose an installed detector, edit settings, start it, stop each running one, see the tracked objects in text and 3D |
| **Strategie** (Strategies) | Choose an installed strategy, edit settings, provide an optional 3D goal, start/stop the run; the strategy gets the tracked objects in `Observation.objects` |
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
does not search for objects itself; it draws what the running detectors find
(see [Detectors and objects](#detectors-and-objects)). For a detector that
reports cube silhouettes, the twin fits the cube across camera views, masks arm
occlusions as unknown and rejects poor fits. Near the gripper, a single view can
confuse a held object with one lying on the table; the tracker applies additional
checks. When grasp evidence is present, the tracker follows the object with the
TCP using its last relative pose. Detections retain their capture timestamps, and
stale or repeated frames do not extend tracking lifetime.

## Detectors and objects

A detector is the fourth plugin kind, next to robots, backends and strategies.
It is a class from an installed package, registered under the
`machinekind_twin.detectors` entry point group. It looks at camera frames and
says what it sees where **in the image**. The twin does the geometry and the
tracking, and strategies read the result as `Observation.objects[label]`: an
`ObjectPose` in the robot's base frame, with `source` and `age_s`. The contract
lives in [`machinekind_twin_api.detector`](../packages/machinekind_twin_api);
its README has a minimal detector.

### The contract

The twin calls `detect(request)` on a thread of its own, one call at a time,
starting a call no sooner than `interval_s` after the previous one started.
Nothing waits for a detector: a slow one only makes the poses older.

`request.frames` maps camera names to frames. It offers the cameras the twin can
place detections from now: enabled, with a calibrated pose that is trusted (or
simulated), and fixed to the base. The set is read again for every call, so a
camera calibrated during a run is offered from the next call. Frames are
fetched only when read. Iterating and `len` list the offered cameras;
`name in frames` and `frames.get(name)` fetch the frame and tell whether the
camera has one younger than a second; `frames[name]` raises `KeyError` when it
has none. Images are raw RGB, never undistorted.

Each `Detection` carries a label, a camera and a pixel (OpenCV convention, in
the frame's own resolution), and optionally a box, a confidence, a bool
silhouette `mask`, the height above the table of the point the pixel marks
(`height_m`), the object's size (`size_m`) and the pixel error of that one
detection. Labels are compared stripped and lower-cased. The twin drops and
counts malformed detections (unknown camera, a pixel outside the frame, a mask
of the wrong shape); the Detektory tab shows the count.

`start(info)` runs once before the first call and may raise to refuse.
`stop(reason)` runs once after a successful start, possibly while `detect` runs
on the detector thread; the twin discards a late answer. A `detect` that raises
is shown in the panel and retried after a wait that doubles with each
consecutive failure, up to 300 s.

### Where an object is

Per label, the twin keeps the surest detection from each camera and places the
object one of three ways. The Detektory tab names the path in Polish:

| Path | When | Result |
|---|---|---|
| silhouette (`sylwetka`) | every camera gave a mask of a full-size frame and `size_m` is a cube | the cube silhouette fit across all masks: position, yaw and the fit's IoU as confidence |
| triangulation (`triangulacja`) | two or more cameras, rays that pass within the detectors' pixel error scaled by depth | the point nearest to all rays, in front of every camera and between 3 cm below and 60 cm above the table |
| one ray (`promien`) | one camera, or rays that disagree | the surest detection's ray meeting the plane at `height_m`, else half of `size_m`, else the table top |

The geometry uses the calibrated camera poses with lens distortion, never the
simulator's ground-truth poses, so simulated perception tests the calibration.
A detection's `t` is the capture time of the oldest frame it used.

### The example cube detector

`examples/detectors/cube` (`machinekind-twin-cube`, installed by `uv sync`)
replaces the Map tab's old built-in cube search. It thresholds one colour in
HSV with the same ranges and opening the old `CubeDetector.mask` used, and
answers per camera with the largest blob's centroid, box and mask and the
cube's size. The twin then runs the old multi-view silhouette fit on those
masks. The only difference from the old pipeline is the start of the fit: the
mask is computed on the raw frame and then warped to the table, instead of
thresholding the warped colour map. In simulation both land within 2 mm and
7 degrees of the true pose. Settings: `color` (red, green, blue or yellow),
`size_m`, `label` (default `cube`, the key soarm-lab's policy reads),
`min_area_px` and `interval_s` (default 0.33 s, the old Map tab rate).

### Tracking, `source` and `age_s`

Each label gets the twin's cube tracker (`perception.ObjectTracker`), which also
follows an object held in the gripper. A measurement is fed to it once, and
only when its confidence is above 0.4. A track with no measurement for
`hold_s` is dropped: 6 s, or three times the detector's interval plus its
latency when that is longer, so a slow detector's object survives between
answers.

`age_s` is the time since the frames the pose was measured from were captured,
and 0 while the object follows the jaws. `source` says what the pose rests on:

- `kamery`: the cameras saw it recently. It holds while the detector's latest
  answer contains the label and `age_s <= fresh_s`, where `fresh_s` is twice
  the detector's interval, kept within 0.5 to 1.0 s. The cube detector at
  0.33 s keeps `kamery` within 0.66 s of capture.
- `ostatnie widziane`: the last known pose, older than `fresh_s`, or the label
  is missing from the latest answer. A measurement that is already older than
  `fresh_s` when it arrives is published like this at once. A detector that
  answers in tens of seconds, such as a language model, therefore never
  publishes `kamery`, and a strategy never mistakes its pose for a live one.
  Read `age_s` to decide how old is too old.
- `w dloni`: the gripper holds it and the pose follows the TCP. The closed jaw
  is the hardware's limit on a connected arm, so an empty closed gripper is
  not taken for a grasp.
- Annotated variants such as `ostatnie widziane (upuszczona)` come from the
  tracker unchanged.

The tracker's thresholds on the jaw (its `still` test, 0.02 rad) are per
update, not per second. They therefore depend on who steps the tracks and how
often, as they depended on the policy's rate before.

### Threads and who steps the tracks

| Thread | Rate | Does |
|---|---|---|
| `detektor:<name>`, one per running detector | `interval_s`, longer after failures | reads frames, calls `detect`, places the answer, hands it to the tracks |
| `detektor-start:<name>`, short-lived | once per start | creates the detector, calls its `start`, starts its thread |
| strategy runner | the strategy's `hz` | reads `Observation.objects`, which steps the tracks |
| panel loop | 5 Hz | steps the tracks when nobody did for 0.15 s, then draws them |

There is no tracker thread. Whoever reads the objects steps the tracks, at
most once per 10 ms. With a strategy running, the tracks move at its `hz`;
without one, the panel steps them at 5 Hz, so the Detektory tab, the 3D
markers and `w dloni` keep working while the operator drives the arm from the
panel. Reading objects never waits for a detector.

### Running detectors

Start a detector in the **Detektory** tab. Its form is prefilled from the
config section of its name. Starting a running detector again restarts it with
the new settings. Several detectors may run at once; when two of them report
the same label, the measurement with the newer capture time wins. STOP does not
stop detectors, because they never move the arm; closing the panel does.

`twin ui --detector cube` starts the named detector at panel start with its
config section's settings, as **Uruchom detektor** would. The flag repeats for
several detectors. A detector that fails to start is logged and shown in its
row; the panel runs on.

Loader messages and panel labels are Polish, like the rest of the panel. A
plugin's own exception texts are English, as the plugin wrote them, and reach
the panel verbatim after a Polish label such as `blad:`. This mix is
intentional.

### Limits

- Only cameras fixed to the base are used. A camera on the arm is never
  offered to a detector.
- A point detection (triangulation or one ray) has no orientation; its
  rotation is the identity. Only the cube silhouette fit gives yaw.
- One object per label. Two red pens need two labels.

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
