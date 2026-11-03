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
in the panel. A workspace loaded from a stand folder keeps its presets in that
folder instead ([Stands](#stands)).

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
   `workspace/intrinsics/<camera>-<time>/`. Turn autofocus off first: a lens that
   refocuses changes its focal length between frames. In **Kamery**, folder
   **Ustawienia kamery**, set autofocus off and a fixed focus; the twin sets them
   whenever it opens the camera ([Camera controls](#camera-controls)). Compute/save the result,
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
| `twin board --square-mm 28` | Printable ChArUco board; `--layout 7x5` for OpenCV's sample board |
| `twin calib-sim --n 5 --cameras 2` | Calibration against simulated ground truth; optional `--robot` |
| `twin workspace` | Inspect workstation state; optional `--path` |
| `twin export workspace/export` | Export scene, assets and station data; optional `--path` |
| `twin episode [PATH]` | Report the latest recording, report a specified file or list a folder; see [episode recording](EPISODES.md) |
| `twin stand save/load/export/list/obsolete` | Stand folders; see [Stands](#stands) |

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
Episode recording is optional. See [episode recording](EPISODES.md) for configuration,
file format and latency and drift reports.

## Calibration and perception notes

Fixed-camera calibration jointly estimates camera poses relative to the robot
base and the card's mount in the gripper. Each real camera needs trusted
intrinsics first. The card uses AprilTag 36h11; the twin's ChArUco board uses ArUco
5x5, and the 7 by 5 layout of OpenCV's sample board uses ArUco 6x6.
The solver rejects outlier observations and refits while retaining enough
observations per camera. See [the calibration tests](../tests/test_twin_calib.py).

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
is not the arm itself: the table, objects, or a second robot model. The collision
check grows those bodies' primitive geoms by that clearance, so a body added to
the scene needs no box. Mesh-only geoms get a best-effort margin. The card and
the gripper also stay 4 cm above the table top.

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
silhouettes and the object's size, the twin fits a box of that size across
camera views, masks arm occlusions as unknown and rejects poor fits. Near the
gripper, a single view can confuse a held object with one lying on the table,
so the tracker applies extra checks. When grasp evidence is present, the
tracker follows the object with the TCP using its last relative pose.
Detections retain their capture timestamps, and stale or repeated frames do not
extend tracking lifetime.

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
| silhouette (`sylwetka`) | Every camera gave a mask of a full-size frame, and `size_m` is given | A box of `size_m` resting on the table, fitted to all masks at once. It gives the position, the yaw (`rotation[:, 0]` along `size_m[0]`) and the fit's IoU as confidence |
| triangulation (`triangulacja`) | Two or more cameras whose rays pass within the detectors' pixel error, scaled by depth | The point nearest to all rays. It must lie in front of every camera and between 3 cm below and 60 cm above the table |
| one ray (`promien`) | One camera, or rays that disagree | The most confident detection's ray meets the plane at `height_m`. Without `height_m` the plane is at half of `size_m`, and without both it is the table top |

The geometry uses the calibrated camera poses with lens distortion, also in
simulation. Simulated perception therefore tests the calibration. A placed
object's `t` is the capture time of the oldest frame used to place it.

Only the silhouette path gives yaw, so a detector that knows an object's size
should send it with a mask, whatever the object's shape. The fit takes
`size_m` as the box's x, y and z sides, z being its height above the table. The
yaw is folded by the footprint's symmetry: into [-45, 45) degrees for a square
footprint such as a cube, into [-90, 90) otherwise. A 50x25x25 mm block lies
within 0.5 mm and 1 degree of its true pose on synthetic masks from two cameras
at about 45 degrees, and within 1.3 mm and 1 degree in the simulated stand.

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
  `kamery`. A strategy reads `age_s` to decide whether a pose is too old.
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
  rotation is the identity. Only the silhouette fit gives yaw; it needs a mask
  and `size_m` from every camera that saw the label.
- The box is fitted resting on the table in the pose `size_m` describes:
  `size_m[2]` is its height above the table. A cuboid stood on its end, or a
  cylinder on its end, with the lying size listed is gated out (IoU below
  0.75) and the label shows `ostatnie widziane`. List the size as the block
  rests.
- A cylinder is fitted as its bounding box. Lying on its side, it fits worse
  than a box (IoU 0.74-0.82 at the true pose from 45-degree cameras) and may
  fail the 0.75 gate at some yaws.
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
default 10 s, the detector costs up to 360 questions an hour.
Four things bound it:

- one question in flight at a time;
- `interval_s`, at least 2 s;
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

## Seeded detector

[`machinekind-twin-seeded`](../packages/machinekind_twin_seeded), installed by
`uv sync`, registers the detector `seeded`. It follows blocks by colour at the
twin's call rate after the LLM detector has said which blob is which block. A
vision model behind the `claude` or `codex` CLI names the blocks every
`seed_interval_s`, 10 s by default. Between its answers the detector grows
each block from the model's pixel by colour and follows the blob. For every
block it sees, it returns the block's mask and size, so the silhouette fit
places the block with its yaw five times a second.

The panel never starts it by itself. Start it in the **Detektory** tab or with
`twin ui --detector seeded`, after filling a `seeded:` section in the config
(`configs/default.yaml` has a commented example). The package README has the
details.

### Threads

| Thread | Rate | Does |
|---|---|---|
| `detektor:seeded`, the runner | `interval_s`, 0.2 s | Takes the model's latest answer, hands the worker a copy of the frames when a question is due, grows every track in the current frames |
| `detektor-seed:<cli>`, the worker | at most every `seed_interval_s` | Asks the LLM detector about the copied frames, one question at a time |

A label is `<colour> <shape>` as the model names a block (`red cube`,
`wooden cuboid`): the shape word gives its size, the colour word its colour.

A track is one label in one camera. A fresh seed takes the blob of the
label's colour whose box lies within the seed's pixel error. A blob longer
than `max_blob_ratio` times the model's box is not one block and is not
taken: two touching blocks of one colour, or a block the colour of the table. From then on the track follows the blob nearest its last centroid,
within `track_error_px`. A followed track outranks a new seed, which only
confirms it. A track the model leaves out of three answers in a row is
dropped.

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `cli` | `codex` | `codex` or `claude`, as for the LLM detector |
| `model` | `gpt-6-astra` | the model as the CLI names it; `""` = the CLI's default |
| `effort` | `low` | reasoning effort passed to the CLI |
| `labels` | `""` | the blocks to follow, comma-separated; `""` = every block the model names |
| `labels_sizes` | `""` | size overrides by label or shape word, `name=LxWxH` [m] |
| `labels_hsv` | `""` | colour overrides by label or colour word, OpenCV HSV ranges |
| `cameras` | `""` | cameras to track and to send, comma-separated; `""` = every camera offered |
| `prompt` | `""` | extra instructions appended to the LLM's prompt |
| `interval_s` | 0.2 | shortest time between two colour-tracking calls [s], at least 0.1 |
| `seed_interval_s` | 10 | shortest time between the starts of two LLM questions [s], at least 2 |
| `seed_error_px` | 25 | how far off the model's pixel may be, in the sent image's pixels [px] |
| `track_error_px` | 40 | how far a followed blob may move between two calls, in frame pixels [px], 1 to 500 |
| `lost_after_s` | 3 | a track without a blob for this long is dropped [s] |
| `min_area_px` | 30 | smallest patch of the colour that counts as a blob [px] |
| `max_blob_ratio` | 2 | a blob longer than this times the model's box is not one block; at least 1 |
| `timeout_s` | 120 | the CLI's process group is killed after this [s] |
| `max_side_px` | 1280 | long edge of the images sent [px] |
| `max_calls_per_hour` | 0 | cap on questions per hour; 0 = no cap |
| `min_confidence` | 0.3 | seeds the model is less sure of are dropped |
| `executable` | `""` | the CLI binary; `""` = `cli` on PATH |
| `claude_images` | `stream` | claude only: `stream` or `read` |

The settings the LLM detector also has go to it unchanged. `seed_interval_s`
becomes its `interval_s`, `seed_error_px` its `pixel_error_px`, and `labels`
its `objects`. A line naming the colours and shapes goes before `prompt`.

### The blocks

| Shape | Size `LxWxH` [m] | Rests |
|---|---|---|
| `cube` | `0.029x0.029x0.029` | on a face |
| `cuboid` | `0.060x0.029x0.014` | flat |
| `cylinder` | `0.060x0.029x0.029` | on its side, as its bounding box |

Colours, OpenCV HSV (H 0..180, S and V 0..255): `red`, `yellow`, `green`,
`light blue` (or `blue`) and `wooden` (or `wood`, `natural`, `unpainted`); the
package README lists the ranges. `labels_sizes` (`name=LxWxH`) and
`labels_hsv` (`name=H0-H1,S0-S1,V0-V1`, `|` between ranges) override them,
`;` between entries; a key is a whole label or a shape or colour word.

Bare wood is about the colour of a wooden table. On such a table a wooden
block's blob grows into the table, is longer than `max_blob_ratio` times the
model's box, and is not followed: the label gets no pose rather than a wrong
one. On a table of another colour bare wood is followed like paint.

A block stood on its end does not fit its lying box and its label shows
`ostatnie widziane` until it is laid down. Blocks the model names alike share
one label.

### What it returns

For every followed blob, per camera, a `Detection` with the blob's `mask`, its
`box`, its centroid as `pixel`, the label's `size_m`, confidence 1 and
`pixel_error_px` 6. The twin fits a box of `size_m` to the masks of every
camera that sees the label, which gives the position, the yaw and the fit's
IoU as confidence.

`pixel` is the colour blob's centroid: with `size_m` the twin reads it as the
centre at half the object's height (`height_m`'s rule), which the silhouette
centroid approximates; the silhouette path never reads it. The model's own
pixel, the footprint's centre, only anchors a fresh seed.

### Freshness

`fresh_s` is 0.5 s at `interval_s` 0.2. A reader sees the pose aged by the
frame's age at the call (the panel's frame cache refreshes every 0.2 s), the
time since the call (up to `interval_s`) and the fit (about 10 to 40 ms per
label). That comes to about 0.55 s with four labels in the panel, so the
source may read `ostatnie widziane` for the tail of a cycle. A strategy treats
`ostatnie widziane` under about 1 s as the current pose. Headless perception
reads its frames live, so its poses stay under about 0.25 s old.

### Lost blobs and failed questions

A label whose blob is not found is left out of that call: the twin shows the
last pose as `ostatnie widziane` and drops it after `hold_s`, 6 s. The
detector drops a track without a blob for `lost_after_s` and waits for the
next seed. For a seed that never found its blob, the time counts from the
answer's arrival.

A failed question (a timeout, a non-zero exit, a subscription limit) is
logged, and the next one goes out after `seed_interval_s`. While any block is
followed, the detector keeps returning it and raises nothing. With nothing to
follow, the call after a failed answer raises `seeded: <the error>`. The
panel shows it and calls again after a wait that doubles with each failure,
which also paces the questions.

`stop` returns at once. It cancels a running question and wakes the worker,
which ends by itself: within 0.2 s when the CLI obeys SIGTERM, and after the
CLI's 2 s grace and SIGKILL otherwise. Only the runner's own join waits.

Do not run `llm` and `seeded` on the same labels at once. Two detectors that
report one label compete for it, and the newer capture time wins: every LLM
answer would replace the followed pose with a point without yaw.

### Privacy

The frames leave the machine every `seed_interval_s`, as for the LLM detector:
JPEGs of at most `max_side_px` go to OpenAI (codex) or Anthropic (claude)
under the terms of the subscription the CLI is logged in with. The colour
tracking runs on the machine.

### Calibrating the colours on the stand

The default ranges cover the simulation and plain paint. Painted blocks under
the stand's lamp may need their own ranges, or every blob is missed and no
label appears. Take a
still from each camera, find a block's pixel (u, v) and read the HSV values
around it:

```python
import cv2
hsv = cv2.cvtColor(cv2.imread("still.png"), cv2.COLOR_BGR2HSV)
print(hsv[v - 3:v + 4, u - 3:u + 4].reshape(-1, 3))
```

Take H about 8 either side of what you see, and S and V from about two thirds
of the lowest values up to 255. Check the shaded side of the block too. Red
lies near both H = 0 and H = 180 and needs one range on each side.

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
| `cameras` | `name`, `source` (prefer the stable `/dev/v4l/by-id/...` path), `width`, `height`, `fps`, `enabled`, optional `controls`, and `sim_pose` for a simulated camera |
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

A camera record can carry `controls`. The twin sets them on a V4L2 camera
whenever it opens one, and again on **Zastosuj i zapisz** in the panel. A
script that sets focus before the twin starts is no longer needed.

| Control | Value |
|---|---|
| `autofocus` | `true` or `false` |
| `focus_absolute` | Fixed focus in the camera's units, often 0-255; turns autofocus off |
| `auto_exposure` | `true` (aperture priority, the UVC auto mode) or `false` (manual) |
| `exposure_absolute` | Exposure time in 100 µs units; turns auto exposure off |

A control left out keeps the camera's own setting. The twin uses the
`VIDIOC_S_CTRL` ioctl on the device node, so it needs Linux and a `/dev/...`
source or a camera index. Elsewhere, for example on macOS, nothing is set. A
control the camera refuses is logged, and the panel shows it under the camera's
settings.

## Historical notes

Earlier research and removed prototypes are kept in [docs/archive](archive/README.md).
They describe past experiments and possible directions, not supported workflows.
