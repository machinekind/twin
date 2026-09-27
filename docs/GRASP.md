# Grasp student

The grasp student is a published policy that does the last centimetres of a
top-down grasp on the Galaxea A1X from its wrist camera. The twin runs it as
one more owner of the arm: every target goes through `Twin.set_target`, the
safety supervisor and the A1X backend. This page covers the commands
`lerobot-twin grasp`, `grasp-bench`, `grasp-eval` and `grasp-model`, the
per-rig file `workspace/grasp.json` and the code in
`src/lerobot_mp/twin/grasp/`. The runbook for the bench is in sections 2, 3,
[4](#4-bench-measurements) and [8](#8-the-shadow-episodes-and-the-first-real-grasp),
in the order you do it.

Status: tested in the twin's simulation, on a fake CAN bus and against
fixtures recorded from the training environment. **Nothing has moved the real
arm yet.**

## 1. What it is

### The policy

- Published as `hvsr-robotics/a1x-wrist-grasp-sim` (file `student.pt`,
  45.7 MB). It was trained in the MuJoCo simulation of galaxeo-manipulators,
  directory `sim/grasp`, by imitation of a teacher that sees the true object
  pose. On 100 held-out simulated episodes it grasps 79 % of a hard mix and
  94 % of plain objects.
- A planner brings the tool to a hover 6 cm above a proposed top-down grasp.
  From there the policy moves the tool until the object is lifted. A failed
  close is not the end: it opens, aims again and retries.
- Every tick, 20 times a second, it reads:
  - the wrist image, 160 x 120 RGB;
  - a fourth image channel: a Gaussian blob at the pixel where the believed
    grasp point projects through the wrist camera;
  - 19 numbers: the six joints normalised by the model's joint ranges, the
    fingertip gap, the believed grasp relative to the TCP, sin and cos of
    twice the yaw error, the gripper target, the previous action and the
    number of close attempts divided by 3.
- It returns five numbers in [-1, 1]: a TCP step in x, y and z (8 mm per
  unit), a yaw step (5 degrees per unit) and a gripper target (-1 closed,
  +1 open).

### Where each input comes from on the real arm

| input | in training | in the twin |
|---|---|---|
| wrist image | a render at the calibrated camera pose, jittered by 3 mm and 0.7 degrees | the raw 1920 x 1080 frame, undistorted with the wrist camera's K, rotated by 180 degrees, resized to 160 x 120 |
| believed pixel | projection through the camera pose and the gripper's forward kinematics | the same projection, at the joint sample nearest the frame's exposure time |
| six joints | measured | the newest fresh joint sample of the twin loop |
| fingertip gap | the finger slide joints | **modelled**: the opening the backend streams to the jaw, limited to the measured jaw speed. The A1X reports no jaw position |
| believed grasp | the planner's proposal plus perception noise | a point taught by touch on the bench (in simulation: the cube in the scene plus training's noise) |
| yaw error, gripper target, previous action, attempts | the environment's bookkeeping | the same bookkeeping, copied from training line by line |

### The code

| module | what it holds |
|---|---|
| `grasp/contract.py` | the constants of training: camera pose and lens, steps, box, floors, IK tolerances, checkpoint hash |
| `grasp/rig.py` | the per-rig file `workspace/grasp.json` and the refusals that follow from it |
| `grasp/command.py` | the command bookkeeping of training's `env.step` |
| `grasp/keys.py` | the one stdin reader: once the arm can move, a bare Enter is always STOP |
| `grasp/hover.py` | the hover above the believed grasp, and training's envelope |
| `grasp/gripper.py` | the gap estimate |
| `grasp/observer.py` | the 19-number observation |
| `grasp/net.py`, `grasp/policy.py` | the network rebuilt without torchvision, the checkpoint loader, the download, the timing |
| `grasp/camera.py` | the wrist camera: the policy frame, the believed pixel, the live and simulated frame sources, the dry-run dump |
| `grasp/motion.py` | the arm session: claim, clutch, joint history, impact guard, pre-positioning, holding |
| `grasp/driver.py` | the 20 Hz loop and the episode endings |
| `grasp/run.py`, `grasp/evaluate.py`, `grasp/bench.py` | `lerobot-twin grasp`, `grasp-eval` and `grasp-bench` |

Changes to existing code, each small:

- `twin/kinematics.py`: `RobotKinematics.ik_step`, one warm IK step that
  reports position and rotation error. It is training's solver transcribed and
  takes about 0.05 ms.
- `robot/base.py`, `robot/a1x.py`: a read-only `gripper_state()` (commanded,
  streamed opening, effort, freeze, feedback age, freeze threshold), and a
  count of command frames from another sender on the CAN link.
- `twin/runtime.py`: `Twin.set_target` refuses non-finite values,
  `Twin.speed_limits()` reports the effective speed limits, and
  `Twin.hold_measured(gripper=...)` holds with a given jaw value.
- `twin/scene.py`: a camera mounted on the hand in the twin's scene
  (`HandCamera`, `Scene.render_hand`).

### Data flow

```
wrist camera --> WristCamera: undistort, rotate 180, resize --> image 160x120 ----+
                                                                                  |
CAN 0x052 --> Twin loop --> joint history --> believed pixel --> heat channel ----+--> policy (CPU)
                                 |                                                |       |
                                 +--> GraspObserver: 19 numbers ------------------+       |
                                          ^                                               |
                  gap estimate -----------+                                               | 5 actions
                                                                                          |
GraspCommand: step, box, floors <---------------------------------------------------------+
    --> ik_step --> accept: converged, jump, collision
    --> Twin.set_target (owner "chwyt") --> supervisor: limits, 90 deg/s
    --> A1X backend --> CAN 0x050 (arm), 0x051 (gripper)
```

**Base frame.** Origin on the joint 1 axis, table top at z = 0, x forward, y
to the arm's left, z up, metres. Yaw is the angle of the jaw closing axis
about +z. Joints towards the twin are degrees; the gripper is 0 (closed) to
100 (open).

**Training's region.** Every training tick had the object on the arm's left:
x from -0.20 to 0.20 m and y from 0.22 to 0.40 m. The twin refuses a hover
outside training's envelope: joint 1 from 40 to 139 degrees, radius 0.22 to
0.44 m from the base axis, hover z 0.05 to 0.13 m.

## 2. Before anything moves

Read these rules before the first bench session.

- **One CAN sender and one camera process.** Close the panel, the galaxeo jog
  tool (`jog_a1x.py`), the bridge (`so101_bridge.py`), any ROS driver and any
  other twin process before a grasp command connects. If one is left running:
  - on macOS the XCAN adapter admits one process, so the grasp command cannot
    connect;
  - on Linux the grasp command listens for 0.3 s before it moves anything, and
    a command frame (0x050, 0x051, FF) that it did not send makes it refuse
    with `busy`;
  - such a frame during a move makes the twin STOP with the fault
    `inny nadawca na laczu CAN (ramki rozkazu spoza blizniaka)`.
- **Enable the arm from the panel first**, if it needs enabling at all.
  `a1x.enable_on_connect` must be false: every grasp command refuses with
  `enable on connect` otherwise, because the enable sequence FF 1-5-6 releases
  the motors for a moment. A freshly powered arm listens without it. When the
  arm does not follow the panel's sliders, use **Wlacz silniki A1X (FF 1-5-6)**
  in the panel ([TWIN.md](TWIN.md#galaxea-a1x), first session, step 1), then
  **Rozlacz** and close the panel. The arm keeps holding its pose.
- **Launch from Terminal.app on macOS.** Camera capture fails
  (`not authorized to capture video`) in processes started by an agent or by
  another app's built-in terminal.
- **The cube goes on the arm's left**, about 0.33 m out, at a bearing of about
  90 degrees, for example at x = 0, y = 0.33 m. A cube in front of the arm is
  outside training's region and the run refuses it.
- **The workspace table must lie under the grasp point.** The table in
  `workspace/twin.json` must contain the believed grasp with 8 cm to spare on
  every side, or the run refuses with `table not under grasp`. The collision
  check uses this table. The default A1X table ends at y = 0.30 m and the
  example one (`lerobot-twin demo --robot a1x`) at y = 0.40 m, so neither
  covers a cube at y = 0.33 m: the table must reach at least y = 0.41 m. Set the real table in
  the panel (tab **Ramie**, folder **Stanowisko (stol)**) or in the `table`
  entry of `workspace/twin.json` (`size`, `base_xy` of the arm base from the
  table centre, `base_yaw`).
- **The workspace is an A1X one.** A workspace file with another robot is
  refused (`not a1x`). Without a file the commands use an A1X workspace in
  memory and never save it.
- **Keep the wrist camera out of the workspace's camera list.** The grasp
  commands open it themselves from `wrist.source` in `workspace/grasp.json`.
  Calibration and the scene treat every camera in `workspace/twin.json` as
  fixed to the table.
- **Use the same YAML config for every grasp command**
  (`lerobot-twin --config configs/local.yaml grasp ...`). The run compares the
  backend's joint offsets and gripper freeze threshold with what the bench
  recorded.
- **The A1X has no brakes and no compliance.** The arm does not yield on
  contact. Keep a hand on the keyboard: Enter alone is STOP.

## 3. Get the model

```bash
lerobot-twin grasp-model fetch       # download student.pt, verify sha256, compare the camera spec
lerobot-twin grasp-model check       # find, hash, load, print the schema and the statistics
lerobot-twin grasp-model time        # inference time on this CPU, 200 calls (--n N)
```

- `fetch` downloads `student.pt` over HTTPS to
  `workspace/policies/grasp/student.pt` (or `--checkpoint PATH`), verifies its
  sha256 and renames it into place. An existing file with another hash is not
  overwritten. It then downloads `wrist_camera_spec.json` next to it and
  compares the camera pose and the vertical field of view with the twin's
  contract. A difference prints a loud warning and exits with 1: do not run the
  model until the contract has been checked.
- `check` prints where the file is, its size and sha256, and whether it is the
  published one. It loads the weights without running code from the file
  (`weights_only`), checks the 130 weight entries strictly and prints the
  observation statistics.
- `time` prints the median and 95th percentile of one inference. The limits
  are 30 ms and 45 ms. It exits with 1 when either is exceeded.

Where a command looks for the checkpoint, first match wins: `--checkpoint`,
then `$LEROBOT_GRASP_STUDENT`, then `workspace/policies/grasp/student.pt`.
Every load verifies the sha256 first. No run mode ever downloads anything.

**A different checkpoint** needs `--checkpoint PATH` and the environment
variable `LEROBOT_GRASP_SHA256=<its sha256>`. Without the variable a file with
another hash is refused. The load logs a warning naming both hashes.

## 4. Bench measurements

Every value the real arm cannot supply is measured on the bench and written
to `workspace/grasp.json`. Do the steps in this order. Each step says what you
should see when it works and what it writes.

### The per-rig file

`workspace/grasp.json` sits next to `workspace/twin.json` and is not in git.
`--rig PATH` points to another file. The `grasp-bench` tools write it; the
operator writes a few keys with an editor. A missing file means nothing is
measured yet. An unknown top-level key or another `version` is an error.

```json
{
  "version": 1,
  "table":   {"touch_tcp_z": null, "tcp": null, "joints": null, "taught_at": null},
  "grasp":   {"tcp": null, "yaw": null, "joints": null, "taught_at": null,
              "object_height": null, "joint_offsets_deg": null},
  "gripper": {"stroke_close_s_effort": null, "stroke_close_s_operator": null,
              "stroke_open_s_effort": null, "stroke_open_s_operator": null,
              "free_effort_peak": null, "loaded_effort_peak": null, "separable": null,
              "freeze_threshold": null, "gap_mm": {}},
  "wrist":   {"source": null, "K": null, "dist": null, "upside_down": true, "dot_error_px": null},
  "latency": {"camera_s": null, "actuation_s": null, "infer_p95_s": null, "total_s": null},
  "guard":   {"effort_abs": 12.0},
  "overrides": {}
}
```

| key | written by | a real backend refuses without it in |
|---|---|---|
| `table.touch_tcp_z`, `table.tcp`, `table.taught_at` (and `table.joints`) | `grasp-bench table` | dry, shadow, real |
| `grasp.tcp`, `grasp.yaw`, `grasp.taught_at`, `grasp.joint_offsets_deg` (and `grasp.joints`) | `grasp-bench point` | dry, shadow, real |
| `grasp.object_height` | `grasp-bench point --object-height` (default 0.030 m) | shadow, real |
| `wrist.source` | you | dry, shadow, real |
| `wrist.K`, `wrist.dist` | you, both together from one calibration | never: the default from 2026-09-26 is used and the dot check judges it |
| `wrist.dot_error_px` | `grasp-bench dot --error-px` | shadow, real |
| `latency.camera_s`, `latency.actuation_s`, `latency.infer_p95_s`, `latency.total_s` | `grasp-bench latency` | `total_s`: shadow, real |
| `gripper.stroke_close_s_*` | `grasp-bench gripper` | real |
| `gripper.stroke_open_s_*`, `gripper.free_effort_peak`, `gripper.loaded_effort_peak`, `gripper.separable`, `gripper.suggested_threshold` | `grasp-bench gripper` | never: they inform the freeze decision |
| `gripper.gap_mm` | `grasp-bench gripper --gap-mm` | never |
| `gripper.freeze_threshold` | `grasp-bench gripper --set-freeze` | real |
| `guard.effort_abs` | you (12.0 is a guess) | never |
| `overrides.shadow_done` | you, after ten clean shadow episodes | real |
| `overrides.table`, `.dot`, `.latency`, `.taught`, `.joint_offsets` | you: a note saying who and why | each lifts the matching limit |

Each tool re-reads the file before it writes and changes only its own
section, so edits you made in between survive.

`lerobot-twin grasp-bench show` prints the file, how old the table and the
point teaching are, and, for dry, shadow and real mode on a real backend,
what is still missing (`nic - moze ruszac` when nothing is). Run it between
the steps. Four refusals need the running arm and the workspace, so `show`
cannot judge them: `table not under grasp`, `joint offsets changed`,
`joint offsets differ` and `gripper freeze differs`.

### The bench tools

```bash
lerobot-twin grasp-bench {gripper,table,point,latency,dot,show} [--backend B] [--port P] [--i-know]
                   [--rig PATH] [--workspace PATH] [--checkpoint PATH] [--xy X Y] [--yaw-deg D] [--start-z M]
                   [--gap-mm O:MM ...] [--set-freeze V] [--error-px V] [--force] [--seconds S]
                   [--object-height M] [--camera SOURCE]
```

| flag | what it does |
|---|---|
| `--backend B`, `--port P` | the arm. Default `sim`: a real arm needs `--backend a1x` and the port (`can0` on Linux, `xcan` on macOS; default: the port saved in the workspace) |
| `--i-know` | required by every tool that moves a real arm |
| `--rig PATH`, `--workspace PATH` | the per-rig file and the workspace file |
| `--checkpoint PATH` | the model, for the inference time inside the latency measurement |
| `--xy X Y`, `--yaw-deg D`, `--start-z M` | where teaching starts: a top-down pose over (X, Y) at TCP height M (default 0.12 m), closing axis at D degrees (default 0). Without `--xy`: the x and y of the other taught point (the table for `point`, the grasp point for `table`), else (0, 0.33). `latency` on the simulated arm uses `--xy` and `--yaw-deg` for its point |
| `--gap-mm O:MM` | a ruler reading: the tip gap in mm at opening O. Several points after one flag, or the flag repeated |
| `--set-freeze V` | records the freeze threshold you chose |
| `--error-px V` | records the dot error you measured |
| `--force` | saves a table touch more than 15 mm from the expected height, and writes `overrides.table` |
| `--object-height M` | the object's height for `point`, 0.005 to 0.10 m (default 0.030 m) |
| `--camera SOURCE` | `latency`: the wrist camera when the per-rig file names none |
| `--seconds S` | `table`, `point`: end after S seconds without saving (used by tests) |

Put the tool name right after `grasp-bench`. A `--gap-mm` list written before
it takes the tool name as one more reading, and the command exits with 2.

Every moving tool takes the arm the same way the driver does: one claim as
owner `chwyt`, the clutch, the impact guard and pre-positioning at 20 degrees
per second. On a real arm it prints what will move and waits for Enter; without
a terminal, or without that Enter, it refuses with `not confirmed`. From then on
a bare Enter is STOP, and so is Ctrl-C.

On the simulated arm (the default backend) the tools run the same way but
write the per-rig file only when `--rig PATH` names one, so a rehearsal never
fills `workspace/grasp.json` with simulated values.

**There is no free drive.** The A1X cannot be moved by hand. The teaching
tools jog the arm under position control in small Cartesian steps, and the arm
does not yield when it meets the table.

### 0. Prepare

1. Power the arm. If it does not follow the panel, enable it from the panel.
   Press **Dom** in the panel so every joint stands inside its limits, then
   **Rozlacz** and close the panel. (Engaging the clutch brings a joint that
   stands outside its limits back inside them. The folded rest pose is outside
   them.)
2. Check `configs/local.yaml`: `a1x.enable_on_connect: false`, and the
   `a1x.joint_offsets_deg` you want to run with (see
   [the joint offset rule](#the-joint-offset-rule)).
3. Check the table in `workspace/twin.json`
   ([Before anything moves](#2-before-anything-moves)).
4. Choose the cube spot on the arm's left, about 0.33 m out. Mark it with tape.

### 1. Gripper log

The jaw has no position feedback, so the twin models the gap from the
streamed opening at the jaw's measured speed. This step measures that speed
and whether the gripper effort can tell a free close from a close on the cube.

1. In `configs/local.yaml` set `a1x.gripper_force: 100` for the measurement.
   This turns the backend's effort freeze off. A freeze during a free close
   would cut the stroke short, and the short time would become the jaw speed
   of the gap model. The tool refuses with `freeze active` otherwise.
2. Run:

   ```bash
   lerobot-twin --config configs/local.yaml grasp-bench gripper --backend a1x --port can0 --i-know
   ```

3. The arm holds where it stands. A measured arm joint outside its limits
   makes the tool refuse with `arm outside limits` before anything is sent.
   After your Enter the jaw moves 99, 100 (a deliberate open, so the
   backend's first-command rule lets it through), then 0, 100, 0, 100 with
   nothing in the jaws, 2 s per move. Each time you see the jaws stop, type
   `s` and Enter: that is the stroke timed by eye. A bare Enter is STOP.
4. When the tool asks, put a 3 cm cube between the plates and type `d` and
   Enter. It closes on the cube and opens again. Anything else at that prompt
   is STOP, and the tool keeps only the free moves (exit code 1).

You should see one line per move: the target opening, the stream stroke time,
the effort peak and mean, the time of the last effort above the noise band,
your time by eye, and whether the move is valid. The stream stroke is about
1.3 s. The tool samples the gripper state at 150 Hz or more; below that it
says so, writes nothing and exits with 1. A move in which the freeze fired is
marked invalid (`NIE (zamrozenie)`) and none of its numbers are written. The
arrays go to `workspace/grasp/bench/gripper_<stamp>.npz`.

It writes, from valid moves only: `gripper.stroke_close_s_effort`,
`gripper.stroke_close_s_operator`, `gripper.stroke_open_s_effort`,
`gripper.stroke_open_s_operator` (the slowest of each kind),
`gripper.free_effort_peak`, `gripper.loaded_effort_peak`,
`gripper.separable` (true when the loaded peak exceeds 1.5 times the free
peak) and `gripper.suggested_threshold` (halfway between the two peaks when
they separate, else null). The slower of the two close strokes becomes the
jaw speed of the gap model. It never writes `gripper.freeze_threshold`.

**Record the freeze decision.** Choose the value of `a1x.gripper_force`:

- `gripper.separable` is true: the suggested threshold, between the free and
  the loaded peak. The jaw then stops closing when it meets the cube.
- `gripper.separable` is false: keep 100. The freeze stays off and the jaw
  squeezes with its own compliance only.

Set the value in `configs/local.yaml`, then record the same number:

```bash
lerobot-twin grasp-bench gripper --set-freeze 100      # or the threshold you chose
```

`--set-freeze` and `--gap-mm` only write the per-rig file: they connect
nothing and move nothing. Real mode refuses without this decision
(`gripper freeze undecided`) and when the running backend's threshold differs
from it (`gripper freeze differs`).

**Ruler.** Measure the tip gap with a ruler at openings 0, 50 and 100 (for
example with `g 0`, `g 50`, `g 100` in the teaching tool below) and record it:

```bash
lerobot-twin grasp-bench gripper --gap-mm 0:3 --gap-mm 50:53 --gap-mm 100:103
```

The gap model is 3 mm plus 1 mm per unit of opening. The tool adds the
readings to `gripper.gap_mm` and warns (`UWAGA: rozwarcie ...`) about a point
more than 3 mm off the model. Such a warning means the jaw's opening does not
follow what the policy was trained with. Check the jaw's zero and span
(`a1x.gripper_closed_pdes`, `a1x.gripper_open_pdes`) before going on.

### 2. Table touch

Teach the table at the cube spot, before you put the cube there. Anywhere
within 10 cm of the grasp point works; further away the run refuses with
`table touch far`.

```bash
lerobot-twin --config configs/local.yaml grasp-bench table --backend a1x --port can0 --i-know --xy 0.0 0.33 --yaw-deg 0
```

1. The arm moves at 20 degrees per second to a top-down pose 0.12 m above
   (0, 0.33).
2. Jog it down with line commands, each followed by Enter:

   | command | step |
   |---|---|
   | `d MM`, `u MM` | down, up; at most 5 mm, and at most 1 mm down below TCP z 0.06 m |
   | `x MM`, `y MM` | along base x or y, signed, at most 5 mm |
   | `r DEG` | yaw, at most 5 degrees |
   | `g N` | gripper opening 0..100 |
   | `ok` | store the measured pose |
   | `q` | end the tool without saving |

   An empty line is STOP. A longer step is cut to its limit
   (`Krok przyciety do limitu`). After every line the tool prints the
   measured TCP, the commanded height, the yaw and the jaw.
3. Go down in 1 mm steps near the table. Three rules watch every step and the
   first one to fire ends the step with `Kontakt prawdopodobny`: the tool
   holds the measured pose and goes up 2 mm.
   - Height: below TCP z 0.06 m, the measured tip fell less than 0.4 mm over
     the last two 1 mm steps. In free air it follows 1 mm per step; on the
     table about 0.1 mm.
   - Lag: an arm joint fell more than 1.5 degrees further behind its command
     than at the start of the step, and stayed there for the 0.3 s after it.
   - Effort: the impact guard (lag 3 degrees for 0.15 s, or an effort over
     `guard.effort_abs`).
4. When the tool says contact is likely, it tells you what to do: come back
   down the 2 mm with `d 1` twice, so the fingertips rest on the table, and
   type `ok`.

You should see the stored touch height and its difference from 33.9 mm (the
TCP sits 33.9 mm above the fingertips when the tool points down), and how far
the touch lies from the grasp point once that is taught. In the twin's
simulation the stored height lies within 2 mm of 33.9 mm. More than 15 mm off
means a mount or kinematics problem. The tool then refuses to save
(`Nie zapisano`) unless you give `--force`, which also writes `overrides.table`.

It writes: `table.touch_tcp_z`, `table.tcp`, `table.joints` and
`table.taught_at` (ISO 8601, UTC). The pose is the measured one, averaged
over 0.5 s.

### 3. Grasp point

Put the 3 cm cube on the spot, one face across the closing axis.

```bash
lerobot-twin --config configs/local.yaml grasp-bench point --backend a1x --port can0 --i-know \
    --xy 0.0 0.33 --yaw-deg 0 --object-height 0.030
```

Jog as for the table (`g 100` opens the jaw) until the open plates straddle
the cube and the fingertips touch the table. `r DEG` turns the closing axis
across the cube's faces. Check by eye that the plates stand beside the cube
and not on it. Then `ok`.

You should see the stored TCP in mm and the closing-axis yaw in degrees
(note both: the dot check below starts from them), then the believed grasp
that follows: the taught x and y at the touch height plus 6 mm (training
keeps the fingertips 6 mm above the table), and whether its hover lies
inside training's envelope (`Zawis w obszarze treningu`). The taught height
itself is stored for a cross-check only.

It writes: `grasp.tcp`, `grasp.yaw` (radians), `grasp.joints`,
`grasp.taught_at`, `grasp.object_height` and `grasp.joint_offsets_deg` (read
from the running backend).

Teaching is valid for 12 hours. After that every mode refuses with
`teaching old` until you teach again, or write `overrides.taught`.

### 4. Wrist camera source

Edit `workspace/grasp.json` and set `wrist.source` to the wrist camera, in
the form the workspace cameras use: an OpenCV index such as `"2"` on macOS,
or on Linux the stable path `/dev/v4l/by-id/...-video-index0` (a
`/dev/videoN` number moves when another USB camera is plugged in). The
camera must deliver 1920 x 1080 frames. Keep `wrist.upside_down: true`: the
camera is mounted upside down and the twin folds the 180-degree rotation into
the undistortion map. `--camera SOURCE` on `grasp` and `grasp-bench latency`
stands in for a missing `wrist.source`, but `grasp-bench show` only reads the
file.

`wrist.K` and `wrist.dist` stay null unless you have a better lens
calibration of the upright 1920 x 1080 image. The default was measured on
2026-09-26 from 15 views with 0.85 px RMS and is not trusted. The dot check
below judges it. A pair whose view reaches noticeably past the sensor, or
whose lens model folds the image, is refused when the camera opens
(`Kamera na nadgarstku: wrist camera K and dist ...`, exit code 1); a thin
strip past one edge only logs a warning and stays black.

You should see `grasp-bench show` no longer list `wrist camera not set`.

### 5. Dry run and the dot check

Dry mode never moves the arm, so first bring it to the hover over the taught
point with the teaching tool. The hover is the believed grasp plus 6 cm: TCP
z = `table.touch_tcp_z` + 0.066 m, about 0.10 m. Give `--xy` the taught
`grasp.tcp` x and y (`grasp-bench show`) and `--yaw-deg` the yaw the `point`
tool printed (`grasp.yaw` is stored in radians):

```bash
lerobot-twin --config configs/local.yaml grasp-bench point --backend a1x --port can0 --i-know \
    --xy <x> <y> --yaw-deg <yaw> --start-z 0.10
```

Type `q` once the arm stands there: the tool ends without saving (exit code
1) and the arm keeps holding the pose. Then, with the cube on its spot:

```bash
lerobot-twin --config configs/local.yaml grasp --mode dry --backend a1x --port can0 --seconds 20
```

Dry mode never claims the arm and never sends a target. Each tick it reads
the newest frame and joints, builds the tick-0 observation at the hover, runs
the network and logs the step it would take. It ends on Enter, Ctrl-C or
after `--seconds`. Every `--dump-every` ticks (default 20) and at the first
tick it writes to `workspace/grasp/dry/<stamp>/`:

- `NNNN.png`: the policy frame enlarged 4 times, a cross at the believed
  pixel, a circle of 8 px (at 160 x 120) around it, and a small square where
  the camera model expects the centre of the cube's top face;
- `NNNN_heat.png`: the fourth input channel;
- `ticks.jsonl`: one line per tick.

Repeat the dry run at two other poses over the cube: one lower
(`--start-z 0.08`) and one with `--xy` 2 cm off the point.

**The dot check.** In each dump, find the centre of the cube's top face in the
image and measure its distance to the cross, in pixels of the 160 x 120 frame
(the dump is 4 times larger). The limit is 8 px, the radius of the circle.
The distance between the square and the real top face is the error of the
camera pose. Record the largest error you measured:

```bash
lerobot-twin grasp-bench dot --error-px 5.5
```

Above 8 px shadow and real refuse with `dot off`. The fix is a new calibration
of the camera on the hand, not a change to the driver. `overrides.dot` lifts
the limit.

You should see, at the end of the dry run, a summary (`Podsumowanie dry`)
with p50, p95 and max of frame age, feedback age, joint age, inference, tick
time and loop gaps; the counts of non-finite actions, stale ticks and skips
by reason; the largest arm effort per joint; whether pairing used a measured
camera latency; and two verdicts: frame age p95 under 120 ms, inference
median under 30 ms and p95 under 45 ms. Expect no stale ticks and no
non-finite action. The arm efforts are logged once a second while the arm
stands. Set `guard.effort_abs` above them if the calm values are far from 12.0.

#### The joint offset rule

The wrist camera pose was calibrated on raw joint angles, with every
`a1x.joint_offsets_deg` at zero. A 1.49-degree offset on joint 3 moves the
TCP by 9 mm.

- `joint offsets differ`: the running backend's offsets are not all zero
  (tolerance 0.05 degrees). Every mode on a real backend refuses. With
  `overrides.joint_offsets` the run goes on and prints a warning: for a point
  taught by touch the believed pixel moves by under 0.3 px, for a point from
  any other source by about 4 to 5 px.
- `joint offsets changed`: the offsets differ from those in force when the
  grasp point was taught (`grasp.joint_offsets_deg`). There is no override.
  Teach the point again.
- Teaching is valid for 12 hours. After that every mode on a real backend
  refuses with `teaching old` (`overrides.taught` lifts it). A table or a
  point without a `taught_at` stamp counts as old.

### 6. Latency

```bash
lerobot-twin --config configs/local.yaml grasp-bench latency --backend a1x --port can0 --i-know
```

It needs the model (for the inference time) and the wrist camera, and checks
both before anything moves. It times the policy on this machine with the twin
loop and the camera running, asks for Enter, and moves the arm at 20 degrees
per second to the hover over the taught point. There it steps joint 6 by
3 degrees and back, five steps, at full speed. For each step it times the
first fresh joint sample that moved more than 0.2 degrees (`actuation_s`),
and from that sample the first frame that changed by more than 4 sigma from
the still frames (`camera_s`, measured against the frame's stamp). The arm
ends at the hover and holds there.

You should see one line per step, then the medians and `total_s` =
`camera_s` + inference p95 + `actuation_s`. The limit is 150 ms; above it
shadow and real refuse with `latency high` (`overrides.latency` lifts it).
With fewer than three good steps the tool writes nothing and exits with 1.
A negative camera latency (a joint sample stamped later than the frame that
already shows the motion) is stored as 0: pairing then uses the frame's own
stamp.

It writes: `latency.camera_s`, `latency.actuation_s`, `latency.infer_p95_s`
and `latency.total_s`. From now on the driver pairs each frame with the joints
at the frame's stamp minus `camera_s`.

After this step `grasp-bench show` should list nothing for shadow mode. Real
mode still misses `no shadow episodes`.

## 5. Run it

### The command

```bash
lerobot-twin grasp [--mode dry|sim|shadow|real] [--backend B] [--port P] [--i-know]
                   [--checkpoint PATH] [--rig PATH] [--workspace PATH] [--camera SOURCE]
                   [--episodes N] [--seed N] [--outside-training]
                   [--floor bench|training] [--gap scene|model] [--jaw-stroke-s S]
                   [--dump-every N] [--seconds S] [--object-height M]
```

| flag | what it does |
|---|---|
| `--mode` | Default `dry`. See [The four modes](#the-four-modes). |
| `--backend B` | The arm backend. Required in `shadow` and `real`. `dry` and `sim` default to `sim` and never take the backend from the workspace file. |
| `--port P` | The CAN port: `can0` on Linux, `xcan` on macOS. Default: the port saved in the workspace. |
| `--i-know` | Required in `shadow` and `real`. Without it the command exits with code 2. |
| `--checkpoint PATH` | The model file. See [Get the model](#3-get-the-model). |
| `--rig PATH` | The per-rig file. Default `workspace/grasp.json`. |
| `--workspace PATH` | The workspace file. Default `workspace/twin.json`. |
| `--camera SOURCE` | The wrist camera when the per-rig file names none. |
| `--episodes N` | Episodes in the series. Default 1. |
| `--seed N` | Seed of the cube draw and the hover noise on the simulated arm. Default 0, so every run without `--seed` draws the same cubes. |
| `--outside-training` | Run a hover outside training's envelope, with a warning. |
| `--floor bench\|training` | `sim` only. `training` uses training's floor (TCP 20 mm above the table) and turns the measured-side rules and the collision check off. Default `bench`. |
| `--gap scene\|model` | Where the gap in the observation comes from. `scene` (`sim` only) reads the simulated jaw; `model` is the estimator the real arm uses. Default: `scene` in `sim` without `--jaw-stroke-s`, else `model`. |
| `--jaw-stroke-s S` | `sim` only. Slows the commanded jaw to a full stroke in S seconds. |
| `--dump-every N` | `dry` only. Dump every N ticks. Default 20. |
| `--seconds S` | `dry` only. End after S seconds; without it the dry run ends on Enter or Ctrl-C. |
| `--object-height M` | `dry` only, for the cube-top square in the dump. Default: the per-rig value, else 0.030 m. |

`--config FILE` gives the YAML config, before or after the command name:
`lerobot-twin --config configs/local.yaml grasp ...`.

Exit codes: 0 when the run completed, 1 on a failure (connect, camera, arm
disconnected, a guard ending on a real backend, Ctrl-C), 2 on a refusal.

### The four modes

| mode | what moves | needs |
|---|---|---|
| `dry` | nothing: no claim, no target | on a real backend: the table and the grasp point taught, the wrist camera set |
| `sim` | the simulated arm, with a cube in the twin's scene | a simulated backend; on a real backend it refuses with `not armed` |
| `shadow` | the real arm follows the policy with the jaw held open at 100 and the floor raised above the object | `--backend`, `--i-know`, every bench value for shadow |
| `real` | the real arm, jaw live | `--backend`, `--i-know`, every bench value, and `overrides.shadow_done` |

Run them in this order on the rig: `dry`, then `shadow`, then `real`.
`shadow` and `real` also run on the simulated arm as a rehearsal:
`--mode shadow --backend sim --i-know`.

**Consent.** `shadow` and `real` need `--i-know`. On a real backend the
command then prints a warning with the age of the table and the point
teaching, and asks:

```
Enter = ramie jedzie nad punkt chwytu (20 st./s); Ctrl-C = koniec.
```

Without a terminal on stdin it refuses. Engaging the clutch brings any joint
that stands outside its limits back inside them.

After each episode the command prints one line (`Epizod k/N: <ending>`) and
a second line with the numbers of the episode report (`EpisodeReport` in
`grasp/driver.py`). The checklists in section 8 read them:

| printed | report field | meaning |
|---|---|---|
| `tiki N (w tym pominiete S)` | `ticks`, `skipped` | ticks run, skipped ones included |
| `proby A` | `attempts` | close attempts |
| `IK odrzucone I` | `ik_refused` | steps refused by the IK rule (not converged, or a joint over 6 degrees) |
| `kolizje odrzucone C` | `collision_refused` | steps refused by the collision check |
| `ogr. predkosci R` | `rate_limited` | ticks whose target the twin's slew (90 degrees per second) had not reached, within 0.5 degrees, by the next tick |
| `max opoznienie L st.` | `max_lag_deg` | largest gap between the command and the measured joints on a scored tick |
| `min TCP z Z mm` | `min_tcp_z` | lowest measured TCP height of a scored tick |
| `podloga F0 -> F1 mm` | `floor_z`, `floor_z_end` | command floor at the start and at the end |
| `ugiecie S mm` | `sag` | sag the floor absorbed |
| `podloga podniesiona Nx` | `floor_raised` | ticks on which the measured tool raised the floor |
| `brzeg pudelka Nx`, `podloga dotknieta Nx` | `box_hits`, `floor_hits` | proposals clipped at the box edge or at the floor |
| `ogr. predkosci P % tikow` | `rate_limited` / `ticks` | shadow mode only |
| `tik median ... ms` | `tick_ms` | tick time |
| `bez reguly wysilku: ...` | `no_effort_rule` | joints whose effort rule was off (loaded at rest) |
| `log PATH` | `log` | the episode log |

The three `podloga podniesiona`, `brzeg pudelka` and `podloga dotknieta`
counts are printed in shadow mode, or when one of the first two is not zero.
Each tick is one JSON line in the episode log,
`workspace/grasp/runs/<stamp>/episode_<n>.jsonl`. The log holds numbers only;
the dry run is the only mode that writes images.

### STOP

- **Enter alone always stops**, from the moment the arm may move until the
  command ends. There is no moment in which a bare Enter means "go on".
- **Ctrl-C** stops, at any point: while connecting, at a prompt, during
  pre-positioning or during an episode.
- **To go on to the next episode on a real backend, type `d` and Enter** at
  the prompt (on the simulated arm the episodes follow one another without
  it):

  ```
  d + Enter = kolejny epizod (szczeki otworza sie nad punktem - zabierz kostke); sam Enter = STOP.
  ```

- Anything else at that prompt is a STOP and ends the series: a bare Enter,
  other text, a line typed before the prompt appeared, or an answer within
  1 s of the end of an episode. Episodes end by themselves, so an Enter meant
  as STOP at that moment must not start the next one.
- Before you type `d`, take the cube out of the jaws if they hold it and put
  it back on its spot. The next pre-positioning rises straight up and opens
  the jaw at the hover.
- STOP holds the arm where it is and stops the jaw at the opening the stream
  has reached. It does not cut the motors: the A1X has no brakes. The command
  never clears a STOP. It closes, and the arm keeps holding the last pose it
  was sent.
- On a real backend every ending except `lifted`, `tick cap`, `time cap` and
  `success` also ends the series, with exit code 1. So does a refused
  pre-positioning (exit code 2 for `ik`, `collision`, `not arrived`,
  `j5 offset` and `speed limited`, 1 for anything else).

## 6. What happens in an episode

### Taking the arm

Before anything moves, the driver checks, in order: an arm is connected
(`not connected`), no STOP is active (`stop`), nobody else owns the arm
(`busy`), the mode fits the backend (`not armed`: `sim` on a real arm,
`shadow` or `real` without consent, or a backend without a gripper state).
Then, on a real backend, it listens 0.3 s for another CAN sender (`busy`).

It claims the arm once as owner `chwyt` and engages the clutch. It releases
the arm only when the command ends. Before any motion it checks that every
arm joint may move at 90 degrees per second. The supervisor's
`safety.velocity_scale`, a panel speed cap and the backend's own
`a1x.max_step_deg_s` each lower that limit. Anything under 90 refuses with
`speed limited`. The speed cap is 20 degrees per second during
pre-positioning and after every ending, and 90 only while an episode's ticks
run. When the command ends it puts back the cap it found.

### Pre-positioning

1. **The hover.** The believed grasp plus 6 cm. The twin's IK solves it from
   the home pose for the taught yaw and for the yaw plus 180 degrees. A
   solution counts when it is within 1 mm and 5 mrad by forward kinematics,
   joint 5 is within 0.5 degrees of zero, and joint 6 stands at least 15
   degrees inside its limits. The one with more room on joint 6 wins. No
   solution: refusal `ik`.
2. **Straight up first.** The tool first rises straight up to the hover
   height, in steps of at most 5 mm, when three things hold: it points down
   within 10 degrees, it stands more than 1 cm below the hover height, and
   either an earlier move of this command left it there or it stands within
   15 cm of the hover in x and y. The tool keeps its measured orientation and
   the jaw command does not change; the joint plan squares the tool at the
   hover height. There is no collision check on this straight path; the
   impact guard watches it. From the second episode on the arm starts low,
   beside the cube; the joint plan alone would sweep the tool sideways by 4
   to 7 cm before it rises. At long reach a vertical line can leave the IK
   step's reach before the hover height (joint 4 at its limit). A refused
   step ends the rise there when the TCP already stands 15 mm above the
   touch height plus the object's height; the collision-checked joint plan
   goes on from there and the log says so. A refused step lower down ends
   the move with `ik`.
3. **The joint plan.** Joint groups in stages, as the panel's Home does, each
   path checked by the collision checker (`collision`). Each stage is a
   smoothstep ramp at 20 degrees per second, at least 0.5 s, all seven joints
   in every target. The impact guard watches with a lag limit of 4 degrees
   for 0.15 s.
4. **The jaw opens at the hover, never earlier:** 99 for 0.1 s, then 100 (the
   A1X backend starts the gripper stream only for a command that differs from
   the reported opening). Then it waits until the streamed opening reaches 99,
   at most 2 s.
5. **Arrival**, within 3 s: every arm joint within 2 degrees of the hover and
   slower than 10 degrees per second, the TCP within 3 mm in x and y, and in
   z between 11 mm below and 3 mm above the hover (the servos sag under the
   arm's weight). Else `not arrived`, with the misses in the detail.
6. On a real backend, measured joint 5 more than 1 degree from zero: refusal
   `j5 offset`.
7. **Sag.** The hover height minus the measured TCP height, averaged over
   0.3 s and clipped to 0..8 mm. It starts the command-side floor.

### Each tick

The policy tick runs at 20 Hz and writes one target. The twin loop (50 Hz)
slews the arm towards it at up to 90 degrees per second.

| step | what the driver does | when it fails |
|---|---|---|
| 1 | Hard checks: a cancel (STOP, preempt, impact, `stop()`), another owner, STOP active, arm disconnected. | ending `stop`, `preempted`, `impact ...`, `not connected` |
| 2 | Caps: 120 ticks including skipped ones (6 s), or 12 s. | ending `tick cap`, `time cap` |
| 3 | Joints: the newest fresh joint sample and the backend's feedback age. Older than 0.1 s, or the twin loop frozen: skip. Older than 0.3 s: end. | skip or ending `stale feedback` |
| 4 | Frame: the newest wrist frame. None, or older than 0.12 s: skip. Older than 0.5 s, or the stream died: end. | skip or ending `camera` |
| 5 | Pairing: the joint sample nearest the frame's exposure time, which is the frame's stamp minus `latency.camera_s`. None within 60 ms: skip. | skip `pairing` |
| 6 | Tracking lag: the largest gap between the command and the measured joints. Over 8 degrees: skip. The command holds and the policy does not run. | skip `lag` |
| 7 | The measured floor (below): the sag estimate, the floor raised by a low measured tool, the `table` check. | ending `table` |
| 8 | Gap: the estimate from the streamed opening. In shadow mode, from a virtual jaw that follows the policy's gripper command. | |
| 9 | Observation: the 19 numbers from the newest joints, the gap and the bookkeeping, and the believed pixel from the paired joints. | ending `non-finite` |
| 10 | Policy: one forward pass on the CPU. | ending `non-finite` |
| 11 | Bookkeeping: the step, clipped to the box (8 cm around the hover in x and y), the floor and the ceiling (8 cm above the hover); the yaw step; the gripper value. | |
| 12 | IK and collision: one warm IK step from the last accepted joints. It is accepted when it converged (1 mm and 5 mrad within 60 iterations), no joint moves more than 6 degrees, and the collision checker clears the new joints (the bench scene: arm and table, no cube). A refusal holds the command, as in training, and is counted. | counted |
| 13 | Gripper: opening = 100 x the gripper value. With `--jaw-stroke-s` it is slewed. In shadow mode the value sent is always 100. | |
| 14 | Send: seven finite joints to `Twin.set_target`. | ending `non-finite` |
| 15 | Success or lift (below). | ending `success`, `lifted` |
| 16 | Record: one JSON line. The tick counts as rate limited when, at the next tick, the twin's command is more than 0.5 degrees from this target. | |
| 17 | Sleep until the next tick on an absolute schedule. | |

**Skips.** A skipped tick sends the previous target again and changes no
bookkeeping. It counts towards the caps. Five skips in a row end the episode
with the reason of the last one; for `lag` it takes ten (0.5 s).

**Success and lift.**

- In simulation the driver reads training's condition from the scene: both
  jaw contact sensors touch the cube, the cube is more than 4 cm above its
  start, and the gripper is commanded closed. The ending is `success`.
- On a real backend there is no success detector. The episode ends with
  `lifted` when the gripper is commanded closed and the measured TCP has
  risen 4 cm above its lowest point of the episode. The operator and the
  video decide whether the cube came along.
- In shadow mode there is neither.

### The floors

The policy may push the tool down onto the table. Two floors keep the
fingertips off it on the real arm.

**Command side.** The commanded TCP never goes below the taught touch height
plus 4 mm plus the sag. The sag starts at the value measured at the hover. It
only grows within an episode, from an estimate taken while every joint moves
slower than 10 degrees per second (time constant 0.3 s), up to 12 mm. The sag
grows with reach, so one value from the hover is not enough: in the twin's
simulation the tips touched the table at the edge of the box with it.

**Measured side.** Each tick the driver computes the measured tool height
from the newest joints: the lower of the TCP and the lowest fingertip corner
plus 33.9 mm, with the jaw at the widest opening that can be true at that
moment. At full lateral speed the wrist lags and the tool tilts, and an open
finger then sits a few millimetres lower than the TCP says. When this height
is under the touch height plus 4 mm, the floor rises by the deficit, up to
the touch height plus 16 mm, and the tick counts in `floor_raised`. When it
is under the touch height plus 0.5 mm on two ticks in a row, the episode ends
with `table`. The floor only rises within an episode. A command below a
raised floor comes up at the next tick.

**The shadow floor.** The touch height plus the object's height plus 10 mm
plus the sag, and never below the command-side floor or the believed grasp
plus 20 mm. With the jaw open the plates stand 51.5 mm either side of the TCP
and the box allows 8 cm of travel, so a finger can stand over the cube. With
the 3 cm cube the fingertips stay 40 mm above the table after the sag, 10 mm
above the cube's top.

**Training floor.** `--floor training`, simulation only: TCP 20 mm above the
table, no measured-side rules and no collision check. Training's floor pushes
the tips 13 mm into the table, which only a simulated table forgives.

Both floors are on heights of the TCP, not of the fingertips. The measured
TCP sits about the sag below the command, so at the command-side floor it is
about 4 mm above the touch height: 2 mm under the believed grasp (touch
height plus 6 mm), with the fingertips about 4 mm above the table.

### The impact guard

The same guard as the Spectacles driver (`twin/impact.py`), listening to every
tick of the twin loop:

| rule | trips when |
|---|---|
| lag | a joint is more than 3 degrees behind its command for 0.15 s while it barely moves; 4 degrees during pre-positioning |
| effort | a joint's effort is over the limit for 3 samples in a row. Real arm: `guard.effort_abs` (12.0, a guess) for every arm joint. Simulation: half the actuator's force range |

A joint whose effort is already above half its limit when a move starts gets
only the lag rule for that move. The log names it and the report lists it
(`no_effort_rule`). A hit ends the episode with `impact lag <joint>` or
`impact effort <joint>`. There is no backoff: the arm holds the measured
pose.

The twin's own stops stay behind the guard: the A1X "deaf or blocked" check
(3 degrees for 1 s while still), the supervisor's tracking stop (25 degrees
for 0.5 s) and the other-sender fault. Each ends the episode with `stop`.

### Endings

| ending | meaning |
|---|---|
| `success` | simulation: the cube is lifted between the jaws |
| `lifted` | real backend: jaw commanded closed and the TCP 4 cm above its lowest point |
| `tick cap` | 120 ticks |
| `time cap` | 12 s |
| `stop` | Enter, Ctrl-C, or a STOP in the twin: a fault, another CAN sender, the deaf check, the tracking stop |
| `preempted` | something else took the arm |
| `not connected` | the arm disconnected |
| `impact lag <joint>`, `impact effort <joint>` | the impact guard tripped |
| `lag` | ten skips in a row for a tracking lag over 8 degrees |
| `table` | the measured tool (TCP or lowest fingertip corner) under the touch height plus 0.5 mm on two ticks in a row |
| `stale feedback` | joints or CAN feedback older than 0.3 s, or five stale skips |
| `camera` | no frame for 0.5 s, the stream died, or five camera skips |
| `pairing` | five pairing skips |
| `non-finite` | a NaN or infinity in the observation, the action or the target |
| `speed limited` | a speed limit under 90 degrees per second when the loop starts |
| `error` | an exception in the tick (the log has the traceback), or an episode without a pre-positioning |
| any other text | the reason given to `stop()`, for example `cube lost` in the sim evaluation |

**Holding after an ending.** Every ending holds, in one call to the twin.

- After `impact ...`, `lag` and `table` the arm holds the plain measured
  reading: a blocked joint stops pushing.
- After every other ending it holds the twin's own hold pose, which accounts
  for braking. The measured reading is up to two read periods old, so holding
  it would drive a moving joint back by 2 to 4 degrees.
- After `lifted`, `tick cap`, `time cap` and `success` the jaw keeps its
  command and keeps squeezing. After every other ending the jaw stops at the
  opening the stream has reached.
- When the twin itself took the arm (STOP, preempt), it already holds. The
  driver then only stops the jaw at the streamed opening. On the A1X the
  twin's STOP holds the last gripper command, so a closing jaw would
  otherwise finish its stroke.

The driver never ends by leaving the last target in place.

### Dry mode

`dry` never claims, never engages, never sets a target, a speed or a hold.
It attaches a tick listener to fill the joint history. Each tick is a tick-0
observation at the hover: yaw error 0, gripper open, previous action 0, no
attempts. Soft guards are logged and end nothing. On a real backend that was
never engaged the twin sends no CAN frame at all.

## 7. Sim evaluation

```bash
lerobot-twin grasp-eval                                   # 50 episodes, seed 50000, the sim jaw
lerobot-twin grasp-eval --jaw measured                    # the jaw slowed to the measured stroke
lerobot-twin grasp-eval --floor training --gap scene      # training's floor and the scene's gap
```

```bash
lerobot-twin grasp-eval [--episodes 50] [--seed 50000] [--jaw sim|measured] [--stroke-s S]
                   [--gap scene|model] [--floor bench|training] [--jitter] [--checkpoint PATH] [--rig PATH] [--out PATH]
                   [--workspace PATH]
```

The evaluation always connects the simulated backend, whatever the workspace
says, and never opens a CAN link. It builds one scene: a 1.2 x 1.2 m table, a
3 cm cube, contact sensors on both jaws and the wrist camera on the hand. The
table has training's colour and the scene training's two lights and headlight,
so the wrist image looks like training's: at the same joints the median pixel
differs by about 5 levels from training's recorded images, against 57 with the
twin's own look (`tests/test_twin_grasp_scene.py`). With the twin's own look
the rate at the sim jaw was 68 %. Each episode:

1. Draws the cube in training's region (x from -0.20 to 0.20 m, y from 0.22
   to 0.40 m, yaw within 45 degrees), again until its hover lies inside
   training's envelope.
2. Adds training's hover noise to the believed grasp (normal, sigma 15 mm in
   x and y, 10 mm in z, 10 degrees in yaw).
3. Pre-positions the arm, which rises straight up from where the last episode
   left it and opens the jaw.
4. Places the cube and lets it settle for 0.2 s.
5. Runs the episode. Success is training's condition read from the scene. The
   evaluation stops an episode with `cube lost` when the cube moves more than
   15 cm.

| flag | what it does |
|---|---|
| `--episodes N`, `--seed N` | default 50 and 50000. The same seed draws the same cubes and the same hover noise at both jaw speeds, so the two runs compare episode by episode |
| `--jaw sim` | the simulated jaw at its own speed (a full stroke in about 0.46 s); the gap from the scene by default |
| `--jaw measured` | the driver slews its jaw command to a full stroke in `--stroke-s`, else the per-rig stroke (`--rig`), else 1.3 s; the gap from the estimator by default |
| `--stroke-s S` | with `--jaw measured` only; with the sim jaw the command refuses (exit code 2) |
| `--gap scene\|model` | override the gap source |
| `--floor training` | training's floor, no measured-side rules, no collision check |
| `--jitter` | per episode, jitter the camera by up to 3 mm, 0.7 degrees and 0.3 degrees of field of view, in the image only. The believed pixel keeps the nominal pose |
| `--out PATH` | the result file. Default `workspace/grasp/eval/<stamp>.json` |
| `--workspace PATH` | must describe an A1X (else `not a1x`); its backend is ignored |

It prints one line per episode (with `SUKCES` or `porazka`) and a summary.
The result file is rewritten after every episode and holds:

- `rate`, `successes`, `episodes`;
- `endings`: the count of every ending. `impact`, `lag` and `table` are rows
  of their own;
- `tick_ms` (median, p95, p99, max) and `infer_ms` (median, p95);
- `rate_limited_frac` (share of all ticks), `ik_refused_frac`,
  `jump_refused_frac` (the part of the IK refusals that only the 6-degree
  rule refused; training took those steps) and `collision_refused_frac`
  (shares of the ticks that ran the policy);
- `config`: jaw, stroke and where it came from, gap, floor, jitter, seed,
  the checkpoint and its sha256;
- `per_episode`: the cube, the hover, the ending, the cube's rise and the
  log of each episode.

The episode logs go next to the result file, into a directory of the same
name without `.json` (`<stamp>/episode_<n>.jsonl`).

The evaluation stops early after an ending that leaves nothing to move
(`stop`, `preempted`, `not connected`, `busy`) or after three pre-positioning
refusals in a row; it then says `Przebieg niepelny`. Exit codes: 0 when the
run completed, 1 when it could not run or stopped early, 2 on a refusal.

### Reading the result

The summary names the verdict (`Werdykt: ...`) and checks the tick time: p99
under 40 ms.

- **70 % or more at the sim jaw speed** is good. Go on to the bench.
- **50 to 70 %.** The verdict says to read the endings (`impact`, `lag`,
  `table`, `tick cap`) and the episode logs before changing anything.
  `table`, `impact ...` and `lag` are rows of their own. A rerun with
  `--floor training --gap scene` shows how much the twin's own floor and gap
  model cost. Record both numbers before the shadow episodes.
- **Below 50 %** the summary says: "ponizej 50 %: przyczyna moze byc sciezka
  obserwacji, fizyka kontaktu albo render sceny blizniaka". The three causes
  weigh the same, because the twin's scene does not reproduce training's
  contact physics (condim, friction, noslip, cone), and its render matches
  training's only in the table and the lights (no other objects, no dog, one
  cube colour). Separate them:
  1. Run the component tests ([Tests](#11-tests)):
     `tests/test_twin_grasp_policy.py`, `tests/test_twin_grasp_observation.py`
     and `tests/test_twin_grasp_camera.py`. The network, the observation and
     the projection are checked against fixtures recorded from training. When
     they pass, the observation path is proven.
  2. Rerun with `--floor training --gap scene`. A rate that recovers points at
     the twin's floor or the gap model.
  3. What remains is contact physics or the render.
- **The slowed jaw** (`--jaw measured`): record the number. Below 50 % open a
  ticket to retrain the student with the jaw slewed to the measured stroke
  and a lagging gap observation.

Never raise a driver threshold to raise the score.

### Results on this branch

Record both 50-episode runs here with the date, the seed and the checkpoint's
sha256. First runs, 2026-09-27, seed 50000, sha256 `05e5aaf6...cdcb8536`, no
per-rig file (so the slowed jaw used the default 1.3 s stroke), on a machine
busy with other test runs:

| run | rate | endings | tick p99 | inference median / p95 |
|---|---|---|---|---|
| `--jaw sim` | 26 of 41 (63 %) | success 26, tick cap 12, ik 3 | 21.4 ms | 9.2 / 12.3 ms |
| `--jaw measured` (1.3 s) | 0 of 24 (0 %) | tick cap 21, ik 3 | 19.4 ms | 9.3 / 11.9 ms |

Neither of those runs reached 50 episodes. Both stopped with
`Przebieg niepelny` after three pre-positioning refusals in a row: at long
reach a step of the straight-up retreat was refused by the IK rule (joint 4
at its limit), and the arm could not leave that pose. The retreat now hands
over to the joint plan when the tool already stands well above the object.
Second runs the same day, same seed and checkpoint:

| run | rate | endings | tick p99 | inference median / p95 |
|---|---|---|---|---|
| `--jaw sim` | 35 of 50 (70 %) | success 35, tick cap 15 | 22.4 ms | 9.7 / 13.0 ms |
| `--jaw measured` (1.3 s) | 0 of 50 (0 %) | tick cap 50 | 27.1 ms | 9.9 / 13.6 ms |

Third runs the same day, same seed and checkpoint, after the scene got training's
table colour and lights. The first action at the same joints, pixel and 19-vector had
differed from training's by up to 0.8 (y and yaw); the failed episodes closed the jaw
next to the cube again and again. The observation vector and the believed pixel were
already equal to training's (1e-7 and 0 px at the first tick):

| run | rate | endings | tick p99 | inference median / p95 |
|---|---|---|---|---|
| `--jaw sim`, the twin's own look | 34 of 50 (68 %) | success 34, tick cap 16 | 27.2 ms | 9.9 / 12.7 ms |
| `--jaw sim` | 46 of 50 (92 %) | success 46, tick cap 4 | 32.3 ms | 10.1 / 12.7 ms |
| `--jaw measured` (1.3 s) | 0 of 50 (0 %) | tick cap 50 | 22.2 ms | 9.5 / 12.3 ms |

Of the four failures at the sim jaw, three close beside the cube seven times and one
stands at the hover for all 120 ticks: every step the policy asks for is refused by the
6-degree rule (a full yaw step with a sideways step turns joint 6 by more than 6 degrees),
and the same image brings the same request.

With the slowed jaw every episode runs into the tick cap. By the verdict this calls for the retraining
ticket (slowed jaw, lagging gap observation) before the first real grasp.

## 8. The shadow episodes and the first real grasp

### The shadow checklist

In shadow mode the policy runs on the real arm at full speed with the jaw held
open at 100 and the shadow floor. The observation is built as if the jaw
obeyed the policy: the gap comes from a virtual jaw. The fingers touch
nothing.

Before: every bench step done, `grasp-bench show` lists nothing for shadow,
the cube on its taped spot, a camera recording the arm.

```bash
lerobot-twin --config configs/local.yaml grasp --mode shadow --backend a1x --port can0 --i-know --episodes 10
```

Read every episode's report against the **shadow floor**. The fingertips stay
10 mm above the cube's top after the sag; with the 3 cm cube the TCP stays
about 74 mm above the table.

| check | report field | pass |
|---|---|---|
| no automatic stop | `ending` | `tick cap` or `time cap` only |
| tracking lag | `max_lag_deg` | under 5 degrees |
| TCP above the floor | `min_tcp_z` against `floor_z` minus `sag` | not lower |
| no measured-side floor raise | `floor_raised` | 0 |
| TCP inside the box | `box_hits` | few; many mean the policy wants to go elsewhere, so check the dot again |
| rate limiting | `rate_limited` over `ticks` | under 10 %. Expect it to be close: random actions give about 10 % |
| the policy aims at the cube | the video | the tool descends to the floor and yaws towards the cube |

After ten clean episodes write a note into `overrides.shadow_done` in
`workspace/grasp.json`, for example
`"shadow_done": "10 clean episodes 2026-10-02, M.W."`. Real mode refuses
without it.

### First real grasp checklist

Before: ten clean shadow episodes, the jaw stroke measured, the freeze
decision recorded with the same `a1x.gripper_force` in the running config,
teaching younger than 12 hours, a camera recording, a hand on Enter.

```bash
lerobot-twin --config configs/local.yaml grasp --mode real --backend a1x --port can0 --i-know --episodes 3
```

Three attempts on the 3 cm cube at the taught point. Between them, take the
cube from the jaws, put it back on the spot, then `d` and Enter.

| check | report field | pass |
|---|---|---|
| no automatic stop, no table strike | `ending` | `lifted`, `tick cap` or `time cap`; never `table`, `impact ...` or `lag` |
| measured floor rule | `floor_raised` | 0; more means the measured tool came within 4 mm of the touch height |
| lowest TCP | `min_tcp_z` against `table.touch_tcp_z` | about 6 mm above it at the grasp |
| close attempts | `attempts` | 1 for a clean grasp; more means the policy reopened and tried again |
| the cube came along | the video | `lifted` alone does not prove it |

Three attempts without an automatic stop or a strike mean the pipeline is
trusted. A lifted cube in at least one of them is the first grasp. A stop or
a strike sends you back to the dry run and the shadow episodes, with the
episode logs.

## 9. Known limits

- **The gap is modelled.** The A1X reports no jaw position. The gap follows
  the streamed opening at the measured jaw speed. In training the gap stopped
  at the object; on the real arm the estimate stops only when the backend's
  effort freeze fires. The policy may reopen and retry with the cube already
  between the plates.
- **The contact cue is unproven.** The freeze fires on a gripper effort above
  a threshold, and one free jaw move already exceeded the default 1.2. The
  gripper log decides whether free and loaded closes separate.
- **The camera pose is good to about 1 cm and 2 degrees.** Training jittered
  it by 3 mm and 0.7 degrees. The dot check decides on the day.
- **The intrinsics are not trusted.** 15 views, 0.85 px RMS.
- **Latency.** Frame age, joint age, inference and the twin loop add up,
  against a policy with no memory that trained with an instant response. The
  limit is 150 ms. The remedy, if needed, is retraining with an action delay.
- **A STOP from the twin itself** (a fault) stops the jaw one loop period
  later than the arm. The twin's STOP holds the last gripper command; the
  driver then re-holds the jaw at the streamed opening.
- **Table clearance** rests on the taught touch, the sag estimate and the
  measured-height rule. None of them sees the table. A table that moved or
  is not flat since teaching is not detected. Your hand on Enter is the last
  guard.
- **No success detection on the real arm.** `lifted` is a convenience ending.
- **Joint 5.** Training's joint 5 had a standard deviation of 0.65 degrees.
  A zero offset on the real joint shifts every observation. The run refuses
  more than 1 degree at the hover (`j5 offset`).
- **The other-sender check** sees only the frames the CAN adapter delivers to
  this process. Whether the XCAN adapter on macOS echoes frames is unknown.
- **The gripper over XCAN on macOS is unverified**
  ([TWIN.md](TWIN.md#galaxea-a1x), open questions). Run the first attempts on
  Linux with SocketCAN.
- **The hover comes from a taught point only.** The cube detector and the
  glasses do not feed it yet.

## 10. Troubleshooting

| symptom | cause and fix |
|---|---|
| a message naming `lerobot-twin grasp-model fetch` | No checkpoint found. Run `lerobot-twin grasp-model fetch`, or give `--checkpoint` or `$LEROBOT_GRASP_STUDENT`. |
| the model is refused | The file's sha256 is not the published one, or its schema is wrong. Fetch again, or set `LEROBOT_GRASP_SHA256` for a checkpoint you trust. |
| `not a1x` | The workspace file has another robot. Use an A1X workspace (`--workspace`). |
| `enable on connect` | `a1x.enable_on_connect` is true in the config. Set it to false and enable the arm from the panel. |
| a per-rig file error (`nieznane pola`, `wersja`, `oczekiwano ...`) | `workspace/grasp.json` is malformed. The message names the key. |
| `table not taught`, `grasp point not taught` | Bench steps 2 and 3. |
| `table touch implausible` | The touch height is more than 15 mm from 33.9 mm: check the gripper mount, the kinematics and the table. `overrides.table` lifts it. |
| `table touch far` | The table was taught more than 10 cm from the grasp point. Teach it at the cube spot. |
| `teaching old` | Teaching is older than 12 hours or has no stamp. Teach again, or write `overrides.taught`. |
| `object height not set` | `grasp.object_height` is missing or outside 5 mm to 10 cm. Teach the point with `--object-height`. |
| `wrist camera not set` | Set `wrist.source` (bench step 4). |
| `dot not checked`, `dot off` | Bench step 5. Over 8 px: calibrate the camera on the hand again. `overrides.dot` lifts it. |
| `latency not measured`, `latency high` | Bench step 6. Over 150 ms: look at frame age and inference in the dry-run summary. `overrides.latency` lifts it. |
| `jaw stroke not measured` | Bench step 1. |
| `gripper freeze undecided` | Record the decision with `grasp-bench gripper --set-freeze`. |
| `gripper freeze differs` | The running `a1x.gripper_force` is not the recorded one. Run with the same `--config`. |
| `no shadow episodes` | Write `overrides.shadow_done` after ten clean shadow episodes. |
| `table not under grasp` | The workspace table does not cover the believed grasp with 8 cm to spare. Fix the table in `workspace/twin.json`. |
| `joint offsets differ` | `a1x.joint_offsets_deg` are not zero. See [the joint offset rule](#the-joint-offset-rule). |
| `joint offsets changed` | The offsets changed since the point was taught. Teach the point again. |
| `outside training` | The hover is outside training's envelope. Move the cube to the arm's left, about 0.33 m out. `--outside-training` runs anyway. |
| `outside volume` | The workspace's `reach` box does not contain the hover. Widen it or remove it. |
| `ik` before any motion | No hover solution with joint 5 near zero and room on joint 6. Turn the cube's yaw, or move it. |
| `ik` during pre-positioning | A step of the straight-up retreat was refused while the tool was still near the object. Bring the arm to the home pose and start again. |
| `wrist camera silent` | No frame within 3 s. Check `wrist.source`, close any other process that holds the camera, and on macOS start from Terminal.app. |
| `camera` | The frame is older than 0.5 s, the stream died, the frame is not 1920 x 1080, or frames keep coming late. |
| `busy` | Another owner in this process, or another sender on the CAN link. Close the panel, the jog tool, the bridge, any ROS driver. |
| `stop` refusal | A STOP is active in the twin, for example after a fault right after connecting. The log names the fault. Fix it and start the command again. |
| `not armed` | `sim` on a real backend, `shadow` or `real` without `--i-know`, or a backend without a gripper state. |
| `speed limited` | A speed limit under 90 degrees per second: `safety.velocity_scale`, a speed cap, or `a1x.max_step_deg_s` in the config. |
| `not arrived` | The arm did not reach the hover within 3 s. The detail lists the misses. A large z miss means a sag over 8 mm. |
| `j5 offset` | Measured joint 5 is more than 1 degree from zero at the hover. Check joint 5's zero. |
| `collision` during pre-positioning | The staged path hits the table or the arm. Start from the home pose. |
| `impact lag <joint>`, `impact effort <joint>` | The arm met something, or the effort limit is too tight: compare the dry-run efforts with `guard.effort_abs`. |
| `lag` | The arm fell more than 8 degrees behind for 0.5 s. Look for rate limiting and speed caps. |
| `table` | The tool reached the touch height. Teach the table again and check the sag in the report. |
| `stale feedback` | CAN feedback or joint samples are late. Check the link (`ip -det -s link show can0`) and the machine's load. |
| `pairing` | No joint sample near the frame's exposure time. Check `latency.camera_s`. |
| `non-finite` | A NaN or infinity reached the observation, the action or the target. The log has the tick. |
| `error` | An exception in the driver. The log has the traceback. |
| STOP with `inny nadawca na laczu CAN` | Another process sent command frames. Close it. |
| the gripper log says `freeze active` | Set `a1x.gripper_force: 100` for the measurement. |
| the gripper log says `arm outside limits` | Press Dom in the panel first, or bring the arm to the hover with `grasp-bench point`. |
| the gripper log samples under 150 Hz | The machine is busy. Close other programs and repeat. |

## 11. Tests

```bash
export LEROBOT_GRASP_STUDENT=<path to student.pt>      # optional: `grasp-model fetch` puts it where the tests look
.venv/bin/python -m pytest -q tests/test_twin_grasp_*.py \
    tests/test_twin_kinematics_step.py tests/test_a1x_gripper_state.py
```

Tests that need the weights skip without them. None of them open a CAN link,
a serial port or a camera. `GRASP_FIXTURES=<dir>` points them at another
fixture directory.

- `tests/test_twin_grasp_contract.py`: the constants equal `meta.json` of the
  recording; `grasp_R`, `yaw_of`, `wrap` against hand values.
- `tests/test_twin_grasp_command.py`: 1000 bookkeeping ticks from training
  with the recorded IK result injected; box, floor and ceiling hits; a
  refused IK; a raised floor lifts the next proposal.
- `tests/test_twin_grasp_rig.py`: the per-rig file round trip, unknown keys,
  the refusals per mode in order, overrides, teaching age, the table touch
  distance.
- `tests/test_twin_grasp_keys.py`: a fake stdin and clock. A bare Enter
  confirms before the arm may move and is STOP after; only `d` and Enter
  typed after the prompt and outside the 1 s window goes on.
- `tests/test_twin_kinematics_step.py`: the two arm models agree; `ik_step`
  reproduces training's IK and its `ok` flag; `converged` is training's flag;
  joint limits; timing.
- `tests/test_a1x_gripper_state.py`: the gripper state on the fake CAN bus:
  stream speed, freeze, stale effort, no side effects; a foreign 0x050 frame
  raises the count, puts the fault in `faults()` and ends in STOP; an echo of
  the backend's own frame does not.
- `tests/test_twin_grasp_runtime.py`: `set_target` refuses NaN and infinity;
  the scene backend's gripper state; `speed_limits()`;
  `hold_measured(gripper=...)` stops a closing jaw, also under STOP.
- `tests/test_twin_grasp_scene.py`: the hand camera sits where the
  kinematics say; a rendered dot lands where the projection says; jitter
  changes the image and is undone.
- `tests/test_twin_grasp_camera.py`: the projection against training's
  pixels; the undistortion and 180-degree rotation on a synthetic pattern;
  a fake camera opener.
- `tests/test_twin_grasp_policy.py`: the network's weight names and shapes;
  the heat channel; with the checkpoint: hash, safe load, a corrupted copy
  refused, 59 recorded ticks reproduced within 1e-5, `LEROBOT_GRASP_SHA256`,
  the download refusing plain HTTP and a wrong hash; timing.
- `tests/test_twin_grasp_observation.py`: the 19 numbers against training's
  at the recorded joints; the gap estimator; the hover solution and the
  envelope.
- `tests/test_twin_grasp_driver.py`: the simulated twin with a scripted
  policy: every ending, ownership, holding, the floors at the box corners,
  the collision check near the base, the second episode's straight-up
  retreat, pairing with camera latency, dry mode, speed caps, the effort rule.
- `tests/test_twin_grasp_driver_a1x.py`: the fake CAN bus: dry mode sends no
  frame; shadow never sends an opening under 99; every STOP path stops a
  closing jaw; a stalled joint trips the guard before the backend's deaf
  fault; stale feedback; another sender during the listen gives `busy`.
- `tests/test_twin_grasp_run.py`: the refusals before connecting, the
  in-memory workspace, `enable on connect`, the evaluation on the literal sim
  backend, a sim run, Ctrl-C, an early Enter after an episode, the dry-run
  dump, a short evaluation; with the checkpoint, 6 episodes with at least 3
  successes.
- `tests/test_twin_grasp_bench.py`: the gripper log on the fake bus; the
  freeze and limit refusals; teaching the table by touch in simulation; the
  point's stored values; latency; `show`.
- `tests/test_twin_grasp_cli.py`: help, defaults and dispatch of the four
  commands.

### The fixtures

`tests/fixtures/grasp/` holds values recorded from the training environment
(about 1.3 MB): 59 ticks the student drove, 1000 bookkeeping ticks, 300
projections, 300 forward-kinematics poses, 200 hovers, 200 IK steps, the heat
channel, the weight names and `meta.json`. Its `README.md` describes each
file and the 2 ms substep lag that the `*_fk` fields correct for.

A test that finds a key missing fails with "re-record the fixtures
(docs/GRASP.md)". Re-record when the training environment, the camera
contract or the published model changes. The recorder imports the training
code from galaxeo-manipulators and writes nothing there. Run it with the
training repository's own environment, from a scratch directory:

```bash
mkdir -p <scratch>/fixtures_out && cd <scratch>/fixtures_out && PYTHONDONTWRITEBYTECODE=1 \
  <galaxeo>/sim/.venv-lerobot/bin/python <twin>/scripts/record_grasp_fixtures.py \
  --galaxeo <galaxeo> --out <scratch>/fixtures_out \
  --seed 50000 --shard-episodes 2 --shard-ticks 40 --book 1000 --proj 300 --fk 300 --hover 200 --ik 200
cp <scratch>/fixtures_out/*.npz <scratch>/fixtures_out/*.json <twin>/tests/fixtures/grasp/
```

`--galaxeo` defaults to `$GALAXEO_REPO`. After re-recording, run the contract
test: a changed camera pose, lens or step in `meta.json` must be copied into
`grasp/contract.py` too.
