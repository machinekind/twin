# Grasp student

A published policy (`hvsr-robotics/a1x-wrist-grasp-sim`, `student.pt`) that does the last
centimetres of a top-down grasp on the Galaxea A1X from its wrist camera: from a hover 6 cm
above a grasp point until the object is lifted, 20 ticks a second. The twin runs it as the
owner `chwyt`, through `Twin.set_target`, the supervisor and the A1X backend. The grasp point
is taught by touch on the bench. Code: `src/lerobot_mp/twin/grasp/` (each module's docstring
explains its part). Status: tested in the twin's simulation, on a fake CAN bus and against
fixtures recorded from training. **Nothing has moved the real arm yet.**

## Before anything moves

- **One CAN sender, one camera process.** Close the panel, `jog_a1x.py`, `so101_bridge.py`,
  any ROS driver and any other twin process. On Linux the grasp commands listen 0.3 s before
  moving and refuse with `busy` on a foreign command frame; during a move such a frame is a
  STOP (`inny nadawca na laczu CAN`). On macOS the XCAN adapter admits one process only.
- **`a1x.enable_on_connect: false`**, or every command refuses with `enable on connect`.
  If the arm does not follow the panel, enable it there (**Wlacz silniki A1X (FF 1-5-6)**),
  press **Dom**, then **Rozlacz** and close the panel. The arm keeps holding.
- **On macOS launch from Terminal.app.** Elsewhere the camera fails with
  `not authorized to capture video`.
- **The cube goes on the arm's left**, about x = 0, y = 0.33 m. Training only saw objects at
  x -0.20..0.20 m, y 0.22..0.40 m; a hover outside joint 1 40..139 degrees, radius 0.22..0.44 m
  or hover z 0.05..0.14 m is refused (`outside training`).
- **The workspace table must cover the grasp point with 8 cm to spare** (`table not under
  grasp`), so for y = 0.33 m it must reach y = 0.41 m. Neither the default A1X table nor the
  demo one does. Set it in the panel (tab **Ramie**, **Stanowisko (stol)**) or in `table` of
  `workspace/twin.json`. The workspace must be an A1X one (`not a1x`).
- **Keep the wrist camera out of `workspace/twin.json`'s cameras**; the grasp commands open it
  from `wrist.source` in `workspace/grasp.json`.
- **Use the same `--config` for every grasp command**: the run compares the running joint
  offsets and gripper freeze threshold with what the bench recorded.
- **The A1X has no brakes and does not yield on contact.** A bare Enter is STOP from the moment
  the arm may move until the command ends; Ctrl-C too. STOP holds the pose; it never cuts the
  motors.

## Get the model

```bash
lerobot-twin grasp-model fetch    # HTTPS download to workspace/policies/grasp/student.pt, sha256, camera spec
lerobot-twin grasp-model check    # hash, safe load, schema, statistics
lerobot-twin grasp-model time     # inference time; exit 1 over 30 ms median or 45 ms p95
```

Lookup order: `--checkpoint`, `$LEROBOT_GRASP_STUDENT`, `workspace/policies/grasp/student.pt`.
Every load checks the sha256; no run mode downloads. Another checkpoint needs
`LEROBOT_GRASP_SHA256=<its sha256>`. `fetch` exiting 1 on a camera spec mismatch means: do not
run the model until the contract is checked.

## The first bench day, in order

Every moving bench tool asks for Enter before it moves (`not confirmed` without it), moves at
20 degrees per second, and then treats a bare Enter as STOP. On the simulated backend (the
default) the tools write only to a file named with `--rig`, so rehearsals never fill
`workspace/grasp.json`. Put the tool name right after `grasp-bench`. Run
`lerobot-twin grasp-bench show` between the steps: it prints the file, the teaching ages and
what each mode still misses (`nic - moze ruszac` when nothing). The commands use Linux's
`--port can0`; on macOS it is `--port xcan`.

**0. Prepare.** Power the arm, enable it from the panel if needed, **Dom**, **Rozlacz**. Check
`a1x.enable_on_connect: false` and all `a1x.joint_offsets_deg` zero. Tape the cube spot.

**1. Gripper.** Set `a1x.gripper_force: 100` for this step (otherwise `freeze active`).

```bash
lerobot-twin --config configs/local.yaml grasp-bench gripper --backend a1x --port can0 --i-know
```

The jaw moves 99, 100, then 0, 100, 0, 100 empty, 2 s per move; type `s` + Enter when you see
the jaws stop. At the prompt put a 3 cm cube between the plates and type `d` + Enter for a
loaded close. Gate: every move valid, stream stroke about 1.3 s, at least 150 Hz sampling (else
nothing is written, exit 1). Then record the freeze decision: the printed suggested threshold
when `separable` is true, else 100 (freeze off). Put the same value into `a1x.gripper_force`
and:

```bash
lerobot-twin grasp-bench gripper --set-freeze 100          # or the threshold you chose
lerobot-twin grasp-bench gripper --gap-mm 0:3 50:53 100:103  # ruler readings at openings 0, 50, 100
```

A `UWAGA: rozwarcie ...` warning (more than 3 mm off 3 mm + 1 mm per unit) means the jaw's zero
or span is wrong: check `a1x.gripper_closed_pdes` and `a1x.gripper_open_pdes` first.

**2. Table touch**, at the cube spot, before the cube is there:

```bash
lerobot-twin --config configs/local.yaml grasp-bench table --backend a1x --port can0 --i-know \
    --xy 0.0 0.33 --yaw-deg 0
```

The arm goes to 0.12 m above the point. Jog with lines ended by Enter: `d MM`/`u MM` (down, up,
at most 5 mm; 1 mm below TCP z 0.06 m), `x MM`/`y MM` (at most 5 mm), `r DEG` (at most 5),
`g N` (jaw 0..100), `ok` (store), `q` (quit without saving); an empty line is STOP. Go down 1 mm
at a time. On `Kontakt prawdopodobny` the tool backs up 2 mm: `d 1` twice, then `ok`. Gate: the
stored touch within 15 mm of 33.9 mm (else `Nie zapisano`; `--force` saves and writes
`overrides.table`).

**3. Grasp point.** Cube on the spot, one face across the closing axis:

```bash
lerobot-twin --config configs/local.yaml grasp-bench point --backend a1x --port can0 --i-know \
    --xy 0.0 0.33 --yaw-deg 0 --object-height 0.030
```

Jog (`g 100` opens) until the open plates straddle the cube and the fingertips touch the table;
`r DEG` squares the plates to the cube. `ok`. Note the printed TCP and yaw in degrees. Gate:
`Zawis w obszarze treningu`. The table touch must be within 10 cm of this point
(`table touch far`).

**4. Wrist camera.** Set `wrist.source` in `workspace/grasp.json` (macOS: an index such as
`"2"`; Linux: `/dev/v4l/by-id/...-video-index0`). It must deliver 1920 x 1080. Keep
`wrist.upside_down: true`. Gate: `show` no longer lists `wrist camera not set`.

**5. Dry run and the dot check.** Dry mode never moves the arm, so park it at the hover first
and quit the tool with `q` (it holds the pose, exit 1):

```bash
lerobot-twin --config configs/local.yaml grasp-bench point --backend a1x --port can0 --i-know \
    --xy <taught x> <taught y> --yaw-deg <taught yaw> --start-z 0.10
lerobot-twin --config configs/local.yaml grasp --mode dry --backend a1x --port can0 --seconds 20
```

Repeat at `--start-z 0.08` and with `--xy` 2 cm off. Every 20 ticks (`--dump-every`) the dry
run writes `workspace/grasp/dry/<stamp>/NNNN.png` (the policy frame, 4 times enlarged),
`NNNN_heat.png` and `ticks.jsonl`. In the PNG the **small square** is where the camera model
expects the centre of the cube's top face. The cross (with its 8 px circle) is the believed
grasp at TCP height, about 1 cm above the top face, so it lies a few pixels away from the
square even with a perfect camera: use it to see where the policy is told to go, not to
measure. **Measure the distance between the square and the real top-face centre**, in pixels
of the 160 x 120 frame (the PNG divided by 4), and record the largest:

```bash
lerobot-twin grasp-bench dot --error-px 5.5
```

Gate: at most 8 px (else `dot off`; the fix is a new hand-camera calibration). The summary
`Podsumowanie dry` must show frame age p95 under 120 ms, inference median under 30 ms and p95
under 45 ms, no stale ticks and no non-finite action. It also logs the arm efforts at rest
(see [the effort limit](#the-effort-limit)).

**6. Latency.**

```bash
lerobot-twin --config configs/local.yaml grasp-bench latency --backend a1x --port can0 --i-know
```

The arm goes to the hover and steps joint 6 by 3 degrees five times at full speed. Gate:
`total_s` (camera + inference p95 + actuation) at most 150 ms (else `latency high`); at least
three good steps (else nothing is written, exit 1). It also prints the joints' peak efforts over
the steps (`Szczyt |wysilku| ...`, see [the effort limit](#the-effort-limit)). After this `show`
lists nothing for shadow.

**7.** [Shadow episodes](#shadow-ten-episodes), **8.** [the first real grasp](#first-real-grasp-three-attempts).

**Every day again:** teaching expires after 12 hours (`teaching old`), so repeat steps 2 and 3
and the step 5 dump at the new point (a new `--error-px` only if it got worse).

## workspace/grasp.json

Next to `workspace/twin.json`, not in git; `--rig PATH` picks another file. Each tool re-reads
the file and changes only its own keys, so your edits survive. Unknown keys and another
`version` are errors. Sim mode and the simulated backend need nothing from it; `dry`, `shadow`
and `real` on a real backend refuse without:

| key | how you get it | refused without it in |
|---|---|---|
| `table.touch_tcp_z`, `.tcp`, `.joints`, `.taught_at` | `grasp-bench table` | dry, shadow, real |
| `grasp.tcp`, `.yaw` (rad), `.joints`, `.taught_at`, `.joint_offsets_deg` | `grasp-bench point` | dry, shadow, real |
| `grasp.object_height` | `grasp-bench point --object-height` (0.005..0.10 m, default 0.030) | shadow, real |
| `wrist.source` | you, by editor | dry, shadow, real |
| `wrist.K`, `wrist.dist` | you, both from one calibration of the upright 1920 x 1080 image; null uses the 2026-09-26 default (15 views, 0.85 px RMS) | never |
| `wrist.dot_error_px` | `grasp-bench dot --error-px` | shadow, real |
| `latency.camera_s`, `.actuation_s`, `.infer_p95_s`, `.total_s` | `grasp-bench latency` | shadow, real (`total_s`) |
| `gripper.stroke_close_s_effort`, `.stroke_close_s_operator` | `grasp-bench gripper` | real |
| `gripper.stroke_open_s_*`, `.free_effort_peak`, `.loaded_effort_peak`, `.separable`, `.suggested_threshold` | `grasp-bench gripper` | never; they inform the freeze decision |
| `gripper.gap_mm` | `grasp-bench gripper --gap-mm` | never |
| `gripper.freeze_threshold` | `grasp-bench gripper --set-freeze` (100 = off) | real |
| `guard.effort_abs` | you, from the logged peaks (default 12.0) | never |
| `overrides.shadow_done` | you, a note after ten clean shadow episodes | real |
| `overrides.table`, `.dot`, `.latency`, `.taught`, `.joint_offsets` | you, a note saying who and why | each lifts its limit |

### The effort limit

`guard.effort_abs` (default 12.0) is the impact guard's per-joint effort limit on the real
arm, next to its lag rule. The dry run logs the efforts at rest (`arm efforts (dry)`). Each
pre-positioning, each episode and the latency steps log the per-joint peak |effort| of the move
(`arm effort peak (pre-positioning)`, `(episode)`, `(latency steps)`); `grasp-eval` keeps
it per episode as `effort_peak`. Set the limit to about 1.5 times the largest peak in motion,
never below the at-rest maximum. `impact effort <joint>` with no contact on the video means it
is too tight. A joint above half its limit at rest gets only the lag rule for that move
(`bez reguly wysilku: ...`).

## Running

`lerobot-twin grasp --help` lists every flag. Modes, in order on the rig:

| mode | what moves | needs |
|---|---|---|
| `dry` | nothing: no claim, no target | table, point, `wrist.source` |
| `sim` | the simulated arm with a cube in the twin's scene | a simulated backend (else `not armed`) |
| `shadow` | the real arm follows the policy, jaw held at 100, floor 10 mm above the cube's top | `--backend`, `--i-know`, the shadow keys |
| `real` | the real arm, live jaw | `--backend`, `--i-know`, every key, `overrides.shadow_done` |

On the simulated backend the cube is drawn from `--seed` (default from the clock, printed at
start as `Symulacja: ziarno N`). `shadow` and `real` rehearse on the simulated arm with
`--backend sim --i-know`. On a real backend the command prints the teaching ages and asks for
Enter before the arm goes to the hover. `--floor`, `--gap` and `--jaw-stroke-s` are for the
simulated arm only.

An episode pre-positions at 20 degrees per second (the jaw opens at the hover), then runs at
most 120 ticks or 12 s at up to 90 degrees per second. It prints `Epizod k/N: <ending>` and a
line of numbers the checklists read; the per-tick log is
`workspace/grasp/runs/<stamp>/episode_<n>.jsonl`.

**Next episode on a real arm:** take the cube out of the jaws, put it back on its spot, then
type `d` + Enter at `d + Enter = kolejny epizod ...`. Anything else, including an Enter within
1 s of the episode's end, is STOP and ends the series. On a real backend every ending except
`lifted`, `tick cap`, `time cap` and `success` ends the series with exit code 1.

Exit codes: 0 done, 1 failure (connect, camera, a guard ending on a real arm, Ctrl-C), 2 refusal.
A STOP during pre-positioning or an episode exits 1, the same as Ctrl-C; a STOP at the
`d + Enter` prompt ends the series with 0.

## Checklists

### Shadow: ten episodes

Before: `show` lists nothing for shadow, the cube on its spot, a camera recording the arm, a
rehearsal on the simulated arm (`--backend sim --i-know`).

```bash
lerobot-twin --config configs/local.yaml grasp --mode shadow --backend a1x --port can0 --i-know --episodes 10
```

| check | printed as | pass |
|---|---|---|
| no automatic stop | the ending | `tick cap` or `time cap` |
| tracking | `max opoznienie` | under 8 degrees (the lag skip threshold); clean sim episodes reach 7.7 |
| clearance over the cube | `min TCP z` minus `table.touch_tcp_z` minus `grasp.object_height` | 5 mm or more (commanded 10 mm, full-speed tilt and sag take a few) |
| measured floor rule | `podloga podniesiona` | 0x |
| aiming | `brzeg pudelka` and the video | few box hits; the tool descends and yaws towards the cube. Many hits: check the dot again |
| rate limiting | `ogr. predkosci ... % tikow` | under 10 % |

After ten clean episodes write `"shadow_done": "10 clean episodes <date>, <initials>"` in `overrides`.

### First real grasp: three attempts

Before: ten clean shadow episodes, the freeze decision matching the running
`a1x.gripper_force`, teaching younger than 12 hours, a camera recording, a hand on Enter.

```bash
lerobot-twin --config configs/local.yaml grasp --mode real --backend a1x --port can0 --i-know --episodes 3
```

| check | printed as | pass |
|---|---|---|
| no stop, no strike | the ending | `lifted`, `tick cap` or `time cap`; never `table`, `impact ...` or `lag` |
| measured floor rule | `podloga podniesiona` | 0x or not printed |
| lowest TCP | `min TCP z` against `table.touch_tcp_z` | 4 to 6 mm above it |
| close attempts | `proby` | 1 for a clean grasp |
| the cube came along | the video | `lifted` only means the closed jaw rose 4 cm |

Three attempts without a stop or strike mean the pipeline is trusted; a lifted cube in one is
the first grasp. A stop or strike sends you back to the dry run and shadow, with the logs.

## Refusals and endings

The message names the code. Refusals come before anything moves (exit 2, a silent camera 1); endings end an
episode, and after an ending the arm holds (the measured pose after `impact`, `lag`, `table`).

| code | what to do |
|---|---|
| `table not taught`, `grasp point not taught`, `table touch far` | steps 2 and 3; the table within 10 cm of the point |
| `table touch implausible` | more than 15 mm from 33.9 mm: check the gripper mount, kinematics, table; `overrides.table` |
| `teaching old` | older than 12 h or unstamped: teach again, or `overrides.taught` |
| `object height not set`, `wrist camera not set` | step 3 with `--object-height` (5 mm to 10 cm); step 4 |
| `dot not checked`, `dot off` | step 5; over 8 px recalibrate the hand camera; `overrides.dot` |
| `latency not measured`, `latency high` | step 6; over 150 ms read frame age and inference in the dry summary; `overrides.latency` |
| `jaw stroke not measured`, `gripper freeze undecided` | step 1, then `--set-freeze` |
| `gripper freeze differs` | the running `a1x.gripper_force` is not the recorded one: same `--config` |
| `no shadow episodes` | `overrides.shadow_done` after ten clean shadow episodes |
| `table not under grasp`, `not a1x`, `enable on connect` | see [Before anything moves](#before-anything-moves) |
| `joint offsets differ` | the camera was calibrated on raw joints: zero `a1x.joint_offsets_deg` (0.05 degree tolerance), or `overrides.joint_offsets` (under 0.3 px for a touched point) |
| `joint offsets changed` | the offsets changed since the point was taught: teach it again (no override) |
| `outside training` | cube on the arm's left, about 0.33 m out; or `--outside-training` |
| `outside volume` | the workspace's `reach` box excludes the hover: widen or remove it |
| `wrist camera silent` | no frame in 3 s: `wrist.source`, other camera users, Terminal.app on macOS |
| `busy` | another owner or CAN sender: close the panel, jog tool, bridge, ROS driver |
| `stop` before moving | a STOP is active, e.g. a fault after connecting; the log names it |
| `not armed` | `sim` on a real backend, `shadow`/`real` without `--i-know`, or no gripper state |
| `not confirmed` | a bench tool got no Enter, or no terminal |
| `speed limited` | a limit under 90 deg/s: `safety.velocity_scale`, a speed cap, `a1x.max_step_deg_s` |
| `ik` before moving | no hover solution with joint 5 near 0 and room on joint 6: turn or move the cube |
| `ik` in pre-positioning, `collision` | the retreat or the staged path failed near the object or table: Dom, start again |
| `not arrived` | hover not reached in 3 s; the detail lists misses; a large z miss means sag over 8 mm |
| `j5 offset` | joint 5 more than 1 degree from 0 at the hover: check its zero |
| `freeze active`, `arm outside limits` | gripper tool: `a1x.gripper_force: 100`; press Dom first |
| `success`, `lifted` | sim: cube lifted between the jaws; real: jaw closed and TCP up 4 cm, check the video |
| `tick cap`, `time cap` | 120 ticks or 12 s; normal in shadow |
| `stop` | Enter, Ctrl-C, or a twin STOP (fault, foreign CAN sender, deaf joint, tracking stop) |
| `preempted`, `not connected` | something else took the arm; the arm disconnected |
| `impact lag <joint>`, `impact effort <joint>` | contact, or the limit too tight: [the effort limit](#the-effort-limit) |
| `lag` | 0.5 s over 8 degrees behind: look for rate limiting and speed caps |
| `table` | the measured tool reached the touch height: teach the table again, check `ugiecie` |
| `stale feedback` | samples older than 0.3 s: the link (`ip -det -s link show can0`) and machine load |
| `camera`, `pairing` | frame older than 0.5 s, stream died, not 1920 x 1080; no joint sample near the exposure time (`latency.camera_s`) |
| `non-finite`, `error` | the episode log has the tick or traceback |

## Sim evaluation

```bash
lerobot-twin grasp-eval                                # 50 episodes, seed 50000, simulated jaw
lerobot-twin grasp-eval --jaw measured                 # jaw slowed to the measured stroke (else 1.3 s)
lerobot-twin grasp-eval --floor training --gap scene   # training's floor and the scene's gap
```

Always the simulated backend. The result goes to `workspace/grasp/eval/<stamp>.json` with the
episode logs beside it. Verdict at the simulated jaw: 70 % or more is good; 50 to 70 %, read
the endings and logs before changing anything; under 50 %, run
`tests/test_twin_grasp_policy.py`, `test_twin_grasp_observation.py` and
`test_twin_grasp_camera.py`, then rerun with `--floor training --gap scene` to separate the
twin's floor and gap model from contact physics and the render. At the slowed jaw, under 50 %
calls for retraining with a slewed jaw and a lagging gap observation. Tick p99 must be under
40 ms. Never raise a driver threshold to raise the score.

Results on this branch, 2026-09-27, seed 50000, checkpoint sha256 `05e5aaf6...cdcb8536`, no
per-rig file:

| run | rate | endings | tick p99 | inference median / p95 |
|---|---|---|---|---|
| `--jaw sim` | 46 of 50 (92 %) | success 46, tick cap 4 | 32.3 ms | 10.1 / 12.7 ms |
| `--jaw measured` (1.3 s) | 0 of 50 (0 %) | tick cap 50 | 22.2 ms | 9.5 / 12.3 ms |

The slowed jaw fails every episode, so the retraining above is due before the first real grasp.

## Not built, and known limits

- **No jaw position.** The gap is modelled from the streamed opening and stops only when the
  effort freeze fires, so the policy may reopen with the cube between the plates.
- **No success detection on the real arm.** `lifted` is a convenience ending.
- **The camera pose** (about 1 cm, 2 degrees; training jittered 3 mm, 0.7 degrees) and the
  intrinsics are not trusted; the dot check decides. Latency up to 150 ms meets a memoryless
  policy trained with an instant response.
- **A table moved since teaching is not detected.**
- **About 10 % of simulated ticks are refused by the 6-degree joint jump rule**; training took them.
- **XCAN on macOS**: the other-sender check and the gripper are unverified there. Run the first
  attempts on Linux with SocketCAN.
- **The hover comes only from a taught point**; the cube detector and glasses do not feed it.

## Tests

```bash
.venv/bin/python -m pytest -q tests/test_twin_grasp_*.py tests/test_twin_kinematics_step.py \
    tests/test_a1x_gripper_state.py
```

Tests needing the weights skip without them; none opens a CAN link, serial port or camera.
Fixtures and re-recording: [tests/fixtures/grasp/README.md](../tests/fixtures/grasp/README.md).
