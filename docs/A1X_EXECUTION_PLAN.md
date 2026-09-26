# A1X + G1 hackathon execution plan

Updated: 2026-09-26. Status: **execution active; offline validation first**.

This is the working plan for the current Codex task. Update it as work progresses;
keep completed evidence, unresolved questions, and the next action explicit.

## Objective and scope

Use Przemek's existing `a1x-twin` setup to obtain reliable A1X + G1 control,
validate simulation and GPU training, then demonstrate a learned manipulation task
on the physical setup. Reuse the team's work and existing camera calibration.

Proposed first manipulation task: pick up one small block, lift it about 6 cm, and
hold it still. The exact object and success criteria remain to be agreed with the
team. Reaching is the first technical checkpoint; pick-and-place follows only if
the lift is reliable and time permits.

The active goal authorizes implementing and validating this plan. The user has
deferred control-computer access and physical tests until later. Work on code,
fake transport, simulation, and local compute can proceed. Do not create or
update Linear issues for this project.

## Starting point and evidence

| Item | Current understanding | Evidence / qualification |
|---|---|---|
| Checkout | `a1x-twin`, commit `7edfceb`; clean before this plan | Locally checked 2026-09-26; recheck on resumption |
| Physical setup | A1X with G1, calibrated against two cameras watching the scene | User confirmed; setup photo supplied. Calibration accuracy and files have not been independently checked |
| Simulation | A1X geometry, six arm joints plus gripper control, IK, contacts, robot-specific limits and units | Inspected source and assets |
| RL | `reach`, `lift`, `pickplace`; CPU evaluation and MuJoCo Warp/PPO training code | State-vector observations; camera perception supplies object pose separately |
| Saved policies | Bundled `reach-v3` and `lift-v3` belong to SO-101 | No bundled A1X policy found; cross-robot use is rejected |
| Hardware integration | CAN backend, browser controls, staged homing, camera calibration | Latest commit reports physical calibration work; an end-to-end learned physical grasp is not established |
| Verification | Fake-CAN regression tests, CPU scripted grasps and bounded GPU training executed | See [offline evidence](A1X_OFFLINE_VALIDATION.md). Physical tests remain deferred |

The README and handoff contain historical claims. Some cited tests were deleted,
and the README's claim of no physical testing predates the latest hardware-session
commit. Neither documentation nor the photo is proof of end-to-end readiness.

## Prioritized findings to reproduce and resolve

These began as source findings. F1, F2, F4 and F5 were reproduced against the
original checkout with offline regression tests. Corrections and remaining
limits are recorded in [offline evidence](A1X_OFFLINE_VALIDATION.md); none of
these tests establishes physical readiness.

| ID | Finding | Required outcome |
|---|---|---|
| F1 | First gripper command: the supervisor bypasses an assumed-open ramp, but `robot/a1x.py` still initializes its own stream from the assumed opening when feedback is unavailable | Exercise the real A1X backend with a fake CAN bus. Starting while holding a card must not send an unintended opening command. Keep measured, commanded, and unknown positions distinct |
| F2 | A main control-loop exception exits the supervisor loop without clearly cancelling the independent CAN sender's old targets | Inject loop failures and stale commands while telemetry remains healthy. Cancel obsolete targets and stop further commanded travel without releasing motor holding torque; require deliberate recovery |
| F3 | Collision checks cover calibration and staged home, but are not common to RL and tag-directed movement | Validate every autonomous movement path against joint/workspace limits and prohibited collisions. Permit intended finger-object grasp contact. Reject invalid/NaN targets; do not rely on tracking error after impact |
| F4 | Default training gripper command can traverse its range in about 0.35 s; hardware's default ramp needs 1.3 s | Make the action timing and gripper limits consistent through training, runtime, and hardware. Measure actual response before claiming physical equivalence |
| F5 | Angular noise is added to all joint coordinates, including the A1X sliding gripper in metres; default 0.3 degrees becomes about 5.2 mm on that coordinate | Use explicit units and separate realistic noise scales; test both CPU and GPU observation paths |
| F6 | Grasp inference depends on measured opening, but default G1 feedback calibration is absent and the backend can report the command instead | Verify feedback endpoints and an intermediate opening. Do not treat commanded opening as measured contact or certify a physical grasp from it |
| F7 | A1X GPU compatibility/contact capacities are unverified; simulated finger coupling permits documented millimetre-scale errors; tests cited as evidence are missing | Restore only useful targeted checks. Measure contact stability and finger symmetry, compare CPU/GPU behavior, and reconcile claims with current evidence |

Primary code locations: `src/lerobot_mp/robot/a1x.py`, `twin/runtime.py`,
`twin/rl/{task,env,batch,runner}.py`, `twin/ui/app.py`, and
`assets/robots/a1x/{README.md,a1x.xml}` (all Python paths under `src/lerobot_mp/`).
The model XML is generated: change its generator if model changes are required.

## Execution stages

### 0. Preserve and identify the working setup

- [x] Re-read workspace instructions, this plan, current branch/diff, and relevant
  driver safety documentation. Check for concurrent team changes before editing.
- [ ] Identify the control computer, OS, CAN adapter, installed `galaxeo` version,
  and the exact code/configuration that produced the working camera calibration.
- [ ] Preserve camera intrinsics, extrinsics, device identities, arm offsets,
  gripper settings, and table/base geometry. Keep originals and dated copies.
- [ ] Record what has physically succeeded: jogging, G1 opening/closing, homing,
  calibration, grasping. Do not repeat calibration merely because it exists in the plan.
- [x] Identify available GPU/runtime and set a bounded local smoke-run budget. No cloud
  provisioning or paid compute without scoped authorization.

Exit: a reproducible baseline and a clear distinction between measured facts,
simulation defaults, and unknowns. Next-stage fixes can be developed offline.

### 1. Close control and gripper gaps offline

- [x] Reproduce F1-F3 and F6 using the actual backend through fake transport;
  include startup while holding an object, STOP, loop failure, telemetry loss,
  reconnection, and prevention of automatic motion resumption.
- [x] Implement minimal fixes and meaningful regression tests. Preserve the
  working SO-101 path and team changes; avoid a general architecture rewrite.
- [x] Align gripper timing and observation units (F4-F5).
- [x] Check that physical mode cannot consume simulated-camera observations or
  simulation-only calibration as evidence of a real object position.
- [x] Record passed, failed, and skipped checks. Fix documentation relevant to the
  execution path instead of restoring every deleted test.

Exit: focused tests demonstrate the actual command path and failure behavior.
Changes affecting the live setup are reviewed with its operator before use.

### 2. Establish an A1X simulation baseline on CPU

- [ ] Verify joint directions, offsets, units, limits, TCP location, and G1 geometry.
- [x] Exercise open/close, reaching, and a scripted block grasp/lift with collisions.
- [x] Measure finger coupling error, object slip, penetrations, and numerical
  failures. Finite numbers alone do not establish realistic grasp physics.
- [x] Run repeated seeded cases; keep a replay/video and per-attempt outcomes.
- [x] Exercise the policy runtime/supervisor path as well as the bare RL environment.

Exit: one reproducible scripted A1X block lift and trustworthy task success/failure
checks. Scripted performance is a baseline, not an RL result.

### 3. Verify GPU training before a full run

- [x] Confirm the selected runtime can load the A1X model in MuJoCo Warp.
- [x] Measure contact/constraint usage; check overflow and numerical failures.
- [x] Compare representative CPU/GPU trajectories, observations, rewards, resets,
  and success detection. Derive tolerances from observations, not guesses.
- [x] Run a bounded smoke test (256 environments, 3 iterations, 2 CPU
  evaluation episodes); save and reload its checkpoint. This checks plumbing only.

Exit: demonstrated A1X GPU compatibility, with logs and checkpoint replay.

### 4. Train and evaluate a narrow policy

- [x] Train an initial A1X `reach` candidate; evaluate on held-out seeds using CPU
  and the runtime path. Result is below demo quality; improvement remains necessary.
- [ ] Train A1X `lift` for the agreed object/workspace. SO-101 weights are not an
  A1X starting checkpoint. Freeze each experiment's scope, budget, and stop rule.
- [ ] Include measured actuator behavior and realistic perception delay/noise.
  Keep contact-force protection and other runtime restrictions consistent with training.
- [ ] Evaluate sustained holds, not momentary height/contact. Log failures and
  report learned-policy results separately from scripted demonstrations.

Proposed simulation acceptance: at least 18/20 held-out lifts, each with a 2 s
hold, and no prohibited collisions or numerical failures. Confirm these criteria
with the team before treating them as agreed requirements.

### 5. Validate on the physical two-camera setup

- [ ] Confirm the attending operator, fixed mounting, clear swept volume,
  hardware limits, and stop/power-cut procedure. The A1X has no holding brakes;
  cutting power drops the arm. Re-read the current driver safety guide.
- [ ] Verify camera-to-arm alignment at several measured points and G1 feedback
  calibration. Reuse existing calibration if it passes; recalibrate only as needed.
- [ ] Compare read-only measured state with the twin, then small low-speed
  movements, gripper behavior, and a scripted lift under operator supervision.
- [ ] Run the evaluated reaching policy, then the lift policy. Stop on stale
  observations, unexpected motion, tracking error, limit/collision rejection, or faults.
- [ ] Record target versus actual motion, latency/jitter, gripper state, and repeated
  successes/failures. Proposed demo acceptance: five consecutive supervised lifts
  with a 2 s hold; this is a hackathon checkpoint, not general reliability certification.

Exit: a repeatable physical demonstration with its exact configuration, checkpoint,
limitations, startup/stop instructions, and operator handoff recorded.

## Responsibility and scope limits

- Codex: code investigation/fixes, focused tests, simulation, training/evaluation
  when authorized and compute is accessible, evidence and handoff instructions.
- Team/operator: control-machine access, calibration/configuration provenance,
  physical measurements, supervision, object handling, and physical stopping.
- Defer broader perception models, arbitrary-object grasping, second-arm work,
  and pick-and-place until the narrow baseline succeeds.
- If RL does not become reliable in the available time, retain the verified
  scripted demo as an explicitly labelled fallback; do not present it as learned.
- Preserve vendor/upstream attribution. No remote publication or merging is part
  of merely writing or following the local plan.

## Progress and resumption

Current next action: **improve the learned reaching candidate and reevaluate it
through the common controller before training lift**. See [offline evidence](A1X_OFFLINE_VALIDATION.md)
for commands, versions, results and remaining gaps.

Selected available compute: local RTX 2000 Ada laptop GPU, 8188 MiB total,
7474 MiB free at the initial WSL check; Windows driver 560.76, Ubuntu on WSL 2.
Both 256-world, three-iteration smoke runs passed: peak total GPU usage was about
1.29 GiB for reach and 1.39 GiB for lift. Saved checkpoints reloaded and ran on CPU;
neither short run learned the task. Five scripted lifts held for two seconds on
both CPU and GPU, with zero reported capacity overflow. These are narrow,
measured workloads, not a promise that every training configuration fits 8 GB.
The adjacent driver checkout has concurrent team changes and must remain untouched.

| Date | Work / decision | Evidence | Next action |
|---|---|---|---|
| 2026-09-26 | Reviewed `a1x-twin` at `7edfceb`; identified F1-F7 | Static source/test/history inspection; no execution | Preserve live setup and reproduce prioritized findings |
| 2026-09-26 | User confirmed A1X + G1 calibrated against two scene cameras; requested a persistent plan | Current task and supplied setup photo | Execution not yet started |
| 2026-09-26 | Execution goal activated; user deferred physical access/tests and identified local GPU via WSL | WSL GPU query confirms device, memory, and driver above; no physical commands | Isolated dependencies, fake-CAN regressions, CPU simulation |
| 2026-09-26 | Isolated WSL environment installed; original backend regressions reproduced and corrected | `workspace/validation/baseline-regressions.log` (11 failures on original source); focused corrected runs pass | Finish full-suite verification and document control contract |
| 2026-09-26 | Scripted CPU lift succeeds on 5/5 seeded block positions | `workspace/validation/cpu-lift-005/`: traces, reports, seed-0 video; two-second holds, no prohibited contacts | Validate through runtime/supervisor |
| 2026-09-26 | GPU reach/lift smoke checks pass; scripted CPU/GPU comparison passes 5/5 | `gpu-reach-001`, `gpu-lift-001`, `cpu-gpu-comparison-001.json` under validation; comparison maximum cube drift 0.00381 mm, zero overflow | Broaden diagnostics and train a bounded reach candidate |
| 2026-09-26 | Full suite passes; scripted common-controller path passes 5/5 | `offline-suite-002.log`: 138 passed; `runtime-lift-004.json`: 2 s holds, no prohibited contacts or warnings | Final regressions and learned-policy evaluation |
| 2026-09-26 | First longer reach attempt fails on per-world constraint capacity, not VRAM | `reach-candidate-001/report.json`: 69 rows required vs 64 allocated; 1336.5 MiB peak total usage | Verify larger A1X reach allocation and collision termination, then rerun a bounded candidate |
| 2026-09-26 | Corrected training candidate completes 3,686,400 transitions; held-out performance remains insufficient | `reach-candidate-002/report.json`: 9/20 nominal CPU, 10/20 randomized CPU; `reach-runtime-003.json`: 6/20 common-controller successes, all 20 completed without collision/fault | Improve precision/stability with another explicitly bounded local experiment; do not deploy this candidate |
| 2026-09-26 | Current control/training changes verified, including GPU and render paths | `offline-suite-003.log`: 145 passed; six additional backend cases added afterward, all 14 backend-contract cases passed in `a1x-backend-contracts-final.log` | Preserve the local checkpoint and continue from these artifacts |

On resumption: check the current commit and team changes, read this log, and
continue from the first unfinished stage. After meaningful progress, update the
checkboxes and append the relevant commit, test command/result, or artifact path.
For runs, record code/dependency versions, configuration/calibration identity,
seed, resource/time limit, checkpoint, metrics, failures, and unresolved risks.
Do not mark a stage complete from documentation claims or skipped tests alone.

### Adjacent driver update from the offline knife-grasp task, 2026-09-26

The user reported `SyntaxError` at `<<<<<<< HEAD` while starting the adjacent
`galaxeo-manipulators/jog_a1x.py` with their Windows PCAN environment. That task
repaired the four conflict blocks in the working file, preserving local CAN-fault
and startup validation checks alongside the incoming protocol/status changes.
Four fake-transport/CLI regression tests were added to the two existing jog
tests; all six pass, as does `galaxeo.protocol.selftest()`. The original files
are backed up under `../reconstructions/scyzoryk/grasps/driver_repair/`.

Read-only access to `PCAN_USBBUS1` opens successfully but reports no `0x052`
joint feedback. A separate 3-second receive-only check observed four `0x001`
frames of length 4, PCAN status 0, and no application transmissions. This is
not a ready-for-motion result. Current operator, arm power/LED state and adapter
mapping still need confirmation. Other team changes and the existing Git merge
index were preserved; no merge commit was made. This does not complete any
digital-twin validation stage above.

Follow-up in the knife-grasp task: after the user powered the arm, read-only
`PCAN_USBBUS1` feedback succeeded at approximately 209-210 Hz. All seven status
groups reported the documented `0x0010` baseline and PCAN status was 0. A full
precision joint snapshot was saved to
`../reconstructions/scyzoryk/grasps/arm_feedback_snapshot.json` at
2026-09-26T16:30:23Z; all reads sent zero application frames. Connectivity is
confirmed; placement/orientation and physical motion checks remain outstanding.

The knife-grasp task subsequently received a setup photo and confirmation that
the base and knife share a tabletop. A provisional hardware-free IK study at
X=439.63 mm (390 mm plus the CAD housing radius), table Z=0, tested all 20 cached
grasps at four illustrative object yaws: 33/80 sampled descents passed. This
does not establish the real pose or a collision-free route out of the folded
arm. Mounting-plate versus housing measurement reference, base fastening and
metric object yaw remain unresolved. Evidence is in
`../reconstructions/scyzoryk/grasps/nominal_reach_probe.json`; no motor commands
were sent.

Knife-setup clarification: the user confirms secure base fastening and measures
390 mm from the rectangular mounting plate. The CAD base mesh does not include
that external plate; its front overhang and thickness are now requested. The
earlier 439.63 mm / table-Z=0 IK study remains explicitly provisional and must
not supply physical targets. Facts and unknowns are saved in
`../reconstructions/scyzoryk/grasps/placement_observations.json`.
