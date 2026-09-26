# A1X + G1 hackathon execution plan

Updated: 2026-09-27. Status: **execution active; offline validation first**.

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

- [x] Verify model joint directions, offsets, units, limits, TCP location, and G1
  geometry against the vendor URDF and generator. Physical agreement remains unmeasured.
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

- [x] Train and improve A1X `reach`; evaluate on held-out seeds using CPU and the
  runtime path. Development sets reached 19/20 through the guarded controller;
  a fresh frozen-policy holdout reached 20/20 with no actual contacts. This is
  simulation evidence only.
- [ ] Train A1X `lift` for the agreed object/workspace. SO-101 weights are not an
  A1X starting checkpoint. Freeze each experiment's scope, budget, and stop rule.
- [ ] Include measured actuator behavior and realistic perception delay/noise,
  plus persistent translation and rotation errors in camera-to-arm calibration.
  Keep contact-force protection and other runtime restrictions consistent with training.
- [ ] Measure success versus perception-error magnitude and report the achieved
  tolerance using true simulated object/contact state; follow the separate
  [perception robustness protocol](A1X_PERCEPTION_ROBUSTNESS.md).
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

Current next action: **a bounded observation-feature diagnostic for nominal
fingertip alignment, before further collection or longer fitting**. The completed
sampling-weight experiment improves correction-label prediction but leaves
autonomous success at 16/20. Test whether geometric features derived exclusively
from existing inputs make millimetre-scale corrections easier to learn; this is
a hypothesis, not established missing information. Preserve the observation-only
teacher, fixed episode splits and autonomous guarded comparison. Do not enlarge
collection or extend training blindly.
The user explicitly requests scripted demonstrations -> imitation -> corrective
teaching, short diagnostic comparisons, no return to PPO and no new long run.
Physical access and tests remain deferred. No training job is running.

Candidate 011 is the current best at **16/20 autonomous guarded lifts**, with zero
guard rejections, prohibited contacts or numerical failures. Its frozen bundle is
`workspace/validation/baselines/lift-imitation-011/`; SHA256 starts `e7975d9b`.
It is still below the 18/20 development gate; qualification remains deferred.
Candidate 007 is the preserved reference at **13/20 guarded lifts**. It is copied with its
metadata, training report and controller evaluation under
`workspace/validation/baselines/lift-imitation-007/`; SHA256 starts `b3b196c3`.
A further 20-case pilot and 200-case corrective batch using candidate 007 yield
20/20 and 192/200 teacher-assisted holds. A combined fresh fit (candidate 008)
reaches only 10/20 guarded; a short fine-tune from candidate 007 with frozen input
statistics (009) reaches 12/20. Neither replaces the baseline. Their two fits
take about 34 s each. All three controller evaluations use identical development
seeds, task settings, evaluator and runtime source, with zero actual prohibited
contacts and numerical failures.

All seven baseline failures now have matched teacher traces: **four grasp
execution, two maintaining lift, one approach clearance**. The 5 mm higher
teacher grasp already increased measured closing clearance from 2.85 to 7.85 mm
on matched cases; preserve that improvement. Further clearance work is secondary
to grasp execution unless new traces change the classification.

The actor sees current joint/command positions, TCP/cube pose and previous action.
The original teacher also uses its waypoint phase/history, initial cube pose, chosen IK
sequence and bilateral simulated touch signals. A recorded closing-state probe
changes the teacher's gripper label by 0.492 when only its contact branch changes;
touch signals are absent from policy input. This is a synthetic branch comparison,
not proof of two physically reachable states with identical observations, and
does not establish the cause of every failure. Exact teacher reproducibility is
unproven for that original teacher.

A separate observation-only teacher now uses exactly the actor's 36 inputs and
fixed robot geometry, with no hidden phase, initial target or simulated touch
signals. It completes all seven diagnosed cases. A 20-case corrective pilot on
fresh training seeds 46000-46019 completes 20/20 guarded two-second holds, with
631 learner actions restricted to close/lift stages. Its matched teacher-only
control also completes 20/20; both have zero prohibited contacts or rejections.
All 2,655 labels reproduce exactly when replayed in reverse order from stored
inputs. These are scripted/assisted results, not autonomous qualification. The
position-based grip proxy and model constants still require physical validation.

A deliberately small fresh fit on this pilot (16 training / 4 validation episodes,
3,000 updates in 8.63 s) produces candidate 010: **0/20 autonomous guarded lifts**,
six guard rejections and fourteen timeouts, with no prohibited contacts or
numerical failures. It is rejected; candidate 007 remains preserved at 13/20.
The evaluator, seeds, task and runtime sources match the baseline. Teacher, data
size and initialization differ, so this is not a causal comparison of teacher
observability alone. Shared inputs remove explicit hidden label dependencies but
do not establish learnability, adequate coverage or successful autonomous control.
Do not scale from the assisted success rate. Inspect the candidate's first wrong
commands; if refining 007 later, relabel compatible trajectories while preserving
their original training/validation roles rather than mixing teacher contracts.

The subsequent channel diagnosis shows arm alignment is a major contributor:
teacher-controlled arms with the learned gripper complete 4/5 baseline cases,
versus 1/5 with teacher-controlled gripper and learned arms. These are assisted
diagnostics. A fresh 20-case pilot injects baseline arm actions across approach,
close and lift; all 20 finish with the teacher's corrections. The baseline's
25,087 saved observations also accept observation-only labels without errors,
preserving every original episode's training/validation role. This relabeling
does not rerun or establish success of a new controller.

Candidate 011 refines 007 on the relabeled data and two small correction pilots:
180 training / 45 validation episodes, frozen normalizer, 3,000 updates in 9.84 s.
It fixes seeds 50003/50004/50010/50015 but regresses 50000, yielding 16/20 under
identical nominal guarded evaluation. Remaining failures 50000/50007/50008/50009
all stall before any bilateral grip; there are no remaining hold failures on
this development set. Read-only teacher queries still demand arm corrections
while the actor produces nearly zero movement. Next expose sustained learner
deviations during a bounded corrective pilot instead of relying only on occasional
single-step substitutions. Keep all successes autonomous during final evaluation.

The sustained-control pilot is now complete: 80 uninterrupted learner steps
(4 s), then teacher recovery, on fresh training seeds 48000-48019. Both it and
the matched intermittent-control dataset complete 20/20 assisted full holds.
The prefix dataset contains 269 frames with near-zero learned arm actions while
the teacher requests substantial correction, versus zero in the control. This
is a diagnostic action-disagreement signature, not proof of stable fixed points.

Two matched 3,000-update refinements from frozen 011 take 13.11/13.78 s:
prefix candidate 012 ties at 16/20 guarded (fixes 50000, regresses 50015), while
intermittent-data control 013 reaches 15/20 with one guarded rejection. All have
zero prohibited contacts/numerical failures. Neither replaces 011. Prefix data
reduce correction-subset validation arm MSE from 0.0624 to 0.0231, still far above
the aggregate arm MSE of 0.0030. Those states occupy only 197/26,724 training
frames. Next test their sampling priority without changing the validation split
or controller criteria. Both matched sources use the same episode seeds: **never
aggregate `lift-demos-prefix-001` and `lift-demos-prefix-control-001` together**.

The sampling comparison is complete. Candidate 014 uses exactly candidate 012's
four datasets, initialization from frozen 011, seed 23, learning rate 0.0001,
3,000-update budget and unchanged validation/checkpoint-selection metric.
Its 197 rare training corrections receive weight 20, increasing their expected
sampling share from 0.74% to 12.93%; validation frames are never weighted.
The priority artifact is hash-bound to exact data sources and split/order.
Training takes 11.93 s. Focused checks pass (20 existing/contract tests plus one
new behavior test); explicit uniform weighting reproduces the original fit.

On the same 72 correction-validation frames, arm MSE falls from candidate 012's
0.02309 to 0.01953; training-subset MSE falls from 0.01377 to 0.00340. Aggregate
validation loss instead increases from 0.00406 to 0.00436. Autonomous evaluation
`lift-imitation-runtime-016.json` remains **16/20**, zero guard rejections,
prohibited contacts or numerical failures, with exactly baseline 011's four
failed seeds. Candidate 014 (`10978b9a4c3da90b23718b8b0e91ee2afac513c684c945c0fbe2c5bbb1b1d667`)
does not replace 011. `weighted-correction-comparison-001.json` verifies identical
data, split roles, initialization, fit budget, task, runtime and frozen normalizer;
both saved baseline bundles and the fixed evaluation protocol remain unchanged.
Artifacts here and below are under `workspace/validation/`.

New autonomous traces in `weighted-policy-drift-001/` show seeds 50000/50007/50008
still stall before bilateral contact, with final lateral errors of 10.0/23.1/22.6
mm. Seed 50009 now first grips at step 185 (9.25 s), reaching only 27.08 mm lift
before the unchanged 10 s deadline; it has no qualifying hold. This is late grasp
execution, not evidence of losing an established two-second lift. Teacher queries
are read-only and never execute. `correction-clipping-audit-001.json` finds no
clipped input features at any final stalled state of 011 or 014; some earlier
observations clip, so an effect on preceding trajectories is not ruled out.
The raw inputs determine the teacher's geometry, but successful reproduction by
the present small network remains unproven. Do not report better label fitting
as task improvement or extend the deadline to turn late grasps into passes.

`configs/a1x_lift_evaluation.json` fixes development seeds **50000-50019** and
reserved held-out seeds **51000-51019**. The artifact audit found no prior use of
the held-out seeds. Keep them out of demonstrations, fitting and development
tuning; run only after freezing a candidate that meets the proposed 18/20 nominal
development gate. Every success requires the full 2 s hold through the guarded
controller, without teacher intervention, prohibited contacts or numerical faults.
If qualification fails, subsequent reuse is regression evidence, not a fresh holdout.

After nominal qualification, perform short comparisons of persistent calibration
translation/rotation bias, camera-pose noise and delay before robustness training.
Corrupt only the policy-supplied pose; use true simulated object/contact state
for the full two-second success check. Plot success versus error magnitude and
report achieved tolerance per tested direction and combined condition. Follow
the [perception robustness protocol](A1X_PERCEPTION_ROBUSTNESS.md); jitter does
not substitute for persistent calibration errors. Keep those synthetic tests
separate from measurement/validation of the actual two-camera pipeline. Existing
randomized CPU scores are exploratory and do not satisfy this stage. The adjacent
knife task's reported model/table/tool clearance discrepancy also remains a
physical-validation issue; the higher simulated grasp does not resolve it.
See [offline evidence](A1X_OFFLINE_VALIDATION.md)
for commands, versions, results and remaining gaps.

Selected available compute: local RTX 2000 Ada laptop GPU, 8188 MiB total,
7474 MiB free at the initial WSL check; Windows driver 560.76, Ubuntu on WSL 2.
Both 256-world, three-iteration smoke runs passed: peak total GPU usage was about
1.29 GiB for reach and 1.39 GiB for lift. Saved checkpoints reloaded and ran on CPU;
neither short run learned the task. Five scripted lifts held for two seconds on
both CPU and GPU, with zero reported capacity overflow. These are narrow,
measured workloads, not a promise that every training configuration fits 8 GB.
The adjacent driver checkout has concurrent team changes and must remain untouched.

The user has now provided Brev H100 compute for this training: organization
`wysocki-55ff9e-nih0`, machine `empty-peach-tortoise`, verified unique ID
`cdeczltwl`. Direct inspection shows H100 PCIe, 81,559 MiB, driver 570.148.08.
Use the separate task-local SSH config at `workspace/validation/brev/ssh_config`
from WSL. The existing active Brev organization belongs to another project:
do not switch it, run a global refresh, change shared SSH configuration, or
stop/reset/delete other instances. Shared WSL SSH config hashes were unchanged
after connecting. Remote project root: `/home/ubuntu/a1x-twin-training`.
The isolated Python 3.12 environment is ready. A 1,024-world, three-iteration
GPU smoke check passed, including checkpoint reload and two CPU evaluations.
Headless commands require `MUJOCO_GL=disable`; the first EGL attempt failed on
missing graphics libraries, and remote rendering remains unverified. No system
graphics packages or drivers were changed. See the offline evidence for commands.

Other Brev work is running concurrently on this computer. A later hash check
found that the shared Brev-generated config had changed while another project's
`brev exec` was active; the main SSH config was unchanged. Do not restore an old
shared config or interfere with that process. This task continues to use only
its separate `ssh -F` configuration.

A 1,024-world lift smoke check used 2,074.5 MiB total GPU memory without capacity
overflow. The 8 GB GPU is sufficient for these measured workloads. Learned reach
now achieves 19/20 runtime successes on each of two seed sets, with one guarded
rejection and zero actual contacts per set. Learned lift is still unproven.

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
| 2026-09-26 | Local implementation checkpoint preserved | `5de2292`: control corrections, scripted physics/runtime and initial learned reach evidence | Improve learned policies without changing the live driver |
| 2026-09-26 | Independent model geometry audit passes | `a1x-geometry-001.log`: 2 tests, 100 random URDF/MuJoCo link poses, jaw units, TCP offset and regenerated XML agreement | Physical directions and offsets remain deferred |
| 2026-09-26 | Reach fine-tuning substantially improves accuracy | `reach-candidate-003`: 6,144,000 additional transitions, 19/20 nominal and 20/20 randomized CPU; fresh seeds 30000-30019 repeat those counts | Common-controller evaluation exposed two marginal table contacts |
| 2026-09-26 | Guard includes measured start and provisional 3 mm environmental clearance | `reach-runtime-buffered-001.json`, `reach-holdout-buffered-003.json`: 19/20 each, one rejection each, zero actual contacts; `runtime-lift-buffered-001.json`: scripted 5/5 | Keep clearance provisional until measured tracking/calibration error is known |
| 2026-09-26 | Physical observation-loss stops and modeled object poses corrected offline | `physical-observation-loss-002.log`, `observed-geometry-001.log`, `observed-geometry-002.log` | Verify real perception only when physical access resumes |
| 2026-09-26 | Lift approach reward diagnosed as nearly absent at home | `lift-reward-diagnosis.json`: old float32 reward exactly zero on 99/100 starts; corrected coarse/fine fingertip reward nonzero on all 100. `lift-reward-contracts-001.log`: 11 passed | Complete original bounded baseline; compare new task definition separately |
| 2026-09-26 | Updated suite and fresh guarded reach holdout pass | `offline-suite-004.log`: 166 passed; `reach-final-holdout-003.json`: 20/20 on seeds 32000-32019, zero contacts at every substep; `runtime-lift-substeps-001.json`: scripted 5/5 | Preserve source checkpoint and proceed with lift |
| 2026-09-26 | Original-reward lift baseline completed with no learned lifts | `lift-candidate-001/report.json`: 606 iterations, 14,893,056 transitions in 900.2 s; 0/20 nominal and 0/20 randomized, no numerical/capacity failure | Corrected reward comparison: same seed/worlds, at most 606 iterations and 900 s; same validation seeds |
| 2026-09-26 | Corrected reward learns approach but no sustained lift | `lift-candidate-002`: 476 iterations, 11,698,176 transitions in 900.9 s, 0/20 nominal and randomized. Intermediate 150-iteration inspection approaches within 26-37 mm while leaving jaws open | Use grasp demonstrations or a measured curriculum; do not equate reward growth with task success |
| 2026-09-26 | Nominal CPU evaluation now preserves saved cube-yaw convention | `evaluation-yaw-contract-001.log`: 11 passed, including two actual CPU/runtime observation agreement cases | Reevaluate lift candidates with corrected input convention; previous run reports retain original evaluator hashes |
| 2026-09-26 | Scripted demonstration collection explored broader task region | `lift-demos-005`: 18/20 completed 2 s holds, two collision failures retained; observations/actions saved separately for every attempted seed | Successful samples can seed imitation, but these are not learned-policy results or hardware instructions |
| 2026-09-26 | User supplied H100; isolated read-only connection verified | `brev/target.json` and shared-config hashes under validation; remote hostname `brev-cdeczltwl` | Isolated dependencies and simulation-only source transfer; short smoke test before longer training |
| 2026-09-26 | Corrected-yaw reevaluation confirms lift candidate 002 still fails | `lift-runtime-002.json`: 0/20 nominal, randomized and runtime lifts; no prohibited contacts or numerical failures | Use grasp demonstrations or curriculum instead of interpreting approach reward as success |
| 2026-09-26 | Isolated H100 setup and GPU smoke check pass | `brev/h100-lift-1024-002`: 73,728 transitions, 1,708.9 MiB peak total GPU usage, zero overflow; `h100-cpu-contracts-001.log`: 28 passed; `h100-gpu-contracts-001.log`: 2 passed | Prepare demonstration warm start and a bounded larger-batch experiment |
| 2026-09-26 | Broader scripted collection and first imitation candidate complete | `lift-demos-006`: 196/200 scripted holds, four collision failures excluded; `lift-imitation-001`: 0/20 nominal and 2/20 randomized CPU holds, 6/7 collision-ended trials respectively; `lift-imitation-runtime-001.json`: 0/5 with guarded rejections and zero actual prohibited contacts | Five-minute PPO refinement, retain current success and collision criteria |
| 2026-09-26 | H100 scaling and bounded refinement completed | `brev/h100-lift-4096-001`: smoke passed, 4,266.9 MiB peak; `brev/h100-lift-imitation-ppo-001`: 23,298,048 steps, 237 iterations, 300.5 s training, 0/20 nominal/randomized holds | Diagnose lost behavior; preserve this failed experiment |
| 2026-09-26 | Timed demonstration ambiguity reproduced and state-triggered collector validated | `demo-phase-boundaries-001.json`: all 196 successful timed episodes contain near-identical observations with conflicting phase-transition commands; `lift-demos-state-002`: matched seeds, 196/200 holds | Train on the state-triggered examples |
| 2026-09-26 | Improved imitation learns nominal lifts but fails controller qualification | `lift-imitation-002`: 16/20 nominal and 5/20 randomized; `lift-imitation-runtime-002.json`: 0/20, all guarded rejections, zero actual prohibited contacts; `lift-imitation-video-002`: inspected successful CPU replay | Resolve training/runtime approach mismatch and learning stability before fresh qualification |
| 2026-09-26 | Fixed excess clearance rejection and premature hold completion | Analytic 5/3.1 mm gaps incorrectly rejected before fix; 2.9 mm gaps and penetration remain rejected. Fake-arm tests reproduce 40 samples spanning only 1.95 s. `clearance-runtime-contracts-002.log`: 50 passed | Check full offline suite and established reach/scripted-lift regressions |
| 2026-09-26 | Unchanged imitation checkpoint completes first guarded lifts | `lift-imitation-runtime-004.json`: 3/20, independent holds 2.010-2.050 s, zero actual prohibited contacts/numerical failures; 16 guarded rejections and one timeout | Train with controller-accepted demonstrations; retain the 3 mm threshold and full-duration hold |
| 2026-09-26 | Full regressions and established runtime baselines remain valid | `offline-suite-005.log`: 181 passed; `reach-clearance-regression-001.json`: 20/20; `runtime-lift-distance-001.json`: scripted 5/5, 3.815-3.825 s continuous holds | Build guarded-runtime teaching data; no new long H100 job is running |
| 2026-09-26 | Scripted teacher now runs through the guarded controller | `lift-demos-runtime-001` to `005`: matched 20-case development comparisons improve 10/20 to 20/20; `lift-demos-runtime-006`: 197/200, three guarded rejections, no actual prohibited contacts or numerical failures | Train a separate imitation candidate on successful runtime examples |
| 2026-09-26 | Runtime demonstrations and longer fitting do not yet improve learned reliability | `lift-imitation-003` / `004`: 3/20 and 2/20 guarded holds; lower prediction loss does not imply grasp retention | Diagnose trajectory drift and collect corrective examples; retain earlier candidate and unchanged criteria |
| 2026-09-26 | Reproduced and corrected IK home-restart gripper reset | Requests for 0% / 100% both returned 60% before fix; two regressions now pass. `runtime-demo-contracts-003.log`: 18 passed; `ik-ui-regression-001.log`: existing branch-jump test passes | Preserve scoped code/data evidence; physical access remains deferred |
| 2026-09-26 | Corrective teaching substantially improves autonomous guarded lifts | `runtime-drift-001` identifies height oscillation despite retained grip; `lift-demos-corrective-002`: 188/200 teacher-assisted holds; candidate 005: 17/20 nominal CPU, 11/20 randomized CPU, 12/20 guarded | Keep candidate 005; investigate approach clearance before fresh qualification |
| 2026-09-26 | Second corrective round and aggregation verified, but not promoted | `lift-demos-corrective-003`: 160/200 teacher-assisted holds; candidate 006: 10/20 guarded. Per-source validation splits preserved; 15 focused tests passed | Retain both experiments; do not equate more data with improvement |
| 2026-09-26 | Guarded learned lift replay inspected | `lift-imitation-runtime-video-001/seed-50001.mp4`: 72.03 mm final lift, 2.010 s hold; metrics exactly match unrendered evaluation | Simulation-only demonstration; physical tests remain deferred |
| 2026-09-26 | Higher teacher grasp adds measured table clearance | `grasp-height-comparison-001`: original and +5 mm teacher both 20/20; minimum close-phase gap 2.85 versus 7.85 mm | Keep physics, collision thresholds and task criteria unchanged |
| 2026-09-26 | Raised corrective teaching gives the best guarded candidate so far | `lift-demos-raised-001`: 185/194 teacher-assisted holds; `lift-imitation-007`: 15/20 nominal CPU, 13/20 randomized CPU; `lift-imitation-runtime-009.json`: 13/20 guarded, two rejections, five timeouts, zero actual contacts/numerical failures | Retain candidate 007; collect corrections for its own deviations |
| 2026-09-26 | Fixed-normalization PPO refinement still degrades the learned lift | `lift-ppo-fixed-001`: 167 iterations, 2,052,096 transitions, 300.06 s; guarded checkpoints 3/1/5/0 out of 20 | Reject refinement; preserve checkpoint history and candidate 007 |
| 2026-09-26 | New training options and evaluation labels verified | `raised-refinement-contracts-001.log`: 22 passed; `nominal-runner-contract-001`: one GPU iteration and actual nominal/randomized CPU evaluations | Historical nominal run's mislabeled duplicate evaluation is documented; independent evaluations retained |
| 2026-09-26 | Raised-policy failure traces isolate remaining learning errors | `runtime-drift-002`: four policy timeouts, four matched teacher successes; two held-block height failures and two cases without a final grip | Bounded corrective pilot before another fit; physical access remains deferred |
| 2026-09-26 | Candidate-007 corrective teaching completes, but both new fits regress | `lift-demos-raised-pilot-002`: 20/20; `lift-demos-raised-002`: 192/200; candidates 008/009: 10/20 and 12/20 guarded versus baseline 13/20 | Preserve baseline; stop scaling data/fitting and audit teacher observability |
| 2026-09-26 | Conservative imitation refinement and split protection verified | `imitation-refinement-contracts-001.log`: 16 passed; actor restored, normalizer frozen, source untouched, mismatched checkpoints/contracts and reassigned/omitted prior split episodes rejected | Refinement is available but not a demonstrated improvement |
| 2026-09-26 | Failure/input audit and evaluation separation saved | `lift-failure-audit-002`: identical recorded comparison conditions, four grasp/two lift/one clearance failures, teacher-only state identified, no prior held-out seed use; fixed protocol in `configs/a1x_lift_evaluation.json` | Target grasp decisions and information sufficiency before further training; no PPO/long runs; physical tests deferred |
| 2026-09-26 | Observation-only teacher and targeted corrective pilot pass | `observable-teacher-pilot-005`: 7/7 diagnosed cases; `lift-demos-observable-001` and matched scripted control: 20/20 each; `lift-observable-label-audit-001.json`: all 2,655 labels exactly reproducible from actor inputs; 34 focused tests pass | Keep the teacher a separate data contract; grasp proxies remain model-specific |
| 2026-09-26 | Small observable-teacher imitation fit fails autonomous comparison | Candidate 010: 3,000 updates / 8.63 s, 0/20 guarded, six rejections and fourteen timeouts; identical evaluation conditions verified in `lift-observable-candidate-comparison-001.json` | Retain 007 at 13/20; trace first policy divergence before another fit or larger collection; held-out and physical tests remain deferred |
| 2026-09-26 | Channel diagnosis supports correcting arm alignment as well as closure | `observable-policy-drift-001` and `observable-channel-ablation-001`: teacher arm 4/5 versus teacher gripper 1/5 on baseline cases; correction controls verified by 21 tests | Collect approach/close/lift arm deviations with unchanged teacher information contract |
| 2026-09-26 | Consistent relabeling and short refinement improve autonomous guarded lifts | `lift-demos-observable-relabel-001`: 25,087 labels, original split preserved; arm-correction pilot 20/20 assisted; candidate 011: 16/20 autonomous, zero rejections/contacts/numerical failures, 9.84 s fit; frozen bundle and comparison audit saved | Retain 011 and original 007; target four pre-grasp stalls with sustained learner rollout/recovery; no held-out, robustness or physical testing yet |
| 2026-09-26 | Sustained learner control exposes corrections absent from intermittent data | `learner-prefix-comparison-001`: matched 20/20 assisted holds, 269 versus zero slow-arm/correction frames; 27 schedule/runtime tests pass | Preserve separate matching datasets; train/validation roles remain fixed |
| 2026-09-26 | Matched short refinements improve corrective action accuracy without increasing task success | `learner-prefix-fit-comparison-001`: prefix 012 16/20, control 013 15/20; no actual contacts; unchanged initial checkpoint, splits, hyperparameters, runtime and normalization | Keep 011 at 16/20; test bounded correction sampling priority before more data or updates; held-out/robustness/physical stages remain deferred |
| 2026-09-27 | Training-only correction weighting improves action prediction but not nominal grasp success | `weighted-correction-comparison-001`: candidate 014 16/20, zero rejections/contacts/numerical failures; 11.93 s fit; 21 focused tests pass; both baseline bundles unchanged | Retain 011; test derived fingertip features with a bounded comparison before more data or longer fitting |
| 2026-09-27 | New autonomous failure traces retain grasp execution as the priority | `weighted-policy-drift-001`: three pre-grasp stalls and one first grip at 9.25 s; clipping audit finds no final-state clipping | Keep the 10 s deadline and full 2 s hold; no teacher intervention in evaluation |
| 2026-09-27 | User explicitly adds persistent camera-to-arm translation/rotation error to the target | [Perception robustness protocol](A1X_PERCEPTION_ROBUSTNESS.md): bias/noise/delay separate, corrupt supplied pose only, score true simulated state, paired magnitude/direction sweeps and tolerance reporting | Nominal development first; no calibration tolerance measured and no real two-camera validation yet |

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

Knife-setup measurements now supplied: plate front overhang **1 mm**, thickness
**11 mm**. The corrected provisional object origin is `[440.627, 0, -11] mm`
in the CAD base frame, conditional on forward-axis alignment. A new offline
probe saved separately as `plate_corrected_reach_probe.json` gives **44/80**
sampled descent IK passes across four illustrative yaws (7/12/9/16). All 44
also pass sampled mesh-to-table checks along descent and a direct joint-space
arrival from the stored folded pose. Checks include six moving arm links and
G1/dual mount envelopes; they do not establish self/obstacle/plate collision,
installed camera coverage, calibration, or continuous controller safety. Eight
hardware-free geometry/frame tests pass. No motor commands were sent. This
remains adjacent knife-grasp work and does not complete a twin-training stage.

Further knife-task follow-up: the user aligned the knife and confirmed they
will supervise. Fresh receive-only feedback remains healthy. The hover path
was screened with the existing MuJoCo collision model plus mount, plate and
knife proxies: no sampled intersections, but a 2.17 mm nominal base/arm_link2
gap warrants reviewing the full route. Only an initial two-degree arm step
is prepared for operator-controlled execution; its sampled nonadjacent gap is
3.38 mm. The new `../grasping/hover_step_a1x.py` is disarmed on opening,
hold-to-run, single-use, and sends no gripper or enable frames. Six fake-arm
tests, a fake GUI smoke check, and a real receive-only check pass. No physical
movement has been performed by this task; this does not advance twin-training
validation. Reports and instructions are in the knife-grasp README.

Knife-task operator follow-up: four attempted first-step sessions transmitted
15/17/80/0 frames and none completed; maximum requested J2 change was 0.073 deg.
The assistant's new controller incorrectly rejected normal active status 0x00
after idle 0x10. This is corrected without weakening other fault checks. The
original stop reason and progress are now printed/persisted, and GUI status
height is fixed to prevent button movement. Ten controller/GUI tests pass.
Fresh receive-only feedback and offline re-screening preserve the same 2 deg
limit (3.51 mm sampled nonadjacent clearance). No new motor commands were sent
during diagnosis; physical step completion remains pending.
