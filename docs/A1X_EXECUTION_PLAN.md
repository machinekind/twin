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
- [x] Train and qualify nominal A1X `lift` for the configured 30 mm simulated cube
  and workspace. Frozen candidate 027 passes the first 20-case held-out evaluation
  at 20/20 with full holds and zero prohibited contacts. Actual object/workspace
  agreement remains part of physical validation; SO-101 weights were not used.
- [ ] Include measured actuator behavior and realistic perception delay/noise,
  plus persistent translation and rotation errors in camera-to-arm calibration.
  Keep contact-force protection and other runtime restrictions consistent with training.
- [ ] Measure success versus perception-error magnitude and report the achieved
  tolerance using true simulated object/contact state; follow the separate
  [perception robustness protocol](A1X_PERCEPTION_ROBUSTNESS.md).
- [x] Evaluate sustained holds, not momentary height/contact. Log failures and
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
- Preserve vendor/upstream attribution. On 2026-09-27 the user authorized a
  comparison-package pull request in `machinekind/digital_twin_training` so Marcin
  can compare policies. Publishing a simulation package does not authorize merging
  into the team branch or powered-arm tests.

## Progress and resumption

Current next action: **freeze candidate 029, verify the publication branch against
its evaluated behavior, and run the reserved perception qualification before a
comparison-package PR**. `calibration-correction-comparison-001` reproduces 028
exactly and compares two matched 1,000-update fits (3.66/3.52 s). Control 029 uses
the previous examples with one explicitly re-encoded/relabelled orientation frame;
corrective 030 adds 19 successful training recoveries. Both use identical old plus
new validation episodes, unchanged normalization and saved runtime conventions.
At zero / +2 mm Y / -0.5 degree Z, autonomous development scores are 028: 19/17/18,
029: **20/20/19**, 030: 19/19/16 (out of 20 each). All prohibited-contact and
numerical-fault counts are zero. Both new fits recover training seed 53076; only
029 meets the predeclared no-regression promotion rule. No improvement can be
attributed to the added corrective examples. Broader tolerance and fresh held-out
qualification for 029 are pending; the following 028 evidence remains historical.
Nominal development is 19/20; the first fixed held-out nominal qualification is
**20/20**, with true holds of 2.005-2.055 s, zero guard rejections, prohibited
contacts or numerical faults for reference 027. Preferred development candidate
028 now fixes all 51 premature completions in the 320-case calibration sweep,
without regressions or any actor retraining. Its fixed 5 mm height-error allowance
is saved with the policy and used by the actual completion check; the true 60 mm
lift, two-second hold and collision criteria are unchanged. The remaining 18
failures are collision-guard rejections, with zero actual prohibited contacts or
numerical faults. The additional +/-1 mm sweep gives 19/20 in every axis direction;
the +/-0.5 degree rotational sweep gives 18-19/20. These are separate-factor
development tolerances, not combined or physical validation. +2 mm Y remains
17/20. Robustness holdout 52000-52019 remains untouched; new robustness training
is assigned 53000-53999. The actual two-camera pipeline is unvalidated.

Keep both the original reference 027 (`22d79622`) and preferred development 028
(`851b34fe`) under `workspace/validation/baselines/lift-imitation-0xx/policy.pt`.
Both use `cube_yaw_mode=continuous`, `gripper_decoder=a1x-templates-v1`;
028 adds `cube_height_error_bound_m=0.005`. Their actor and normalizer are identical
to 024/026; the decoder selects the nearest supported
gripper action using the learned output and measured jaw, with no teacher phase
logic or intervention. Preserve all earlier baselines, including 024/026 at
17/20 with zero development guard rejections and their training regression.
The remaining nominal development failure is 50009 (approach/grasp guard rejection).
028 also passes 20/20 nominal held-out regression with true holds 2.155-2.240 s
and zero rejections/contacts/faults; this reused set is not fresh qualification.
The 51000-51019 set has now been used once for qualification: future reuse is
regression evidence, never fresh held-out proof. Keep it excluded from training.
Do not aggregate original and replacement variants of a training episode or
silently drop saved action/observation conventions during refinement.
The user explicitly requests scripted demonstrations -> imitation -> corrective
teaching, short diagnostic comparisons, no return to PPO and no new long run.
Physical access and tests remain deferred. No training job is running.

The corruption wrapper applies persistent translation AND rotation to the pose
supplied to the policy, with rotation moving position about the arm base as well
as orientation. Jitter and timestamped delay are separate. True scoring and
scene collision geometry stay independent of that input. The successful zero
control reproduces the saved baseline; 73 focused checks pass. Traces and the
42,021-frame audit/curves are in `workspace/validation/perception-development-002/`
and `perception-analysis-001/`; the detailed results and limitations are in the
[perception protocol](A1X_PERCEPTION_ROBUSTNESS.md#first-frozen-policy-development-results-2026-09-27).
The earlier attempt 001 is retained after a stop-message-format mismatch.

A counterfactual audit also shows an information limit: the same policy input
can describe distinct true horizontal object positions with opposite persistent
calibration offsets. Do not train truth-informed correction labels that the
policy cannot reproduce from its observations. The fixed-bound completion
correction is now validated in development (`perception-development-004/005`,
`height-bound-comparison-001`, 66 focused tests). It recovered 51 cases with
identical policy inputs/actions until the former stop; 42,926 corrected frames
and 18,662 boundary-check frames are audited. Five failure traces queried against
the observation-only teacher support targeting premature closing and grasp
misalignment next (`perception-clearance-audit-001`). Do not assume early closure
explains every failure, and do not fit these development traces. Use fresh cases
from 53000-53999 and preserve yaw/decoder/completion conventions; the current
imitation runner still rejects incompatible yaw/decoder datasets. Preserve all
checkpoints and keep the robustness holdout reserved.

Historical candidate 022 reached **17/20 autonomous guarded
lifts**, with one guard rejection and zero actual prohibited contacts/numerical
failures. Its frozen bundle is `workspace/validation/baselines/lift-imitation-022/`;
SHA256 starts `a701802f`. This is a modest development improvement, not held-out
qualification or evidence of calibration-error tolerance.
Candidate 011 remains the **16/20 zero-rejection reference**, with zero
prohibited contacts or numerical failures. Its frozen bundle is
`workspace/validation/baselines/lift-imitation-011/`; SHA256 starts `e7975d9b`.
Both were below the 18/20 development gate; neither was used for qualification.
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

The explicit-geometry experiment is complete. The optional
`a1x-alignment-v1` encoder derives 24 scaled fingertip/cube alignment, orientation,
jaw position/lag and observed cube-height features from the existing 36 inputs
and fixed model geometry. It uses no scene state, teacher decisions, contacts or
IK outputs. Its model signature is saved and checked on reload. Added first-layer
weights start at zero, retaining the initial baseline function; input statistics
stay frozen. The raw policy path remains the default. This adds 6,144 weights,
so the experiment tests feature augmentation, not a parameter-matched claim.

Matched 3,000-update fits use the same four datasets and frozen 011 checkpoint:

| Candidate | Change | Guarded success | Guard rejections | Actual contacts / numerical faults |
|---|---|---|---|---|
| 015 | Raw reproduction control | 16/20 | 0 | 0 / 0 |
| 016 | Derived geometry features | 16/20 | 1 | 0 / 0 |
| 017 | Raw inputs; remove previous arm-action history | 14/20 | 0 | 0 / 0 |

`alignment-feature-comparison-001.json` verifies 015 reproduces every saved 012
tensor exactly. Feature candidate 016 takes 11.38 s fitting plus 3.60 s feature
preparation. Its validation loss falls from raw control's 0.00406 to 0.00341,
and correction-subset validation arm MSE from 0.02309 to 0.01953. It fixes baseline
seeds 50000/50008 but regresses 50014/50015. The guard rejects a table-directed
move on 50009; this is not an actual contact. Candidate SHA256 starts `f0d0d2aa`.
No replacement or held-out qualification follows this tie with a new rejection.

`previous-arm-dependence-001.json` checks 12 final failure observations from
011/014/016: all 156 counterfactual teacher labels are unchanged when only the
six previous arm-action inputs change. Learner actions can change by 0.47 for
a single previous-input perturbation of 0.2. These are input probes, not proof
of equivalent physically reachable states or of the cause of a failed trajectory.
The subsequent raw-input history ablation 017 removes columns 29:35 from the
saved network, retaining previous gripper action at index 35. It intentionally
changes the initial function. After 3,000 updates / 12.88 s, its guarded score is
14/20 (bare CPU 15/20), so it is rejected. `arm-history-comparison-001.json`
verifies exact arm-history invariance on all 6,602 validation observations,
unchanged data/splits/sampling/runtime, and preserved baseline bundles. The first
audit attempt rejected a floating-point representation of 400 x 5 ms
(`1.9999999999999793`); the corrected audit uses the evaluator's existing 10 ns
tolerance. No episode, hold criterion or evaluator changed. Candidate SHA256
starts `9835a721`. Input dependence alone did not establish an effective remedy.

`alignment-policy-drift-001` shows three close-stage stalls without bilateral
grip (50007/50014/50015), and the early clearance rejection (50009). The matching
12-case diagnostic `alignment-channel-ablation-001` changes the correction
priority for this candidate: **teacher gripper + learned arms 4/4; teacher arms
+ learned gripper 1/4; teacher both 4/4**. All have zero actual contacts, guarded
rejections and numerical failures; all successes meet the full two-second hold.
`alignment-channel-audit-001.json` verifies executed channels, learned outputs
and 144 replayed teacher labels. These runs are assisted diagnostics only and
must never enter autonomous success counts or the training dataset. The gripper
also changes later arm inputs, so this does not isolate a mechanically independent
actuator defect. It supports a fresh training-only gripper-correction pilot.

Validation: `alignment-feature-contracts-001.log` passes 66 controller, imitation
and geometry checks; `alignment-orientation-contract-001.log` adds a supplied-roll
check. After the history ablation, `arm-history-contracts-001.log` passes all 30
focused tests. The original baseline checkpoints and fixed evaluation protocol
remain unchanged. All experiment/audit paths above are under
`workspace/validation/`; no long run, PPO, held-out, robustness or physical
evaluation was started.

The first gripper-focused pilot is complete: `lift-demos-gripper-001` uses
candidate 016's arms continuously and the observation-only teacher's gripper
on fresh seeds 49000-49019. It finishes **17/20 assisted holds** in 27.83 s,
with zero guarded rejections, actual contacts or numerical failures. The three
failures remain in approach throughout (49003/49005/49015). All 3,032 labels and
submitted action-channel combinations are verified in `gripper-pilot-audit-001`.
Successful episodes split into 14 training / 3 validation, with 3 excluded.

The pilot contains 668 close-phase frames, but its coverage is limited:
`gripper-coverage-description-001.json` finds a minimum predicted closure lead
of 0.371 mm in training and 0.527 mm in validation, versus approximately
0.127/0.390/0.226 mm at the three diagnosed candidate-016 closing stalls.
These are single-jaw joint-coordinate differences, not total aperture changes.
Continuous teaching avoids the most sluggish closing states rather than
demonstrating recovery from them. This is evidence of a collection mismatch,
not proof of the only reason the policy fails.

Gripper-only fitting is now available through `--train-channels gripper`. It
updates only the final gripper row and bias (129 weights), leaving all hidden
weights, arm output rows, input statistics and observation mapping fixed. An
internal check and saved-checkpoint audits enforce that restriction. Identical
observations produce identical arm actions; changed gripper feedback can still
change later arm trajectories. The collection CLI permits learner probability 1
only with the observable teacher continuously controlling the gripper, or with
the explicitly bounded gripper-burst schedule below.

Two matched 3,000-update refinements from 016 take 6.96/7.16 s, plus 4.05/4.66 s
feature preparation. Control **018** uses the original four data sources; **019**
adds the successful new pilot episodes, retaining the original roles. Guarded
results are **16/20 and 15/20**, respectively, each with one table-directed
rejection and zero actual contacts/numerical faults. Neither replaces 011.
Bare CPU scores were 17/20 and 16/20 with one prohibited-contact episode each;
those results do not qualify either policy for guarded operation.
`gripper-refinement-comparison-001.json` verifies frozen arm mapping, source
checkpoint, fit settings, prior split roles and unchanged controller/evaluator.
On the same 115 new validation close frames, command-clipped gripper MSE changes
only from 0.003758 (016) to 0.003720 (018) / 0.003676 (019). Small supervised gains
did not improve task success. Checkpoint SHA256 prefixes: 018 `e185a1ef`,
019 `05491261`. Both original baseline bundles remain intact.

A subsequent **five-case collection probe, with no further fit**, runs seeds
49100-49104 with learned arms throughout. When closing first starts, it executes
50 steps (2.5 s) of learned gripper control, then returns gripper control to the
teacher. The matched control uses teacher gripper throughout. Both finish 4/5
assisted holds; 49100 stalls in approach in both. All have zero actual contacts,
guarded rejections or numerical failures, and the original 10 s deadline holds.
`gripper-burst-comparison-001.json` verifies identical starts/pre-burst trajectories,
all 1,653 labels, submitted control channels and previous-action observations.
Neither variant contains close-phase frames with predicted closure lead 0-0.4 mm
while the teacher requests at least 0.8 mm. The burst variant's minimum is
0.522 mm, versus 0.483 mm for its control. This proxy comes from earlier stalls;
it is not a fixed-point proof. Do not train or scale this probe as though it solved
failure-state coverage. Never aggregate `lift-demos-gripper-burst-001` with
`lift-demos-gripper-burst-control-001`: their seeds overlap.

Tests: `gripper-refinement-contracts-001.log` passes 61 focused cases;
`gripper-burst-contracts-001.log` passes 36 runtime/collection cases. A final
report-counter correction for pre-existing full-action/prefix modes passes both
selected cases in `gripper-action-counts-001.log`; it changes no controls or
labels. Exact pre-correction collection helper sources are archived under
`gripper-burst-source-001/`, hash-matching both probe manifests. All artifacts
above are under `workspace/validation/`. No job is running; no qualification,
robustness campaign, PPO or physical test has been started.

The subsequent training-only screen locates genuine failures without consuming
evaluation cases. `gripper-training-screen-001` runs all 14 already assigned
new-pilot training cases: **14/14 autonomous successes** in 19.61 s, including one
transient slowdown that recovers. `gripper-training-screen-002` then runs a fixed
subset (the first 20 sorted original training seeds): **17/20 successes** in
27.57 s. Seed 44013 stalls before bilateral grasp; 44006 and 44022 grasp too late
and achieve only 1.825/1.790 s of qualifying hold. There are zero prohibited
contacts, guard rejections or numerical failures in either screen. Teacher
queries diagnose decisions but never supply actions in these screens.

`gripper-training-recovery-001` replays those three failures. The learner controls
all actions until steps 53/82/85, respectively (20 steps after the first observed
slow-closing frame, capped at 100); the teacher then supplies only the gripper.
The arms remain learned throughout. All **3/3 assisted recoveries** finish the
full two-second hold by the unchanged 10 s deadline, with zero actual contacts,
rejections or numerical faults. Recorded pre-intervention observations exactly
match the autonomous traces; all 492 teacher labels replay from the 36 inputs.
The recovered trajectories include **57 slow-closing frames**, using the same
0-0.4 mm learner / at least 0.8 mm teacher proxy as the earlier burst probe.

`lift-demos-targeted-gripper-001` replaces exactly those three training episodes
in the original observable-relabel source. It retains every episode's role and
report order, and all other episode files and validation arrays are identical.
The original source remains intact; never aggregate it with its replacement.
The combined four-source data now contain 26,795 training frames and the same
6,602 validation frames, with 196 training / 49 validation / 9 excluded episodes.

Candidate **020** changes only those three data trajectories relative to control
018: initialization 016, frozen arm/hidden mapping and input statistics, uniform
sampling, seed 23, learning rate 0.0001, and 3,000 updates all match. Fitting takes
6.45 s plus 4.32 s feature preparation, selecting update 2,600. Guarded evaluation
is **15/20**, one table-directed rejection, zero actual contacts/numerical faults;
failures are 50007/50008/50009/50014/50015. Bare CPU is 17/20 with one prohibited
contact and does not qualify the candidate. Checkpoint SHA256 starts `773f8e3c`.
`targeted-gripper-comparison-001.json` verifies identical evaluation conditions,
validation arrays, arm outputs, source/configuration, old roles and frozen
baseline bundles. Candidate 020 is rejected as a replacement for 011.

Separate autonomous replays on the three selected training failures show that
018 and 020 each solve only 44022. The persistent stall at 44013 remains; 020
improves 44006's hold from the control's 1.32 s to about 1.99 s, still a failure.
On the 57 corrective frames, mean predicted closure changes only from 0.286 mm
(018) to 0.291 mm (020), versus 1 mm teacher targets. Corrective coverage has
improved; reliable learned closure has not. Teacher label replay establishes
that the information is available in the observation, not that this frozen
network representation can fit or generalize the decision boundary.

`gripper-linear-diagnostic-001.json` solves the 129-parameter final gripper layer
directly in float64, with uniform training-only least squares, on the original
and replacement datasets. Both matrices have full rank; each solve takes under
0.2 s, and the whole diagnostic takes 11.89 s including feature extraction. On
the replacement data, training MSE falls from 0.001201 to 0.001145, but raw
validation MSE slightly worsens (0.0009983 to 0.0010039). Mean closure on the
targeted frames rises only to 0.333 mm. No checkpoint or runtime action was
produced. This tests the optimum of the current uniform linear-head objective;
it does not prove that all representations or loss weightings are inadequate.
The next comparison changes the gripper's representation without changing arm
outputs or collecting additional examples. All diagnostic checks pass; no
production code changed in this cycle, so earlier unit tests were not rerun.

That nonlinear comparison is now complete. The optional `mlp64-v1` gripper
residual adds an independent 60-64-64-1 ELU network (8,129 parameters) to the
gripper output. It consumes the same normalized observations and existing derived
geometry, has no teacher at runtime, and starts with a zero final layer. The
original actor and observation normalizer are frozen under
`--train-channels gripper-residual --gripper-residual mlp64-v1`. Legacy checkpoints
default to no residual; saved residual policies retain their inference path.

`gripper-residual-experiment-001.json` fixes one 3,000-update fit from 016 on
exactly candidate 020's data, splits, seed 23, learning rate 0.0001 and uniform
sampling. Candidate **021** takes 7.43 s plus 3.94 s feature preparation, selecting
update 3,000. Raw validation gripper MSE is 0.0009965 versus control 020's
0.0009983. On the 57 targeted slow-closing inputs, mean closure rises from
0.291 mm (020) to 0.323 mm (021), still far below the 1 mm teacher target.

Guarded autonomous development is **16/20**, versus 020's 15/20, but with **two
guard rejections**, zero actual prohibited contacts and zero numerical faults.
Failures are 50007/50008/50009/50015. This is not better than retained 011's
16/20 with zero rejections. Bare CPU is 17/20 with one prohibited contact.
Separate autonomous training-case replays fail all three selected cases:
44006 holds only 1.18 s, 44013 never grasps, and 44022 holds only 1.70 s.
Do not promote 021 or infer reliable recovery from lower prediction error.
Its checkpoint SHA256 begins `05166d4a`.

`gripper-residual-comparison-001.json` verifies identical dataset/split/validation
content, initial predictions, and every original actor/normalizer tensor. Arm
outputs on identical inputs remain exactly unchanged, although changed gripper
feedback changes later arm trajectories. Both frozen baseline bundles and the
fixed evaluation protocol remain unchanged. Exact experiment sources are saved
under `gripper-residual-source-001/`. `gripper-residual-contracts-002.log` passes
41 focused tests, including controlled nonlinear response, raw/geometry inputs,
save/reload, inherited residual fitting and legacy checkpoint compatibility;
`gripper-residual-runner-001.log` passes all 11 policy-runner tests.

`gripper-residual-rejections-001` traces the two guard stops without teacher
intervention. The new 50008 rejection is **finger-link-2 versus cube at the
commanded end pose**, during closing with 1.27 mm lateral error, 13.89 mm measured
jaw position and 12.01 mm command, before bilateral contact. The teacher's
observable blockage proxy is true and requests closure; this is not evidence of
premature approach closing. Seed 50009 retains the earlier **finger/table start
clearance rejection**: about 121 mm lateral error, nearly closed jaws and 17 deg
orientation error while the teacher requests approach with open jaws. Neither
trace contains actual prohibited contact. Inspect the geometry and guard inputs
for the new closing rejection before deciding the next correction target; do
not weaken contact restrictions to raise the score. All artifacts above are
under `workspace/validation/`; no job or physical test is running.

The geometry investigation is complete in `grasp-clearance-inspection-002`.
On seed 50008, autonomous 016 succeeds just before the deadline, 020 obtains only
1.425 s hold, and 021 rejects during step 137. At rejection, enabled finger-body
collision geometry remains 6.05 mm from the true cube. The guard's reconstructed
start is 6.54 mm clear, but the requested end is only **2.83 mm clear**, below
the unchanged 3 mm buffer. Keeping the measured jaw opening in that hypothetical
end pose gives 5.35 mm clearance. Only one fingertip is touching, so the existing
bilateral-contact clamp is inactive. This supports a consistent guard decision,
not a reason to lower clearance. The original `inspection-001` included disabled
visual meshes in its minimum-distance summary; use `inspection-002`'s explicitly
collision-enabled pairs for contact interpretation. No prohibited physical
simulation contact occurs, and inspection operates on copied geometry only.

`closing-clearance-channels-001` checks six teacher-assisted interventions on
that development trajectory. Switching at the rejection step 137 is too late:
arm-only, gripper-only and both-channel corrections all reject immediately.
At step 117, teacher arm or both channels avoid rejection but achieve only
1.48/1.88 s holds; gripper-only still rejects. All six remain failures under the
unchanged two-second rule, with zero actual prohibited contacts or numerical
faults. Pre-intervention observations match the learner trace and all 188 queried
post-switch teacher labels replay exactly. These inspected development traces
are never training examples. The result supports earlier coordinated alignment
and closing, not a gripper-strength-only correction.

The following bounded comparison changes the trainable channels on the same
fixed four-source targeted dataset: candidate **022** starts from 016 and updates
the full original actor, keeping the normalizer frozen and adding no residual
head. Relative to gripper-only control 020, data, split roles, architecture,
initial checkpoint, seed 23, learning rate 0.0001, uniform sampling and 3,000-update
budget match. Training takes 10.83 s plus 3.50 s feature preparation; validation
selects update 2,800. Selected arm/gripper MSE is 0.001943/0.000970, versus
control 020's 0.002370/0.000998. No guard, task or evaluation condition changes.

`lift-imitation-runtime-024.json` records **17/20 autonomous guarded holds**, one
guard rejection, zero actual prohibited contacts and zero numerical faults.
Relative to frozen 011, 022 fixes 50000 and 50008 but regresses 50015. It improves
both same-data controls (020: 15/20, 021: 16/20), and separately completes all
**3/3 targeted training recoveries autonomously** with at least 2.035 s hold.
Bare CPU is 17/20 with two prohibited-contact episodes and does not qualify the
policy outside the guarded path. The 20 development cases are repeatedly used;
this is not a claim of general reliability. `joint-grasp-refinement-comparison-001`
verifies the comparison, original baseline hashes, dataset roles, normalizer,
geometry findings and unchanged evaluation protocol. Both the higher-success
022 checkpoint and zero-rejection 011 reference are preserved.

`joint-refined-policy-drift-001` confirms the remaining failures: 50007 never
gets bilateral grip and stalls in close with a 16.35 mm jaw coordinate; 50009
retains the approach/table-clearance rejection; 50015 first grasps at step 137
(6.85 s) and obtains only 1.59 s qualifying hold. Teacher queries never intervene
in these traces. No production code changed in this cycle, so unit tests were
not rerun; the bounded simulation comparisons and artifact audits complete
successfully. Held-out seeds remain unused and calibration-error tolerance
remains unmeasured. The saved persistent translation/rotation-bias protocol is
still required after nominal qualification, alongside separate jitter/delay
tests and later validation of the actual two-camera pipeline.

The subsequent `joint-training-screen-001` completes **33/34 autonomous holds**
on the same previously assigned training cases. It fixes all three targeted
failures but regresses 44016. This is a separate representation failure: true
yaw moves from 45.22 to 44.98 degrees, while canonical square-cube folding jumps
from -44.78 to +44.98 degrees. The resulting near-quarter-turn demand ends in
a finger-body/cube guard rejection, with no actual prohibited contact. The
three remaining development failures have no comparable yaw jump.

`yaw-continuity-probe-001` chooses the square-equivalent yaw nearest the last
delivered observation. With unchanged 022 weights, training case 44016 completes
a true two-second hold and 74.0 mm lift with no rejection/contact/numerical fault.
The production option is explicitly saved as `PolicyMeta.cube_yaw_mode`:
legacy/default `canonical` behavior is preserved, while `continuous` applies
the same observation-history rule in CPU evaluation and `PolicyRunner` and
resets each episode. Both retain the upright-square-cube approximation; this
does not correct calibration bias or add hidden teacher/physics information.
Unsupported imitation/batch refinement fails before training rather than silently
discarding the observation convention; continuous-mode demonstration support
must be explicit before any such fit. No new training occurred in this experiment.

Candidate **023** differs from 022 only in saved observation-mode metadata;
all actor/normalizer tensors are identical. `cube-yaw-comparison-001` pairs both
variants under the same guarded controller, seeds, 10 s deadline and true 2 s
hold criterion. Development remains **17/20 for both**, failing 50007/50009/50015.
Each has one development guard rejection and zero actual prohibited contacts
or numerical faults. On the matched training subset, 022 scores **33/34**, while
023 scores **34/34** with zero rejections, contacts or numerical faults. The full
paired comparison takes 198.62 s; the previous 022 development and training
results reproduce exactly. All actor/normalizer tensors are identical, parent
hashes and runtime sources are unchanged during the comparison, and the frozen
007/011/022 bundles still verify. Candidate 023 is preserved separately in
`workspace/validation/baselines/lift-imitation-023/`, SHA256 `7251688c`, with seven
verified artifact hashes. This is nominal diagnostic evidence, not calibration
tolerance or held-out qualification. The fix addresses a real input discontinuity
but does not explain the remaining development closing/clearance/late-grasp failures.
`cube-yaw-contracts-001.log` passes 48 checkpoint/continuity/runner/imitation
checks; `cube-yaw-environment-contracts-001.log` passes 48 CPU evaluation,
guarded-demonstration and NumPy/Torch checks. The source overlay is archived
under `cube-yaw-source-001/` against commit `90f4391`. No PPO, corrective-data
collection, held-out evaluation, calibration sweep or physical test was run in
that yaw-only cycle.

### Targeted recovery, gripper action decoding and nominal qualification

`joint-training-screen-002` selects the next 20 original-source training seeds
not previously screened. Frozen 022 and 023 both score **18/20**, with identical
closing stalls at 44026/44034, no bilateral grasp, and no rejection/contact/fault.
Those two full traces are identical between observation modes. The screen takes
100.64 s for 40 trials; no validation/development/held-out seeds enter the data.

`joint-training-recovery-002` runs the learner to steps 85/83 (20 steps beyond
the first slow-closing observation), then the observation-only teacher controls
both arm and gripper. Both recover with full holds, zero rejection/contact/fault.
All **328 labels** replay exactly from saved actor inputs; **45 frames** cover
the diagnosed slow-closing range. The replacement `lift-demos-targeted-joint-002`
changes only these two training episodes; every other episode and all validation
arrays remain byte-identical, and all source/split roles are retained.

`joint-recovery-fit-002` compares two 3,000-update, at-most-60-second full-actor
refinements from frozen 022, with unchanged normalization, seed, learning rate
and four-source split. Existing-data control **024** takes 18.13 s, selects update
2600, and scores **17/20 guarded with zero rejections/contacts/faults**. Targeted
candidate **025** takes 9.90 s, selects update 2500, but regresses to **15/20**,
with two rejections and zero actual contacts/faults. The targeted fit has lower
whole-episode validation prediction loss (0.002581 vs 0.002633), showing again
that this metric does not establish autonomous success. Reject 025 as a replacement.

Both fits solve the two new training stalls without intervention. In the six-case
training replay, 024 scores 5/6 (44006 has only 1.80 s hold), while 025 scores 4/6
(44013 stalls and 44016 hits the canonical-yaw guard failure). `joint-recovery-audit-002`
verifies matched settings, unchanged validation, frozen normalizers and all prior
baseline hashes. Continuous-yaw variant **026** has identical 024 tensors and
also scores 17/20 development with zero rejections and 5/6 selected training cases.
Both are preserved separately, with their regression recorded. The 026 traces show
50007 under-closing, 50009 stalled in approach with closed jaws, and 50014 first
grasping at step 146 with only 1.125 s final hold. Inspected development traces
remain excluded from training.

`gripper-template-audit-001` finds that **all 26,795 training and 6,602 validation
gripper labels** match exactly one of three observation-derived targets: open to
30 mm single-jaw coordinate, close 1 mm from the measured jaw, or command fully
closed. Maximum normalized residual is 6.62e-8; there are no unmatched or ambiguous
labels. `gripper-template-probe-001` projects only the learned gripper output to
its nearest supported target. Arm outputs and network weights are unchanged;
there is no online teacher/phase lookup. In a paired 41.90 s runtime comparison,
the original scores **17/20**, while decoding scores **19/20**. Only 50009 fails
with a guard rejection; both modes have zero prohibited contacts/faults.

The decoder is now an explicit saved `PolicyMeta.gripper_decoder` option, default
`identity` for old checkpoints. `a1x-templates-v1` is restricted to the audited
A1X cube-lift/jaw contract, preserves arm outputs, clips decoded actions to their
range, and rejects nonfinite inputs. Unsupported refinement fails before training
instead of discarding the mode. `gripper-decoder-contracts-001.log` passes 68
initial checks; the final bounds correction and runner suite pass 23 checks in
`gripper-decoder-contracts-002.log`. This is action decoding, not another fit.

Saved and reloaded **027** reproduces the probe's **19/20** through the actual
evaluator, including exact true hold durations (`gripper-decoder-package-audit-001`).
It is frozen before first use of the reserved seeds. `lift-heldout-027.json` then
records **20/20** on 51000-51019: every controller finishes, independent true holds
span 2.005-2.055 s, and there are zero guard rejections, prohibited contacts or
numerical faults. `lift-heldout-027-audit.json` verifies the exact seed set, frozen
checkpoint/bundle/runtime hashes and the complete success conditions. This passes
the nominal simulation gate only; no perception-error tolerance or physical result
is implied. Source/diagnostic overlay: `gripper-decoder-source-001/` on `e10acf2`.

`configs/a1x_lift_evaluation.json` fixes development seeds **50000-50019** and
held-out seeds **51000-51019**. A preflight found no prior use before frozen 027's
first qualification above. Keep them out of demonstrations, fitting and development
tuning. Every success requires the full 2 s hold through the guarded controller,
without teacher intervention, prohibited contacts or numerical faults. Subsequent
reuse is regression evidence, not a fresh holdout, even after a successful first run.

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
| 2026-09-27 | Derived geometry improves action prediction but ties autonomous success | `alignment-feature-comparison-001`: raw 015 exactly reproduces 012; feature 016 scores 16/20 with one rejection and zero actual contacts; 66 regression checks pass | Keep 011; diagnose the feature candidate's changed failures before scaling |
| 2026-09-27 | Removing teacher-irrelevant arm history does not improve grasping in a matched short fit | `previous-arm-dependence-001`: 156 invariant teacher labels; `arm-history-comparison-001`: 017 scores 14/20 guarded, 30 focused tests pass | Reject 017; do not infer an effective intervention from sensitivity alone |
| 2026-09-27 | Assisted channel comparison identifies gripper correction as the next candidate-specific priority | `alignment-channel-audit-001`: learned arms + teacher grip 4/4, teacher arms + learned grip 1/4, scripted control 4/4; no actual contacts | Collect at most 20 fresh training-only gripper-correction episodes for experimental 016; first preserve its arm mapping in refinement; baseline and held-out set stay untouched |
| 2026-09-27 | Continuous learner-arm / teacher-gripper pilot succeeds but misses the slowest closing states | `gripper-pilot-audit-001`: 17/20 assisted, all 3,032 labels verified; 668 closing frames; zero actual contacts/rejections; 14 training / 3 validation / 3 excluded | Keep all outcomes; diagnose coverage rather than scale from assisted success |
| 2026-09-27 | Gripper-only refinements preserve arm mapping but fail the autonomous comparison | `gripper-refinement-comparison-001`: existing-data control 018 16/20, added-data 019 15/20; one rejection each, no actual contacts; 61 tests pass | Reject both replacements; retained 011 remains 16/20 with zero rejections |
| 2026-09-27 | Matched closing-burst probe still misses the diagnosed slow-closing range | `gripper-burst-comparison-001`: 4/5 assisted for each schedule, 1,653 labels verified, no slow-closing proxy frames; 36 collection tests and two final counter checks pass | Screen existing training cases for actual learner failures, then collect recovery examples; no fit on this probe and no blind expansion |
| 2026-09-27 | Training-only screens find three actual learner failures and targeted teaching recovers them | `gripper-training-screen-001/002`: 14/14 and 17/20 autonomous; `gripper-training-recovery-001`: 3/3 assisted, 57 slow-closing frames, 492 labels replayed, original splits and all validation data preserved | Coverage is now relevant; successful assisted recovery remains distinct from learned success |
| 2026-09-27 | Targeted-data refinement fails autonomous comparison and the linear-head optimum still under-closes | `targeted-gripper-comparison-001`: 020 15/20 versus control 018 16/20, one rejection each, zero actual contacts; `gripper-linear-diagnostic-001`: optimum predicts 0.333 mm versus 1 mm target | Retain 011; test a separate nonlinear gripper residual on fixed data with frozen arm mapping, not longer training of the same uniform linear head |
| 2026-09-27 | Nonlinear gripper residual improves one development case but adds a guard rejection and fails targeted recoveries | `gripper-residual-comparison-001`: 021 16/20, two rejections, zero actual contacts; 0/3 selected training recoveries; 41 focused and 11 runner tests pass | Keep 011; investigate the new 50008 finger-body/cube clearance rejection before more fitting |
| 2026-09-27 | Grasp-clearance inspection supports earlier coordinated corrections without weakening the guard | `grasp-clearance-inspection-002`: requested 2.83 mm versus 3 mm margin, measured 6.05 mm; `closing-clearance-channels-001`: early arm correction avoids rejection, gripper-only does not; all six assisted probes remain below full hold | Keep guard unchanged; test joint arm/gripper refinement on existing training data, never the inspected development traces |
| 2026-09-27 | Joint refinement improves nominal development success and all targeted training recoveries | `joint-grasp-refinement-comparison-001`: 022 17/20, one rejection, zero actual contacts; 3/3 training recoveries; 10.83 s fit; frozen bundle SHA256 `a701802f` | Preserve 022 alongside zero-rejection 011; screen existing training cases for remaining failure coverage; held-out, calibration tolerance and physical validation remain pending |
| 2026-09-27 | Continuous cube-yaw input fixes a boundary failure with identical learned weights | `cube-yaw-comparison-001`: 023 ties 022 at 17/20 development and improves 33/34 to 34/34 assigned training cases; zero actual contacts; legacy results reproduce exactly; 96 tests pass; separate frozen 023 bundle | Screen the next bounded training subset for remaining closing/approach failures; require matching observation conventions before further teaching/fitting; calibration tolerance remains unmeasured |
| 2026-09-27 | Two training stalls are recovered, but targeted fitting regresses autonomous development | `joint-training-screen-002`: both 18/20; `joint-training-recovery-002`: 2/2 assisted, 328 labels replayed; matched fits 024/025: 17/20 vs 15/20; validation unchanged | Reject 025; preserve 024/026 with zero development rejections and their 44006 training regression; inspect action representation instead of scaling fitting |
| 2026-09-27 | Explicit learned-output gripper decoding raises nominal development success | `gripper-template-audit-001`: all 33,397 labels match three supported commands; paired probe and saved 027: 19/20 versus 17/20, one rejection, zero actual contacts; compatibility and controller checks pass | Freeze 027 before first held-out use; no teacher intervention, new fitting, guard relaxation or deadline change |
| 2026-09-27 | Frozen 027 passes first nominal held-out qualification | `lift-heldout-027-audit.json`: 20/20, true holds 2.005-2.055 s, zero rejections/contacts/faults; frozen source/checkpoint hashes verified | Begin separate perception-robustness stage with persistent translation/rotation bias, noise and delay; tolerance and physical validation remain pending |
| 2026-09-27 | Persistent calibration-error evaluation exposes premature completion | `perception-development-002`: 320 development trials in 425.11 s; 51 early completions and 18 guarded rejections, zero actual prohibited contacts/numerical faults; 42,021 input frames audited; 73 focused tests pass | Test a fixed uncertainty-aware completion bound before more fitting; no nonzero all-direction tolerance demonstrated at 2 mm / 0.5 degree; robustness holdout remains unused |
| 2026-09-27 | Saved completion uncertainty bound removes all early completions | 028 preserves actor tensors; paired sweep recovers 51 failures with no regressions, true holds 2.050-2.325 s, zero actual contacts; nominal holdout regression 20/20; 66 focused tests pass | Preferred development candidate frozen as `851b34fe`; target remaining clearance/grasp failures without discarding saved runtime contracts |
| 2026-09-27 | Initial nonzero calibration tolerance measured in development | +/-1 mm translation: 19/20 in every signed axis; +/-0.5 degree rotation: 18-19/20; separate factors, 20 starts each; +2 mm Y remains 17/20; all contacts/numerical faults zero | Screen fresh training-only starts and test observable teacher recovery; combined-error testing and fresh robustness holdout remain pending |
| 2026-09-27 | Fresh training screen isolates a recoverable biased-grasp failure | 40 initial and 40 targeted trials; seed 53076 passes nominally but rejects +2 mm Y. Teacher from start or after 48 learner steps succeeds; takeover at 56 fails. All labels replay from actor inputs | Teach early corrections; retain targeted-screen duplicate-close cleanup failure and reconciled evidence |
| 2026-09-27 | Calibration corrective examples and runtime-compatible refinement verified | `calibration-teaching-001`: 19/20 training and 18/20 validation assisted successes; all 40 attempts retained. Prior 33,397 frames keep split roles; one orientation frame explicitly converted and relabelled. 95 distinct focused checks pass, three PPO training cases excluded | Compare matched short fits before scaling; no teacher intervention in policy evaluation |
| 2026-09-27 | Short control fit outperforms added corrective data | `calibration-correction-comparison-001`: 183 autonomous trials, 235.27 s total; 029 scores 20/20 nominal, 20/20 at +2 mm Y, 19/20 at -0.5 degree Z versus 028 at 19/17/18 and corrective 030 at 19/19/16. Baseline reproduced exactly; zero prohibited contacts/numerical faults | Freeze 029; reserved perception qualification and user-authorized comparison PR; physical validation still deferred |

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
