# A1X offline validation

Updated 2026-09-26. Working branch: `a1x-twin`, starting commit `7edfceb`.
This records simulation and fake transport evidence. No physical commands were
sent by this validation task. Physical setup access is deferred by the user.

## Reproduce

WSL Ubuntu, Python 3.12.14 in `/home/kuba/.venvs/a1x-twin`; RTX 2000 Ada Laptop,
8188 MiB total; Windows driver 560.76. Installed PyTorch 2.11.0+cu126,
MuJoCo 3.14.0, mujoco-warp 3.14.0, Warp 1.17.0, NumPy 2.2.6.
Dependencies follow `constraints.txt`; PyTorch uses its cu126 wheel index.
The environment is isolated from the adjacent driver's working environment.

From the repository in WSL, with that environment activated:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../galaxeo-manipulators MUJOCO_GL=egl python -m pytest -q --tb=short
MUJOCO_GL=egl python scripts/validate_a1x_cpu.py --out workspace/validation/cpu-new --seeds 5 --video
python scripts/compare_a1x_physics.py --cpu workspace/validation/cpu-new --out workspace/validation/comparison-new.json
python scripts/validate_a1x_runtime.py --cpu workspace/validation/cpu-new --out workspace/validation/runtime-new.json
python scripts/validate_a1x_gpu.py --task reach --out workspace/validation/gpu-reach-new
python scripts/validate_a1x_gpu.py --task lift --out workspace/validation/gpu-lift-new
python scripts/train_a1x_candidate.py --out workspace/validation/reach-new --task reach --envs 512 --iters 300 --seconds 900 --seed 11
python scripts/evaluate_a1x_runtime.py workspace/validation/reach-new/policy/policy.pt --out workspace/validation/reach-runtime-new.json
```

Use new output paths. Tests use fake transport; do not launch the hardware UI or
select the `a1x` backend for these commands. The scripts contain no hardware calls.
Artifacts are local, ignored under `workspace/validation/`. GPU reports record
source hashes and package versions; they must accompany a checkpoint when shared.

## Control corrections

- Unknown G1 opening cannot initiate gripper motion. Calibrated feedback is
  required, and commands containing an invalid gripper request are rejected
  before any joint target changes. A command estimate is not measured opening.
- A1X's asynchronous sender latches a stop after command timeout (default 250 ms),
  stale/frozen feedback, or an I/O exception. Runtime STOP and loop failure cancel
  targets. Recovery requires explicit clearing with fresh healthy telemetry;
  old targets are discarded. This does not transmit a torque-release command.
- Common A1X target submission and the final command before sending are checked
  for joint limits and model collisions. The checker follows current object poses.
  Only fixed mounting contacts and intended fingertip/cube contacts are exempt.
  Unmodelled real obstacles and physical calibration errors remain outside this check.
  With bilateral fingertip contact on the allowed cube, a tighter squeeze setpoint
  is checked at measured jaw opening: the object prevents the fingers from reaching
  that setpoint. This preserves pressure commands without exempting palm collisions.
- G1 training commands use the configured 1.3-second full-travel rate. Actual
  physical actuator response is still unmeasured here.
- Observation noise uses radians for arm joints and metres for sliding jaws.
  The 0.3 mm jaw noise is a provisional simulation parameter, not a measurement.
- Physical policy start requires measured gripper feedback. UI camera-driven
  physical movement requires real, enabled cameras with trusted calibration.
- New A1X lift tasks require two seconds of consecutive success. CPU evaluation
  distinguishes a completed hold from merely being above the height threshold
  on the last frame. Existing saved policies retain their saved task definition.

Original-source reproduction: `baseline-regressions.log` contains 11 failures,
including stale sender motion, nonfinite requests, unknown gripper startup,
gripper rate mismatch and about 5.1 mm of accidental jaw observation noise.
Full suite after the control and training corrections: `offline-suite-003.log`,
**145 passed**, including rendering and GPU tests (180.01 s, with the candidate
training process running concurrently). GPU checks cover mixed-unit observation
noise, injected capacity overflow, matching CPU/GPU observations/rewards, timeout
reset and termination on a real folded-wrist/shoulder collision.
Six backend regression cases were added afterward: feedback endpoints and midpoint,
first calibrated squeeze while holding an object, sender I/O failure/recovery, and
reconnection. All **14 backend-contract cases passed** in
`a1x-backend-contracts-final.log` (0.87 s). These results describe checkpoint
`5de2292`; subsequent changes and their verification are recorded below.
The fake-CAN tests imported the adjacent `galaxeo` working tree (HEAD `d9fa677`);
that checkout contains team changes and was neither edited nor installed by this task.

## Scripted physics baseline

`cpu-lift-005`: five seeds, 3 cm / 30 g cube around x=0.35 m, y=0 with independent
position offsets within +/-1 cm. Scripted approach, close, lift 7 cm and hold 2 s.
All five succeeded; no prohibited contacts or MuJoCo warnings were recorded.
Final lift was about 68.7 mm, maximum jaw coupling error below 0.6 mm, peak contact
penetration about 1.6 mm, peak 12 contacts / 49 constraint rows. These values
describe this model and these trajectories; friction/dynamics remain provisional.

The TCP is about 29.89 mm behind the fingertip midpoint along its approach axis.
The script derives that offset from geometry while preserving the team's existing
TCP and calibration. A first attempt placed the TCP at the cube and drove the
fingers into the table (about 9.9 mm coupling error). That failure is retained in
`cpu-lift-001`; the successful baseline corrects the grasp target, not the model's
frame or stiffness. `cpu-lift-005/seed-0.mp4` was rendered and visually inspected.

`cpu-gpu-comparison-001.json`: replayed all 212 actions per seed from the successful
CPU traces in five Warp worlds. All five held for two seconds. Maximum differences:

| Quantity | Observed maximum |
|---|---:|
| Arm joint position | 0.000141 degrees |
| Controlled jaw position | 0.000108 mm |
| Cube position | 0.00381 mm |
| GPU jaw coupling error | 0.578 mm |
| Contacts, all five worlds / capacity | 60 / 320 |
| Constraint rows per world / capacity | 49 / 384 |
| Capacity overflow | 0 worlds |

GPU diagnostics accumulate every physics substep, including flags that could be
lost between policy ticks. Capacity overflow aborts a run. Solver iteration-limit
flags are reported separately through their raw bitmask; they do not by themselves
mean an allocation overflow. No universal CPU/GPU tolerance is inferred from five
successful scripted trajectories.

`runtime-lift-004.json`: the same five scripted action sequences through
`PolicyRunner -> Twin -> SafetySupervisor -> SceneBackend`, with policy at 20 Hz
and runtime at 50 Hz. **5/5 two-second holds**, no prohibited contacts or warnings;
final heights 68.77-68.84 mm. Runtime tick p99 was 0.97-1.18 ms on this machine.
The first replay correctly exposed virtual overclosure into the solid cube;
bilateral-contact geometry handling fixed this without relaxing palm/table checks.

## GPU capacity and learning status

| Run | Worlds x iterations | Transitions | Peak total GPU memory used | Outcome |
|---|---:|---:|---:|---|
| `gpu-reach-001` | 256 x 3 | 18,432 | 1286.5 MiB | Checkpoint reload + 2 CPU episodes passed; 0 task successes |
| `gpu-lift-001` | 256 x 3 | 18,432 | 1418.5 MiB | Checkpoint reload + 2 CPU episodes passed; 0 task successes |

These were plumbing checks before the new capacity instrumentation and two-second
hold default. They establish that this GPU can run small A1X training batches.
They do not establish a usable learned policy. Memory includes Warp and desktop
usage; Torch's own allocator count alone substantially understates total usage.

`reach-candidate-001` attempted 512 worlds x 300 iterations, at most 900 s, seed 11,
with 20 nominal and 20 randomized CPU evaluation episodes reserved at seeds
20000-20019. It stopped after 18.2 s on an actual constraint-capacity overflow:
69 rows required, 64 allocated, one world affected. Peak total GPU use 1336.5 MiB.
No evaluated checkpoint resulted. The report/manifest and failure are retained.
A1X reach now receives the same 64-contact / 384-row capacity as A1X lift.

A1X CPU/GPU environments now terminate an episode with reward -10 on prohibited
model contact, checked every physics substep. Intended fingertip/cube and fixed
mounting contacts remain allowed. Training/evaluation report collision episodes
and numerical failures separately; numerical failure stops bounded candidates.
These constraints were absent from the original training environment even though
the physical runtime needs them.

`reach-candidate-002`: same 512-world, 300-iteration, 900-second ceiling and seed 11
after the capacity/collision corrections. Completed 3,686,400 transitions in 283.5 s
including setup; training plus CPU evaluation took 298.1 s. Peak total GPU memory
was **1592.5 MiB**, including the concurrent test process. No numerical failures or
capacity overflow. Training accumulated 849 collision-ended exploration episodes;
they were penalized/reset, not counted as success. Final training success was 15.2%.

Independent evaluation (saved/reloaded deterministic policy, seeds 20000-20019):

| Evaluation path | Final successes | Other results |
|---|---:|---|
| CPU, nominal physics/perception | 9/20 | Median final error 27.3 mm; no collisions or numerical failures |
| CPU, provisional randomization | 10/20 | Median final error 21.4 mm; no collisions or numerical failures |
| `PolicyRunner -> Twin -> SafetySupervisor` | 6/20 | All 20 completed; no prohibited contacts or numerical failures |

Runtime evidence is `reach-runtime-003.json`. The previous evaluator report `002`
mistakenly treated the normal end-of-episode message as a fault; it is superseded.
Success here means final TCP error below 20 mm; reach has no sustained-hold
requirement. This checkpoint demonstrates learning, but is **not ready for a demo
or hardware deployment**. The scripted lift remains the stronger current baseline.
The improved candidate below supersedes it for reaching experiments.

## Follow-up geometry, perception and learned reach

`a1x-geometry-001.log`: two independent geometry checks pass. URDF forward
kinematics agree with every MuJoCo robot link across 100 random joint poses;
joint limits, jaw unit/sign conversion, the 29.89 mm TCP/fingertip offset and
normalized generated XML agree. This validates conversion from vendor assets,
not the physical assembly, calibration offsets or installed accessories.

`reach-candidate-003` fine-tunes candidate 002 for 500 iterations, 512 worlds,
seed 14, 900 s ceiling and initial action std 0.3. It restores actor, critic and
normalizer, with a fresh optimizer/action variance. Additional training:
6,144,000 transitions, 464.2 s including setup, 477.5 s through CPU evaluation,
1,592.5 MiB peak total GPU memory. No numerical failures or capacity overflow.

| Seed set | Nominal CPU | Default-randomized CPU | Guarded runtime |
|---|---:|---:|---:|
| 21000-21019, validation | 19/20 | 20/20 | 19/20; 1 safe rejection |
| 30000-30019, fresh policy evaluation | 19/20 | 20/20 | 19/20; 1 safe rejection |

CPU median final errors were about 1.5-1.8 mm. Each nominal CPU set had one
collision-ended episode. The old runtime guard also allowed one table contact
in each set. Seed 21007 exposed 0.1194 mm finger/table penetration even though
the commanded endpoint was clear: the measured tracking pose was not included
when a short path used only its endpoint. Evidence: `reach-contact-21007.json`.

The guard now includes the measured start and uses 3 mm environmental clearance
on a private model copy. Simulated dynamics remain unchanged. Positive internal
housing clearances are exempt from that buffer; actual self penetration remains
prohibited. Both replays now have zero actual contacts and one safe rejection:
`reach-runtime-buffered-001.json`, `reach-holdout-buffered-003.json`. Since the
guard changed after inspecting these sets, they are regression evidence for the
guard, not untouched guard holdouts. `clearance-003.log`: eight focused tests
pass. All five scripted lifts still pass (`runtime-lift-buffered-001.json`).
This 3 mm choice is provisional and is not a measured physical error bound.

After freezing the policy and guard, a fresh seed set 32000-32019 passed **20/20**
through the common controller (`reach-final-holdout-003.json`). The evaluator
observes every 5 ms physics substep: zero prohibited contacts or numerical
failures. The same monitor passes all five scripted lifts, with continuous
bilateral contact above 6 cm for 3.815-3.825 s, and maximum jaw coupling below
0.565 mm (`runtime-lift-substeps-001.json`). Learned reaching is still a narrow
simulation result; this does not qualify a hardware policy.

Physical-mode policy observations now refuse stale, simulated or unknown cube
sources. Observation loss and exceptions after movement hold measured arm pose
while preserving an existing squeeze. Current camera/held-object estimates
update free-object collision geometry using the saved task's cube size, while
retaining unrelated modeled obstacles. Simulation truth is not overwritten by
camera estimates. Tests: `physical-observation-loss-002.log` (12 passed),
`observed-geometry-001.log` (38 passed), `observed-geometry-002.log` (8 passed).
These are fake/simulated observations; live camera operation remains untested.
The complete updated suite passed **166 tests** in 193.41 s
(`offline-suite-004.log`), including GPU and rendering. The later evaluation
script changes were exercised by the fresh reaching set and scripted lifts above.

## Lift learning experiments

`gpu-lift-1024/report.json`: 1,024 worlds x 3 iterations, 73,728 transitions,
2,074.5 MiB peak total GPU memory, approximately 38-40k transitions/s after
warm-up. No overflow; peak 4,098/65,536 contacts and 23/384 constraint rows.
Save/reload and two CPU episodes passed; this smoke run did not learn a lift.

`lift-candidate-001` is the bounded original-reward baseline: 1,024 worlds,
1,000 iterations or 900 seconds, seed 17, CPU evaluation seeds 22000-22019.
Its manifest preserves the source/task loaded before reward changes below.
Do not overwrite it or relabel it as the corrected experiment.
It completed at its 900 s training limit: 606 iterations, 14,893,056 transitions,
2,074.5 MiB peak total GPU memory; 979.5 s including CPU evaluation. Success was
0/20 nominal and 0/20 randomized, with no evaluation collisions or numerical
failures. Training had 226 collision-ended episodes and no capacity overflow.

The original lift approach reward uses a 5 cm scale centered on the TCP. For
100 actual A1X starting poses, its float32 value is exactly zero in 99 cases
and at most 5.96e-8. The A1X starts roughly half a metre from the cube, and its
gripping surfaces are 29.89 mm ahead of the TCP. A new A1X lift task therefore
uses a coarse 20 cm plus fine 5 cm approach reward centered on the fingertip
midpoint. On the same starts, its range is 0.00100-0.00757 (median 0.00334),
with no exact zeros. This gives a learning signal; it does not prove learning.

The calibrated TCP frame and observation layout are unchanged. Saved task
definitions and SO-101 defaults retain the legacy reward. Success still requires
bilateral contact, over 6 cm height and a consecutive 2 s hold. CPU/GPU reward
and fingertip position agreement plus legacy compatibility pass 11 focused
tests (`lift-reward-contracts-001.log`); diagnostic samples are in
`lift-reward-diagnosis.json`.

`lift-candidate-002`: same seed 17, 1,024 worlds, at most 606 iterations / 900 s.
The time limit ended this run at 476 iterations (11,698,176 transitions), so this
is a same-time-ceiling comparison, not equal training steps. Total elapsed
935.97 s, peak total GPU memory 2,074.5 MiB. No numerical failures or capacity
overflow; 7,990 collision-ended exploration episodes. CPU results remained 0/20
nominal and 0/20 randomized, with no evaluation collisions. At iteration 150,
five CPU traces showed approach within 25.9-36.5 mm and fully open jaws, with no
bilateral grasp (`lift-inspection-002a`). Reward growth alone did not solve lift.

While that run was active, evaluation was corrected to preserve the saved
policy's folded cube-yaw representation even when randomization is disabled.
Previously nominal CPU evaluation supplied raw yaw, unlike the physical runner.
Two new tests compare actual initial CPU and runtime observations; all 11 focused
RL tests passed (`evaluation-yaw-contract-001.log`). Both earlier lift reports
used the old evaluator loaded at process start and are retained as such.

The corrected reevaluation is saved separately as `lift-runtime-002.json`:
0/20 nominal CPU, 0/20 randomized CPU and 0/20 common-controller lifts. All use
the saved folded-yaw convention. Runtime monitoring found no prohibited contact
or numerical failure, but no sustained lift either. Checkpoint SHA256:
`c2513640ffd2f36fab8edd0eafe07644315f69f9e890af10ce1f6ebc1a2efcb5`.

`collect_a1x_lift_demos.py` provides a bounded scripted-demonstration experiment.
It samples the full original lift region, cube yaw and noisy starting joints,
records observations/actions and retains failed cases. Its own first attempt
had an incorrect world/base-frame offset; `lift-demos-001` is invalid and retained.
The corrected collector initially completed 15/20 (`002`, `003`), then 16/20
when IK searched within configured controller bounds (`004`). A 7.5 cm approach
waypoint instead of 10 cm completed 18/20 (`005`); two collision-ended episodes
remain failures, not training successes. No limits or success thresholds were
relaxed. Successful trajectories may seed imitation; no imitator has yet been
trained, and these broader scripted sequences have not passed the runtime guard.

`reach-video-003/seed-32000.mp4` is a labeled, visually inspected CPU replay of
the learned reach candidate (1.05 mm final error, zero collisions). Reproduce
with `scripts/render_a1x_candidate.py`; the report includes checkpoint hash.

## Additional H100 compute

### Demonstration warm start

`lift-demos-006` collected seeds 41000-41199 in 45.60 s: 196/200 scripted lifts
held above 6 cm for two seconds; four ended on collisions, zero numerical faults.
All attempts remain recorded. These are development demonstrations, not a learned
holdout or guarded-runtime result. Only successful collision-free episodes enter
`rl/imitation.py`; it rejects invalid shapes, nonfinite samples, out-of-range
actions and repeated episode IDs. The training/validation split is by whole
episode, and validation observations never enter normalization.

`lift-imitation-001`: 5,000 supervised updates, seed 23, CPU, 120 s ceiling;
actual training 15.81 s. The checkpoint at update 4,700 had the lowest held-out
episode action error. This score is not manipulation success. Reloaded physics
evaluation on seeds 50000-50019 achieved 0/20 nominal and 2/20 randomized holds,
with 6 and 7 collision-ended episodes respectively, no numerical failures.
Three nominal traces in `lift-imitation-inspection-001` show bilateral grasp
and maximum lifts of 51.6-52.8 mm, below the required 60 mm.

`lift-imitation-runtime-001.json`: five same-seed controller trials, zero lifts;
all stopped on collision-guard rejection, with zero actual prohibited contacts
at every physics substep. This policy must not be deployed. Checkpoint SHA256:
`031af15422f82427597af2c1386162198679f47baa1ca5054768fab3f0b3185e`.
Six data/training tests passed (`imitation-contracts-001.log`).
The later `imitation-contracts-002.log` passes all seven, including an actual
GPU PPO iteration proving critic-only warm-up preserves every actor and
observation-normalizer value exactly.

The frozen refinement experiment used this exact checkpoint, 4,096 GPU worlds,
PPO seed 29, initial action standard deviation 0.15, 20 critic-only warm-up
iterations, and at most 500 iterations / 300 training seconds. The outer process
limit was 480 s including setup/evaluation. A three-iteration 4,096-world
compatibility check ran first. Evaluation used the same development seeds
50000-50019; final qualification still needs fresh seeds. Numerical/capacity
faults were stop conditions; height/hold requirements and runtime guards remained
unchanged.

`brev/h100-lift-4096-001` passed at 294,912 transitions, 4,266.9 MiB peak total
GPU memory, zero overflow and 35/384 peak constraint rows. Warm iterations ran
at 111-156k transitions/s; the total including CPU evaluation was 11.13 s.
The untrained smoke checkpoint had 0/2 CPU lifts and one collision-ended episode.

`brev/h100-lift-imitation-ppo-001` completed 237 iterations / 23,298,048
transitions in 300.51 training seconds, 321.85 s including evaluation; peak total
GPU memory 4,268.9 MiB. Source snapshot: `0538669987de325f8c10e1afe5fef033332b0658`,
archive SHA256 `07d3d78f50fb7ada4d7e7aa2d962aa78ff518af11d4520597d3f0181c0a6acc6`.
Training had 24,710 collision-ended episodes, zero numerical failures and zero
overflow. Final CPU results were 0/20 nominal and randomized, without evaluation
collisions. The grasp behavior was lost despite early training successes.
The historical manifest's generic continuation string says the critic was
restored; the imitation checkpoint had no critic. The actual configured 20
critic-only iterations initialized it; future runner descriptions are corrected.

`imitation-refinement-ablation-001.json` exchanges only the observation
normalizers between the frozen pre/post-refinement policies. On five development
seeds, the original actor with the new normalizer collides in 5/5, and the new
actor with the old normalizer also collides in 5/5. Both full checkpoints fail
the lift. Normalization changed enough to break the old actor, but this does
not establish it as the sole cause of the failed refinement. Future work should
test fixed normalization and an imitation constraint during PPO, not just run
the same configuration longer.

### State-triggered examples and remaining controller gap

The timed expert had a hidden-clock problem: in all 196 successful episodes,
the observations just before/at each descend, close and lift transition differed
by less than 0.001 in every coordinate while an action channel jumped by more
than 0.5. Median observation changes were 2.33e-10, 1.53e-13 and 2.79e-5
respectively (`demo-phase-boundaries-001.json`; mixed observation units).
Thus the deterministic observation-only policy was given conflicting labels
for effectively the same state. This was a problem in the newly generated
demonstrations, not evidence of an upstream hardware issue.

The collector's optional `--state-triggered` mode advances when the commanded
arm waypoint is reached and measured arm error is below 0.02 rad. Closure also
requires bilateral contact and a completed closing setpoint. Phase time limits
remain enforced; failure to reach a phase condition is retained as failure.
This changes example generation, not the policy observation schema, physical
limits, 6 cm lift requirement or 2 s hold. `lift-demos-state-001` passed its
first 20/20 scripted trials. `lift-demos-state-002` repeats the same 200 seeds
as the timed dataset: 196/200 holds, four collision failures, zero numerical
faults, 32.54 s. The successful episodes still need separate runtime validation.

`lift-imitation-002` uses the same learner, seed 23, 5,000-update / 120 s ceiling
and development evaluation seeds as candidate 001, with the state-triggered
dataset. Training took 15.62 s; update 5,000 had the best action validation loss
(0.009824 versus 0.020098 for candidate 001). Reloaded CPU results: **16/20
nominal holds**, three collision-ended episodes; **5/20 randomized holds**,
12 collision-ended episodes; no numerical failures. These are development
results, not a final untouched holdout. Checkpoint SHA256:
`eb06853be2011ca11ddfa17551ca0276a2c4e044d2b256ea6b860906563af887`.

The same candidate through `PolicyRunner -> Twin -> SafetySupervisor` achieved
**0/20** (`lift-imitation-runtime-002.json`): every trial stopped on finger-body
versus cube guard rejection. Every-substep monitoring found zero actual
prohibited contacts and zero numerical failures. Training must produce paths
accepted by that guard; bare-environment successes are insufficient.

`lift-imitation-video-002/seed-50001.mp4` is a visually inspected, labeled
successful bare-CPU replay (89 frames, 74.03 mm final lift, completed 2 s hold).
It is imitation, not PPO performance, and does not pass the guarded runtime.

### Remote environment and isolation

The user supplied Brev organization `wysocki-55ff9e-nih0` and machine
`empty-peach-tortoise` (ID `cdeczltwl`). Read-only inspection confirms hostname
`brev-cdeczltwl`, H100 PCIe with 81,559 MiB and driver 570.148.08. Brev CLI v0.6.335
can list a specific organization without changing context, but its connection
lookup uses the active organization. The other project remains active.

The task read only the designated workspace metadata through the same official
Brev API used by the CLI, using existing authentication without printing tokens.
It verified organization and instance identities, then wrote a separate local
SSH config/known-hosts file under `workspace/validation/brev/`. No shared config
or organization switch was required. WSL shared file hashes before/after:

- `/home/kuba/.brev/ssh_config`: `64a3b978e9f0bb1c53530f103c72e239fdd4fe99e4630ab28e0eabcea96c8cf9`
- `/home/kuba/.ssh/config`: `1d4cd26da5cc01120cc0576762e9dc28e40c614dfeffd67775ab6bbc6898a3c0`

Remote files and dependencies belong only under `/home/ubuntu/a1x-twin-training`.
Transfer repository source/assets and explicit experiment artifacts, excluding
local credentials, live workspace/calibration files and the adjacent driver.
Snapshot manifests accompany non-Git source exports. Start with a bounded smoke
check; the longer training budget is pending the user's answer.

The remote isolated environment is `/home/ubuntu/a1x-twin-training/venv`, Python
3.12.14, Torch 2.11.0+cu126, MuJoCo / MuJoCo Warp 3.14.0 and Warp 1.17.0. Its
source export is commit `d7497a34a873f9aaeff7fddf1d8142683b14ae77`; all 161
exported files are hashed in `SOURCE_SNAPSHOT.json`. The archive SHA256, verified
after transfer, is
`6fdcd7c4b284b9dff66c5b3b89d0c51bb766ec6b2638cb6a13bbf7c83de68483`.
The tracked assets include the repository's two camera-intrinsic text files;
these are not a verified copy of the current physical calibration and are not
used for state-vector training. No live camera or hardware access was attempted.

`MUJOCO_GL=egl` failed during import because the host lacks an EGL loader
(`brev/h100-lift-1024-001.log`). Training works with `MUJOCO_GL=disable`, without
changing system libraries or drivers. Remote image rendering remains unverified.

The corrected smoke run `brev/h100-lift-1024-002` passed: 1,024 worlds x three
iterations, 73,728 transitions, checkpoint reload and two CPU episodes. Total
elapsed 119.49 s includes initial compilation and CPU evaluation. Warm iterations
measured approximately 38-44k transitions/s; this small workload does not yet
show a substantial speed advantage over the laptop. Peak total GPU use was
1,708.9 MiB, with 4,099/65,536 peak contacts, 26/384 constraint rows and zero
overflow. CPU evaluation had no collisions/numerical failures and 0/2 lifts,
as expected from a plumbing check. Report and checkpoint were copied locally.

Reproduce from WSL in the local repository, using a new output directory:

```bash
ssh -F workspace/validation/brev/ssh_config a1x-h100 \
  'cd /home/ubuntu/a1x-twin-training/source && PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=disable timeout 600 ../venv/bin/python scripts/validate_a1x_gpu.py --task lift --envs 1024 --iters 3 --out workspace/validation/NEW_RUN'
```

`brev/h100-cpu-contracts-001.log`: 28 tests passed in 4.32 s, covering geometry,
RL action/observation/reward conventions, collision checks and physical-observation
rejection using simulated inputs. Tests ran with rendering disabled and
`OMP_NUM_THREADS=2`; they do not test real hardware. The adjacent driver was not
installed or transferred, and remote renderer/backend checks are not claimed.

`brev/h100-gpu-contracts-001.log`: both GPU tests passed in 13.86 s. These check
CPU/GPU observation and reward agreement, resets, actual collision termination,
mixed angular/linear sensor noise and fatal handling of injected capacity overflow.

A subsequent shared-config check observed a different Brev-generated file hash
(`2514ddf1917e125bc0acabbe35baeee93013e5d06dabd94710752ce2cb1ad391`)
while another project's `brev exec` process was active. The main `.ssh/config`
hash remained unchanged. This task did not write either shared file or switch
organizations. Preserve concurrent changes; never restore the earlier snapshot.
The task-local SSH configuration is independent of those refreshes.

## Remaining validation

- Preserve the verified implementation checkpoint; relevant historical README/test
  claims have been reconciled with the evidence above.
- Scripted runtime path passed; retain the regression and evaluate learned
  candidates through that same controller path.
- Extend grasp/contact stress coverage as needed; the five successful lift traces
  remain a narrow nominal comparison, not physical dynamics validation.
- Retain the improved reach candidate as simulation-only. Train lift with explicit experiment scope,
  recorded limits, and independent sustained-hold/runtime evaluation.
- Preserve and verify live calibration/configuration, actual G1 feedback, dynamics,
  camera latency, and physical success when the user returns to hardware testing.

Full execution-plan completion still requires the physical demonstration. Existing
read-only hardware findings from other tasks do not substitute for that work.
