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
`a1x-backend-contracts-final.log` (0.87 s). No source changes followed the full-suite
run; the later changes are tests, validation scripts and documentation.
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
The next experiment should improve reaching precision/stability and repeat the
independent controller evaluation before proceeding to learned lift.

## Remaining validation

- Preserve the verified implementation checkpoint; relevant historical README/test
  claims have been reconciled with the evidence above.
- Scripted runtime path passed; retain the regression and evaluate learned
  candidates through that same controller path.
- Extend grasp/contact stress coverage as needed; the five successful lift traces
  remain a narrow nominal comparison, not physical dynamics validation.
- Improve the reach candidate. Train lift only with explicit experiment scope,
  recorded limits, and independent sustained-hold/runtime evaluation.
- Preserve and verify live calibration/configuration, actual G1 feedback, dynamics,
  camera latency, and physical success when the user returns to hardware testing.

Full execution-plan completion still requires the physical demonstration. Existing
read-only hardware findings from other tasks do not substitute for that work.
