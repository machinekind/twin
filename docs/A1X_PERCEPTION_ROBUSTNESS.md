# A1X perception robustness protocol

Status: next active stage, simulation only; no calibration-error tolerance measured yet.
The user explicitly requires persistent camera-to-arm translation and rotation
errors as well as pose noise and delay. Frozen candidate 027 has now passed
nominal qualification: 19/20 development and 20/20 first held-out lifts, full
true two-second holds and zero prohibited contacts. Use its preserved checkpoint
(`workspace/validation/baselines/lift-imitation-027/policy.pt`, SHA256 `22d79622`),
continuous cube yaw and explicit `a1x-templates-v1` gripper decoder. Reserve a
separate robustness evaluation set before corruption fitting. The previously
used nominal held-out set is now regression evidence, never fresh qualification.

## Injection and independent truth

- Keep the real simulated cube pose, dynamics, robot state and contact geometry
  unchanged. Wrap the pose supplied to `PolicyRunner` through `cube_provider`.
- Express a fixed calibration error as a rigid transform `(R_bias, t_bias)` in
  the arm-base frame. Apply it to the delayed true object pose:
  `p_biased = R_bias @ p_true(t-delay) + t_bias` and
  `R_biased = R_bias @ R_true(t-delay)`.
- Add independent zero-mean position/orientation noise after that transform.
  Corrupt the full pose before the runner's existing yaw folding. Calibration
  rotation changes both object orientation and apparent position about the base
  origin; rotating only the object orientation is a different test.
- Record the checkpoint's `cube_yaw_mode` and keep it fixed in each comparison.
  The optional continuous mode selects the nearest equivalent square-cube yaw
  using delivered observation history; it does not estimate or remove calibration
  bias. Both modes retain the existing upright-cube approximation and discard
  roll/pitch in the actor's orientation input. Inject the full transform first:
  its effect on perceived position must survive even when tilt is discarded.
  Include starts near the +/-45-degree yaw boundary. Continuity within an episode
  does not eliminate a biased first-frame choice of grasp orientation.
- Keep the bias constant throughout each episode and across the paired trials
  for a given condition. Do not redraw it every frame or average opposite signs
  into a single apparent zero-bias result. Noise and delay are separate factors.
- Feed the independent `PhysicsMonitor` only true simulation state. Count success
  from a true lift above 60 mm, bilateral contact for the full 2 s, zero prohibited
  contacts at every physics substep, and no controller/numerical fault. A biased
  controller's premature success declaration must not pass this check.
- Log true, delayed and policy-supplied pose with timestamps, applied transform,
  noise seed, requested/effective delay, and controller stop reason. The actual
  guarded controller remains in the loop, with no teacher intervention.

## Proposed diagnostic ranges

Freeze each small batch's exact configuration before execution. These are initial
exploration ranges, not measured camera errors or promised operating limits.

| Factor | Magnitudes | Direction / interpretation |
|---|---|---|
| Persistent translation | 0, 2, 5, 10, 20 mm | Separate positive and negative base X/Y/Z offsets |
| Persistent rotation | 0, 0.5, 1, 2, 5 degrees | Separate positive and negative rotations about base X/Y/Z |
| Position noise | 0, 1, 2, 5 mm standard deviation | Independent zero-mean samples; log per-axis convention |
| Orientation noise | 0, 0.5, 1, 2 degrees standard deviation | Independent rotation-vector noise; log per-axis convention |
| Delay | 0, 50, 100, 200 ms | Timestamped stale pose; verify actual delivered age |

First measure each factor separately against zero corruption. Then measure
specified translation-plus-rotation combinations and combinations with noise/delay,
especially near the observed failure boundary. Do not infer combined tolerance
by combining separate one-factor limits. A two-camera fused-pose bias is only a
surrogate here; this does not reproduce the actual cameras' estimation pipeline.

## Comparison, training and reporting

- Use the same frozen policy, object starts and truth-based success criterion
  across magnitudes/directions. Keep the nominal development cases explicitly
  labeled as development. Reserve a fresh robustness evaluation set before any
  robustness training; do not reuse an inspected qualification set as fresh proof.
- Run small bounded batches, initially at most 20 episodes per condition. Keep
  failures, rejections and incomplete batches. Record checkpoint/config/source
  hashes, seeds and denominators; success is not a prediction-loss metric.
- Plot success rate against translation magnitude, rotation angle, noise and
  delay, with separate directional curves and trial counts/uncertainty. Report
  true hold durations, actual contacts and guarded rejections alongside success.
- Proposed empirical tolerance: the largest tested magnitude for which every
  tested direction and all smaller tested magnitudes achieve at least 18/20 true
  successes with zero prohibited contacts. State the tested poses/directions,
  sample size and combined-error conditions. Do not interpolate an untested bound
  or call this a physical calibration tolerance. Report zero/undemonstrated
  tolerance if the zero-error policy does not meet the criterion.
- Train on explicitly separated development corruptions only after diagnosing
  the measured curve. Preserve nominal performance and rerun independent nominal
  and robustness evaluations for each proposed replacement checkpoint.

Physical validation remains deferred. When access resumes, measure the actual
two-camera calibration residuals and pose latency against known arm-frame points,
then compare them with the simulated curves. Jitter tests cannot certify persistent
calibration accuracy, and simulated tolerance cannot certify the physical setup.
