# A1X perception robustness protocol

Status: active development measurements, simulation only; no nonzero tolerance
across all tested calibration-error directions demonstrated yet.
The user explicitly requires persistent camera-to-arm translation and rotation
errors as well as pose noise and delay. Frozen candidate 027 has now passed
nominal qualification: 19/20 development and 20/20 first held-out lifts, full
true two-second holds and zero prohibited contacts. Use its preserved checkpoint
(`workspace/validation/baselines/lift-imitation-027/policy.pt`, SHA256 `22d79622`),
continuous cube yaw and explicit `a1x-templates-v1` gripper decoder. The separate
robustness holdout is reserved at seeds 52000-52019; training uses 53000-53999.
Development remains 50000-50019. The previously
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

## Implemented evaluation and information boundary

`rl/perception.py` applies the rigid transform and separate jitter to a causal
pose history. Repeated reads at the same simulation timestamp return the same
sample. `RuntimeLiftEnv` installs it before the first observation, including
continuous-yaw initialization. The independent `PhysicsMonitor` remains unchanged.
The scene collision guard also retains its true simulated geometry: these runs
test supplied-pose error, not an erroneous collision map or robot/table model.

`scripts/evaluate_a1x_perception.py` freezes each batch's checkpoint hash, seeds,
conditions, source archive and runtime versions before execution. It saves full
pose/action traces, partial results on interruption, and per-condition counts
with Wilson intervals. A baseline control must reproduce saved success, true
holds, final joint positions/commands, height, coupling and underlying stop cause.
It distinguishes the adapter's status prefix from the original evaluator's
exception prefix without ignoring a changed guard cause.

Example bounded development invocation, in the existing Python environment:

```sh
python scripts/evaluate_a1x_perception.py \
  workspace/validation/baselines/lift-imitation-027/policy.pt \
  --out workspace/validation/perception-development-002 \
  --translation-mm 2 --rotation-deg 0.5 \
  --position-noise-mm 1 --rotation-noise-deg 0.5 --delay-ms 50 \
  --seconds 600
```

Use a new output directory for any rerun. Both demonstration collection and
imitation training now reject development cases as fitting data and protect both
nominal and robustness holdouts. The robustness holdout has not been evaluated.

`workspace/validation/calibration-observability-001.json` records eight
pre-contact counterfactual pairs from four training-only starts. Opposite 2/5 mm
horizontal biases and corresponding object displacements produce identical
36-element policy observations and identical learned actions, while the true
grasp centres differ by 4/10 mm and truth-informed teacher actions differ.
These are diagnostic counterfactuals, not executed demonstrations or training
labels. Exact oracle corrections are not identifiable from the present input
alone. Wider grasp margins, contact feedback/history or an independent frame
reference can still help; this is not a claim that robust grasping is impossible.
Known tabletop geometry can constrain vertical bias separately.

## First frozen-policy development results, 2026-09-27

`workspace/validation/perception-development-002/report.json` contains 320 trials
in 425.11 s, 20 identical development starts for each of 16 conditions. The
zero-error control reproduces saved candidate 027 exactly under the comparisons
above. The first attempt (`perception-development-001`) stopped after its control
because of the stop-message prefix difference; it is retained as a failed audit
attempt, not merged into these denominators. No fitting or policy change occurred.

| Persistent error, applied alone | +X | -X | +Y | -Y | +Z | -Z |
|---|---:|---:|---:|---:|---:|---:|
| 2 mm translation | 20/20 | 19/20 | 17/20 | 19/20 | 1/20 | 19/20 |
| 0.5 degree rotation about base origin | 15/20 | 15/20 | 19/20 | 0/20 | 19/20 | 18/20 |

Separate factors: zero error **19/20**; position jitter with 1 mm per-axis SD
**13/20**; rotation-vector jitter with 0.5 degree per-axis SD **19/20**; requested
50 ms delay **19/20**. With the current capture cadence, non-warmup delivered
ages are **60 ms (1,322 frames) or 100 ms (1,303 frames)**. Initial frames explicitly
mark shorter-history warmup. Do not interpret the requested value as exact.
There are zero prohibited contacts and zero numerical failures in all 320 trials.
These small development samples and one noise realization per start are not
statistical certification. Factors have not yet been combined.

Of 69 failures, **51 are premature controller completion**: the true continuous
hold is 1.900-1.995 s even though the estimated-pose timer reaches two seconds.
The +2 mm Z condition accounts for 18 of these; -0.5 degree Y accounts for 19.
Rotation about Y changes the apparent height of objects forward of the base,
so orientation jitter alone would not expose this failure. The true scorer
continues to reject every shortened hold; no acceptance threshold was relaxed.
The other **18 failures are collision-guard rejections**, including one predicted
table/finger collision. Fifteen involve the existing difficult start 50009;
three additional rejections occur at 50014/50015 (+Y translation) and 50012
(-Z rotation). Early bilateral sensor contact alone is not proof of a secure grasp.

**Achieved calibration tolerance:** only zero error currently passes the stated
criterion across all tested directions. No nonzero all-direction tolerance is
demonstrated at 2 mm or 0.5 degree. This does not mean arbitrarily small errors
always fail: intermediate magnitudes are untested. Do not extrapolate a physical
or combined-error limit from these results.

`workspace/validation/perception-analysis-001/` contains `audit.json`,
`curves.csv`, and PNG/PDF curves. The audit checks 42,021 logged input frames,
identical true initial states across conditions, rigid-transform/noise math,
causal delay selection, supplied policy position, true success, and source/trace
hashes. The frozen checkpoint is unchanged. `calibration-observability-001.json`
supports the information-boundary finding above. Focused checks: 73 passed,
four PPO/warmup tests excluded; two comparison checks rerun after the prefix fix.

Next: test a conservative completion check using an explicit, fixed pose-height
uncertainty bound, without feeding it the injected error or simulator truth.
Compare it on zero error and the same failing conditions; retain the full true
two-second hold and collision guards. Then generate training-only corrections
for the residual approach/grasp rejections, after confirming the teacher can
make its decisions from the supplied observations. Do not broaden fitting or
spend the reserved robustness holdout before this diagnostic correction.
