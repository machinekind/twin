# Episode recording

Recording is disabled by default. Enable it for a strategy run:

```python
from machinekind_twin.episode import EpisodeOptions
from machinekind_twin.strategy import StrategyRunner

runner = StrategyRunner(twin, strategy, episode=EpisodeOptions(
    folder=campaign, name="run-001", meta={"seed": 42},
))
runner.start()
```

Omit `folder` to use `episodes/` beside the workspace file. Names default to a
local timestamp; duplicates receive a numeric suffix. In the Strategies tab,
**Record episode** enables the same recorder.

A worker handles file creation, JSON serialization, Zstandard compression,
image encoding and disk writes. Control callbacks only snapshot and enqueue data.
The queue holds at most 1,024 events and 64 MiB of pending image data. If it fills,
or a write fails, recording stops and `runner.status.episode_error` reports why.
The strategy continues. Queue overflow leaves an unfinished recording.

Stopping a run queues completion without waiting for disk I/O. Before reading
its completed file, wait explicitly:

```python
runner.stop()
if not runner.wait_for_episode(timeout=5.0):
    raise TimeoutError("Episode recording is still finishing")
path = runner.status.episode  # None if recording never opened
```

A recording also captures a strategy's startup failure. A failed arm claim
creates no recording. Files are kept until deleted; there is no automatic retention.
The repository ignores episode folders, `*.jsonl.zst` and `*.frames/` directories.

## Files and storage

Each run produces `<name>.jsonl.zst`: UTF-8 JSON Lines compressed with Zstandard
level 1 and a content checksum. The worker flushes compressed blocks every
250 ms and flushes and syncs the completed file. After process termination, the
loader can recover complete records from flushed blocks and marks the run
`unfinished`. Corrupt compressed data raises an error. Existing uncompressed
`.jsonl` files remain readable.

Frames are optional. `keep_frames_s=None` saves only camera metadata;
`keep_frames_s=1.0` saves at most one image per camera per second;
`keep_frames_s=0` saves every frame read by the strategy. Images are stored in
`<name>.frames/`. The default is JPEG at quality 85. Set
`frame_format="webp"` to use WebP, or `frame_quality=75` to adjust quality (1–100).
Recording does not fetch additional frames.
The UI's **Save camera frames** uses the one-second interval;
**Frame format** selects JPEG or WebP.

Image storage is approximately `cameras × saved frames/second × average encoded
bytes × seconds`. For example, three cameras at one frame/second and 50 KB/image
add about 9 MB/minute (540 MB/hour). Measure representative images: scene content
and resolution change image size substantially. Images are already compressed and
are not passed through Zstandard.

## Inspect a recording

```bash
uv run twin episode                       # Latest recording beside the default workspace
uv run twin episode campaign/run-001.jsonl.zst
uv run twin episode campaign/             # List runs, newest name first
uv run twin episode --gripper marker --offset-mm 0 0 20
```

`--path` selects a different workspace file when locating recordings.
`EpisodeLog.load(path)` returns arrays and step records. `EpisodeLog.summary(path)`
streams just the header and final status into memory; compressed logs must still
be scanned from the beginning. Loading compressed files requires `zstandard`.

## Record format

Each line has a `kind`. Joint arrays follow `header.joints`; angles are degrees
and gripper values use the robot's 0–100 scale. Positions are metres and rotations
are 3×3 matrices in the robot's base frame. Non-finite numbers become `null`.

| Kind | Contents |
|---|---|
| `header` | Format/version, robot and joint limits, backend, workspace, strategy class/rate/time limit, recording options, package versions and caller metadata |
| `loop` | Sample time, command sent to actuators, fresh joint readings and their timestamp, joints at a limit |
| `step` | Strategy step/time, observation and action timestamps, observed joints/command/objects/goal, validated target (`intent`), frame metadata, action result and model decisions |
| `end` | Termination code/detail, duration, accepted step count, whether startup returned, and the last action's result |

Timestamps use seconds since recording started on `Twin.clock`; `header.t0`
contains that origin. `run_t` is the strategy's elapsed time. `age_s` is an age,
so an observed pose's capture time is `obs_t - age_s`. Synthetic tests using a
virtual clock must supply ages on the same time scale.

`RunnerStatus.stopped_code` and `end.code` use `done`, `gave_up`, `time_limit`,
`disconnected`, `stopped`, `preempted`, `bad_target`, `error` or `refused`.
`reason` is free-form detail. A missing end record or incomplete compressed
stream loads as `unfinished`.

## Report interpretation

The report estimates three delays for each moving joint:

| Signals | Interpretation |
|---|---|
| Target → measured position | Total observed response delay |
| Target → actuator command | Supervisor ramp, which depends on move distance and speed limits |
| Actuator command → measured position | Actuator response plus sampling and control-loop ordering |

The estimator scans shifts from 0 to 1 second in 5 ms steps and selects the
lowest RMS error during motion. The displayed band contains similarly fitting
shifts; it is not a statistical confidence interval. RMS error, residual and bias
are in joint units. A ramp or obstruction may not be explained by a single delay.
Stationary joints, missing signals and targets that merely copy measured motion
do not provide usable delay estimates. These are measurements of the recorded
signals, not a pure hardware specification.

Gripper drift compares the marker position predicted from measured joints with
an independently observed camera marker at the same capture time. Positive drift
means predicted minus camera position, in the base frame. The report gives mean,
RMS and maximum displacement, plus a subset measured below 10 mm/s TCP speed.
It requires the robot's kinematics package and MuJoCo, a marker label supplied
with `--gripper` or `EpisodeOptions.gripper`, and camera observations spanning
joint readings. The current perception source value for camera evidence is
`kamery`; inferred in-hand and stale poses are excluded. Recording does not itself
add gripper-marker detection. Without independent marker observations, drift is
unavailable. `--offset-mm` specifies the marker's location in the TCP frame.

## Model decisions

Strategies using `machinekind-twin-api>=0.1.3,<0.2` can attach diagnostic responses
to an action:

```python
from machinekind_twin_api import Action, Decision

action = Action(target=target, decisions=(Decision(
    question="reach-cube", answer=response_text,
    state={"joints": observed_joints}, confidence=0.8, response_s=1.25,
),))
```

Return each decision once, with the next action after its answer arrives.
The recorder stores it on that step; it does not drive the arm. `state` should
contain JSON data. Arrays larger than 64 elements are represented by type and
shape. A plain target mapping cannot carry decisions. Startup decisions must be
returned with an action; decisions arriving after the run ends are not recorded.

## Recording an external controller

Without a `StrategyRunner`, use `EpisodeWriter.create(folder, header, twin)` with
`header={**twin_header(twin), "mode": "observation", "strategy": None}`.
The writer captures twin loop samples. Submit `StepRecord`s with `run_t=None`
and any externally observed commands as `intent`; leave `intent={}` when unknown.
Close with an `EpisodeEnd`. `close(..., wait=False)` returns immediately;
`wait(timeout)` lets a non-control caller wait for completion.

When converting `Perception.snapshot()` views to `ObjectPose`s, add
`perception.clock() - perception.tracks.last_step()` to each view's `age_s`.
The snapshot's ages were calculated at the last tracker step. Using them unchanged
would align camera poses with joint readings too late.
