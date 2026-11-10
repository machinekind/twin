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
**Record episode** enables the same recorder. `twin eval-strategy` always records;
see [strategy evaluation](TWIN.md#strategy-evaluation).

Telemetry and images use separate background workers. Image encoding or storage
backlog cannot end telemetry collection. The recorder streams throughout the run;
there is no duration limit or growing in-memory history.

The telemetry queue holds 8,192 records by default, about 117 seconds at a 50 Hz
control loop plus a 20 Hz strategy. Set `EpisodeOptions.queue_capacity` to change
it. Retained image copies are limited to 64 MiB; `frame_buffer_bytes` changes
that budget. Encoding uses additional working memory. Capacity is reserved before copying images. Queues use short standard
synchronization; no shared lock covers encoding, disk I/O or a capacity wait.
Selected image copies and data snapshots still cost CPU on the caller.

Frame pressure skips only affected images and capture resumes automatically.
`RunnerStatus.episode_warning` shows gaps and high telemetry backlog. A genuine
telemetry overflow is counted, marks the recording incomplete and allows later
records to be collected. Disk failures set `episode_error`. Neither stops motion.
No finite buffer can preserve unlimited input during a permanent storage failure.

`close(end)` is nonblocking, even with full queues. Explicit completion waits have
finite timeouts. After close, outstanding images may be abandoned after three seconds and
become explicit gaps so a stuck image writer cannot prevent telemetry's
end record. Daemon workers cannot hold process exit indefinitely.

Before reading a completed file, wait explicitly:

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
are not passed through Zstandard. Each saved image is flushed and synced before
its completion is recorded. Camera filenames include an identity hash to avoid
collisions on case-insensitive filesystems.

## Inspect a recording

```bash
uv run twin episode                       # Latest recording beside the default workspace
uv run twin episode campaign/run-001.jsonl.zst
uv run twin episode campaign/             # List runs, newest name first
uv run twin episode --gripper marker --offset-mm 0 0 20
```

`--path` selects a different workspace file when locating recordings.
`EpisodeLog.load(path)` returns arrays, step records and collection accounting in
`log.recording`. Check `log.recording["telemetry_complete"]` before using a run for
evaluation or training. Frame gaps are allowed independently: affected entries in
`step["frames"]` have `file=None` and a `drop_reason`. Counts distinguish requested,
saved, skipped, failed and unfinished images. Pending images never count as saved.

`twin episode` shows these counts and exits with status 2 for incomplete telemetry.
Latency and drift reports are disabled for known telemetry gaps. A successful
strategy outcome is separate from collection completeness.

`EpisodeLog.summary(path)` streams the header, final outcome and collection counts
without retaining samples. Compressed logs must be scanned from the beginning.
Loading compressed files requires `zstandard`.

## Record format

Writers emit format version 2; version 1 logs remain readable.
Each line has a `kind`. Joint arrays follow `header.joints`; angles are degrees
and gripper values use the robot's 0–100 scale. Positions are metres and rotations
are 3×3 matrices in the robot's base frame. Non-finite numbers become `null`.

| Kind | Contents |
|---|---|
| `header` | Format/version, robot and joint limits, backend, workspace, strategy class/rate/time limit, recording options, package versions and caller metadata |
| `loop` | Sample time, command sent to actuators, fresh joint readings and their timestamp, joints at a limit |
| `step` | Strategy step/time, observation and action timestamps, observed joints/command/objects/goal, validated target (`intent`), frame metadata, action result and model decisions |
| `frame` | Completion or failure of an asynchronous image save, identified by strategy step and camera |
| `recording` | Periodic queue-loss and image accounting, also included in `end.recording` |
| `end` | Termination code/detail, duration, accepted step count, last action result and final collection accounting |

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
contain plain JSON data. Custom properties, mappings and object representations
are not evaluated during snapshots; unsupported values become type placeholders.
Arrays larger than 64 elements are represented by type and
shape. A plain target mapping cannot carry decisions. Startup decisions must be
returned with an action; decisions arriving after the run ends are not recorded.

## Recording an external controller

Without a `StrategyRunner`, use `EpisodeWriter.create(folder, header, twin)` with
`header={**twin_header(twin), "mode": "observation", "strategy": None}`.
The writer captures twin loop samples. Submit `StepRecord`s with `run_t=None`
and any externally observed commands as `intent`; leave `intent={}` when unknown.
Close with an `EpisodeEnd`; `close(end)` returns immediately.
`wait(timeout)` or `close(end, wait=True, timeout=5.0)` lets a non-control caller
wait for completion with a finite timeout.

When converting `Perception.snapshot()` views to `ObjectPose`s, add
`perception.clock() - perception.tracks.last_step()` to each view's `age_s`.
The snapshot's ages were calculated at the last tracker step. Using them unchanged
would align camera poses with joint readings too late.

## Collection checks

The manual soak uses synthetic 50 Hz telemetry, a 20 Hz strategy and three
640×480 camera images per second. Accelerated mode paces ten-second input batches;
`--block-images` deliberately stalls image saving while telemetry continues.

```bash
uv run python tests/soak_episode.py --seconds 28800 --block-images
uv run python tests/soak_episode.py --seconds 60 --realtime --frame-format webp
```

Use `--output /tmp/episode-soak` to retain the synthetic recordings and counters.

## Drive log of another robot

A walking robot's odometry drifts, and its base tag lets the cameras correct it
(see [other robots](TWIN.md#other-robots-in-the-scene)). The drive log writes a
drive down, so the drift can be measured afterwards. Tick **Zapisuj przejazd**
in the panel's **Inne roboty** folder for the robot picked there, or call
`Twin.log_drive(name, True)`; the robot's panel line names the file while it is
written. Unticking it, saving the robots and closing the twin write the last
line. A log that cannot be written, on a full disk say, ends by itself: its file
is closed, the robot and the twin go on, and the panel line says why. The file
is `other-<robot>-<YYYYmmdd-HHMMSS>.jsonl` in the episodes folder: plain JSON
Lines, written a line at a time, without compression.

`t` is seconds since the log started, taken from when the pose was measured or
the tag's frame was taken, so a tag line may carry an earlier `t` than the
pose line before it. Poses are a position [m] and a quaternion (w, x, y, z).
The first line is the header: the robot's name and description, its record's
`xyz` and `yaw_deg`, its `base_tag` or null, the start time and the units.

```json
{"format":"machinekind-twin-other-robot","version":1,"robot":"walker","description":"walker","xyz":[0.35,0.15,0.0],"yaw_deg":90.0,"base_tag":{"id":7,"size":0.08},"started":"2026-11-10T19:23:12+01:00","units":{"t":"s","position":"m","quaternion":"w, x, y, z","velocity":"m/s, m/s, rad/s","drift_m":"m","drift_deg":"deg","path_m":"m"}}
```

One line for every base pose the robot reports: `odometry` in its own fixed
frame, `base` where the twin puts it in the driven arm's base frame (the record's
pose, corrected by the tag, times odometry), and `velocity`, the base velocity
being commanded (forward, left, turn), or null when none is.

```json
{"t":0.199732,"odometry":{"position":[0.3,0.4,0.27],"quaternion":[0.707107,0.0,0.0,0.707107]},"base":{"position":[-0.05,0.45,0.27],"quaternion":[0.0,0.0,0.0,1.0]},"velocity":null}
```

One line for every camera whose sighting of the tag corrected the copy: the base
pose that camera measured in the driven arm's base frame, how many cameras saw
the tag in the same attempt, and the drift right after the correction.

```json
{"t":0.199732,"tag":{"base":{"position":[-0.01,0.45,0.27],"quaternion":[0.0,0.0,0.0,1.0]},"cameras":1,"drift_m":0.02,"drift_deg":0.0}}
```

The last line: `t` is the drive's length, `path_m` the path odometry reports in
the plane (a walking trunk's bobbing is no distance), the number of sightings
and the drift at the last pose line.

```json
{"t":84.2,"end":{"path_m":3.1,"sightings":230,"drift_m":0.041,"drift_deg":3.2}}
```

The drift is how far the copy stands from where the record and odometry alone
would put it, in the plane [m] and in heading [deg]. For every pose line it is
worked out from that line's `odometry` and `base` and the header's `xyz` and
`yaw_deg`: the final drift is the last pose line's, right also when the robot
walked on after the last sighting, and the largest drift the largest of them. It
counts from when the twin started reading the robot, not from the log's start,
and the first sightings also take up any error in the record's `xyz` and
`yaw_deg`. Odometry's own drift over the drive is therefore the change from the
first sighting to the last. A correction moves half of the error the cameras see
(all of it past 10 cm or 30 degrees), so the drift trails the cameras by a few
frames. A tag line's `base` against the `base` of the pose line nearest its `t`
is the error the cameras saw before that correction.

`twin episode <file>` prints a drive log's length, path, sightings, and final and
largest drift; `drivelog.summary(path)` gives the same numbers. `twin episode`
without a path, and with a folder, skips drive logs: they are not strategy runs.
