# Remote robot in the twin: the RealAnt copy pattern

The twin used to carry a copy of the RealAnt ant (8 AX-12A servos). The ant code is deleted, and this note records
the pattern that stays: **a remote robot streams its joint angles, and a copy in the twin follows them**. The
ROS 2 adapter for remote robots is meant to build on it.

The last version of the code is in commit `86a9e2c`:

```
git show 86a9e2c:src/lerobot_mp/twin/antmirror.py      # AntMirror, StandInServos
git show 86a9e2c:src/lerobot_mp/twin/antwalk.py        # ZmqAnt, servo maps, DT
git show 86a9e2c:src/lerobot_mp/twin/scene.py          # Scene.command_ant, ant_pose, set_ant_pose
git show 86a9e2c:src/lerobot_mp/twin/ant.py            # Ant (tag mount), locate, snap_yaw
git show 86a9e2c:src/lerobot_mp/twin/antdrive.py       # the stand-in ant in the demo
```

## The pattern in short

1. The real ant walks on its own policy, which runs on its side (realant-rl `robot_teleop.py` or
   `robot_policy.py`).
2. `AntMirror` reads its servo angles over ZeroMQ, through `ZmqAnt` from `twin/antwalk.py`. It only listens.
3. Those angles are the servo targets of a **physical copy** of the ant in the twin's MuJoCo scene. The copy takes
   the same steps and moves by itself, also when no camera sees it.
4. Every frame that has tags pulls the copy's position and heading towards the pose from the tags.
5. The pose error is computed against the copy's pose **at the time the frame was taken**, so camera latency does
   not drag the copy back.
6. `AntMirror` calibrates by itself where the tag sits on the body.
7. The twin sends nothing to the real ant.

## The angle stream: `ZmqAnt` (`twin/antwalk.py`)

- On the ant's side `ant_server.py` from realant-hw runs behind `antproxy.py`. The proxy has an XPUB socket on
  port 3001 and fans the measurements out to every subscriber (PUB/SUB). The twin is one more subscriber next to
  `rollout_server.py`, so it takes nothing away from anyone.
- `ZmqAnt` opens a `SUB` socket on `tcp://<host>:3001` (`ZmqAntConfig`, default `127.0.0.1`), subscribes to the
  prefix `b"{"` (JSON messages only) and sets `RCVTIMEO` to 200 ms so the receive thread can stop.
- The `mrowka-serwa` thread reads `recv_multipart()` and parses the first part as JSON. It takes only messages with
  `"id": "serial"`. It uses the fields `s1_angle` ... `s8_angle` (raw AX-12A servo units, about 0.29 degrees per
  unit) and `ant_time` (the ant's clock in ms). Messages without these fields are skipped.
- The servo maps come from `rollout_server.py` (marked "TODO validate" there) in semberecki/realant-rl, which is
  MIT-licensed (Aalto University, Ote Robotics Ltd); `NOTICE` at `86a9e2c` carries the attribution. Policy joint `k`
  reads servo `SERVO_OF_JOINT = (2, 3, 0, 1, 6, 7, 4, 5)`. Hip: 350 / 510 / 661 units give -0.80031 / 0 / +0.80031
  rad. Ankle: 288 (leg down) / 511 (leg horizontal) give 1.14549 / 0 rad. Between those points the conversion is
  linear (`units_to_joints`).
- Joint velocities are angle differences divided by the `ant_time` difference. The servos report only an angle.
- `ZmqAnt.joints()` returns `(angles [rad], velocities [rad/s], reading age [s])`, or `None` before the first
  reading. The age counts from the moment the twin received the message.
- `pyzmq` was an optional dependency (`pip install -e .[ant]`). The import sat inside `ZmqAnt.__init__`, so the
  twin without `pyzmq` kept working.

The contract of an angle source was only `joints() -> (q, v, age) | None` and `close()`. The demo (`StandInServos`,
below) met the same contract, so `AntMirror` did not know where the angles came from.

## The copy in the scene physics

- The scene held the ant under the prefix `ant/` (`scene.ANT_PREFIX`), inserted from the RealAnt v2 model
  (`assets/robots/realant/realant.xml`) with `attach_body`. The torso stood on a free joint `ant/root`, and the legs
  had position-servo joints from the model.
- The model's torso was rotated by 180 degrees relative to the ant's front (`ANT_TORSO_IN_ANT = diag(-1, -1, 1, 1)`).
  All poses outside the scene were in the ant frame (x forward, z up) relative to the arm base.
- The `Scene` methods: `command_ant(ctrl, prefix)` sets the leg servo targets (clipped to `ctrlrange`),
  `ant_pose(prefix)` reads the torso pose from the model, `set_ant_pose(T, prefix)` moves the free joint and
  rotates the linear velocity with it, and `ant_joints(prefix)` reads the leg angles and velocities.
- A scene rebuild (`Twin.rebuild`) carried the copy's free-joint state over with `_copy_state`, so a copy that was
  already walking kept walking. A new copy stood in the pose stored in the workspace.

## The `AntMirror` loop (`twin/antmirror.py`)

`AntMirror(twin, get_ant, servos, gain=0.5, auto_yaw=True, auto_level=True, snap=True, on_missing=None)`.
`start()` launches the `kopia-mrowki` thread, which calls `tick()` every `antwalk.DT` = 50 ms (20 Hz). Deadlines
count from the start (`t0 + k * DT`); after a delay of more than 1 s the counter restarts. An exception from
`tick()` goes to `status.note` and the loop keeps running. Tests called `tick()` by hand.

One `tick()`:

1. Reads `servos.joints()` and, under `twin.lock`, sets the copy's servo targets from it (`command_ant`).
2. Steps the physics by `DT`, but only if nobody else did. If the simulation time grew by less than half of `DT`
   since the previous tick, the arm loop is stopped and `AntMirror` steps the scene itself. Otherwise time would
   run twice as fast.
3. Records the copy's physics pose in a history (`deque`, 80 entries, about 4 s) with the time `time.monotonic()`.
   It also adds the in-plane displacement to the odometry. The odometry counts motion from physics only, without
   tag corrections, because the previous position is taken after the correction already.
4. Looks for tags (`_tags`). From every enabled camera that has a pose it takes the frame with its capture time
   (`twin.cameras.frame_t(name)`). It skips frames that are not newer than the last one used and frames older than
   0.5 s. The ant pose comes from `ant.locate` (the median across cameras through `tagbody.locate`). The measurement
   time is the earliest capture time among the frames used.
5. When tags are seen but the copy is not in the scene, it calls `on_missing(T)`. The panel then saved the pose
   into the workspace and rebuilt the scene, at most once every 3 s.
6. When tags are seen and the copy exists, it corrects the pose (`_correct`) and then calibrates the mount.
7. Sets the state: `czeka na tagi` (no copy yet), `czeka na serwa` (no readings), `bez tagow` (the last frame with
   tags is older than 1 s, the copy runs on servos alone), `sledzi`.

## Pulling the pose from tags, and camera latency

- Only the in-plane position and the heading are corrected. Height, pitch and roll stay from physics, because the
  copy stands on its legs on the table.
- The error is the difference between the pose from the tags and the copy's pose **at the time the frame was
  taken** (`_at(t_frame)`: the nearest history entry). The correction is applied to the copy's pose **now**: a
  rotation by `g * heading error` about the vertical through the ant's centre, and a shift by
  `g * position error`.
- Why: a frame arrives late. An ant in motion is already further along than the frame shows. Comparing against
  the copy's pose from now would drag the copy backwards on every frame. Comparing against the pose from the same
  moment measures the real mismatch and leaves the motion since the frame untouched.
- `g = 0.5` for a normal frame. A jump (`g = 1`) on the first frame and when the error exceeds 0.10 m or 30 degrees
  (the ant was lost or moved by hand). After a jump, the history and the heading-calibration segment start over.

## Self-calibrating the tag mount on the body

The cameras see only tags, not the body. The mount (`Ant.mount`: x, y, z, yaw, pitch and roll of the tag in the
ant frame) splits into two parts, and a different source calibrates each.

- **Heading (the mount yaw)**: from a driving segment. When the tags have moved by at least 6 cm, `AntMirror`
  compares the direction of the copy's odometry displacement (from servos) with the direction of the displacement
  from the tags. The angle difference is the mount yaw error and is added to `mount["yaw"]`. A segment is skipped
  when the two lengths differ by more than a factor of two (slip, or the copy did not reproduce the motion). The
  result is snapped to 45 degrees + k * 90 degrees (`ant.snap_yaw`, when the deviation is at most 20 degrees),
  because the tag has edges parallel to the body plate and its axes run along the legs. A change below 0.5 degrees
  is skipped.
- **Tag height and pitch**: from the tag relative to the copy's body at the frame time. The body stands on its legs,
  so its height and pitch come from physics, not from the cameras. Computed only after a normal correction, when
  the copy has caught up with the tags (error below 12 mm and 3 degrees). `z`, pitch and roll of the mount go
  through a low-pass filter with coefficient 0.1.
- By hand there remained `ant.level_mount` (the "level from the table" button): height and pitch from an ant lying
  or standing flat on the table.

## Why the twin sends nothing to the real ant

Earlier the twin sent commands to the ant over PUB. Commands from the twin and from the ant's own controller
(`robot_teleop.py`) arrived at the same time and fought each other. Since that change `ZmqAnt` only listens. The
ant's own policy drives it, and the twin only mirrors it.

## The demo: a stand-in ant (`twin/antdrive.py`)

- The demo had no real ant. A second ant in the scene under the prefix `antprawda/` (`scene.STANDIN_PREFIX`), with
  real tags, played its part.
- `twin/antdrive.py` drives only this simulated stand-in. `AntDrive` and `SimAnt` are the ported
  `robot_teleop_v2.py`: the `exp004_smooth2` walking policy in numpy, WSAD keys, 20 Hz, a 0.4 s dead-man switch. It
  never drove the real ant.
- `StandInServos` (in `antmirror.py`) read the stand-in's angles from the scene, rounded them to AX-12A units
  (`joints_to_units` then `units_to_joints`) and returned them through the same `joints()` contract as `ZmqAnt`.
- The copy was then hidden from the simulated cameras (geom group 4, semi-transparent, no tags), so the cameras saw
  only the stand-in's tags. The 3D panel drew it.

## What was verified

In simulation only (`tests/test_twin_antdrive.py`, `tests/test_twin_ant.py` at `86a9e2c`):

- a copy that started with a wrong mount (45 degrees instead of 135) had the mount at 135 degrees after a few
  driving segments, the tag height less than 6 mm from the truth, and stood less than 2 cm / 5 degrees from the
  stand-in ant;
- without tags the copy reproduced a drive from servos alone with a 1.5 cm difference over 13 cm;
- `ZmqAnt` on a fake proxy only listened.

**On the real ant this never ran.** The servo maps were not validated. The v2 model differs from the v1 ant on the
bench in its hip mounts, so the copy reproduced the motion only approximately, and the tags pulled out the
mismatch.

## What to carry over to the ROS 2 adapter

- An angle source with the same contract: `joints() -> (q, v, age) | None`, `close()`. A `sensor_msgs/JointState`
  subscription instead of `ZmqAnt`, receive only, no command publishing.
- The measurement time from the sender, not from the receiver: `header.stamp` in ROS 2, like `ant_time` and the
  frame capture time here.
- A copy in the scene physics under its own prefix, driven by servo targets from the readings.
- A pose correction from an external measurement, compared with the copy's pose history at the measurement time and
  applied to the pose now, with a small gain and a jump on a large error.
- A physics step only when nobody else steps.
