# Plan: generic calibration twin

This plan turns `lerobot_mp` into a generic calibration and digital-twin product. Part 1 states the goal.
Part 2 removes the code that serves one device, one robot or one app. Part 3 defines how a robot enters the
product. Part 4 defines how a policy drives the twin and how the cameras measure its error. Part 5 describes
the room-scale deployment with two arms and the Wojtek quadruped. Part 6 describes the sim-to-real loop.

Every fact below was checked against commit `86a9e2c`. Paths are relative to `src/lerobot_mp/` unless they
start with `tests/`, `docs/`, `assets/`, `scripts/`, `examples/` or `configs/`.

## 1. Goal

The product calibrates cameras against a measured rig and builds a MuJoCo twin of that rig.

- The product owns a contract for adding any robot from a URDF or MJCF model and a definition file. The package ships no robot.
- One scene holds several robots, and each robot has its own parameters.
- The product calibrates the cameras against the measured rig and builds the twin from that calibration.
- Any policy runs against the twin or the real robot through one contract. The contract feeds the policy live joint positions and tracking error.
- The calibrated cameras measure the policy's error against reality.

These items are out of scope.

- Hardware backends for specific robots live outside the package.
- Spectacles, RealAnt, the PMD camera, the uEye camera and the MediaPipe hand-control app leave the repository.
- Training recipes stay as they are. Nothing in this plan changes how `twin/rl/` trains.

## 2. The cut

The cut has five pull requests in this order.

1. Step 1 deletes Spectacles.
2. Step 2 deletes RealAnt.
3. Step 3 deletes the PMD and uEye cameras.
4. Step 4 moves the twin's dependencies out of the hand-control app.
5. Step 5 deletes the hand-control app.

Each PR leaves `pytest -q` green. The cut deletes 12 of the 32 test files. Those 12 files hold 127 of the 278
tests that pytest collects today.

**The panel.** `twin/ui/app.py` has 3,971 lines, and its `TwinApp` class spans about 3,470 of them. The other
modules in `twin/ui/` add about 2,300 lines. 161 lines of `app.py` mention Spectacles, the ant, PMD or tag
bodies. Steps 1 to 3 each remove their own tab and imports from the panel and keep the panel running. Nobody refactors
`TwinApp` during the cut. After the cut, the team rebuilds a thin panel against the policy contract of part 4.

### Step 1. Spectacles

Only `twin/cli.py` and `twin/ui/specs_panel.py` import `twin/specsreach.py`. No core module imports it.

| Deleted | Tests |
|---|---|
| `twin/specsreach.py` with 2,499 lines | `tests/test_twin_specsreach.py`, 35 |
| `twin/reach.py`, `twin/follow.py`, `twin/impact.py` | `tests/test_twin_specsreach_a1x.py`, 15 |
| `twin/ui/specs_panel.py` | `tests/test_twin_specsreach_so101.py`, 2 |
| `docs/SPECS.md`, `docs/RIG.md` | `tests/test_twin_follow.py`, 26 |
| | `tests/test_twin_impact.py`, 10 |
| | `tests/test_twin_reach.py`, 8 |

The PR also makes these edits.

- `twin/cli.py` loses the `specs` subcommand. Its handler `_specs` imports `SpecsReach` at line 525, and its subparser starts at line 752.
- The `ui` subcommand loses `--specs`, `--i-know`, `--relay`, `--follow-speed` and `--hold-off`. `_ui_startup_line` keeps only the `--arm` part.
- `twin/ui/app.py` drops the `add_specs_folder` import at line 50, the folder it builds and the glasses arguments of `TwinApp._startup`.
- `twin/ant.py` imports `ReachVolume` from `reach` at line 427. `ant_xy_at_table_end` uses it to keep the ant behind the glasses' reach box. The PR deletes that check and `REACH_CLEARANCE`.
- `tests/test_twin_ant.py` drops the `ReachVolume` part of `test_add_ant_stands_it_at_the_far_table_end_clear_of_the_reach_box`.
- `twin/workspace.py` drops the `reach` field and its key in `Workspace.to_dict`.
- `tests/twinbench.py` mixes a generic virtual-clock `Bench` with the Spectacles relay. The PR keeps `Bench` and deletes `FakeRelay`, `Bench.driver` and the `websockets` imports.
- `tests/test_twin_a1x_sim.py` keeps using `Bench` for its staged-home test and drops `importorskip("websockets")`.
- `tests/test_twin_ui.py` deletes `test_spectacles_mode_change_interrupts_a_move_and_disconnect_clears_real_consent` and `test_ui_specs_real_without_i_know_refuses_before_the_panel`. `test_startup_flags_connect_the_arm_and_start_the_glasses` keeps only its `--arm` checks.
- `pyproject.toml` drops `websockets` from the `twin` extra. `constraints.txt` keeps its `websockets` pin because `viser` requires `websockets`.
- `docs/TWIN.md` loses its Spectacles section.

Step 1 is done when no file under `src/` or `tests/` imports `specsreach`, `reach`, `follow`, `impact` or
`websockets`.

### Step 2. RealAnt

| Deleted | Tests |
|---|---|
| `twin/ant.py`, `twin/antdrive.py`, `twin/antmirror.py`, `twin/antwalk.py`, `twin/antpad.py` | `tests/test_twin_ant.py`, 4 |
| `twin/ui/ant_panel.py`, `assets/robots/realant/`, `assets/policies/ant/` | `tests/test_twin_antdrive.py`, 10 |

The PR also makes these edits.

- `twin/scene.py` loses about 200 of its 654 lines. These are `AntPlacement`, `ANT_PREFIX`, `STANDIN_PREFIX`, `ANT_TORSO_IN_ANT`, `ANT_JOINTS`, `GHOST_RGBA`, `CARDBOARD_RGBA` and `_add_ant`.
- `SceneConfig` loses its `ant` and `ant_standin` fields.
- `Scene` loses `ants`, `_ant`, `place_ant`, `ant_joints`, `command_ant`, `ant_pose` and `set_ant_pose`.
- `twin/workspace.py` loses the `ant` field, `ant_obj()` and the `with_ant` argument of `scene_config`.
- `twin/runtime.py` loses `with_ant=True` at lines 322 and 379, the ant loop in `rebuild` and `Twin.place_ant`.
- `twin/cli.py` loses `demo --ant`, `workspace --ant` and their handlers.
- `twin/collision.py` drops the ant from the docstring of `set_world`.
- `pyproject.toml` drops the `ant` extra, and `constraints.txt` drops the `pyzmq` pin.
- `docs/TWIN.md` loses its RealAnt section.
- Part 5 records the pattern of `twin/antmirror.py`, so the deletion loses no design knowledge.

Step 2 is done when a search for `AntPlacement`, `ANT_PREFIX`, `with_ant` and `realant` in `src/` and `tests/`
finds no match.

### Step 3. PMD and uEye cameras

| Deleted | Tests |
|---|---|
| `twin/pmd.py`, `twin/ueye.py` | `tests/test_twin_pmd.py`, 5 |
| `pmd_view.py` and `ueye_view.py` at the repository root, `assets/calib/ueye_calib.txt` | `tests/test_twin_ueye.py`, 6 |

`twin/tagreach.py` stays. It finds a desk tag's pose from the trusted cameras and computes a point above it.
It imports only `calib.tags` and `calib.handeye`. Part 4 uses it to measure object poses.

`twin/tagbody.py` stays with `tests/test_twin_tagbody.py`. It locates a rigid body that carries several tags
from the calibrated cameras. Part 5 uses it for arm bases and for Wojtek's back. The PR removes its PMD wording.

The PR also makes these edits.

- `twin/cameras.py` drops the `pmd` and `ueye` branches of `probe_all` at line 80 and `open_live` at line 89.
- `twin/workspace.py` drops the `pmd` field and the `intrinsics_from == "pmd"` branch at line 130.
- `twin/ui/app.py` drops the `is_pmd` import at line 42, the PMD section and the PMD and uEye cases of the camera probe. The `tagbody` import at line 43 goes too, because only the PMD section uses it.
- `tests/test_twin_ui.py` deletes `test_pmd_camera_with_tags_on_its_housing_is_learned_found_drawn_and_saved`. `test_shipped_lens_files_parse` drops its `ueye_calib.txt` case.
- `pyproject.toml` drops the `ueye` extra, and `constraints.txt` drops the `pyueye` pin.
- `docs/TWIN.md` loses the PMD section and the reference to `ueye_calib.txt`.

Step 3 is done when `twin/cameras.py` opens every real camera through `_Live` and `CameraStream`.

### Step 4. Move the twin's dependencies out of the app

The twin imports five modules of the hand-control app.

| Module | What the twin uses | Importer | Destination |
|---|---|---|---|
| `control/safety.py` | `SafetySupervisor`, `SafetyState` | `twin/runtime.py` line 40 | `twin/safety.py` |
| `control/filters.py` | `RateLimiter` | `control/safety.py` line 15 | `twin/safety.py` |
| `vision/camera.py` | `CameraStream` | `twin/cameras.py` line 29 | `twin/camera_stream.py` |
| `preview/model.py` | `load_model` | `twin/cli.py` line 77, `tests/test_twin_robots.py` line 41 | `tests/articulus.py` |
| `config.py` | dataclasses, constants, loaders | `twin/robots.py`, `twin/kinematics.py`, `twin/cameras.py`, `twin/cli.py`, `twin/rl/sysid.py`, all of `robot/` | `twin/config.py` |

`load_model` reads the Articulus reference model in `assets/so101_preview.npz`. The twin uses it only to
compare the SO-101 MJCF with that reference. `lerobot-twin check` drops that comparison.
`test_mjcf_matches_the_articulus_reference_in_every_pose` keeps it through `tests/articulus.py`. Part 3 moves
that test, the `.npz` file and `scripts/import_articulus_model.py` into the SO-101 example folder.

`twin/config.py` receives `JointConfig`, `SafetyConfig`, `CameraConfig`, `RobotConfig`, `A1XConfig`, `GRIPPER`,
`JOINT_NAMES`, `read_config_data`, `config_from_data` and `load_config`. A new `TwinConfig` replaces
`AppConfig` for the twin and the backends. It holds `safety`, `robot`, `a1x` and `joints`, and it keeps the
method `joint(name)`. `twin/` and `robot/` use `cfg.robot` 31 times and `cfg.joint(...)` 17 times, so the names
stay. Step 4 changes no behavior. `_default_joints` and the SO-101 fields of `RobotConfig` move unchanged, and
part 3 removes them. The hand-app sections stay in `config.py` until step 5.

Step 4 is done when no module under `twin/` or `robot/` imports `control`, `vision`, `preview` or `config`.

### Step 5. Delete the hand-control app

- The PR deletes `app.py`, `cli.py`, `__main__.py`, `config.py`, `control/`, `vision/`, `preview/`, `ui/` and `utils/`.
- The PR deletes `robot/sim.py`. The twin simulates through `SceneBackend`, and only the app and the `auto` branch of `create_backend` use `SimulatedArm`.
- The PR deletes `models/`, `start.bat` and the MediaPipe pose helpers in `tests/conftest.py`.
- The PR deletes `tests/test_keyboard.py` with 2 tests and `tests/test_pipeline.py` with 4 tests. `tests/test_twin_runtime.py` still covers `SafetySupervisor` through `Twin`.
- `configs/default.yaml` keeps only its `safety` and `robot` sections.
- `pyproject.toml` drops `mediapipe`, the `lerobot-mp` entry point and the hand-control description.
- `src/lerobot_mp/__init__.py` stops exporting `AppConfig`, `JOINT_NAMES` and `load_config`.
- `README.md` describes the twin. The distribution name `lerobot-mp` stays for now, and renaming it is an open decision.

Step 5 is done when `src/lerobot_mp/` holds only `twin/`, `robot/`, `paths.py`, `__init__.py` and `py.typed`,
and a fresh `pip install -e ".[twin]"` installs no `mediapipe`.

## 3. Robot definition contract

### Today

`twin/robots.py` defines `RobotSpec` with the fields `name`, `title`, `mjcf`, `joints`, `gripper`,
`tcp_site`, `hand_body`, `base_body`, `home`, `tcp_approach`, `tcp_closing`, `fingertips`, `roll_joint`,
`wave_ranges`, `onboard_cameras`, `jaw_bodies`, `limits`, `max_vel`, `gripper_range_q`, `mirror_joints`, `jaw`,
`desk` and `home_stages`. The defaults of `JawThresholds` and `Desk` are SO-101 values. The module defines the
constants `SO101` and `A1X` and `REGISTRY = {SO101.name: SO101, A1X.name: A1X}`. `get_spec(name)` reads the
registry. `robot_config(spec, path, overrides)` builds the app config for one robot.

`RobotKinematics(spec, model=None, prefix="", gripper_ticks=None)` takes a spec. `SceneConfig.robot`, the
calibration modules, `rl/task.make_task` and `CubeTracker.for_spec` take a spec too. `Workspace.robot` is a
name with the default `"so101"`, and `Workspace.spec()` calls `get_spec`.

`RobotBackend` in `robot/base.py` is a clean ABC. Its methods are `connect`, `disconnect`, `read_joints`,
`send_joints`, `is_connected`, `link_silent`, `faults`, `joint_limits`, `gripper_ticks`,
`calibration_warnings`, `efforts`, `velocities` and `step`. `robot/factory.py` `create_backend(cfg)` picks
`sim`, `lerobot`, `feetech`, `a1x` or `auto` by name.

`scripts/import_a1x_model.py` converts the vendor URDF to MJCF. It derives from galaxeo's `sim/urdf2mjcf.py`
and adds six A1X rules.

1. The joint `gripper_finger_joint1` becomes `gripper`.
2. Each actuator carries the name of its joint.
3. The bent jaw tips are named `finger1_tip` and `finger2_tip`.
4. The model's `<option>` sets integrator, timestep, cone and impratio, and the scene copies them.
5. The `tcp` site keeps galaxeo's axes, with x along the approach and y along the closing direction.
6. The equality that couples the second jaw gets a stiffer `solref` and `solimp`.

The script also hardcodes the A1X position gains, the jaw collision boxes and the `home` keyframe.
`scripts/import_articulus_model.py` produces only the SO-101 preview mesh. No generic contract exists.

### SO-101 is the default in 24 files

Outside the code that the cut deletes, `twin/`, `robot/` and `config.py` reference SO-101 112 times.

| Place | Default today | Change |
|---|---|---|
| `twin/calib/card.py` `Card.nominal` | `spec=SO101` | required argument |
| `twin/calib/simulate.py` `sample_cameras`, `run` | `spec=SO101` | required argument |
| `twin/calib/intrinsics.py` lines 296 and 317 | builds a scene with `SO101` | takes a spec |
| `twin/rl/sysid.py` | `JOINTS`, `Recording`, `record`, `Replayer`, `identify`, `fit` fall back to SO-101 | required argument, error on a recording without joints |
| `twin/rl/policy.py` | `PolicyMeta.robot = "so101"` | required field |
| `twin/rl/runner.py` line 128 | `getattr(policy.meta, "robot", "so101")` | error |
| `twin/rl/task.py` | `TaskConfig.n_joints = 6` | required field |
| `twin/rl/batch.py` | `CONTACTS` keyed by name, with an SO-101 fallback | `sim.contacts` in the definition |
| `twin/workspace.py` | `robot: str = "so101"` | required field |
| `twin/kinematics.py` | `TICKS_PER_REV` and `backend_gripper_ticks()` read STS3215 ticks from `RobotConfig` | move to the Feetech backend |
| `twin/runtime.py` `_gripper_mismatch` | compares backend ticks with `backend_gripper_ticks()` | moves into the backend's `calibration_warnings` |
| `twin/robots.py` | `JawThresholds` and `Desk` defaults, `robot_config` strips SO-101 tables from the YAML | required in the definition |
| `twin/cli.py` | `DEMO_WORKSPACES` keyed by robot, `--robot` defaults to `so101` | takes a robot folder |
| `twin/ui/app.py` | `JOINT_LABELS` keyed by robot, A1X CAN choices at line 1247, A1X enable button near line 1078 | labels in the definition, connection choices from the backend |
| `twin/ui/guide.py` lines 190 and 388 | branches on `backend == "a1x"` | asks the backend |
| `config.py` | `_default_joints`, `JOINT_NAMES` and the `RobotConfig` fields `robot_id`, `kind`, `use_degrees`, `calibration_dir`, `baudrate`, `center_ticks`, `gripper_closed_ticks`, `gripper_open_ticks`, `torque_off_on_exit` | SO-101 definition and its backend |

`config._default_joints` repeats the limits and `max_vel` of `SO101`. A comment in `robots.py` says a test
keeps them equal. Commit `3eea2f8` deleted that test, so nothing enforces the equality today.

`robot/feetech.py`, `robot/lerobot_backend.py` and `robot/serial_bridge.py` drive SO-101 hardware, and
`robot/a1x.py` drives the A1X. `assets/robots/so101/` holds the Menagerie MJCF. `assets/policies/reach-v3`
and `assets/policies/lift-v3` were trained on SO-101.

### The contract

A robot is a folder with a model file and a `robot.yaml` definition. The model is an MJCF file, or a URDF
file plus import rules in the definition. The definition carries every `RobotSpec` field that the model does
not carry.

```yaml
name: a1x
title: Galaxea A1X
model: a1x.xml                 # or urdf: a1x.urdf plus an import: section
joints: [arm_joint1, arm_joint2, arm_joint3, arm_joint4, arm_joint5, arm_joint6, gripper]
gripper: gripper
gripper_range_q: [0.0, 0.05]
tcp: {site: tcp, approach: [1, 0, 0], closing: [0, 1, 0]}
hand_body: gripper_link
base_body: base_link
fingertips: [finger2_tip, finger1_tip]
jaw_bodies: [gripper_finger_link1, gripper_finger_link2]
roll_joint: arm_joint6
mirror_joints: {gripper_finger_joint2: [gripper, -1.0]}
home_stages: [[arm_joint3], [arm_joint4, arm_joint5, arm_joint6], [arm_joint2, arm_joint1], [gripper]]
jaw: {still: 0.001, margin: 0.003, creep: 0.002, open: 0.020}
# home, limits, max_vel, wave_ranges, desk and labels list every joint, shortened here
home: {arm_joint2: 57.3, arm_joint3: -91.7, arm_joint4: 34.4, gripper: 0.0}
limits: {arm_joint1: [-160, 160], gripper: [0, 100]}
labels: {arm_joint1: [J1 Base, J1 base rotation]}
sim: {contacts: [64, 384]}
backend: a1x_backend:A1XArm    # optional plugin module path, placeholder name
```

The contract has these rules.

- `RobotSpec.load(path)` reads a folder and validates every field against the model. It replaces `SO101`, `A1X`, `REGISTRY` and `get_spec`.
- The package ships no robot. `examples/robots/so101/` and `examples/robots/a1x/` hold the two robots for tests, and a separate repository can hold more.
- Every robot with a gripper sets `gripper_range_q`. The SO-101 definition writes the range that today's servo ticks produce.
- Every `= SO101` default becomes a required argument. Every fallback that reads a missing value as SO-101 becomes an error or a required field.
- `_default_joints` and the SO-101 fields of `RobotConfig` move into the SO-101 definition and its backend. `TwinConfig` keeps only rig-level settings, which are safety timing and the backend and port of each robot.
- `twin/urdf_import.py` implements the six A1X rules as options under `import:`. The options also take position gains, collision-box replacements and the home keyframe.
- Backends become plugins against `RobotBackend`. `create_backend` finds one by the `backend:` module path or by a package entry point, and passes it the `RobotSpec` and its own config mapping.
- `SceneBackend` stays in the package as the `sim` backend.
- The Feetech, LeRobot and serial-bridge backends leave the package with the SO-101 folder. The home of their code is an open decision.
- The A1X backend leaves too. `robot/a1x.py` already imports the `galaxeo` package, which can host it.
- `tests/test_feetech.py`, `tests/test_lerobot_backend_guard.py`, `tests/test_serial_bridge.py` and `tests/test_a1x.py` move with their backends.

### Several robots in one scene

Today the scene and the twin hold exactly one robot.

- `SceneConfig.robot` holds one spec, and `Table.base_xy` and `Table.base_yaw` place it.
- `scene.build` attaches that robot under the fixed prefix `PREFIX = "robot/"`. `CollisionChecker` and `twin/ui/watch.py` use `PREFIX` too.
- `Scene` builds one `RobotKinematics(cfg.robot, model, PREFIX)`, one `act_ids` array and one `T_base2world`. The `prefix` argument serves the arm, and the ant never used it.
- Cameras, objects and panels are posed in the robot's base frame through `CameraView.T_cam2base`, `Box.xy` and `Panel.T_panel2base`.
- `Workspace` holds one `robot`, `backend`, `port`, `home` and `dynamics`.
- `Twin` holds one `spec`, `_backend`, `_supervisor`, owner with its `_gen` counter, `_target` and `RobotStatus`. One loop thread runs `Twin._tick`.
- `SceneBackend.step` steps the physics of the whole scene.

The change has these parts.

- A new `RobotMount` holds `name`, `spec` and `T_base2rig`. `SceneConfig.robots: list[RobotMount]` replaces `SceneConfig.robot`.
- The rig frame defaults to the base frame of the first mount, whose `T_base2rig` is then the identity. Existing camera poses stay valid. Part 5 fixes the rig frame to a tag board at room scale.
- `Table.base_xy` and `Table.base_yaw` place the rig origin on the table.
- `scene.build` attaches each robot under the prefix `<name>/`. `Scene.robots` maps each name to its `RobotKinematics`, `act_ids` and `T_base2world`.
- `Scene.kin`, `Scene.act_ids` and `Scene.T_base2world` stay as properties of the first robot until every caller passes a name.
- `CollisionChecker` and `watch.arm_mask` loop over all mounts.
- `Workspace.robots` lists one entry per robot with `name`, `definition`, `backend`, `port`, `home`, `dynamics` and `T_base2rig`. The loader converts a top-level `robot` key into a one-entry list, and a file without robots is an error.
- A new `RobotChannel` owns the per-robot state of today's `Twin`. That is the spec, backend, supervisor, owner, claim, preempt, target, status, loop state, tick listeners, home and estop.
- `Twin` keeps the scene, cameras, renderer, clock and lock, and it maps names to channels. `Twin.robot(name)` returns a channel, and `Twin.estop` stops every channel.
- Each channel runs its own loop thread, so a blocking read on one backend cannot stall another robot.
- Physics stepping moves from `SceneBackend.step` into `Twin`. One scene thread steps physics when any robot is simulated.
- `Twin.step(dt)` in manual mode ticks every channel in order and then steps physics once, so tests stay deterministic.
- The first robot calibrates the cameras with the card wave, as today. Each further robot gets its `T_base2rig` from a card wave with the camera poses held fixed. `calib/handeye.solve` needs a new mode for that.

### Order and done

1. `RobotSpec.load` and the two example folders land, and the constants become thin loaders. Done when every test passes on loaded specs.
2. Required arguments replace the SO-101 defaults and fallbacks. Done when no `= SO101` default remains.
3. The config split removes `_default_joints` and the SO-101 fields. Done when `TwinConfig` has no robot-specific field.
4. Backends become plugins and leave with their tests. Done when `robot/` holds only `base.py` and `factory.py`.
5. `twin/urdf_import.py` lands. Done when a test shows that it rebuilds `assets/robots/a1x/a1x.xml` byte for byte from the A1X definition.
6. Several robots per scene land. Done when a test connects SO-101 and A1X in one simulated scene and moves each under its own owner.

Part 3 is done when a search for `so101`, `SO101`, `a1x` and `A1X` in `src/` finds no match. The test suite
runs against the `examples/robots/` fixtures. Adding a robot requires only its folder. The `toy_spec` fixture
in `tests/conftest.py` becomes a third folder and proves that.

## 4. Policy contract and error measurement

### Today

- `Twin.add_tick_listener(fn)` at `twin/runtime.py` line 1311 calls `fn` once per loop tick with `TickSample(t, sent, measured, measured_t, fresh)`.
- `RobotStatus` holds `measured`, `command`, `at_limit`, `faults`, `owner` and `loop_hz`, among other fields.
- Motion goes through `Twin.claim(owner)`, `Twin.set_target(joints, owner)`, the `SafetySupervisor`, `Twin.preempt`, `Twin.hold_measured` and `Twin.estop`.
- The owner's asked target lives in `Twin._target`. Neither `TickSample` nor `RobotStatus` carries it.
- `Twin._check_tracking` computes the sent command minus the measured position per joint. It stops the arm after 25 degrees for 0.5 s and publishes nothing.
- `rl/runner.py` `PolicyRunner(twin, policy, max_tracking_deg=25.0, ...)` is the only policy consumer. The panel creates it at `app.py` line 3593. It accepts a torch `Policy` whose observation formula is `rl/task.py` `observe`.
- The runner computes `lag = max|q_measured - q_cmd|` at line 351 and uses it only as a stop.
- The runner also holds the episode rules. These are the step limit, `end_on_success`, goal projection, the lift and pickplace success checks and the wait for the first cube pose.
- `perception.py` gives the cube pose from the cameras through `CubeDetector`, `CubeTracker` and `TableMapper`.
- `TwinApp._tick_simreal` computes the median pixel distance from render edges to frame edges.
- `calib/handeye.py` solves camera poses from the card in the gripper and provides `pose_error`. No code measures the arm's own TCP pose from the cameras outside calibration.
- `tagreach.tag_from_frames` gives a tag pose in the base frame from the trusted cameras, with the spread between cameras.
- `rl/sysid.py` `Recording` saves `t`, the sent command and `measured` per tick. It is the only per-tick log today.

### The observation

`twin/strategy.py` defines `Observation`. The loop publishes one per tick per robot through
`add_tick_listener`, and it replaces `TickSample`.

```python
@dataclass(frozen=True)
class Observation:
    robot: str
    t: float                                   # loop clock
    measured: dict[str, float]
    measured_t: float
    fresh: bool
    sent: dict[str, float]                     # supervisor output
    asked: dict[str, float]                    # owner's target
    clip_err: dict[str, float]                 # asked - sent: supervisor clipping
    lag_err: dict[str, float]                  # sent - measured: servo lag
    tcp: np.ndarray                            # 4x4 in the base frame, from kinematics
    objects: dict[str, np.ndarray] | None      # poses from perception
    efforts: dict[str, float] | None           # RobotBackend.efforts()
    velocities: dict[str, float] | None        # RobotBackend.velocities()
```

The loop computes `tcp` with its own `RobotKinematics`, so it shares no `MjData` with the panel. `efforts()`
and `velocities()` return the latest fresh reading and cost no bus transaction. `rl/sysid.py` keeps working
because `t`, `sent`, `measured`, `measured_t` and `fresh` keep their names.

### The strategy

```python
class Strategy(Protocol):
    def start(self, twin: Twin, robot: str) -> None: ...
    def act(self, obs: Observation) -> Mapping[str, float] | None: ...
    def stop(self) -> None: ...
```

- `act` returns joint targets, or `None` to hold the current target.
- A strategy ends its episode by raising `EpisodeDone(reason, success)`. Task-specific success checks stay inside each strategy.
- `StrategyRunner(twin, robot, strategy, rate_hz, max_tracking_deg=25.0, max_steps=None)` wraps every strategy. It claims the robot, registers the preempt callback and paces `act` on the twin's clock. It sends targets through `Twin.set_target` and the supervisor.
- The runner keeps today's divergence stop. It compares the asked target with the measured position, which equals `clip_err + lag_err`. On any stop the runner calls `hold_measured` and releases the robot.
- `PolicyRunner` becomes `TorchPolicyStrategy`, the first adapter. It keeps `observe`, `apply_action`, the goal and the success rules, and it loses its thread and lag check.
- `IKStrategy` is the second adapter. It solves `RobotKinematics.ik` for TCP waypoints, checks each segment with `CollisionChecker.path_clear` and ramps toward each solution.
- The panel's "go above the tag" action at `app.py` line 3442 already chains `tagreach.hover_point`, `ik` and `Twin.move`. It becomes the first user of `IKStrategy`.

### Error against reality

Three measurements ship, and a fourth check is optional.

1. **Joint tracking error.** It comes from `clip_err` and `lag_err` of every observation. The episode summary stores the per-joint RMS and maximum of `lag_err` and the share of ticks with clipping.
2. **Sim-versus-real appearance misfit.** It exists inside `TwinApp._tick_simreal`. It moves to `twin/measure.py` as `edge_misfit(real, sim, K, dist) -> float`, and the panel calls that function.
3. **Task outcome error.** It does not exist yet. After an episode, the calibrated cameras measure the object pose and compare it with the goal. The cube uses `CubeDetector` and `TableMapper`, and a tagged object uses `tagreach.tag_from_frames`. The result holds position error in mm, yaw error in degrees and the spread between cameras.
4. **Camera TCP check.** The robot holds the calibration card, and the cameras find its tags with `tag_from_frames`. `Card.nominal` and `pinch_point` give the card pose in the TCP frame. `handeye.pose_error` compares the camera pose with the kinematic TCP pose. The check runs on demand and never per tick.

### The episode log

`twin/episode_log.py` defines `EpisodeLog`. `StrategyRunner` writes one `.npz` file per episode under
`workspace/episodes/`. The file holds per-tick arrays for `t`, `measured`, `sent`, `asked`, `clip_err`,
`lag_err`, `tcp` and `fresh`. Its metadata holds the joint order, the strategy name, the stop reason, the
tracking summary and the outcome error. `EpisodeLog.load(path)` reads any episode back, whichever strategy
wrote it.

### Order and done

1. `Observation` replaces `TickSample` in the loop. Done when the `rl/sysid.py` tests pass on it.
2. `StrategyRunner` and `TorchPolicyStrategy` land, and the panel switches to them. Done when `tests/test_twin_runner.py` passes on the new path.
3. `IKStrategy` lands, and the panel's tag action uses it.
4. `EpisodeLog` lands.
5. `twin/measure.py` lands with `edge_misfit` and the outcome error. A test compares the outcome error from simulated cameras with simulation truth.
6. The camera TCP check lands.

Part 4 starts after step 5 of the cut and uses the single-robot `Twin`. Step 6 of part 3 later moves
`Observation` publishing into `RobotChannel`.

Part 4 is done when two different strategies run on the same twin through the same code path. A test runs
`reach-v3` through `TorchPolicyStrategy` and a waypoint list through `IKStrategy` on one simulated twin. The
test reads both episode files with `EpisodeLog.load` and finds per-tick observations and an outcome error in
each.

## 5. How it runs at room scale

One room holds two arms and the Wojtek quadruped. The twin tracks all three, AI controllers drive all three,
and every machine sits on one network. Wojtek runs the ROS 2 workspace of `machinekind/w01-tek`. Its policy
node runs on the robot, and the workspace holds `wojtek_policy` and `wojtek_odometry` packages.

### Topology

- The twin hub is one process that owns the world. It holds the world frame, the camera calibration, the scene with every robot and part, perception, the per-robot streams and errors, logging and replay.
- An arm attaches through its backend. The Feetech backend reaches an arm on another machine through `robot/serial_bridge.py` and a `socket://` port, so a Feetech arm can hang off whichever machine it is plugged into.
- The A1X backend speaks CAN on the machine it is plugged into and has no network transport today. Whether the A1X runs on the hub's machine or behind a CAN bridge is open.
- Wojtek attaches through a ROS 2 adapter, a second robot type next to the driven backends.
- The adapter subscribes to Wojtek's joint states, its leg odometry and the intent its policy node publishes. It publishes corrections back. The topic names come from the w01-tek workspace.
- The adapter implements a new `ObservedRobot` interface with `read_telemetry`, `read_intent` and `publish`. It has no `send_joints`, so the hub cannot command Wojtek.

**The RealAnt mirror is this adapter in miniature.** `twin/antmirror.py` `AntMirror` reads the real ant's servo
angles over ZeroMQ through `twin/antwalk.py` `ZmqAnt`. It drives a physics copy of the ant with those angles.
Each camera frame with tags pulls the copy's position and heading toward the tag pose. The mirror compares the
tag pose with the copy's pose at the frame's capture time, so camera latency does not drag the copy back. The
mirror calibrates the tag mount on the body by itself. The twin sends the real ant nothing, because commands
from two places fought each other. `twin/antdrive.py` only drives the simulated stand-in ant of the demo. The
ROS 2 adapter copies the mirror's pattern.

### Frames

- Today every pose is expressed in one arm's base frame.
- The room fixes a world frame with a tag board on the floor or a wall. Every camera is calibrated against that board.
- `twin/tagbody.py` locates each arm's base from tags on it. Wojtek carries tags on its back, as the ant does today.
- The card wave of `twin/calib/session.py` and `twin/calib/handeye.py` stays as the per-arm refinement.
- Wojtek's mount in the scene has a free base. The hub updates that pose every tick from the fused estimate.

### Three streams per robot

| Stream | Arm | Wojtek |
|---|---|---|
| Intent | the strategy's asked target | the policy node's velocity command or body-pose plan |
| Telemetry | servo positions from the backend | joint states and leg odometry |
| Camera observation | gripper tag pose, part poses on the table, the render-versus-frame overlay | body tag pose, part poses on its back |

Every sample carries the time of its source. Part 4's `Observation` bundles one tick of intent and telemetry
with the latest camera observation of objects.

### Three errors

- Supervisor clipping is intent minus sent. Part 4 calls it `clip_err`.
- Actuation lag is sent minus telemetry. Part 4 calls it `lag_err`.
- Ground-truth drift is telemetry minus camera observation.
- For an arm the drift compares the kinematic TCP with the camera TCP check of part 4, and it stays small.
- For Wojtek the drift compares leg odometry with the body tag pose. It matters most there, because leg odometry drifts and the cameras do not.
- Wojtek's clipping and lag need its stack to publish the sent command beside the intent.

### Feedback at two speeds

Fast feedback is a correction and never a command. The hub runs no robot's controller over the network.

| Message | Fields | Receiver |
|---|---|---|
| `PoseFix` | source time, robot, `T_body2world`, spread between cameras | Wojtek's state estimator, which fuses it |
| `PartPose` | source time, part, `T_part2world`, source, spread | arm strategies |
| `Verdict` | source time, robot, verdict such as "part is on the back", "part fell" or "arm stopped short" | strategies and the log |
| `AdvisoryStop` | source time, robot, reason | Wojtek |

`PartPose` generalises what `twin/perception.py` `CubeTracker` already does for the lift policy through the
`cube_provider` of `PolicyRunner`.

Slow feedback is the data flywheel. The hub logs every intent, telemetry and camera observation.
`twin/rl/sysid.py` refits the actuator dynamics from those logs. Training randomises around the refit.
Failures with their camera-measured outcome become correction data.

### Time

- Every sample is stamped at its source, and the hub aligns the streams by time.
- `twin/clock.py` provides `WallClock` and `VirtualClock`. `TickSample` carries `measured_t` and `fresh`.
- `CameraHub.frame_t` in `twin/cameras.py` returns each frame with its capture time from `time.monotonic`.
- `time.monotonic` is local to one machine. Wojtek's ROS time and the hub clock need a shared reference.
- This plan makes the shared reference a requirement. Every machine syncs to one time server, and the hub drops samples from a link whose clock offset it does not know.

### Safety

- The arms' supervisors run inside the hub process.
- Wojtek's safety stays on Wojtek. The hub observes and advises Wojtek and never owns it.
- A stop from the hub over the network is advisory.
- Every robot's hardware cutoff works without the hub.

### Replay

A logged session replays into the simulation on `VirtualClock` with no robot attached. The replay shows the
recorded camera frames beside the render. The session log therefore stores camera frames with their capture
times. `EpisodeLog` from part 4 stores no frames, so the session log extends it.

### Contracts this part adds

1. World-frame calibration against a fixed tag board. Done when a simulated room calibrates every camera and both arm bases against the board within a tested tolerance of truth.
2. The ROS 2 adapter behind `ObservedRobot`. Done when the hub replays a recorded ROS 2 bag of Wojtek into the scene.
3. Intent as a first-class input stream. Done when every robot's log row carries intent, telemetry and camera observation with source times.
4. The shared time reference. Done when the hub reports the clock offset of every link and drops samples from links without one.
5. The correction message set. Done when Wojtek's estimator fuses `PoseFix` messages and the logged drift error shrinks.

## 6. The sim-to-real loop

The goal is to train in the twin, transfer well, and know why when a transfer fails.

### What the measured twin closes

- The geometric gap closes. Cameras sit at their measured poses with measured intrinsics, and robots sit at their measured base poses.
- The actuator gap closes in part. `twin/rl/sysid.py` fits damping, armature and delay.
- `twin/rl/randomize.py` states that position gain and dry friction are not identifiable from the recording. `UNFITTED_RANGES` randomises them wider.
- The perception gap closes. Training feeds the policy the cube pose the way the cameras deliver it. `Randomization.cube_delay`, `cube_period` and `cube_noise` make that pose late, sparse and noisy.
- The visual gap is avoided. The policies observe state through `rl/task.py` `observe` and never see pixels, so the renderer never has to match the room.
- The Sim-Real overlay serves diagnosis only.

### What it does not close

- Contact stays open. No camera measures friction, gripper compliance, backlash or slip in the jaws, and no motion recording identifies them.
- Load-dependent servo behaviour stays open. Sysid identifies the arm from 12-degree steps around home with no payload. The tracking error logged under load is the only measurement.
- Arbitrary parts stay open. A state-based policy needs pose and shape, so each part enters the twin as a scanned mesh or hull. `docs/RESEARCH_3D.md` is the starting point.
- Wojtek's locomotion stays in its own training pipeline in w01-tek. The twin provides ground-truth pose during carries and measured payload behaviour, and it does not duplicate that pipeline.

### The loop

1. The team calibrates the cameras, locates the robots, runs sysid and scans the parts.
2. Training randomises around the measurements. `Randomization.around` already centres on the identified dynamics and widens each fitted range to the band that sysid reports. Widths from real-episode residuals are new.
3. Evaluation in simulation runs on the CPU engine. `twin/rl/evaluate.py` already runs policies on plain MuJoCo instead of MuJoCo Warp.
4. The bench run starts in shadow mode. The strategy drives a simulated copy, the real arm holds still and the real cameras feed the part poses. The log compares intent with the simulated response. The guarded real run follows. `StrategyRunner` needs a shadow flag for this.
5. Every episode logs its intent, telemetry and camera outcome.
6. Attribution maps each failure to a gap. A geometric residual means recalibration. A lag residual means a refit or wider actuator randomisation. A grasp failure after a correct approach means contact, and it goes to real-data correction.
7. Correction refits what is identifiable and widens what is not. Fine-tuning on real episodes uses the camera-measured outcome as the label.

### Open decisions

- The choice between state-based and image-based policies is open. The twin is built for state-based policies. Image-based policies need a renderer good enough to train on, or they use the twin only as a state and outcome oracle.
- How parts enter the twin is open. The options are scanned meshes and primitive approximations.

### Done

Part 6 is done when the logged streams of one failed real episode attribute the failure to one named gap. The
correction for that gap is a documented procedure.
