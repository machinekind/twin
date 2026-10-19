# Maintainer handoff

This file describes the current repository. For installation, calibration and
hardware procedures, use [the workstation guide](docs/TWIN.md). For plugin
contracts, use [machinekind-twin-api](packages/machinekind_twin_api).

## Scope and validation

Machinekind Twin is a workstation digital twin framework whose first application
is robot arms. The current runtime and plugin contracts are centered on arms;
broader workstation support remains a direction for development.

Simulation, calibration, perception, dynamics identification and supervised
strategy execution are covered by integration tests. End-to-end physical SO-101
and Galaxea A1X validation is still pending. Do not interpret synthetic accuracy
or past policy success rates as measurements on hardware.

## Package boundaries

| Location | Owns |
|---|---|
| [src/machinekind_twin](src/machinekind_twin) | MuJoCo scene, cameras, calibration, perception, runtime, supervisor, strategy runner, UI, dynamics fitting and export |
| [packages/machinekind_twin_api](packages/machinekind_twin_api) | Backend and strategy contracts, observations/actions and entry-point group names; NumPy only |
| [examples/robots/arm6](examples/robots/arm6) | Default example robot and in-memory backend installed by `uv sync` |
| [examples/robots/testbot](examples/robots/testbot) | Smaller robot loaded by the test session |
| [examples/strategies/swing](examples/strategies/swing) | Complete example strategy package |
| `soarm` in [so-arm101](https://github.com/machinekind/so-arm101) | SO-101 model and hardware backends |
| `soarm-lab` in so-arm101 | Training, policies and strategy implementations for SO-101 |
| [galaxeo-manipulators](https://github.com/machinekind/galaxeo-manipulators) | A1X description, model, demo workstation and CAN-FD backend |

Training code and checkpoint assets were split out of this repository. Historical
references to local RL modules, policy checkpoints and training commands are not
instructions for the current twin. The integration boundary is `twin export`
(scene, assets and station data) and installed strategy plugins. The panel runs
strategies; it does not train policies. `twin check` checks the twin's runtime,
installed models and serial ports, not CUDA or a training stack.

Plugins depend on `machinekind-twin-api`, not on the twin. Installed entry points
register robot folders, backend factories and strategy classes. The twin
re-exports the API's contract classes for compatibility. Strategy settings are
dataclasses validated before a run; selecting a strategy imports its package.

## Important implementation paths

| Area | Implementation | Tests |
|---|---|---|
| Scene and station export | `scene.py`, `workspace.py` | [test_twin_export.py](tests/test_twin_export.py), [test_twin_look.py](tests/test_twin_look.py) |
| Robot/backend discovery | `robots.py`, `backends/factory.py` | [test_robot_plugins.py](tests/test_robot_plugins.py), [test_example_arm.py](tests/test_example_arm.py) |
| Motion ownership, faults and Home | `runtime.py`, `safety.py`, `clock.py` | [test_twin_runtime.py](tests/test_twin_runtime.py), [test_twin_home_stages.py](tests/test_twin_home_stages.py), [test_twin_clock.py](tests/test_twin_clock.py) |
| Strategy runner and settings | `strategy.py`, `strategies.py`, `waypoints.py` | [test_strategy.py](tests/test_strategy.py), [test_strategy_plugins.py](tests/test_strategy_plugins.py) |
| Camera calibration | `calib/handeye.py`, `calib/session.py`, `calib/intrinsics.py` | [test_twin_calib.py](tests/test_twin_calib.py), [test_twin_intrinsics.py](tests/test_twin_intrinsics.py) |
| Mounted cameras and frame routing | `scene.py`, `cameras.py` | [test_twin_robot_cameras.py](tests/test_twin_robot_cameras.py) |
| Dynamics identification | `sysid.py`, `dynamics.py` | [test_twin_sysid.py](tests/test_twin_sysid.py) |
| Panel, guide and streams | `ui/app.py`, `ui/guide.py`, `ui/strategy_tab.py`, `ui/stream.py` | [test_twin_ui.py](tests/test_twin_ui.py), [test_twin_stream.py](tests/test_twin_stream.py) |
| Multi-tag body geometry | `tagbody.py` | [test_twin_tagbody.py](tests/test_twin_tagbody.py); not currently exposed in the panel |
| Optional A1X integration | External `galaxeo` plugin | [test_a1x_in_twin.py](tests/test_a1x_in_twin.py); skips without that package |

Implementation names in the table are relative to `src/machinekind_twin/`.
Keep the existing source-provenance notes in calibration modules copied from
`galaxeo-manipulators`; robot model and driver provenance belongs with the
external robot packages.

Cube detection/tracking and table mapping live in `perception.py`; they are not a
general object reconstruction pipeline.

## Local state and reproducibility

- `uv sync` installs the locked environment, local API and examples. Separately
  installed plugins survive `uv run` and `uv sync --inexact`; plain `uv sync`
  removes extra packages. Reinstall plugins after entry-point metadata changes.
- `workspace/twin.json` and `configs/local.yaml` are machine-local and ignored.
  `workspace/calib/*.json` is the exception: saved example camera presets are
  tracked. Do not describe all of `workspace/` as either bundled or ignored.
- Default data paths resolve from the clone root; explicit relative flags resolve
  from the current directory. See [paths.py](src/machinekind_twin/paths.py).
- First launch writes a demo only when the selected workstation file is missing.
  `twin demo --force` replaces it; use a separate `--path` for experiments.

## Invariants and pitfalls to preserve

- **One motion owner.** Every target goes through the supervisor. STOP, Home and
  reconnect preempt running or starting tasks. A slow strategy from an old run
  must never send targets into a later run.
- **Stop from measured motion.** Do not restore a stale command or drive a
  blocked joint toward an obstacle. Preserve an established gripper squeeze on
  STOP/Home; handle gripper faults separately. Reconnection can release a held
  object depending on the backend.
- **Observe real hardware.** The scene mirrors real joints. A physical wrist
  camera must supply real frames through a workspace camera of the same name;
  never substitute a simulated frame when real feedback is missing.
- **Keep capture times.** Old frames and repeated detections must not renew an
  object's lifetime. Dynamics recordings use control-loop tick timestamps so
  sampling jitter is not fitted as servo delay.
- **Separate truth from estimates.** Simulated camera `sim_pose` and estimated
  calibration are distinct. The calibration wave uses frames captured after the
  arm stops. It already rejects outlier observations and refits; adding that
  rejection is not an outstanding task.
- **Respect pixel conventions.** MuJoCo principal-point signs and the AprilTag
  half-pixel correction matter. Verify projection with segmentation masks, not
  centers of shaded color blobs.
- **Average rotations locally.** Calibration takes rotation medians in a tangent
  space; directly taking global Rodrigues-vector medians fails near +/-pi.
- **Use the rendering thread.** OpenGL context ownership is per thread. After
  stepping physics, update kinematics; mounted cameras also need `mj_camlight`
  before rendering so they follow the current body pose.
- **Report identifiable dynamics honestly.** The current excitation fits damping,
  armature and delay. Gain and dry friction stay nominal with wider uncertainty
  ranges. A narrow fit residual does not make all parameters observable.
- **Treat arm occlusions as unknown.** A tabletop-only fit can confidently put a
  held cube on the table. Keep fit-quality and near-gripper evidence gates, and
  use grasp evidence to track with the TCP.
- **Keep frame buffers intact.** Camera-watch correlation uses copies where
  OpenCV operations can modify input arrays. Do not let repeated comparisons
  fade the reference image.
- **Minimize browser updates.** The scene bridge updates body transforms when
  changed, rather than sending every geom each frame. Hidden browser tabs can
  pause viser drawing.

## Checks and remaining work

```bash
uv run pytest -q
uv run pyright
uv run twin check
```

Rendering needs a working OpenGL backend; UI/stream tests bind localhost sockets.
The `render` marker does not yet cover every test that creates a renderer, so
`-m "not render"` alone does not guarantee a headless run. Tests use virtual time
where appropriate ([tests/twinbench.py](tests/twinbench.py)); avoid introducing
wall-clock sleeps into deterministic motion tests.

Priorities:

1. Validate real hardware using [the first-session procedure](docs/TWIN.md#first-hardware-session),
   including feedback, STOP/Home, gripper calibration, camera alignment and dynamics.
   Use the A1X package's hardware guide for CAN and power sequencing.
2. Keep hardware-specific limits, offsets, motor enabling and protocol questions
   with the owning package. The panel still has A1X-specific motor controls;
   general backend actions would remove that special case.
3. Estimate a wrist camera's body-relative transform with a dedicated hand-eye
   workflow. Rendering and frame routing already exist; fixed-camera calibration
   does not solve this mount calibration.
4. Continue object reconstruction research separately from the working cube
   perception. See [the historical 3D study](docs/RESEARCH_3D.md) and
   [prototype requirements](research/3d/README.md).
5. Improve release readiness with reproducible CI, complete rendering markers and
   wheel-install checks. Training improvements and policy benchmarks belong to
   the training package, rather than this maintenance checklist.

For the removed remote RealAnt prototype, see
[the historical remote-robot note](docs/NOTES_REMOTE_ROBOT.md). That note records a
receive-only mirroring design; it is not a currently available ROS 2 integration.
