# Machinekind Twin

A framework for building digital twins of physical workstations, starting with
robot arms. It combines MuJoCo simulation, camera calibration, perception and a
browser control panel.

The goal is to make simulation match the real setup through measured geometry,
camera positions and optics, and hardware dynamics. Control strategies use the
same execution and safety interfaces in simulation and on hardware.

Robot arms are the first supported application. Hardware models, backends and
strategies are provided as plugins, keeping the core separate from individual
devices and tasks.

## Workflows

- **Experiment in simulation.** Start with the included example arm and two
  simulated cameras. Try control strategies, camera placement and calibration
  without hardware.
- **Build a twin of your workstation.** For a robot workstation, connect the
  hardware backend, calibrate camera intrinsics with a ChArUco board, and locate
  multiple cameras using a tagged card held in the gripper. Identify servo
  dynamics from recorded motion.
- **Inspect perception and alignment.** View camera streams, a stitched tabletop
  map and cube tracking. Compare real images with simulated overlays and detect
  moved cameras.
- **Export for training.** Export the MuJoCo scene, assets, camera calibration and
  identified dynamics for external training tools. Training lives outside this
  repository; `soarm-lab` provides training and policy strategies for the SO-101.
- **Run strategies on hardware.** Switch from simulation to the robot's backend.
  Scripted motion, trained policies and other strategy plugins run through the
  same supervisor, which handles motion ownership, joint limits, STOP and Home.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and OpenGL. From a clone:

```bash
uv sync
uv run twin check
uv run twin ui
```

Open **http://localhost:8080**. The first launch creates an example workstation
with a six-joint arm and two calibrated simulated cameras. Select `sim`, connect
the arm and try the included `swing` strategy.

The panel currently uses Polish labels: **Ramię** = Arm, **Połącz** = Connect,
**Strategie** = Strategies.

Useful commands:

```bash
uv run twin workspace                  # Inspect the workstation
uv run twin calib-sim --n 5 --cameras 2 # Evaluate calibration in simulation
uv run twin export workspace/export    # Export scene.xml, assets/ and station.json
```

Your workstation is saved in `workspace/twin.json`. For backend and strategy
settings, copy `configs/default.yaml` to `configs/local.yaml`, then launch with:

```bash
uv run twin --config configs/local.yaml ui
```

## Hardware and strategies

Hardware models, backends and control strategies are installed as separate Python
packages and discovered through entry points.

Existing robot integrations include the **SO-101** through
[`soarm`](https://github.com/machinekind/so-arm101/tree/main/packages/soarm) and the
**Galaxea A1X** through
[`galaxeo`](https://github.com/machinekind/galaxeo-manipulators).
To add your own, start with the [example robot](examples/robots/arm6),
[example strategy](examples/strategies/swing) and
[plugin API](packages/machinekind_twin_api).

## Status and development

Validated in simulation and with simulated hardware interfaces. End-to-end
validation on physical SO-101 and A1X arms is still pending.

```bash
uv run pytest -q
uv run pytest -q -m "not render" # Skip tests explicitly marked render
uv run pyright
```

See [the workstation guide](docs/TWIN.md) for setup and hardware procedures.

## License

Copyright 2026 Machinekind and contributors.

Machinekind Twin, its plugin API and the example packages in this repository are
licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE)
for attribution. Redistributions must preserve the applicable copyright and
attribution notices as required by the license.

Third-party dependencies and separately installed robot or strategy packages are
covered by their own licenses.
