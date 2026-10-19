# 3D reconstruction and SO-101 grasp prototypes

Research scripts and recorded results for [the historical 3D study](../../docs/archive/RESEARCH_3D.md).
These are standalone experiments, not features exposed by the current panel.
They use the twin's scene and kinematics and specifically request the `so101`
robot; the bundled `arm6` and `testbot` models do not replace it.

| File | Experiment |
|---|---|
| [hull_and_grasp_bench.py](hull_and_grasp_bench.py) | Scene compilation/recompilation, rendering, a shared visual hull from four views, occlusion handling and simulated grasps |
| [grasp_reachability.py](grasp_reachability.py) | SO-101 top/side grasp reachability, IK position and orientation errors |
| [jaw_geometry.py](jaw_geometry.py) | Fingertip geometry relative to the TCP at different gripper openings |
| `*_out.json` | Recorded laptop results (i5-7200U, Intel HD 620); not current performance guarantees |

## Setup

From the clone, install the research extra and the external SO-101 robot package:

```bash
uv sync --extra research --inexact
uv pip install --no-sources "soarm @ git+https://github.com/machinekind/so-arm101.git#subdirectory=packages/soarm"
uv run twin check
```

The `research` extra provides `trimesh`. The hardware backend extras are not
needed for these simulation-only experiments. Keep `soarm` installed when
syncing by using `--inexact`.

## Running

Scripts write output JSON and images to the **current directory**. Run them
from a scratch directory with the project's interpreter. For example, on a Unix
shell, substitute the absolute path to your clone below:

```bash
mkdir -p /tmp/twin-research-3d
cd /tmp/twin-research-3d
uv run --project /absolute/path/to/twin --no-sync python -B /absolute/path/to/twin/research/3d/hull_and_grasp_bench.py
```

Use the other script filenames in the same way. `--no-sync` uses the environment
prepared above, including the research extra and separately installed robot.
Rendering requires OpenGL. On a Linux server without a display, configure
`MUJOCO_GL=egl` for a compatible GPU or `osmesa` for software rendering.

The saved outputs document earlier experiments; this cleanup does not rerun the
benchmarks or validate model recommendations from the historical study.
