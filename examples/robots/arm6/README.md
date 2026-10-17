# arm6

The digital twin's example arm: six joints and a parallel gripper, built from MuJoCo primitives
(cylinders, capsules and boxes), about the size of a desk arm. It has no meshes, so the package is
a few KB. `uv sync` in the twin's clone installs it, editable, and the first `twin ui` opens the
example scene with it.

It is packaged the way a robot's own repository packages its arm: a robot folder (`arm6/robot/`:
`robot.yaml`, with a wrist camera on the hand, and the MJCF model `arm6.xml`) and a backend plugin
(`arm6/backend.py`, an arm in memory). `pyproject.toml` registers both through entry points:

```toml
[project.entry-points."machinekind_twin.robots"]
arm6 = "arm6:ROBOT_DIR"

[project.entry-points."machinekind_twin.backends"]
arm6 = "arm6.backend:create"
```

The package depends on `machinekind-twin-api`, the contracts, never on the twin.

```bash
uv sync
uv run twin ui      # the first start opens the example scene with arm6; Ramie: "Ramie przykladowe (w pamieci)"
```

The backend is a perfect position servo in memory, so **Polacz** works without hardware and runs the
twin's real-arm code path. Real arms live in their own packages, for example the SO-101 in
so-arm101.
