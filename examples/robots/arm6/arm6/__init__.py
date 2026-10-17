"""The twin's example arm as an installable package: a robot folder and a backend plugin.

Six joints and a parallel gripper built from MuJoCo primitives, so the package stays a few KB.
`pyproject.toml` registers both halves:

* `machinekind_twin.robots`: `arm6 = "arm6:ROBOT_DIR"`, the folder with `robot.yaml` and the model;
* `machinekind_twin.backends`: `arm6 = "arm6.backend:create"`, an arm in memory.

Nothing here imports the twin: the backend uses only the contracts (`machinekind-twin-api`).
"""

from pathlib import Path

ROBOT_DIR = Path(__file__).parent / "robot"
