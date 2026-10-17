"""The twin's test robot as an installable package: a robot folder and a backend plugin.

It is the template for a robot's own repository. `pyproject.toml` registers both halves:

* `machinekind_twin.robots`: `testbot = "testbot:ROBOT_DIR"`, the folder with `robot.yaml` and the model;
* `machinekind_twin.backends`: `testbot = "testbot.backend:create"`, the backend factory.

Nothing here imports the twin: the backend uses only the contracts (`machinekind-twin-api`).
"""

from pathlib import Path

ROBOT_DIR = Path(__file__).parent / "robot"
