"""SO-101: the robot folder and its two hardware backends, `feetech` and `lerobot`, plus the COM<->TCP bridge.

This package is the SO-101's own and leaves the twin whole for the so101 repository. Until then
this repository's `pyproject.toml` registers it like any robot package: the folder under
`lerobot_mp.robots`, the backends under `lerobot_mp.backends`.
"""

from pathlib import Path

ROBOT_DIR = Path(__file__).parent / "robot"
