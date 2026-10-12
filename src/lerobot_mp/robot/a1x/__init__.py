"""Galaxea A1X: the robot folder and its CAN-FD hardware backend (`a1x`).

This package is the A1X's own and leaves the twin whole for galaxeo. Until then this repository's
`pyproject.toml` registers it like any robot package: the folder under `lerobot_mp.robots`, the
backend under `lerobot_mp.backends`.
"""

from pathlib import Path

ROBOT_DIR = Path(__file__).parent / "robot"
