"""Robot backends for real arms: SO-101 (`feetech`, `lerobot`) and Galaxea A1X (`a1x`).

The twin's simulated arm is `twin.runtime.SceneBackend`, not a backend from this package.
"""

from .base import RobotBackend, RobotInfo
from .factory import create_backend

__all__ = ["RobotBackend", "RobotInfo", "create_backend"]
