"""Robot backends: the contract (`base`, from `lerobot_mp_api.backend`) and the plugin registry (`factory`).

Backends come from installed packages through the `lerobot_mp.backends` entry point group: the
SO-101's `feetech` and `lerobot` from the `soarm` package (so-arm101 repository), the test robot's
from `examples/robots/testbot`. `a1x/` is the A1X package that still lives here, registered in this
repository's `pyproject.toml` like any robot package until it moves to galaxeo. The twin's
simulated arm is `twin.runtime.SceneBackend`.
"""

from .base import BackendContext, BackendPlugin, RobotBackend, RobotInfo
from .factory import backend_context, backends, create_backend

__all__ = ["BackendContext", "BackendPlugin", "RobotBackend", "RobotInfo", "backend_context", "backends",
           "create_backend"]
