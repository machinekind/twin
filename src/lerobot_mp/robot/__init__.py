"""Robot backends: the contract (`base`), the plugin registry (`factory`) and the backends that
still live here - SO-101 (`feetech`, `lerobot`) and Galaxea A1X (`a1x`), registered as plugins
in this repository's `pyproject.toml` until they move to their robots' repositories.

The twin's simulated arm is `twin.runtime.SceneBackend`, not a backend from this package.
"""

from .base import BackendContext, BackendPlugin, RobotBackend, RobotInfo
from .factory import backend_context, backends, create_backend

__all__ = ["BackendContext", "BackendPlugin", "RobotBackend", "RobotInfo", "backend_context", "backends",
           "create_backend"]
