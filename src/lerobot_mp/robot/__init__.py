"""Robot backends: the contract (`base`, from `lerobot_mp_api.backend`) and the plugin registry (`factory`).

`so101/` and `a1x/` are robot packages that still live here: each holds its robot folder and its
backends, registered in this repository's `pyproject.toml` like any robot package until they move
to the so101 repository and to galaxeo. The twin's simulated arm is `twin.runtime.SceneBackend`.
"""

from .base import BackendContext, BackendPlugin, RobotBackend, RobotInfo
from .factory import backend_context, backends, create_backend

__all__ = ["BackendContext", "BackendPlugin", "RobotBackend", "RobotInfo", "backend_context", "backends",
           "create_backend"]
