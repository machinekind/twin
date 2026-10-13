"""The robot backend contract: how the twin talks to an arm.

The twin talks to every arm through `RobotBackend`, so its simulated arm and a real one are
interchangeable. The contract (`BackendContext`, `BackendPlugin`, `RobotInfo`, `RobotBackend`)
lives in the `lerobot-mp-api` package, `lerobot_mp_api.backend`: a backend from another package
(a robot's own repository) depends on that package, never on the twin. It implements the
`BackendPlugin` protocol, and `factory.create_backend` wraps it into a `RobotBackend`. Such a
package registers a factory under the `lerobot_mp.backends` entry point group; the twin calls it
with a `BackendContext`, plain data with everything the backend may use.

Backends come from installed packages. This module re-exports the same classes, so
`lerobot_mp.robot.base.RobotBackend is lerobot_mp_api.RobotBackend`.
"""

from lerobot_mp_api.backend import BackendContext, BackendPlugin, RobotBackend, RobotInfo

__all__ = ["BackendContext", "BackendPlugin", "RobotBackend", "RobotInfo"]
