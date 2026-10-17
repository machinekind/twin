"""The digital twin's plugin contracts: what a robot, backend or strategy package implements.

A plugin package depends on this package (only numpy), never on the twin (`machinekind-twin`). The
twin re-exports the same classes from `machinekind_twin.strategy` and `machinekind_twin.backends.base`.

* `strategy`: `Observation` in, `Action` out, the `Strategy` protocol;
* `backend`: `BackendContext` in, the `BackendPlugin` protocol or a `RobotBackend` subclass out;
* `entry_points`: the groups under which a package registers its robots, backends and strategies.
"""

from .backend import BackendContext, BackendPlugin, RobotBackend, RobotInfo
from .entry_points import BACKENDS, ROBOTS, STRATEGIES
from .strategy import Action, ActionLike, Frame, ObjectPose, Observation, RunInfo, S, Strategy

__version__ = "0.1.1"

__all__ = [
    "BACKENDS", "ROBOTS", "STRATEGIES", "Action", "ActionLike", "BackendContext", "BackendPlugin", "Frame",
    "ObjectPose", "Observation", "RobotBackend", "RobotInfo", "RunInfo", "S", "Strategy", "__version__",
]
