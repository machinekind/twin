"""The digital twin's plugin contracts: what a robot, backend, strategy or detector package implements.

A plugin package depends on this package (only numpy), never on the twin (`machinekind-twin`). The
twin re-exports the same classes from `machinekind_twin.strategy` and `machinekind_twin.backends.base`.

* `strategy`: `Observation` in, `Action` out, the `Strategy` protocol;
* `backend`: `BackendContext` in, the `BackendPlugin` protocol or a `RobotBackend` subclass out;
* `detector`: frames in, Detections (pixels) out, the `Detector` protocol;
* `entry_points`: the groups under which a package registers its robots, backends, strategies and
  detectors.
"""

from .backend import BackendContext, BackendPlugin, RobotBackend, RobotInfo
from .detector import Detection, Detector, DetectorInfo, DetectRequest
from .entry_points import BACKENDS, DETECTORS, ROBOTS, STRATEGIES
from .strategy import Action, ActionLike, Decision, Frame, ObjectPose, Observation, RunInfo, S, Strategy

__version__ = "0.1.3"

__all__ = [
    "BACKENDS", "DETECTORS", "ROBOTS", "STRATEGIES", "Action", "ActionLike", "BackendContext",
    "BackendPlugin", "Decision", "DetectRequest", "Detection", "Detector", "DetectorInfo", "Frame",
    "ObjectPose", "Observation", "RobotBackend", "RobotInfo", "RunInfo", "S", "Strategy", "__version__",
]
