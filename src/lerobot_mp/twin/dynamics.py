"""The arm's identified servo dynamics: multipliers on the MuJoCo Menagerie model.

`sysid` measures them on the connected arm, the panel saves them as `Workspace.dynamics`
(`Dynamics.to_dict`), and `lerobot-twin export` writes them to `station.json`, where a
training package reads them as the centre of its randomization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

#: Multiplier ranges of the parameters identification does NOT fit (`Dynamics.fitted`): kp and
#: dry friction. The fit leaves them at the model's 1.0, and the fitted damping, armature and
#: delay compensate for their true value: with a true kp of 0.8 the fit had armature -13 %, so it
#: holds only together with the kp the arm really has. `sysid` measures the uncertainty of its
#: result over these ranges and names them in `Dynamics.source` as the ranges for training.
UNFITTED_RANGES: dict[str, tuple[float, float]] = {"kp": (0.6, 1.5), "frictionloss": (0.5, 2.0)}


@dataclass
class Dynamics:
    """Identified arm dynamics: multipliers relative to the Menagerie model."""

    kp: float = 1.0                 # servo position gain
    damping: float = 1.0            # joint damping
    armature: float = 1.0           # rotor inertia
    frictionloss: float = 1.0       # dry friction
    #: Delay from command to servo motion [policy ticks].
    delay: float = 0.0
    #: Where the numbers come from: "menagerie" (nominal) or a description of the measurement.
    source: str = "menagerie"
    #: Identification fit error [deg], if there was one.
    fit_deg: float | None = None
    #: Parameters identification really fitted (the rest stayed at the model).
    #: Empty = nothing measured (Menagerie, or dynamics saved before this field).
    fitted: tuple[str, ...] = ()
    #: Uncertainty of the fitted parameters (`sysid.FitResult.band`): multipliers relative
    #: (0.1 = +-10 %), delay in policy TICKS (like `delay`); inf = the measurement says nothing
    #: about the parameter. Empty = no measurement (Menagerie or older dynamics).
    band: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["fitted"] = list(self.fitted)                 # the workspace JSON has no tuples
        # inf as "inf": json.dumps writes Infinity, which strict JSON (a browser) does not read.
        out["band"] = {k: (float(v) if np.isfinite(v) else "inf") for k, v in self.band.items()}
        return out

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> Dynamics:
        data = dict(data or {})
        data["fitted"] = tuple(data.get("fitted") or ())
        data["band"] = {str(k): float(v) for k, v in (data.get("band") or {}).items()}
        return Dynamics(**data)

    def is_fitted(self, name: str) -> bool:
        """Whether `name` comes from the measurement (and not from the Menagerie model)."""
        return name in self.fitted
