"""Twin configuration: dataclasses and YAML loading.

`TwinConfig` holds what the twin itself reads: safety, which backend and port to use, and the
joint table of the robot it runs. Every other top-level mapping in the file belongs to a backend
plugin or a strategy of that name and is kept raw in `TwinConfig.sections`, keyed by the section
name: a backend validates its own section, and `strategies` builds a strategy's settings from its.
`from_mapping` is the strict dataclass builder the twin uses for its sections, and the backends
still in this repository use it for theirs.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import types
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from ..paths import CONFIG_ENV, config_from_env

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    #: Camera index (0, 1, ...) or a path to a video file / stream.
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    #: Mirrored image, so that a hand moving right moves right in the picture.
    mirror: bool = True
    #: Forcing the MJPG codec often unlocks 30+ FPS on USB cameras.
    fourcc: str | None = "MJPG"
    #: Loop a video file (for demos without a camera).
    loop_video: bool = True


@dataclass
class JointConfig:
    """Limits and dynamics of one joint (degrees, gripper in 0..100)."""

    min: float
    max: float
    #: Maximum commanded speed [unit/s]: a hard safety limit.
    max_vel: float = 120.0
    #: Gain of the hand-feature to joint mapping. The unit depends on what drives the joint:
    #:   * hand position/depth (pan, lift, elbow) -> degrees per normalized unit
    #:     (moving the hand across half the frame is about 0.5),
    #:   * hand angle (wrist_flex, wrist_roll) -> dimensionless ratio
    #:     (1.0 = the wrist turns 1:1 with the hand).
    gain: float = 1.0
    #: Reverses the direction (when your calibration has the opposite sign).
    invert: bool = False
    #: Constant offset added to the target.
    offset: float = 0.0


@dataclass
class SafetyConfig:
    #: No hand for longer than this many seconds -> freeze the target.
    hold_timeout_s: float = 0.4
    #: No hand for longer than this many seconds -> return to the home pose.
    return_home_timeout_s: float = 6.0
    #: Smooth approach to the home pose at start [s].
    startup_ramp_s: float = 2.5
    #: Global multiplier of the speed limits (0.5 = half, the "slow" mode).
    velocity_scale: float = 1.0
    #: Home/rest pose [app units]. Empty = the robot description's home (`robots.robot_config`).
    home: dict[str, float] = field(default_factory=dict)
    #: Return to the home pose before disconnecting.
    home_on_exit: bool = True


@dataclass
class RobotConfig:
    #: "sim" (the twin's own physics) or the name of a backend plugin (`robot.factory.backends`).
    backend: str = "sim"
    #: Where the backend finds the arm: a serial port (/dev/ttyACM0, COM5), a CAN interface
    #: (can0, xcan) - whatever that backend understands. Empty = the backend's default.
    port: str | None = None


@dataclass
class TwinConfig:
    """What the twin reads, plus the raw sections that belong to backend plugins."""

    safety: SafetyConfig = field(default_factory=SafetyConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    #: The joint table of the robot the twin runs; `robots.robot_config` fills it from the description.
    joints: dict[str, JointConfig] = field(default_factory=dict)
    #: Every other top-level mapping of the file, by name, unvalidated: a backend plugin or a strategy reads its own.
    sections: dict[str, dict[str, Any]] = field(default_factory=dict)

    def joint(self, name: str) -> JointConfig:
        try:
            return self.joints[name]
        except KeyError as exc:  # pragma: no cover - configuration error
            raise KeyError(f"Brak konfiguracji stawu '{name}' w sekcji `joints`") from exc


# --------------------------------------------------------------------------
# Loading YAML -> dataclasses
# --------------------------------------------------------------------------


def _coerce(value: Any, target_type: Any, where: str) -> Any:
    """Casts a YAML value to the dataclass field type: tuples, nested mappings, dataclasses."""
    origin = get_origin(target_type)
    args = get_args(target_type)

    if origin in (Union, types.UnionType):
        if value is None:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(value, inner[0], where) if len(inner) == 1 else value
    if is_dataclass(target_type):
        return from_mapping(target_type, value, where)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], where) for v in value)
        if args and len(value) != len(args):
            raise ValueError(f"{where}: oczekiwano {len(args)} wartosci, dostalem {len(value)}")
        return tuple(_coerce(v, t, where) for v, t in zip(value, args)) if args else tuple(value)
    if origin is dict:
        if not isinstance(value, Mapping):
            raise TypeError(f"{where}: oczekiwano mapowania, dostalem {type(value).__name__}")
        _, val_t = args
        return {k: _coerce(v, val_t, f"{where}.{k}") for k, v in value.items()}
    if target_type is Path:
        return Path(value)
    return value


def from_mapping(cls: Any, data: Any, where: str | None = None) -> Any:
    """Builds the dataclass `cls` from a mapping, strictly: unknown keys fail, gaps take defaults.

    `where` names the section in error messages (default: the class name).
    """
    where = where or cls.__name__
    if not isinstance(data, Mapping):
        raise TypeError(f"Oczekiwano mapowania dla {where}, dostalem {type(data).__name__}")

    # `from __future__ import annotations` turns annotations into strings, so resolve them to
    # real types before casting anything.
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Nieznane klucze w sekcji {where}: {sorted(unknown)}")
    missing = [f.name for f in fields(cls)
               if f.name not in data and f.default is MISSING and f.default_factory is MISSING]
    if missing:
        raise ValueError(f"Brak kluczy w sekcji {where}: {missing}")

    kwargs = {name: _coerce(value, hints[name], f"{where}.{name}") for name, value in data.items()}
    return cls(**kwargs)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def read_config_data(path: str | Path | None = None) -> dict[str, Any]:
    """Raw content of the YAML config file, `{}` without a file.

    Without `path` the file comes from the `LEROBOT_MP_CONFIG` environment variable
    (if it is set).
    """
    if path is None:
        path = config_from_env()
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{CONFIG_ENV}={path}: nie ma takiego pliku konfiguracji")
    if path is None:
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Plik konfiguracyjny {path} musi zawierac mapowanie YAML")
    return raw


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    joints: Mapping[str, JointConfig] | None = None,
) -> TwinConfig:
    """Loads the configuration: defaults <- YAML file <- overrides from code.

    Without `path` the file comes from the `LEROBOT_MP_CONFIG` environment variable (if set).
    That way the twin, which loads the configuration in several places including the training
    process, gets the port and the backend sections from the file given once in
    `lerobot-twin --config ...`.

    `joints` is the joint table of the robot (`robots.robot_config` passes the description's);
    the YAML `joints` section and the overrides merge into it, so a joint outside it is an error.
    """
    return config_from_data(read_config_data(path), overrides, joints=joints)


def config_from_data(
    data: Mapping[str, Any],
    overrides: dict[str, Any] | None = None,
    *,
    joints: Mapping[str, JointConfig] | None = None,
) -> TwinConfig:
    """`load_config` for file content that is already read (`read_config_data`)."""
    merged = _deep_merge(data, overrides or {})
    own = {f.name for f in fields(TwinConfig)} - {"sections"}
    sections: dict[str, dict[str, Any]] = {}
    for key, value in merged.items():
        if key in own:
            continue
        if isinstance(value, Mapping):
            sections[str(key)] = copy.deepcopy(dict(value))
        else:
            logger.debug("Skipping top-level config value %r: not a section", key)

    # The `joints` section merges into the joint table joint by joint, so a single limit can be
    # overridden without rewriting the whole table.
    twin_data = {k: v for k, v in merged.items() if k in own}
    joints_override = twin_data.pop("joints", None)
    cfg = from_mapping(TwinConfig, twin_data)
    cfg.sections = sections
    if joints is not None:
        cfg.joints = {k: copy.deepcopy(v) for k, v in joints.items()}
    if joints_override:
        for joint_name, joint_data in joints_override.items():
            if joint_name not in cfg.joints:
                raise ValueError(f"Nieznany staw w sekcji `joints`: {joint_name}")
            merged_joint = _deep_merge(dataclasses.asdict(cfg.joints[joint_name]), joint_data)
            cfg.joints[joint_name] = from_mapping(JointConfig, merged_joint, f"joints.{joint_name}")
    return cfg
