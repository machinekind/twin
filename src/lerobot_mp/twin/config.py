"""Twin and robot backend configuration: dataclasses and YAML loading.

Moved here from the hand-control app's `lerobot_mp/config.py`, deleted with that app.
`TwinConfig` holds only the sections the twin and the backends read.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from ..paths import CONFIG_ENV, config_from_env

logger = logging.getLogger(__name__)

# Same joint order as LeRobot uses for the SO-100/SO-101.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

#: LeRobot always drives the gripper in the normalized range 0..100, whether the other
#: joints are in degrees or in -100..100.
GRIPPER = "gripper"


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


def _default_joints() -> dict[str, JointConfig]:
    return {
        # Slightly tighter than the SO-101 calibration ranges
        # (+-110 / +-100 / +-96.8 / +-95 / -157..163), to keep a margin
        # before the mechanical end stop.
        "shoulder_pan": JointConfig(min=-105.0, max=105.0, max_vel=140.0, gain=90.0),
        "shoulder_lift": JointConfig(min=-95.0, max=95.0, max_vel=120.0, gain=90.0),
        "elbow_flex": JointConfig(min=-92.0, max=92.0, max_vel=120.0, gain=90.0),
        # The wrist is driven by the hand ANGLE, so its gain is a ratio (1.0 = 1:1).
        # Values in the tens would hit the limit after a few degrees of hand motion.
        "wrist_flex": JointConfig(min=-95.0, max=95.0, max_vel=160.0, gain=1.2),
        "wrist_roll": JointConfig(min=-150.0, max=150.0, max_vel=220.0, gain=1.0),
        "gripper": JointConfig(min=0.0, max=100.0, max_vel=300.0, gain=100.0),
    }


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
    #: Home/rest pose: a ready pose with the tip in the middle of the workspace and the gripper
    #: pointing slightly down (from inverse kinematics, not eyeballed). The all-zero pose
    #: (centre of the LeRobot calibration range) is valid but leaves the arm fully extended.
    home: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.0,
            "shoulder_lift": -23.3,
            "elbow_flex": 46.4,
            "wrist_flex": 3.3,
            "wrist_roll": 0.0,
            "gripper": 35.0,
        }
    )
    #: Return to the home pose before disconnecting.
    home_on_exit: bool = True


@dataclass
class RobotConfig:
    #: "sim" | "lerobot" | "feetech" | "a1x" | "auto".
    #:   lerobot - full LeRobot ecosystem compatibility (needs torch),
    #:   feetech - talks to the servos directly over the serial port (pyserial only).
    #:   a1x     - Galaxea A1X over CAN-FD (`galaxeo` package, `a1x` section; twin only).
    #:   auto    - lerobot if installed, otherwise feetech.
    backend: str = "sim"
    #: Serial port of the control board, e.g. /dev/ttyACM0 or COM5.
    #: For a1x: the CAN interface (can0 / xcan / xcan:6 / virtual:x); empty = can0 (macOS: xcan).
    port: str | None = None
    #: Arm id in LeRobot (selects the calibration file).
    robot_id: str = "so101_follower"
    #: True -> joints in degrees (the default in LeRobot >= 0.4).
    use_degrees: bool = True
    #: Hardware limit of a step relative to the current position (None = off).
    max_relative_target: float | None = 12.0
    calibration_dir: str | None = None
    #: Arm type passed to LeRobot: "so101" or "so100".
    kind: str = "so101"
    #: How often to read the real joint positions [Hz]. Every read is a transaction over the
    #: serial port, so doing it every frame makes no sense.
    read_state_hz: float = 10.0

    # --- the rest applies only to the `feetech` backend ----------------------
    #: Port speed. The STS3215 servos in the SO-101 ship set to 1 Mbaud.
    baudrate: int = 1_000_000
    #: Tick that corresponds to joint zero. The servo has 4096 ticks per turn, and the standard
    #: SO-101 assembly mounts it centred, which is 2048.
    center_ticks: int = 2048
    #: Gripper jaw range in ticks: 0 in app scale is `closed`, 100 is `open`. The values come
    #: from the limits the gripper servo had stored in its own EEPROM, i.e. from the jaw travel
    #: measured on a real arm, not from an estimate.
    #: Check yours: open the jaw by hand and read `Present_Position`.
    gripper_closed_ticks: int = 1986
    gripper_open_ticks: int = 2670
    #: Torque off on exit. Off by default, because a limp arm drops under its own weight;
    #: enable it only when you know the arm is supported.
    torque_off_on_exit: bool = False


@dataclass
class A1XConfig:
    """`a1x` backend: Galaxea A1X over CAN-FD (protocol and link from the `galaxeo` package).

    The arm is purely position-controlled: only p_des acts and kp/kd/v_des/t_ff have no effect,
    but kp 0 leaves it deaf to commands (galaxeo diag/REPORT.md), hence kp > 0.
    The arm has no brakes: disconnecting never releases the motors (FF 2), and an uncontrolled
    A1X holds the pose it stands in (galaxeo docs/SAFETY.md).
    """

    #: Gains sent in 0x050 (never 0; only p_des has authority).
    kp: float = 20.0
    kd: float = 1.0
    gripper_kp: float = 25.0
    #: Gripper p_des [rad, same units as group 7 of frame 0x052] at 100 (open) and at 0
    #: (closed); more negative = wider. Measured 2026-09-27 (galaxeo `xcan_usb --grip-test`, kp 20):
    #: closed stop about +0.039 rad (+2.2 deg), open about -1.99 rad (-114 deg). The old +0.6
    #: pushed the jaws 0.56 rad past the stop: the stall current (|effort| 7.8 instead of 0.2)
    #: overheated the gripper motor. Closed = 0.04 rad of pressure past the stop (objects are
    #: held by the `gripper_force` contact stop anyway), open = a hair before the stop so it
    #: does not lean on it.
    gripper_open_pdes: float = -1.95
    gripper_closed_pdes: float = 0.08
    #: Group 7 position from 0x052 [rad] at 0 and at 100: the stops measured 2026-09-27 (above).
    #: Only valid with a driven gripper (from the first 0x051 in the connection): at rest after
    #: power-on group 7 reads an arbitrary value (-165 deg with the jaws at -85 deg).
    #: None (both) = feedback not measured: the backend reports the last command (open loop).
    gripper_fb_closed: float | None = 0.039
    gripper_fb_open: float | None = -1.992
    #: What the backend reports as the gripper before the first command when feedback is not
    #: measured. 0 = closed: the station convention (home = jaws closed, "Set home = here"
    #: confirms it).
    gripper_start: float = 0.0
    #: Gripper p_des slew [rad/s]; the full 2.03 travel takes about 1.0 s.
    gripper_rate_pdes_s: float = 2.0
    #: |effort| of group 7 this far above rest (median of the first 0.1 s of closing) = contact:
    #: the target freezes (it grips, does not crush).
    gripper_force: float = 1.2
    #: Arm joint zero offset [deg]: subtracted on read, added on send
    #: (J3 reads about +1.49 deg at rest - measure yours).
    joint_offsets_deg: tuple[float, ...] = (0.0,) * 6
    #: Rate of the 0x050 / 0x051 stream [Hz] (the arm answers 0x052 every 5 ms).
    stream_hz: float = 200.0
    #: The backend's own p_des speed limit [deg/s]: a second layer below the twin's supervisor.
    max_step_deg_s: float = 90.0
    #: No 0x052 for longer than this [ms] = the link is silent: sending stops, the twin gets an error.
    stale_ms: float = 150.0
    #: This many identical 0x052 frames in a row = frozen telemetry (arm released, FF 2).
    frozen_frames: int = 50
    #: A command-measurement gap [deg] lasting `deaf_s` on a stationary joint = arm deaf or blocked.
    deaf_err_deg: float = 3.0
    deaf_s: float = 1.0
    #: FF 1 -> 5 -> 6 enable sequence right after connecting. Off by default: a freshly powered
    #: arm listens right away, and FF 5 briefly releases the motors. The panel has a button for it.
    enable_on_connect: bool = False
    #: Gripper over XCAN (macOS): on by default. Measured 2026-09-27 (`python -m galaxeo.xcan_usb
    #: --grip-test`): 0x051 gets through, the jaws travelled 85-116 degrees. The earlier refusal
    #: rested on an unverified comment. False = refuse ("no gripper"); env A1X_GRIPPER_XCAN=1 forces True.
    gripper_over_xcan: bool = True

    def __post_init__(self) -> None:
        if self.kp <= 0 or self.gripper_kp <= 0:
            raise ValueError("a1x.kp i a1x.gripper_kp musza byc > 0 - przy kp 0 ramie A1X przestaje "
                             "sluchac rozkazow")
        if len(self.joint_offsets_deg) != 6:
            raise ValueError("a1x.joint_offsets_deg: potrzeba 6 wartosci (arm_joint1..6)")
        self.joint_offsets_deg = tuple(float(x) for x in self.joint_offsets_deg)
        if (self.gripper_fb_closed is None) != (self.gripper_fb_open is None):
            raise ValueError("a1x.gripper_fb_closed i gripper_fb_open: podaj oba albo zadne")
        if self.gripper_fb_closed is not None and abs(self.gripper_fb_open - self.gripper_fb_closed) < 1e-6:
            raise ValueError("a1x.gripper_fb_closed i gripper_fb_open nie moga byc rowne")
        if self.stream_hz <= 0 or self.max_step_deg_s <= 0 or self.gripper_rate_pdes_s <= 0:
            raise ValueError("a1x.stream_hz, max_step_deg_s i gripper_rate_pdes_s musza byc > 0")


@dataclass
class TwinConfig:
    """What the twin and the robot backends read: safety, the arm link, A1X and the joint table."""

    safety: SafetyConfig = field(default_factory=SafetyConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    a1x: A1XConfig = field(default_factory=A1XConfig)
    joints: dict[str, JointConfig] = field(default_factory=_default_joints)

    def joint(self, name: str) -> JointConfig:
        try:
            return self.joints[name]
        except KeyError as exc:  # pragma: no cover - configuration error
            raise KeyError(f"Brak konfiguracji stawu '{name}' w sekcji `joints`") from exc


# --------------------------------------------------------------------------
# Loading YAML -> dataclasses
# --------------------------------------------------------------------------


def _coerce(value: Any, target_type: Any) -> Any:
    """Casts a YAML value to the dataclass field type, as far as needed."""
    origin = get_origin(target_type)

    if origin is tuple:
        return tuple(value)
    if is_dataclass(target_type) and isinstance(value, dict):
        return _from_dict(target_type, value)
    if origin is dict:
        key_t, val_t = get_args(target_type)
        if is_dataclass(val_t):
            return {k: _from_dict(val_t, v) for k, v in value.items()}
        return dict(value)
    return value


def _from_dict(cls: Any, data: dict[str, Any]) -> Any:
    """Builds the dataclass `cls`, filling gaps with defaults."""
    if not isinstance(data, dict):
        raise TypeError(f"Oczekiwano mapowania dla {cls.__name__}, dostalem {type(data).__name__}")

    # `from __future__ import annotations` turns annotations into strings, so resolve them to
    # real types before casting anything.
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Nieznane klucze w sekcji {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        kwargs[name] = _coerce(value, hints[name])
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
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
    """Loads the configuration: defaults <- YAML file <- CLI overrides.

    Without `path` the file comes from the `LEROBOT_MP_CONFIG` environment variable (if set).
    That way the twin, which calls `load_config()` in several places including the training
    process, gets the gripper ticks and the port from the file given once in
    `lerobot-twin --config ...`.

    `joints` replaces the default SO-101 joint table (the twin passes the joints of its own
    arm, `robots.robot_config`); the YAML `joints` section and the overrides merge into it,
    so a joint outside it is an error.

    Top-level file sections `TwinConfig` has no field for are skipped, so a user file copied
    from the old `configs/default.yaml` still loads. Unknown keys in `overrides` still fail.
    """
    return config_from_data(read_config_data(path), overrides, joints=joints)


def config_from_data(
    data: Mapping[str, Any],
    overrides: dict[str, Any] | None = None,
    *,
    joints: Mapping[str, JointConfig] | None = None,
) -> TwinConfig:
    """`load_config` for file content that is already read (`read_config_data`)."""
    # Only file sections are skipped; overrides come from code and stay strict.
    known = {f.name for f in fields(TwinConfig)}
    skipped = sorted(str(k) for k in data if k not in known)
    if skipped:
        logger.debug("Skipping config sections the twin does not read: %s", ", ".join(skipped))
    data = {k: v for k, v in data.items() if k in known}
    if overrides:
        data = _deep_merge(data, overrides)

    # The `joints` section merges into the default dictionary joint by joint, so a single
    # limit can be overridden without rewriting the whole table.
    joints_override = data.pop("joints", None)
    cfg = _from_dict(TwinConfig, data)
    if joints is not None:
        cfg.joints = {k: copy.deepcopy(v) for k, v in joints.items()}
    if joints_override:
        for joint_name, joint_data in joints_override.items():
            if joint_name not in cfg.joints:
                raise ValueError(f"Nieznany staw w sekcji `joints`: {joint_name}")
            merged = _deep_merge(dataclasses.asdict(cfg.joints[joint_name]), joint_data)
            cfg.joints[joint_name] = _from_dict(JointConfig, merged)
    return cfg
