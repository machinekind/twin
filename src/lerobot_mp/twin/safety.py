"""Safety supervision: the last link before a command goes to the servos.

Rule: *nothing* reaches the robot without passing this class. Even if the mapping or the
hand detection went wild, the target is clamped to the joint limits and rate-limited here,
and a missing hand freezes the motion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from .config import GRIPPER, TwinConfig

logger = logging.getLogger(__name__)


class RateLimiter:
    """Limits how fast a signal changes (slew rate): a hard safety limit."""

    def __init__(self, max_rate: float, value: float = 0.0):
        if max_rate <= 0:
            raise ValueError("max_rate musi byc dodatnie")
        self.max_rate = max_rate
        self._value = value

    def __call__(self, target: float, dt: float) -> float:
        max_step = self.max_rate * max(dt, 0.0)
        delta = target - self._value
        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step
        self._value += delta
        return self._value

    @property
    def value(self) -> float:
        return self._value

    def reset(self, value: float) -> None:
        self._value = value


class SafetyState(str, Enum):
    """State of the safety state machine."""

    STARTING = "STARTING"      # smooth approach to the home pose after start
    IDLE = "IDLE"              # ready, control inactive
    ACTIVE = "ACTIVE"          # the robot follows the hand
    HOLDING = "HOLDING"        # hand lost / clutch released: stand still
    HOMING = "HOMING"          # returning to the home pose
    ESTOP = "ESTOP"            # emergency stop


@dataclass
class SafetyReport:
    """What the supervisor did in this step, for the HUD."""

    state: SafetyState = SafetyState.IDLE
    #: Joints that hit a position limit.
    at_limit: list[str] = field(default_factory=list)
    #: Joints whose commanded motion was cut by the speed limit.
    rate_limited: list[str] = field(default_factory=list)
    seconds_without_hand: float = 0.0


def _smoothstep(t: float) -> float:
    """Smooth 0->1 transition with zero speed at both ends."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class SafetySupervisor:
    """Position and speed limits, hand watchdog, emergency stop."""

    def __init__(self, cfg: TwinConfig):
        self.cfg = cfg
        self.state = SafetyState.STARTING
        self._limiters: dict[str, RateLimiter] = {}
        self._command: dict[str, float] = {}
        self._ramp_from: dict[str, float] = {}
        self._ramp_to: dict[str, float] = {}
        self._ramp_t = 0.0
        self._ramp_duration = max(cfg.safety.startup_ramp_s, 1e-3)
        #: Ramp time [s] from which a joint's own ramp counts (missing = from zero).
        #: Twin only: a joint outside its limits starts its ramp only after it is back in range.
        self._ramp_start: dict[str, float] = {}
        self._no_hand_for = 0.0
        self._estop = False
        self._was_active = False
        #: Supervised joints: those of the configuration, in its order (the SO-101 in the hand
        #: app, the arm's own joints from its description in the twin, `robots.robot_config`).
        self.names: tuple[str, ...] = tuple(cfg.joints)
        self.home = self._clamped_home()
        #: Twin mode (`start(..., keep_outside=True)`): a joint standing outside the configured
        #: limits gets its own widened limits up to where it stands. The limits follow the
        #: command inwards and vanish once the joint is back in range. Empty dict = plain
        #: limits (hand app).
        self._soft: dict[str, tuple[float, float]] = {}
        self._keep_outside = False
        #: Speed [unit/s] at which a joint outside the limits returns into range.
        #: Measured on arm no. 2 (shoulder_lift at -101 deg with a limit of -95): clamping at
        #: start gave a 6 deg jump at full servo speed.
        self.outside_vel = 15.0

    # ------------------------------------------------------------- helpers
    def _clamped_home(self) -> dict[str, float]:
        home = {}
        for name in self.names:
            jc = self.cfg.joint(name)
            value = self.cfg.safety.home.get(name, 0.0)
            home[name] = min(max(value, jc.min), jc.max)
        return home

    def set_speed_cap(self, deg_s: float | None) -> None:
        """Upper speed bound of every arm joint [units/s], like 'speed' in jog_a1x.

        None = no bound (only `max_vel` x `velocity_scale`). The gripper is unchanged.
        The limiters take the new pace at once, without a jump (the step counts from the last
        command). A joint outside its limits still returns at most at `outside_vel` (`step`).
        """
        self.speed_cap = None if deg_s is None else max(float(deg_s), 1e-3)
        for name, limiter in self._limiters.items():
            limiter.max_rate = self._max_vel(name)

    def _max_vel(self, name: str) -> float:
        v = max(self.cfg.joint(name).max_vel * self.cfg.safety.velocity_scale, 1e-3)
        cap = getattr(self, "speed_cap", None)
        if cap is not None and name != GRIPPER:
            v = min(v, cap)
        return v

    # ------------------------------------------------------------------ start
    def start(self, measured: dict[str, float], go_home: bool = True, *, keep_outside: bool = False) -> None:
        """Initializes the supervisor with the robot's current, *measured* pose.

        We start exactly where the arm really stands and only from there move smoothly to the
        home pose; otherwise the first command would be a jump of several tens of degrees.
        `go_home=False` leaves the arm where it stands: the twin panel connects without motion
        and has a separate button.

        `keep_outside=True` (twin): a joint outside the limits is NOT clamped at start. The
        command begins exactly at the measured pose, and the joint returns into range slowly
        (`outside_vel`) only once control is on (ACTIVE) or in the ramp home. Without it,
        "connect without motion" sent a jump to the limit as its first command.
        """
        self._command = {}
        self._limiters = {}
        self._soft = {}
        self._keep_outside = keep_outside
        for name in self.names:
            jc = self.cfg.joint(name)
            value = float(measured.get(name, self.home[name]))
            if keep_outside and not jc.min <= value <= jc.max:
                self._soft[name] = (min(jc.min, value), max(jc.max, value))
            else:
                value = min(max(value, jc.min), jc.max)
            self._command[name] = value
            self._limiters[name] = RateLimiter(self._max_vel(name), value)

        self._ramp_from = dict(self._command)
        self._ramp_to = dict(self.home)
        self._ramp_t = 0.0
        self._ramp_start = {}
        self._no_hand_for = 0.0
        self._was_active = False
        if not go_home:
            self.state = SafetyState.IDLE
            logger.info("Nadzor wystartowal - ramie zostaje tam, gdzie stoi.")
            return
        self.state = SafetyState.STARTING
        logger.info("Nadzor wystartowal, plynne dojscie do pozycji domowej (%.1f s).", self._ramp_duration)

    # ------------------------------------------------------------------- step
    def step(
        self,
        desired: dict[str, float] | None,
        dt: float,
        *,
        hand_present: bool,
        engaged: bool,
    ) -> tuple[dict[str, float], SafetyReport]:
        """Returns the safe command for the robot and the report for the HUD."""
        if not self._command:
            raise RuntimeError("Wywolaj SafetySupervisor.start() przed step()")

        dt = max(dt, 0.0)
        self._no_hand_for = 0.0 if hand_present else self._no_hand_for + dt

        target = self._resolve_target(desired, dt, hand_present=hand_present, engaged=engaged)

        report = SafetyReport(state=self.state, seconds_without_hand=self._no_hand_for)
        command: dict[str, float] = {}
        for name in self.names:
            jc = self.cfg.joint(name)
            raw = float(target.get(name, self._command[name]))
            outside = name in self._soft
            lo, hi = self._soft.get(name, (jc.min, jc.max))
            if outside and self.state == SafetyState.ACTIVE:
                # Control is on: the joint returns into range (slowly, see below) even if the
                # target does not move it. The drivers add to the target the current command,
                # so the joint would otherwise stay outside the limits for the rest of the
                # session, in a pose outside what the policies were trained on.
                lo, hi = jc.min, jc.max

            clamped = min(max(raw, lo), hi)
            if abs(clamped - raw) > 1e-6 or outside:
                report.at_limit.append(name)

            limiter = self._limiters[name]
            limiter.max_rate = self._max_vel(name)
            if outside and not jc.min <= limiter.value <= jc.max:
                limiter.max_rate = min(limiter.max_rate, self.outside_vel)
            value = limiter(clamped, dt)
            if abs(value - clamped) > 1e-6:
                report.rate_limited.append(name)
            command[name] = value
            if outside:
                # The limit follows the command inwards, never back outwards.
                lo, hi = self._soft[name]
                lo, hi = min(jc.min, max(lo, value)), max(jc.max, min(hi, value))
                if lo >= jc.min and hi <= jc.max:
                    del self._soft[name]
                    if self.state in (SafetyState.STARTING, SafetyState.HOMING):
                        # The joint came back into range during the ramp: restart its ramp
                        # from here. The ramp target had run far ahead (the joint crawled at
                        # 15 deg/s) and the limiter went back to the full max_vel in range.
                        # Measured: wrist_roll 160 -> home, command -15 deg/s for 8 ticks,
                        # then at once -220 deg/s.
                        self._ramp_from[name] = value
                        self._ramp_start[name] = self._ramp_t
                else:
                    self._soft[name] = (lo, hi)

        self._command = command
        return dict(command), report

    def _resolve_target(
        self,
        desired: dict[str, float] | None,
        dt: float,
        *,
        hand_present: bool,
        engaged: bool,
    ) -> dict[str, float]:
        """State machine: decides *where* we want to go at all."""
        if self._estop:
            self.state = SafetyState.ESTOP
            return dict(self._command)

        if self.state in (SafetyState.STARTING, SafetyState.HOMING):
            self._ramp_t += dt
            target = {}
            for name in self.names:
                alpha = _smoothstep((self._ramp_t - self._ramp_start.get(name, 0.0)) / self._ramp_duration)
                target[name] = self._ramp_from[name] + alpha * (self._ramp_to[name] - self._ramp_from[name])
            # A joint outside the limits (twin only, `keep_outside`) ends the ramp only after it
            # is back in range and has run its own ramp from there. Measured: wrist_roll 45 deg
            # past the limit (the 15 deg/s return takes longer than the 2.5 s ramp): Home
            # finished (IDLE) with a command of 158 deg, and outside ACTIVE the widened limits
            # stay, so the joint stayed out of range. The hand app has no `_soft`: unchanged.
            if not self._soft and self._ramp_t >= self._ramp_duration + max(self._ramp_start.values(), default=0.0):
                self.state = SafetyState.IDLE
                self._was_active = False
            return target

        if engaged and desired is not None and hand_present:
            self.state = SafetyState.ACTIVE
            self._was_active = True
            return desired

        # Control inactive: hold the position...
        if self._no_hand_for >= self.cfg.safety.return_home_timeout_s > 0:
            # ...and after a long absence of the hand return to the home pose.
            self.begin_homing()
            return dict(self._command)

        # HOLDING means "we were in control and lost the hand", not "waiting for the first
        # hand"; otherwise the app would start in an alarm state.
        lost_hand = self._no_hand_for > 0 and self._was_active
        self.state = SafetyState.HOLDING if lost_hand else SafetyState.IDLE
        return dict(self._command)

    # ------------------------------------------------------------- control
    def begin_homing(self, duration: float | None = None, *, keep: dict[str, float] | None = None) -> None:
        """Starts a smooth return to the home pose.

        `keep` (twin): joints that the ramp should take to the given value instead of home,
        e.g. a gripper squeezing a cube does not open on the way.
        """
        self._ramp_from = dict(self._command)
        self._ramp_to = dict(self.home)
        for name, value in (keep or {}).items():
            if name in self._ramp_to:
                jc = self.cfg.joint(name)
                self._ramp_to[name] = min(max(float(value), jc.min), jc.max)
        self._ramp_duration = max(duration or self.cfg.safety.startup_ramp_s, 1e-3)
        self._ramp_t = 0.0
        self._ramp_start = {}
        self._no_hand_for = 0.0
        self._was_active = False
        self.state = SafetyState.HOMING

    def reseed(self, joints: dict[str, float]) -> None:
        """Restarts the command and speed limiters of the given joints at this pose, keeping the state.

        For the twin: when the backend sent something other than the command (the servo clamps
        to its own limits), the supervisor must continue from what really went out.
        """
        if not self._command:
            return
        for name, raw in joints.items():
            if name not in self._command:
                continue
            jc = self.cfg.joint(name)
            value = float(raw)
            if self._keep_outside and not jc.min <= value <= jc.max:
                self._soft[name] = (min(jc.min, value), max(jc.max, value))
            else:
                self._soft.pop(name, None)
                value = min(max(value, jc.min), jc.max)
            self._command[name] = value
            self._limiters[name].reset(value)

    def hold(self, joints: dict[str, float]) -> None:
        """Stand where the arm IS (the measured pose), not where the command was heading.

        Interrupts the ramp and the control. After a collision the last command lies several
        tens of degrees behind the obstacle; holding it (as `set_engaged(False)` alone does)
        left the servos pressing against it at full torque. An emergency STOP stays a STOP,
        it just holds the measured pose.
        """
        self.reseed(joints)
        self._no_hand_for = 0.0
        self._was_active = False
        if not self._estop:
            self.state = SafetyState.IDLE

    def trigger_estop(self) -> None:
        self._estop = True
        self.state = SafetyState.ESTOP
        logger.warning("STOP AWARYJNY - ruch zatrzymany.")

    def clear_estop(self) -> None:
        if not self._estop:
            return
        self._estop = False
        self.state = SafetyState.IDLE
        logger.info("Stop awaryjny skasowany.")

    @property
    def estopped(self) -> bool:
        return self._estop

    @property
    def command(self) -> dict[str, float]:
        return dict(self._command)

    @property
    def outside(self) -> dict[str, tuple[float, float]]:
        """Joints outside the configured limits and their temporarily widened limits (`keep_outside` only)."""
        return dict(self._soft)

    @property
    def is_homing_done(self) -> bool:
        return self.state not in (SafetyState.STARTING, SafetyState.HOMING)
