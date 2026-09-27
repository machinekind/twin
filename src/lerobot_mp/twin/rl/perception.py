"""Simulation pose perturbations; independent of robot control and success truth."""
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


def rotation_vector_matrix(degrees) -> np.ndarray:
    """Axis-angle vector in degrees, expressed in the arm-base frame."""
    vector = np.asarray(degrees, float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Rotation vector must contain three finite degree values")
    vector = np.radians(vector)
    angle = float(np.linalg.norm(vector))
    if angle == 0:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross @ cross)


@dataclass(frozen=True)
class PoseCorruption:
    translation_m: tuple[float, float, float] = (0., 0., 0.)
    rotation_vector_deg: tuple[float, float, float] = (0., 0., 0.)
    position_noise_std_m: float = 0.
    rotation_noise_std_deg: float = 0.
    delay_s: float = 0.

    def __post_init__(self):
        for name in ('translation_m', 'rotation_vector_deg'):
            values = np.asarray(getattr(self, name), float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        for name in ('position_noise_std_m', 'rotation_noise_std_deg', 'delay_s'):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


class PoseCorruptor:
    """Timestamped sample-and-hold pose stream, with constant episode bias.

    Rotation bias acts on BOTH position about the base origin and orientation.
    Independent orientation jitter acts only on orientation, after that bias.
    Frames are captured at provider calls (20 Hz in the evaluator, with actual
    simulation timestamps). Delay uses the latest captured frame no newer than
    now-delay, without interpolation. During startup, the oldest available frame
    is delivered and its shorter effective age is explicitly marked as warmup.
    Duplicate reads at the same timestamp return the same noise realization.
    """

    def __init__(self, config: PoseCorruption, *, seed: int):
        self.config = config
        self.bias_rotation = rotation_vector_matrix(config.rotation_vector_deg)
        self.translation = np.asarray(config.translation_m, float)
        streams = np.random.SeedSequence(seed).spawn(2)
        self.position_rng, self.rotation_rng = [np.random.default_rng(s) for s in streams]
        self.frames = deque()
        self.last_log = None
        self._last_time = None
        self._last_truth = self._last_pose = None

    def sample(self, time_s: float, position, rotation) -> tuple[np.ndarray, np.ndarray]:
        time_s = float(time_s)
        position, rotation = np.asarray(position, float), np.asarray(rotation, float)
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Pose time must be finite and nonnegative")
        if (position.shape != (3,) or rotation.shape != (3, 3)
                or not np.isfinite(position).all() or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
            raise ValueError("Expected a finite position and proper 3x3 rotation")
        if self._last_time is not None:
            if time_s < self._last_time:
                raise ValueError("Pose timestamps must not move backwards")
            if time_s == self._last_time:
                if not all(np.array_equal(a, b) for a, b in zip((position, rotation), self._last_truth)):
                    raise ValueError("True pose changed without advancing its timestamp")
                return tuple(value.copy() for value in self._last_pose)
        self.frames.append((time_s, position.copy(), rotation.copy()))
        cutoff = time_s - self.config.delay_s
        while len(self.frames) > 1 and self.frames[1][0] <= cutoff + 1e-10:
            self.frames.popleft()
        source_time, delayed_position, delayed_rotation = self.frames[0]
        position_noise = self.position_rng.normal(size=3) * self.config.position_noise_std_m
        rotation_noise = self.rotation_rng.normal(size=3) * self.config.rotation_noise_std_deg
        if any(self.config.rotation_vector_deg):
            biased_position = self.bias_rotation @ delayed_position
            biased_rotation = self.bias_rotation @ delayed_rotation
        else:
            biased_position, biased_rotation = delayed_position.copy(), delayed_rotation.copy()
        biased_position = biased_position + self.translation
        supplied_position = biased_position + position_noise
        supplied_rotation = (rotation_vector_matrix(rotation_noise) @ biased_rotation
                             if self.config.rotation_noise_std_deg else biased_rotation.copy())
        self._last_time = time_s
        self._last_truth = (position.copy(), rotation.copy())
        self._last_pose = (supplied_position.copy(), supplied_rotation.copy())
        age = time_s - source_time
        self.last_log = dict(time_s=time_s, source_time_s=source_time,
            requested_delay_s=self.config.delay_s, effective_delay_s=age,
            delay_warmup=bool(age < self.config.delay_s - 1e-10),
            true_position_m=position.tolist(), true_rotation=rotation.tolist(),
            delayed_position_m=delayed_position.tolist(), delayed_rotation=delayed_rotation.tolist(),
            biased_position_m=biased_position.tolist(), biased_rotation=biased_rotation.tolist(),
            supplied_position_m=supplied_position.tolist(), supplied_rotation=supplied_rotation.tolist(),
            position_noise_m=position_noise.tolist(), rotation_noise_vector_deg=rotation_noise.tolist(),
            corruption=asdict(self.config))
        return supplied_position, supplied_rotation
