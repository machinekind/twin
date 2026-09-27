"""Sim evaluation of the grasp student in the twin's own scene: `lerobot-twin grasp-eval`.

    res = evaluate(twin, policy, episodes=50, seed=50000)                     # the sim jaw
    res = evaluate(twin, policy, episodes=50, jaw="measured", stroke_s=1.3)   # the jaw slowed

The twin is connected to the literal backend `"sim"`, never the workspace's, and its scene is
configured ONCE per evaluation: a large table under the arm placed as in training, one 3 cm cube
with a contact sensor on each jaw, and the wrist camera on the hand. The scene is never rebuilt
between episodes: a rebuild copies every joint state by name from the old scene, so a new cube
position given to `Box` would be overwritten by where the last episode left the cube.

Each episode draws a cube in training's region beside the arm, solves the hover with training's
hover noise, pre-positions (the arm leaves the last episode's pose straight up first, the cube
still wherever it lies), then writes the cube's full free-joint state and lets it settle before
the policy runs. Success is training's condition read from the scene inside the driver's tick:
both jaws touch the cube, its centre has risen more than `LIFT`, the jaws are commanded closed.
Every other ending is a failure and keeps its own row (`impact ...`, `lag` and `table` included).

Both configurations draw the same cubes and the same hover noise for the same seed, so the sim
jaw and the slowed jaw can be compared episode by episode.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

from ...paths import data_path
from ..kinematics import RobotKinematics, inverse
from ..runtime import Twin
from ..scene import Box, HandCamera, HandCameraJitter, Light, Table
from . import driver as driver_mod
from . import motion
from . import policy as policy_mod
from .camera import Frame, SimWristSource, WristCamera
from .contract import (
    FOVY_DEG,
    HOVER,
    IK_MAX_JUMP_DEG,
    IMG_H,
    IMG_W,
    LIFT,
    STREAM_STROKE_S,
    T_GRIPPER_CAMERA,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
)
from .driver import EpisodeReport, TickRecord
from .hover import Hover, HoverError, from_scene, training_envelope
from .rig import Rig, RigError

logger = logging.getLogger(__name__)

CUBE = "cube"
CUBE_HALF = 0.015                       # m, half side of the 3 cm cube
CUBE_X = (-0.20, 0.20)                  # m, training's object region in the base frame
CUBE_Y = (0.22, 0.40)
CUBE_YAW = (-math.pi / 4, math.pi / 4)
CUBE_LOST_XY = 0.15                     # m, training's "pushed" failure: the cube moved this far in xy
JITTER_POS_M = 0.003                    # training's wrist camera jitter, uniform half-ranges
JITTER_ROT_DEG = 0.7
JITTER_FOVY_DEG = 0.3
DRAW_ATTEMPTS = 200
SETTLE_WALL_S = 5.0                     # the scene must advance `settle_s` within this much wall time
GOOD_RATE = 0.70
BAD_RATE = 0.50
TICK_P99_MAX_MS = 40.0
OUT_DIR = Path("workspace") / "grasp" / "eval"
NOT_A1X = "polecenia chwytu dzialaja tylko na ramieniu A1X (robot w pliku stanowiska)"
#: Refusals after which nothing more can move in this evaluation.
HARD_REFUSALS = ("stop", "preempted", "not connected", "busy")
#: Pre-positioning refused this many times in a row: the arm cannot leave where it stands, and
#: every further episode would be refused the same way.
PREPOSITION_STUCK = 3
# Training's scene as the wrist camera sees it (sim/pickplace_scene.py): table colour, the two
# lights (positions above the table top) and the headlight. With the twin's own look the policy's
# first action differed from training's by up to 0.8 at the same joints, pixel and 19-vector.
TRAINING_TABLE_RGBA = (0.55, 0.45, 0.35, 1.0)
TRAINING_LIGHTS = (Light(pos=(0.0, 0.0, 1.8), dir=(0.0, 0.0, -1.0), directional=True),
                   Light(pos=(1.0, -1.0, 1.3), dir=(-0.4, 0.4, -1.0), diffuse=0.5))
TRAINING_HEADLIGHT = (0.3, 0.6, 0.0)               # ambient, diffuse, specular


class EvalRefused(RuntimeError):
    """The evaluation could not start: `reason` is a refusal code (`not simulated`, a driver refusal)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class EvalResult:
    episodes: int
    successes: int
    rate: float
    endings: dict[str, int]
    tick_ms: dict[str, float]            # median, p95, p99, max
    infer_ms: dict[str, float]           # median, p95
    rate_limited_frac: float
    ik_refused_frac: float
    collision_refused_frac: float
    config: dict                          # jaw, stroke_s, gap, floor, jitter, seed, checkpoint sha256
    jump_refused_frac: float = 0.0        # of `ik_refused_frac`: steps over IK_MAX_JUMP_DEG (training took)
    per_episode: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Draw:
    """One episode's cube and the hover solved over it (base frame)."""

    xy: np.ndarray
    yaw: float
    hover: Hover
    redraws: int


# ---------------------------------------------------------------------- scene
def sim_extras() -> dict:
    """Extras for `Twin.configure`: training's table placement and look, the cube with jaw sensors,
    the hand camera."""
    return {
        "table": Table(size=(1.2, 1.2), base_xy=(0.0, -0.20), base_yaw=0.0, rgba=TRAINING_TABLE_RGBA),
        "lights": list(TRAINING_LIGHTS),
        "headlight": TRAINING_HEADLIGHT,
        "objects": [Box(CUBE, size=(CUBE_HALF, CUBE_HALF, CUBE_HALF), xy=(0.0, 0.30))],
        "grasp_sensors": [CUBE],
        "hand_cameras": [HandCamera("wrist", T_GRIPPER_CAMERA, FOVY_DEG, IMG_W, IMG_H)],
    }


def draw_cube(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Cube xy (base frame) uniform in training's region and a yaw in (-pi/4, pi/4).

    Redrawn until the hover over it lies inside the training envelope. Joint 1 of a top-down
    A1X hover is the bearing of the point from the base axis (to about 0.01 deg), so the
    envelope is judged without solving the IK; `draw_hover` still judges the solved hover.
    """
    z = TOUCH_EXPECTED_Z + TIP_CLEAR + HOVER
    for _ in range(DRAW_ATTEMPTS):
        x = float(rng.uniform(*CUBE_X))
        y = float(rng.uniform(*CUBE_Y))
        yaw = float(rng.uniform(*CUBE_YAW))
        bearing = math.degrees(math.atan2(y, x))
        if not training_envelope({"arm_joint1": bearing}, np.array([x, y, z])):
            return np.array([x, y]), yaw
    raise RuntimeError(f"no cube inside the training envelope in {DRAW_ATTEMPTS} draws")


def draw_hover(kin: RobotKinematics, rng: np.random.Generator, *, limits: dict[str, tuple[float, float]],
               accept: Callable[[np.ndarray], bool] | None = None) -> Draw:
    """A cube and its hover with training's noise; redrawn while the solved hover is unusable.

    A hover is unusable when the IK refuses it, when the noise takes it out of the training
    envelope, or when `accept(p)` refuses its position (for example a reach volume set in the
    workspace). With no usable hover in `DRAW_ATTEMPTS` draws the `HoverError` names the most
    frequent reason.
    """
    refused: Counter[str] = Counter()
    for k in range(DRAW_ATTEMPTS):
        xy, yaw = draw_cube(rng)
        # The position before the noise first: a volume that misses training's region then
        # costs no IK solve (two full solves per draw).
        if accept is not None and not accept(np.array([xy[0], xy[1], TOUCH_EXPECTED_Z + TIP_CLEAR + HOVER])):
            refused["outside volume"] += 1
            continue
        try:
            hv = from_scene(kin, xy, yaw, touch_z=TOUCH_EXPECTED_Z, limits=limits, rng=rng)
        except HoverError as exc:
            logger.debug("cube %s yaw %.1f deg: no hover (%s), drawing again", xy.round(3).tolist(),
                         math.degrees(yaw), exc)
            refused[exc.reason] += 1
            continue
        if hv.outside_training:
            refused["outside training"] += 1
            continue
        if accept is not None and not accept(hv.p):
            refused["outside volume"] += 1
            continue
        return Draw(xy, yaw, hv, k)
    reason = refused.most_common(1)[0][0] if refused else "ik"
    raise HoverError(reason, f"no usable hover in {DRAW_ATTEMPTS} draws: {dict(refused)}")


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _free_joint(model: mujoco.MjModel, name: str) -> int:
    # The cube's free joint has no name: it is found through its body.
    body = model.body(name).id
    jnt = int(model.body_jntadr[body])
    if jnt < 0 or int(model.jnt_type[jnt]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"body {name!r} has no free joint")
    return jnt


def cube_pose(twin: Twin, name: str = CUBE) -> tuple[np.ndarray, float]:
    """Centre (3,) and yaw of the cube in the base frame, as the scene holds it now."""
    with twin.lock:
        s = twin.scene
        body = s.model.body(name).id
        pos = s.data.xpos[body].copy()
        R = s.data.xmat[body].reshape(3, 3).copy()
        T = s.T_base2world.copy()
    centre = (inverse(T) @ np.append(pos, 1.0))[:3]
    R_base = T[:3, :3].T @ R
    return centre, math.atan2(R_base[1, 0], R_base[0, 0])


def _settle(twin: Twin, until: float) -> None:
    """Waits until the scene's clock reaches `until`; a manual twin is stepped here by loop periods."""
    deadline = time.monotonic() + SETTLE_WALL_S
    while True:
        with twin.lock:
            now = float(twin.scene.data.time)
        if now >= until:
            return
        if getattr(twin, "_manual", False):
            twin.step(1.0 / twin.loop_hz)
        elif time.monotonic() > deadline:
            raise RuntimeError("the scene did not advance: is the twin connected?")
        else:
            time.sleep(0.01)


def place_cube(twin: Twin, xy: np.ndarray, yaw: float, *, name: str = CUBE,
               settle_s: float = 0.2) -> np.ndarray:
    """Puts the cube on the table at `xy` (base frame) turned by `yaw`; returns the settled centre (3,).

    Writes the FULL free-joint state (position, orientation, zero velocity) under the twin's
    lock, so the cube does not keep the pose or the motion the last episode gave it.
    """
    x, y = float(xy[0]), float(xy[1])
    with twin.lock:
        s = twin.scene
        m, d = s.model, s.data
        jnt = _free_joint(m, name)
        body = m.body(name).id
        half_z = float(m.geom_size[int(m.body_geomadr[body]), 2])
        T = s.T_base2world
        p = T @ np.array([x, y, half_z, 1.0])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (T[:3, :3] @ _rot_z(float(yaw))).ravel())
        qa, va = int(m.jnt_qposadr[jnt]), int(m.jnt_dofadr[jnt])
        d.qpos[qa:qa + 3] = p[:3]
        d.qpos[qa + 3:qa + 7] = quat
        d.qvel[va:va + 6] = 0.0
        mujoco.mj_forward(m, d)
        t0 = float(d.time)
    _settle(twin, t0 + settle_s)
    return cube_pose(twin, name)[0]


def scene_success(twin: Twin, z0: float, *, name: str = CUBE) -> Callable[[], bool]:
    """Training's success read from the scene: both jaws touch the cube and it rose more than `LIFT`.

    The driver adds the closed jaw command (`g_cmd < CLOSE_BELOW`) before it calls this.
    """
    sensors = (f"{name}_jaw0", f"{name}_jaw1")

    def success() -> bool:
        with twin.lock:
            s = twin.scene
            touching = all(float(s.data.sensor(k).data[0]) > 0.0 for k in sensors)
            if not touching:
                return False
            pos = s.data.xpos[s.model.body(name).id].copy()
            T = s.T_base2world.copy()
        z = float((inverse(T) @ np.append(pos, 1.0))[2])
        return z - z0 > LIFT

    return success


def _draw_jitter(rng: np.random.Generator) -> HandCameraJitter:
    return HandCameraJitter(
        dpos=rng.uniform(-JITTER_POS_M, JITTER_POS_M, 3),
        rotvec=np.radians(rng.uniform(-JITTER_ROT_DEG, JITTER_ROT_DEG, 3)),
        dfovy_deg=float(rng.uniform(-JITTER_FOVY_DEG, JITTER_FOVY_DEG)),
    )


class _EpisodeFrames:
    """The driver's frame source; each episode gets a fresh `SimWristSource` carrying its jitter."""

    def __init__(self, twin: Twin, clock: Callable[[], float] = time.monotonic) -> None:
        self.twin = twin
        self.clock = clock
        self.src = SimWristSource(twin, clock=clock)

    def use(self, jitter: HandCameraJitter | None) -> None:
        self.src = SimWristSource(self.twin, jitter=jitter, clock=self.clock)

    def latest(self) -> Frame | None:
        return self.src.latest()

    def close(self) -> None:
        self.src.close()


# ---------------------------------------------------------------------- reports
def _stats(values: list[float], qs: dict[str, float]) -> dict[str, float]:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], float)
    if v.size == 0:
        return {}
    return {name: float(np.percentile(v, q)) for name, q in qs.items()}


def refused_report(ending: str, detail: str = "") -> EpisodeReport:
    """An `EpisodeReport` for an episode that never ran its ticks (pre-positioning refused)."""
    values: dict[str, Any] = {}
    for f in fields(EpisodeReport):
        kind = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
        if kind.startswith("str"):
            values[f.name] = ""
        elif kind.startswith("int"):
            values[f.name] = 0
        elif kind.startswith("float"):
            values[f.name] = math.nan if f.name.startswith("min_") else 0.0
        elif kind.startswith("list"):
            values[f.name] = []
        elif kind.startswith("dict"):
            values[f.name] = {}
        else:
            values[f.name] = None
    values.update(ending=ending, detail=detail)
    return EpisodeReport(**values)


def _mm(v: float | None) -> str:
    return "-" if v is None or not math.isfinite(v) else f"{v * 1000.0:.1f}"


def episode_line(k: int, total: int, rep: EpisodeReport, *, success: bool | None = None,
                 shadow: bool = False) -> str:
    """One Polish line per episode for the terminal."""
    head = f"Epizod {k + 1}/{total}: {rep.ending}"
    if rep.detail:
        head += f" ({rep.detail})"
    if success is not None:
        head += " - SUKCES" if success else " - porazka"
    parts = [f"tiki {rep.ticks} (w tym pominiete {rep.skipped})", f"proby {rep.attempts}",
             f"IK odrzucone {rep.ik_refused}", f"kolizje odrzucone {rep.collision_refused}",
             f"ogr. predkosci {rep.rate_limited}", f"max opoznienie {rep.max_lag_deg:.1f} st.",
             f"min TCP z {_mm(rep.min_tcp_z)} mm", f"podloga {_mm(rep.floor_z)} -> {_mm(rep.floor_z_end)} mm",
             f"ugiecie {_mm(rep.sag)} mm"]
    if shadow or rep.floor_raised or rep.box_hits:
        parts += [f"podloga podniesiona {rep.floor_raised}x", f"brzeg pudelka {rep.box_hits}x",
                  f"podloga dotknieta {rep.floor_hits}x"]
    if shadow:
        share = rep.rate_limited / rep.ticks if rep.ticks else 0.0
        parts.append(f"ogr. predkosci {100.0 * share:.0f} % tikow")
    if rep.tick_ms:
        parts.append("tik " + ", ".join(f"{k} {v:.1f}" for k, v in rep.tick_ms.items()) + " ms")
    if rep.no_effort_rule:
        parts.append(f"bez reguly wysilku: {', '.join(rep.no_effort_rule)}")
    if rep.log:
        parts.append(f"log {rep.log}")
    return head + "\n    " + "; ".join(parts)


def verdicts(res: EvalResult) -> list[str]:
    """The Polish verdict lines for a finished evaluation."""
    out: list[str] = []
    pct = 100.0 * res.rate
    jaw = res.config.get("jaw")
    if jaw == "sim":
        if res.rate >= GOOD_RATE:
            out.append(f"Werdykt: {pct:.0f} % przy szczece sim - dobrze (>= {GOOD_RATE * 100:.0f} %).")
        elif res.rate >= BAD_RATE:
            out.append(f"Werdykt: {pct:.0f} % przy szczece sim - ponizej {GOOD_RATE * 100:.0f} %. Przejrzyj "
                       f"zakonczenia (impact, lag, table, tick cap) i logi epizodow, zanim cos zmienisz.")
        else:
            out.append(f"Werdykt: {pct:.0f} % przy szczece sim - ponizej 50 %: przyczyna moze byc sciezka "
                       f"obserwacji, fizyka kontaktu albo render sceny blizniaka.")
            out.append("  Scena blizniaka nie odtwarza fizyki kontaktu z treningu (condim, losowanie tarcia, "
                       "noslip, stozek) ani jego oswietlenia, wiec te trzy przyczyny waza tyle samo.")
            out.append("  Sciezke obserwacji sprawdzaja testy skladowych (siec, obserwacja, rzut wzgledem "
                       "nagranych fixture): pytest tests/test_twin_grasp_policy.py "
                       "tests/test_twin_grasp_observation.py tests/test_twin_grasp_camera.py.")
            out.append("  Podloge i model szczeliny oddziela ponowny przebieg z "
                       "--floor training --gap scene.")
    else:
        out.append(f"Wynik przy spowolnionej szczece ({res.config.get('stroke_s')} s): {pct:.0f} % "
                   f"- zapisany do porownania.")
        if res.rate < BAD_RATE:
            out.append("  Ponizej 50 %: otworz zgloszenie o ponowny trening ze szczeka spowolniona do "
                       "zmierzonego skoku i opozniona obserwacja szczeliny.")
    p99 = res.tick_ms.get("p99")
    if p99 is not None:
        ok = p99 < TICK_P99_MAX_MS
        out.append(f"Czas tiku p99 {p99:.1f} ms: {'OK' if ok else 'ZA WOLNO'} "
                   f"(limit {TICK_P99_MAX_MS:.0f} ms).")
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def episode_log_dir(out: Path) -> Path:
    """Directory of the episode logs next to the result file `out`: its name without the suffix."""
    out = Path(out)
    # Without a suffix the directory would take the result file's own path.
    return out.with_suffix("") if out.suffix else out.with_name(out.name + "_episodes")


def write_result(res: EvalResult, path: Path) -> Path:
    """Writes the result as JSON (non-finite numbers as null), through a `.tmp` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(asdict(res)), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------- evaluation
def _checkpoint_sha(policy: Any) -> str | None:
    sha = getattr(policy, "sha256", None)
    if sha:
        return str(sha)
    path = getattr(policy, "path", None)
    if path is not None and Path(path).is_file():
        return policy_mod.sha256_of(Path(path))
    return None


def _result(n: int, reports: list[tuple[EpisodeReport, bool]], ticks: list[TickRecord], config: dict,
            per_episode: list[dict]) -> EvalResult:
    successes = sum(1 for _, ok in reports if ok)
    endings = Counter(rep.ending for rep, _ in reports)
    # `ticks` counts every tick run, skipped ones included; only the others asked the IK.
    all_ticks = sum(rep.ticks for rep, _ in reports)
    policy_ticks = sum(rep.ticks - rep.skipped for rep, _ in reports)
    return EvalResult(
        episodes=n,
        successes=successes,
        rate=successes / n if n else 0.0,
        endings=dict(endings.most_common()),
        tick_ms=_stats([r.tick_ms for r in ticks], {"median": 50, "p95": 95, "p99": 99, "max": 100}),
        infer_ms=_stats([r.infer_ms for r in ticks], {"median": 50, "p95": 95}),
        rate_limited_frac=sum(rep.rate_limited for rep, _ in reports) / all_ticks if all_ticks else 0.0,
        ik_refused_frac=sum(rep.ik_refused for rep, _ in reports) / policy_ticks if policy_ticks else 0.0,
        collision_refused_frac=(sum(rep.collision_refused for rep, _ in reports) / policy_ticks
                                if policy_ticks else 0.0),
        config=dict(config),
        jump_refused_frac=(sum(rep.jump_refused for rep, _ in reports) / policy_ticks
                           if policy_ticks else 0.0),
        per_episode=list(per_episode),
    )


def evaluate(twin: Twin, policy: policy_mod.GraspStudentPolicy, *, episodes: int = 50, seed: int = 50000,
             jaw: Literal["sim", "measured"] = "sim", stroke_s: float | None = None,
             gap: Literal["scene", "model"] | None = None, floor: Literal["bench", "training"] = "bench",
             jitter: bool = False, out: Path | None = None,
             on_episode: Callable[[int, EpisodeReport, bool], None] | None = None,
             rig: Rig | None = None, clock: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] | None = None) -> EvalResult:
    """Runs `episodes` sim episodes on a connected sim twin and scores them by training's condition.

    `jaw="measured"` slews the driver's jaw command at `100 / stroke_s`; `stroke_s`
    defaults to the stroke recorded in `rig` (the per-rig file), else `STREAM_STROKE_S`. `gap`
    defaults to `"scene"` at the sim jaw and `"model"` at the measured one. `out` is the result
    file, rewritten after every episode; the episode logs go to `episode_log_dir(out)`.
    `on_episode(k, report, success)` gets k from 0. Raises `EvalRefused` when nothing can run.
    `clock` and `sleep` go to the driver and the frame source: a twin connected with
    `threaded=False` passes its own clock and a `sleep` that steps it (the driver's clock rule).
    """
    if jaw not in ("sim", "measured"):
        raise ValueError(f"jaw must be 'sim' or 'measured', not {jaw!r}")
    if floor not in ("bench", "training"):
        raise ValueError(f"floor must be 'bench' or 'training', not {floor!r}")
    if not twin.connected or not twin.status.simulated:
        raise EvalRefused("not simulated", f"backend {twin.status.backend!r}")
    gap = gap or ("scene" if jaw == "sim" else "model")
    if gap not in ("scene", "model"):
        raise ValueError(f"gap must be 'scene' or 'model', not {gap!r}")
    stroke_from = None
    if jaw == "measured":
        if stroke_s is not None:
            stroke_from = "argument"
        elif rig is not None and rig.stroke_s() is not None:
            stroke_s, stroke_from = rig.stroke_s(), f"per-rig file {rig.path}"
        else:
            stroke_s, stroke_from = STREAM_STROKE_S, "default STREAM_STROKE_S (nothing measured)"
        if not (stroke_s is not None and math.isfinite(stroke_s) and stroke_s > 0.0):
            raise ValueError(f"stroke_s must be a positive number, not {stroke_s!r}")
    else:
        stroke_s = None

    rng = np.random.default_rng(seed)
    jitter_rng = np.random.default_rng([seed, 1])       # the cubes do not depend on the jitter setting
    config = {"jaw": jaw, "stroke_s": stroke_s, "stroke_from": stroke_from, "gap": gap, "floor": floor,
              "jitter": bool(jitter), "seed": int(seed), "episodes_asked": int(episodes),
              "checkpoint": str(getattr(policy, "path", "")) or None,
              "checkpoint_sha256": _checkpoint_sha(policy), "backend": "sim"}
    log_dir = episode_log_dir(Path(out)) if out is not None else None

    twin.configure(**sim_extras())
    kin = RobotKinematics(twin.spec)
    frames = _EpisodeFrames(twin, clock)
    # The first render builds the hand camera's renderer (about 0.1 s): not inside a tick.
    frames.latest()
    ticks: list[TickRecord] = []
    lost: dict[str, Any] = {"start": None, "sent": False}
    drv: driver_mod.GraspDriver | None = None

    def on_tick(rec: TickRecord) -> None:
        ticks.append(rec)
        start = lost["start"]
        if start is None or lost["sent"] or drv is None:
            return
        centre, _ = cube_pose(twin)
        if math.hypot(centre[0] - start[0], centre[1] - start[1]) > CUBE_LOST_XY:
            lost["sent"] = True
            drv.stop("cube lost")

    try:
        drv = driver_mod.GraspDriver(
            twin, policy, frames, WristCamera(), Rig.for_sim(), mode="sim", floor=floor, gap=gap,
            jaw_stroke_s=stroke_s, camera_latency_s=0.0, log_dir=log_dir, on_tick=on_tick,
            clock=clock, sleep=sleep)
    except BaseException:
        frames.close()
        raise
    reports: list[tuple[EpisodeReport, bool]] = []
    per_episode: list[dict] = []
    refused_in_row = 0
    try:
        why = drv.check()
        if why is not None:
            raise EvalRefused(why, getattr(drv, "detail", "") or "driver check")
        why = drv.take()
        if why is not None:
            raise EvalRefused(why, getattr(drv, "detail", "") or "driver take")
        for k in range(int(episodes)):
            draw = draw_hover(kin, rng, limits=twin.joint_limits())
            frames.use(_draw_jitter(jitter_rng) if jitter else None)
            info: dict[str, Any] = {"episode": k + 1, "cube_xy": draw.xy, "cube_yaw": draw.yaw,
                                    "hover": draw.hover.p, "hover_yaw": draw.hover.yaw,
                                    "believed": draw.hover.believed, "redraws": draw.redraws}
            lost["start"], lost["sent"] = None, False
            why = drv.preposition(draw.hover)
            refused_in_row = refused_in_row + 1 if why is not None else 0
            if why is not None:
                detail = getattr(drv, "detail", "")
                rep = refused_report(why, f"pre-positioning: {detail}" if detail else "pre-positioning")
                ok = False
            else:
                centre = place_cube(twin, draw.xy, draw.yaw)
                info["cube_placed"] = centre
                lost["start"] = centre[:2].copy()
                drv.success = scene_success(twin, float(centre[2]))
                rep = drv.episode(draw.hover)
                ok = rep.ending == "success"
                end_centre, _ = cube_pose(twin)
                info["cube_end"] = end_centre
                info["cube_rise"] = float(end_centre[2] - centre[2])
            info.update(ending=rep.ending, detail=rep.detail, success=ok, ticks=rep.ticks,
                        attempts=rep.attempts, ik_refused=rep.ik_refused, jump_refused=rep.jump_refused,
                        log=rep.log)
            reports.append((rep, ok))
            per_episode.append(info)
            if on_episode is not None:
                on_episode(k, rep, ok)
            res = _result(len(reports), reports, ticks, config, per_episode)
            if out is not None:
                write_result(res, Path(out))
            if why in HARD_REFUSALS or rep.ending in HARD_REFUSALS:
                logger.warning("evaluation stops after episode %d: %s", k + 1, rep.ending)
                break
            if refused_in_row >= PREPOSITION_STUCK:
                logger.warning("evaluation stops after episode %d: pre-positioning refused %d times in a row "
                               "(%s)", k + 1, refused_in_row, rep.detail)
                break
    except KeyboardInterrupt:
        motion.stop_now(twin, "Ctrl-C (ewaluacja chwytu)")
        raise
    finally:
        drv.close()
        frames.close()
    res = _result(len(reports), reports, ticks, config, per_episode)
    if out is not None:
        write_result(res, Path(out))
    return res


# ---------------------------------------------------------------------- command
def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _refuse(code: str, detail: str = "") -> int:
    print(f"Odmowa: {code}" + (f" - {detail}" if detail else ""), file=sys.stderr, flush=True)
    return 2


def _fmt_ms(stats: dict[str, float]) -> str:
    return ", ".join(f"{k} {v:.1f}" for k, v in stats.items()) + " ms" if stats else "brak danych"


def main_eval(a: argparse.Namespace) -> int:
    """`lerobot-twin grasp-eval`: 0 when the run completed, 1 when it could not run, 2 on a refusal."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    episodes = getattr(a, "episodes", None)
    episodes = 50 if episodes is None else int(episodes)
    seed = getattr(a, "seed", None)
    seed = 50000 if seed is None else int(seed)
    jaw = getattr(a, "jaw", None) or "sim"
    floor = getattr(a, "floor", None) or "bench"
    stroke_arg = getattr(a, "stroke_s", None)
    if jaw == "sim" and stroke_arg is not None:
        print("--stroke-s dziala tylko z --jaw measured (szczeka sim jedzie z predkoscia sceny).",
              file=sys.stderr)
        return 2
    if stroke_arg is not None and not (math.isfinite(stroke_arg) and stroke_arg > 0.0):
        print(f"--stroke-s musi byc dodatnia liczba sekund, jest {stroke_arg}.", file=sys.stderr)
        return 2

    path = policy_mod.find_checkpoint(getattr(a, "checkpoint", None))
    if path is None:
        print(f"Brak modelu. Pobierz: lerobot-twin grasp-model fetch  (albo --checkpoint PATH, "
              f"{policy_mod.CHECKPOINT_ENV})", file=sys.stderr)
        return 2
    try:
        policy = policy_mod.GraspStudentPolicy.load(path)
    except policy_mod.CheckpointError as exc:
        print(f"Model odrzucony: {exc}", file=sys.stderr)
        return 2
    ws = motion.load_workspace(getattr(a, "workspace", None))
    if ws is None:
        return _refuse("not a1x", NOT_A1X)
    try:
        rig = Rig.load(getattr(a, "rig", None))
    except RigError as exc:
        print(f"Plik stanowiska chwytu odrzucony: {exc}", file=sys.stderr)
        return 2

    out = Path(a.out) if getattr(a, "out", None) else data_path(OUT_DIR / f"{_stamp()}.json")
    twin = Twin(ws)
    try:
        # The literal "sim": the workspace backend on the rig is the real arm.
        twin.connect("sim", go_home=False, threaded=True)
    except Exception as exc:
        print(f"Nie polaczono z ramieniem symulowanym: {exc}", file=sys.stderr)
        twin.close()
        return 1
    if not twin.status.simulated:
        twin.close()
        return _refuse("not simulated", f"backend {twin.status.backend}")

    print(f"Ewaluacja w symulacji: {episodes} epizodow, ziarno {seed}, szczeka {jaw}, podloga {floor}, "
          f"model {path}", flush=True)

    def show(k: int, rep: EpisodeReport, ok: bool) -> None:
        print(episode_line(k, episodes, rep, success=ok), flush=True)

    code = 0
    res: EvalResult | None = None
    try:
        res = evaluate(twin, policy, episodes=episodes, seed=seed, jaw=jaw, stroke_s=stroke_arg,
                       gap=getattr(a, "gap", None), floor=floor, jitter=bool(getattr(a, "jitter", False)),
                       out=out, on_episode=show, rig=rig)
    except EvalRefused as exc:
        code = _refuse(exc.reason, exc.detail)
    except HoverError as exc:
        print(f"Nie wylosowano punktu nad kostka: {exc}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        print("\nCtrl-C: STOP, ewaluacja przerwana.", file=sys.stderr, flush=True)
        code = 1
    finally:
        twin.close()
    if res is None:
        if out.is_file():
            print(f"Wynik czesciowy: {out}")
        return code

    cfg = res.config
    print(f"\nWynik: {res.successes}/{res.episodes} udanych ({100.0 * res.rate:.0f} %), szczeka {cfg['jaw']}"
          + (f" (skok {cfg['stroke_s']} s: {cfg['stroke_from']})" if cfg.get("stroke_s") else "")
          + f", szczelina {cfg['gap']}, podloga {cfg['floor']}, "
            f"drgania kamery {'tak' if cfg['jitter'] else 'nie'}")
    print("Zakonczenia: " + ", ".join(f"{k} {v}" for k, v in res.endings.items()))
    print(f"Tik: {_fmt_ms(res.tick_ms)}; inferencja: {_fmt_ms(res.infer_ms)}")
    print(f"Ograniczenie predkosci {100.0 * res.rate_limited_frac:.1f} % tikow, "
          f"IK odrzucone {100.0 * res.ik_refused_frac:.1f} % "
          f"(w tym skok ponad {IK_MAX_JUMP_DEG:.0f} st. {100.0 * res.jump_refused_frac:.1f} %), "
          f"kolizje odrzucone {100.0 * res.collision_refused_frac:.1f} %")
    print(f"Zapisano: {out}")
    for line in verdicts(res):
        print(line)
    if res.episodes < episodes:
        print(f"Przebieg niepelny: {res.episodes} z {episodes} epizodow.", file=sys.stderr)
        return 1
    return code
