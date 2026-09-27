"""`lerobot-twin grasp`: the grasp student on the A1X, in four modes.

    lerobot-twin grasp                                        # dry: sim backend, nothing moves
    lerobot-twin grasp --mode sim --episodes 5                # the simulated arm grasps a simulated cube
    lerobot-twin grasp --mode dry --backend a1x               # dry on the real arm: what the policy sees
    lerobot-twin grasp --mode shadow --backend a1x --i-know   # the policy moves the arm, jaw open
    lerobot-twin grasp --mode real --backend a1x --i-know     # the first real grasp

* `dry` never claims, never engages and never writes a target. Each tick is a tick-0 observation
  at the hover; the step the policy would take is computed and logged, and every few ticks the
  model input is dumped with the believed pixel and the projected cube top drawn.
* `sim` runs only on the simulated backend: the cube is placed in the scene before every episode
  and success is read from the scene.
* `shadow` moves the arm as the policy says, with the jaw held open and the floor above the object.
* `real` is everything live.

Shadow and real need `--backend` and `--i-know`; dry and sim take the sim backend unless told
otherwise, never the workspace's. Once motion has started a bare Enter is always STOP (`keys.Keys`);
going on to the next episode needs `d` and Enter. Every STOP goes through `motion.stop_now`, which
also stops the jaw where it is. The workspace file is never written, the checkpoint is never
fetched, and the only camera opened is the wrist camera.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ...paths import data_path
from .contract import (
    BOX_XY,
    CALIBRATION_JOINT_OFFSETS_DEG,
    FRAME_AGE_MAX_S,
    JOINT_OFFSET_TOL_DEG,
    OBJECT_HEIGHT_DEFAULT,
    TICK_S,
    TIP_AHEAD,
)
from .keys import Keys
from .rig import MODES, OBJECT_HEIGHT_RANGE, Rig, RigError

if TYPE_CHECKING:
    from ..kinematics import RobotKinematics
    from ..runtime import Twin
    from ..workspace import Workspace
    from .camera import FrameSource, WristCamera
    from .driver import DryInputs, EpisodeReport, GraspDriver, TickRecord
    from .hover import Hover
    from .policy import GraspStudentPolicy

logger = logging.getLogger(__name__)

OUT_ROOT = Path("workspace") / "grasp"
FIRST_FRAME_S = 3.0                    # s, the wrist camera must deliver a frame within this
POLL_S = 0.05                          # s, the main thread watches the keys and the link this often
JOIN_S = 5.0                           # s, longest wait for a worker after a STOP
FREEZE_TOL = 1e-6
DUMP_EVERY_DEFAULT = 20
INFER_MEDIAN_MAX_MS = 30.0
INFER_P95_MAX_MS = 45.0
#: Pre-positioning refusals of the arm's own checks (exit 2); anything else that ends a move is a failure.
MOVE_REFUSALS = ("ik", "collision", "not arrived", "j5 offset", "speed limited")

NOT_A1X = "polecenia chwytu dzialaja tylko na ramieniu A1X (robot w pliku stanowiska)"
RIG_REJECTED = "Plik wartosci stanowiska chwytu odrzucony"
STOP_KEYS = "STOP z klawiatury (chwyt)"
STOP_CTRL_C = "Ctrl-C (chwyt)"
CONFIRM_PROMPT = "Enter = ramie jedzie nad punkt chwytu (20 st./s); Ctrl-C = koniec. "
PROCEED_PROMPT = ("d + Enter = kolejny epizod (szczeki otworza sie nad punktem - zabierz kostke); "
                  "sam Enter = STOP. ")

REAL_WARNING = """\
UWAGA: chwyt rusza PRAWDZIWYM ramieniem (tryb {mode}: {mode_note}).
  - Brak hamulcow: odciecie zasilania zrzuca ramie. STOP trzyma poze strumieniem,
    a szczeki staja tam, gdzie sa.
  - Wlaczenie sprzegla sprowadza kazdy staw stojacy poza limitami z powrotem w limity - to tez jest ruch.
  - Stol nauczony {table_age} temu, punkt chwytu {grasp_age} temu. Stol chroni tylko nauczony dotyk,
    ocena ugiecia i regula zmierzonej wysokosci - zadna z nich nie widzi stolu.
  - Progi wysilku sa zgadywane (guard.effort_abs). Najpierw --mode dry, potem shadow, dopiero real.
  - Rece z dala od ramienia. Sam Enter ZAWSZE zatrzymuje (STOP), Ctrl-C tez.
    Kolejny epizod tylko po wpisaniu d i Enter, nie wczesniej niz sekunde po koncu epizodu."""

_MODE_NOTES = {
    "shadow": "szczeki zostaja otwarte, podloga nad obiektem",
    "real": "szczeki zamykaja sie naprawde",
}


@dataclass
class Session:
    """Everything a run mode holds between connecting and closing."""

    twin: Twin
    driver: GraspDriver
    policy: GraspStudentPolicy
    frames: FrameSource
    rig: Rig
    mode: str
    camera: WristCamera
    hover: Hover | None = None
    ws: Workspace | None = None
    kin: RobotKinematics | None = None               # the main thread's own
    simulated: bool = True
    rng: np.random.Generator | None = None           # simulated backend: draws the cubes
    cube: tuple[np.ndarray, float] | None = None     # simulated backend: the cube under `hover`
    stopped: bool = False                            # `motion.stop_now` was called
    worker: threading.Thread | None = None


def out_dir(kind: str) -> Path:
    """`workspace/grasp/<kind>/<stamp>/`; tests point it elsewhere."""
    return data_path(OUT_ROOT / kind / datetime.now().strftime("%Y%m%d-%H%M%S"))


def refuse(code: str, detail: str = "") -> int:
    print(f"Odmowa: {code}" + (f" - {detail}" if detail else ""), file=sys.stderr, flush=True)
    return 2


def jsonable(value: Any) -> Any:
    """`value` with numpy values, paths and non-finite floats turned into what JSON takes."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def load_site(a: argparse.Namespace) -> tuple[Workspace, Rig] | int:
    """The workspace (an A1X one) and the per-rig file; 2, after the message, when either is refused."""
    from . import motion

    try:
        ws = motion.load_workspace(getattr(a, "workspace", None))
    except (OSError, ValueError, TypeError) as exc:
        print(f"Nie wczytano pliku stanowiska: {exc}", file=sys.stderr)
        return 2
    if ws is None:
        return refuse("not a1x", NOT_A1X)
    try:
        rig = Rig.load(getattr(a, "rig", None))
    except RigError as exc:
        print(f"{RIG_REJECTED}: {exc}", file=sys.stderr)
        return 2
    return ws, rig


def first_frame(frames: FrameSource, clock: Callable[[], float], wait: Callable[[float], bool]) -> bool:
    """Waits up to `FIRST_FRAME_S` for the wrist camera's first frame; `wait(dt)` returns True to give up.

    On False the reason is printed already.
    """
    from .camera import CameraError

    t_end = clock() + FIRST_FRAME_S
    while clock() < t_end:
        try:
            if frames.latest() is not None:
                return True
        except CameraError as exc:
            print(f"Kamera na nadgarstku: {exc}", file=sys.stderr)
            return False
        if wait(POLL_S):
            return False
    print(f"Blad: wrist camera silent - brak klatki z kamery na nadgarstku przez {FIRST_FRAME_S:.0f} s",
          file=sys.stderr)
    return False


def backend_offsets(twin: Twin) -> np.ndarray:
    """The running backend's joint zero offsets [deg]; a backend without them applies none."""
    offs = getattr(getattr(twin.backend, "acfg", None), "joint_offsets_deg", None)
    return np.zeros(6) if offs is None else np.asarray(offs, float).reshape(6)


def _age(seconds: float | None) -> str:
    if seconds is None:
        return "(nieznany czas)"
    minutes = max(0.0, seconds) / 60.0
    if minutes < 60.0:
        return f"{minutes:.0f} min"
    return f"{int(minutes // 60)} h {int(minutes % 60)} min"


# ---------------------------------------------------------------------- checks with the workspace and backend
def _table_holds(ws: Workspace, xy: np.ndarray) -> bool:
    """The workspace table contains `xy` (base frame) with `BOX_XY` to spare on every side."""
    t = ws.table_obj()
    c, s = math.cos(t.base_yaw), math.sin(t.base_yaw)
    wx = t.base_xy[0] + c * float(xy[0]) - s * float(xy[1])
    wy = t.base_xy[1] + s * float(xy[0]) + c * float(xy[1])
    return abs(wx) <= t.size[0] / 2.0 - BOX_XY and abs(wy) <= t.size[1] / 2.0 - BOX_XY


def rig_refusals(rig: Rig, mode: str, simulated: bool, *, camera: str | None = None,
                 now: float | None = None) -> list[str]:
    """`rig.missing`, with `--camera` standing in for a wrist camera source the per-rig file lacks."""
    missing = rig.missing(mode, simulated, now=now)
    if camera:
        missing = [m for m in missing if m != "wrist camera not set"]
    return missing


def site_refusals(ws: Workspace, rig: Rig, twin: Twin, mode: str) -> list[str]:
    """Refusals that need the workspace or the running backend; always empty on a simulated backend."""
    if twin.status.simulated:
        return []
    out: list[str] = []
    believed = rig.believed()
    if believed is not None and not _table_holds(ws, believed[:2]):
        out.append("table not under grasp")
    offs = backend_offsets(twin)
    taught = rig.grasp.get("joint_offsets_deg")
    if taught is None or float(np.max(np.abs(offs - np.asarray(taught, float)))) > JOINT_OFFSET_TOL_DEG:
        out.append("joint offsets changed")
    differ = float(np.max(np.abs(offs - np.asarray(CALIBRATION_JOINT_OFFSETS_DEG, float))))
    if differ > JOINT_OFFSET_TOL_DEG:
        if rig.overrides.get("joint_offsets"):
            bar = "!" * 72
            print(f"{bar}\nUWAGA: przesuniecia zera stawow backendu {offs.round(3).tolist()} st.\n"
                  f"roznia sie od tych, przy ktorych kalibrowano kamere na nadgarstku (same zera).\n"
                  f"Nadpisane: overrides.joint_offsets = {rig.overrides.get('joint_offsets')!r}.\n"
                  f"Kropka punktu nauczonego dotykiem przesunie sie o mniej niz 0,3 px,\n"
                  f"punktu z innego zrodla o okolo 4-5 px.\n{bar}", file=sys.stderr, flush=True)
        else:
            out.append("joint offsets differ")
    if mode == "real":
        b = twin.backend
        gs = b.gripper_state() if b is not None else None
        thr = rig.freeze_threshold()
        if (gs is None or gs.force_threshold is None or thr is None
                or abs(float(gs.force_threshold) - thr) > FREEZE_TOL):
            out.append("gripper freeze differs")
    return out


# ---------------------------------------------------------------------- opening and closing
def _shutdown(twin: Twin | None, driver: GraspDriver | None, frames: FrameSource | None, *,
              stopped: bool, worker: threading.Thread | None = None) -> None:
    """Driver, camera, hold unless a STOP holds already, disconnect. Never raises."""
    from ...control.safety import SafetyState

    if worker is not None and worker.is_alive():
        if driver is not None:
            driver.stop("stop")
        worker.join(JOIN_S)
        if worker.is_alive():
            logger.error("grasp worker still running %.0f s after the end", JOIN_S)
    for what, fn in (("driver", getattr(driver, "close", None)), ("camera", getattr(frames, "close", None))):
        if fn is None:
            continue
        try:
            fn()
        except Exception:
            logger.exception("closing the %s failed", what)
    if twin is None:
        return
    try:
        if twin.connected and not stopped and twin.safety_state != SafetyState.ESTOP:
            twin.hold_measured()
    except Exception:
        logger.exception("hold after the run failed")
    try:
        twin.close()
    except Exception:
        logger.exception("closing the twin failed")


def _stop(s: Session, reason: str) -> None:
    """The one STOP of the run mode: `motion.stop_now` (STOP, then the jaw held where it is)."""
    from . import motion

    if s.stopped:
        return
    s.stopped = True
    if s.twin.connected:
        motion.stop_now(s.twin, reason)
    s.driver.stop("stop")


def _reach_filter(ws: Workspace | None) -> Callable[[np.ndarray], bool] | None:
    """The workspace's reach volume as a test of a hover position; None when the workspace sets none.

    Asked about the hover only, never per tick: its 5 cm floor would reject the grasp height.
    """
    if ws is None or not ws.reach:
        return None
    from ..reach import ReachVolume

    vol = ReachVolume.from_dict(ws.reach)
    return lambda p: vol.contains(p)


def _sim_hover(s: Session) -> None:
    """Draws the next cube and its hover on the simulated backend."""
    from . import evaluate

    assert s.kin is not None and s.rng is not None
    draw = evaluate.draw_hover(s.kin, s.rng, limits=s.twin.joint_limits(), accept=_reach_filter(s.ws))
    s.hover, s.cube = draw.hover, (draw.xy, draw.yaw)


def open_session(a: argparse.Namespace, keys: Keys) -> Session | int:
    """Model, workspace, connect, refusals, camera, hover and the driver's check.

    Returns the session, or the exit code of a refusal or failure with everything already
    closed. On an exception (Ctrl-C during connect included) it stops, closes and re-raises.
    """
    parts: dict[str, Any] = {}
    try:
        got = _open(a, keys, parts)
    except BaseException as exc:
        twin = parts.get("twin")
        stopped = False
        if isinstance(exc, KeyboardInterrupt) and twin is not None and twin.connected:
            from . import motion

            motion.stop_now(twin, STOP_CTRL_C)
            stopped = True
        _shutdown(twin, parts.get("driver"), parts.get("frames"), stopped=stopped)
        raise
    if isinstance(got, int):
        _shutdown(parts.get("twin"), parts.get("driver"), parts.get("frames"), stopped=False)
    return got


def _open(a: argparse.Namespace, keys: Keys, parts: dict[str, Any]) -> Session | int:
    from ..kinematics import RobotKinematics
    from ..runtime import Twin
    from . import driver as driver_mod
    from . import evaluate, motion
    from . import policy as policy_mod
    from .camera import CameraError, LiveWristSource, SimWristSource, WristCamera, wrist_intrinsics
    from .hover import HoverError, from_rig

    mode = a.mode
    path = policy_mod._find_for_cli(a)
    policy = policy_mod._load_for_cli(path) if path is not None else None
    if policy is None:
        return 2
    site = load_site(a)
    if isinstance(site, int):
        return site
    ws, rig = site
    # Dry and sim never take the backend from the workspace file: on the rig that is the real arm.
    backend = str(a.backend or "sim").lower()
    port = getattr(a, "port", None)
    if backend != "sim":
        if mode == "sim":
            return refuse("not armed", f"tryb sim na prawdziwym backendzie {backend!r}")
        why = motion.enable_refusal(ws, backend, port)
        if why is not None:
            return refuse(why, "a1x.enable_on_connect wlacza silniki przy laczeniu (ramie bez hamulcow "
                               "na chwile puszcza) - wlacz ramie z panelu i ustaw enable_on_connect: false")

    twin = Twin(ws)
    parts["twin"] = twin
    try:
        twin.connect(backend, port=port, go_home=False, threaded=True)
    except Exception as exc:
        print(f"Nie polaczono z ramieniem: {exc}", file=sys.stderr)
        return 1
    simulated = bool(twin.status.simulated)
    print(f"Ramie: {twin.status.backend} ({'symulacja' if simulated else 'prawdziwe'}), tryb {mode}",
          flush=True)
    if mode == "sim" and not simulated:
        return refuse("not armed", "tryb sim tylko na ramieniu symulowanym")
    missing = rig_refusals(rig, mode, simulated, camera=getattr(a, "camera", None), now=time.time())
    if missing:
        return refuse(", ".join(missing), f"plik {rig.path}: lerobot-twin grasp-bench show")
    site = site_refusals(ws, rig, twin, mode)
    if site:
        return refuse(", ".join(site))

    if simulated:
        twin.configure(**evaluate.sim_extras())
        rig = Rig.for_sim()
        try:
            camera = WristCamera()
            frames = SimWristSource(twin)
        except CameraError as exc:
            print(f"Kamera na nadgarstku (symulacja): {exc}", file=sys.stderr)
            return 1
    else:
        from .. import cameras

        source = rig.wrist_source() or getattr(a, "camera", None)
        try:
            # K and dist of the per-rig file are judged here: a lens model that folds or looks
            # past the sensor is refused before the camera opens.
            camera = WristCamera(wrist_intrinsics(rig))
            # The opener is looked up here, not bound at import, so it can be replaced.
            frames = LiveWristSource(str(source), camera, opener=cameras.open_live)
        except CameraError as exc:
            print(f"Kamera na nadgarstku: {exc}", file=sys.stderr)
            return 1
    parts["frames"] = frames
    if not first_frame(frames, time.monotonic, lambda dt: time.sleep(dt) or False):
        return 1

    kin = RobotKinematics(twin.spec)
    seed = getattr(a, "seed", None)
    if seed is None:
        seed = int(time.time()) % 100000
    s_rng = np.random.default_rng(int(seed)) if simulated else None
    hover: Hover
    cube = None
    try:
        if simulated:
            draw = evaluate.draw_hover(kin, s_rng, limits=twin.joint_limits(), accept=_reach_filter(ws))
            hover, cube = draw.hover, (draw.xy, draw.yaw)
            print(f"Symulacja: ziarno {seed}, kostka ({draw.xy[0]:.3f}, {draw.xy[1]:.3f}) m, "
                  f"obrot {math.degrees(draw.yaw):.1f} st.", flush=True)
        else:
            hover = from_rig(kin, rig, limits=twin.joint_limits())
    except HoverError as exc:
        return refuse(exc.reason, exc.detail)
    if hover.outside_training:
        if not getattr(a, "outside_training", False):
            return refuse("outside training", "; ".join(hover.outside_training)
                           + " (--outside-training nadpisuje)")
        print(f"UWAGA: nad punktem poza obszarem treningu ({'; '.join(hover.outside_training)}) - "
              f"--outside-training.", file=sys.stderr, flush=True)
    inside = _reach_filter(ws)
    if inside is not None and not inside(hover.p):
        return refuse("outside volume",
                      f"nad punktem {np.round(hover.p, 3).tolist()} poza pudelkiem zasiegu")
    print(f"Nad punktem: {np.round(hover.p, 4).tolist()} m, obrot {math.degrees(hover.yaw):.1f} st., "
          f"zapas J6 {hover.j6_margin_deg:.0f} st.", flush=True)

    jaw_stroke_s = getattr(a, "jaw_stroke_s", None)
    gap = getattr(a, "gap", None) or ("scene" if mode == "sim" and jaw_stroke_s is None else "model")
    log_dir = None if mode == "dry" else out_dir("runs")
    drv = driver_mod.GraspDriver(
        twin, policy, frames, camera, rig, mode=mode, real_ok=mode in ("shadow", "real") and bool(a.i_know),
        floor=getattr(a, "floor", None) or "bench", gap=gap, jaw_stroke_s=jaw_stroke_s, log_dir=log_dir)
    parts["driver"] = drv
    why = drv.check()
    if why is not None:
        return refuse(why, getattr(drv, "detail", ""))
    return Session(twin=twin, driver=drv, policy=policy, frames=frames, rig=rig, mode=mode, camera=camera,
                   hover=hover, ws=ws, kin=kin, simulated=simulated, rng=s_rng, cube=cube)


# ---------------------------------------------------------------------- episodes
def _watch(s: Session, keys: Keys, fn: Callable[[], Any], what: str) -> Any:
    """Runs `fn` on a worker thread; the calling thread watches the keys and the link meanwhile."""
    box: dict[str, Any] = {}
    done = threading.Event()

    def work() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:                     # handed to the watching thread
            box["error"] = exc
        finally:
            done.set()

    th = threading.Thread(target=work, name=f"chwyt-{what}", daemon=True)
    s.worker = th
    th.start()
    lost = False
    while True:
        if keys.stop.is_set():
            _stop(s, STOP_KEYS)
        elif not lost and not s.twin.connected:
            lost = True
            s.driver.stop("not connected")
        if done.wait(POLL_S):
            break
    s.worker = None
    if "error" in box:
        raise box["error"]
    return box.get("value")


def run_series(s: Session, keys: Keys, episodes: int) -> int:
    """Consent on a real backend, take the arm, then episode after episode; returns the exit code."""
    from . import evaluate, motion
    from .hover import HoverError

    twin, drv = s.twin, s.driver
    if not s.simulated:
        ages = s.rig.taught_age_s(time.time())
        print(REAL_WARNING.format(mode=s.mode, mode_note=_MODE_NOTES.get(s.mode, s.mode),
                                  table_age=_age(ages.get("table")), grasp_age=_age(ages.get("grasp"))),
              flush=True)
        if not keys.interactive:
            print("Prawdziwe ramie tylko z terminala (TTY): Enter musi moc zatrzymac ramie.", file=sys.stderr)
            return 2
        if not keys.confirm(CONFIRM_PROMPT):
            print("Bez potwierdzenia - nic nie ruszylo.", file=sys.stderr)
            return 2
    keys.arm()
    why = drv.take()
    if why is not None:
        return refuse(why, getattr(drv, "detail", ""))
    code = 0
    refused_in_row = 0
    for k in range(episodes):
        if keys.stop.is_set():
            _stop(s, STOP_KEYS)
            break
        if k > 0 and s.simulated:
            try:
                _sim_hover(s)
            except HoverError as exc:
                print(f"Epizod {k + 1}/{episodes}: brak punktu nad kostka ({exc})", file=sys.stderr,
                      flush=True)
                code = 1
                break
        hover = s.hover
        assert hover is not None
        why = _watch(s, keys, lambda: drv.preposition(hover), "przed-epizodem")
        # A STOP while the arm moves ends with 1, as Ctrl-C does; only the one at the prompt
        # between episodes is the normal end of a series.
        if keys.stop.is_set() or s.stopped:
            _stop(s, STOP_KEYS)
            code = 1
            break
        if why is not None:
            keys.note_end()
            refused_in_row += 1
            detail = getattr(drv, "detail", "")
            print(f"Epizod {k + 1}/{episodes}: odmowa przed epizodem: {why}"
                  + (f" ({detail})" if detail else ""), file=sys.stderr, flush=True)
            if not s.simulated:
                code = 2 if why in MOVE_REFUSALS else 1
                break
            if why in evaluate.HARD_REFUSALS or refused_in_row >= evaluate.PREPOSITION_STUCK:
                code = 1
                break
            continue
        refused_in_row = 0
        if s.simulated and s.cube is not None:
            centre = evaluate.place_cube(twin, *s.cube)
            if s.mode != "shadow":
                drv.success = evaluate.scene_success(twin, float(centre[2]))
        if keys.stop.is_set():
            _stop(s, STOP_KEYS)
            code = 1
            break
        rep: EpisodeReport = _watch(s, keys, lambda: drv.episode(hover), "epizod")
        keys.note_end()
        print(evaluate.episode_line(k, episodes, rep, shadow=s.mode == "shadow"), flush=True)
        if keys.stop.is_set() or s.stopped:
            _stop(s, STOP_KEYS)
            code = 1
            break
        if not twin.connected:
            print(f"Ramie rozlaczone: {twin.status.error}", file=sys.stderr)
            code = 1
            break
        if not s.simulated and rep.ending not in motion.KEEP_GRIP:
            print(f"Koniec serii: {rep.ending} na prawdziwym ramieniu.", file=sys.stderr, flush=True)
            code = 1
            break
        if s.simulated and rep.ending in evaluate.HARD_REFUSALS:
            code = 1
            break
        if k + 1 < episodes and not s.simulated:
            if not keys.proceed(PROCEED_PROMPT):
                _stop(s, STOP_KEYS)
                break
    return code


# ---------------------------------------------------------------------- dry run
def _dump(out: Path, n: int, rec: TickRecord, inp: DryInputs, camera: WristCamera, kin: RobotKinematics,
          believed: np.ndarray, top_z: float) -> None:
    import cv2

    from .camera import draw_dump

    if inp.frame is None:
        return
    joints = inp.pair_joints
    if rec.uv is not None:
        uv = np.asarray(rec.uv, np.float32)
    elif joints is not None:
        uv = camera.believed_uv(kin, joints, believed)
    else:
        uv = np.array([np.nan, np.nan], np.float32)
    extra = {}
    if joints is not None:
        extra["cube top"] = camera.cube_top_uv(kin, joints, believed[:2], top_z)
    img = draw_dump(inp.frame.image, uv, extra=extra)
    cv2.imwrite(str(out / f"{n:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if inp.inputs is not None:
        heat = np.clip(np.asarray(inp.inputs[3], np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
        cv2.imwrite(str(out / f"{n:04d}_heat.png"), heat)


def _pcts(values: list[float | None], scale: float = 1.0) -> str:
    v = np.asarray([x * scale for x in values if x is not None and math.isfinite(x)], float)
    if v.size == 0:
        return "brak danych"
    return f"p50 {np.percentile(v, 50):.1f}  p95 {np.percentile(v, 95):.1f}  max {v.max():.1f}"


def _finite(values: list[float] | None) -> bool:
    return values is None or all(v is not None and math.isfinite(v) for v in values)


def dry_run(session: Session, keys: Keys, *, seconds: float | None, dump_every: int, out: Path,
            object_height: float) -> int:
    """The dry loop at 20 Hz: `driver.dry_tick` scores, this writes the dumps and `ticks.jsonl`.

    Ends on any line typed (Enter), on Ctrl-C (the caller's) or after `seconds`. Nothing moves:
    the driver never claims, engages or writes a target in dry mode.
    """
    from ..kinematics import RobotKinematics

    twin, drv, rig = session.twin, session.driver, session.rig
    hover = session.hover
    if hover is None:
        raise ValueError("dry run needs the session's hover")
    touch = rig.touch_tcp_z()
    if touch is None:
        raise ValueError("dry run needs the taught table touch")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if session.simulated and session.cube is not None:
        from . import evaluate

        # Only the simulated cube is put in place; the arm stays where it stands.
        evaluate.place_cube(twin, *session.cube)
    kin = RobotKinematics(twin.spec)
    believed = np.asarray(hover.believed, float)
    top_z = touch - TIP_AHEAD + float(object_height)
    every = int(dump_every)
    print(f"Tryb dry: ramie stoi, nic nie jest wysylane. Zrzuty co {every} tikow do {out}. "
          f"Enter albo Ctrl-C = koniec.", flush=True)

    recs: list[TickRecord] = []
    starts: list[float] = []
    code = 0
    t0 = time.monotonic()
    next_t = t0
    n = 0
    with open(out / "ticks.jsonl", "w", encoding="utf-8") as log:
        while True:
            now = time.monotonic()
            if seconds is not None and now - t0 >= seconds:
                break
            if keys.readline(timeout_s=0.0) is not None:
                break
            if not twin.connected:
                print(f"Ramie rozlaczone: {twin.status.error}", file=sys.stderr)
                code = 1
                break
            starts.append(now)
            rec, inp = drv.dry_tick(hover)
            recs.append(rec)
            log.write(json.dumps(jsonable(asdict(rec))) + "\n")
            if n % every == 0:
                _dump(out, n, rec, inp, session.camera, kin, believed, top_z)
            n += 1
            next_t += TICK_S
            delay = next_t - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            elif delay < -TICK_S:                        # more than a period late: resynchronise
                next_t = time.monotonic()

    # The driver logs the arm efforts once per second while it scores dry ticks.
    efforts = [e for _, e in drv.arm.efforts_seen]
    _dry_summary(session, recs, starts, efforts, time.monotonic() - t0, out)
    return code


def _dry_summary(s: Session, recs: list[TickRecord], starts: list[float], efforts: list[dict[str, float]],
                 elapsed: float, out: Path) -> None:
    gaps = np.diff(np.asarray(starts)) * 1000.0 if len(starts) > 1 else np.zeros(0)
    skips = Counter(r.skip for r in recs if r.skip)
    nonfinite = sum(1 for r in recs if r.skip == "non-finite" or not _finite(r.action))
    frame_age = [r.frame_age_s for r in recs]
    infer = [r.infer_ms for r in recs if r.infer_ms is not None]
    print(f"\nPodsumowanie dry: {len(recs)} tikow w {elapsed:.1f} s, zrzuty w {out}")
    print(f"  wiek klatki [ms]      {_pcts(frame_age, 1000.0)}")
    print(f"  wiek odczytu [ms]     {_pcts([r.fb_age_s for r in recs], 1000.0)}")
    print(f"  wiek stawow [ms]      {_pcts([r.joint_age_s for r in recs], 1000.0)}")
    print(f"  inferencja [ms]       {_pcts(infer)}")
    print(f"  czas tiku [ms]        {_pcts([r.tick_ms for r in recs])}")
    if gaps.size:
        late = int(np.sum(gaps > 1.5 * TICK_S * 1000.0))
        print(f"  przerwy petli [ms]    {_pcts(list(gaps))}; dluzsze niz 1,5 okresu: {late}")
    print(f"  akcje nieskonczone: {nonfinite}; tiki nieswieze (stale feedback): "
          f"{skips.get('stale feedback', 0)}; pominiete wg powodu: {dict(skips) or 'brak'}")
    if efforts:
        peak = {k: max(abs(e.get(k, 0.0)) for e in efforts) for k in efforts[0]}
        print(f"  wysilek ramienia (raz na sekunde, max |.|): "
              f"{ {k: round(v, 3) for k, v in peak.items()} } - guard.effort_abs ustaw powyzej")
    if s.simulated:
        print("  parowanie klatki: symulacja - znacznik klatki to chwila ekspozycji")
    elif s.rig.latency.get("camera_s") is not None:
        print(f"  parowanie klatki: zmierzone opoznienie kamery {s.rig.camera_latency_s() * 1000.0:.0f} ms")
    else:
        print("  parowanie klatki: opoznienie kamery NIEZMIERZONE - para po znaczniku klatki "
              "(lerobot-twin grasp-bench latency)")
    ages = np.asarray([a for a in frame_age if a is not None and math.isfinite(a)], float)
    if ages.size:
        p95 = float(np.percentile(ages, 95))
        print(f"  werdykt: wiek klatki p95 {p95 * 1000.0:.0f} ms - "
              f"{'OK' if p95 < FRAME_AGE_MAX_S else 'ZA STARE'} (limit {FRAME_AGE_MAX_S * 1000.0:.0f} ms)")
    if infer:
        med, p95i = float(np.median(infer)), float(np.percentile(infer, 95))
        ok = med < INFER_MEDIAN_MAX_MS and p95i < INFER_P95_MAX_MS
        print(f"  werdykt: inferencja mediana {med:.1f} ms, p95 {p95i:.1f} ms - {'OK' if ok else 'ZA WOLNO'} "
              f"(limity {INFER_MEDIAN_MAX_MS:.0f} / {INFER_P95_MAX_MS:.0f} ms)")


# ---------------------------------------------------------------------- command
def _flag_refusal(a: argparse.Namespace) -> str | None:
    mode = a.mode
    if mode not in MODES:
        return f"nieznany tryb {mode!r} (dry, sim, shadow, real)"
    if mode in ("shadow", "real"):
        if not getattr(a, "i_know", False):
            return f"lerobot-twin grasp --mode {mode} rusza prawdziwym ramieniem - dodaj --i-know."
        if not getattr(a, "backend", None):
            return f"lerobot-twin grasp --mode {mode} wymaga jawnego --backend (np. a1x, albo sim na probe)."
    if mode != "dry":
        if getattr(a, "seconds", None) is not None:
            return "--seconds tylko w --mode dry."
        if getattr(a, "object_height", None) is not None:
            return "--object-height tylko w --mode dry (w pozostalych trybach wysokosc z pliku stanowiska)."
        if getattr(a, "dump_every", DUMP_EVERY_DEFAULT) != DUMP_EVERY_DEFAULT:
            return "--dump-every tylko w --mode dry."
    if mode != "sim":
        if getattr(a, "floor", None) == "training":
            return "--floor training tylko w --mode sim (w treningu palce wchodza w stol)."
        if getattr(a, "jaw_stroke_s", None) is not None:
            return "--jaw-stroke-s tylko w --mode sim."
        if getattr(a, "gap", None) == "scene":
            return "--gap scene tylko w --mode sim (szczeline czyta sie ze sceny symulacji)."
    stroke = getattr(a, "jaw_stroke_s", None)
    if stroke is not None and not (math.isfinite(stroke) and stroke > 0.0):
        return f"--jaw-stroke-s musi byc dodatnia liczba sekund, jest {stroke}."
    height = getattr(a, "object_height", None)
    if height is not None and not OBJECT_HEIGHT_RANGE[0] <= height <= OBJECT_HEIGHT_RANGE[1]:
        return f"--object-height {height}: oczekiwano {OBJECT_HEIGHT_RANGE[0]}..{OBJECT_HEIGHT_RANGE[1]} m."
    for flag, value in (("--episodes", getattr(a, "episodes", 1)),
                        ("--dump-every", getattr(a, "dump_every", DUMP_EVERY_DEFAULT))):
        if value < 1:
            return f"{flag} musi byc >= 1, jest {value}."
    return None


def main_grasp(a: argparse.Namespace) -> int:
    """`lerobot-twin grasp`.

    Exit codes: 0 ok, 1 failure (connect, camera, arm lost, guard ending on a real arm, a STOP while
    the arm moves), 2 refusal.
    """
    session: Session | None = None
    code = 0
    try:
        why = _flag_refusal(a)
        if why is not None:
            print(why, file=sys.stderr)
            return 2
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        keys = Keys()
        keys.start()
        got = open_session(a, keys)
        if isinstance(got, int):
            return got
        session = got
        if session.mode == "dry":
            object_height = getattr(a, "object_height", None)
            if object_height is None:
                object_height = session.rig.object_height() or OBJECT_HEIGHT_DEFAULT
            code = dry_run(session, keys, seconds=getattr(a, "seconds", None),
                           dump_every=int(getattr(a, "dump_every", DUMP_EVERY_DEFAULT)), out=out_dir("dry"),
                           object_height=float(object_height))
        else:
            code = run_series(session, keys, int(getattr(a, "episodes", 1)))
    except KeyboardInterrupt:
        print("\nCtrl-C: STOP.", file=sys.stderr, flush=True)
        if session is not None:
            _stop(session, STOP_CTRL_C)
        code = 1
    finally:
        if session is not None:
            _shutdown(session.twin, session.driver, session.frames, stopped=session.stopped,
                      worker=session.worker)
        print("Koniec: ramie rozlaczone." if session is not None else "Koniec.", flush=True)
    return code
