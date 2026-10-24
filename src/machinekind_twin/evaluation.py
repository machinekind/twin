"""Many simulated runs of one strategy, each with a seeded object, judged by the scene: `twin eval-strategy`.

    plugin = load_strategy("touch")
    campaign = Campaign(plugin=plugin, settings=plugin_values(plugin, config_section("touch")),
                        rule="touch", label="cube", obj=ObjectSpec(), area=Area.from_desk(spec.desk),
                        seed=50000, episodes=50)
    facts = preflight(workspace, campaign)             # refuses what would fail in every episode
    result = run_campaign(workspace, campaign, folder, facts=facts, on_episode=print_row)

Every episode builds a fresh twin on a `VirtualClock`, places the object (`scene_truth`) and runs a
fresh instance of the strategy through the same `StrategyRunner` and `SafetySupervisor` as the
panel's Strategie tab. The twin's loop steps at its own rate and the strategy at its own, on one
simulated clock. The strategy sees the object in `Observation.objects`: its true pose
(`SimulationPoses`, source `symulacja`) or a detector's answer through `Perception`
(`DetectorPoses`). The scene judges the run (`scene_truth.judge`); the strategy's own claim (`done`
with or without success) is kept next to that verdict, never mixed into it.

The campaign folder gets one episode log per run (`NNN.jsonl`, written by the runner) and
`summary.json`, which is rewritten after every episode, so a campaign cut off leaves what it ran.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Protocol

import machinekind_twin_api
import mujoco
import numpy as np
from machinekind_twin_api.strategy import ObjectPose

from . import __version__ as twin_version
from .clock import VirtualClock
from .detection import Perception
from .detectors import DetectorPlugin
from .episode import EpisodeOptions, episodes_dir
from .paths import CONFIG_ENV
from .plugin_settings import PluginClass, SettingField
from .runtime import Twin
from .scene_truth import (
    RULES,
    SETTLE_S,
    SIM_SOURCE,
    Area,
    ObjectSpec,
    Placement,
    SceneSample,
    SimulationPoses,
    Verdict,
    Watch,
    check_jaws,
    check_on_table,
    configure_object,
    draw_placement,
    judge,
    place_object,
    robot_contacts,
    rule_constants,
    sample,
)
from .strategies import StrategyPlugin
from .strategy import StrategyRunner
from .ui.watch import arm_mask
from .workspace import Workspace

__all__ = [
    "FORMAT", "SUMMARY", "VERSION", "Campaign", "CampaignResult", "DetectorPoses", "EpisodeRow",
    "EvaluationAborted", "StrategyFacts", "campaign_folder", "drive", "effective_time_limit", "episode_name",
    "footer_lines", "header_lines", "object_label", "object_size_m", "plugin_values", "preflight", "row_line",
    "run_campaign", "run_episode", "setting_from_text", "settle", "strategy_cameras", "summary_data",
    "totals", "wilson", "write_summary",
]

#: The summary's `format`, and the version this twin writes.
FORMAT = "machinekind-twin-evaluation"
VERSION = 1
#: The summary's file in the campaign folder: not `.jsonl`, so `twin episode <folder>` lists only runs.
SUMMARY = "summary.json"
#: Slack when comparing times on the simulated clock [s].
EPS = 1e-9
#: How far past its time limit a run may go before the evaluation stops it [s]. Only a runner that
#: fails to apply its own limit gets there.
OVERRUN_S = 1.0
#: The object's label when neither the operator nor a plugin names one.
DEFAULT_LABEL = "cube"
#: The twin's loop rate [Hz], the panel's.
LOOP_HZ = 50.0
#: How far the arm's mask is widened for the detector's silhouette fit [px], as in the panel.
MASK_DILATE_PX = 5


class EvaluationAborted(RuntimeError):
    """The campaign cannot go on: an episode's arm did not start, which no strategy explains."""


# --------------------------------------------------------------------------- settings


def setting_from_text(setting: SettingField, text: str, where: str) -> Any:
    """A setting typed on the command line, as its field's type, or ValueError.

    The panel's settings are strict about types (`plugin_settings`): text becomes a number or a
    switch here first. An int takes a whole float too ("3.0"), a bool only `true` or `false`.
    """
    name = f"{where}.{setting.name}"
    kind = setting.kind

    if kind is str:
        value: Any = text
    elif kind is bool:
        lowered = text.strip().lower()
        if lowered not in ("true", "false"):
            raise ValueError(f"{name}: potrzeba bool (true albo false), jest {text!r}")
        value = lowered == "true"
    elif kind is int:
        value = _whole_number(text, name)
    else:
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{name}: potrzeba float, jest {text!r}") from None

    return setting.convert(value, where)


def _whole_number(text: str, name: str) -> int:
    try:
        return int(text)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError:
        number = math.nan
    if not number.is_integer():
        raise ValueError(f"{name}: potrzeba int, jest {text!r}")

    return int(number)


def plugin_values(plugin: PluginClass, section: Mapping[str, Any], assignments: Sequence[str] = ()) -> dict[str, Any]:
    """A plugin's settings as the panel builds them: the config section, `KEY=VALUE` texts over it,
    every default the two leave out. A broken section fails even when an assignment would override it.

    A missing required key fails only at `plugin.create`, as in the panel.
    """
    plugin.check(section)
    fields = {setting.name: setting for setting in plugin.fields}
    overrides: dict[str, Any] = {}

    for text in assignments:
        key, sep, value = text.partition("=")
        if not sep or not key:
            raise ValueError(f"--set {text}: potrzeba KLUCZ=WARTOSC")
        setting = fields.get(key)
        overrides[key] = value if setting is None else setting_from_text(setting, value, plugin.name)

    given = plugin.check({**section, **overrides})
    return {setting.name: setting.default for setting in plugin.fields if not setting.required} | given


def own_label(plugin: PluginClass, values: Mapping[str, Any]) -> str | None:
    """The plugin's `label` setting, when it has a text one: the key it reads or writes objects under."""
    if not any(setting.name == "label" and setting.kind is str for setting in plugin.fields):
        return None

    value = values.get("label")
    return value if isinstance(value, str) else None


def object_label(explicit: str | None, strategy: PluginClass, values: Mapping[str, Any],
                 detector: PluginClass | None = None, detector_values: Mapping[str, Any] | None = None) -> str:
    """The object's key in `Observation.objects`: `explicit`, else the strategy's `label` setting, else
    the detector's, else "cube". ValueError when a plugin's own label names another key."""
    strategy_label = own_label(strategy, values)
    detector_label = None if detector is None else own_label(detector, detector_values or {})
    label = explicit if explicit is not None else (strategy_label or detector_label or DEFAULT_LABEL)

    if strategy_label is not None and strategy_label != label:
        raise ValueError(f"{strategy.name}.label = {strategy_label!r}, a obiekt ma etykiete {label!r} - "
                         f"podaj --set label={label} albo --label {strategy_label}")
    if detector is not None and detector_label is not None and detector_label != label:
        raise ValueError(f"{detector.name}.label = {detector_label!r}, a obiekt ma etykiete {label!r} - "
                         f"ustaw label w sekcji {detector.name} pliku konfiguracji albo --label {detector_label}")

    return label


def object_size_m(size_mm: Sequence[float]) -> tuple[float, float, float]:
    """Edge lengths from the command line [mm]: one value is a cube, three are a box."""
    if len(size_mm) == 1:
        edge = size_mm[0] / 1000.0
        return edge, edge, edge
    if len(size_mm) == 3:
        return size_mm[0] / 1000.0, size_mm[1] / 1000.0, size_mm[2] / 1000.0

    raise ValueError(f"--size-mm: podaj 1 wartosc (szescian) albo 3 (krawedzie), jest {len(size_mm)}")


def own_time_limit(strategy: Any) -> float | None:
    """The strategy's own `time_limit_s` when it is a positive finite number."""
    own = getattr(strategy, "time_limit_s", None)
    if isinstance(own, bool) or not isinstance(own, (int, float)) or not math.isfinite(own) or own <= 0:
        return None

    return float(own)


def effective_time_limit(cap_s: float, strategy: Any) -> float:
    """The run's time limit: the cap, or the strategy's own when that is shorter. A limit given to
    `StrategyRunner` replaces the strategy's own, so the cap must never lengthen it."""
    own = own_time_limit(strategy)
    return cap_s if own is None else min(cap_s, own)


def strategy_cameras(workspace: Workspace, twin: Any) -> list[str]:
    """The cameras a strategy may read, as the panel offers them on a simulated arm: the enabled
    simulated workstation cameras, then the cameras on the arm. Frames render only when read."""
    names = [camera.name for camera in workspace.cameras if camera.enabled and camera.simulated]
    return names + [name for name in twin.spec.cameras if name not in names]


def episode_name(index: int, episodes: int) -> str:
    """`000`, `001`, ...: zero-padded, so names sort like numbers and `twin episode` lists them in order."""
    return f"{index:0{max(3, len(str(episodes - 1)))}d}"


def campaign_folder(explicit: Path | None, workspace: Workspace, strategy: str) -> Path:
    """The campaign's folder, created. An explicit one must be new or empty; the default is
    `eval-<strategy>-<time>` next to the workstation's episode logs, with `-2`, `-3` on a clash."""
    if explicit is not None:
        if explicit.exists() and (not explicit.is_dir() or any(explicit.iterdir())):
            raise ValueError(f"--out {explicit}: istnieje i nie jest pustym folderem - podaj nowy")
        explicit.mkdir(parents=True, exist_ok=True)
        return explicit.resolve()

    stem = f"eval-{re.sub(r'[^A-Za-z0-9._-]', '_', strategy)}-{time.strftime('%Y%m%d-%H%M%S', time.localtime())}"
    parent = episodes_dir(workspace.path)
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        folder = parent / (stem if number == 1 else f"{stem}-{number}")
        try:
            folder.mkdir()
        except FileExistsError:
            number += 1
            continue
        return folder.resolve()


# --------------------------------------------------------------------------- the campaign


@dataclass(frozen=True)
class Campaign:
    """What to run: the strategy and its settings, the object and its area, the rule, how many times."""

    plugin: StrategyPlugin
    #: The strategy's settings, every default included (`plugin_values`).
    settings: Mapping[str, Any]
    rule: str
    label: str
    obj: ObjectSpec
    area: Area
    seed: int = 0
    episodes: int = 10
    #: Each run's time limit cap [simulated s]; a strategy's own shorter limit stays.
    time_limit_s: float = 120.0
    #: The joint speed cap of the panel's slider [deg/s]; None = none.
    speed_deg_s: float | None = None
    #: A run cut after this many real seconds [s]; None = none. Results then depend on the machine.
    wall_limit_s: float | None = None
    #: Object poses from this detector instead of the simulation's truth.
    detector: DetectorPlugin | None = None
    detector_settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"regula {self.rule!r}: znam {', '.join(RULES)}")
        if not _whole(self.seed) or self.seed < 0:
            raise ValueError(f"ziarno musi byc liczba calkowita od 0, jest {self.seed!r}")
        if not _whole(self.episodes) or self.episodes < 1:
            raise ValueError(f"przebiegow musi byc co najmniej 1, jest {self.episodes!r}")
        for flag, value in (("--time-limit-s", self.time_limit_s), ("--speed-deg-s", self.speed_deg_s),
                            ("--wall-limit-s", self.wall_limit_s)):
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"{flag} musi byc dodatnie i skonczone, jest {value:g}")

    @property
    def poses(self) -> str:
        """Where the strategy's object poses come from, as the report names it."""
        return SIM_SOURCE if self.detector is None else f"detektor {self.detector.name}"


@dataclass(frozen=True)
class StrategyFacts:
    """What one instance of the strategy tells before any run: its rate and its time limits."""

    hz: float
    own_time_limit_s: float | None
    time_limit_s: float


@dataclass(frozen=True)
class EpisodeRow:
    """One episode: where the object was, how the strategy ended and what it claimed, the scene's verdict."""

    index: int
    name: str
    placement: Placement
    start: SceneSample
    #: The object touched the robot as placed.
    clash: bool
    #: The runner's end code and Polish reason (`episode.EndCode`).
    code: str
    reason: str
    #: The strategy's own claim: True for `done`, False for `gave_up`, None for any other end.
    claim: bool | None
    done: bool
    success: bool | None
    error: float | None
    note: str
    steps: int
    #: `strategy.status.attempt` when the strategy has one.
    attempts: int | None
    #: What `runner.start` raised: a refusal or a failed engage; "" when it started.
    start_error: str
    verdict: Verdict
    #: The run's length on the simulated clock [s], the episode log's end `t`.
    duration_s: float
    #: The whole episode's real time, setup and judging included [s].
    wall_s: float
    #: The wall limit cut the run.
    wall_limit: bool
    #: The episode log's file name in the campaign folder; None when it could not be opened.
    log: str | None
    log_error: str

    @property
    def seed(self) -> int:
        return self.placement.seed

    @property
    def agree(self) -> bool:
        return (self.claim is True) == self.verdict.ok

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "seed": self.seed,
            "log": self.log,
            "log_error": self.log_error,
            "placement": self.placement.to_json(),
            "start": {"position_m": self.start.position_m.tolist(), "contacts": list(self.start.contacts),
                      "clash": self.clash},
            "strategy": {"code": self.code, "reason": self.reason, "claim": self.claim, "done": self.done,
                         "success": self.success, "error": self.error, "note": self.note, "steps": self.steps,
                         "attempts": self.attempts, "start_error": self.start_error},
            "scene": self.verdict.to_json(),
            "agree": self.agree,
            "duration_s": self.duration_s,
            "wall_s": self.wall_s,
            "wall_limit": self.wall_limit,
        }


@dataclass
class CampaignResult:
    """The episodes a campaign ran, and how it ended."""

    folder: Path
    started: str
    facts: StrategyFacts
    planned: int
    rows: list[EpisodeRow] = field(default_factory=list)
    #: Ctrl-C: the episode it cut is not a row.
    interrupted: bool = False
    #: Why the campaign stopped early; "" when it did not.
    aborted: str = ""

    @property
    def finished(self) -> bool:
        return not self.interrupted and not self.aborted and len(self.rows) == self.planned


# --------------------------------------------------------------------------- object poses


class PoseSource(Protocol):
    """Where the strategy's `Observation.objects` come from during one episode."""

    def objects(self) -> Mapping[str, ObjectPose]: ...

    def begin(self) -> None: ...

    def advance(self, now: float) -> None: ...

    def end(self) -> None: ...


class DetectorPoses:
    """Object poses from a detector through `Perception`, as the panel gets them, on the simulated clock.

    Frames are renders of the workstation's simulated cameras, stamped on the twin's clock; a real
    camera is never opened. The detector is called on its own interval, between physics steps.
    """

    def __init__(self, twin: Any, workspace: Workspace, plugin: DetectorPlugin, values: Mapping[str, Any]):
        simulated = {camera.name for camera in workspace.cameras if camera.simulated}

        def frame(name: str) -> tuple[np.ndarray | None, float]:
            now = twin.clock.now()
            return (twin.render(name), now) if name in simulated else (None, now)

        def occluders(times: Mapping[str, float]) -> dict[str, np.ndarray]:
            return twin.render_with(lambda scene: {name: arm_mask(scene, name, dilate=MASK_DILATE_PX)
                                                   for name in times})

        self._twin = twin
        self._plugin = plugin
        self._values = dict(values)
        self._next_due = math.inf
        self._runner: Any = None
        self.perception = Perception(twin, workspace, clock=twin.clock.now, frame=frame, occluders=occluders)

    def objects(self) -> Mapping[str, ObjectPose]:
        return self.perception.objects()

    def begin(self) -> None:
        """Starts the detector and calls it once, so it has looked before the strategy starts."""
        self._runner = self.perception.start_detector(self._plugin, self._values, threaded=False)
        self._call(self._twin.clock.now())

    def advance(self, now: float) -> None:
        if self._runner is not None and now + EPS >= self._next_due:
            self._call(now)

    def end(self) -> None:
        self.perception.stop_all("koniec przebiegu")

    def _call(self, now: float) -> None:
        self._next_due = now + self._runner.step_once()


def _poses(twin: Any, workspace: Workspace, campaign: Campaign) -> PoseSource:
    if campaign.detector is None:
        return SimulationPoses(twin, campaign.label)

    return DetectorPoses(twin, workspace, campaign.detector, campaign.detector_settings)


# --------------------------------------------------------------------------- one episode


def preflight(workspace: Workspace, campaign: Campaign) -> StrategyFacts:
    """Refuses, with ValueError or TypeError, what would fail in every episode: settings the strategy
    or the detector refuses, a robot without jaw sensors, an area off the table, a strategy the
    runner refuses, a detector without a simulated camera or a render. Builds and closes one twin;
    runs nothing."""
    strategy = campaign.plugin.create(campaign.settings)
    if campaign.detector is not None:
        campaign.detector.create(campaign.detector_settings)
    check_jaws(workspace.spec())

    twin = Twin(workspace, loop_hz=LOOP_HZ, clock=VirtualClock(t0=0.0))
    try:
        configure_object(twin, campaign.obj)
        check_on_table(campaign.area, campaign.obj, twin.scene)
        StrategyRunner(twin, strategy)
        if campaign.detector is not None:
            _check_detector_cameras(twin, workspace, campaign.detector.name)

        return StrategyFacts(float(strategy.hz), own_time_limit(strategy),
                             effective_time_limit(campaign.time_limit_s, strategy))
    finally:
        twin.close()


def _check_detector_cameras(twin: Any, workspace: Workspace, name: str) -> None:
    simulated = {camera.name for camera in workspace.cameras if camera.simulated}
    cameras = [camera for camera in Perception(twin, workspace, hand=lambda: None).cameras() if camera in simulated]
    if not cameras:
        raise ValueError(f"--detector {name}: stanowisko nie ma zaufanej kamery symulowanej (zrodlo sim), "
                         "z ktorej detektor by widzial")

    try:
        twin.render(cameras[0])
    except Exception as exc:
        raise ValueError(f"render kamery {cameras[0]} nie dziala: {exc} (bez ekranu: MUJOCO_GL=egl)") from exc


def drive(twin: Any, runner: StrategyRunner, poses: PoseSource, watch: Watch, start_t: float, limit_s: float,
          wall_limit_s: float | None, wall_t0: float) -> bool:
    """Runs the twin's loop and the strategy on one simulated clock until the run ends.

    The strategy ticks at exactly `start_t + k / hz` and the loop steps at its own period in between,
    as the tests' `drive` does: at 20 Hz on a 50 Hz loop, the strategy ticks every 5 cells of a 10 ms
    grid and the loop every 2. The scene is sampled after every physics step. Returns True when the
    wall limit stopped the run (between ticks only: an `act` that never returns cannot be cut).
    """
    step_dt = 1.0 / twin.loop_hz
    period = 1.0 / float(runner.strategy.hz)
    ticks = 0
    wall_cut = False

    while runner.status.running:
        due = start_t + ticks * period
        if twin.connected and twin.clock.now() + step_dt <= due + EPS:
            twin.step(step_dt)
            now = twin.clock.now()
            watch.observe(sample(twin), now - start_t)
            poses.advance(now)
            if wall_limit_s is not None and time.monotonic() - wall_t0 > wall_limit_s:
                runner.stop(f"ocena: limit czasu rzeczywistego {wall_limit_s:g} s")
                wall_cut = True
            elif now - start_t > limit_s + OVERRUN_S:
                runner.stop("ocena: przebieg nie skonczyl sie w limicie")
        else:
            runner.step_once(due)
            ticks += 1

    return wall_cut


def settle(twin: Any) -> SceneSample | None:
    """The object after the scene ran on for `SETTLE_S` with the arm holding; None when it cannot run."""
    step_dt = 1.0 / twin.loop_hz
    for _ in range(round(SETTLE_S * twin.loop_hz)):
        if not twin.connected:
            return None
        twin.step(step_dt)

    return sample(twin) if twin.connected else None


def run_episode(workspace: Workspace, campaign: Campaign, folder: Path, index: int) -> EpisodeRow:
    """One episode on a fresh twin: place, run, judge. A strategy that refuses or fails ends only
    its own episode. Ctrl-C ends the run with an end line in its log, then propagates."""
    seed = campaign.seed + index
    placement = draw_placement(campaign.area, seed)
    episode_t0 = time.monotonic()
    twin = Twin(workspace, loop_hz=LOOP_HZ, clock=VirtualClock(t0=0.0))
    poses: PoseSource | None = None
    runner: StrategyRunner | None = None

    try:
        if campaign.speed_deg_s is not None:
            twin.set_speed(campaign.speed_deg_s)
        configure_object(twin, campaign.obj, (placement.x_m, placement.y_m))
        with twin.lock:
            twin.scene.set_joints(twin.home_pose)
        twin.connect("sim", threaded=False)
        place_object(twin, campaign.obj, placement)
        start = sample(twin)
        clash = robot_contacts(twin) > 0

        poses = _poses(twin, workspace, campaign)
        poses.begin()
        strategy = campaign.plugin.create(campaign.settings)
        limit_s = effective_time_limit(campaign.time_limit_s, strategy)
        name = episode_name(index, campaign.episodes)
        options = EpisodeOptions(folder=folder, name=name, meta=_meta(campaign, folder, index, placement))
        runner = StrategyRunner(twin, strategy, objects=poses.objects, cameras=strategy_cameras(workspace, twin),
                                time_limit_s=limit_s, episode=options)

        start_t = twin.clock.now()
        run_t0 = time.monotonic()
        start_error = _start(runner, index)
        watch = Watch(start)
        wall_cut = drive(twin, runner, poses, watch, start_t, limit_s, campaign.wall_limit_s, run_t0)
        duration_s = twin.clock.now() - start_t
        end = sample(twin)
        settled = settle(twin)
        verdict = judge(campaign.rule, start, clash, watch, end, settled)
        status = runner.status

        return EpisodeRow(
            index=index, name=name, placement=placement, start=start, clash=clash,
            code=status.stopped_code, reason=status.stopped_because, claim=_claim(status.stopped_code),
            done=status.done, success=_bool_or_none(status.success), error=_float_or_none(status.error),
            note=str(status.note), steps=status.step,
            attempts=_attempts(strategy), start_error=start_error, verdict=verdict, duration_s=duration_s,
            wall_s=time.monotonic() - episode_t0, wall_limit=wall_cut,
            log=None if status.episode is None else status.episode.name, log_error=status.episode_error,
        )
    except KeyboardInterrupt:
        if runner is not None and runner.status.running:
            runner.stop("przerwana przez operatora")
        raise
    finally:
        if poses is not None:
            poses.end()
        twin.close()


def _start(runner: StrategyRunner, index: int) -> str:
    """Starts the run; what it raised when the strategy refused or the arm was taken at once."""
    try:
        runner.start(threaded=False)
    except Exception as exc:
        # The runner has recorded a refusal or a failed engage; with no end code it never began.
        if not runner.status.stopped_code:
            raise EvaluationAborted(f"przebieg {index}: ramie nie ruszylo: {exc}") from exc
        return f"{type(exc).__name__}: {exc}"

    return ""


def _meta(campaign: Campaign, folder: Path, index: int, placement: Placement) -> dict[str, Any]:
    """The log header's `meta`: the panel's three keys, the run and its seed, and the evaluation."""
    plugin = campaign.plugin
    return {
        "strategy": plugin.name,
        "source": plugin.source,
        "settings": dict(campaign.settings),
        "run": index,
        "seed": placement.seed,
        "evaluation": {"campaign": folder.name, "rule": campaign.rule, "label": campaign.label,
                       "poses": campaign.poses, "object": campaign.obj.to_json(),
                       "placement": placement.to_json()},
    }


def _claim(code: str) -> bool | None:
    return {"done": True, "gave_up": False}.get(code)


def _attempts(strategy: Any) -> int | None:
    attempt = getattr(getattr(strategy, "status", None), "attempt", None)
    return attempt if _whole(attempt) else None


def _bool_or_none(value: Any) -> bool | None:
    """A strategy's `success` as a bool; None when it gave none or one that has no truth value."""
    if value is None:
        return None
    try:
        return bool(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    """A strategy's `error` as a float, whatever number type it came in; None when it is not a number."""
    if value is None:
        return None
    try:
        return float(np.asarray(value))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- the campaign


def run_campaign(workspace: Workspace, campaign: Campaign, folder: Path, *, facts: StrategyFacts | None = None,
                 on_episode: Callable[[EpisodeRow], None] | None = None) -> CampaignResult:
    """Runs every episode into `folder` and rewrites its summary after each one.

    The summary exists before the first episode, so the plan is on disk whatever happens. Ctrl-C
    ends the campaign with `interrupted`; an `EvaluationAborted` with `aborted`. Any other exception
    is written down as `aborted` and propagates.
    """
    facts = facts if facts is not None else preflight(workspace, campaign)
    folder.mkdir(parents=True, exist_ok=True)
    result = CampaignResult(folder, time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()), facts, campaign.episodes)
    write_summary(folder, summary_data(workspace, campaign, result))

    try:
        for index in range(campaign.episodes):
            row = run_episode(workspace, campaign, folder, index)
            result.rows.append(row)
            if on_episode is not None:
                on_episode(row)
            write_summary(folder, summary_data(workspace, campaign, result))
    except KeyboardInterrupt:
        result.interrupted = True
    except EvaluationAborted as exc:
        result.aborted = str(exc)
    except Exception as exc:
        result.aborted = f"{type(exc).__name__}: {exc}"
        # The error that ended the campaign propagates even when its summary cannot be written.
        with contextlib.suppress(Exception):
            write_summary(folder, summary_data(workspace, campaign, result))
        raise

    write_summary(folder, summary_data(workspace, campaign, result))
    return result


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float] | None:
    """The Wilson score interval of a success rate, 95% by default; None without trials."""
    if trials <= 0:
        return None

    rate = successes / trials
    scale = 1.0 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / scale
    half = z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4 * trials * trials)) / scale
    return max(0.0, centre - half), min(1.0, centre + half)


def totals(rows: Sequence[EpisodeRow]) -> dict[str, Any]:
    """Counts over the episodes: the scene's successes, the strategy's claims, where they disagree."""
    successes = sum(row.verdict.ok for row in rows)
    interval = wilson(successes, len(rows))
    claimed = [row.claim for row in rows]

    return {
        "episodes": len(rows),
        "scene": {"success": successes, "rate": successes / len(rows) if rows else None,
                  "interval_95": None if interval is None else list(interval)},
        "claim": {"success": claimed.count(True), "failure": claimed.count(False), "none": claimed.count(None)},
        "agreement": {
            "both_success": sum(row.claim is True and row.verdict.ok for row in rows),
            "both_failure": sum(row.claim is not True and not row.verdict.ok for row in rows),
            "false_claim": sum(row.claim is True and not row.verdict.ok for row in rows),
            "unclaimed_success": sum(row.claim is not True and row.verdict.ok for row in rows),
            "disagree": [row.name for row in rows if not row.agree],
        },
        "codes": dict(Counter(row.code for row in rows).most_common()),
        "duration_s": _spread([row.duration_s for row in rows]),
        "steps": _spread([row.steps for row in rows]),
        "attempts": _spread([row.attempts for row in rows if row.attempts is not None]),
        "wall_s": sum(row.wall_s for row in rows),
    }


def _spread(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None

    return {"median": statistics.median(values), "max": max(values)}


def summary_data(workspace: Workspace, campaign: Campaign, result: CampaignResult) -> dict[str, Any]:
    """`summary.json`: the conditions, the totals and one entry per episode. SI units, angles in degrees."""
    plugin, facts, detector = campaign.plugin, result.facts, campaign.detector
    return {
        "kind": "evaluation",
        "format": FORMAT,
        "version": VERSION,
        "started": result.started,
        "finished": result.finished,
        "interrupted": result.interrupted,
        "aborted": result.aborted,
        "versions": {"twin": twin_version, "api": machinekind_twin_api.__version__, "mujoco": mujoco.__version__},
        "conditions": {
            "robot": workspace.robot,
            "workspace": None if workspace.path is None else str(workspace.path),
            "config": os.environ.get(CONFIG_ENV) or None,
            "loop_hz": LOOP_HZ,
            "speed_deg_s": campaign.speed_deg_s,
            "seed": campaign.seed,
            "episodes": campaign.episodes,
            "rule": {"name": campaign.rule, **rule_constants()},
            "object": campaign.obj.to_json(),
            "area": campaign.area.to_json(),
            "poses": campaign.poses,
            "detector": None if detector is None else {"name": detector.name, "source": detector.source,
                                                       "settings": dict(campaign.detector_settings)},
            "time_limit_cap_s": campaign.time_limit_s,
            "wall_limit_s": campaign.wall_limit_s,
            "strategy": {"name": plugin.name, "source": plugin.source, "hz": facts.hz,
                         "settings": dict(campaign.settings), "own_time_limit_s": facts.own_time_limit_s,
                         "time_limit_s": facts.time_limit_s},
        },
        "totals": totals(result.rows),
        "episodes": [row.to_json() for row in result.rows],
    }


def write_summary(folder: Path, data: Mapping[str, Any]) -> Path:
    """Writes the summary as strict JSON, atomically: a reader never sees half a file."""
    path = folder / SUMMARY
    temporary = folder / f"{SUMMARY}.tmp"
    text = json.dumps(_json_value(data), indent=2, ensure_ascii=False, allow_nan=False)
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _json_value(value: Any) -> Any:
    """`value` as strict JSON values, never rounded, as the episode log writes them. It never raises:
    a non-finite number becomes null, any other number type (a Decimal, a tensor's scalar) a float,
    and anything JSON has no type for its `repr`."""
    try:
        return _json_cleaned(value)
    except Exception:  # noqa: BLE001 - a value the summary cannot read costs only that value
        return f"<{type(value).__name__}>"


def _json_cleaned(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, PurePath):
        return str(value)
    try:
        number = float(np.asarray(value))
    except (TypeError, ValueError):
        return repr(value)
    return number if math.isfinite(number) else None


def _whole(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


# --------------------------------------------------------------------------- the terminal


def header_lines(campaign: Campaign, workspace: Workspace, folder: Path) -> list[str]:
    plugin = campaign.plugin
    first = (f"Ocena strategii {plugin.name} ({plugin.source}): ramie {workspace.robot}, regula {campaign.rule}, "
             f"ziarno {campaign.seed}, przebiegow {campaign.episodes}, pozy {campaign.poses}")
    return [first, f"Zapis: {folder}"]


def row_line(row: EpisodeRow) -> str:
    """One episode as a terminal row; the scene's reason and NIEZGODNE when they apply."""
    claim = "-" if row.claim is None else ("tak" if row.claim else "nie")
    scene = "tak" if row.verdict.ok else "nie"
    attempts = "-" if row.attempts is None else str(row.attempts)
    placement = row.placement
    line = (f"  {row.name} {row.code:<13} strategia {claim:<3} scena {scene:<3} "
            f"x {placement.x_m * 100:+6.1f} y {placement.y_m * 100:+6.1f} cm  "
            f"obrot {math.degrees(placement.yaw_rad):+4.0f} st  czas {row.duration_s:7.2f} s  "
            f"krokow {row.steps:<5} prob {attempts:<3} {row.reason}")
    if not row.verdict.ok:
        line += f"  [scena: {row.verdict.reason}]"
    if not row.agree:
        line += "  NIEZGODNE"

    return line


def footer_lines(campaign: Campaign, result: CampaignResult) -> list[str]:
    counts = totals(result.rows)
    lines = []
    if result.interrupted:
        lines.append(f"Przerwano po {len(result.rows)} z {campaign.episodes} przebiegow")

    scene = counts["scene"]
    interval = scene["interval_95"]
    share = "" if interval is None else (f" ({100 * scene['rate']:.0f}%, 95%: "
                                         f"{100 * interval[0]:.0f}-{100 * interval[1]:.0f}%)")
    lines.append(f"Scena ({campaign.rule}): {scene['success']}/{counts['episodes']}{share}")

    claim, disagree = counts["claim"], counts["agreement"]["disagree"]
    names = f" ({', '.join(disagree)})" if disagree else ""
    lines.append(f"Strategia: sukces {claim['success']}, porazka {claim['failure']}, bez zdania {claim['none']}; "
                 f"niezgodne ze scena: {len(disagree)}{names}")
    if counts["codes"]:
        lines.append("Konce: " + ", ".join(f"{code} {count}" for code, count in counts["codes"].items()))
    lines.append(f"Podsumowanie: {result.folder / SUMMARY}")

    return lines
