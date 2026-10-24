"""`twin eval-strategy`: many simulated runs of one strategy, each with a seeded object, judged by the scene.

The physics tests run in-test strategies on the test robot, whose reach ends near r = 0.12 m, so the
object lies at r = 0.10 m unless a test says otherwise. Every CLI call names its own workstation file
and output folder, so nothing lands in the repository's `workspace/`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")

from conftest import BOT, CubeScene, install_detector
from test_strategy_plugins import install_strategy

from machinekind_twin.calib.simulate import look_at
from machinekind_twin.cli import main as twin_cli
from machinekind_twin.detectors import load_detector
from machinekind_twin.episode import EpisodeLog
from machinekind_twin.evaluation import (
    Campaign,
    plugin_values,
    run_campaign,
    setting_from_text,
    totals,
    wilson,
)
from machinekind_twin.paths import CONFIG_ENV
from machinekind_twin.plugin_settings import SettingField
from machinekind_twin.robots import get_spec
from machinekind_twin.scene_truth import Area, ObjectSpec
from machinekind_twin.strategies import StrategyPlugin, load_strategy
from machinekind_twin.workspace import CameraRecord, Workspace

#: The object at r = 0.10 m straight ahead of the base, not turned.
FIXED = ("--radius-m", "0.1", "0.1", "--bearing-deg", "0", "0", "--yaw-deg", "0", "0")

PRESS = '''
from machinekind_twin_api.strategy import Action


class Strategy:
    """Presses the closed jaws down at r 0.10 m for 2 s, then claims success."""

    hz = 20.0
    time_limit_s = 10.0

    def act(self, obs):
        return Action(target={"j1": 0.0, "j2": 45.0, "gripper": 0.0}, done=obs.t >= 2.0, success=True)
'''

HOIST = '''
from machinekind_twin_api.strategy import Action

#: Until when [s], j2 [deg] and the gripper [0..100]: open, reach down, close, lift.
PHASES = ((1.0, 20.0, 100.0), (2.5, 49.0, 100.0), (3.5, 49.0, 0.0), (float("inf"), 20.0, 0.0))


class Strategy:
    """Grips the object at r 0.10 m and lifts it, then claims success at 5 s."""

    hz = 20.0
    time_limit_s = 10.0

    def act(self, obs):
        j2, gripper = next((j2, gripper) for until, j2, gripper in PHASES if obs.t < until)
        return Action(target={"j1": 0.0, "j2": j2, "gripper": gripper}, done=obs.t >= 5.0, success=True)
'''

REFUSING = '''
class Strategy:
    hz = 20.0

    def start(self, run, obs):
        raise ValueError("bez kostki")

    def act(self, obs):
        return {}
'''

FAILING = '''
class Strategy:
    hz = 20.0
    time_limit_s = 10.0

    def act(self, obs):
        if obs.step == 2:
            raise RuntimeError("zepsuta")
        return {}
'''

ENDLESS = '''
class Strategy:
    """Never done, and no time limit of its own."""

    hz = 20.0

    def act(self, obs):
        return {}
'''

SHORT = '''
class Strategy:
    hz = 20.0
    time_limit_s = 0.5

    def act(self, obs):
        return {}
'''

SLOW = '''
import time


class Strategy:
    hz = 20.0
    time_limit_s = 10.0

    def act(self, obs):
        time.sleep(0.05)
        return {}
'''

INTERRUPTED = '''
from machinekind_twin_api.strategy import Action


class Strategy:
    """The second run it starts gets Ctrl-C at its fourth tick."""

    started = 0
    hz = 20.0
    time_limit_s = 10.0

    def start(self, run, obs):
        Strategy.started += 1
        self.interrupt = Strategy.started == 2

    def act(self, obs):
        if self.interrupt and obs.step == 3:
            raise KeyboardInterrupt
        return Action(done=obs.step >= 5, success=True)
'''

LABELLED = '''
from dataclasses import dataclass


@dataclass
class Settings:
    label: str = "cube"


class Strategy:
    hz = 20.0
    settings = Settings

    def __init__(self, settings):
        self.config = settings

    def act(self, obs):
        return {}
'''

ZERO_HZ = '''
class Strategy:
    hz = 0

    def act(self, obs):
        return {}
'''

SEES = '''
from machinekind_twin_api.strategy import Action


class Strategy:
    """Done once the cameras see the cube."""

    hz = 20.0
    time_limit_s = 2.0

    def act(self, obs):
        cube = obs.objects.get("cube")
        seen = cube is not None and cube.source == "kamery"
        return Action(done=seen, success=True if seen else None)
'''

ODD_VALUES = '''
from decimal import Decimal

from machinekind_twin_api.strategy import Action


class Strategy:
    """Reports its error and note in types JSON has none for."""

    hz = 20.0
    time_limit_s = 10.0

    def act(self, obs):
        return Action(done=obs.step >= 3, success=True, error=Decimal("0.25"), note=Decimal("1.5"))
'''

FINE = '''
from dataclasses import dataclass


@dataclass
class Settings:
    tolerance_m: float = 3e-7


class Strategy:
    hz = 20.0
    time_limit_s = 0.2
    settings = Settings

    def __init__(self, settings):
        self.config = settings

    def act(self, obs):
        return {}
'''

BELOW_ONE_HERTZ = '''
class Strategy:
    """Ticks every 3.33 s, so its own limit falls between two ticks."""

    hz = 0.3
    time_limit_s = 8.0

    def act(self, obs):
        return {}
'''

REFUSING_DETECTOR = '''
class Detector:
    settings = None
    interval_s = 1.0
    pixel_error_px = 3.0

    def start(self, info):
        raise RuntimeError("brak klucza")

    def detect(self, request):
        return []
'''

LATE_REFUSING_DETECTOR = '''
class Detector:
    """Its first start passes, every later one refuses."""

    settings = None
    started = 0
    interval_s = 1.0
    pixel_error_px = 3.0

    def start(self, info):
        Detector.started += 1
        if Detector.started > 1:
            raise RuntimeError("limit zapytan")

    def detect(self, request):
        return []
'''


def installed(install_dist, dist: str, source: str) -> StrategyPlugin:
    return load_strategy(install_strategy(install_dist, dist, source))


def campaign(plugin: StrategyPlugin, *, radius: tuple[float, float] = (0.10, 0.10),
             bearing_deg: tuple[float, float] = (0.0, 0.0), **options) -> Campaign:
    options = {"rule": "touch", "label": "cube", "obj": ObjectSpec(), "episodes": 1, **options}
    area = Area(radius, (np.radians(bearing_deg[0]), np.radians(bearing_deg[1])), (0.0, 0.0))
    return Campaign(plugin=plugin, settings=plugin_values(plugin, {}), area=area, **options)


def run(plugin: StrategyPlugin, folder, **options):
    return run_campaign(Workspace(robot=BOT), campaign(plugin, **options), folder)


def summary_of(folder) -> dict:
    return json.loads((folder / "summary.json").read_text(encoding="utf-8"))


@pytest.fixture
def station(tmp_path):
    """A workstation file with the test robot and no cameras."""
    path = tmp_path / "twin.json"
    path.write_text(json.dumps({"robot": BOT}), encoding="utf-8")
    return path


def evaluate(station, out, *args: str) -> int:
    return twin_cli(["eval-strategy", *FIXED, *args, "--path", str(station), "--out", str(out)])


def watched_station() -> Workspace:
    """The test robot with `CubeScene`'s two trusted simulated cameras, looking at r = 0.2 m."""
    workspace = Workspace(robot=BOT)
    for name, eye in CubeScene.EYES:
        pose = look_at(eye, (0.2, 0.0, 0.0)).tolist()
        workspace.add_camera(CameraRecord(name, "sim", 640, 480, K=[list(row) for row in CubeScene.K],
                                          sim_pose=pose, T_cam2base=pose, intrinsics_from="symulacja",
                                          calibration={"trusted": True}))
    return workspace


# --------------------------------------------------------------------------- settings and counts


def test_settings_from_command_line_text():
    def field(kind: type) -> SettingField:
        return SettingField("value", kind, 0)

    accepted = [setting_from_text(field(float), "15", "x"), setting_from_text(field(int), "3", "x"),
                setting_from_text(field(int), "3.0", "x"), setting_from_text(field(bool), "true", "x"),
                setting_from_text(field(bool), "False", "x"), setting_from_text(field(str), "5", "x")]

    assert accepted == [15.0, 3, 3, True, False, "5"]
    assert [type(value) for value in accepted] == [float, int, int, bool, bool, str]
    for kind, text in ((int, "2.5"), (bool, "yes"), (float, "abc"), (float, "nan")):
        with pytest.raises(ValueError, match=f"x.value: potrzeba {kind.__name__}"):
            setting_from_text(field(kind), text, "x")


def test_wilson_interval_of_46_of_50_is_81_to_97_percent():
    low, high = wilson(46, 50)

    assert (low, high) == pytest.approx((0.812, 0.968), abs=0.001)
    assert wilson(0, 0) is None


# --------------------------------------------------------------------------- the scene's verdict


def test_press_is_a_touch_and_the_scene_agrees(install_dist, tmp_path):
    row = run(installed(install_dist, "eval-press", PRESS), tmp_path / "out").rows[0]

    assert (row.code, row.claim) == ("done", True)
    assert row.verdict.ok and row.agree
    assert max(row.verdict.contacts_end) > 0.5
    assert row.verdict.push_m < 0.01


def test_press_out_of_reach_is_a_false_claim(install_dist, tmp_path):
    result = run(installed(install_dist, "eval-press-far", PRESS), tmp_path / "out", radius=(0.2, 0.2), episodes=2)

    assert [(row.claim, row.verdict.ok, row.verdict.reason) for row in result.rows] \
        == [(True, False, "brak kontaktu na koncu")] * 2
    assert totals(result.rows)["agreement"]["false_claim"] == 2
    assert summary_of(tmp_path / "out")["totals"]["agreement"]["disagree"] == ["000", "001"]


def test_lift_needs_both_jaws_and_four_centimetres(install_dist, tmp_path):
    hoist = installed(install_dist, "eval-hoist", HOIST)
    press = installed(install_dist, "eval-press-lift", PRESS)

    lifted = run(hoist, tmp_path / "lift", rule="lift").rows[0]
    pushed = run(hoist, tmp_path / "touch", rule="touch").rows[0]
    pressed = run(press, tmp_path / "press", rule="lift").rows[0]

    assert lifted.verdict.ok
    assert lifted.verdict.rise_settled_m >= 0.04
    assert min(lifted.verdict.contacts_end) > 0.5
    assert not pushed.verdict.ok and pushed.verdict.reason.startswith("obiekt przesuniety")
    assert not pressed.verdict.ok and pressed.verdict.reason.startswith("uniesiony")


def test_episode_i_is_the_same_in_any_campaign(install_dist, tmp_path):
    press = installed(install_dist, "eval-press-seed", PRESS)
    options = {"radius": (0.095, 0.105), "bearing_deg": (-3.0, 3.0)}

    three = run(press, tmp_path / "three", seed=100, episodes=3, **options).rows
    one = run(press, tmp_path / "one", seed=102, episodes=1, **options).rows

    def same(row) -> dict:
        data = row.to_json()
        return {key: value for key, value in data.items() if key not in ("index", "name", "seed", "log", "wall_s")}

    assert three[2].seed == one[0].seed == 102
    assert same(three[2]) == same(one[0])
    assert three[0].placement != three[1].placement


# --------------------------------------------------------------------------- how runs end


def test_a_refusing_strategy_is_counted_and_the_campaign_goes_on(install_dist, tmp_path):
    rows = run(installed(install_dist, "eval-refusing", REFUSING), tmp_path / "out", episodes=2).rows

    assert [(row.code, row.reason, row.claim, row.verdict.ok) for row in rows] \
        == [("refused", "strategia odmowila: bez kostki", None, False)] * 2
    assert all(row.start_error == "ValueError: bez kostki" for row in rows)
    for name in ("000", "001"):
        assert EpisodeLog.summary(tmp_path / "out" / f"{name}.jsonl").end.began is False


def test_a_strategy_that_raises_ends_as_error_and_the_campaign_goes_on(install_dist, tmp_path):
    rows = run(installed(install_dist, "eval-failing", FAILING), tmp_path / "out", episodes=3).rows

    assert [row.code for row in rows] == ["error"] * 3
    assert all(row.reason.startswith("RuntimeError:") for row in rows)
    assert [row.steps for row in rows] == [2] * 3


def test_a_strategy_below_one_hertz_ends_at_its_own_time_limit(install_dist, tmp_path):
    row = run(installed(install_dist, "eval-below-one-hertz", BELOW_ONE_HERTZ), tmp_path / "out").rows[0]

    assert (row.code, row.steps) == ("time_limit", 3)
    assert row.duration_s == pytest.approx(10.0, abs=0.03)


def test_a_strategy_without_a_limit_ends_at_the_cap(install_dist, tmp_path):
    row = run(installed(install_dist, "eval-endless", ENDLESS), tmp_path / "out", time_limit_s=1.0).rows[0]

    assert (row.code, row.steps, row.claim) == ("time_limit", 20, None)
    assert row.duration_s == pytest.approx(1.0)
    assert EpisodeLog.summary(tmp_path / "out" / "000.jsonl").end.t == pytest.approx(row.duration_s)


@pytest.mark.parametrize("cap, limit, steps", [(120.0, 0.5, 10), (0.25, 0.25, 5)])
def test_the_cap_never_lengthens_the_strategy_s_own_limit(install_dist, tmp_path, cap, limit, steps):
    row = run(installed(install_dist, "eval-short", SHORT), tmp_path / "out", time_limit_s=cap).rows[0]
    header = EpisodeLog.summary(tmp_path / "out" / "000.jsonl").header

    assert header["strategy"]["time_limit_s"] == limit
    assert (row.code, row.steps) == ("time_limit", steps)
    assert summary_of(tmp_path / "out")["conditions"]["strategy"]["time_limit_s"] == limit


def test_the_wall_limit_stops_a_slow_strategy_between_ticks(install_dist, tmp_path):
    row = run(installed(install_dist, "eval-slow", SLOW), tmp_path / "out", wall_limit_s=0.2).rows[0]

    assert row.code == "stopped"
    assert "limit czasu rzeczywistego" in row.reason
    assert row.wall_limit
    assert EpisodeLog.summary(tmp_path / "out" / "000.jsonl").end.code == "stopped"


def test_ctrl_c_leaves_a_partial_summary(install_dist, station, tmp_path, capsys):
    plugin = installed(install_dist, "eval-interrupted", INTERRUPTED)

    result = run(plugin, tmp_path / "out", episodes=3)
    summary = summary_of(tmp_path / "out")
    end = EpisodeLog.summary(tmp_path / "out" / "001.jsonl").end

    assert result.interrupted
    assert (summary["interrupted"], summary["finished"], len(summary["episodes"])) == (True, False, 1)
    assert (end.code, end.reason) == ("stopped", "przerwana przez operatora")
    assert not (tmp_path / "out" / "002.jsonl").exists()

    plugin.cls.started = 0
    assert evaluate(station, tmp_path / "again", plugin.name, "--episodes", "3") == 130
    assert "Przerwano po 1 z 3 przebiegow" in capsys.readouterr().out


def test_an_unexpected_error_is_written_into_the_summary_and_propagates(install_dist, tmp_path):
    press = installed(install_dist, "eval-press-broken-pipe", PRESS)

    def print_row(row):
        raise BrokenPipeError("stdout zamkniete")

    with pytest.raises(BrokenPipeError):
        run_campaign(Workspace(robot=BOT), campaign(press, episodes=2), tmp_path / "out", on_episode=print_row)
    summary = summary_of(tmp_path / "out")

    assert (summary["finished"], summary["aborted"]) == (False, "BrokenPipeError: stdout zamkniete")
    assert len(summary["episodes"]) == 1


# --------------------------------------------------------------------------- the summary


def test_a_value_json_has_no_type_for_does_not_stop_the_campaign(install_dist, tmp_path):
    rows = run(installed(install_dist, "eval-odd-values", ODD_VALUES), tmp_path / "out", episodes=2).rows
    summary = summary_of(tmp_path / "out")
    logged = EpisodeLog.load(tmp_path / "out" / "000.jsonl").steps[-1]

    assert [(row.error, row.success) for row in rows] == [(0.25, True)] * 2
    assert summary["finished"] is True
    assert [entry["strategy"]["error"] for entry in summary["episodes"]] == [0.25, 0.25]
    assert summary["episodes"][0]["strategy"]["note"] == logged["note"]


def test_the_summary_keeps_the_settings_the_strategy_got(install_dist, tmp_path):
    run(installed(install_dist, "eval-fine", FINE), tmp_path / "out")
    meta = EpisodeLog.load(tmp_path / "out" / "000.jsonl").header["meta"]

    assert summary_of(tmp_path / "out")["conditions"]["strategy"]["settings"] == meta["settings"] \
        == {"tolerance_m": 3e-7}


# --------------------------------------------------------------------------- poses from a detector


@pytest.mark.render
def test_the_cube_detector_feeds_the_strategy_through_perception(cube_example, install_dist, tmp_path):
    detector = load_detector(cube_example)
    plan = campaign(installed(install_dist, "eval-sees", SEES), radius=(0.2, 0.2), time_limit_s=2.0,
                    detector=detector, detector_settings=plugin_values(detector, {}))

    row = run_campaign(watched_station(), plan, tmp_path / "out").rows[0]
    seen = [step["objects"]["cube"] for step in EpisodeLog.load(tmp_path / "out" / "000.jsonl").steps]

    assert row.claim is True
    assert seen[-1]["source"] == "kamery"
    assert np.linalg.norm(np.asarray(seen[-1]["position"]) - [0.2, 0.0, 0.015]) < 0.005
    assert summary_of(tmp_path / "out")["conditions"]["poses"] == "detektor cube"


@pytest.mark.render
def test_a_detector_that_refuses_to_start_is_refused_before_anything_is_written(swing, install_dist, tmp_path,
                                                                                capsys):
    station = watched_station().save(tmp_path / "watched.json")
    name = install_detector(install_dist, "eval-refusing-detector", REFUSING_DETECTOR)
    out = tmp_path / "out"

    code = evaluate(station, out, "swing", "--set", "joint=j1", "--detector", name)

    assert code == 1
    assert f"detektor {name} odmowil: brak klucza" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.render
def test_a_detector_that_refuses_in_an_episode_aborts_the_campaign(swing, install_dist, tmp_path, capsys):
    station = watched_station().save(tmp_path / "watched.json")
    name = install_detector(install_dist, "eval-late-detector", LATE_REFUSING_DETECTOR)
    out = tmp_path / "out"

    code = evaluate(station, out, "swing", "--set", "joint=j1", "--detector", name, "--episodes", "2")
    err = capsys.readouterr().err
    summary = summary_of(out)

    assert code == 1
    assert "przebieg 0" in err and "limit zapytan" in err
    assert (summary["finished"], summary["episodes"]) == (False, [])
    assert summary["aborted"].startswith("przebieg 0") and "limit zapytan" in summary["aborted"]


# --------------------------------------------------------------------------- the command


def test_swing_claims_success_and_the_scene_says_no(swing, station, tmp_path, capsys):
    out = tmp_path / "out"

    code = evaluate(station, out, "swing", "--set", "joint=j1", "--set", "period_s=1", "--set", "cycles=1",
                    "--episodes", "2", "--seed", "7")
    printed = capsys.readouterr().out
    rows = [line for line in printed.splitlines() if line.startswith("  0")]

    assert code == 0
    assert sorted(path.name for path in out.iterdir()) == ["000.jsonl", "001.jsonl", "summary.json"]
    assert len(rows) == 2
    assert all("NIEZGODNE" in row for row in rows)
    assert "Scena (touch): 0/2" in printed
    assert "niezgodne ze scena: 2" in printed

    assert twin_cli(["episode", str(out)]) == 0
    listed = capsys.readouterr().out.splitlines()[1:]
    assert len(listed) == 2
    assert not any("nieczytelny" in line for line in listed)


def test_each_log_carries_the_placement_and_every_step_the_true_pose(swing, station, tmp_path):
    out = tmp_path / "out"
    assert evaluate(station, out, "swing", "--set", "joint=j1", "--set", "period_s=1", "--set", "cycles=1",
                    "--seed", "7", "--episodes", "1") == 0

    log = EpisodeLog.load(out / "000.jsonl")
    meta = log.header["meta"]

    assert (meta["run"], meta["seed"], meta["strategy"]) == (0, 7, "swing")
    assert meta["settings"]["period_s"] == 1.0
    assert meta["evaluation"]["placement"] == summary_of(out)["episodes"][0]["placement"]
    assert log.steps
    for step in log.steps:
        cube = step["objects"]["cube"]
        assert cube["source"] == "symulacja"
        assert cube["position"] == pytest.approx([0.1, 0.0, 0.015], abs=1e-3)


def test_settings_come_from_the_config_section_and_set(swing, station, tmp_path):
    config = tmp_path / "local.yaml"
    config.write_text("swing:\n  joint: j1\n  period_s: 1.0\n", encoding="utf-8")
    out = tmp_path / "out"

    assert evaluate(station, out, "swing", "--config", str(config), "--set", "cycles=1", "--set", "amplitude=5",
                    "--episodes", "1") == 0

    settings = {"joint": "j1", "amplitude": 5.0, "period_s": 1.0, "cycles": 1, "wait_for_return": True}
    conditions = summary_of(out)["conditions"]
    assert conditions["strategy"]["settings"] == settings
    assert conditions["config"] == str(config.resolve())
    assert EpisodeLog.load(out / "000.jsonl").header["meta"]["settings"] == settings


#: argv after `eval-strategy` and what stderr says; `{name}` fields are filled in by the test.
REFUSALS = {
    "an unknown strategy": (["nope"], "Nieznana strategia"),
    "a setting that is not a number": (["swing", "--set", "joint=j1", "--set", "amplitude=abc"],
                                       "swing.amplitude: potrzeba float"),
    "an unknown setting": (["swing", "--set", "joint=j1", "--set", "nope=1"], "Nieznane klucze w sekcji swing"),
    "a missing setting": (["swing"], "Brak kluczy w sekcji swing"),
    "a blank required setting": (["swing", "--set", "joint="], "Brak kluczy w sekcji swing"),
    "a broken config section that --set overrides": (["swing", "--config", "{broken}", "--set", "amplitude=5"],
                                                     "swing.amplitude: potrzeba float"),
    "a setting without a value": (["swing", "--set", "joint=j1", "--set", "amplitude"], "KLUCZ=WARTOSC"),
    "an output folder in use": (["swing", "--set", "joint=j1"], "nie jest pustym folderem"),
    "an inverted range": (["swing", "--set", "joint=j1", "--radius-m", "0.2", "0.1"], "poczatek wiekszy od konca"),
    "two edges": (["swing", "--set", "joint=j1", "--size-mm", "30", "30"], "--size-mm"),
    "no episodes": (["swing", "--set", "joint=j1", "--episodes", "0"], "przebiegow musi byc co najmniej 1"),
    "an area off the table": (["swing", "--set", "joint=j1", "--radius-m", "0.8", "0.9"], "poza blat"),
    "a label the strategy does not read": (["{labelled}", "--label", "pen"], "label = 'cube', a obiekt ma etykiete"),
    "a strategy without a rate": (["{zero_hz}"], "hz musi byc dodatnie"),
    "a detector without cameras": (["swing", "--set", "joint=j1", "--detector", "{cube}"], "kamery symulowanej"),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_refusals(case, swing, cube_example, install_dist, station, tmp_path, capsys):
    args, expected = REFUSALS[case]
    broken = tmp_path / "broken.yaml"
    broken.write_text("swing:\n  joint: j1\n  amplitude: duzo\n", encoding="utf-8")
    names = {"broken": str(broken), "cube": cube_example,
             "labelled": install_strategy(install_dist, "eval-labelled", LABELLED),
             "zero_hz": install_strategy(install_dist, "eval-zero-hz", ZERO_HZ)}
    out = tmp_path / "out"
    if case == "an output folder in use":
        out.mkdir()
        (out / "notes.txt").write_text("moje", encoding="utf-8")

    code = evaluate(station, out, *[arg.format(**names) for arg in args])
    err = capsys.readouterr().err

    assert code == 1
    assert expected in err
    assert "'Nieznana" not in err
    if case == "an output folder in use":
        assert [path.name for path in out.iterdir()] == ["notes.txt"]
    else:
        assert not out.exists()


def test_without_a_workspace_file_the_campaign_lands_in_the_default_episodes_folder(swing, tmp_path):
    code = twin_cli(["eval-strategy", *FIXED, "swing", "--robot", BOT, "--set", "joint=j1", "--set", "period_s=0.5",
                     "--set", "cycles=1", "--episodes", "1"])
    folders = list((tmp_path / "workspace" / "episodes").glob("eval-swing-*"))

    assert code == 0
    assert len(folders) == 1
    assert sorted(path.name for path in folders[0].iterdir()) == ["000.jsonl", "summary.json"]


@pytest.mark.cli
def test_the_command_runs_from_installed_packages(tmp_path):
    station = tmp_path / "station.json"
    station.write_text(json.dumps({"robot": "arm6"}), encoding="utf-8")
    joint = get_spec("arm6").joints[0]
    environment = {key: value for key, value in os.environ.items() if key != CONFIG_ENV}

    completed = subprocess.run(
        [sys.executable, "-m", "machinekind_twin.cli", "eval-strategy", "swing", "--path", str(station),
         "--out", str(tmp_path / "out"), "--episodes", "1", "--set", f"joint={joint}", "--set", "period_s=0.5",
         "--set", "cycles=1"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "out" / "000.jsonl").is_file()
    assert (tmp_path / "out" / "summary.json").is_file()
