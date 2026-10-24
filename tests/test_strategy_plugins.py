"""Strategies from installed packages: found through entry points, built strictly from their settings,
run by the same `StrategyRunner` as the twin's test strategy.

The example package `examples/strategies/swing` lives outside `src/machinekind_twin/`; the `swing`
fixture installs it the way pip would.
"""

from __future__ import annotations

import math
import subprocess
import sys

import pytest
from machinekind_twin_api.entry_points import STRATEGIES

pytest.importorskip("mujoco")

from conftest import BOT, SWING
from test_strategy import drive

from machinekind_twin.clock import VirtualClock
from machinekind_twin.episode import EpisodeLog, EpisodeOptions
from machinekind_twin.episode_report import NOT_NAMED, DelayFit, report
from machinekind_twin.episode_report import STILL as STILL_JOINT
from machinekind_twin.runtime import Twin
from machinekind_twin.strategies import ENTRY_POINT_GROUP, config_section, load_strategy, strategies
from machinekind_twin.strategy import StrategyRunner
from machinekind_twin.workspace import Workspace

GROUP = "machinekind_twin.strategies"


@pytest.fixture
def twin():
    twin = Twin(Workspace(robot=BOT), clock=VirtualClock(t0=0.0))
    try:
        yield twin
    finally:
        twin.close()


def install_strategy(install_dist, dist: str, source: str) -> str:
    """Another installed package: a module named like it, whose `Strategy` it registers under that name.

    Every test needs its own `dist`: a module once imported stays in `sys.modules`.
    """
    module = dist.replace("-", "_")
    install_dist(dist, {GROUP: {module: f"{module}:Strategy"}}, {module: source})
    return module


# --------------------------------------------------------------------------- discovery


def test_the_twin_reads_the_entry_point_group_machinekind_twin_api_names():
    assert ENTRY_POINT_GROUP == STRATEGIES == GROUP


def test_a_strategy_from_an_installed_package_is_found_through_its_entry_point(swing):
    assert strategies()[swing].value == "swing:Swing"

    plugin = load_strategy(swing)

    assert plugin.name == "swing"
    assert plugin.cls.__name__ == "Swing"
    assert plugin.cls.__module__ == "swing"
    assert plugin.source == "machinekind-twin-swing 0.1.0 (swing:Swing)"
    assert plugin.summary == "Swings one joint around where it starts, then waits for it to come back."


def test_the_example_package_lives_outside_the_twin():
    assert "src" not in SWING.parts
    assert (SWING / "swing" / "__init__.py").is_file()


def test_the_example_package_imports_nothing_from_the_twin_at_runtime():
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(SWING)!r})\n"
        "import swing\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'machinekind_twin'))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "[]"


def test_its_settings_dataclass_becomes_the_form_fields(swing):
    fields = {field.name: field for field in load_strategy(swing).fields}

    assert list(fields) == ["joint", "amplitude", "period_s", "cycles", "wait_for_return"]
    assert fields["joint"].kind is str
    assert fields["joint"].required
    assert fields["amplitude"].kind is float
    assert fields["amplitude"].default == 10.0
    assert fields["cycles"].kind is int
    assert fields["cycles"].default == 2
    assert fields["wait_for_return"].kind is bool
    assert fields["wait_for_return"].default is True


def test_an_unknown_strategy_names_the_installed_ones(swing):
    with pytest.raises(KeyError, match="Zainstalowane: .*swing"):
        load_strategy("pi05")


def test_two_strategies_with_one_name_are_refused(swing, install_dist):
    install_dist("another-swing", {GROUP: {"swing": "another_swing:Swing"}})

    with pytest.raises(ValueError, match="Dwie strategie o nazwie 'swing'"):
        strategies()


# --------------------------------------------------------------------------- strict settings


def test_an_unknown_key_is_refused(swing):
    plugin = load_strategy(swing)

    with pytest.raises(ValueError, match=r"Nieznane klucze w sekcji swing: \['amplitud'\]"):
        plugin.create({"joint": "j1", "amplitud": 5.0})


def test_a_missing_key_is_refused(swing):
    plugin = load_strategy(swing)

    with pytest.raises(ValueError, match=r"Brak kluczy w sekcji swing: \['joint'\]"):
        plugin.create({"amplitude": 5.0})


@pytest.mark.parametrize("key, value", [
    ("joint", 3),
    ("amplitude", "15"),
    ("amplitude", True),
    ("amplitude", math.nan),
    ("cycles", 2.5),
    ("wait_for_return", 1),
])
def test_a_value_of_another_type_is_refused(swing, key, value):
    plugin = load_strategy(swing)
    raw = {"joint": "j1", key: value}

    with pytest.raises(ValueError, match=f"swing.{key}: potrzeba"):
        plugin.create(raw)


def test_values_take_their_fields_types(swing):
    """The panel's number field gives floats, YAML gives ints for whole numbers."""
    settings = load_strategy(swing).build_settings({"joint": "j2", "amplitude": 15, "cycles": 3.0})

    assert settings.joint == "j2"
    assert settings.amplitude == 15.0
    assert isinstance(settings.amplitude, float)
    assert settings.cycles == 3
    assert isinstance(settings.cycles, int)
    assert settings.wait_for_return is True


def test_the_strategy_is_created_with_its_settings(swing):
    strategy = load_strategy(swing).create({"joint": "j1", "period_s": 2.0, "cycles": 3})

    assert strategy.config.joint == "j1"
    assert strategy.time_limit_s == 2.0 * 3 + strategy.return_s


STILL = '''
class Strategy:
    """Holds the arm where it is."""

    hz = 10.0

    def act(self, obs):
        return {}
'''


INT_DEFAULTS = '''
from dataclasses import dataclass

@dataclass
class Settings:
    speed: float = 1

class Strategy:
    hz = 10.0
    settings = Settings

    def __init__(self, settings):
        self.config = settings

    def act(self, obs):
        return {}
'''

BAD_DEFAULT = '''
from dataclasses import dataclass

@dataclass
class Settings:
    cycles: int = "two"

class Strategy:
    hz = 10.0
    settings = Settings

    def act(self, obs):
        return {}
'''


def test_an_int_default_of_a_float_setting_is_a_float(install_dist):
    """The panel's number field keeps the type of its first value: an int would cut 1.5 to 1."""
    name = install_strategy(install_dist, "int-defaults", INT_DEFAULTS)
    speed = load_strategy(name).fields[0]

    assert speed.default == 1.0
    assert isinstance(speed.default, float)


def test_a_default_of_another_type_is_refused_when_loaded(install_dist):
    name = install_strategy(install_dist, "bad-default", BAD_DEFAULT)

    with pytest.raises(ValueError, match="cycles: potrzeba int, jest 'two'"):
        load_strategy(name)


def test_a_strategy_without_settings_is_created_without_arguments(install_dist):
    name = install_strategy(install_dist, "still-strategy", STILL)
    plugin = load_strategy(name)

    assert plugin.settings is None
    assert plugin.fields == ()
    assert plugin.create({}).hz == 10.0

    with pytest.raises(ValueError, match=r"nie ma ustawien, a dostala klucze \['speed'\]"):
        plugin.create({"speed": 1.0})


LIST_SETTING = '''
from dataclasses import dataclass, field

@dataclass
class Settings:
    poses: list = field(default_factory=list)

class Strategy:
    hz = 10.0
    settings = Settings

    def __init__(self, settings):
        self.config = settings

    def act(self, obs):
        return {}
'''

NOT_A_DATACLASS = '''
class Strategy:
    hz = 10.0
    settings = {"speed": 1.0}

    def act(self, obs):
        return {}
'''

NO_ACT = '''
class Strategy:
    hz = 10.0
'''

A_FUNCTION = '''
def Strategy(settings=None):
    return None
'''


@pytest.mark.parametrize("dist, source, problem", [
    ("list-setting", LIST_SETTING, "ustawienie poses ma typ"),
    ("dict-settings", NOT_A_DATACLASS, "`settings` musi byc klasa dataclass"),
    ("no-act", NO_ACT, "potrzeba klasy z `act"),
    ("a-function", A_FUNCTION, "potrzeba klasy z `act"),
])
def test_a_strategy_the_panel_cannot_build_is_refused_when_loaded(install_dist, dist, source, problem):
    name = install_strategy(install_dist, dist, source)

    with pytest.raises(TypeError, match=problem):
        load_strategy(name)


def test_the_config_section_named_like_the_strategy(tmp_path, monkeypatch):
    config = tmp_path / "local.yaml"
    config.write_text("swing:\n  joint: j2\n  cycles: 1\nempty:\nbroken: 3\n", encoding="utf-8")
    monkeypatch.setenv("MACHINEKIND_TWIN_CONFIG", str(config))

    assert config_section("swing") == {"joint": "j2", "cycles": 1}
    assert config_section("empty") == {}
    assert config_section("pi05") == {}

    with pytest.raises(ValueError, match="Sekcja konfiguracji broken"):
        config_section("broken")


# --------------------------------------------------------------------------- the same runner


def test_a_strategy_from_a_package_drives_the_twin_through_the_strategy_runner(swing, twin, monkeypatch):
    twin.connect("sim", threaded=False)
    centre = twin.status.measured["j1"]

    sent = []
    send = twin.set_target

    def recording_send(target, owner=None):
        sent.append(target["j1"])
        return send(target, owner)

    monkeypatch.setattr(twin, "set_target", recording_send)

    strategy = load_strategy(swing).create({"joint": "j1", "amplitude": 8.0, "period_s": 2.0,
                                             "cycles": 1})
    runner = StrategyRunner(twin, strategy)
    runner.start(threaded=False)
    assert twin.owner == "strategia"

    drive(twin, runner, 6.0)

    assert runner.status.done
    assert runner.status.success
    assert runner.status.stopped_because == "zadanie wykonane"
    assert twin.owner is None
    assert max(sent) == pytest.approx(centre + 8.0, abs=0.5)
    assert min(sent) == pytest.approx(centre - 8.0, abs=0.5)
    assert twin.status.measured["j1"] == pytest.approx(centre, abs=strategy.tolerance)


def test_a_swing_run_leaves_a_log(swing, twin, tmp_path):
    twin.connect("sim", threaded=False)
    strategy = load_strategy(swing).create({"joint": "j1", "amplitude": 8.0, "period_s": 2.0, "cycles": 1})
    runner = StrategyRunner(twin, strategy, episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 6.0)

    assert runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)
    result = report(log)
    latency = {joint.joint: joint for joint in result.latency}
    text = result.lines()

    assert log.header["strategy"]["class"] == "swing:Swing"
    assert log.end.code == "done"
    assert isinstance(latency["j1"].total, DelayFit)
    assert isinstance(latency["j1"].actuator, DelayFit)
    for joint in ("j2", "gripper"):
        assert latency[joint].total == NOT_NAMED
        assert latency[joint].actuator == STILL_JOINT
        assert f"  {joint} zamiar->odczyt: {NOT_NAMED}" in text


def test_a_joint_the_robot_does_not_have_refuses_the_run(swing, twin):
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, load_strategy(swing).create({"joint": "arm_joint1"}))

    with pytest.raises(ValueError, match="nie ma stawu 'arm_joint1'"):
        runner.start(threaded=False)

    assert twin.owner is None
    assert not runner.status.running
