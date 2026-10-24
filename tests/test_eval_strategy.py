"""`twin eval-strategy`: many simulated runs of one strategy, each with a seeded object, judged by the scene.

The physics tests run in-test strategies on the test robot, whose reach ends near r = 0.12 m, so the
object lies at r = 0.10 m unless a test says otherwise.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

from conftest import BOT, CubeScene
from test_strategy_plugins import install_strategy

from machinekind_twin.calib.simulate import look_at
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
from machinekind_twin.plugin_settings import SettingField
from machinekind_twin.scene_truth import Area, ObjectSpec
from machinekind_twin.strategies import StrategyPlugin, load_strategy
from machinekind_twin.workspace import CameraRecord, Workspace

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


def test_ctrl_c_leaves_a_partial_summary(install_dist, tmp_path):
    plugin = installed(install_dist, "eval-interrupted", INTERRUPTED)

    result = run(plugin, tmp_path / "out", episodes=3)
    summary = summary_of(tmp_path / "out")
    end = EpisodeLog.summary(tmp_path / "out" / "001.jsonl").end

    assert result.interrupted
    assert (summary["interrupted"], summary["finished"], len(summary["episodes"])) == (True, False, 1)
    assert (end.code, end.reason) == ("stopped", "przerwana przez operatora")
    assert not (tmp_path / "out" / "002.jsonl").exists()


# --------------------------------------------------------------------------- poses from a detector


@pytest.mark.render
def test_the_cube_detector_feeds_the_strategy_through_perception(cube_example, install_dist, tmp_path):
    workspace = Workspace(robot=BOT)
    for name, eye in CubeScene.EYES:
        pose = look_at(eye, (0.2, 0.0, 0.0)).tolist()
        workspace.add_camera(CameraRecord(name, "sim", 640, 480, K=[list(row) for row in CubeScene.K],
                                          sim_pose=pose, T_cam2base=pose, intrinsics_from="symulacja",
                                          calibration={"trusted": True}))
    detector = load_detector(cube_example)
    plan = campaign(installed(install_dist, "eval-sees", SEES), radius=(0.2, 0.2), time_limit_s=2.0,
                    detector=detector, detector_settings=plugin_values(detector, {}))

    row = run_campaign(workspace, plan, tmp_path / "out").rows[0]
    seen = [step["objects"]["cube"] for step in EpisodeLog.load(tmp_path / "out" / "000.jsonl").steps]

    assert row.claim is True
    assert seen[-1]["source"] == "kamery"
    assert np.linalg.norm(np.asarray(seen[-1]["position"]) - [0.2, 0.0, 0.015]) < 0.005
    assert summary_of(tmp_path / "out")["conditions"]["poses"] == "detektor cube"
