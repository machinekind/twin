"""The grasp run modes and the sim evaluation, without hardware.

Refusals before anything connects, the in-memory A1X workspace, the literal "sim" backend of the
evaluation, one sim episode through `main_grasp`, Ctrl-C during connect, the first-frame wait, the
consent prompt and pre-positioning, an Enter just after an ending, the dry run's dumps (also on the
fake A1X bus, where it sends no frame), the evaluation's cube placement and, with the checkpoint,
six sim episodes. Episodes of the scripted policy are cut to a few ticks to keep the file short.
"""

from __future__ import annotations

import _thread
import argparse
import io
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from lerobot_mp.control.safety import SafetyState  # noqa: E402
from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin.grasp import driver as driver_mod  # noqa: E402
from lerobot_mp.twin.grasp import evaluate as ev  # noqa: E402
from lerobot_mp.twin.grasp import motion  # noqa: E402
from lerobot_mp.twin.grasp import policy as policy_mod  # noqa: E402
from lerobot_mp.twin.grasp import run  # noqa: E402
from lerobot_mp.twin.grasp.contract import (  # noqa: E402
    HOVER,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
    TRAIN_J1_DEG,
    TRAIN_RADIUS,
)
from lerobot_mp.twin.grasp.driver import GraspDriver  # noqa: E402
from lerobot_mp.twin.grasp.hover import solve_hover  # noqa: E402
from lerobot_mp.twin.grasp.keys import Keys  # noqa: E402
from lerobot_mp.twin.grasp.policy import GraspStudentPolicy, find_checkpoint  # noqa: E402
from lerobot_mp.twin.grasp.rig import Rig  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics  # noqa: E402
from lerobot_mp.twin.robots import get_spec  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

FIX = Path(os.environ.get("GRASP_FIXTURES", Path(__file__).parent / "fixtures" / "grasp"))
CKPT = find_checkpoint()
SHORT_TICKS = 10


class ScriptedPolicy(GraspStudentPolicy):
    """No weights: the same action every tick. `inputs` is the real one."""

    def __init__(self, action=(0.0, 0.0, 0.0, 0.0, 0.0)) -> None:     # no network to build
        self.action = np.asarray(action, np.float32)
        self.obs_mean = np.zeros(19, np.float32)
        self.obs_std = np.ones(19, np.float32)
        self.path = Path("scripted.pt")
        self.sha256 = None
        self.schema = {}
        self.calls = 0

    def act(self, image, obs, uv):
        self.calls += 1
        return self.action.copy()


class ShortDriver(GraspDriver):
    def __init__(self, *args, **kw):
        kw.setdefault("max_ticks", SHORT_TICKS)
        super().__init__(*args, **kw)


class FakeTTY:
    """stdin of a terminal: lines come when the test pushes them; "" is end of file."""

    def __init__(self) -> None:
        self.lines: queue.Queue[str] = queue.Queue()

    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        return self.lines.get()

    def push(self, line: str) -> None:
        self.lines.put(line)


class Watch(io.TextIOBase):
    """stdout that lets a test wait for a prompt."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.cond = threading.Condition()

    def write(self, s: str) -> int:
        with self.cond:
            self.text.append(s)
            self.cond.notify_all()
        return len(s)

    def flush(self) -> None:
        pass

    def wait_for(self, needle: str, timeout: float) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: needle in "".join(self.text), timeout)


def grasp_args(**kw) -> argparse.Namespace:
    base = dict(mode="dry", backend=None, port=None, i_know=False, checkpoint=None, rig=None, workspace=None,
                camera=None, episodes=1, seed=7, outside_training=False, floor="bench", gap=None,
                jaw_stroke_s=None, dump_every=20, seconds=None, object_height=None)
    base.update(kw)
    return argparse.Namespace(**base)


def eval_args(**kw) -> argparse.Namespace:
    base = dict(episodes=2, seed=50000, jaw="sim", stroke_s=None, gap=None, floor="bench", jitter=False,
                checkpoint=None, rig=None, out=None, workspace=None)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def hermetic(monkeypatch, tmp_path):
    """Outputs under tmp_path, no default workspace file, the per-rig file in tmp_path."""
    monkeypatch.setattr(run, "out_dir", lambda kind: tmp_path / "out" / kind)
    monkeypatch.setattr(motion, "default_path", lambda: tmp_path / "no-such-twin.json")

    def no_save(self, *a, **k):
        raise AssertionError("a grasp command saved a file it must not write")

    monkeypatch.setattr(Workspace, "save", no_save)
    monkeypatch.setattr(Rig, "save", no_save)
    return tmp_path


@pytest.fixture
def scripted(monkeypatch, hermetic):
    """The scripted policy instead of the checkpoint, and episodes of a few ticks."""
    pol = ScriptedPolicy()
    monkeypatch.setattr(policy_mod, "find_checkpoint", lambda explicit=None: Path("scripted.pt"))
    monkeypatch.setattr(GraspStudentPolicy, "load", classmethod(lambda cls, path=None, **kw: pol))
    monkeypatch.setattr(driver_mod, "GraspDriver", ShortDriver)
    return pol


@pytest.fixture
def no_connect(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("connected although the run should have refused first")

    monkeypatch.setattr(rt.Twin, "connect", fail)
    monkeypatch.setattr(rt, "create_backend", fail)


def _move_to_hover(s: run.Session) -> None:
    """Test shortcut: the sim arm to the hover, so pre-positioning does not start from home."""
    s.twin.move(s.hover.joints, 2.0, owner="test")
    s.twin.release("test")


def _above_the_cubes(twin: rt.Twin) -> None:
    """Test shortcut: the sim arm above the middle of training's region instead of at home.

    From home the first pre-positioning at 20 deg/s takes about 10 s; from here a few.
    """
    kin = RobotKinematics(twin.spec)
    believed = np.array([0.0, 0.31, TOUCH_EXPECTED_Z + TIP_CLEAR])
    hv = solve_hover(kin, believed, 0.0, limits=twin.joint_limits())
    twin.move(hv.joints, 2.0, owner="test")
    twin.release("test")


def _stepper(twin: rt.Twin):
    """The driver's `sleep` on a twin connected with `threaded=False`: it steps the twin by loop periods."""
    period = 1.0 / twin.loop_hz

    def sleep(dt: float) -> None:
        t_end = twin._clock + dt
        twin.step(min(period, max(dt, 1e-3)))
        while twin._clock < t_end - 1e-9:
            twin.step(min(period, t_end - twin._clock))

    return sleep


# ---------------------------------------------------------------------- refusals before connecting
@pytest.mark.parametrize("kw, needle", [
    (dict(mode="real"), "--i-know"),
    (dict(mode="shadow"), "--i-know"),
    (dict(mode="shadow", i_know=True), "--backend"),
    (dict(mode="real", i_know=True), "--backend"),
    (dict(mode="dry", floor="training"), "--floor training"),
    (dict(mode="real", i_know=True, backend="a1x", floor="training"), "--floor training"),
    (dict(mode="dry", jaw_stroke_s=1.3), "--jaw-stroke-s"),
    (dict(mode="sim", jaw_stroke_s=0.0), "--jaw-stroke-s"),
    (dict(mode="shadow", i_know=True, backend="a1x", gap="scene"), "--gap scene"),
    (dict(mode="sim", seconds=1.0), "--seconds"),
    (dict(mode="sim", object_height=0.03), "--object-height"),
    (dict(mode="sim", dump_every=5), "--dump-every"),
    (dict(mode="dry", object_height=5.0), "--object-height"),
    (dict(mode="dry", dump_every=0), "--dump-every"),
    (dict(mode="sim", episodes=0), "--episodes"),
    (dict(mode="sim", episodes=-1), "--episodes"),
])
def test_flag_refusals_return_2_before_anything_connects(kw, needle, monkeypatch, capsys, no_connect):
    def fail(*a, **k):
        raise AssertionError("the checkpoint was looked for although the flags refuse")

    monkeypatch.setattr(policy_mod, "find_checkpoint", fail)
    assert run.main_grasp(grasp_args(**kw)) == 2
    assert needle in capsys.readouterr().err


@pytest.mark.parametrize("kw, needle", [
    (dict(jaw="sim", stroke_s=1.3), "--stroke-s"),
    (dict(jaw="measured", stroke_s=-1.0), "--stroke-s"),
    (dict(episodes=0), "--episodes"),
    (dict(episodes=-2), "--episodes"),
])
def test_evaluation_flags_refuse_before_anything_connects(kw, needle, monkeypatch, capsys, no_connect,
                                                          hermetic):
    def fail(*a, **k):
        raise AssertionError("the checkpoint was looked for although the flags refuse")

    monkeypatch.setattr(policy_mod, "find_checkpoint", fail)
    assert ev.main_eval(eval_args(**kw)) == 2
    assert needle in capsys.readouterr().err
    assert not list(hermetic.rglob("*.json"))


def test_a_missing_checkpoint_names_the_fetch_command(monkeypatch, capsys, hermetic, no_connect):
    monkeypatch.setattr(policy_mod, "find_checkpoint", lambda explicit=None: None)
    assert run.main_grasp(grasp_args(mode="sim")) == 2
    assert "lerobot-twin grasp-model fetch" in capsys.readouterr().err
    assert ev.main_eval(eval_args()) == 2
    assert "lerobot-twin grasp-model fetch" in capsys.readouterr().err


def test_a_workspace_file_with_another_robot_refuses_not_a1x(capsys, scripted, no_connect, tmp_path):
    path = tmp_path / "twin.json"
    path.write_text(json.dumps({"robot": "so101"}), encoding="utf-8")
    assert run.main_grasp(grasp_args(mode="sim", workspace=str(path))) == 2
    assert "not a1x" in capsys.readouterr().err
    assert ev.main_eval(eval_args(workspace=str(path))) == 2
    assert "not a1x" in capsys.readouterr().err


def test_enable_on_connect_on_a_real_backend_refuses_before_connect(monkeypatch, capsys, scripted, no_connect,
                                                                    tmp_path):
    cfg = tmp_path / "local.yaml"
    cfg.write_text("a1x:\n  enable_on_connect: true\n", encoding="utf-8")
    monkeypatch.setenv("LEROBOT_MP_CONFIG", str(cfg))
    built = []
    orig_init = rt.Twin.__init__

    def init(self, *a, **k):
        built.append(1)
        orig_init(self, *a, **k)

    monkeypatch.setattr(rt.Twin, "__init__", init)
    assert run.main_grasp(grasp_args(mode="dry", backend="a1x", rig=str(tmp_path / "grasp.json"))) == 2
    assert "enable on connect" in capsys.readouterr().err
    assert built == []


def test_sim_mode_on_a_real_backend_refuses_not_armed_before_connect(capsys, scripted, no_connect, tmp_path):
    assert run.main_grasp(grasp_args(mode="sim", backend="a1x", rig=str(tmp_path / "grasp.json"))) == 2
    assert "not armed" in capsys.readouterr().err


# ---------------------------------------------------------------------- checks with the workspace and backend
def _fake_twin(*, simulated=False, offsets=(0.0,) * 6, force=1.2):
    backend = SimpleNamespace(acfg=SimpleNamespace(joint_offsets_deg=tuple(offsets)),
                              gripper_state=lambda: SimpleNamespace(force_threshold=force))
    return SimpleNamespace(status=SimpleNamespace(simulated=simulated), backend=backend)


def _taught_rig(**grasp) -> Rig:
    return Rig(table={"touch_tcp_z": 0.034, "tcp": [0.0, 0.33, 0.034]},
               grasp={"tcp": [0.0, 0.33, 0.04], "yaw": 0.0, "joint_offsets_deg": [0.0] * 6, **grasp},
               gripper={"freeze_threshold": 100.0})


def test_site_refusals_on_a_real_backend(capsys):
    training_table = {"size": [1.2, 1.2], "height": 0.75, "base_xy": [0.0, -0.20], "base_yaw": 0.0}
    ws = Workspace(robot="a1x", table=training_table)
    rig = _taught_rig()
    assert run.site_refusals(ws, rig, _fake_twin(force=100.0), "real") == []
    assert run.site_refusals(ws, rig, _fake_twin(simulated=True, offsets=(5.0,) * 6), "real") == []
    # The default workspace table ends at y = 0.30 m, inside the training region.
    assert run.site_refusals(Workspace(robot="a1x"), rig, _fake_twin(force=100.0), "dry") == \
        ["table not under grasp"]
    assert run.site_refusals(ws, _taught_rig(joint_offsets_deg=None), _fake_twin(force=100.0), "dry") == \
        ["joint offsets changed"]
    j3 = (0.0, 0.0, 1.49, 0.0, 0.0, 0.0)
    rig_j3 = _taught_rig(joint_offsets_deg=list(j3))
    assert run.site_refusals(ws, rig_j3, _fake_twin(offsets=j3, force=100.0), "shadow") == \
        ["joint offsets differ"]
    assert run.site_refusals(ws, rig, _fake_twin(offsets=j3, force=100.0), "shadow") == \
        ["joint offsets changed", "joint offsets differ"]
    rig_j3.overrides["joint_offsets"] = "operator, J3 zero measured"
    capsys.readouterr()
    assert run.site_refusals(ws, rig_j3, _fake_twin(offsets=j3, force=100.0), "shadow") == []
    assert "UWAGA" in capsys.readouterr().err
    assert run.site_refusals(ws, rig, _fake_twin(force=1.2), "real") == ["gripper freeze differs"]
    assert run.site_refusals(ws, rig, _fake_twin(force=1.2), "shadow") == []
    assert run.site_refusals(ws, rig, _fake_twin(force=None), "real") == ["gripper freeze differs"]


def test_on_the_fake_a1x_bus_site_refusals_read_the_backend_and_a_dry_run_sends_no_frame(
        monkeypatch, tmp_path):
    pytest.importorskip("galaxeo")
    from test_twin_grasp_driver_a1x import STILL, segments
    from test_twin_grasp_driver_a1x import Rig as A1XRig

    j3 = [0.0, 0.0, 1.49, 0.0, 0.0, 0.0]
    cfg = tmp_path / "local.yaml"
    cfg.write_text(f"a1x:\n  joint_offsets_deg: {j3}\n  gripper_force: 100.0\n", encoding="utf-8")
    monkeypatch.setenv("LEROBOT_MP_CONFIG", str(cfg))
    a = A1XRig(monkeypatch)
    try:
        assert not a.twin.status.simulated
        ws = Workspace(robot="a1x", table={"size": [1.2, 1.2], "height": 0.75, "base_xy": [0.0, -0.20],
                                           "base_yaw": 0.0})
        taught = _taught_rig()
        assert run.site_refusals(ws, taught, a.twin, "real") == \
            ["joint offsets changed", "joint offsets differ"]
        taught.grasp["joint_offsets_deg"] = j3
        taught.overrides["joint_offsets"] = "operator, J3 zero measured"
        assert run.site_refusals(ws, taught, a.twin, "real") == []
        taught.gripper["freeze_threshold"] = 1.2
        assert run.site_refusals(ws, taught, a.twin, "real") == ["gripper freeze differs"]

        drv = a.driver(segments((1, STILL)), mode="dry")
        assert drv.check() is None
        a.pump(0.2)                                       # joint samples for the listener
        s = run.Session(twin=a.twin, driver=drv, policy=drv.policy, frames=drv.frames, rig=Rig.for_sim(),
                        mode="dry", camera=drv.camera, hover=a.hover, simulated=False)
        orig_close, closes = a.twin.close, []

        def close():
            if closes:
                return
            closes.append(1)
            if a.twin.connected:
                a.pump(0.5)                               # the loop runs on after the shutdown's hold
            orig_close()

        monkeypatch.setattr(a.twin, "close", close)
        out = tmp_path / "dry"
        keys = Keys(io.StringIO(""))
        assert run.dry_run(s, keys, seconds=0.3, dump_every=2, out=out, object_height=0.03) == 0
        run._shutdown(s.twin, s.driver, s.frames, stopped=False)
        assert a.bus.tx == [] and a.twin.owner is None and not a.twin.connected
        ticks = [json.loads(line) for line in (out / "ticks.jsonl").read_text(encoding="utf-8").splitlines()]
        assert ticks and all(t["kind"] == "policy" for t in ticks)
        assert (out / "0000.png").is_file() and (out / "0000_heat.png").is_file()
    finally:
        a.close()


def test_the_camera_flag_stands_in_for_a_wrist_source_the_per_rig_file_lacks():
    rig = _taught_rig()
    assert "wrist camera not set" in run.rig_refusals(rig, "dry", False)
    assert run.rig_refusals(rig, "dry", False, camera="/dev/video2") == []
    assert run.rig_refusals(rig, "dry", True) == []


def test_the_real_warning_is_polish_ascii_and_says_how_to_stop_and_go_on():
    text = run.REAL_WARNING.format(mode="real", mode_note="x", table_age="1 h 2 min", grasp_age="3 min")
    assert text.isascii() and run.CONFIRM_PROMPT.isascii() and run.PROCEED_PROMPT.isascii()
    assert "limity" in text and "Sam Enter ZAWSZE zatrzymuje" in text and "d i Enter" in text
    assert "1 h 2 min" in text and "3 min" in text


# ---------------------------------------------------------------------- the sim evaluation's pieces
def test_drawn_cubes_lie_in_training_region_and_their_hover_inside_the_envelope():
    rng = np.random.default_rng(1)
    draws = [ev.draw_cube(rng) for _ in range(200)]
    for xy, yaw in draws:
        assert -0.20 <= xy[0] <= 0.20 and 0.22 <= xy[1] <= 0.40
        assert -math.pi / 4 <= yaw <= math.pi / 4
        assert TRAIN_J1_DEG[0] <= math.degrees(math.atan2(xy[1], xy[0])) <= TRAIN_J1_DEG[1]
        assert TRAIN_RADIUS[0] <= math.hypot(*xy) <= TRAIN_RADIUS[1]
    # Joint 1 of the solved top-down hover is the bearing that `draw_cube` judges by.
    spec = get_spec("a1x")
    kin = RobotKinematics(spec)
    limits = {n: (-170.0, 170.0) for n in spec.arm_joints}
    for xy, yaw in draws[:3]:
        hv = solve_hover(kin, np.array([xy[0], xy[1], TOUCH_EXPECTED_Z + TIP_CLEAR]), yaw, limits=limits)
        assert abs(hv.joints["arm_joint1"] - math.degrees(math.atan2(xy[1], xy[0]))) < 0.05
        assert hv.outside_training == []
        assert abs(hv.p[2] - (TOUCH_EXPECTED_Z + TIP_CLEAR + HOVER)) < 1e-12


def test_the_image_jitter_stays_inside_trainings_half_ranges():
    rng = np.random.default_rng(3)
    draws = [ev._draw_jitter(rng) for _ in range(500)]
    assert max(float(np.max(np.abs(j.dpos))) for j in draws) <= 0.003
    assert max(float(np.max(np.abs(np.degrees(j.rotvec)))) for j in draws) <= 0.7
    assert max(abs(j.dfovy_deg) for j in draws) <= 0.3
    assert max(float(np.max(np.abs(j.dpos))) for j in draws) > 0.0025          # the whole range is used


def test_episode_logs_go_beside_the_result_file_never_onto_it(tmp_path):
    assert ev.episode_log_dir(tmp_path / "eval.json") == tmp_path / "eval"
    assert ev.episode_log_dir(tmp_path / "eval") == tmp_path / "eval_episodes"


def test_verdicts_below_half_name_three_causes_and_the_slowed_jaw_asks_for_retraining():
    def res(rate, jaw):
        return ev.EvalResult(episodes=10, successes=int(rate * 10), rate=rate, endings={},
                             tick_ms={"p99": 12.0}, infer_ms={}, rate_limited_frac=0.0, ik_refused_frac=0.0,
                             collision_refused_frac=0.0, config={"jaw": jaw, "stroke_s": 1.3})

    low = "\n".join(ev.verdicts(res(0.4, "sim")))
    assert ("ponizej 50 %: przyczyna moze byc sciezka obserwacji, fizyka kontaktu albo render sceny blizniaka"
            in low)
    assert "--floor training --gap scene" in low and "test_twin_grasp_policy.py" in low
    assert "dobrze" in "\n".join(ev.verdicts(res(0.8, "sim")))
    assert "ponowny trening" in "\n".join(ev.verdicts(res(0.4, "measured")))
    assert "ponowny trening" not in "\n".join(ev.verdicts(res(0.6, "measured")))
    assert all(line.isascii() for r in (0.1, 0.6, 0.9) for line in ev.verdicts(res(r, "sim")))


# ---------------------------------------------------------------------- whole runs on the sim backend
def test_without_a_workspace_file_the_run_uses_an_in_memory_a1x_and_writes_no_file(monkeypatch, scripted,
                                                                                  tmp_path):
    seen = []
    orig_connect = rt.Twin.connect

    def connect(self, backend=None, port=None, go_home=False, *, threaded=True):
        seen.append((self.workspace.robot, self.workspace.path, backend))
        return orig_connect(self, backend, port, go_home, threaded=threaded)

    monkeypatch.setattr(rt.Twin, "connect", connect)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert run.main_grasp(grasp_args(mode="dry", seconds=0.3, rig=str(tmp_path / "grasp.json"))) == 0
    assert seen == [("a1x", None, "sim")]
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    written = [p for p in after if p not in before]
    assert all(p.parts[0] == "out" for p in written), written
    assert not (tmp_path / "no-such-twin.json").exists() and not (tmp_path / "grasp.json").exists()


def test_main_eval_connects_the_literal_sim_backend_and_places_the_cube_where_drawn(monkeypatch, scripted,
                                                                                  tmp_path):
    ws_file = tmp_path / "twin.json"
    ws_file.write_text(json.dumps({"robot": "a1x", "backend": "a1x", "port": "can0"}), encoding="utf-8")

    def no_bus(*a, **k):
        raise AssertionError("the evaluation opened a backend other than the sim")

    monkeypatch.setattr(rt, "create_backend", no_bus)
    connects = []
    orig_connect = rt.Twin.connect

    def connect(self, backend=None, port=None, go_home=False, *, threaded=True):
        connects.append(backend)
        orig_connect(self, backend, port, go_home, threaded=threaded)
        _above_the_cubes(self)

    monkeypatch.setattr(rt.Twin, "connect", connect)
    placed = []
    orig_place = ev.place_cube

    def place(twin, xy, yaw, **kw):
        centre = orig_place(twin, xy, yaw, **kw)
        got, got_yaw = ev.cube_pose(twin)
        placed.append((np.asarray(xy, float), float(yaw), got, got_yaw, twin.version))
        return centre

    monkeypatch.setattr(ev, "place_cube", place)
    out = tmp_path / "eval.json"
    assert ev.main_eval(eval_args(episodes=2, workspace=str(ws_file), out=str(out))) == 0
    assert connects == ["sim"]

    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["episodes"] == 2 and res["config"]["backend"] == "sim" and res["config"]["jaw"] == "sim"
    assert res["config"]["gap"] == "scene" and res["config"]["seed"] == 50000
    assert sum(res["endings"].values()) == 2 and len(res["per_episode"]) == 2
    assert all(Path(e["log"]).is_file() for e in res["per_episode"])

    # At the start of episode 2 the cube is where it was drawn, in a scene that was not rebuilt.
    assert len(placed) == 2
    xy, yaw, got, got_yaw, version = placed[1]
    assert np.hypot(*(got[:2] - xy)) < 0.002
    d = (math.degrees(got_yaw - yaw) + 45.0) % 90.0 - 45.0               # a cube looks the same every 90 deg
    assert abs(d) < 2.0
    assert version == placed[0][4]


def test_sim_mode_runs_one_episode_and_closes(monkeypatch, scripted, tmp_path, capsys):
    if CKPT is not None:
        monkeypatch.setattr(GraspStudentPolicy, "load", classmethod(_real_load))
        monkeypatch.setattr(policy_mod, "find_checkpoint", find_checkpoint)
    twins = []
    orig_open = run.open_session

    def opened(a, keys):
        s = orig_open(a, keys)
        if isinstance(s, run.Session):
            twins.append(s.twin)
            _move_to_hover(s)
        return s

    monkeypatch.setattr(run, "open_session", opened)
    code = run.main_grasp(grasp_args(mode="sim", episodes=1, rig=str(tmp_path / "grasp.json")))
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Epizod 1/1" in out
    tw = twins[0]
    assert not tw.connected and tw.owner is None
    logs = list((tmp_path / "out" / "runs").glob("episode_*.jsonl"))
    assert len(logs) == 1 and logs[0].read_text(encoding="utf-8").strip()


_ORIG_LOAD = GraspStudentPolicy.load.__func__


def _real_load(cls, path=None, **kw):
    return _ORIG_LOAD(GraspStudentPolicy, path, **kw)


def test_ctrl_c_during_prepositioning_ends_with_stop_hold_and_close(monkeypatch, scripted, tmp_path, capsys):
    started, returned = threading.Event(), []
    orig_pre = GraspDriver.preposition

    def preposition(self, hover, *a, **k):
        started.set()
        why = orig_pre(self, hover, *a, **k)
        returned.append(why)
        return why

    monkeypatch.setattr(GraspDriver, "preposition", preposition)
    stops = []
    orig_stop_now = motion.stop_now

    def stop_now(twin, reason):
        stops.append(reason)
        orig_stop_now(twin, reason)

    monkeypatch.setattr(motion, "stop_now", stop_now)
    closed = []
    orig_close = rt.Twin.close

    def close(self):
        closed.append((self.safety_state, self.connected))
        orig_close(self)

    monkeypatch.setattr(rt.Twin, "close", close)

    def ctrl_c():
        if started.wait(60.0):
            time.sleep(0.3)
            if not returned:
                _thread.interrupt_main()

    threading.Thread(target=ctrl_c, daemon=True).start()
    code = run.main_grasp(grasp_args(mode="sim", episodes=2, rig=str(tmp_path / "grasp.json")))
    assert code == 1
    assert stops == [run.STOP_CTRL_C]
    assert returned and returned[0] in ("stop", "preempted")           # the move ended, it did not run on
    assert closed == [(SafetyState.ESTOP, True)]                      # held by the STOP, then closed
    assert "Ctrl-C" in capsys.readouterr().err


def _record_stops_and_closes(monkeypatch) -> tuple[list, list]:
    stops, closed = [], []
    orig_stop_now, orig_close = motion.stop_now, rt.Twin.close

    def stop_now(twin, reason):
        stops.append(reason)
        orig_stop_now(twin, reason)

    def close(self):
        closed.append((self.safety_state, self.connected))
        orig_close(self)

    monkeypatch.setattr(motion, "stop_now", stop_now)
    monkeypatch.setattr(rt.Twin, "close", close)
    return stops, closed


@pytest.mark.parametrize("where", ["connect", "first frame"])
def test_ctrl_c_during_connect_or_the_first_frame_wait_stops_and_closes(
        where, monkeypatch, scripted, tmp_path, capsys):
    from lerobot_mp.twin.grasp import camera as camera_mod

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    if where == "connect":
        orig_connect = rt.Twin.connect

        def connect(self, *a, **k):
            orig_connect(self, *a, **k)
            interrupt()

        monkeypatch.setattr(rt.Twin, "connect", connect)
    else:
        monkeypatch.setattr(camera_mod.SimWristSource, "latest", interrupt)
    stops, closed = _record_stops_and_closes(monkeypatch)
    assert run.main_grasp(grasp_args(mode="sim", rig=str(tmp_path / "grasp.json"))) == 1
    assert stops == [run.STOP_CTRL_C]
    assert closed == [(SafetyState.ESTOP, True)]
    assert "Ctrl-C" in capsys.readouterr().err


def test_ctrl_c_at_the_consent_prompt_stops_before_anything_moves(monkeypatch, scripted, tmp_path):
    sessions = []
    orig_open = run.open_session

    def opened(a, keys):
        s = orig_open(a, keys)
        if isinstance(s, run.Session):
            s.simulated = False              # the real-backend path of the series on the simulated arm
            sessions.append(s)
        return s

    monkeypatch.setattr(run, "open_session", opened)
    chwyt_targets = []
    orig_set_target = rt.Twin.set_target

    def set_target(self, joints, owner=None):
        if owner == "chwyt":
            chwyt_targets.append(dict(joints))
        return orig_set_target(self, joints, owner)

    monkeypatch.setattr(rt.Twin, "set_target", set_target)
    stops, closed = _record_stops_and_closes(monkeypatch)
    watch, tty = Watch(), FakeTTY()
    monkeypatch.setattr(sys, "stdout", watch)
    monkeypatch.setattr(sys, "stdin", tty)

    def operator():
        if watch.wait_for(run.CONFIRM_PROMPT, 30.0):
            time.sleep(0.2)
            _thread.interrupt_main()

    threading.Thread(target=operator, daemon=True).start()
    try:
        code = run.main_grasp(grasp_args(mode="real", backend="sim", i_know=True, episodes=2,
                                         rig=str(tmp_path / "grasp.json")))
    finally:
        tty.push("")
    assert code == 1
    assert "UWAGA" in "".join(watch.text)
    assert stops == [run.STOP_CTRL_C]
    assert closed == [(SafetyState.ESTOP, True)]
    assert chwyt_targets == [] and sessions and sessions[0].driver.arm.gen is None


def test_a_silent_wrist_camera_ends_with_1_before_anything_moves(monkeypatch, scripted, tmp_path, capsys):
    from lerobot_mp.twin.grasp import camera as camera_mod

    monkeypatch.setattr(camera_mod.SimWristSource, "latest", lambda self: None)
    monkeypatch.setattr(run, "FIRST_FRAME_S", 0.3)
    assert run.main_grasp(grasp_args(mode="sim", rig=str(tmp_path / "grasp.json"))) == 1
    assert "wrist camera silent" in capsys.readouterr().err


def test_a_hover_outside_training_refuses_unless_overridden(monkeypatch, scripted, tmp_path, capsys):
    spec = get_spec("a1x")
    front = np.array([0.33, 0.0, TOUCH_EXPECTED_Z + TIP_CLEAR])                 # in front of the arm
    hv = solve_hover(RobotKinematics(spec), front, 0.0, limits=spec.limits)
    assert hv.outside_training
    monkeypatch.setattr(ev, "draw_hover", lambda *a, **k: ev.Draw(front[:2], 0.0, hv, 0))
    keys = Keys(io.StringIO(""))
    args = grasp_args(mode="sim", rig=str(tmp_path / "grasp.json"))
    assert run.open_session(args, keys) == 2
    assert "outside training" in capsys.readouterr().err
    args = grasp_args(mode="sim", rig=str(tmp_path / "grasp.json"), outside_training=True)
    s = run.open_session(args, keys)
    assert isinstance(s, run.Session)
    run._shutdown(s.twin, s.driver, s.frames, stopped=False)
    assert "UWAGA" in capsys.readouterr().err


def test_a_workspace_reach_volume_that_misses_the_cubes_refuses_outside_volume(monkeypatch, scripted,
                                                                              tmp_path, capsys):
    ws = tmp_path / "twin.json"
    # The A1X's default reach box starts at x = 0.20 m: training's region lies beside it.
    reach = {"lo": [0.20, -0.30, 0.05], "hi": [0.50, 0.30, 0.30]}
    ws.write_text(json.dumps({"robot": "a1x", "reach": reach}), encoding="utf-8")
    takes = []
    monkeypatch.setattr(GraspDriver, "take", lambda self: takes.append(1) or "busy")
    assert run.main_grasp(grasp_args(mode="sim", workspace=str(ws), rig=str(tmp_path / "grasp.json"))) == 2
    assert "outside volume" in capsys.readouterr().err and takes == []


def test_an_enter_50_ms_after_a_tick_cap_ending_ends_the_series_and_no_set_target_follows(
        monkeypatch, scripted, tmp_path):
    ended, reports = threading.Event(), []
    t_end: list[float] = []
    orig_episode = GraspDriver.episode

    def episode(self, hover):
        rep = orig_episode(self, hover)
        t_end.append(time.monotonic())
        reports.append(rep)
        ended.set()
        return rep

    monkeypatch.setattr(GraspDriver, "episode", episode)
    targets: list[float] = []
    orig_set_target = rt.Twin.set_target

    def set_target(self, joints, owner=None):
        targets.append(time.monotonic())
        return orig_set_target(self, joints, owner)

    monkeypatch.setattr(rt.Twin, "set_target", set_target)
    watch = Watch()
    monkeypatch.setattr(sys, "stdout", watch)
    tty = FakeTTY()
    keys = Keys(tty)
    keys.start()

    s = run.open_session(grasp_args(mode="real", backend="sim", i_know=True, episodes=3,
                                    rig=str(tmp_path / "grasp.json")), keys)
    assert isinstance(s, run.Session)
    try:
        _move_to_hover(s)
        # The real-backend path of the series (consent, prompts between episodes) on the simulated arm.
        s.simulated = False

        def operator():
            if watch.wait_for(run.CONFIRM_PROMPT, 30.0):
                time.sleep(0.1)
                tty.push("\n")
            if ended.wait(60.0):
                time.sleep(0.05)
                tty.push("\n")

        threading.Thread(target=operator, daemon=True).start()
        code = run.run_series(s, keys, 3)
    finally:
        run._shutdown(s.twin, s.driver, s.frames, stopped=s.stopped, worker=s.worker)
        tty.push("")
    assert [r.ending for r in reports] == ["tick cap"]
    assert code == 0
    assert keys.stop.is_set() and s.stopped
    assert targets and max(targets) <= t_end[0]


@pytest.mark.parametrize("during", ["preposition", "episode"])
def test_a_bare_enter_while_the_arm_moves_ends_the_series_with_1(during, monkeypatch, scripted, tmp_path):
    keys = Keys(io.StringIO(""))
    s = run.open_session(grasp_args(mode="sim", episodes=3, rig=str(tmp_path / "grasp.json")), keys)
    assert isinstance(s, run.Session)
    episodes = []

    def preposition(self, hover, *a, **k):
        if during == "preposition":
            keys.stop.set()
        return None

    def episode(self, hover):
        episodes.append(hover)
        keys.stop.set()
        return self._empty_report("tick cap", "")

    monkeypatch.setattr(GraspDriver, "preposition", preposition)
    monkeypatch.setattr(GraspDriver, "episode", episode)
    try:
        code = run.run_series(s, keys, 3)
    finally:
        run._shutdown(s.twin, s.driver, s.frames, stopped=s.stopped, worker=s.worker)
    assert code == 1 and s.stopped
    assert len(episodes) == (0 if during == "preposition" else 1)


def test_a_prepositioning_refusal_on_a_real_backend_ends_the_series_as_a_refusal(monkeypatch, scripted,
                                                                                tmp_path):
    episodes = []
    monkeypatch.setattr(GraspDriver, "preposition", lambda self, hover, *a, **k: "not arrived")
    monkeypatch.setattr(GraspDriver, "episode", lambda self, hover: episodes.append(hover))
    # Both streams are the test's own, not capsys: capsys would close the stream monkeypatch puts
    # back, and pytest -s would then end on a closed stdout.
    watch, err = Watch(), Watch()
    monkeypatch.setattr(sys, "stdout", watch)
    monkeypatch.setattr(sys, "stderr", err)
    tty = FakeTTY()
    keys = Keys(tty)
    keys.start()
    s = run.open_session(grasp_args(mode="real", backend="sim", i_know=True, episodes=3,
                                    rig=str(tmp_path / "grasp.json")), keys)
    assert isinstance(s, run.Session)
    try:
        s.simulated = False

        def operator():
            if watch.wait_for(run.CONFIRM_PROMPT, 30.0):
                time.sleep(0.1)
                tty.push("\n")

        threading.Thread(target=operator, daemon=True).start()
        code = run.run_series(s, keys, 3)
    finally:
        run._shutdown(s.twin, s.driver, s.frames, stopped=s.stopped, worker=s.worker)
        tty.push("")
    assert code == 2 and episodes == []
    assert "Epizod 1/3: odmowa przed epizodem: not arrived" in "".join(err.text)


def test_dry_run_writes_png_dumps_with_the_dot_and_the_cube_top_a_heat_png_and_ticks(monkeypatch, scripted,
                                                                                    tmp_path, capsys):
    chwyt_targets = []
    orig_set_target = rt.Twin.set_target

    def set_target(self, joints, owner=None):
        if owner == "chwyt":
            chwyt_targets.append(dict(joints))
        return orig_set_target(self, joints, owner)

    monkeypatch.setattr(rt.Twin, "set_target", set_target)
    keys = Keys(io.StringIO(""))
    s = run.open_session(grasp_args(mode="dry", rig=str(tmp_path / "grasp.json")), keys)
    assert isinstance(s, run.Session)
    out = tmp_path / "dry"
    try:
        _move_to_hover(s)
        # The arm stands at the hover through the dry run: where the dumps must draw the cube top.
        kin = RobotKinematics(s.twin.spec)
        top_uv = s.camera.cube_top_uv(kin, dict(s.twin.status.measured), s.hover.believed[:2], 0.030)
        assert run.dry_run(s, keys, seconds=0.8, dump_every=4, out=out, object_height=0.030) == 0
    finally:
        run._shutdown(s.twin, s.driver, s.frames, stopped=s.stopped, worker=s.worker)
    assert chwyt_targets == [] and s.twin.owner is None
    summary = capsys.readouterr().out
    assert "Podsumowanie dry" in summary and "wysilek ramienia" in summary
    assert "werdykt: inferencja" in summary

    ticks = [json.loads(line) for line in (out / "ticks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(ticks) >= 5 and {"tick", "uv", "action", "frame_age_s"} <= set(ticks[0])
    dumps = sorted(out.glob("[0-9][0-9][0-9][0-9].png"))
    heats = sorted(out.glob("*_heat.png"))
    assert dumps and heats and dumps[0].name == "0000.png" and len(heats) == len(dumps)

    img = cv2.imread(str(dumps[0]))                                      # BGR
    b, g, r = (img[..., k].astype(int) for k in range(3))
    assert img.shape == (480, 640, 3)

    def at_scale(uv):                                                    # policy pixel -> dump pixel
        return (float(uv[0]) + 0.5) * 4 - 0.5, (float(uv[1]) + 0.5) * 4 - 0.5

    # The cross and the circle around the believed pixel of that tick: centred on it.
    gy, gx = np.nonzero((g > 200) & (r < 90) & (b < 90))
    assert gx.size >= 10
    cx, cy = at_scale(ticks[0]["uv"])
    assert abs(gx.mean() - cx) < 3.0 and abs(gy.mean() - cy) < 3.0
    # The square of the projected cube top: its border lies 4 px (one policy pixel) around the point.
    magenta = (r > 200) & (b > 200) & (g < 90)
    x0, y0 = (round(v) for v in at_scale(top_uv))
    assert 0 <= x0 < 640 and 0 <= y0 < 480
    assert magenta[y0 - 6:y0 + 7, x0 - 6:x0 + 7].sum() >= 4
    heat = cv2.imread(str(heats[0]), cv2.IMREAD_GRAYSCALE)
    assert heat.shape == (120, 160) and heat.max() >= 200                # the dot is in the frame


# ---------------------------------------------------------------------- with the checkpoint
@pytest.mark.skipif(CKPT is None, reason="no checkpoint (LEROBOT_GRASP_STUDENT or workspace/policies/grasp)")
def test_with_the_checkpoint_six_sim_episodes_at_seed_50000_grasp_at_least_three_cubes(tmp_path, capsys):
    # The twin is stepped by the driver's own clock and sleep, as training steps its scene: the
    # count does not depend on how busy the machine is (on the real-time twin it moved between 2
    # and 4 of 6 from run to run), and six episodes take seconds instead of a minute.
    policy = GraspStudentPolicy.load(CKPT)
    twin = rt.Twin(Workspace(robot="a1x"))
    twin.connect("sim", go_home=False, threaded=False)
    try:
        res = ev.evaluate(twin, policy, episodes=6, seed=50000, jaw="sim", out=tmp_path / "ckpt.json",
                          clock=lambda: twin._clock, sleep=_stepper(twin))
    finally:
        twin.close()
    with capsys.disabled():
        print(f"\n[grasp sim evaluation] {res.successes}/{res.episodes} at seed 50000, sim jaw, "
              f"stepped twin; endings {res.endings}; tick {res.tick_ms}; infer {res.infer_ms}")
    assert res.episodes == 6 and all(Path(e["log"]).is_file() for e in res.per_episode)
    assert res.config["checkpoint_sha256"] == policy_mod.CHECKPOINT_SHA256
    assert res.successes >= 3, f"{res.successes}/6 successes: a finding, not a tolerance to loosen"
