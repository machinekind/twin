"""`lerobot-twin grasp`, `grasp-bench`, `grasp-eval` and `grasp-model`: argument parsing and dispatch.

The commands only parse arguments and hand the namespace to the `main_*` of their module. Parsing
and dispatch are checked with stand-ins for those modules, so they need neither torch nor MuJoCo.
A second group runs the real `main_*` on paths that end before an arm is connected (flag refusals,
the per-rig notes, `grasp-model check` with the checkpoint): a flag the parser names differently
from the module that reads it fails there. Nothing connects, and files go only under `tmp_path`.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from lerobot_mp.twin import cli

FIX = Path(os.environ.get("GRASP_FIXTURES", Path(__file__).parent / "fixtures" / "grasp"))
SRC = Path(__file__).resolve().parents[1] / "src"
HEAVY = ("mujoco", "torch", "cv2")

# command -> (module, function in it, handler in cli)
COMMANDS = {
    "grasp": ("lerobot_mp.twin.grasp.run", "main_grasp", "_grasp"),
    "grasp-bench": ("lerobot_mp.twin.grasp.bench", "main_bench", "_grasp_bench"),
    "grasp-eval": ("lerobot_mp.twin.grasp.evaluate", "main_eval", "_grasp_eval"),
    "grasp-model": ("lerobot_mp.twin.grasp.policy", "main_model", "_grasp_model"),
}

# The shortest valid command line of each.
MINIMAL = {
    "grasp": ["grasp"],
    "grasp-bench": ["grasp-bench", "show"],
    "grasp-eval": ["grasp-eval"],
    "grasp-model": ["grasp-model", "check"],
}

EXISTING = [
    "check", "demo", "card", "board", "calib-sim", "workspace", "train", "eval", "policies", "ui", "specs",
]


def _ns(argv: list[str]) -> dict:
    a = vars(cli.parser().parse_args(argv))
    a.pop("fn")
    return a


def _stand_ins(monkeypatch, calls: list, code: int = 17) -> None:
    """Every grasp module becomes a stand-in whose `main_*` records the namespace and returns `code`."""
    for cmd, (module, func, _) in COMMANDS.items():
        mod = types.ModuleType(module)

        def main(a, cmd=cmd):
            calls.append((cmd, a))
            return code
        setattr(mod, func, main)
        monkeypatch.setitem(sys.modules, module, mod)


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "LEROBOT_MP_CONFIG"}
    env["PYTHONPATH"] = str(SRC)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


HELP_FLAGS = {
    "grasp": ["{dry,sim,shadow,real}", "--mode", "--backend", "--port", "--i-know", "--checkpoint", "--rig",
              "--workspace", "--camera", "--episodes", "--seed", "--outside-training", "{bench,training}",
              "--floor", "{scene,model}", "--gap", "--jaw-stroke-s", "--dump-every", "--seconds",
              "--object-height"],
    "grasp-bench": ["{gripper,table,point,latency,dot,show}", "--backend", "--port", "--i-know", "--rig",
                    "--workspace", "--checkpoint", "--xy X Y", "--yaw-deg", "--start-z", "--gap-mm O:MM",
                    "--set-freeze", "--error-px", "--force", "--seconds", "--object-height", "--camera"],
    "grasp-eval": ["--episodes", "--seed", "{sim,measured}", "--jaw", "--stroke-s", "{scene,model}", "--gap",
                   "{bench,training}", "--floor", "--jitter", "--checkpoint", "--rig", "--out",
                   "--workspace"],
    "grasp-model": ["{fetch,check,time}", "--checkpoint", "--n"],
}


@pytest.mark.parametrize("cmd", list(COMMANDS))
def test_each_grasp_command_prints_its_help_in_polish_ascii(cmd, capsys, monkeypatch):
    calls: list = []
    _stand_ins(monkeypatch, calls)
    with pytest.raises(SystemExit) as exc:
        cli.main([cmd, "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith(f"usage: lerobot-twin {cmd} ")
    for flag in HELP_FLAGS[cmd]:
        assert flag in out, flag
    assert out.isascii()
    assert calls == []


def test_grasp_eval_help_offers_no_backend(capsys):
    with pytest.raises(SystemExit):
        cli.main(["grasp-eval", "--help"])
    assert "--backend" not in capsys.readouterr().out


def test_the_top_level_help_lists_the_four_grasp_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in COMMANDS:
        assert cmd in out
    assert out.isascii()


def test_the_module_docstring_shows_how_to_call_the_four_grasp_commands():
    doc = cli.__doc__
    for line in ("lerobot-twin grasp  ", "lerobot-twin grasp --mode sim", "lerobot-twin grasp-bench ",
                 "lerobot-twin grasp-eval ", "lerobot-twin grasp-model fetch"):
        assert line in doc, line


def test_grasp_defaults_are_those_of_the_specification():
    assert _ns(["grasp"]) == {
        "config": None, "cmd": "grasp", "mode": "dry", "backend": None, "port": None, "i_know": False,
        "checkpoint": None, "rig": None, "workspace": None, "camera": None, "episodes": 1, "seed": None,
        "outside_training": False, "floor": "bench", "gap": None, "jaw_stroke_s": None, "dump_every": 20,
        "seconds": None, "object_height": None,
    }


@pytest.mark.parametrize("action", ["gripper", "table", "point", "latency", "dot", "show"])
def test_grasp_bench_defaults_are_those_of_the_specification(action):
    assert _ns(["grasp-bench", action]) == {
        "config": None, "cmd": "grasp-bench", "action": action, "backend": "sim", "port": None,
        "i_know": False, "rig": None, "workspace": None, "checkpoint": None, "xy": None, "yaw_deg": 0.0,
        "start_z": 0.12, "gap_mm": None, "set_freeze": None, "error_px": None, "force": False,
        "seconds": None, "object_height": 0.030, "camera": None,
    }


def test_grasp_eval_defaults_are_those_of_the_specification_and_it_has_no_backend():
    assert _ns(["grasp-eval"]) == {
        "config": None, "cmd": "grasp-eval", "episodes": 50, "seed": 50000, "jaw": "sim", "stroke_s": None,
        "gap": None, "floor": "bench", "jitter": False, "checkpoint": None, "rig": None, "out": None,
        "workspace": None,
    }


@pytest.mark.parametrize("action", ["fetch", "check", "time"])
def test_grasp_model_defaults_are_those_of_the_specification(action):
    assert _ns(["grasp-model", action]) == {
        "config": None, "cmd": "grasp-model", "action": action, "checkpoint": None, "n": 200,
    }


def test_every_grasp_flag_parses_to_its_value_and_type():
    assert _ns(["grasp", "--mode", "real", "--backend", "a1x", "--port", "can0", "--i-know",
                "--checkpoint", "s.pt", "--rig", "g.json", "--workspace", "t.json", "--camera", "2",
                "--episodes", "3", "--seed", "7", "--outside-training", "--floor", "training",
                "--gap", "scene", "--jaw-stroke-s", "1.3", "--dump-every", "5", "--seconds", "2.5",
                "--object-height", "0.04"]) == {
        "config": None, "cmd": "grasp", "mode": "real", "backend": "a1x", "port": "can0", "i_know": True,
        "checkpoint": "s.pt", "rig": "g.json", "workspace": "t.json", "camera": "2", "episodes": 3, "seed": 7,
        "outside_training": True, "floor": "training", "gap": "scene", "jaw_stroke_s": 1.3, "dump_every": 5,
        "seconds": 2.5, "object_height": 0.04,
    }


def test_every_grasp_bench_flag_parses_to_its_value_and_type():
    assert _ns(["grasp-bench", "point", "--backend", "a1x", "--port", "can0", "--i-know", "--rig", "g.json",
                "--workspace", "t.json", "--checkpoint", "s.pt", "--xy", "-0.05", "0.3", "--yaw-deg", "-12.5",
                "--start-z", "0.1", "--gap-mm", "0:3", "--gap-mm", "50:53", "--gap-mm", "100:103",
                "--set-freeze", "100", "--error-px", "2.5", "--force", "--seconds", "4",
                "--object-height", "0.025", "--camera", "1"]) == {
        "config": None, "cmd": "grasp-bench", "action": "point", "backend": "a1x", "port": "can0",
        "i_know": True, "rig": "g.json", "workspace": "t.json", "checkpoint": "s.pt", "xy": [-0.05, 0.3],
        "yaw_deg": -12.5, "start_z": 0.1, "gap_mm": ["0:3", "50:53", "100:103"], "set_freeze": 100.0,
        "error_px": 2.5, "force": True, "seconds": 4.0, "object_height": 0.025, "camera": "1",
    }


def test_ruler_points_may_follow_one_flag_or_repeat_it():
    assert _ns(["grasp-bench", "gripper", "--gap-mm", "0:3", "50:53", "--gap-mm", "100:103"])["gap_mm"] == \
        ["0:3", "50:53", "100:103"]


def test_every_grasp_eval_flag_parses_to_its_value_and_type():
    assert _ns(["grasp-eval", "--episodes", "6", "--seed", "1", "--jaw", "measured", "--stroke-s", "1.3",
                "--gap", "model", "--floor", "training", "--jitter", "--checkpoint", "s.pt",
                "--rig", "g.json", "--out", "r.json", "--workspace", "t.json"]) == {
        "config": None, "cmd": "grasp-eval", "episodes": 6, "seed": 1, "jaw": "measured", "stroke_s": 1.3,
        "gap": "model", "floor": "training", "jitter": True, "checkpoint": "s.pt", "rig": "g.json",
        "out": "r.json", "workspace": "t.json",
    }


def test_every_grasp_model_flag_parses_to_its_value_and_type():
    assert _ns(["grasp-model", "time", "--checkpoint", "s.pt", "--n", "50"]) == {
        "config": None, "cmd": "grasp-model", "action": "time", "checkpoint": "s.pt", "n": 50,
    }


def test_the_config_file_is_accepted_before_and_after_a_grasp_command():
    assert _ns(["--config", "c.yaml", "grasp-eval"])["config"] == "c.yaml"
    assert _ns(["grasp", "--config", "c.yaml"])["config"] == "c.yaml"


@pytest.mark.parametrize("argv", [
    ["grasp", "--mode", "teleop"],
    ["grasp", "--floor", "table"],
    ["grasp", "--gap", "command"],
    ["grasp", "--episodes", "two"],
    ["grasp-bench"],
    ["grasp-bench", "jog"],
    ["grasp-bench", "point", "--xy", "0.1"],
    ["grasp-eval", "--backend", "a1x"],
    ["grasp-eval", "--jaw", "slow"],
    ["grasp-model"],
    ["grasp-model", "download"],
])
def test_a_malformed_grasp_command_line_exits_with_2_and_calls_nothing(argv, monkeypatch, capsys):
    calls: list = []
    _stand_ins(monkeypatch, calls)
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert calls == []
    assert capsys.readouterr().err.startswith("usage: lerobot-twin")


@pytest.mark.parametrize("cmd", list(COMMANDS))
def test_each_grasp_command_hands_its_namespace_to_its_main_and_returns_its_code(cmd, monkeypatch):
    calls: list = []
    _stand_ins(monkeypatch, calls)
    argv = MINIMAL[cmd] + (["--checkpoint", "s.pt"] if cmd != "grasp-eval" else ["--seed", "3"])
    assert cli.main(argv) == 17
    assert len(calls) == 1
    got, a = calls[0]
    assert got == cmd
    assert isinstance(a, argparse.Namespace)
    assert a.cmd == cmd
    assert a.fn is getattr(cli, COMMANDS[cmd][2])
    if cmd == "grasp-eval":
        assert a.seed == 3
    else:
        assert a.checkpoint == "s.pt"


@pytest.mark.parametrize("cmd", list(COMMANDS))
def test_each_grasp_command_reaches_the_main_of_its_real_module(cmd, monkeypatch):
    for lib in HEAVY:
        pytest.importorskip(lib)
    module, func, _ = COMMANDS[cmd]
    mod = importlib.import_module(module)
    assert callable(getattr(mod, func))
    calls: list = []
    monkeypatch.setattr(mod, func, lambda a: calls.append(a) or 5)
    assert cli.main(MINIMAL[cmd]) == 5
    assert len(calls) == 1 and calls[0].cmd == cmd


@pytest.fixture
def unplugged(monkeypatch) -> list:
    """The real grasp modules run with an empty stdin; an attempt to connect an arm is recorded and fails."""
    for lib in HEAVY:
        pytest.importorskip(lib)
    from lerobot_mp.twin.runtime import Twin

    tried: list = []

    def connect(self, *args, **kwargs):
        tried.append((args, kwargs))
        raise AssertionError("a grasp command connected an arm")
    monkeypatch.setattr(Twin, "connect", connect)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    return tried


@pytest.mark.parametrize(("argv", "says"), [
    (["grasp", "--mode", "real"], "--i-know"),
    (["grasp", "--mode", "shadow", "--i-know"], "--backend"),
    (["grasp", "--floor", "training"], "--floor training"),
    (["grasp", "--mode", "real", "--i-know", "--backend", "a1x", "--gap", "scene"], "--gap scene"),
    (["grasp", "--jaw-stroke-s", "1.3"], "--jaw-stroke-s"),
    (["grasp", "--mode", "sim", "--jaw-stroke-s", "0"], "--jaw-stroke-s"),
    (["grasp-eval", "--stroke-s", "1.3"], "--stroke-s"),
    (["grasp-eval", "--jaw", "measured", "--stroke-s", "-1"], "--stroke-s"),
    (["grasp-bench", "dot"], "--error-px"),
    (["grasp-bench", "gripper", "--gap-mm", "50"], "--gap-mm"),
])
def test_the_real_commands_read_the_parsed_flags_and_refuse_before_connecting(argv, says, unplugged, capsys):
    assert cli.main(argv) == 2
    assert says in capsys.readouterr().err
    assert unplugged == []


def test_ruler_points_and_the_freeze_decision_reach_the_per_rig_file_without_connecting(tmp_path, unplugged):
    rig = tmp_path / "grasp.json"
    assert cli.main(["grasp-bench", "gripper", "--rig", str(rig), "--gap-mm", "0:3", "50:53",
                     "--gap-mm", "100:103", "--set-freeze", "100"]) == 0
    gripper = json.loads(rig.read_text(encoding="utf-8"))["gripper"]
    assert gripper["gap_mm"] == {"0": 3.0, "50": 53.0, "100": 103.0}
    assert gripper["freeze_threshold"] == 100.0
    assert unplugged == []


def test_the_dot_error_reaches_the_per_rig_file_and_show_prints_it(tmp_path, unplugged, capsys):
    rig = tmp_path / "grasp.json"
    assert cli.main(["grasp-bench", "dot", "--rig", str(rig), "--error-px", "2.5"]) == 0
    assert json.loads(rig.read_text(encoding="utf-8"))["wrist"]["dot_error_px"] == 2.5
    capsys.readouterr()
    assert cli.main(["grasp-bench", "show", "--rig", str(rig)]) == 0
    out = capsys.readouterr().out
    assert f"Plik: {rig}" in out and '"dot_error_px": 2.5' in out
    assert unplugged == []


def test_grasp_model_check_loads_the_checkpoint_named_on_the_command_line(monkeypatch, capsys):
    pytest.importorskip("torch")
    from lerobot_mp.twin.grasp.policy import CHECKPOINT_SHA256, find_checkpoint

    ckpt = find_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint: lerobot-twin grasp-model fetch, or LEROBOT_GRASP_STUDENT")
    # Only the flag can lead to the file now (unless it is the default path itself).
    monkeypatch.delenv("LEROBOT_GRASP_STUDENT", raising=False)
    monkeypatch.delenv("LEROBOT_GRASP_SHA256", raising=False)
    assert cli.main(["grasp-model", "check", "--checkpoint", str(ckpt)]) == 0
    out = capsys.readouterr().out
    assert f"Model: {ckpt.resolve()}" in out
    assert f"sha256 {CHECKPOINT_SHA256} (opublikowany)" in out


def test_parsing_the_grasp_commands_imports_neither_torch_nor_mujoco_nor_the_grasp_modules():
    code = (
        "import json, sys\n"
        "from lerobot_mp.twin.cli import parser\n"
        f"for argv in {list(MINIMAL.values())!r}:\n"
        "    parser().parse_args(argv)\n"
        "print(json.dumps(sorted(m for m in sys.modules\n"
        "                        if m in ('torch', 'mujoco') or m.startswith('lerobot_mp.twin.grasp'))))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=_env(), timeout=60)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == []


def test_the_grasp_model_help_works_as_a_module_run():
    out = subprocess.run([sys.executable, "-m", "lerobot_mp.twin.cli", "grasp-model", "--help"],
                         capture_output=True, text=True, env=_env(), timeout=60)
    assert out.returncode == 0, out.stderr
    assert "{fetch,check,time}" in out.stdout


def test_the_existing_commands_come_first_in_their_old_order_and_the_grasp_commands_after_them():
    sub = next(a for a in cli.parser()._actions if isinstance(a, argparse._SubParsersAction))
    assert list(sub.choices) == EXISTING + list(COMMANDS)


@pytest.mark.parametrize(("argv", "fn", "expected"), [
    (["check"], "_check", {"cmd": "check"}),
    (["demo"], "_demo", {"cmd": "demo", "path": None, "force": False, "robot": "so101"}),
    (["card"], "_card", {"cmd": "card", "out": "karta_kalibracyjna.png", "tag_mm": 50.0, "dpi": 300}),
    (["board"], "_board", {"cmd": "board", "out": "tablica_charuco.png", "square_mm": 28.0, "dpi": 300}),
    (["calib-sim"], "_calib_sim", {"cmd": "calib-sim", "n": 5, "cameras": 2, "seed": 0, "extreme": False,
                                   "robot": "so101", "verbose": False}),
    (["workspace"], "_workspace", {"cmd": "workspace", "path": None}),
    (["train"], "_train", {"cmd": "train", "task": "reach", "envs": 4096, "iters": 300, "seed": 0,
                           "name": None, "out": None, "workspace": None, "no_rand": False, "spread": 1.0,
                           "eval_episodes": 50, "stop_file": None, "init": None}),
    (["eval", "p.pt"], "_eval", {"cmd": "eval", "policy": "p.pt", "episodes": 50, "rand": False,
                                 "workspace": None}),
    (["policies"], "_policies", {"cmd": "policies", "dir": None}),
    (["ui"], None, {"cmd": "ui", "host": "127.0.0.1", "port": 8080, "workspace": None, "stream_port": None,
                    "stream_host": "0.0.0.0"}),
    (["specs"], "_specs", {"cmd": "specs", "relay": "ws://127.0.0.1:8080/twin", "backend": None, "port": None,
                           "mode": "dry", "grasp": False, "trusted_only": False, "i_know": False,
                           "max_jump_deg": 150.0, "hold_off": 1.0, "start_center": False, "seconds": None,
                           "workspace": None}),
    (["specs", "--mode", "real", "--i-know", "--backend", "a1x", "--start-center", "--hold-off", "0"],
     "_specs", {"cmd": "specs", "relay": "ws://127.0.0.1:8080/twin", "backend": "a1x", "port": None,
                "mode": "real", "grasp": False, "trusted_only": False, "i_know": True, "max_jump_deg": 150.0,
                "hold_off": 0.0, "start_center": True, "seconds": None, "workspace": None}),
])
def test_the_existing_commands_parse_as_before(argv, fn, expected):
    a = vars(cli.parser().parse_args(argv))
    handler = a.pop("fn")
    assert a == {"config": None, **expected}
    if fn is None:
        assert callable(handler)
    else:
        assert handler is getattr(cli, fn)
    assert handler not in {getattr(cli, h) for _, _, h in COMMANDS.values()}
