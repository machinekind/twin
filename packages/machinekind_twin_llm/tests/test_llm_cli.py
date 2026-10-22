"""The CLI runner against a fake claude and codex (tests/fake_cli.py): argv, stdin, environment,
timeouts, cancels, failures and the call directory. No model, no network."""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from machinekind_twin_llm import cli
from machinekind_twin_llm.images import SentImage

FAKE = Path(__file__).with_name("fake_cli.py")

_NUM = {"type": "number"}
SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["objects"],
    "properties": {"objects": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["label", "camera", "u", "v"],
        "properties": {"label": {"type": "string"}, "camera": {"type": "string"}, "u": _NUM, "v": _NUM}}}}}
ANSWER = {"objects": [{"label": "red pen", "camera": "top", "u": 10.0, "v": 20.0}]}


class Fake:
    def __init__(self, bin_dir: Path, log: Path) -> None:
        self.claude = str(bin_dir / "claude")
        self.codex = str(bin_dir / "codex")
        self._log = log

    def runs(self) -> list[dict[str, Any]]:
        if not self._log.is_file():
            return []
        return [json.loads(line) for line in self._log.read_text(encoding="utf-8").splitlines()]

    def last(self) -> dict[str, Any]:
        return [r for r in self.runs() if "argv" in r][-1]


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fake:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    body = FAKE.read_text(encoding="utf-8")
    for name in ("claude", "codex"):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(ANSWER), encoding="utf-8")
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_ANSWER", str(answer))
    monkeypatch.setenv("FAKE_MODE", "ok")
    monkeypatch.setenv("FAKE_AUTH", "subscription")
    monkeypatch.delenv(cli.KEEP_ENV, raising=False)
    return Fake(bin_dir, log)


@pytest.fixture
def call(tmp_path: Path) -> Path:
    path = tmp_path / "call"
    path.mkdir()
    return path


def jpegs(call: Path, sizes: tuple[tuple[int, int], ...] = ((320, 240), (160, 120))) -> list[SentImage]:
    sent = []
    for i, (w, h) in enumerate(sizes):
        path = call / f"img-{i + 1}.jpg"
        cv2.imwrite(str(path), np.full((h, w, 3), 128, np.uint8))
        sent.append(SentImage(("top", "side")[i], path, (w, h), (2 * w, 2 * h)))
    return sent


def after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.mark.parametrize("turns", [True, False])
def test_claude_stream_argv_and_one_line_of_stdin_with_every_image(
        fake: Fake, call: Path, monkeypatch: pytest.MonkeyPatch, turns: bool):
    monkeypatch.setattr(cli, "MAX_TURNS_OK", turns)
    sent = jpegs(call)

    got = cli.ask_claude(fake.claude, sent, "find things", SCHEMA, cwd=call, model="sonnet", effort="low")

    assert got == ANSWER
    run = fake.last()
    argv = run["argv"]
    assert "-p" in argv and "--bare" not in argv
    assert after(argv, "--input-format") == "stream-json"
    assert after(argv, "--output-format") == "stream-json"
    assert json.loads(after(argv, "--json-schema")) == SCHEMA
    assert after(argv, "--tools") == ""
    assert "--safe-mode" in argv and "--no-session-persistence" in argv and "--strict-mcp-config" in argv
    assert after(argv, "--permission-prompts") == "none"
    assert after(argv, "--model") == "sonnet" and after(argv, "--effort") == "low"
    assert ("--max-turns" in argv) is turns
    assert Path(run["cwd"]).resolve() == call.resolve()
    lines = run["stdin"].splitlines()
    assert len(lines) == 1
    content = json.loads(lines[0])["message"]["content"]
    assert [b["type"] for b in content] == ["text", "image", "text", "image", "text"]
    assert content[0]["text"] == 'Image 1: camera "top", 320x240 px.'
    assert content[-1]["text"] == "find things"
    assert run["images"] == [{"source": "inline", "jpeg": True, "size": [320, 240]},
                             {"source": "inline", "jpeg": True, "size": [160, 120]}]


def test_claude_without_model_or_effort_passes_neither(fake: Fake, call: Path):
    cli.ask_claude(fake.claude, jpegs(call), "p", SCHEMA, cwd=call)

    argv = fake.last()["argv"]
    assert "--model" not in argv and "--effort" not in argv


def test_claude_read_route_names_existing_files_in_its_working_directory(fake: Fake, call: Path):
    sent = jpegs(call)
    prompt = "\n".join(f"camera {s.camera}: file {s.path} (open it with the Read tool)" for s in sent)

    got = cli.ask_claude(fake.claude, sent, prompt, SCHEMA, cwd=call, route="read")

    assert got == ANSWER
    run = fake.last()
    argv = run["argv"]
    assert after(argv, "--tools") == "Read" and after(argv, "--allowedTools") == "Read"
    assert after(argv, "--output-format") == "json" and "--input-format" not in argv
    assert "--add-dir" not in argv and "--bare" not in argv
    assert [Path(i["source"]).parent for i in run["images"]] == [call, call]
    assert all(i["jpeg"] for i in run["images"])


def test_claude_refuses_an_unknown_route(fake: Fake, call: Path):
    with pytest.raises(ValueError, match="stream or read"):
        cli.ask_claude(fake.claude, jpegs(call), "p", SCHEMA, cwd=call, route="inline")


def test_claude_refuses_an_answer_given_on_an_api_key(fake: Fake, call: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_AUTH", "apikey")

    with pytest.raises(cli.CliError, match="API key"):
        cli.ask_claude(fake.claude, jpegs(call), "p", SCHEMA, cwd=call)


def test_codex_argv_images_in_order_then_the_end_of_options(fake: Fake, call: Path):
    sent = jpegs(call)

    got = cli.ask_codex(fake.codex, sent, "find things", SCHEMA, cwd=call, model="gpt-6-astra", effort="low")

    assert got == ANSWER
    run = fake.last()
    argv = run["argv"][1:]
    assert argv[:4] == ["exec", "--json", "-o", str(call / "last.txt")]
    for f in ("--skip-git-repo-check", "--ephemeral", "--ignore-user-config"):
        assert f in argv
    assert after(argv, "--sandbox") == "read-only"
    assert after(argv, "--cd") == str(call)
    assert after(argv, "-m") == "gpt-6-astra"
    assert after(argv, "-c") == 'model_reasoning_effort="low"'
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--image"] == [str(s.path) for s in sent]
    assert argv[-2:] == ["--", "-"]
    assert run["stdin"] == "find things"
    assert [i["size"] for i in run["images"]] == [[320, 240], [160, 120]]
    schema = json.loads(Path(after(argv, "--output-schema")).read_text(encoding="utf-8"))
    assert schema == SCHEMA
    _assert_strict(schema)


def _assert_strict(node: Any) -> None:
    """Strict structured output: every object closes its properties and requires all of them."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert sorted(node["required"]) == sorted(node["properties"])
        for value in node.values():
            _assert_strict(value)
    elif isinstance(node, list):
        for value in node:
            _assert_strict(value)


def test_codex_without_model_or_effort_passes_neither(fake: Fake, call: Path):
    cli.ask_codex(fake.codex, jpegs(call), "p", SCHEMA, cwd=call)

    argv = fake.last()["argv"]
    assert "-m" not in argv and "-c" not in argv


@pytest.mark.parametrize("who", ["claude", "codex"])
def test_api_keys_base_urls_and_the_calling_session_never_reach_the_cli(
        fake: Fake, call: Path, monkeypatch: pytest.MonkeyPatch, who: str):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY", "CLAUDECODE", "CMUX_SURFACE_ID",
                 "CLAUDE_CODE_SESSION_ID"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "subscription-token")

    ask = cli.ask_claude if who == "claude" else cli.ask_codex
    ask(getattr(fake, who), jpegs(call), "p", SCHEMA, cwd=call)

    keys = set(fake.last()["env_keys"])
    assert not keys & {"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY", "CLAUDECODE",
                       "CMUX_SURFACE_ID", "CLAUDE_CODE_SESSION_ID"}
    assert "CLAUDE_CODE_OAUTH_TOKEN" in keys


def test_clean_env_strips_exactly_the_listed_names(monkeypatch: pytest.MonkeyPatch):
    for name in cli.STRIPPED_ENV:
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("KEEP_ME", "y")

    env = cli.clean_env()

    assert not set(cli.STRIPPED_ENV) & set(env)
    assert env["KEEP_ME"] == "y"


def test_a_hanging_cli_times_out(fake: Fake, call: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_MODE", "sleep")
    t0 = time.monotonic()

    with pytest.raises(cli.CliTimeout, match="no answer in"):
        cli.ask_claude(fake.claude, jpegs(call), "p", SCHEMA, cwd=call, timeout_s=0.5)

    assert time.monotonic() - t0 < 4.0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_timeout_kills_the_whole_process_group(fake: Fake, call: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_MODE", "child")

    with pytest.raises(cli.CliTimeout):
        cli.ask_codex(fake.codex, jpegs(call), "p", SCHEMA, cwd=call, timeout_s=1.0)

    pid = next(r["grandchild"] for r in fake.runs() if "grandchild" in r)
    deadline = time.monotonic() + 3.0
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)


@pytest.mark.parametrize("who", ["claude", "codex"])
def test_a_cancel_mid_call_stops_it_within_a_second(fake: Fake, call: Path, monkeypatch: pytest.MonkeyPatch,
                                                    who: str):
    monkeypatch.setenv("FAKE_MODE", "sleep")
    cancel = threading.Event()
    fired: list[float] = []

    def fire() -> None:
        fired.append(time.monotonic())
        cancel.set()

    timer = threading.Timer(0.5, fire)
    timer.start()
    with pytest.raises(cli.CliCancelled):
        ask = cli.ask_claude if who == "claude" else cli.ask_codex
        ask(getattr(fake, who), jpegs(call), "p", SCHEMA, cwd=call, timeout_s=30, cancel=cancel)

    assert time.monotonic() - fired[0] < 1.0


@pytest.mark.parametrize("who", ["claude", "codex"])
def test_a_failing_cli_reports_the_end_of_its_stderr(fake: Fake, call: Path, monkeypatch: pytest.MonkeyPatch,
                                                     who: str):
    monkeypatch.setenv("FAKE_MODE", "exit1")
    ask = cli.ask_claude if who == "claude" else cli.ask_codex

    with pytest.raises(cli.CliError, match=f"{who} exit 1: boom: something broke"):
        ask(getattr(fake, who), jpegs(call), "p", SCHEMA, cwd=call)


@pytest.mark.parametrize(("who", "mode", "message"), [
    ("claude", "garbage", "no result"),
    ("claude", "no_structured", "no structured output"),
    ("claude", "limit", "subscription limit"),
    ("codex", "garbage", "not JSON"),
    ("codex", "no_structured", "empty answer"),
    ("codex", "turn_failed", "the model refused"),
    ("codex", "limit", "subscription limit"),
])
def test_unusable_answers_raise(fake: Fake, call: Path, monkeypatch: pytest.MonkeyPatch, who: str, mode: str,
                                message: str):
    monkeypatch.setenv("FAKE_MODE", mode)
    ask = cli.ask_claude if who == "claude" else cli.ask_codex

    with pytest.raises(cli.CliError, match=message):
        ask(getattr(fake, who), jpegs(call), "p", SCHEMA, cwd=call)


def test_claude_read_route_garbage_is_not_json(fake: Fake, call: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_MODE", "garbage")

    with pytest.raises(cli.CliError, match="not JSON"):
        cli.ask_claude(fake.claude, jpegs(call), "p", SCHEMA, cwd=call, route="read")


def test_a_missing_executable_is_a_cli_error(call: Path):
    with pytest.raises(cli.CliError, match="could not be started"):
        cli.run_cli([str(call / "nope")], stdin="", cwd=call, timeout_s=5, cancel=threading.Event())


def test_codex_noise_is_left_out_of_the_error():
    noisy = ("2026-10-22T23:45:26Z ERROR codex_rmcp_client::oauth::refresh_transaction: failed\n"
             "real problem here\n")
    assert cli._tail(noisy) == "real problem here"


@pytest.mark.parametrize(("who", "auth"), [("claude", "subscription"), ("claude", "oauth_token"),
                                           ("codex", "subscription"), ("codex", "oauth_token")])
def test_login_check_passes_a_subscription(fake: Fake, monkeypatch: pytest.MonkeyPatch, who: str, auth: str):
    monkeypatch.setenv("FAKE_AUTH", auth)

    method = cli.check_login(who, getattr(fake, who))

    assert method
    assert "API key" not in method


@pytest.mark.parametrize("who", ["claude", "codex"])
def test_login_check_refuses_an_api_key(fake: Fake, monkeypatch: pytest.MonkeyPatch, who: str):
    monkeypatch.setenv("FAKE_AUTH", "apikey")

    with pytest.raises(RuntimeError, match="not a subscription"):
        cli.check_login(who, getattr(fake, who))


@pytest.mark.parametrize(("who", "hint"), [("claude", "run `claude`"), ("codex", "run `codex login`")])
def test_login_check_refuses_a_logged_out_cli(fake: Fake, monkeypatch: pytest.MonkeyPatch, who: str, hint: str):
    monkeypatch.setenv("FAKE_AUTH", "logged_out")

    with pytest.raises(RuntimeError, match=f"not logged in; {hint}"):
        cli.check_login(who, getattr(fake, who))


def test_the_call_directory_goes_after_success_and_after_failure(tmp_path: Path):
    with cli.call_dir(tmp_path) as ok:
        (ok / "img-1.jpg").write_bytes(b"x")
    assert not ok.exists()

    with pytest.raises(RuntimeError), cli.call_dir(tmp_path) as failed:
        raise RuntimeError("boom")
    assert not failed.exists()
    assert list(tmp_path.iterdir()) == []


def test_keep_copies_each_call_directory_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: Fake):
    keep = tmp_path / "keep"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv(cli.KEEP_ENV, str(keep))

    with cli.call_dir(work) as call:
        cli.ask_claude(fake.claude, jpegs(call), "p", SCHEMA, cwd=call)

    assert not call.exists()
    (kept,) = keep.iterdir()
    assert {p.name for p in kept.iterdir()} == {"img-1.jpg", "img-2.jpg", "stdin.txt", "stdout.txt", "stderr.txt"}
    sent = json.loads((kept / "stdin.txt").read_text(encoding="utf-8"))
    data = sent["message"]["content"][1]["source"]["data"]
    assert base64.b64decode(data)[:2] == b"\xff\xd8"
