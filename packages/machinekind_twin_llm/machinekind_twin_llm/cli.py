"""Running the claude and codex CLIs headless: one question, one subprocess, the subscription's login.

The CLIs answer on the subscription they are logged in with (claude.ai, ChatGPT). The environment
they get has API keys and base-URL overrides removed: a key in the twin's shell would take
precedence over the login, and a base URL could route around the subscription through a proxy.
`--bare` is never passed to claude, because it turns OAuth off.

Every call runs in a call directory of its own (`call_dir`): the images, the prompt (`stdin.txt`),
`stdout.txt` and `stderr.txt` live there, so a chatty CLI can never fill a pipe and block. The
process gets a session of its own, and a timeout or a cancel kills the whole process group, MCP
servers and shells included.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .images import SentImage

__all__ = [
    "KEEP_ENV", "MAX_TURNS_OK", "STRIPPED_ENV", "CliCancelled", "CliError", "CliTimeout", "ask_claude",
    "ask_codex", "call_dir", "check_login", "claude_command", "clean_env", "codex_command", "run_cli",
]

#: True when the installed claude accepts the hidden `--max-turns` flag. The smoke run checked it
#: on claude 2.1.309 (see README.md); set it to False if a newer claude refuses the flag.
MAX_TURNS_OK = True

#: Removed from the CLI's environment. API keys and base URLs: the login decides. The Claude Code
#: and cmux session variables: a twin started from a Claude Code or cmux terminal must not make
#: the CLI a child of that session. `CLAUDE_CODE_OAUTH_TOKEN` stays: it is the subscription token
#: of `claude setup-token`, the usual login on a headless machine.
STRIPPED_ENV = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "CODEX_API_KEY", "CODEX_ACCESS_TOKEN", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CMUX_SURFACE_ID",
    "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_CODE_EXECPATH", "CLAUDE_PID",
    "CLAUDE_EFFORT",
)

#: Set it to a directory to keep a copy of every call directory: images, prompt, stdout, stderr.
KEEP_ENV = "MACHINEKIND_TWIN_LLM_KEEP"

#: How long the login check may take [s].
LOGIN_TIMEOUT_S = 15.0

#: How often a running CLI is checked for a cancel or the deadline [s].
_POLL_S = 0.2

#: Stderr lines that say nothing about why a call failed: codex's MCP and OAuth start-up chatter.
_NOISE = (
    re.compile(r"\bmcp\b", re.IGNORECASE),
    re.compile(r"oauth", re.IGNORECASE),
    re.compile(r"^\s*Reading (prompt|additional input) from stdin", re.IGNORECASE),
)

_LIMIT = re.compile(r"usage limit|rate limit|quota|\b429\b", re.IGNORECASE)
_LOGIN = re.compile(r"not logged in|login", re.IGNORECASE)
_API_KEY = re.compile(r"api[\s_-]?key", re.IGNORECASE)

#: The field of claude's stream-json `system`/`init` event that names where an API key came from;
#: "none" on a subscription login (seen in the smoke run on claude 2.1.309).
CLAUDE_KEY_FIELD = "apiKeySource"


class CliError(RuntimeError):
    """The CLI failed or gave no usable answer."""


class CliTimeout(CliError):
    """The CLI gave no answer in time; its process group was killed."""


class CliCancelled(CliError):
    """The call was cancelled (the detector stopped); its process group was killed."""


def clean_env() -> dict[str, str]:
    """This process's environment without `STRIPPED_ENV`."""
    return {k: v for k, v in os.environ.items() if k not in STRIPPED_ENV}


def _hint(name: str) -> str:
    return "run `codex login`" if name == "codex" else "run `claude` and log in"


def _tail(text: str, limit: int = 600) -> str:
    """The last `limit` characters of a CLI's output, without its start-up noise."""
    lines = [line for line in text.splitlines() if line.strip() and not any(p.search(line) for p in _NOISE)]
    return "\n".join(lines)[-limit:].strip()


def _failure(name: str, detail: str) -> CliError:
    """A CliError for a failed call, saying so plainly for a spent subscription or a lost login."""
    if _LIMIT.search(detail):
        return CliError(f"subscription limit: {name} {detail}")
    if _LOGIN.search(detail):
        return CliError(f"{name} {detail}; {_hint(name)}")
    return CliError(f"{name} {detail}")


def _kill(proc: subprocess.Popen[Any]) -> None:
    """Terminate the group, give its parent up to 2 s to exit, then kill any remaining members."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(2)
    except subprocess.TimeoutExpired:
        pass
    # The parent exiting does not mean its children accepted SIGTERM.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_cli(cmd: Sequence[str], *, stdin: str, cwd: Path, timeout_s: float, cancel: threading.Event,
            name: str = "") -> tuple[str, str]:
    """Run `cmd` in `cwd` with `stdin` as its input and return (stdout, stderr).

    Input and output go through files in `cwd` (`stdin.txt`, `stdout.txt`, `stderr.txt`), never
    pipes. Raises CliCancelled within 0.2 s of `cancel` being set, CliTimeout after `timeout_s`,
    and CliError on a non-zero exit, with the end of stderr (or of stdout when stderr is empty).
    """
    name = name or Path(cmd[0]).name
    src, out, err = cwd / "stdin.txt", cwd / "stdout.txt", cwd / "stderr.txt"
    src.write_text(stdin, encoding="utf-8")
    with open(src, encoding="utf-8") as fi, open(out, "w", encoding="utf-8") as fo, \
            open(err, "w", encoding="utf-8") as fe:
        try:
            proc = subprocess.Popen(list(cmd), stdin=fi, stdout=fo, stderr=fe, cwd=cwd, env=clean_env(),
                                    start_new_session=True)
        except OSError as e:
            raise CliError(f"{name} could not be started: {e}") from e
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                proc.wait(timeout=_POLL_S)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel.is_set():
                _kill(proc)
                raise CliCancelled("stopped")
            if time.monotonic() > deadline:
                _kill(proc)
                raise CliTimeout(f"{name} gave no answer in {timeout_s:.0f} s")
    stdout, stderr = out.read_text(encoding="utf-8"), err.read_text(encoding="utf-8")
    if proc.returncode != 0:
        raise _failure(name, f"exit {proc.returncode}: {_tail(stderr) or _tail(stdout)}")
    return stdout, stderr


@contextmanager
def call_dir(work: Path) -> Iterator[Path]:
    """A fresh directory for one call under `work`, removed afterwards, also after a failure.

    With `MACHINEKIND_TWIN_LLM_KEEP=<dir>` set, a copy goes to `<dir>/<time>-<name>` first.
    """
    call = Path(tempfile.mkdtemp(prefix="call-", dir=work))
    try:
        yield call
    finally:
        keep = os.environ.get(KEEP_ENV)
        if keep:
            try:
                dest = Path(keep).expanduser() / f"{time.strftime('%Y%m%d-%H%M%S')}-{call.name}"
                shutil.copytree(call, dest, dirs_exist_ok=True)
            except OSError:
                pass
        shutil.rmtree(call, ignore_errors=True)


def check_login(cli: str, exe: str, *, timeout_s: float = LOGIN_TIMEOUT_S) -> str:
    """How `cli` is logged in, e.g. "claude.ai (max)" or "Logged in using ChatGPT".

    Refuses API-key auth only: any other login passes and its name is returned for the log.
    Raises RuntimeError when the CLI is not logged in, uses an API key, or cannot be run.
    """
    args = [exe, "auth", "status"] if cli == "claude" else [exe, "login", "status"]
    try:
        with subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=clean_env(), start_new_session=True) as r:
            try:
                stdout, stderr = r.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired as e:
                _kill(r)
                raise RuntimeError(f"{cli} login check gave no answer in {timeout_s:.0f} s") from e
    except OSError as e:
        raise RuntimeError(f"{cli} could not be run: {e}") from e
    text = _tail(stdout + "\n" + stderr, 300)
    logged_out = RuntimeError(f"{cli} is not logged in; {_hint(cli)}")
    if cli == "claude":
        try:
            status = json.loads(stdout)
        except ValueError:
            status = None
        if isinstance(status, dict):
            method = str(status.get("authMethod") or "")
            source = str(status.get("apiKeySource") or "")
            if _API_KEY.search(method) or source:
                raise RuntimeError(f"claude uses {method or source}, not a subscription: {text}")
            if r.returncode != 0 or not status.get("loggedIn", True):
                raise logged_out
            plan = status.get("subscriptionType")
            return f"{method or 'logged in'}" + (f" ({plan})" if plan else "")
    if r.returncode != 0:
        raise logged_out
    if _API_KEY.search(text):
        raise RuntimeError(f"{cli} uses an API key, not a subscription: {text}")
    return text.splitlines()[0] if text else "logged in"


def _lines(stdout: str) -> list[dict[str, Any]]:
    """The JSON objects of an NDJSON stream, skipping lines that are not JSON objects."""
    events = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _structured(result: Mapping[str, Any] | None) -> dict[str, Any]:
    """The structured answer of claude's `result` event, or a CliError saying why there is none."""
    if result is None:
        raise CliError("claude gave no result")
    if result.get("is_error") or result.get("subtype") != "success":
        detail = result.get("result") or result.get("errors") or result.get("subtype")
        raise _failure("claude", f"failed: {_tail(str(detail))}")
    answer = result.get("structured_output")
    if not isinstance(answer, dict):
        raise CliError(f"claude gave no structured output: {_tail(str(result.get('result') or ''), 200)}")
    return answer


def claude_command(exe: str, *, route: str, schema: Mapping[str, Any], model: str = "", effort: str = "",
                   max_turns: bool | None = None) -> list[str]:
    """claude's argv for one question. Route "stream": images inline over stream-json stdin, no
    tools. Route "read": the prompt names image files in the working directory, Read only."""
    turns = MAX_TURNS_OK if max_turns is None else max_turns
    common = ["--json-schema", json.dumps(schema), "--no-session-persistence", "--safe-mode",
              "--strict-mcp-config", "--permission-prompts", "none",
              *(["--model", model] if model else []), *(["--effort", effort] if effort else [])]
    if route == "stream":
        return [exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                "--tools", "", *(["--max-turns", "3"] if turns else []), *common]
    if route == "read":
        return [exe, "-p", "--output-format", "json", "--tools", "Read", "--allowedTools", "Read",
                *(["--max-turns", "6"] if turns else []), *common]
    raise ValueError(f"claude image route must be stream or read, not {route!r}")


def ask_claude(exe: str, sent: Sequence[SentImage], prompt: str, schema: Mapping[str, Any], *, cwd: Path,
               model: str = "", effort: str = "", route: str = "stream", timeout_s: float = 120.0,
               cancel: threading.Event | None = None, max_turns: bool | None = None) -> dict[str, Any]:
    """Ask claude one question about `sent` and return its structured answer.

    Route "stream" puts each image inline as a base64 block, after a line naming its camera and
    size, and the prompt last. Route "read" sends the prompt alone: it must name the image files,
    which live in `cwd`.
    """
    cmd = claude_command(exe, route=route, schema=schema, model=model, effort=effort, max_turns=max_turns)
    if route == "stream":
        content: list[dict[str, Any]] = []
        for i, s in enumerate(sent):
            w, h = s.sent_size
            content.append({"type": "text", "text": f'Image {i + 1}: camera "{s.camera}", {w}x{h} px.'})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": base64.b64encode(s.path.read_bytes()).decode()}})
        content.append({"type": "text", "text": prompt})
        stdin = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"
    else:
        stdin = prompt
    stdout, _ = run_cli(cmd, stdin=stdin, cwd=cwd, timeout_s=timeout_s, cancel=cancel or threading.Event(),
                        name="claude")
    if route == "stream":
        events = _lines(stdout)
        for event in events:
            if event.get("type") == "system" and event.get("subtype") == "init":
                source = str(event.get(CLAUDE_KEY_FIELD) or "none")
                if source.lower() != "none":
                    raise CliError(f"claude answered with an API key ({source}), not a subscription")
        results = [e for e in events if e.get("type") == "result"]
        return _structured(results[-1] if results else None)
    try:
        parsed: Any = json.loads(stdout)
    except ValueError as e:
        raise CliError(f"claude answer is not JSON: {_tail(stdout, 200)}") from e
    if isinstance(parsed, list):
        results = [e for e in parsed if isinstance(e, dict) and e.get("type") == "result"]
        parsed = results[-1] if results else None
    return _structured(parsed if isinstance(parsed, dict) else None)


def codex_command(exe: str, *, call: Path, sent: Sequence[SentImage], model: str = "",
                  effort: str = "") -> list[str]:
    """codex's argv for one question; the schema is `call/schema.json`, the answer `call/last.txt`."""
    cmd = [exe, "exec", "--json", "-o", str(call / "last.txt"), "--skip-git-repo-check", "--ephemeral",
           "--ignore-user-config", "--sandbox", "read-only", "--cd", str(call), "--color", "never",
           "--output-schema", str(call / "schema.json"),
           *(["-m", model] if model else []),
           *(["-c", f"model_reasoning_effort={json.dumps(effort)}"] if effort else [])]
    for s in sent:
        cmd += ["--image", str(s.path)]
    return [*cmd, "--", "-"]  # `--` ends the variadic --image; `-` reads the prompt from stdin


def ask_codex(exe: str, sent: Sequence[SentImage], prompt: str, schema: Mapping[str, Any], *, cwd: Path,
              model: str = "", effort: str = "", timeout_s: float = 120.0,
              cancel: threading.Event | None = None) -> dict[str, Any]:
    """Ask codex one question about `sent` (attached with `--image`, in order) and return its answer.

    `--ignore-user-config` drops the user's config.toml (their effort, MCP servers, notify hook)
    and keeps the login.
    """
    (cwd / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    last = cwd / "last.txt"
    cmd = codex_command(exe, call=cwd, sent=sent, model=model, effort=effort)
    stdout, _ = run_cli(cmd, stdin=prompt, cwd=cwd, timeout_s=timeout_s, cancel=cancel or threading.Event(),
                        name="codex")
    events = _lines(stdout)
    message = ""
    for event in events:
        kind = event.get("type")
        if kind in ("turn.failed", "error"):
            error = event.get("error")
            detail = error.get("message") if isinstance(error, dict) else event.get("message")
            raise _failure("codex", f"{kind}: {detail or event}")
        item = event.get("item")
        if kind == "item.completed" and isinstance(item, dict) and item.get("type") == "agent_message":
            message = str(item.get("text") or "")
    text = last.read_text(encoding="utf-8") if last.is_file() else ""
    text = text.strip() or message.strip()
    if not text:
        raise CliError("codex gave an empty answer")
    try:
        answer = json.loads(text)
    except ValueError as e:
        raise CliError(f"codex answer is not JSON: {text[:200]}") from e
    if not isinstance(answer, dict):
        raise CliError(f"codex answer is not a JSON object: {text[:200]}")
    return answer
