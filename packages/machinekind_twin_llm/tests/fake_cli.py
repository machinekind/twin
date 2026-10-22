"""A stand-in for the claude and codex CLIs: the tests copy it into a bin folder under both names,
with a shebang naming the test's own Python.

It answers as the CLI it is named after and logs what it got, so a test sees exactly what the
real CLI would have seen. No model, no network.

Environment:
    FAKE_LOG     a file; one JSON line per run: argv, stdin, cwd, env_keys, images
    FAKE_ANSWER  a JSON file with the answer (default: no objects)
    FAKE_MODE    ok | sleep | garbage | exit1 | no_structured | turn_failed | limit | child
    FAKE_AUTH    subscription | oauth_token | apikey | logged_out
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

NAME = Path(sys.argv[0]).name
ARGS = sys.argv[1:]
MODE = os.environ.get("FAKE_MODE", "ok")
AUTH = os.environ.get("FAKE_AUTH", "subscription")


def log(entry: dict) -> None:
    path = os.environ.get("FAKE_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def image(data: bytes, source: str) -> dict:
    """What a model would get from these bytes: a JPEG or not, and its size."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    return {"source": source, "jpeg": data[:2] == b"\xff\xd8",
            "size": None if img is None else [img.shape[1], img.shape[0]]}


def from_file(path: str) -> dict:
    p = Path(path)
    return image(p.read_bytes(), path) if p.is_file() else {"source": path, "jpeg": False, "size": None}


def auth_status() -> int:
    if NAME == "claude":
        status = {
            "subscription": {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"},
            "oauth_token": {"loggedIn": True, "authMethod": "oauth_token"},
            "apikey": {"loggedIn": True, "authMethod": "api_key", "apiKeySource": "ANTHROPIC_API_KEY"},
            "logged_out": {"loggedIn": False, "authMethod": "none"},
        }[AUTH]
        print(json.dumps(status, indent=2))
        return 1 if AUTH == "logged_out" else 0
    text = {"subscription": "Logged in using ChatGPT", "oauth_token": "Logged in using ChatGPT",
            "apikey": "Logged in using an API key - sk-proj-***ABCD", "logged_out": "Not logged in"}[AUTH]
    print(text, file=sys.stderr)  # codex prints its status on stderr
    return 1 if AUTH == "logged_out" else 0


def answer() -> dict:
    path = os.environ.get("FAKE_ANSWER")
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {"objects": []}


def flag(name: str) -> str | None:
    return ARGS[ARGS.index(name) + 1] if name in ARGS else None


def claude(stdin: str) -> int:
    images = []
    if flag("--input-format") == "stream-json":
        for line in stdin.splitlines():
            for block in json.loads(line)["message"]["content"]:
                if block["type"] == "image":
                    images.append(image(base64.b64decode(block["source"]["data"]), "inline"))
    else:
        images = [from_file(p) for p in re.findall(r"(/\S+?\.jpg)", stdin)]
    log({"argv": sys.argv, "stdin": stdin, "cwd": os.getcwd(), "env_keys": sorted(os.environ), "images": images})
    run_mode()
    stream = flag("--output-format") == "stream-json"
    result: dict = {"type": "result", "subtype": "success", "is_error": False, "result": "done",
                    "structured_output": answer()}
    if MODE == "garbage":
        print("this is not json at all")
        return 0
    if MODE == "no_structured":
        del result["structured_output"]
    if MODE == "limit":
        result.update(is_error=True, result="Claude AI usage limit reached|1792718400")
    if stream:
        source = "ANTHROPIC_API_KEY" if AUTH == "apikey" else "none"
        print(json.dumps({"type": "system", "subtype": "init", "apiKeySource": source, "tools": []}))
        print(json.dumps({"type": "assistant", "message": {"content": []}}))
        print(json.dumps(result))
    else:
        print(json.dumps(result))
    return 1 if MODE == "limit" else 0


def codex(stdin: str) -> int:
    images = [from_file(ARGS[i + 1]) for i, a in enumerate(ARGS) if a == "--image"]
    log({"argv": sys.argv, "stdin": stdin, "cwd": os.getcwd(), "env_keys": sorted(os.environ), "images": images})
    run_mode()
    last = flag("-o")
    print(json.dumps({"type": "thread.started", "thread_id": "fake"}))
    print(json.dumps({"type": "turn.started"}))
    if MODE in ("turn_failed", "limit"):
        message = "You've hit your usage limit." if MODE == "limit" else "the model refused"
        print(json.dumps({"type": "error", "message": message}))
        print(json.dumps({"type": "turn.failed", "error": {"message": message}}))
        return 1
    text = "this is not json" if MODE == "garbage" else "" if MODE == "no_structured" else json.dumps(answer())
    print(json.dumps({"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": text}}))
    print(json.dumps({"type": "turn.completed", "usage": {}}))
    if last:
        Path(last).write_text(text, encoding="utf-8")
    return 0


def run_mode() -> None:
    """Act out the modes that never answer: a crash, a hang, a hang with a grandchild."""
    if MODE == "exit1":
        print("boom: something broke", file=sys.stderr)
        sys.exit(1)
    if MODE == "child":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        log({"grandchild": child.pid})
    if MODE in ("sleep", "child"):
        time.sleep(60)


def main() -> int:
    if ARGS[:2] in (["auth", "status"], ["login", "status"]):
        return auth_status()
    stdin = sys.stdin.read()
    return claude(stdin) if NAME == "claude" else codex(stdin)


if __name__ == "__main__":
    sys.exit(main())
