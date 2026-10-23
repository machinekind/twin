"""One real question to the claude or codex CLI, by hand, to check what the installed version accepts.

    python -m machinekind_twin_llm.smoke --cli claude [--route stream|read] [--model M] [IMAGE...]
    python -m machinekind_twin_llm.smoke --cli codex [--model M] [--effort E] [IMAGE...]

Without IMAGE it draws two synthetic photos (a red square, a blue disc) and checks that the model
puts each object within 40 px of where it was drawn. It prints the login check, the argv (schema
and images elided), the duration and the answer, then one line per check. A failed check fails
loudly (exit 1). The outcome of a run on the installed versions is recorded in README.md.

It makes one call to the model, two for codex: the second one sends an effort codex does not know,
which the server refuses before any model turn. That proves `-c model_reasoning_effort` reaches the
request with `--ignore-user-config`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import cli
from .images import SentImage
from .prompt import SCHEMA

#: The detector's answer shape (strict, so codex accepts it too).
SMOKE_SCHEMA: dict[str, Any] = SCHEMA

#: Synthetic photos: (camera, drawing, expected centre (u, v)).
_SIZE = (640, 480)
_RED_AT = (200, 300)
_BLUE_AT = (450, 150)
#: How far off a synthetic object's answer may be [px].
_TOLERANCE_PX = 40.0


def _synthetic() -> list[tuple[str, np.ndarray, tuple[float, float]]]:
    """Two BGR photos of a grey table: one with a red square, one with a blue disc."""
    w, h = _SIZE
    red = np.full((h, w, 3), 170, np.uint8)
    cv2.rectangle(red, (_RED_AT[0] - 30, _RED_AT[1] - 30), (_RED_AT[0] + 30, _RED_AT[1] + 30), (30, 30, 220), -1)
    blue = np.full((h, w, 3), 170, np.uint8)
    cv2.circle(blue, _BLUE_AT, 35, (200, 60, 20), -1)
    return [("cam1", red, (float(_RED_AT[0]), float(_RED_AT[1]))),
            ("cam2", blue, (float(_BLUE_AT[0]), float(_BLUE_AT[1])))]


def _write(call: Path, photos: list[tuple[str, np.ndarray]], max_side: int = 1280) -> list[SentImage]:
    """Each photo as `img-<k>.jpg` in `call`, its long edge at most `max_side`."""
    sent = []
    for i, (camera, bgr) in enumerate(photos):
        h, w = bgr.shape[:2]
        scale = min(1.0, max_side / max(w, h))
        img = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1.0 else bgr
        path = call / f"img-{i + 1}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sent.append(SentImage(camera, path, (img.shape[1], img.shape[0]), (w, h)))
    return sent


def _prompt(sent: list[SentImage], read: bool) -> str:
    lines = []
    for i, s in enumerate(sent):
        w, h = s.sent_size
        where = f"file {s.path} (open it with the Read tool)" if read else f"image {i + 1}"
        lines.append(f'- camera "{s.camera}": {w}x{h} px, {where}')
    return "\n".join([
        f"You get {len(sent)} photos, one per camera.", *lines,
        ("Find every distinct coloured object. For every object, in every image where you can see it, give one "
         "entry: label (lowercase colour plus noun); camera (the camera name of that image); u, v (the pixel at "
         "the object's centre in that image's own pixels: u from the left edge, v from the top); box (its "
         "bounding box in the same pixels, or null); confidence (0..1). Answer only with the JSON."),
    ])


def _shown(cmd: list[str]) -> str:
    out, skip = [], False
    for i, arg in enumerate(cmd):
        if skip:
            skip = False
            continue
        if arg == "--json-schema":
            out += [arg, f"<schema, {len(cmd[i + 1])} chars>"]
            skip = True
        else:
            out.append(repr(arg) if arg == "" or " " in arg else arg)
    return " ".join(out)


class _Checks:
    def __init__(self) -> None:
        self.failed = 0

    def __call__(self, ok: bool, what: str, detail: str = "") -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {what}" + (f": {detail}" if detail else ""))
        self.failed += not ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m machinekind_twin_llm.smoke", description=(__doc__ or "").splitlines()[0])
    p.add_argument("--cli", choices=("claude", "codex"), required=True)
    p.add_argument("--model", default="")
    p.add_argument("--effort", default="low")
    p.add_argument("--route", choices=("stream", "read"), default="stream", help="claude only")
    p.add_argument("--executable", default="", help="the CLI binary; default: --cli found on PATH")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("images", nargs="*", type=Path)
    a = p.parse_args(argv)

    exe = str(Path(a.executable).expanduser()) if a.executable else shutil.which(a.cli)
    if not exe:
        print(f"{a.cli} not found on PATH; pass --executable", file=sys.stderr)
        return 2
    version = subprocess.run([exe, "--version"], capture_output=True, text=True, env=cli.clean_env(), check=False)
    print(f"executable: {exe} ({version.stdout.strip() or version.stderr.strip()})")
    check = _Checks()
    try:
        print(f"login: {cli.check_login(a.cli, exe)}")
    except RuntimeError as e:
        check(False, "login check", str(e))
        return 1

    photos: list[tuple[str, np.ndarray]] = []
    expected: list[tuple[str, tuple[float, float]]] = []
    if a.images:
        for i, path in enumerate(a.images):
            img = cv2.imread(str(path))
            if img is None:
                print(f"cannot read {path}", file=sys.stderr)
                return 2
            photos.append((f"cam{i + 1}", np.asarray(img)))
    else:
        synthetic = _synthetic()
        photos = [(camera, img) for camera, img, _ in synthetic]
        expected = [(camera, at) for camera, _, at in synthetic]

    work = Path(tempfile.mkdtemp(prefix="twin-llm-smoke-"))
    try:
        with cli.call_dir(work) as call:
            sent = _write(call, photos)
            read = a.cli == "claude" and a.route == "read"
            prompt = _prompt(sent, read)
            if a.cli == "claude":
                cmd = cli.claude_command(exe, route=a.route, schema=SMOKE_SCHEMA, model=a.model, effort=a.effort,
                                         max_turns=True)
            else:
                cmd = cli.codex_command(exe, call=call, sent=sent, model=a.model, effort=a.effort)
            print(f"argv: {_shown(cmd)}")
            t0 = time.monotonic()
            try:
                if a.cli == "claude":
                    answer = cli.ask_claude(exe, sent, prompt, SMOKE_SCHEMA, cwd=call, model=a.model,
                                            effort=a.effort, route=a.route, timeout_s=a.timeout, max_turns=True)
                else:
                    answer = cli.ask_codex(exe, sent, prompt, SMOKE_SCHEMA, cwd=call, model=a.model,
                                           effort=a.effort, timeout_s=a.timeout)
            except cli.CliError as e:
                print(f"duration: {time.monotonic() - t0:.1f} s")
                check(not isinstance(e, cli.CliTimeout), "the CLI exits on stdin EOF", type(e).__name__)
                if "max-turns" in str(e):
                    check(False, "--max-turns accepted", "set cli.MAX_TURNS_OK = False")
                check(False, "the call", str(e))
                return 1
            print(f"duration: {time.monotonic() - t0:.1f} s")
            print(f"answer: {json.dumps(answer)}")
            check(True, "the CLI exits on stdin EOF")
            if a.cli == "claude":
                check(True, "--max-turns accepted")
                check(isinstance(answer.get("objects"), list), "--json-schema returns structured_output"
                      + (' with --tools ""' if a.route == "stream" else " with --tools Read"))
            else:
                check(isinstance(answer.get("objects"), list), "--output-schema answer parsed from -o")
            _check_objects(check, answer, expected, sent)
            if a.cli == "claude" and a.route == "stream":
                _check_init(check, (call / "stdout.txt").read_text(encoding="utf-8"))
            if a.cli == "codex":
                _check_codex_effort(check, exe, call, a.model)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("all checks passed" if not check.failed else f"{check.failed} check(s) FAILED")
    return 1 if check.failed else 0


def _check_objects(check: _Checks, answer: dict[str, Any], expected: list[tuple[str, tuple[float, float]]],
                   sent: list[SentImage]) -> None:
    objects = [o for o in answer.get("objects") or [] if isinstance(o, dict)]
    cameras = {str(o.get("camera")) for o in objects}
    if not expected:
        check(bool(objects), "the model saw the images", f"{len(objects)} objects from cameras {sorted(cameras)}")
        return
    check(cameras >= {s.camera for s in sent}, f"every image reached the model ({len(sent)} images)",
          f"answers from {sorted(cameras)}")
    for camera, (u, v) in expected:
        near = [o for o in objects if o.get("camera") == camera
                and abs(float(o.get("u", 1e9)) - u) <= _TOLERANCE_PX and abs(float(o.get("v", 1e9)) - v) <= _TOLERANCE_PX]
        best = min((abs(float(o.get("u", 1e9)) - u) + abs(float(o.get("v", 1e9)) - v) for o in objects
                    if o.get("camera") == camera), default=None)
        check(bool(near), f"{camera}: an object within {_TOLERANCE_PX:.0f} px of ({u:.0f}, {v:.0f})",
              f"closest |du|+|dv| = {best:.0f} px" if best is not None else "nothing on that camera")


def _check_init(check: _Checks, stdout: str) -> None:
    init = next((e for e in cli._lines(stdout) if e.get("type") == "system" and e.get("subtype") == "init"), None)
    if init is None:
        check(False, "a system/init event on stream-json")
        return
    fields = {k: v for k, v in init.items() if any(s in k.lower() for s in ("key", "auth", "source"))}
    check(cli.CLAUDE_KEY_FIELD in init, f"init event names the API-key source in {cli.CLAUDE_KEY_FIELD!r}",
          json.dumps(fields))


def _check_codex_effort(check: _Checks, exe: str, call: Path, model: str) -> None:
    """An effort the server does not know must be refused: proof that `-c model_reasoning_effort`
    reaches the request with `--ignore-user-config`. The server refuses it before any model turn."""
    probe = call / "probe"
    probe.mkdir()
    (probe / "schema.json").write_text(json.dumps(SMOKE_SCHEMA), encoding="utf-8")
    cmd = cli.codex_command(exe, call=probe, sent=[], model=model, effort="no-such-effort")
    try:
        cli.run_cli(cmd, stdin="Answer with an empty objects list.", cwd=probe, timeout_s=60,
                    cancel=threading.Event(), name="codex")
    except cli.CliError as e:
        check("effort" in str(e).lower(), "-c model_reasoning_effort is read with --ignore-user-config",
              str(e)[:200])
        return
    check(False, "-c model_reasoning_effort is read with --ignore-user-config", "a bogus effort was accepted")


if __name__ == "__main__":
    sys.exit(main())
