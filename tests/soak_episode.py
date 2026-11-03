"""Synthetic collector soak, outside the regular test suite.

    uv run python tests/soak_episode.py --seconds 28800 --block-images
    uv run python tests/soak_episode.py --seconds 60 --realtime --frame-format webp

Accelerated mode flushes each ten-second batch to pace synthetic input. Neither
mode controls a robot. Use --output to retain the generated recordings.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from machinekind_twin_api import Frame

from machinekind_twin.episode import EpisodeEnd, EpisodeLog, EpisodeWriter, StepRecord


class Clock:
    virtual = True
    t = 10000.0

    def now(self):
        return self.t


class Twin:
    status = SimpleNamespace(at_limit=())

    def __init__(self):
        self.clock = Clock()

    def add_tick_listener(self, fn):
        return 1

    def remove_tick_listener(self, handle):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--block-images", action="store_true")
    parser.add_argument("--frame-format", choices=("jpeg", "webp"), default="jpeg")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    with tempfile.TemporaryDirectory(prefix="episode-soak-") as temporary:
        folder = args.output or Path(temporary)
        image = np.zeros((480, 640, 3), np.uint8)
        image[:, :, 0] = np.arange(640, dtype=np.uint16)[None, :] % 256
        image[:, :, 1] = np.arange(480, dtype=np.uint16)[:, None] % 256
        release = threading.Event()
        original = EpisodeWriter._keep

        def blocked(self, step, camera, pixels):
            release.wait()
            return original(self, step, camera, pixels)

        if args.block_images:
            EpisodeWriter._keep = blocked
        twin = Twin()
        writer = EpisodeWriter.create(folder, {
            "robot": "synthetic-soak", "joints": ["j1", "j2", "gripper"], "loop_hz": 50.0,
            "strategy": {"class": "synthetic-soak", "hz": 20.0, "time_limit_s": None},
            "cameras": ["a", "b", "c"],
        }, twin, name=f"soak-{args.seconds}s", keep_frames_s=0, frame_format=args.frame_format)
        started, steps = time.monotonic(), 0
        try:
            for tick in range(1, args.seconds * 100 + 1):
                if args.realtime:
                    time.sleep(max(0.0, started + tick * 0.01 - time.monotonic()))
                elapsed = tick / 100.0
                twin.clock.t = 10000.0 + elapsed
                joints = {"j1": round(20 * math.sin(elapsed), 4), "j2": 10.0, "gripper": 50.0}
                if tick % 2 == 0:
                    writer.on_tick(SimpleNamespace(t=twin.clock.t, sent=joints, measured=joints,
                                                   measured_t=twin.clock.t, fresh=True))
                if tick % 5 == 0:
                    frames = {name: Frame(image, 0.0) for name in ("a", "b", "c")} if tick % 100 == 0 else {}
                    writer.step(StepRecord(steps, elapsed, elapsed, elapsed, joints, joints, joints,
                                           None, {}, frames))
                    steps += 1
                if tick % 1000 == 0:
                    assert writer.flush(wait=True, timeout=15.0), writer.error
                if tick % 60000 == 0:
                    print(json.dumps({"virtual_minutes": elapsed / 60,
                                      "wall_seconds": round(time.monotonic() - started, 2)}), flush=True)
        finally:
            release.set()
            completed = writer.close(EpisodeEnd("done", "soak complete", float(args.seconds), True, True,
                                                0.0, "", steps, True), wait=True, timeout=15.0)
            EpisodeWriter._keep = original
            assert completed, writer.error
        status = EpisodeLog.summary(writer.path).recording
        assert status["telemetry_complete"], status
        assert status["loop_written"] == args.seconds * 50
        assert status["step_written"] == args.seconds * 20
        assert status["frames_requested"] == args.seconds * 3
        assert status["frames_pending"] == 0
        if args.realtime and not args.block_images:
            assert status["frames_complete"], status
        result = {"virtual_seconds": args.seconds, "wall_seconds": round(time.monotonic() - started, 3),
                  "compressed_bytes": writer.path.stat().st_size, "recording": status}
        if args.output:
            result["path"] = str(writer.path)
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
