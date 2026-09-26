"""Render saved A1X candidates on CPU; this never connects to hardware."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from lerobot_mp.twin.rl.evaluate import evaluate
from lerobot_mp.twin.rl.policy import Policy
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.workspace import Workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=32000)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 20:
        parser.error("episodes must be in [1, 20]")
    policy = Policy.load(args.policy)
    if policy.meta.robot != "a1x":
        parser.error("Expected an A1X checkpoint")
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in range(args.seed, args.seed + args.episodes):
        writer = None
        frame_count = 0

        def frame(rgb):
            nonlocal writer, frame_count
            if rgb is None:
                raise RuntimeError("The evaluation scene has no render camera")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if writer is None:
                writer = cv2.VideoWriter(str(args.out / f"seed-{seed}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         policy.task.control_hz, (bgr.shape[1], bgr.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("Could not open video output")
            cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 47), (20, 20, 20), -1)
            for y, text in ((18, f"A1X learned {policy.task.name} | CPU simulation | seed {seed}"),
                            (38, f"{frame_count / policy.task.control_hz:.2f} s | provisional model, no hardware validation")):
                cv2.putText(bgr, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .42, (240, 240, 240), 1, cv2.LINE_AA)
            writer.write(bgr)
            frame_count += 1

        try:
            result = evaluate(policy, episodes=1, workspace=Workspace(robot="a1x"),
                              randomization=Randomization.none(), seed=seed, on_frame=frame)
            result.update(seed=seed, frames=frame_count)
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            if writer is not None:
                writer.release()
            (args.out / "report.json").write_text(json.dumps({
                "policy": str(args.policy), "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                "simulation_only": True, "path": "saved policy -> TwinEnv (CPU)", "episodes": results,
            }, indent=2) + "\n")


if __name__ == "__main__":
    main()
