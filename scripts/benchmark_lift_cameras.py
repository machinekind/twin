"""Run the existing image-to-grasp harness over several calibrated camera layouts.

Requires development dependencies. This tests perfect calibration, not calibration
recovery after moving a real camera. Results are independent of checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LAYOUTS = {
    "reference_two": {"eyes": [[.55, -.45, .45], [.5, .5, .5]], "focal": 560.},
    "wide_two": {"eyes": [[.65, -.65, .65], [.65, .65, .65]], "focal": 560.},
    "high_two": {"eyes": [[.3, -.35, .8], [.3, .35, .8]], "focal": 560.},
    "low_two": {"eyes": [[.55, -.45, .22], [.5, .5, .25]], "focal": 560.},
    "single_camera": {"eyes": [[.55, -.45, .45]], "focal": 560.},
    "wide_angle_three": {"eyes": [[.55, -.45, .45], [.5, .5, .5], [.7, .05, .6]], "focal": 400.},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[5, 17, 29])
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    run_case = runpy.run_path(str(ROOT / "tests/test_twin_lift_from_cameras.py"))["run_camera_case"]
    report = {"policy": str(args.policy.resolve()), "perfect_calibration": True,
              "resolution": [640, 480], "layouts": LAYOUTS, "results": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, config in LAYOUTS.items():
        rows = []
        report["results"][name] = rows
        for seed in args.seeds:
            result = run_case(args.policy, seed=seed, **config)
            rows.append(result)
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{name} seed={seed}: success={result['success']}, "
                  f"finished={result['finished']}, {result['height_mm']:.1f}mm, {result['reason']}", flush=True)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
