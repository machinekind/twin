"""One real question to a real model: does it find a red cube in a rendered frame?

Skipped unless `MACHINEKIND_TWIN_LLM` names the CLI to ask (claude or codex). It runs on your
subscription, sends one frame off the machine and takes seconds to minutes. Optional:
`MACHINEKIND_TWIN_LLM_EXECUTABLE` (e.g. ~/.local/bin/codex, to bypass terminal wrappers) and
`MACHINEKIND_TWIN_LLM_MODEL` (default: the CLI's own default for claude, gpt-6-astra for codex).

    MACHINEKIND_TWIN_LLM=codex uv run pytest -q packages/machinekind_twin_llm/tests/test_llm_real.py

The frame comes from a minimal MuJoCo scene (a grey table, a 4 cm red cube, one camera), so the
test needs MuJoCo and OpenGL but not the twin: the package's tests stay as pure as the package.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from machinekind_twin_api import DetectorInfo, DetectRequest, Frame
from machinekind_twin_llm import LlmDetector, LlmSettings

pytestmark = pytest.mark.llm

W, H = 640, 480
FOVY_DEG = 50.0
EYE = np.array([0.05, -0.25, 0.45])
#: The cube's footprint centre on the table: what the prompt asks the model to point at.
CUBE = np.array([0.03, 0.02, 0.0])
HALF = 0.02
#: How far off the model's pixel may be [px].
TOLERANCE_PX = 40.0


def camera_axes(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MuJoCo `xyaxes` of a camera at `eye` looking at `target`: x right, y up, looking along -z."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(forward, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(-forward, x)
    return x, y


def render() -> tuple[np.ndarray, tuple[float, float]]:
    """The frame (RGB uint8) and where the cube's footprint centre is in it (OpenCV pixel centres)."""
    mujoco = pytest.importorskip("mujoco")
    x, y = camera_axes(EYE, CUBE)
    xml = f"""
    <mujoco>
      <visual><global offwidth="{W}" offheight="{H}"/></visual>
      <worldbody>
        <light pos="0 0 2" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba="0.62 0.6 0.58 1"/>
        <geom type="box" size="{HALF} {HALF} {HALF}" pos="{CUBE[0]} {CUBE[1]} {HALF}" rgba="0.85 0.2 0.2 1"/>
        <camera name="top" pos="{' '.join(map(str, EYE))}" xyaxes="{' '.join(map(str, [*x, *y]))}" fovy="{FOVY_DEG}"/>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, H, W) as renderer:
        renderer.update_scene(data, "top")
        image = renderer.render().copy()

    # Pinhole projection of the footprint centre: MuJoCo's camera looks along -z with y up.
    R, pos = data.cam_xmat[0].reshape(3, 3), data.cam_xpos[0]
    pc = R.T @ (CUBE - pos)
    f = (H / 2) / math.tan(math.radians(FOVY_DEG) / 2)
    return image, ((W - 1) / 2 + f * pc[0] / -pc[2], (H - 1) / 2 - f * pc[1] / -pc[2])


def test_a_real_model_finds_the_red_cube():
    cli = os.environ.get("MACHINEKIND_TWIN_LLM", "")
    if cli not in ("claude", "codex"):
        pytest.skip("set MACHINEKIND_TWIN_LLM=claude or codex to ask a real model")
    image, (u, v) = render()
    model = os.environ.get("MACHINEKIND_TWIN_LLM_MODEL", "" if cli == "claude" else "gpt-6-astra")
    detector = LlmDetector(LlmSettings(cli=cli, model=model, objects="red cube",
                                       executable=os.environ.get("MACHINEKIND_TWIN_LLM_EXECUTABLE", ""),
                                       timeout_s=300.0))
    detector.start(DetectorInfo(cameras=("top",), robot="test", simulated=True))
    try:
        found = detector.detect(DetectRequest(frames={"top": Frame(image=image, age_s=0.0)}, step=0))
    finally:
        detector.stop("test")

    assert found, "the model found no red cube"
    best = max(found, key=lambda d: d.confidence)
    assert best.label == "red cube" and best.camera == "top"
    miss = math.hypot(best.pixel[0] - u, best.pixel[1] - v)
    assert miss <= TOLERANCE_PX, f"the model put the cube at {best.pixel}, {miss:.0f} px from ({u:.0f}, {v:.0f})"
