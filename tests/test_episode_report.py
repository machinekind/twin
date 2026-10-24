"""The episode report on synthetic data: the delay scan, the gripper's drift and the missing cases.

The logs here are written by hand, line by line, so each test knows the truth it checks against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import BOT

from machinekind_twin.episode import FORMAT, VERSION, EpisodeLog
from machinekind_twin.episode_report import DelayFit, delay_fit, report
from machinekind_twin.robots import get_spec

SPEC = get_spec(BOT)
JOINTS = list(SPEC.joints)
#: Where the cameras put the marker against the readings' TCP [m]: readings minus cameras.
OFFSET = np.array([0.010, -0.005, 0.0])


def held(ref_t: np.ndarray, ref_v: np.ndarray, prior: float, at: np.ndarray) -> np.ndarray:
    """The reference as the twin holds it: the last value set strictly before, `prior` before the first."""
    index = np.searchsorted(ref_t, at, side="left") - 1
    return np.where(index >= 0, ref_v[np.clip(index, 0, None)], prior)


def swinging() -> tuple[np.ndarray, np.ndarray]:
    """A command as the loop sends it: a new value every 20 ms, swinging 20 degrees."""
    ref_t = np.arange(0.0, 3.0, 0.02)
    return ref_t, np.round(20.0 * np.sin(np.pi * ref_t), 1)


# --------------------------------------------------------------------------- the delay scan


def test_a_delayed_copy_of_a_stepped_reference_gives_its_delay():
    ref_t, ref_v = swinging()
    sample_t = np.arange(0.0, 3.0, 0.04)

    fit = delay_fit(ref_t, ref_v, 0.0, sample_t, held(ref_t, ref_v, 0.0, sample_t - 0.08))

    assert isinstance(fit, DelayFit)
    assert fit.lag_s == pytest.approx(0.08, abs=0.005)
    assert fit.band_s[0] <= fit.lag_s <= fit.band_s[1]
    assert fit.residual_rms == pytest.approx(0.0, abs=1e-9)
    assert fit.error_rms > 1.0
    assert not fit.at_bound


def test_a_still_joint_is_not_counted():
    ref_t = np.arange(0.0, 3.0, 0.02)
    sample_t = np.arange(0.0, 3.0, 0.04)

    fit = delay_fit(ref_t, np.full_like(ref_t, 5.0), 5.0, sample_t, np.full_like(sample_t, 5.2))

    assert isinstance(fit, str)


def test_a_delay_past_the_scan_is_flagged():
    ref_t, ref_v = swinging()
    sample_t = np.arange(0.0, 3.0, 0.04)

    fit = delay_fit(ref_t, ref_v, 0.0, sample_t, held(ref_t, ref_v, 0.0, sample_t - 1.2), max_delay_s=1.0)

    assert isinstance(fit, DelayFit)
    assert fit.at_bound
    assert fit.lag_s == pytest.approx(1.0)


def test_a_constant_offset_is_bias_not_delay():
    ref_t, ref_v = swinging()
    sample_t = np.arange(0.0, 3.0, 0.04)

    fit = delay_fit(ref_t, ref_v, 0.0, sample_t, held(ref_t, ref_v, 0.0, sample_t - 0.06) + 2.0)

    assert isinstance(fit, DelayFit)
    assert fit.lag_s == pytest.approx(0.06, abs=0.005)
    assert fit.bias == pytest.approx(2.0)


def test_samples_before_the_first_event_use_the_prior():
    """One step at 1 s from where the joint stood: readings that stay there for 0.1 s are the delay."""
    ref_t, ref_v = np.array([1.0]), np.array([10.0])
    sample_t = np.arange(1.0, 2.0, 0.01)

    fit = delay_fit(ref_t, ref_v, 0.0, sample_t, held(ref_t, ref_v, 0.0, sample_t - 0.1))

    assert isinstance(fit, DelayFit)
    assert fit.lag_s == pytest.approx(0.1, abs=0.005)


def test_too_few_samples_in_motion_give_no_delay():
    ref_t, ref_v = np.array([1.0]), np.array([10.0])
    sample_t = np.array([1.0, 1.1, 1.2, 5.0, 6.0])

    fit = delay_fit(ref_t, ref_v, 0.0, sample_t, np.full(5, 10.0))

    assert isinstance(fit, str)
    assert "(3)" in fit


# --------------------------------------------------------------------------- logs written by hand


def write_log(path: Path, loops: list[dict], steps: list[dict], **header) -> Path:
    lines = [{"kind": "header", "format": FORMAT, "version": VERSION, "name": path.stem, "robot": BOT,
              "joints": JOINTS, "gripper": SPEC.gripper, "simulated": True, **header}]
    lines += [{"kind": "loop", "at_limit": [], **loop} for loop in loops]
    lines += [{"kind": "step", "intent": {}, "objects": {}, "decisions": [], **step} for step in steps]
    end_t = max([loop["t"] for loop in loops] + [0.0])
    lines.append({"kind": "end", "t": end_t, "code": "stopped", "reason": "zatrzymana", "done": False,
                  "success": None, "error": None, "note": "", "steps": len(steps), "began": True})
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def pose_at(t: float) -> dict[str, float]:
    """The arm in the hand-written logs: the first two joints sweep for a second, then stop."""
    s = min(t, 1.0)
    return dict(SPEC.home, **{JOINTS[0]: 30.0 * s, JOINTS[1]: SPEC.home[JOINTS[1]] - 20.0 * s})


def readings_loops(seconds: float = 1.5, sent: bool = False) -> list[dict]:
    loops = []
    for t in np.arange(0.02, seconds, 0.02):
        pose = [round(pose_at(t)[joint], 6) for joint in JOINTS]
        loops.append({"t": float(t), "sent": pose if sent else None, "measured": pose, "measured_t": float(t)})
    return loops


@pytest.fixture
def tcp():
    pytest.importorskip("mujoco")
    from machinekind_twin.kinematics import RobotKinematics

    kinematics = RobotKinematics(SPEC)
    return lambda t: kinematics.tcp(pose_at(t))


def camera_steps(tcp, label: str = "chwytak", sources=None, rotation: bool = True, age_s: float = 0.03) -> list[dict]:
    steps = []
    for index, t in enumerate(np.arange(0.1, 1.45, 0.05)):
        T = tcp(t - age_s)
        source = "kamery" if sources is None else sources[index % len(sources)]
        pose = {"position": (T[:3, 3] - OFFSET).tolist(),
                "rotation": T[:3, :3].tolist() if rotation else np.eye(3).tolist(),
                "source": source, "age_s": age_s}
        steps.append({"step": index, "obs_t": float(t), "t": float(t), "objects": {label: pose}})
    return steps


def test_drift_is_readings_minus_the_cameras(tmp_path, tcp):
    path = write_log(tmp_path / "drift.jsonl", readings_loops(), camera_steps(tcp), gripper_label="chwytak")

    result = report(EpisodeLog.load(path))
    drift = result.drift

    assert drift is not None
    assert drift.samples == 27
    assert drift.mean_m == pytest.approx(OFFSET, abs=1e-5)
    assert drift.rms_m == pytest.approx(np.linalg.norm(OFFSET), abs=1e-5)
    assert drift.median_age_s == pytest.approx(0.03)
    assert drift.angle_max_deg == pytest.approx(0.0, abs=0.01)
    assert drift.at_rest_samples > 0
    assert any("odczyty - kamery" in line for line in result.lines())


def test_drift_skips_poses_that_are_not_camera_evidence(tmp_path, tcp):
    sources = ["kamery", "w dloni", "ostatnie widziane"]
    path = write_log(tmp_path / "mixed.jsonl", readings_loops(), camera_steps(tcp, sources=sources))

    drift = report(EpisodeLog.load(path), gripper="chwytak").drift

    assert drift is not None
    assert drift.samples == 9
    assert drift.skipped == {"w dloni": 9, "ostatnie widziane": 9}
    assert drift.mean_m == pytest.approx(OFFSET, abs=1e-5)


def test_no_label_means_no_drift_and_says_so(tmp_path, tcp):
    path = write_log(tmp_path / "unnamed.jsonl", readings_loops(), camera_steps(tcp))

    result = report(EpisodeLog.load(path))

    assert result.drift is None
    assert result.drift_note in result.lines()


def test_a_label_the_cameras_never_saw_means_no_drift(tmp_path, tcp):
    path = write_log(tmp_path / "unseen.jsonl", readings_loops(), camera_steps(tcp, sources=["w dloni"]))

    result = report(EpisodeLog.load(path), gripper="chwytak")

    assert result.drift is None
    assert "chwytak" in result.drift_note


def test_an_identity_rotation_means_position_only(tmp_path, tcp):
    path = write_log(tmp_path / "points.jsonl", readings_loops(), camera_steps(tcp, rotation=False))

    result = report(EpisodeLog.load(path), gripper="chwytak")

    assert result.drift is not None
    assert result.drift.angle_median_deg is None
    assert "  obrot: kamery podaja tylko polozenie" in result.lines()


def test_a_log_without_intent_or_command_says_it_does_not_count_latency(tmp_path):
    steps = [{"step": index, "obs_t": float(t), "t": float(t)} for index, t in enumerate(np.arange(0.1, 1.4, 0.1))]
    path = write_log(tmp_path / "watched.jsonl", readings_loops(), steps, mode="observation", strategy=None)

    result = report(EpisodeLog.load(path))

    assert result.latency == ()
    assert result.latency_note
    assert result.latency_note in result.lines()


def test_a_log_with_commands_but_no_intent_counts_only_the_servo(tmp_path):
    path = write_log(tmp_path / "commanded.jsonl", readings_loops(sent=True), [])

    result = report(EpisodeLog.load(path))
    moving = result.latency[0]

    assert result.latency_note in result.lines()
    assert isinstance(moving.total, str)
    assert isinstance(moving.actuator, DelayFit)
    assert moving.actuator.lag_s == pytest.approx(0.0, abs=0.025)


def test_decisions_are_counted_with_their_response_times(tmp_path):
    decisions = [{"question": "q", "answer": "a", "state": {}, "confidence": 0.5, "response_s": 1.0},
                 {"question": "q", "answer": "a", "state": {}, "confidence": None, "response_s": 3.0}]
    steps = [{"step": 0, "obs_t": 0.1, "t": 0.1, "decisions": decisions}]
    path = write_log(tmp_path / "asked.jsonl", readings_loops(), steps)

    stats = report(EpisodeLog.load(path)).decisions

    assert stats.count == 2
    assert stats.without_confidence == 1
    assert stats.response_median_s == pytest.approx(2.0)
    assert stats.response_max_s == pytest.approx(3.0)
    assert stats.confidence_mean == pytest.approx(0.5)
