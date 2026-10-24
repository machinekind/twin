"""The episode report on synthetic data: the delay scan, the gripper's drift and the missing cases.

The logs here are written by hand, line by line, so each test knows the truth it checks against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import BOT

from machinekind_twin.cli import main as twin_cli
from machinekind_twin.episode import FORMAT, VERSION, EpisodeLog
from machinekind_twin.episode_report import (
    FOLLOWS,
    NEITHER,
    NOTHING_SENT,
    ONLY_ACTUATOR,
    ONLY_TOTAL,
    SPEED_HALF_S,
    DelayFit,
    delay_fit,
    report,
)
from machinekind_twin.robots import get_spec

SPEC = get_spec(BOT)
JOINTS = list(SPEC.joints)
#: Where the cameras put the marker against the readings' TCP [m]: readings minus cameras.
OFFSET = np.array([0.010, -0.005, 0.0])
#: A marker 20 mm along the TCP's z axis [m], in the TCP frame.
MARKER = np.array([0.0, 0.0, 0.020])
#: When the hand-written arm stops [s].
SWEEP_S = 1.0


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


def test_a_reference_that_follows_its_samples_gives_no_delay():
    """A target copied from the readings every 50 ms while the joint creeps: the readings lead it."""
    ref_t = np.arange(0.0, 3.0, 0.05)
    sample_t = np.arange(0.0, 3.0, 0.02)

    fit = delay_fit(ref_t, 2.0 * ref_t, 0.0, sample_t, 2.0 * sample_t)

    assert fit == FOLLOWS


def test_a_reference_the_samples_take_up_at_once_gives_zero():
    """A command that is the target from the next loop tick on: the scan's lowest shift fits exactly."""
    ref_t = np.arange(0.0, 3.0, 0.05)
    ref_v = np.where(np.arange(len(ref_t)) // 5 % 2 == 0, 10.0, -10.0)
    sample_t = np.arange(0.01, 3.0, 0.02)

    fit = delay_fit(ref_t, ref_v, 0.0, sample_t, held(ref_t, ref_v, 0.0, sample_t))

    assert isinstance(fit, DelayFit)
    assert fit.lag_s == 0.0
    assert fit.error_rms == 0.0


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
    s = min(t, SWEEP_S)
    return dict(SPEC.home, **{JOINTS[0]: 30.0 * s, JOINTS[1]: SPEC.home[JOINTS[1]] - 20.0 * s})


def readings_loops(seconds: float = 1.5) -> list[dict]:
    loops = []
    for t in np.arange(0.02, seconds, 0.02):
        pose = [round(pose_at(t)[joint], 6) for joint in JOINTS]
        loops.append({"t": float(t), "sent": None, "measured": pose, "measured_t": float(t)})
    return loops


#: The loop's period in the hand-written logs [s].
LOOP_S = 0.02
HOME = np.array([SPEC.home[joint] for joint in JOINTS])


def chain(supervisor_s: float, servo_s: float, seconds: float = 3.0) -> tuple[list[dict], list[dict]]:
    """Loop and step lines with known delays on the first joint. The intent is a square wave set
    every 50 ms; the command is the held intent `supervisor_s` later and the readings are the held
    command `servo_s` later, on a 20 ms loop. The other joints stand at home."""
    intent_t = np.round(np.arange(0.05, seconds - 0.5, 0.05), 6)
    intent_v = np.where(np.arange(len(intent_t)) // 5 % 2 == 0, 10.0, -10.0)
    loop_t = np.round(np.arange(1, round(seconds / LOOP_S)) * LOOP_S, 6)

    sent = np.tile(HOME, (len(loop_t), 1))
    sent[:, 0] = held(intent_t, intent_v, HOME[0], loop_t - supervisor_s)
    measured = np.tile(HOME, (len(loop_t), 1))
    measured[:, 0] = held(loop_t, sent[:, 0], HOME[0], loop_t - servo_s)

    loops = [{"t": float(t), "sent": command.tolist(), "measured": reading.tolist(), "measured_t": float(t)}
             for t, command, reading in zip(loop_t, sent, measured, strict=True)]
    steps = [{"step": index, "obs_t": float(t), "t": float(t), "intent": {JOINTS[0]: float(value)}}
             for index, (t, value) in enumerate(zip(intent_t, intent_v, strict=True))]
    return loops, steps


@pytest.fixture
def tcp():
    pytest.importorskip("mujoco")
    from machinekind_twin.kinematics import RobotKinematics

    kinematics = RobotKinematics(SPEC)
    return lambda t: kinematics.tcp(pose_at(t))


#: When the cameras saw the marker in the hand-written logs [s].
SEEN_T = np.arange(0.1, 1.45, 0.05)


def camera_steps(tcp, label: str = "chwytak", sources=None, rotation: bool = True, age_s: float = 0.03,
                 marker=(0.0, 0.0, 0.0)) -> list[dict]:
    """Steps whose camera pose of `label` is the marker on the readings' TCP minus `OFFSET`."""
    steps = []
    for index, t in enumerate(SEEN_T):
        T = tcp(t - age_s)
        source = "kamery" if sources is None else sources[index % len(sources)]
        pose = {"position": (T[:3, :3] @ np.asarray(marker) + T[:3, 3] - OFFSET).tolist(),
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
    assert drift.at_rest_samples == np.sum(SEEN_T - 0.03 - SPEED_HALF_S >= SWEEP_S)
    assert drift.at_rest_rms_m == pytest.approx(np.linalg.norm(OFFSET), abs=1e-5)
    assert any("odczyty - kamery" in line for line in result.lines())


def test_the_marker_offset_is_in_the_tcp_frame(tmp_path, tcp):
    assert not np.allclose(tcp(0.5)[:3, :3] @ MARKER, MARKER)
    path = write_log(tmp_path / "marker.jsonl", readings_loops(), camera_steps(tcp, marker=MARKER),
                     gripper_label="chwytak")
    log = EpisodeLog.load(path)

    at_marker = report(log, offset_m=MARKER).drift
    at_tcp = report(log).drift

    assert at_marker is not None and at_tcp is not None
    assert at_marker.mean_m == pytest.approx(OFFSET, abs=1e-5)
    assert np.linalg.norm(at_tcp.mean_m - OFFSET) > 0.015


def test_twin_episode_takes_the_gripper_label_and_the_marker_offset(tmp_path, tcp, capsys):
    path = write_log(tmp_path / "marker.jsonl", readings_loops(), camera_steps(tcp, marker=MARKER))

    def drift_lines(*flags: str) -> list[str]:
        assert twin_cli(["episode", str(path), *flags]) == 0
        return [line for line in capsys.readouterr().out.splitlines() if "odczyty - kamery" in line]

    named = drift_lines("--gripper", "chwytak", "--offset-mm", *(f"{value * 1000:g}" for value in MARKER))

    assert drift_lines() == []
    assert len(named) == 1
    mean_mm = [float(value) for value in named[0].split("[")[1].split("]")[0].split(",")]
    assert mean_mm == pytest.approx(OFFSET * 1000, abs=0.1)


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


def test_each_pair_measures_its_own_link_of_the_chain(tmp_path):
    path = write_log(tmp_path / "chain.jsonl", *chain(supervisor_s=0.10, servo_s=0.06))

    result = report(EpisodeLog.load(path))
    moving = result.latency[0]

    assert result.latency_note == ""
    assert isinstance(moving.supervisor, DelayFit)
    assert isinstance(moving.actuator, DelayFit)
    assert isinstance(moving.total, DelayFit)
    assert moving.supervisor.lag_s == pytest.approx(0.10, abs=0.005)
    assert moving.actuator.lag_s == pytest.approx(0.06, abs=0.005)
    assert 0.16 <= moving.total.lag_s <= 0.185


def test_a_log_without_intent_or_command_says_it_does_not_count_latency(tmp_path):
    steps = [{"step": index, "obs_t": float(t), "t": float(t)} for index, t in enumerate(np.arange(0.1, 1.4, 0.1))]
    path = write_log(tmp_path / "watched.jsonl", readings_loops(), steps, mode="observation", strategy=None)

    result = report(EpisodeLog.load(path))

    assert result.latency == ()
    assert result.latency_note == NEITHER
    assert result.latency_note in result.lines()


def test_a_log_with_commands_but_no_intent_counts_only_the_servo(tmp_path):
    loops, _ = chain(supervisor_s=0.10, servo_s=0.06)
    path = write_log(tmp_path / "commanded.jsonl", loops, [])

    result = report(EpisodeLog.load(path))
    moving = result.latency[0]

    assert result.latency_note == ONLY_ACTUATOR
    assert result.latency_note in result.lines()
    assert isinstance(moving.total, str)
    assert isinstance(moving.actuator, DelayFit)
    assert moving.actuator.lag_s == pytest.approx(0.06, abs=0.005)


def test_a_log_with_intents_but_no_command_counts_only_intent_to_readings(tmp_path):
    """A read-only backend that mirrors a foreign controller's command into the intent (#175)."""
    loops, steps = chain(supervisor_s=0.10, servo_s=0.06)
    path = write_log(tmp_path / "mirrored.jsonl", [dict(loop, sent=None) for loop in loops], steps)

    result = report(EpisodeLog.load(path))
    moving = result.latency[0]

    assert result.latency_note == ONLY_TOTAL
    assert result.latency_note in result.lines()
    assert isinstance(moving.total, DelayFit)
    assert 0.16 <= moving.total.lag_s <= 0.185
    assert moving.supervisor == moving.actuator == NOTHING_SENT


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
