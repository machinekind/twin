"""Command bookkeeping against 1000 recorded ticks of the training environment, IK result injected."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from lerobot_mp.twin.grasp.command import GraspCommand
from lerobot_mp.twin.grasp.contract import (ARM_BASE_WORLD, CEILING, HOVER, STEP_XYZ, TRAINING_FLOOR_Z,
                                            grasp_R, wrap)

FIX = Path(os.environ.get("GRASP_FIXTURES", Path(__file__).parent / "fixtures" / "grasp"))
RERECORD = "re-record the fixtures (docs/GRASP.md)"
KEYS = ("action", "p_cmd0", "yaw_cmd0", "q_cmd0", "p0", "g_cmd0", "closing0", "attempts0", "ik_pos", "ik_yaw",
        "ik_R", "ik_q0", "ik_q", "ik_ok", "p_cmd1", "yaw_cmd1", "q_cmd1", "g_cmd1", "closing1", "attempts1",
        "prev_action1", "ctrl_arm", "ctrl_grip", "obs12_18", "episode", "tick")
BASE = np.array(ARM_BASE_WORLD)


@pytest.fixture(scope="module")
def book() -> dict[str, np.ndarray]:
    path = FIX / "bookkeeping.npz"
    if not path.is_file():
        pytest.fail(f"{path} missing: {RERECORD}")
    with np.load(path) as z:
        absent = [k for k in KEYS if k not in z.files]
        if absent:
            pytest.fail(f"bookkeeping.npz has no {absent}: {RERECORD}")
        return {k: z[k] for k in KEYS}


@pytest.fixture(scope="module")
def replay(book):
    """Every tick through one GraspCommand per episode, carried from tick to tick like training."""
    rows, cmd = [], None
    for i in range(len(book["action"])):
        if cmd is None or book["tick"][i] == 0:
            assert book["tick"][i] == 0, "an episode must start at tick 0"
            cmd = GraspCommand(book["p0"][i] - BASE, float(book["yaw_cmd0"][i]), book["q_cmd0"][i],
                               floor_z=TRAINING_FLOOR_Z)
        s = cmd.state
        before = (s.p_cmd.copy(), s.yaw_cmd, s.q_cmd.copy(), s.g_cmd, s.closing, s.attempts)
        prop = cmd.propose(book["action"][i])
        ok = bool(book["ik_ok"][i])
        edge = cmd.commit(prop, book["ik_q"][i] if ok else None)
        after = (s.p_cmd.copy(), s.yaw_cmd, s.q_cmd.copy(), s.g_cmd, s.closing, s.attempts,
                 s.prev_action.copy())
        rows.append((i, before, prop, edge, after, cmd.obs_tail()))
    return rows


def test_the_state_carried_from_tick_to_tick_matches_training(book, replay):
    bad = []
    for i, (p, yaw, q, g, closing, attempts), *_ in replay:
        ok = (np.abs(p - (book["p_cmd0"][i] - BASE)).max() <= 1e-12
              and float(yaw) == book["yaw_cmd0"][i]
              and np.array_equal(q, book["q_cmd0"][i])
              and g == book["g_cmd0"][i] and closing == book["closing0"][i]
              and attempts == book["attempts0"][i])
        if not ok:
            bad.append(i)
    assert bad == []


def test_proposals_match_the_ik_call_of_training(book, replay):
    bad = []
    for i, _, prop, *_ in replay:
        # The recorded IK yaw is atan2 of the float32 yaw's rotation, in float64: one ulp off at most.
        ok = (np.abs(prop.p - (book["ik_pos"][i] - BASE)).max() <= 1e-12
              and isinstance(prop.yaw, np.float32)
              and prop.yaw == np.float32(book["ik_yaw"][i])
              and abs(wrap(float(prop.yaw) - float(book["ik_yaw"][i]))) <= 1e-12
              and np.abs(prop.R - book["ik_R"][i]).max() <= 1e-12
              and np.abs(prop.R - grasp_R(prop.yaw)).max() == 0.0
              and prop.action.dtype == np.float32
              and np.array_equal(prop.action, np.clip(book["action"][i], -1.0, 1.0))
              and book["ctrl_grip"][i] == 0.05 * prop.g)
        if not ok:
            bad.append(i)
    assert bad == []


def test_state_after_each_tick_matches_training(book, replay):
    bad = []
    for i, _, prop, edge, (p, yaw, q, g, closing, attempts, prev), tail in replay:
        ok = (np.abs(p - (book["p_cmd1"][i] - BASE)).max() <= 1e-12
              and float(yaw) == book["yaw_cmd1"][i]                   # float32 values, bit for bit
              and np.array_equal(q, book["q_cmd1"][i])
              and np.array_equal(q, book["ctrl_arm"][i])
              and g == book["g_cmd1"][i] and closing == book["closing1"][i]
              and attempts == book["attempts1"][i]
              and edge == (attempts == book["attempts0"][i] + 1)
              and np.array_equal(prev, book["prev_action1"][i])
              and np.array_equal(tail[2:], book["obs12_18"][i]))
        if not ok:
            bad.append(i)
    assert bad == []


def test_accepted_yaw_is_float32_arithmetic_not_float64(book, replay):
    accepted = [(i, prop) for i, _, prop, *_ in replay if book["ik_ok"][i]]
    assert all(isinstance(prop.yaw, np.float32) for _, prop in accepted)
    f64 = [wrap(float(book["yaw_cmd0"][i]) + float(np.clip(book["action"][i][3], -1, 1)) * math.radians(5.0))
           for i, _ in accepted]
    assert sum(f != float(prop.yaw) for f, (_, prop) in zip(f64, accepted)) > len(accepted) // 2


def test_box_floor_ceiling_hits_and_refused_steps_all_occur(book, replay):
    props = [prop for _, _, prop, *_ in replay]
    assert sum(p.box_hit for p in props) > 0
    assert sum(p.floor_hit for p in props) > 0
    assert sum(p.ceiling_hit for p in props) > 0
    assert (~book["ik_ok"]).sum() > 0
    assert sum(edge for *_, edge, _, _ in replay) > 0
    for i, _, prop, *_ in replay:
        p0 = book["p0"][i] - BASE
        assert np.all(np.abs(prop.p[:2] - p0[:2]) <= 0.08 + 1e-12)
        assert TRAINING_FLOOR_Z <= prop.p[2] <= p0[2] + CEILING + 1e-12


def test_the_observation_tail_equals_training_on_the_student_driven_ticks():
    path = FIX / "shard.npz"
    need = ("obs", "believed_pos", "believed_yaw", "yaw_cmd", "g_cmd", "prev_action", "attempts")
    if not path.is_file():
        pytest.fail(f"{path} missing: {RERECORD}")
    with np.load(path) as z:
        absent = [k for k in need if k not in z.files]
        if absent:
            pytest.fail(f"shard.npz has no {absent}: {RERECORD}")
        shard = {k: z[k] for k in need}
    bad = []
    for i in range(len(shard["obs"])):
        p_hover = shard["believed_pos"][i] - BASE + [0.0, 0.0, HOVER]
        cmd = GraspCommand(p_hover, float(shard["believed_yaw"][i]), np.zeros(6), floor_z=TRAINING_FLOOR_Z)
        s = cmd.state
        # Training's yaw_cmd is the hover's python float until a step is accepted, np.float32 after.
        yaw = float(shard["yaw_cmd"][i])
        s.yaw_cmd = yaw if yaw == s.believed_yaw else np.float32(yaw)
        s.g_cmd = float(shard["g_cmd"][i])
        s.prev_action = shard["prev_action"][i].copy()
        s.attempts = int(shard["attempts"][i])
        if not np.array_equal(cmd.obs_tail(), shard["obs"][i][10:19]):
            bad.append(i)
    assert bad == []
    assert (shard["yaw_cmd"] != shard["believed_yaw"]).sum() > len(shard["obs"]) // 2


def hover_command(floor_z: float = TRAINING_FLOOR_Z) -> GraspCommand:
    return GraspCommand(np.array([0.0, 0.33, 0.099]), 1.2, np.array([1.5, 2.0, -1.2, 0.7, 0.0, 0.8]),
                        floor_z=floor_z)


def test_a_new_command_starts_at_the_hover_with_the_tick0_tail():
    cmd = hover_command()
    s = cmd.state
    np.testing.assert_array_equal(s.p_cmd, s.p0)
    np.testing.assert_allclose(s.believed_pos, s.p0 - [0, 0, HOVER], rtol=0, atol=1e-15)
    assert type(s.yaw_cmd) is float and s.believed_yaw == s.yaw_cmd
    tail = cmd.obs_tail()
    assert tail.dtype == np.float32 and tail.shape == (9,)
    np.testing.assert_array_equal(tail, [0, 1, 1, 0, 0, 0, 0, 0, 0])


def test_a_refused_ik_keeps_p_yaw_and_q_but_updates_g_attempts_and_prev_action():
    cmd = hover_command()
    s = cmd.state
    p, yaw, q = s.p_cmd.copy(), s.yaw_cmd, s.q_cmd.copy()
    prop = cmd.propose(np.array([1.0, -1.0, -1.0, 1.0, -1.0], np.float32))
    assert cmd.commit(prop, None) is True
    np.testing.assert_array_equal(s.p_cmd, p)
    assert s.yaw_cmd == yaw
    np.testing.assert_array_equal(s.q_cmd, q)
    assert s.g_cmd == 0.0 and s.closing and s.attempts == 1 and s.ticks == 1
    np.testing.assert_array_equal(s.prev_action, prop.action)
    # Still closing: no second attempt; opening and closing again is one.
    assert cmd.commit(cmd.propose(np.array([0, 0, 0, 0, -0.9], np.float32)), None) is False
    assert cmd.commit(cmd.propose(np.array([0, 0, 0, 0, 1.0], np.float32)), None) is False
    assert cmd.commit(cmd.propose(np.array([0, 0, 0, 0, -1.0], np.float32)), None) is True
    assert s.attempts == 2
    assert abs(cmd.obs_tail()[8] - 2 / 3) < 1e-7


def test_an_accepted_step_moves_the_command_by_the_step_from_the_command():
    cmd = hover_command()
    s = cmd.state
    q_new = s.q_cmd + 0.01
    prop = cmd.propose(np.array([1.0, 0.5, -0.25, 1.0, 1.0], np.float32))
    assert cmd.commit(prop, q_new) is False
    step = np.array([1.0, 0.5, -0.25], np.float32) * STEP_XYZ          # float32 product, as in training
    assert step.dtype == np.float32
    np.testing.assert_array_equal(s.p_cmd, s.p0 + step)
    assert np.abs(s.p_cmd - s.p0 - np.array([1.0, 0.5, -0.25]) * STEP_XYZ).max() < 1e-9
    assert isinstance(s.yaw_cmd, np.float32)
    assert abs(float(s.yaw_cmd) - (1.2 + math.radians(5.0))) < 1e-6
    np.testing.assert_array_equal(s.q_cmd, q_new)
    tail = cmd.obs_tail()
    assert abs(tail[0] - math.sin(2 * math.radians(5.0))) < 1e-6
    assert abs(tail[1] - math.cos(2 * math.radians(5.0))) < 1e-6


@pytest.mark.parametrize("bad", [
    [math.nan, 0, 0, 0, 0], [0, 0, 0, 0, math.inf], [0, 0, 0, -math.inf, 0], [0, 0, 0, 0], [[0, 0, 0, 0, 0]],
])
def test_a_non_finite_or_misshaped_action_raises_and_changes_nothing(bad):
    cmd = hover_command()
    before = (cmd.state.p_cmd.copy(), cmd.state.ticks)
    with pytest.raises(ValueError):
        cmd.propose(np.array(bad, dtype=np.float64))
    np.testing.assert_array_equal(cmd.state.p_cmd, before[0])
    assert cmd.state.ticks == before[1]


def test_a_non_finite_ik_result_raises_before_anything_changes():
    cmd = hover_command()
    prop = cmd.propose(np.zeros(5, np.float32))
    with pytest.raises(ValueError):
        cmd.commit(prop, np.array([0, 0, math.nan, 0, 0, 0.0]))
    assert cmd.state.ticks == 0


def test_raising_the_floor_between_ticks_lifts_the_next_proposal_to_it():
    cmd = hover_command()
    s = cmd.state
    for _ in range(12):                                   # walk down to the floor
        prop = cmd.propose(np.array([0, 0, -1.0, 0, 1.0], np.float32))
        cmd.commit(prop, s.q_cmd)
    assert prop.floor_hit and s.p_cmd[2] == TRAINING_FLOOR_Z
    cmd.floor_z = 0.031
    prop = cmd.propose(np.zeros(5, np.float32))
    assert prop.p[2] == 0.031 and prop.floor_hit
    cmd.commit(prop, s.q_cmd)
    assert s.p_cmd[2] == 0.031


def test_a_floor_above_the_ceiling_is_refused():
    with pytest.raises(ValueError):
        hover_command(floor_z=0.099 + CEILING + 1e-6)
    cmd = hover_command()
    cmd.floor_z = 0.2
    with pytest.raises(ValueError):
        cmd.propose(np.zeros(5, np.float32))
