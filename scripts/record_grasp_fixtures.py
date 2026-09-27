#!/usr/bin/env python3
"""Record the grasp driver's test fixtures from the training environment.

The training repository (galaxeo-manipulators) is only read: its code is imported, nothing is
written into it and no bytecode is cached there. Run it with the training repository's own
environment, from a scratch directory, never from inside that repository:

    mkdir -p <scratch>/fixtures_out && cd <scratch>/fixtures_out && PYTHONDONTWRITEBYTECODE=1 \\
      <galaxeo>/sim/.venv-lerobot/bin/python <twin>/scripts/record_grasp_fixtures.py \\
      --galaxeo <galaxeo> --out <scratch>/fixtures_out \\
      --seed 50000 --shard-episodes 2 --shard-ticks 40 --book 1000 --proj 300 --fk 300 --hover 200 --ik 200
    cp <scratch>/fixtures_out/*.npz <scratch>/fixtures_out/*.json <twin>/tests/fixtures/grasp/

`--galaxeo` defaults to $GALAXEO_REPO. What each file holds: tests/fixtures/grasp/README.md.

After `mj_step`, MuJoCo's positions (site_xpos, xpos, xmat, the rendered image, believed_uv, the
TCP inside the student observation) belong to the state BEFORE the last 2 ms substep, while qpos
belongs to the state after it. The `*_fk` fields are what training's own model gives AT the
recorded joints (a scratch MjData and mj_kinematics), which is what the twin computes.

The twin never imports this script, and neither does any test.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# The training repository stays untouched even when PYTHONDONTWRITEBYTECODE was forgotten.
sys.dont_write_bytecode = True

CHECKPOINT = "sim/grasp/student_v4.pt"          # relative to the training repository
# The published model's camera; training's defaults (320x240 or 128x96, 10 mm jitter) do not describe it.
WRIST = dict(wrist_size=(160, 120), wrist_jitter=(3.0, 0.7, 0.3))
IK_HOVERS = 20
IK_LIMIT_EVERY = 5                              # every fifth hover has one step that needs J6 past its range
IK_LIMIT_BEYOND = math.radians(10.0)
CALIBRATION_JOINT_OFFSETS_DEG = [0.0] * 6       # the wrist camera pose was calibrated on raw joint angles

mujoco = torch = genv = WRIST_SPEC = Student = heat_channel = BASE = None


def import_training(repo: Path) -> None:
    """Import the training code (read only) into this module's globals."""
    global mujoco, torch, genv, WRIST_SPEC, Student, heat_channel, BASE
    sys.path.insert(0, str(repo / "sim"))
    import mujoco
    import torch
    from grasp import env as genv
    from grasp.collect import WRIST_SPEC
    from grasp.student import Student, heat_channel
    from pickplace_scene import ARM_BASE
    BASE = np.array(ARM_BASE, float)


def inside(path: Path, repo: Path) -> bool:
    return path == repo or repo in path.parents


def repo_head(repo: Path) -> dict[str, str | None]:
    """Branch and commit of the training repository, read from its .git files: git is never run there.
    Uncommitted changes are not seen."""
    git = repo / ".git"
    if git.is_file():                       # a worktree: .git names the real directory
        git = (repo / git.read_text().split(":", 1)[1].strip()).resolve()
    if not (git / "HEAD").is_file():
        return dict(branch=None, commit=None)
    common = git
    if (git / "commondir").is_file():
        common = (git / (git / "commondir").read_text().strip()).resolve()
    head = (git / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return dict(branch=None, commit=head)
    ref = head[5:]
    branch = ref.removeprefix("refs/heads/")
    for base in (git, common):
        if (base / ref).is_file():
            return dict(branch=branch, commit=(base / ref).read_text().strip())
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return dict(branch=branch, commit=parts[0])
    return dict(branch=branch, commit=None)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gl_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    b = env.m.body("arm/gripper_link").id
    return env.d.xpos[b].copy(), env.d.xmat[b].reshape(3, 3).copy()


def at_joints(env: Any, scratch: dict[int, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TCP and gripper_link poses (world) of training's model at the current qpos: (tcp, tcp_R, gl_pos, gl_R).

    `scratch` holds one MjData per pool scene of this environment (the scenes differ in their objects)."""
    s = scratch.get(id(env.m))
    if s is None:
        s = scratch[id(env.m)] = mujoco.MjData(env.m)
    s.qpos[:] = env.d.qpos
    s.mocap_pos[:] = env.d.mocap_pos
    s.mocap_quat[:] = env.d.mocap_quat
    mujoco.mj_kinematics(env.m, s)
    gb = env.m.body("arm/gripper_link").id
    return (s.site_xpos[env.arm.site].copy(), s.site_xmat[env.arm.site].reshape(3, 3).copy(),
            s.xpos[gb].copy(), s.xmat[gb].reshape(3, 3).copy())


def believed_uv_at(env: Any, tw: np.ndarray, Rw: np.ndarray) -> np.ndarray:
    """The body of `GraspEnv.believed_uv` with the gripper_link pose given."""
    if not hasattr(env, "_cam_nominal"):
        env.believed_uv()
    pos_g, R_g, fovy = env._cam_nominal
    cam_pos, cam_R = tw + Rw @ pos_g, Rw @ R_g
    local = cam_R.T @ (env.believed_pos - cam_pos)
    W, H = env.wrist_size
    f = (H / 2) / np.tan(np.radians(fovy) / 2)
    if local[2] <= 1e-3:
        return np.array([np.nan, np.nan], np.float32)
    return np.array([f * local[0] / local[2] + W / 2 - 0.5,
                     f * local[1] / local[2] + H / 2 - 0.5], np.float32)


def student_obs_at(env: Any, tcp: np.ndarray) -> np.ndarray:
    """The body of `GraspEnv.student_obs` with the TCP position given."""
    q = env.d.qpos[env.arm.qadr]
    qn = 2.0 * (q - env.arm.lo) / (env.arm.hi - env.arm.lo) - 1.0
    e_yaw = genv._wrap(env.yaw_cmd - env.believed_yaw)
    parts = [qn, [env.arm.gap(env.d) * 10.0], (env.believed_pos - tcp) * 10.0,
             [math.sin(2 * e_yaw), math.cos(2 * e_yaw)], [env.g_cmd], env.prev_action,
             [env.attempts / 3.0]]
    return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts]).astype(np.float32)


def fk_fields(env: Any, scratch: dict[int, Any], obs: np.ndarray, uv: np.ndarray) -> dict[str, np.ndarray]:
    """The `*_fk` fields at the current qpos, after checking that the two copies above still reproduce
    training's own `student_obs` and `believed_uv` from the lagging poses."""
    tcp, _ = env.arm.tcp_pose(env.d)
    glp, glR = gl_pose(env)
    if not np.array_equal(student_obs_at(env, tcp), obs):
        raise SystemExit("student_obs_at no longer reproduces GraspEnv.student_obs: update the copy")
    if not np.array_equal(believed_uv_at(env, glp, glR), uv, equal_nan=True):
        raise SystemExit("believed_uv_at no longer reproduces GraspEnv.believed_uv: update the copy")
    tcp_fk, tcp_R_fk, gl_pos_fk, gl_R_fk = at_joints(env, scratch)
    obs_fk = student_obs_at(env, tcp_fk)
    others = np.r_[0:7, 10:19]
    if not np.array_equal(obs_fk[others], obs[others]):
        raise SystemExit("obs_fk differs from obs outside indices 7..9")
    return dict(tcp_fk=tcp_fk, tcp_R_fk=tcp_R_fk, gl_pos_fk=gl_pos_fk, gl_R_fk=gl_R_fk,
                uv_fk=believed_uv_at(env, gl_pos_fk, gl_R_fk), obs_fk=obs_fk)


def lag(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0


def record_shard(a: argparse.Namespace, st: Any) -> dict[str, float]:
    env = genv.GraspEnv(seed=a.seed, pool=a.shard_episodes, wrist=WRIST_SPEC, composites=0.5,
                        disturb_p=0.3, force_p=0.3, **WRIST)
    scratch: dict[int, Any] = {}
    rec = {k: [] for k in ("img", "obs", "uv", "act", "q", "gap", "tcp", "tcp_R", "believed_pos",
                           "believed_yaw", "yaw_cmd", "g_cmd", "prev_action", "attempts", "gl_pos", "gl_R",
                           "episode", "tick", "tcp_fk", "tcp_R_fk", "gl_pos_fk", "gl_R_fk", "uv_fk",
                           "obs_fk")}
    for ep in range(a.shard_episodes):
        env.reset(seed=a.seed + ep)
        done, t = False, 0
        while not done and t < a.shard_ticks:
            img, so, uv = env.render_wrist(), env.student_obs(), env.believed_uv()
            act = st.act(img[None], so[None], uv[None])[0]
            tcp, R = env.arm.tcp_pose(env.d)
            glp, glR = gl_pose(env)
            for k, v in (("img", img), ("obs", so), ("uv", uv), ("act", act),
                         ("q", env.d.qpos[env.arm.qadr].copy()), ("gap", env.arm.gap(env.d)), ("tcp", tcp),
                         ("tcp_R", R), ("believed_pos", env.believed_pos), ("believed_yaw", env.believed_yaw),
                         ("yaw_cmd", env.yaw_cmd), ("g_cmd", env.g_cmd),
                         ("prev_action", env.prev_action.copy()), ("attempts", env.attempts), ("gl_pos", glp),
                         ("gl_R", glR), ("episode", a.seed + ep), ("tick", t),
                         *fk_fields(env, scratch, so, uv).items()):
                rec[k].append(np.asarray(v).copy())
            _, _, term, trunc, info = env.step(act)
            done, t = term or trunc, t + 1
        print(f"shard episode {a.seed + ep}: {t} ticks, outcome {info['outcome'] or '(cut)'}, "
              f"attempts {info['attempts']}")
    env.close()
    out = {k: np.stack(v) for k, v in rec.items()}
    out["img"] = out["img"].astype(np.uint8)
    for k in ("obs", "uv", "act", "uv_fk", "obs_fk"):
        out[k] = out[k].astype(np.float32)
    np.savez_compressed(os.path.join(a.out, "shard.npz"), **out)
    stats = dict(obs=lag(out["obs"], out["obs_fk"]), uv_px=lag(out["uv"], out["uv_fk"]),
                 tcp_m=lag(out["tcp"], out["tcp_fk"]))
    print(f"shard: {len(out['obs'])} ticks, max |obs - obs_fk| {stats['obs']:.3e}, "
          f"max |uv - uv_fk| {stats['uv_px']:.3f} px, max |tcp - tcp_fk| {stats['tcp_m'] * 1e3:.3f} mm")
    return stats


def record_bookkeeping(a: argparse.Namespace) -> None:
    env = genv.GraspEnv(seed=a.seed + 7, pool=4, composites=0.5, disturb_p=0.0, force_p=0.0)
    rng = np.random.default_rng(a.seed + 7)
    calls = []
    rec = {k: [] for k in ("action", "p_cmd0", "yaw_cmd0", "q_cmd0", "p0", "g_cmd0", "closing0",
                           "attempts0", "ik_pos", "ik_yaw", "ik_R", "ik_q0", "ik_q", "ik_ok", "p_cmd1",
                           "yaw_cmd1", "q_cmd1", "g_cmd1", "closing1", "attempts1", "prev_action1",
                           "ctrl_arm", "ctrl_grip", "obs12_18", "episode", "tick")}

    def wrap_ik(arm):
        orig = arm.ik

        def ik(q0, pos, R, iters=500, damping=0.05):
            q, ok = orig(q0, pos, R, iters=iters, damping=damping)
            calls.append((np.array(q0, float), np.array(pos, float), np.array(R, float), q.copy(), bool(ok),
                          iters))
            return q, ok
        arm.ik = ik

    ep, n = 0, 0
    while n < a.book:
        env.reset(seed=a.seed + 7000 + ep)
        wrap_ik(env.arm)
        done, t = False, 0
        while not done and n < a.book:
            act = rng.uniform(-1.3, 1.3, 5).astype(np.float32)
            pre = (env.p_cmd.copy(), float(env.yaw_cmd), env.q_cmd.copy(), env.p0.copy(), float(env.g_cmd),
                   bool(env.closing), int(env.attempts))
            calls.clear()
            _, _, term, trunc, _ = env.step(act)
            assert len(calls) == 1 and calls[0][5] == 60, calls
            q0, pos, R, q, ok, _ = calls[0]
            ik_yaw = math.atan2(R[1, 1], R[0, 1])
            vals = dict(action=act, p_cmd0=pre[0], yaw_cmd0=pre[1], q_cmd0=pre[2], p0=pre[3], g_cmd0=pre[4],
                        closing0=pre[5], attempts0=pre[6], ik_pos=pos, ik_yaw=ik_yaw, ik_R=R, ik_q0=q0,
                        ik_q=q, ik_ok=ok, p_cmd1=env.p_cmd.copy(), yaw_cmd1=float(env.yaw_cmd),
                        q_cmd1=env.q_cmd.copy(),
                        g_cmd1=float(env.g_cmd), closing1=bool(env.closing), attempts1=int(env.attempts),
                        prev_action1=env.prev_action.copy(), ctrl_arm=env.d.ctrl[env.arm.acts].copy(),
                        ctrl_grip=float(env.d.ctrl[env.arm.grip]), obs12_18=env.student_obs()[12:19].copy(),
                        episode=a.seed + 7000 + ep, tick=t)
            for k, v in vals.items():
                rec[k].append(np.asarray(v))
            done, t, n = term or trunc, t + 1, n + 1
        ep += 1
    out = {k: np.stack(v) for k, v in rec.items()}
    np.savez_compressed(os.path.join(a.out, "bookkeeping.npz"), **out)
    print(f"bookkeeping: {n} ticks over {ep} episodes, IK ok {out['ik_ok'].mean():.3f}, "
          f"attempts max {out['attempts1'].max()}")
    env.close()


def record_projection(a: argparse.Namespace) -> tuple[dict[str, Any], dict[str, float]]:
    env = genv.GraspEnv(seed=a.seed + 11, pool=6, wrist=WRIST_SPEC, composites=0.5, **WRIST)
    rng = np.random.default_rng(a.seed + 11)
    scratch: dict[int, Any] = {}
    rec = {k: [] for k in ("q", "gl_pos", "gl_R", "believed_world", "believed_base", "uv",
                           "gl_pos_fk", "gl_R_fk", "uv_fk")}
    ep = 0
    while len(rec["q"]) < a.proj:
        env.reset(seed=a.seed + 11000 + ep)
        done, t = False, 0
        while not done and t < 30 and len(rec["q"]) < a.proj:
            # vary the believed point too, so off-centre and off-frame pixels are covered
            if t % 3 == 2:
                env.believed_pos = env.believed_pos + rng.normal(0, 0.03, 3)
            uv = env.believed_uv()
            glp, glR = gl_pose(env)
            fk = fk_fields(env, scratch, env.student_obs(), uv)
            for k, v in (("q", env.d.qpos[env.arm.qadr].copy()), ("gl_pos", glp), ("gl_R", glR),
                         ("believed_world", env.believed_pos.copy()),
                         ("believed_base", env.believed_pos - BASE), ("uv", uv),
                         ("gl_pos_fk", fk["gl_pos_fk"]), ("gl_R_fk", fk["gl_R_fk"]), ("uv_fk", fk["uv_fk"])):
                rec[k].append(np.asarray(v, float))
            _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 5))
            done, t = term or trunc, t + 1
        ep += 1
    out = {k: np.stack(v) for k, v in rec.items()}
    np.savez_compressed(os.path.join(a.out, "projection.npz"), **out)
    u, v = out["uv"][:, 0], out["uv"][:, 1]
    in_frame = (u >= 0) & (u < 160) & (v >= 0) & (v < 120)
    # far off frame the same sub-millimetre lag moves the pixel by several pixels
    stats = dict(uv_px=lag(out["uv"], out["uv_fk"]),
                 uv_px_in_frame=lag(out["uv"][in_frame], out["uv_fk"][in_frame]),
                 gl_pos_m=lag(out["gl_pos"], out["gl_pos_fk"]))
    print(f"projection: {len(out['q'])} samples, nan {np.isnan(u).sum()}, in frame {in_frame.sum()}, "
          f"u range {np.nanmin(u):.1f}..{np.nanmax(u):.1f}, max |uv - uv_fk| {stats['uv_px']:.3f} px "
          f"({stats['uv_px_in_frame']:.3f} px in frame), "
          f"max |gl_pos - gl_pos_fk| {stats['gl_pos_m'] * 1e3:.3f} mm")
    meta = dict(cam_pos_gripper_link=env._cam_nominal[0].tolist(),
                cam_R_gripper_link_opencv=env._cam_nominal[1].tolist(),
                fovy_deg=float(env._cam_nominal[2]))
    env.close()
    return meta, stats


def record_fk(a: argparse.Namespace) -> dict[str, Any]:
    env = genv.GraspEnv(seed=a.seed + 13, pool=1)
    env.reset(seed=a.seed + 13)
    m = env.m
    d = mujoco.MjData(m)
    arm = env.arm
    base_b = m.body("arm/base_link").id
    mujoco.mj_kinematics(m, d)
    base_pos, base_R = d.xpos[base_b].copy(), d.xmat[base_b].reshape(3, 3).copy()
    assert np.allclose(base_pos, BASE) and np.allclose(base_R, np.eye(3)), (base_pos, base_R)
    rng = np.random.default_rng(a.seed + 13)
    gb = m.body("arm/gripper_link").id
    rec = {k: [] for k in ("q", "tcp_base", "tcp_R", "gl_base", "gl_R")}
    for i in range(a.fk):
        q = rng.uniform(arm.lo, arm.hi)
        d.qpos[arm.qadr] = q
        mujoco.mj_kinematics(m, d)
        rec["q"].append(q); rec["tcp_base"].append(d.site_xpos[arm.site] - BASE)
        rec["tcp_R"].append(d.site_xmat[arm.site].reshape(3, 3).copy())
        rec["gl_base"].append(d.xpos[gb] - BASE); rec["gl_R"].append(d.xmat[gb].reshape(3, 3).copy())
    out = {k: np.stack(v) for k, v in rec.items()}
    np.savez_compressed(os.path.join(a.out, "fk.npz"), **out)
    print(f"fk: {a.fk} samples")
    lo, hi = arm.lo.tolist(), arm.hi.tolist()
    timestep = float(m.opt.timestep)
    env.close()
    return dict(arm_base=BASE.tolist(), joint_lo=lo, joint_hi=hi, timestep=timestep)


def record_heat(a: argparse.Namespace) -> None:
    uv = np.array([[79.5, 59.5], [0.0, 0.0], [159.0, 119.0], [46.53, 72.96], [-30.0, 60.0], [200.0, -50.0],
                   [np.nan, np.nan]], np.float32)
    h = heat_channel(uv, 120, 160, "cpu").numpy()
    np.savez_compressed(os.path.join(a.out, "heat.npz"), uv=uv, heat=h)
    print("heat:", h.shape)


def record_hover(a: argparse.Namespace) -> dict[str, float]:
    """Episode starts: the hover training solved, the joints after its 0.1 s settle and the tick-0 inputs."""
    env = genv.GraspEnv(seed=a.seed + 19, pool=8, wrist=WRIST_SPEC, composites=0.5, disturb_p=0.3,
                        force_p=0.3, **WRIST)
    scratch: dict[int, Any] = {}
    rec = {k: [] for k in ("p_hover", "believed", "yaw", "q_hover", "q_measured", "gap0", "uv0", "obs0",
                           "uv0_fk", "obs0_fk", "episode")}
    k = 0
    while len(rec["yaw"]) < a.hover:
        seed = a.seed + 19000 + k
        k += 1
        try:
            env.reset(seed=seed)
        except RuntimeError as e:   # no hover pose in 50 tries: training skips such a draw too
            print(f"hover: reset {seed} skipped ({e})")
            continue
        assert env.yaw_cmd == env.believed_yaw and np.array_equal(env.p_cmd, env.p0)
        uv0, obs0 = env.believed_uv(), env.student_obs()
        fk = fk_fields(env, scratch, obs0, uv0)
        for key, v in (("p_hover", env.p0 - BASE), ("believed", env.believed_pos - BASE),
                       ("yaw", float(env.believed_yaw)), ("q_hover", env.q_cmd.copy()),
                       ("q_measured", env.d.qpos[env.arm.qadr].copy()), ("gap0", env.arm.gap(env.d)),
                       ("uv0", uv0), ("obs0", obs0), ("uv0_fk", fk["uv_fk"]), ("obs0_fk", fk["obs_fk"]),
                       ("episode", seed)):
            rec[key].append(np.asarray(v).copy())
    env.close()
    out = {key: np.stack(v) for key, v in rec.items()}
    for key in ("uv0", "obs0", "uv0_fk", "obs0_fk"):
        out[key] = out[key].astype(np.float32)
    np.savez_compressed(os.path.join(a.out, "hover.npz"), **out)
    stats = dict(obs=lag(out["obs0"], out["obs0_fk"]), uv_px=lag(out["uv0"], out["uv0_fk"]))
    j5 = np.degrees(np.abs(out["q_hover"][:, 4])).max()
    print(f"hover: {len(out['yaw'])} resets, max |J5| at the hover {j5:.4f} deg, "
          f"max |obs0 - obs0_fk| {stats['obs']:.3e}, max |uv0 - uv0_fk| {stats['uv_px']:.3f} px")
    return stats


def record_ik(a: argparse.Namespace) -> None:
    """Single warm IK steps as training's `step` makes them, from IK_HOVERS hovers: random actions, box,
    floor and ceiling; the command advances only on an accepted step, and the next seed is the last
    accepted joints. In every IK_LIMIT_EVERY-th hover the middle step asks for a yaw that needs J6 beyond
    its range, so the file holds refused steps and steps that go on from the held seed."""
    env = genv.GraspEnv(seed=a.seed + 17, pool=4, composites=0.5)
    rng = np.random.default_rng(a.seed + 17)
    per = max(1, math.ceil(a.ik / IK_HOVERS))
    rec = {k: [] for k in ("q0", "pos", "R", "q", "ok", "yaw", "hover", "step", "j6_limit")}
    h = 0
    while len(rec["ok"]) < a.ik:
        env.reset(seed=a.seed + 17000 + h)
        arm = env.arm
        p_cmd, yaw_cmd, q_cmd, p0 = env.p_cmd.copy(), env.yaw_cmd, env.q_cmd.copy(), env.p0.copy()
        for i in range(per):
            if len(rec["ok"]) >= a.ik:
                break
            limit = h % IK_LIMIT_EVERY == IK_LIMIT_EVERY - 1 and i == per // 2
            if limit:
                j6 = math.copysign(arm.hi[5] + IK_LIMIT_BEYOND, q_cmd[5])
                p, yaw = p_cmd.copy(), genv._wrap(q_cmd[0] + math.pi / 2 - j6)
            else:
                act = np.clip(np.asarray(rng.uniform(-1.3, 1.3, 5), np.float32), -1.0, 1.0)
                p = p_cmd + act[:3] * env.step_xyz
                p[:2] = np.clip(p[:2], p0[:2] - genv.BOX_XY, p0[:2] + genv.BOX_XY)
                p[2] = np.clip(p[2], genv.TABLE_TOP + 0.02, p0[2] + 0.08)
                yaw = genv._wrap(yaw_cmd + act[3] * env.step_yaw)
            R = genv._grasp_R(yaw)
            q, ok = arm.ik(q_cmd, p, R, iters=60)
            for k, v in (("q0", q_cmd.copy()), ("pos", p - BASE), ("R", R), ("q", q.copy()), ("ok", bool(ok)),
                         ("yaw", float(yaw)), ("hover", h), ("step", i), ("j6_limit", limit)):
                rec[k].append(np.asarray(v))
            if ok:
                p_cmd, yaw_cmd, q_cmd = p, yaw, q
        h += 1
    out = {k: np.stack(v) for k, v in rec.items()}
    np.savez_compressed(os.path.join(a.out, "ik.npz"), **out)
    lim = out["j6_limit"]
    print(f"ik: {len(out['ok'])} steps from {h} hovers, accepted {int(out['ok'].sum())}, refused "
          f"{int((~out['ok']).sum())} ({int((~out['ok'] & lim).sum())} of {int(lim.sum())} J6-limit steps)")


def record_state_keys(a: argparse.Namespace, ckpt: dict[str, Any]) -> None:
    keys = [[k, list(v.shape), str(v.dtype)] for k, v in ckpt["net"].items()]
    with open(os.path.join(a.out, "state_keys.json"), "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(k) for k in keys) + "\n]\n")
    print(f"state_keys: {len(keys)} entries")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--galaxeo", default=os.environ.get("GALAXEO_REPO"),
                    help="the training repository, read only (default: $GALAXEO_REPO)")
    ap.add_argument("--checkpoint", default=CHECKPOINT,
                    help=f"relative to the training repository (default {CHECKPOINT})")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=50_000)
    ap.add_argument("--shard-episodes", type=int, default=2)
    ap.add_argument("--shard-ticks", type=int, default=40)
    ap.add_argument("--book", type=int, default=1000)
    ap.add_argument("--proj", type=int, default=300)
    ap.add_argument("--fk", type=int, default=300)
    ap.add_argument("--hover", type=int, default=200)
    ap.add_argument("--ik", type=int, default=200)
    a = ap.parse_args()
    if not a.galaxeo:
        ap.error("give --galaxeo or set $GALAXEO_REPO")
    repo = Path(a.galaxeo).expanduser().resolve()
    if not (repo / "sim" / "grasp" / "env.py").is_file():
        ap.error(f"{repo} is not the training repository (no sim/grasp/env.py)")
    out = Path(a.out).expanduser().resolve()
    if inside(out, repo) or inside(Path.cwd().resolve(), repo):
        ap.error("the training repository is read only: run from, and write to, a directory outside it")
    a.out = str(out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = repo / a.checkpoint

    import_training(repo)
    st = Student.load(str(ckpt_path))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    lag_shard = record_shard(a, st)
    record_bookkeeping(a)
    meta, lag_proj = record_projection(a)
    meta.update(record_fk(a))
    record_heat(a)
    lag_hover = record_hover(a)
    record_ik(a)
    record_state_keys(a, ckpt)

    rel = ckpt_path.resolve().relative_to(repo) if inside(ckpt_path.resolve(), repo) else Path(ckpt_path.name)
    sigma = inspect.signature(heat_channel).parameters["sigma_frac"].default
    meta.update(obs_mean=st.obs_mean.tolist(), obs_std=st.obs_std.tolist(), checkpoint=str(rel), seed=a.seed,
                image_hw=[120, 160], sigma_frac=sigma, step_xyz=genv.STEP_XYZ,
                step_yaw_deg=math.degrees(genv.STEP_YAW), box_xy=genv.BOX_XY,
                floor="TABLE_TOP + 0.02 (world z 0.72)", ceiling="p0_z + 0.08", hover=genv.HOVER)
    fovy = meta["fovy_deg"]
    meta.update(f_px=(120 / 2) / math.tan(math.radians(fovy) / 2), cx=160 / 2 - 0.5, cy=120 / 2 - 0.5)
    meta.update(
        calibration_joint_offsets_deg=CALIBRATION_JOINT_OFFSETS_DEG,
        control_hz=genv.CONTROL_HZ, substeps=genv.SUBSTEPS,
        max_steps=inspect.signature(genv.GraspEnv).parameters["max_steps"].default,
        ik_iters=60, ik_damping=0.05, ik_tol_pos=1e-3, ik_tol_rot=5e-3,
        checkpoint_sha256=sha256_of(ckpt_path), checkpoint_bytes=ckpt_path.stat().st_size,
        training_repository=repo_head(repo),
        recorder_args={k: v for k, v in vars(a).items() if k not in ("galaxeo", "out")},
        recorded_with=dict(python=sys.version.split()[0], numpy=np.__version__, mujoco=mujoco.__version__,
                           torch=torch.__version__),
        substep_lag=dict(shard=lag_shard, projection=lag_proj, hover=lag_hover),
    )
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
