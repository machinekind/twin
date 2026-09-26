"""Bounded multi-object curriculum, independent CPU evaluation and selected videos.

Worker processes release all CUDA graphs between blocks. status.json is the
watchdog contract. No hardware connection is opened anywhere in this campaign.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def scenario(name):
    from lerobot_mp.twin.rl.randomize import Randomization, Dynamics
    if name == "nominal":
        return Randomization.none()
    if name == "warm":
        return replace(Randomization.around(Dynamics(), .5), calibration_translation=.001,
                       calibration_rotation_deg=.25, cube_noise=.001, cube_delay=1, cube_period=2)
    if name == "moderate":
        return replace(Randomization(), calibration_translation=.003, calibration_rotation_deg=.75,
                       calibration_scale=.01, joint_bias_deg=.15, cube_z_noise=.001,
                       dropout_probability=.05, dropout_steps=2)
    if name == "stress":
        return replace(Randomization.around(Dynamics(), 1.5), calibration_translation=.006,
                       calibration_rotation_deg=1.5, calibration_scale=.02, joint_bias_deg=.3,
                       cube_noise=.003, cube_z_noise=.002, max_delay=2, cube_delay=4, cube_period=4,
                       dropout_probability=.10, dropout_steps=4)
    if name == "extreme":
        return replace(scenario("stress"), calibration_translation=.010, calibration_rotation_deg=3.,
                       calibration_scale=.03, joint_bias_deg=.6, max_delay=3,
                       dropout_probability=.15, dropout_steps=10)
    raise ValueError(name)


def worker_train(args):
    import torch
    from lerobot_mp.twin.rl.multi import MixedBatchEnv
    from lerobot_mp.twin.rl.objects import OBJECT_KINDS
    from lerobot_mp.twin.rl.policy import Policy
    from lerobot_mp.twin.rl.ppo import PPOConfig, train
    torch.set_num_threads(4)
    initial = Policy.load(args.policy)
    stage = args.stage
    kinds = OBJECT_KINDS[:6] if stage == "warm" else OBJECT_KINDS
    initial.meta.task.update(smooth_velocity=.005 if stage == "warm" else .01,
                             smooth_acceleration=.015 if stage == "warm" else .03,
                             smooth_jerk=.003 if stage == "warm" else .01,
                             action_rate=.03, hold_still=2., limit_penalty=1.)
    stop = threading.Event()
    cfg = PPOConfig(num_envs=args.envs, iterations=args.iters, seed=args.seed,
                    init_std=.18 if stage == "warm" else .12, desired_kl=.005,
                    entropy=.001, save_every=25)
    write(args.out / "worker.json", {"stage":stage,"kinds":list(kinds),"object_scale":args.scale,
          "initial_policy":str(args.policy.resolve()),"until_unix":args.until,"seed":args.seed,
          "note":"Each kind has its own compiled collider; fold_yaw is enabled only for cube"})
    status_path = args.out.parent.parent / "status.json"
    def progress(p):
        if time.time() >= args.until or (args.out.parent.parent / "STOP").exists():
            stop.set()
        write(status_path, {"status":"training", "stage":stage, "block":args.out.name,
                            "updated":now(), "iteration":p.iteration, "iterations":p.iterations,
                            "steps":p.steps, "success_training":p.success, "fps":p.fps,
                            "elapsed_s":p.elapsed, "policy":str(args.out / "policy.pt"),
                            "block_deadline_unix":args.until})
        if p.iteration == 1 or p.iteration % 10 == 0:
            print(f"{stage}/{args.out.name} {p.iteration}/{p.iterations} "
                  f"success={p.success:.3%} fps={p.fps:.0f} time={p.elapsed:.0f}s", flush=True)
    def factory(task, count, **kw):
        return MixedBatchEnv(task, count, kinds=kinds, scale=args.scale, **kw)
    train("lift", cfg, randomization=scenario(stage), out_dir=args.out, init=initial,
          stop=stop, on_progress=progress, env_factory=factory)


def make_video(env, trajectory, path, label):
    import cv2
    import mujoco
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), env.task.control_hz, (480,360))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for step,qpos in enumerate(trajectory):
            env.scene.data.qpos[:] = qpos
            mujoco.mj_forward(env.scene.model, env.scene.data)
            rgb = env.render()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame,(0,0),(480,46),(20,20,20),-1)
            cv2.putText(frame,label[:65],(8,18),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(frame,f"SIMULATION | {step/env.task.control_hz:.2f}s | successful lift",
                        (8,37),cv2.FONT_HERSHEY_SIMPLEX,.40,(80,230,130),1,cv2.LINE_AA)
            writer.write(frame)
        if trajectory:
            for _ in range(12):
                writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size < 1000:
        raise RuntimeError(f"Empty success video: {path}")


def worker_eval(args):
    import numpy as np
    import torch
    from lerobot_mp.twin.rl.env import TwinEnv
    from lerobot_mp.twin.rl.multi import object_task
    from lerobot_mp.twin.rl.objects import OBJECT_KINDS
    from lerobot_mp.twin.rl.policy import Policy
    torch.set_num_threads(4)
    pol = Policy.load(args.policy)
    result = {"policy":str(args.policy), "seed":args.seed, "scale":args.scale,
              "requested_episodes":args.episodes, "complete":False, "cases":{}, "videos":[]}
    video_index_path = args.videos / "index.json"
    video_index = json.loads(video_index_path.read_text(encoding="utf-8")) if video_index_path.exists() else {}
    interrupted = False
    for profile in args.profiles:
        for kind in OBJECT_KINDS:
            if time.time() >= args.until:
                interrupted = True
                break
            task = object_task(pol.task, kind, args.scale)
            rand = replace(scenario(profile), fold_yaw=kind == "cube")
            env = TwinEnv(task, randomization=rand, render_mode="rgb_array")
            rows, best, best_trajectory = [], None, None
            try:
                for ep in range(args.episodes):
                    if time.time() >= args.until:
                        interrupted = True
                        break
                    seed = args.seed + ep
                    obs,_ = env.reset(seed=seed)
                    trajectory = [env.scene.data.qpos.copy()]
                    commands, actual, penalties, pauses = [], [], [], 0
                    for step in range(task.episode_steps):
                        obs,reward,term,trunc,info = env.step(pol.act(obs))
                        trajectory.append(env.scene.data.qpos.copy())
                        commands.append(env.q_cmd.copy())
                        actual.append(env.scene.data.qpos[env.kin.qadr].copy())
                        penalties.append(info["smooth_cost"])
                        pauses += int(info["stale_pause"])
                        if term or trunc:
                            break
                    q = np.asarray(commands)[:,:5]
                    measured = np.asarray(actual)[:,:5]
                    velocity = np.diff(measured,axis=0)*task.control_hz
                    acceleration = np.diff(velocity,axis=0)*task.control_hz
                    jerk = np.diff(acceleration,axis=0)*task.control_hz
                    cmd_jerk = np.diff(q,n=3,axis=0)*task.control_hz**3
                    rms = lambda x: float(np.sqrt(np.mean(x*x))) if x.size else 0.
                    near = (np.minimum(abs(q-env.limits.lo[:5]),abs(q-env.limits.hi[:5])) < np.radians(1)).any(1)
                    row = {"seed":seed,"finished":bool(info["finished"] and info["success"]),
                           "steps":step+1,"height_mm":info["height"]*1000,
                           "velocity_rms":rms(velocity),"acceleration_rms":rms(acceleration),
                           "jerk_rms":rms(jerk),"command_jerk_rms":rms(cmd_jerk),
                           "limit_ticks":int(near.sum()),"stale_pause_ticks":pauses}
                    rows.append(row)
                    if row["finished"] and (best is None or (row["command_jerk_rms"],row["steps"]) <
                                            (best["command_jerk_rms"],best["steps"])):
                        best,best_trajectory = row,trajectory
                key = f"{profile}/{kind}"
                n = len(rows)
                case = {"episodes":n,"finished":sum(r["finished"] for r in rows),
                        "success":sum(r["finished"] for r in rows)/n if n else 0.,
                        "mean_steps":float(np.mean([r["steps"] for r in rows])) if n else 0.,
                        "command_jerk_rms":float(np.mean([r["command_jerk_rms"] for r in rows])) if n else 0.,
                        "actual_jerk_rms":float(np.mean([r["jerk_rms"] for r in rows])) if n else 0.,
                        "limit_ticks_per_episode":float(np.mean([r["limit_ticks"] for r in rows])) if n else 0.,
                        "rows":rows}
                result["cases"][key] = case
                print(f"eval {args.out.stem} {key}: {case['finished']}/{n}, "
                      f"command jerk={case['command_jerk_rms']:.1f}",flush=True)
                # Distinctive only: first success, or >=20% smoother than that
                # family's previous highlight, capped at 80 clips in this folder.
                old = video_index.get(key)
                noteworthy = best is not None and (old is None or best["command_jerk_rms"] < .8*old["command_jerk_rms"])
                if noteworthy and len(list(args.videos.glob("*.mp4"))) < 80:
                    filename = f"{args.out.stem}_{kind}_{profile}_seed{best['seed']}.mp4"
                    video = args.videos / filename
                    make_video(env,best_trajectory,video,f"{kind} | {profile} | {args.out.stem}")
                    result["videos"].append(str(video))
                    video_index[key] = {**best,"path":str(video),"checkpoint":str(args.policy)}
                    write(video_index_path,video_index)
                    print(f"VIDEO {video}",flush=True)
                write(args.out,result)
            finally:
                env.close()
        if interrupted:
            break
    result["complete"] = not interrupted
    write(args.out,result)


def score(report):
    cases = report["cases"]
    nominal = [v["success"] for k,v in cases.items() if k.startswith("nominal/")]
    overall = [v["success"] for v in cases.values()]
    jerk = [v["command_jerk_rms"] for v in cases.values()]
    if not report["complete"] or len(nominal) != 12:
        return (-1.,-1.,-1.)
    return (min(nominal),sum(overall)/len(overall),-sum(jerk)/len(jerk))


def campaign(args):
    import torch
    from lerobot_mp.twin.rl.multi import shape_warm_start
    from lerobot_mp.twin.rl.objects import OBJECT_KINDS, object_spec
    from lerobot_mp.twin.rl.policy import Policy
    torch.set_num_threads(4)
    out = args.out.resolve()
    if (out / "manifest.json").exists():
        raise RuntimeError("Campaign directory already exists. Use a new --out to preserve results.")
    out.mkdir(parents=True,exist_ok=True)
    args.videos.mkdir(parents=True,exist_ok=True)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    start = args.start if args.start else time.time()
    if deadline <= time.time()+120:
        raise ValueError("Campaign deadline must be more than 2 minutes in the future")
    remaining = deadline-start
    training_end = deadline-min(25*60,remaining*.15)
    initial_path = out / "initial/policy.pt"
    shape_warm_start(Policy.load(args.policy),smooth_velocity=.005,
                     smooth_acceleration=.015,smooth_jerk=.003,
                     action_rate=.03,hold_still=2.,limit_penalty=1.).save(initial_path)
    tracked = [*list((ROOT/"src/lerobot_mp/twin/rl").glob("*.py")), Path(__file__)]
    write(out / "manifest.json", {"created":now(),"start_unix":start,"deadline_unix":deadline,
          "source_policy":str(args.policy.resolve()),"training_end_unix":training_end,
          "objects":{k:asdict(object_spec(k)) for k in OBJECT_KINDS},
          "profiles":{k:asdict(scenario(k)) for k in ("nominal","warm","moderate","stress","extreme")},
          "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
          "videos":str(args.videos),"validation_seed":30000,"test_seed":80000,
          "warning":"Known object identity/dimensions and noisy pose, not image recognition of arbitrary objects"})
    latest,best_policy,best_score = initial_path,initial_path,(-1.,-1.,-1.)
    evaluations = []
    def invoke(mode, flags):
        command = [sys.executable,"-u",str(Path(__file__).resolve()),mode,*map(str,flags)]
        print("WORKER " + " ".join(command),flush=True)
        completed = subprocess.run(command,cwd=ROOT)
        if completed.returncode:
            raise RuntimeError(f"{mode} worker exited with {completed.returncode}")
    def evaluate(path,label,episodes,seed,profiles,until,scale=1.):
        target = out / "evaluations" / f"{label}.json"
        write(out / "status.json", {"status":"evaluating","stage":label,"updated":now(),
                                    "policy":str(path),"deadline_unix":deadline})
        invoke("eval",["--policy",path,"--out",target,"--episodes",episodes,"--seed",seed,
                       "--until",until,"--videos",args.videos,"--scale",scale,"--profiles",*profiles])
        return json.loads(target.read_text(encoding="utf-8"))
    baseline = evaluate(initial_path,"initial-validation",5,30000,["nominal","moderate"],training_end)
    best_score = score(baseline)
    evaluations.append({"policy":str(initial_path),"score":best_score})
    block = 0
    while time.time() < training_end and not (out/"STOP").exists():
        elapsed = time.time()-start
        if elapsed < 65*60:
            stage,phase_end,scale = "warm",min(start+65*60,training_end),1.
        elif elapsed < 120*60:
            stage,phase_end,scale = "moderate",min(start+120*60,training_end),[1.,.85,1.15][block%3]
        else:
            stage,phase_end,scale = "stress",training_end,[1.,.85,1.15][block%3]
        block += 1
        checkpoint = out / "checkpoints" / f"block-{block:03d}-{stage}"
        write(out / "status.json", {"status":"initializing_gpu","stage":stage,"block":block,
                                    "updated":now(),"deadline_unix":deadline})
        invoke("train",["--policy",latest,"--out",checkpoint,"--stage",stage,
                        "--envs",args.envs,"--iters",args.iters,"--seed",260926+block,
                        "--scale",scale,"--until",phase_end])
        latest = checkpoint / "policy.pt"
        report = evaluate(latest,f"block-{block:03d}-validation",5,30000,
                          ["nominal","moderate"],min(training_end+180,deadline))
        candidate_score = score(report)
        evaluations.append({"policy":str(latest),"score":candidate_score})
        if candidate_score > best_score:
            best_policy,best_score = latest,candidate_score
        write(out/"selection.json",{"selected":str(best_policy),"score":best_score,"candidates":evaluations,
              "criterion":"worst nominal object success, macro success, negative command jerk; validation only"})
        if not report["complete"]:
            break
    selected = Policy.load(best_policy)
    selected.meta.notes += "; simulation curriculum selected on seeds 30000+, heldout 80000+"
    selected.save(out / "selected/policy.pt")
    heldout = evaluate(out/"selected/policy.pt","heldout-selected",args.test_episodes,80000,
                       ["nominal","moderate","stress"],deadline,scale=1.08)
    complete = heldout["complete"]
    write(out / "summary.json", {"selected":str(best_policy),"validation_score":best_score,
          "heldout_complete":complete,"heldout":{k:{a:b for a,b in v.items() if a != "rows"}
          for k,v in heldout["cases"].items()},"videos":str(args.videos),"finished":now(),
          "deployment":"Simulation only: generic shape/size/pose perception adapter still required"})
    write(out / "status.json", {"status":"complete" if complete else "time_budget_reached",
          "updated":now(),"selected":str(out/"selected/policy.pt"),"videos":str(args.videos),
          "summary":str(out/"summary.json")})
    print(f"CAMPAIGN FINISHED {out / 'summary.json'}",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["campaign","train","eval"])
    p.add_argument("--policy",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--videos",type=Path,default=Path("D:/digital-twin-training/sukcesy"))
    p.add_argument("--deadline",default="2026-09-26T05:39:06+00:00")
    p.add_argument("--start",type=float)
    p.add_argument("--until",type=float,default=float("inf"))
    p.add_argument("--stage",choices=["warm","moderate","stress"],default="warm")
    p.add_argument("--envs",type=int,default=1536)
    p.add_argument("--iters",type=int,default=250)
    p.add_argument("--episodes",type=int,default=5)
    p.add_argument("--test-episodes",type=int,default=50)
    p.add_argument("--seed",type=int,default=30000)
    p.add_argument("--scale",type=float,default=1.)
    p.add_argument("--profiles",nargs="+",default=["nominal","moderate"])
    args = p.parse_args()
    try:
        {"campaign":campaign,"train":worker_train,"eval":worker_eval}[args.mode](args)
    except BaseException as exc:
        if args.mode == "campaign":
            write(args.out / "status.json",{"status":"failed","updated":now(),"error":str(exc)})
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
