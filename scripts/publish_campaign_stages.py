"""Publish completed campaign stages as independent draft PRs (explicit opt-in).

Uses the existing Git credential helper in memory for gh, never writes tokens.
Isolated worktrees prevent switching the running training checkout. Re-running
is safe: ledger and GitHub head-branch lookup avoid duplicate PRs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = "machinekind/digital_twin_training"


def write(path,value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.replace(path)


def command(args,cwd=ROOT,env=None):
    p = subprocess.run(args,cwd=cwd,env=env,text=True,encoding="utf-8",capture_output=True)
    if p.returncode:
        raise RuntimeError(f"{args[0]} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def authenticated_environment():
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0",GCM_INTERACTIVE="Never")
    response = subprocess.run(["git","credential","fill"],cwd=ROOT,env=env,
        input=f"url=https://github.com/{REPO}.git\n\n",text=True,capture_output=True)
    if response.returncode:
        raise RuntimeError("Git credential helper did not return GitHub authentication")
    fields = dict(line.split("=",1) for line in response.stdout.splitlines() if "=" in line)
    if not fields.get("password"):
        raise RuntimeError("No GitHub credential is available")
    env["GH_TOKEN"] = fields["password"]
    return env


def choose_checkpoint(campaign,stage):
    choices=[]
    for directory in (campaign/"checkpoints").glob(f"block-*-{stage}"):
        number = directory.name.split("-")[1]
        evaluation = campaign/"evaluations"/f"block-{number}-validation.json"
        if not evaluation.exists() or not (directory/"policy.pt").exists():
            continue
        report=json.loads(evaluation.read_text(encoding="utf-8"))
        if not report.get("complete"):
            continue
        cases=list(report["cases"].values())
        nominal=[v["success"] for k,v in report["cases"].items() if k.startswith("nominal/")]
        score=(min(nominal),sum(c["success"] for c in cases)/len(cases),
               -sum(c["command_jerk_rms"] for c in cases)/len(cases))
        choices.append((score,directory,evaluation))
    if not choices:
        raise RuntimeError(f"No complete evaluation/checkpoint for stage {stage}")
    return max(choices,key=lambda x:x[0])[1:]


def publish(campaign,stage,worktrees,env):
    branch=f"codex/campaign-20260926-{stage}"
    existing=json.loads(command(["gh","pr","list","--repo",REPO,"--head",branch,
                                 "--state","all","--json","url"],env=env))
    if existing:
        return existing[0]["url"]
    if stage == "final":
        checkpoint=campaign/"selected"
        evaluation=campaign/"evaluations/heldout-selected.json"
    else:
        checkpoint,evaluation=choose_checkpoint(campaign,stage)
    report=json.loads(evaluation.read_text(encoding="utf-8"))
    command(["git","fetch","origin","main"],env=env)
    worktree=worktrees/stage
    if not worktree.exists():
        branch_exists=subprocess.run(["git","show-ref","--verify","--quiet",f"refs/heads/{branch}"],cwd=ROOT).returncode==0
        if branch_exists:
            command(["git","worktree","add",str(worktree),branch],env=env)
        else:
            command(["git","worktree","add","-b",branch,str(worktree),"origin/main"],env=env)
    actual=command(["git","branch","--show-current"],cwd=worktree)
    if actual!=branch:
        raise RuntimeError(f"Unexpected branch in publisher worktree: {actual}")
    destination=worktree/"research/training/20260926"/stage
    destination.mkdir(parents=True,exist_ok=True)
    for name in ("policy.pt","meta.json","config.json","worker.json"):
        if (checkpoint/name).exists():
            shutil.copy2(checkpoint/name,destination/name)
    shutil.copy2(evaluation,destination/"evaluation.json")
    for name in ("manifest.json","selection.json"):
        if (campaign/name).exists():
            shutil.copy2(campaign/name,destination/name)
    if stage=="final" and (campaign/"summary.json").exists():
        shutil.copy2(campaign/"summary.json",destination/"summary.json")
    lines=[f"# SO-101 — etap {stage}","",
           "Eksperymentalny checkpoint i wyniki symulacji. Wymaga kodu z PR #2 oraz adaptera "
           "percepcji typu, wymiarów i pozy obiektu przed wdrożeniem na fizycznym ramieniu.","",
           f"Źródło checkpointu: `{checkpoint.name}`. Ewaluacja kompletna: `{report.get('complete')}`.","",
           "| Profil / obiekt | Sukcesy | Średnie takty | Jerk poleceń RMS [rad/s³] |",
           "|---|---:|---:|---:|"]
    for key,case in report["cases"].items():
        lines.append(f"| {key} | {case['finished']}/{case['episodes']} | {case['mean_steps']:.1f} "
                     f"| {case['command_jerk_rms']:.2f} |")
    lines += ["","Modele geometryczne są parametrycznymi przybliżeniami. Typ i rozmiar obiektu "
              "są wejściami, a nie wynikiem wyuczonego detektora obrazu. Wyniki walidacji służą "
              "do wyboru checkpointu; tylko raport final używa osobnego zbioru testowego.",""]
    (destination/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    relative=destination.relative_to(worktree).as_posix()
    command(["git","add",relative],cwd=worktree,env=env)
    staged=command(["git","diff","--cached","--name-only"],cwd=worktree)
    if staged:
        command(["git","-c","user.name=Codex","-c","user.email=codex@localhost","commit",
                 "-m",f"Publish SO-101 {stage} checkpoint and simulation evaluation"],cwd=worktree,env=env)
    command(["git","push","-u","origin",branch],cwd=worktree,env=env)
    body=campaign/f"pr-body-{stage}.md"
    body.write_text(f"Publikuje checkpoint po etapie **{stage}** kampanii SO-101 wraz z konfiguracją "
                    f"i wynikami każdego obiektu/profilu. Szczegółowy raport: `{relative}/REPORT.md`.\n\n"
                    "Kod środowiska: #2. Artefakty są eksperymentalne i pozostają poza katalogiem "
                    "polityk produkcyjnych. Nie są potwierdzeniem skuteczności na fizycznym ramieniu.\n\n"
                    f"Ewaluacja kompletna: `{report.get('complete')}`. "
                    "Nieudane próby są zachowane w evaluation.json; nie wybierano wyłącznie sukcesów.\n",
                    encoding="utf-8")
    return command(["gh","pr","create","--repo",REPO,"--base","main","--head",branch,"--draft",
                    "--title",f"SO-101: checkpoint i wyniki etapu {stage}","--body-file",str(body)],
                    cwd=worktree,env=env)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign",type=Path)
    parser.add_argument("--worktrees",type=Path,default=Path("D:/digital-twin-training/pr-worktrees/20260926"))
    args=parser.parse_args()
    campaign=args.campaign.resolve()
    ledger_path=campaign/"stage_prs.json"
    ledger=json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    env=authenticated_environment()
    manifest=json.loads((campaign/"manifest.json").read_text(encoding="utf-8"))
    limit=manifest["deadline_unix"]+20*60
    while time.time()<limit:
        status=json.loads((campaign/"status.json").read_text(encoding="utf-8"))
        stage=status.get("stage","")
        final=status.get("status") in ("complete","time_budget_reached")
        ready=[]
        if stage in ("moderate","stress","heldout-selected") or final:
            ready.append("warm")
        if stage in ("stress","heldout-selected") or final:
            ready.append("moderate")
        if stage=="heldout-selected" or final:
            ready.append("stress")
        if final:
            ready.append("final")
        for phase in ready:
            if phase in ledger:
                continue
            try:
                url=publish(campaign,phase,args.worktrees.resolve(),env)
                ledger[phase]={"url":url,"published":datetime.now(timezone.utc).isoformat()}
                write(ledger_path,ledger)
                print(f"PUBLISHED {phase}: {url}",flush=True)
            except Exception as exc:
                # A publishing failure must never stop or corrupt training.
                print(f"PUBLISH_ERROR {phase}: {exc}",flush=True)
                write(campaign/"publisher-status.json",{"status":"retrying","stage":phase,"error":str(exc)})
                break
        write(campaign/"publisher-status.json",{"status":"complete" if len(ledger)==4 else "watching",
              "updated":datetime.now(timezone.utc).isoformat(),"published":ledger})
        if len(ledger)==4:
            return
        if status.get("status")=="failed":
            raise RuntimeError("Campaign failed; stage publisher stopped without publishing incomplete final results")
        time.sleep(45)
    raise RuntimeError("Publisher deadline reached before all stage PRs were created")


if __name__=="__main__":
    main()
