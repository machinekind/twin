"""Create a concise Markdown report and plot from a completed lift experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    root = args.experiment.resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    baseline = json.loads((root / "test-baseline.json").read_text(encoding="utf-8"))
    selected = baseline if summary["selected"] == "baseline" else json.loads(
        (root / "test-selected.json").read_text(encoding="utf-8"))
    profiles = list(baseline)
    text = ["# Wyniki douczania lift", "",
            f"Model wybrany na walidacji: **{summary['selected']}**.", "",
            "Końcowy test używa osobnych seedów i zwykłego MuJoCo na CPU. "
            "Sukces oznacza chwyt powyżej 6 cm utrzymany przez 10 kolejnych taktów.", "",
            "| Warunki | lift-v3 | Wybrany model | 95% CI wybranego | Czas bazy / modelu |",
            "|---|---:|---:|---:|---:|"]
    for name in profiles:
        b, s = baseline[name], selected[name]
        lo, hi = s["wilson95"]
        text.append(f"| {name} | {b['finished']}/{b['episodes']} | {s['finished']}/{s['episodes']} "
                    f"| {lo:.1%}–{hi:.1%} | {b['mean_steps']/20:.2f} / {s['mean_steps']/20:.2f} s |")
    text += ["", "Czas to średnia do końca epizodu; nieudane próby również się liczą. "
             "Przedziały Wilsona opisują niepewność skuteczności, nie różnicy między modelami.", "",
             "| Warunki | Takty przy limitach/ep., baza / model | Maks. ruch po sukcesie, baza / model |",
             "|---|---:|---:|"]
    for name in profiles:
        b, s = baseline[name], selected[name]
        text.append(f"| {name} | {b['limit_ticks_per_episode']:.3f} / {s['limit_ticks_per_episode']:.3f} "
                    f"| {b['max_swing_deg']:.1f} / {s['max_swing_deg']:.1f}° |")
    text += ["", "## Porównanie po seedach", "",
             "| Warunki | Sukces tylko nowego | Sukces tylko bazy |",
             "|---|---:|---:|"]
    for name in profiles:
        b, s = baseline[name]["rows"], selected[name]["rows"]
        assert [r["seed"] for r in b] == [r["seed"] for r in s]
        wins = sum(y["finished"] and not x["finished"] for x,y in zip(b,s))
        losses = sum(x["finished"] and not y["finished"] for x,y in zip(b,s))
        text.append(f"| {name} | {wins} | {losses} |")
    text += ["", "To test symulacyjny z syntetycznym modelem błędów percepcji. "
             "Nie obejmuje fizycznego ramienia ani detekcji z obrazów kamer. "
             "Selekcja jest wynikiem jednego przebiegu treningowego, nie przeszukania wszystkich możliwości.", "",
             f"Commit źródłowy: `{manifest['git_commit']}`. "
             f"Trening: {manifest['ppo']['iterations']} iteracji × "
             f"{manifest['ppo']['num_envs']} światów × {manifest['ppo']['horizon']} kroków.", "",
             "Polityka: [selected/policy.pt](selected/policy.pt). "
             "Konfiguracja: [manifest.json](manifest.json). "
             "Wyniki: [summary.json](summary.json).", "",
             "![Porównanie](comparison.png)", ""]
    (root / "REPORT.md").write_text("\n".join(text), encoding="utf-8")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    xs = np.arange(len(profiles))
    for offset, data, label, color in [(-0.18, baseline, "lift-v3", "#718096"),
                                      (0.18, selected, summary["selected"], "#087f8c")]:
        rates = np.array([data[n]["success"] * 100 for n in profiles])
        bounds = np.array([data[n]["wilson95"] for n in profiles]).T * 100
        axs[0].bar(xs + offset, rates, width=.34, color=color, label=label,
                   yerr=np.maximum(0, np.vstack((rates-bounds[0], bounds[1]-rates))), capsize=4)
        axs[1].bar(xs + offset, [data[n]["mean_steps"]/20 for n in profiles], width=.34, color=color)
        axs[2].bar(xs + offset, [data[n]["limit_ticks_per_episode"] for n in profiles], width=.34, color=color)
    for ax in axs:
        ax.set_xticks(xs, profiles)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axs[0].set(title="Held-out success (95% Wilson CI)", ylabel="%", ylim=(0, 106))
    axs[0].legend(loc="lower left")
    axs[1].set(title="Mean episode duration", ylabel="seconds")
    axs[2].set(title="Near-limit commands after grasp", ylabel="ticks / episode")
    fig.suptitle("SO-101 lift: independent CPU simulation test", fontsize=16, fontweight="bold")
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
