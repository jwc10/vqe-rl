#!/usr/bin/env python3
"""Generate final-report figures from consolidated results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT / "results" / "final_report" / "master_results.json"
OUT = ROOT / "results" / "final_report"
CHEM = 1.6  # mHa

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


def load():
    return json.loads(IN_PATH.read_text())


def fig_h2_cnots(data):
    rows = data["h2"]["rows"]
    methods, cnots = [], []
    for r in rows:
        methods.append(r["method"].replace("raw-gate ", "").replace("ADAPT compiled + ", ""))
        cnots.append(r["cnots"])
    order = np.argsort(cnots)
    methods = [methods[i] for i in order]
    cnots = [cnots[i] for i in order]
    colors = ["#2ecc71" if c == 3 else "#3498db" if c == 7 else "#95a5a6" for c in cnots]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(methods, cnots, color=colors, edgecolor="white", height=0.65)
    ax.set_xlabel("Compiled CNOT count")
    ax.set_title("H₂ (4 qubits): compiled CNOTs by method (exact FCI)")
    ax.axvline(3, color="#2ecc71", ls="--", alpha=0.5, lw=1)
    for b, c in zip(bars, cnots):
        ax.text(c + 0.15, b.get_y() + b.get_height() / 2, str(c), va="center", fontsize=10)
    ax.set_xlim(0, max(cnots) + 3)
    fig.tight_layout()
    fig.savefig(OUT / "fig01_h2_cnot_comparison.png")
    plt.close(fig)


def fig_lih8_fair_pairs(data):
    """Apples-to-apples: same start circuit, greedy vs best RL."""
    pairs = [
        ("1double_chem", "1-double compile → prune @ chem", 7),
        ("adapt_chem", "full ADAPT compile → prune @ chem", 23),
    ]
    greedy_c, rl_best, rl_labels = [], [], []
    for pair_id, _, g in pairs:
        greedy_c.append(g)
        best = None
        label = ""
        for run in data["rl_prune_runs"]:
            if run.get("qubits") != 8 or run.get("pair") != pair_id:
                continue
            if run.get("status", "complete") != "complete":
                continue
            if best is None or run["cnots"] < best:
                best = run["cnots"]
                label = f"s{run.get('seed', '?')}"
        for p in data["partial_runs_log_only"]:
            if p.get("qubits") == 8 and p.get("pair") == pair_id and p.get("cnots"):
                if best is None or p["cnots"] < best:
                    best = p["cnots"]
                    label = f"s{p['seed']}*"
        rl_best.append(best if best is not None else 0)
        rl_labels.append(label)

    x = np.arange(len(pairs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, greedy_c, w, label="Greedy prune", color="#e74c3c")
    b2 = ax.bar(x + w / 2, rl_best, w, label="Best RL prune", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in pairs], fontsize=9)
    ax.set_ylabel("Compiled CNOTs @ chem acc (1.6 mHa)")
    ax.set_title("LiH 8q: fair head-to-head (same starting circuit)")
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{int(h)}",
                        ha="center", va="bottom", fontsize=10)
    for i, lbl in enumerate(rl_labels):
        if lbl and rl_best[i] > 0:
            ax.text(x[i] + w / 2, rl_best[i] + 2.5, lbl, ha="center", fontsize=8, color="#1e8449")
    fig.tight_layout()
    fig.savefig(OUT / "fig02_lih8q_fair_comparison.png")
    plt.close(fig)


def fig_adapt_chem_learning_curves(data):
    runs = [
        r for r in data["rl_prune_runs"]
        if r.get("pair") == "adapt_chem" and r.get("qubits") == 8 and r.get("log")
    ]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for run in sorted(runs, key=lambda r: (r.get("seed", 0), r.get("cnots", 99))):
        log = run["log"]
        xs = [e["update"] for e in log]
        ys = [e["best_cnots"] for e in log]
        dev = run.get("device", "?")[:12]
        seed = run.get("seed", "?")
        final = run["cnots"]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.5,
                label=f"seed {seed} → {final} CNOTs ({dev})")
    ax.axhline(23, color="#e74c3c", ls="--", lw=1.5, label="Greedy floor (23)")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Best CNOTs so far @ chem acc")
    ax.set_title("LiH 8q ADAPT-start RL prune: learning curves")
    ax.legend(fontsize=8, loc="upper right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "fig03_lih8q_adapt_chem_learning.png")
    plt.close(fig)


def fig_multiscale_greedy_rl(data):
    """6q / 8q / 10q: greedy ADAPT-prune vs best RL (where available)."""
    systems = [
        ("LiH 6q", 6, 16, 60),
        ("LiH 8q", 8, 23, 60),
        ("LiH 10q", 10, 60, 60),
    ]
    greedy = [16, 23, 60]
    rl_best = []
    for q in [6, 8, 10]:
        best = None
        for run in data["rl_prune_runs"] + data["partial_runs_log_only"]:
            if run.get("qubits") != q:
                continue
            if run.get("pair") not in ("adapt_chem", "adapt_chem_rl_only"):
                continue
            c = run.get("cnots")
            if c is not None and (best is None or c < best):
                best = c
        rl_best.append(best)

    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, greedy, w, label="Greedy ADAPT→prune @ chem", color="#e74c3c")
    rl_vals = [b if b is not None else 0 for b in rl_best]
    colors = ["#2ecc71" if b else "#bdc3c7" for b in rl_best]
    ax.bar(x + w / 2, rl_vals, w, label="Best RL prune @ chem", color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in systems])
    ax.set_ylabel("Compiled CNOTs")
    ax.set_title("Multi-scale LiH: greedy vs best RL (ADAPT start, chem acc)")
    ax.legend()
    for i, (g, r) in enumerate(zip(greedy, rl_best)):
        ax.text(i - w / 2, g + 1, str(g), ha="center", fontsize=10)
        if r:
            star = "*" if any(p.get("qubits") == [6, 8, 10][i] and p.get("status", "").startswith("timeout")
                              for p in data["partial_runs_log_only"]) else ""
            ax.text(i + w / 2, r + 1, f"{r}{star}", ha="center", fontsize=10)
    ax.text(0.02, 0.02, "* = partial/timeout run (log only, not volume-verified)",
            transform=ax.transAxes, fontsize=8, color="#7f8c8d")
    fig.tight_layout()
    fig.savefig(OUT / "fig04_multiscale_comparison.png")
    plt.close(fig)


def fig_energy_cnot_pareto(data):
    """Energy error vs CNOTs for LiH 8q methods."""
    points = []
    if data.get("fair_greedy_baselines"):
        for r in data["fair_greedy_baselines"]["LiH(2e,4o)"]:
            points.append((r["final_cnots"], r["err_mHa"], r["method"][:40], "greedy/baseline"))

    for run in data["rl_prune_runs"]:
        if run.get("qubits") != 8:
            continue
        if run.get("pair") == "adapt_chem" and run.get("status") == "complete":
            points.append((
                run["cnots"], run["err_mHa"],
                f"RL adapt_chem s{run.get('seed', '?')}", "RL",
            ))
        if run.get("pair") == "1double_chem":
            points.append((
                run["cnots"], run["err_mHa"],
                f"RL 1double s{run.get('seed', '?')}", "RL tie",
            ))

    if data.get("exact_compare_8q"):
        g = data["exact_compare_8q"]["greedy_exact_adapt"]
        points.append((g["cnots"], g["err_mHa"], "Greedy exact ADAPT", "exact"))
        r = data["exact_compare_8q"]["rl_exact_adapt"]
        points.append((r["cnots"], r["err_mHa"], "RL exact ADAPT (partial)", "exact"))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {"greedy/baseline": ("#e74c3c", "s"), "RL": ("#2ecc71", "o"),
              "RL tie": ("#3498db", "o"), "exact": ("#9b59b6", "D")}
    for cnots, err, name, cat in points:
        c, m = styles.get(cat, ("#333", "o"))
        ax.scatter(cnots, err, c=c, marker=m, s=70, edgecolors="white", linewidths=0.5)
        if cnots <= 14 or "RL adapt" in name or "exact" in name.lower():
            ax.annotate(name.split("(")[0].strip()[:22], (cnots, err),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axhline(CHEM, color="#f39c12", ls=":", lw=1.5, label="Chem acc (1.6 mHa)")
    ax.axhline(0.001, color="#9b59b6", ls=":", lw=1, alpha=0.7, label="1 mHa (near-exact)")
    ax.set_xlabel("Compiled CNOT count")
    ax.set_ylabel("|E − E_FCI| (mHa)")
    ax.set_yscale("log")
    ax.set_title("LiH 8q: energy error vs compiled CNOTs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig05_lih8q_energy_cnot_pareto.png")
    plt.close(fig)


def fig_1double_seeds(data):
    runs = [r for r in data["rl_prune_runs"] if r.get("pair") == "1double_chem" and r.get("qubits") == 8]
    if not runs:
        return
    seeds = sorted({r.get("seed", 0) for r in runs})
    cnots = []
    for s in seeds:
        c = next((r["cnots"] for r in runs if r.get("seed") == s), None)
        cnots.append(c or 7)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"seed {s}" for s in seeds], cnots, color="#3498db", width=0.5)
    ax.axhline(7, color="#e74c3c", ls="--", label="Greedy (7)")
    ax.set_ylabel("CNOTs @ chem acc")
    ax.set_title("LiH 8q: 1-double start - RL vs greedy across Modal seeds")
    ax.legend()
    ax.set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig(OUT / "fig06_lih8q_1double_seeds.png")
    plt.close(fig)


def fig_modal_campaign_summary(data):
    """Table-style figure of all completed Modal runs."""
    rows = []
    for run in sorted(data["rl_prune_runs"], key=lambda r: (r.get("qubits", 0), r.get("pair", ""), r.get("seed", 0))):
        if "Modal" not in run.get("device", "") and "modal" not in run.get("source", ""):
            continue
        rows.append([
            f"{run.get('qubits')}q",
            run.get("pair", "?"),
            f"s{run.get('seed', '?')}",
            str(run.get("cnots", "?")),
            f"{run.get('err_mHa', 0):.2f}" if run.get("err_mHa") else "?",
            "✓" if run.get("beats_greedy") else ("tie" if run.get("cnots") == run.get("greedy_baseline_cnots") else "-"),
            run.get("status", "complete")[:12],
        ])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(rows) + 1)))
    ax.axis("off")
    cols = ["System", "Pair", "Seed", "RL CNOTs", "Err mHa", "vs Greedy", "Status"]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Modal GPU campaign - completed RL prune runs", pad=20)
    fig.tight_layout()
    fig.savefig(OUT / "fig07_modal_campaign_table.png")
    plt.close(fig)


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Run consolidate_results.py first ({IN_PATH})")
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    fig_h2_cnots(data)
    fig_lih8_fair_pairs(data)
    fig_adapt_chem_learning_curves(data)
    fig_multiscale_greedy_rl(data)
    fig_energy_cnot_pareto(data)
    fig_1double_seeds(data)
    fig_modal_campaign_summary(data)
    print(f"Wrote figures to {OUT}/")


if __name__ == "__main__":
    main()
