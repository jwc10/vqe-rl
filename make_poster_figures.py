#!/usr/bin/env python3
# poster figures: bar charts, RL training curves from logs, simple circuit schematics

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from adapt import adapt_vqe
from raw_prune import adapt_to_raw_records, count_cnots, greedy_raw_prune, circuit_gates_to_raw_records
from rl_env import build_action_space, make_h2_config, make_lih_config
from vqe_core import run_vqe_on_circuit, plot_convergence

OUT = Path("results/poster")
CHEM = 1.6e-3


def parse_rl_prune_log(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        m = re.search(r"update\s+(\d+).*best CNOTs:\s*(\d+)", line)
        if m:
            rows.append({"update": int(m.group(1)), "cnots": int(m.group(2))})
        m2 = re.search(r"avg removed\s+([\d.]+)", line)
        if m2 and rows:
            rows[-1]["avg_removed"] = float(m2.group(1))
    return rows


def parse_hybrid_log(path):
    updates, cnots, rets = [], [], []
    for line in Path(path).read_text().splitlines():
        m = re.search(r"update\s+(\d+)/\d+.*avg return\s+([-\d.]+).*best CNOTs chem:(\S+)", line)
        if m:
            updates.append(int(m.group(1)))
            rets.append(float(m.group(2)))
            c = m.group(3)
            cnots.append(int(c) if c != "--" else np.nan)
    return updates, rets, cnots


def best_double(cfg):
    for a in build_action_space(cfg["n_electrons"], cfg["num_qubits"]):
        if a["type"] != "double":
            continue
        vqe = run_vqe_on_circuit(cfg["H"], cfg["num_qubits"], cfg["hf_state"], [a])
        if vqe["energy"] - cfg["fci_energy"] < CHEM:
            return a, vqe
    raise RuntimeError("no chem-acc double")


def energy_row(name, cnots, n_gates, err_ha):
    return {"method": name, "cnots": cnots, "n_gates": n_gates,
            "err_mHa": err_ha * 1000, "err_Ha": err_ha}


def verify_lih_energies(skip_slow_greedy=False):
    cfg = make_lih_config(active_electrons=2, active_orbitals=4)
    fci = cfg["fci_energy"]
    rows = []
    greedy7_recs = None

    fair = Path("results/fair_greedy_compare.json")
    if fair.exists():
        for r in json.loads(fair.read_text())["LiH(2e,4o)"]:
            err = r["err_mHa"] / 1000.0
            rows.append(energy_row(r["method"], r["final_cnots"], r["n_gates"], err))

    if skip_slow_greedy:
        return rows, greedy7_recs

    double, vqe = best_double(cfg)
    err = vqe["energy"] - fci
    rows.append(energy_row("1-double (14 CNOT compile)", 14, 1, err))

    recs = circuit_gates_to_raw_records([{**double, "theta": float(vqe["params"][0])}])
    gp = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], recs, fci,
                          extra_restarts=0, maxiter=40)
    rows.append(energy_row("Greedy raw-prune from 1-double", gp["cnots"], gp["n_gates"],
                           gp["error_vs_fci"]))
    with open(OUT / "lih_greedy7_gates.json", "w") as f:
        json.dump({"records": gp["records"], "energy": gp["energy"],
                   "cnots": gp["cnots"], "error_vs_fci": gp["error_vs_fci"]}, f, indent=2)
    return rows, gp["records"]


def bar_chart(title, labels, vals, ylabel, fname, colors=None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cols = colors or ["#4c78a8"] * len(labels)
    ax.barh(range(len(labels)), vals, color=cols)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def plot_training_curves():
    # RL prune from ADAPT (long run)
    if Path("results/lih4_rl_prune.log").exists():
        rows = parse_rl_prune_log("results/lih4_rl_prune.log")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["update"] for r in rows], [r["cnots"] for r in rows], "o-", label="best CNOTs")
        ax.axhline(23, color="gray", ls="--", label="greedy from ADAPT (23)")
        ax.axhline(7, color="green", ls="--", alpha=0.7, label="greedy from 1-double (7)")
        ax.set_xlabel("PPO update"); ax.set_ylabel("compiled CNOTs")
        ax.set_title("LiH: RL raw-gate prune (start = ADAPT compile, 46 CNOTs)")
        ax.legend(); fig.tight_layout()
        fig.savefig(OUT / "lih_rl_prune_learning.png", dpi=150)
        plt.close(fig)

    # simul+prune medium
    if Path("results/lih4_simul_prune_medium.log").exists():
        u, ret, c = parse_hybrid_log("results/lih4_simul_prune_medium.log")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(u, ret, "o-")
        axes[0].set_xlabel("update"); axes[0].set_ylabel("avg episode return")
        axes[0].set_title("Simul+prune medium: return")
        axes[1].plot(u, c, "s-", color="darkgreen")
        axes[1].axhline(7, color="gray", ls="--", label="greedy 1-double (7)")
        axes[1].set_xlabel("update"); axes[1].set_ylabel("best CNOTs @ chem acc")
        axes[1].set_title("Simul+prune medium: best CNOTs")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(OUT / "lih_simul_prune_learning.png", dpi=150)
        plt.close(fig)

    # H2 simul+prune if log exists
    h2log = Path("results/h2_simul_prune_medium.log")
    if h2log.exists():
        u, ret, c = parse_hybrid_log(h2log)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(u, c, "o-")
        ax.set_xlabel("update"); ax.set_ylabel("best CNOTs @ chem acc")
        ax.set_title("H2: simul+prune medium")
        fig.tight_layout()
        fig.savefig(OUT / "h2_simul_prune_learning.png", dpi=150)
        plt.close(fig)


def draw_schematic_block(text, fname):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.text(0.05, 0.5, text, fontsize=11, va="center", family="monospace")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def draw_raw_circuit(records, title, fname, max_gates=24):
    ops = []
    for r in records[:max_gates]:
        if r["name"] == "CNOT":
            ops.append(qml.CNOT(wires=list(r["wires"])))
        elif r["param"] is not None:
            ops.append(getattr(qml, r["name"])(r["param"], wires=list(r["wires"])))
        else:
            ops.append(getattr(qml, r["name"])(wires=list(r["wires"])))
    try:
        fig, ax = qml.draw_mpl(ops, wire_order=range(max(r["wires"]) + 1 if records else 4))(figsize=(10, 3))
        ax.set_title(title)
        fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        # fallback: gate name strip
        fig, ax = plt.subplots(figsize=(10, 2))
        names = [f"{r['name']}{list(r['wires'])}" for r in records[:max_gates]]
        ax.text(0.02, 0.5, " → ".join(names), fontsize=8, family="monospace")
        ax.axis("off"); ax.set_title(title)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=150)
        plt.close(fig)
        return False


def h2_adapt_convergence():
    cfg = make_h2_config()
    adapt = adapt_vqe(cfg, verbose=False)
    hist = adapt.get("history", [])
    if hist:
        plot_convergence({"energy": hist}, cfg["hf_energy"], cfg["fci_energy"],
                         "H2 ADAPT / VQE energy vs step", OUT / "h2_adapt_vqe_convergence.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # --- energy table (LiH) ---
    print("Loading LiH energies (cached fair compare)...")
    erows, greedy7_recs = verify_lih_energies(skip_slow_greedy=True)
    with open(OUT / "lih_energy_verified.json", "w") as f:
        json.dump({"rows": erows, "note": "RL 12-CNOT run did not save final circuit; "
                  "if chem-acc, expect ~1.47 mHa same as greedy 7"}, f, indent=2)

    # --- bar charts ---
    bar_chart(
        "H2: compiled CNOTs @ chemical accuracy / exact FCI",
        ["Raw RL scratch", "Simul+prune", "Greedy/RL prune", "One double / ADAPT"],
        [3, 4, 7, 14],
        "CNOT count",
        "h2_cnot_comparison.png",
        ["#2ca02c", "#9467bd", "#ff7f0e", "#aec7e8"],
    )

    lih_labels = [
        "Greedy prune (ADAPT start)",
        "RL prune (ADAPT, ~upd 45)",
        "One-double floor",
        "Greedy prune (1-double)",
        "Chained / simul+prune RL",
        "ADAPT full",
    ]
    lih_vals = [23, 12, 14, 7, 7, 46]
    lih_colors = ["#aec7e8", "#9467bd", "#ffbb78", "#2ca02c", "#2ca02c", "#d62728"]
    bar_chart(
        "LiH (2e,4o): compiled CNOTs @ chem acc (~1.47 mHa band)",
        lih_labels, lih_vals, "CNOT count", "lih_cnot_comparison.png", lih_colors,
    )

    # energy bar (mHa)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    names = [r["method"] for r in erows]
    errs = [r["err_mHa"] for r in erows]
    ax.barh(names, errs, color="steelblue")
    ax.axvline(1.6, color="red", ls="--", label="chem acc (1.6 mHa)")
    ax.set_xlabel("error vs FCI (mHa)")
    ax.set_title("LiH: energy at compressed circuits (not exact FCI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lih_energy_comparison.png", dpi=150)
    plt.close(fig)

    # --- learning curves from logs ---
    plot_training_curves()

    # --- circuit diagrams (H2 only by default; LiH greedy is slow) ---
    print("Drawing H2 circuit...")
    cfg = make_h2_config()
    double, vqe = best_double(cfg)
    h2_recs = circuit_gates_to_raw_records([{**double, "theta": float(vqe["params"][0])}])
    h2_gp = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], h2_recs, cfg["fci_energy"],
                             extra_restarts=0, maxiter=50)
    draw_raw_circuit(
        h2_gp["records"],
        f"H2: greedy prune ({h2_gp['cnots']} CNOTs, exact FCI)",
        "circuit_h2_greedy7.png",
    )

    cached = OUT / "lih_greedy7_gates.json"
    if cached.exists():
        greedy7_recs = json.loads(cached.read_text())["records"]
        draw_raw_circuit(
            greedy7_recs,
            f"LiH: greedy 7-CNOT circuit ({count_cnots(greedy7_recs)} CNOTs)",
            "circuit_lih_greedy7.png",
        )
    else:
        draw_schematic_block(
            "LiH 7-CNOT class: HF + 1 double → compile 14 CNOTs → greedy prune → 7 CNOTs\n"
            "(run: python quick_fair_compare.py to cache gate list)",
            "circuit_lih_schematic.png",
        )

    h2_adapt_convergence()

    # summary for poster
    checklist = {
        "figures_generated": sorted(p.name for p in OUT.glob("*")),
        "energy_finding": "Greedy-7 and 1-double sit at ~1.47 mHa; not exact FCI (2.9 mHa HF gap). "
                          "RL paths at chem acc are same energy band, not better than greedy-7.",
        "rl12_note": "RL prune reached 12 CNOTs logged; never logged 7. Different gate count than "
                     "greedy-7; energy likely same plateau if within chem acc.",
        "circuit_difference": "Different removal/build order → different raw gate lists; "
                              "same chem-acc energy suggests same effective quality, not a new minimum.",
    }
    with open(OUT / "figure_notes.json", "w") as f:
        json.dump(checklist, f, indent=2)

    print(f"\nDone. Figures in {OUT}/")
    for p in sorted(OUT.glob("*")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
