#!/usr/bin/env python3
"""One-shot LiH(2e,4o) energy table: err vs FCI in Ha and mHa."""
from __future__ import annotations

import json
from pathlib import Path

from adapt import adapt_vqe
from prune_baselines import greedy_backward_elimination
from raw_prune import adapt_to_raw_records, count_cnots, greedy_raw_prune, _optimized_energy
from rl_env import build_action_space, make_lih_config
from vqe_core import compiled_cnots_for_actions, run_vqe_on_circuit

CHEM = 1.6e-3  # Ha


def mha(err_ha):
    return err_ha * 1000.0


def row(method, cnots, err, n_units=None):
    return {
        "method": method,
        "cnots": cnots,
        "n_units": n_units,
        "err_vs_fci_Ha": err,
        "err_vs_fci_mHa": mha(err),
        "within_chem_acc": err < CHEM,
        "within_exact_1e-6": abs(err) < 1e-6,
    }


def main():
    cfg = make_lih_config(active_electrons=2, active_orbitals=4)
    fci = cfg["fci_energy"]
    hf = cfg["hf_energy"]
    gap = hf - fci
    rows = [
        row("HF reference", 0, hf - fci, 0),
        row("FCI (target)", None, 0.0),
    ]
    print(f"LiH(2e,4o)  FCI = {fci:.8f} Ha   HF = {hf:.8f} Ha   correlation gap = {mha(gap):.2f} mHa\n")
    print(f"{'method':42s} {'CNOTs':>5s} {'err (mHa)':>10s} {'err (Ha)':>12s}  chem?  exact?")
    print("-" * 85)

    # 14-CNOT baseline: best single double
    best_d = None
    for a in build_action_space(cfg["n_electrons"], cfg["num_qubits"]):
        if a["type"] != "double":
            continue
        vqe = run_vqe_on_circuit(cfg["H"], cfg["num_qubits"], cfg["hf_state"], [a])
        err = vqe["energy"] - fci
        c = compiled_cnots_for_actions([a])
        if err < CHEM and (best_d is None or c < best_d[0]):
            best_d = (c, err, 1)
    if best_d:
        rows.append(row("14-CNOT baseline (1 double excitation)", best_d[0], best_d[1], best_d[2]))

    # ADAPT full
    adapt = adapt_vqe(cfg, verbose=False)
    rows.append(row("ADAPT full (excitations)", compiled_cnots_for_actions(adapt["actions"]),
                    adapt["error_vs_fci"], adapt["n_excitations"]))

    # Greedy excitation prune
    gp_exc = greedy_backward_elimination(cfg, adapt["actions"])
    rows.append(row("Greedy excitation prune", gp_exc["cnots"], gp_exc["error_vs_fci"],
                    gp_exc["n_excitations"]))

    # Greedy raw-gate prune from compiled ADAPT (slow — fast inner)
    print("Running greedy raw-gate prune (may take several min)...", flush=True)
    records = adapt_to_raw_records(adapt["actions"], adapt["params"])
    gp_raw = greedy_raw_prune(
        cfg["H"], cfg["num_qubits"], cfg["hf_state"], records, fci,
        chem_acc=CHEM, extra_restarts=0, maxiter=40)
    rows.append(row("Greedy raw-gate prune (from ADAPT compile)",
                    gp_raw["cnots"], gp_raw["error_vs_fci"], gp_raw["n_gates"]))

    for r in rows:
        if r["method"] == "FCI (target)":
            continue
        c = r["cnots"] if r["cnots"] is not None else "--"
        print(f"{r['method']:42s} {str(c):>5s} {r['err_vs_fci_mHa']:10.3f} {r['err_vs_fci_Ha']:12.4e}  "
              f"{'Y' if r['within_chem_acc'] else 'N':>5s}  {'Y' if r['within_exact_1e-6'] else 'N':>5s}")

    out = Path("results/lih_energy_table.json")
    out.write_text(json.dumps({"fci": fci, "hf": hf, "chem_acc_Ha": CHEM, "rows": rows}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
