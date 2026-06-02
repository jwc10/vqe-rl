#!/usr/bin/env python3
"""Fair compare: greedy raw-prune from ONE-DOUBLE compile (hybrid's start) vs ADAPT compile."""
from __future__ import annotations

import json
from pathlib import Path

from rl_env import build_action_space, make_h2_config, make_lih_config
from raw_prune import adapt_to_raw_records, count_cnots, greedy_raw_prune
from adapt import adapt_vqe
from vqe_core import compiled_cnots_for_actions, run_vqe_on_circuit

CHEM = 1.6e-3


def best_double_records(cfg):
    for a in build_action_space(cfg["n_electrons"], cfg["num_qubits"]):
        if a["type"] != "double":
            continue
        vqe = run_vqe_on_circuit(cfg["H"], cfg["num_qubits"], cfg["hf_state"], [a])
        err = vqe["energy"] - cfg["fci_energy"]
        if err < CHEM:
            recs = adapt_to_raw_records([a], list(vqe["params"]))
            return recs, err, compiled_cnots_for_actions([a])
    raise RuntimeError("no chem-acc double found")


def run_molecule(name, cfg):
    fci = cfg["fci_energy"]
    rows = []

    recs, err_exc, c_exc = best_double_records(cfg)
    rows.append({
        "method": "1-double excitation (no prune)",
        "start_cnots": count_cnots(recs),
        "final_cnots": count_cnots(recs),
        "n_gates": len(recs),
        "err_mHa": (err_exc) * 1000,
    })
    print(f"\n=== {name} | 1-double compile: {len(recs)} gates, {count_cnots(recs)} CNOTs ===", flush=True)

    gp = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], recs, fci,
                          chem_acc=CHEM, extra_restarts=0, maxiter=40)
    rows.append({
        "method": "Greedy raw-prune FROM 1-double compile",
        "start_cnots": count_cnots(recs),
        "final_cnots": gp["cnots"],
        "n_gates": gp["n_gates"],
        "err_mHa": gp["error_vs_fci"] * 1000,
        "n_evals": gp["n_evals"],
    })

    adapt = adapt_vqe(cfg, verbose=False)
    recs_adapt = adapt_to_raw_records(adapt["actions"], adapt["params"])
    gp2 = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], recs_adapt, fci,
                           chem_acc=CHEM, extra_restarts=0, maxiter=40)
    rows.append({
        "method": "Greedy raw-prune FROM ADAPT compile (reference)",
        "start_cnots": count_cnots(recs_adapt),
        "final_cnots": gp2["cnots"],
        "n_gates": gp2["n_gates"],
        "err_mHa": gp2["error_vs_fci"] * 1000,
    })

    print(f"{'method':42s} {'start':>5s} {'final':>5s} {'err mHa':>8s}")
    for r in rows:
        print(f"{r['method']:42s} {r['start_cnots']:5d} {r['final_cnots']:5d} {r['err_mHa']:8.3f}")
    return rows


def main():
    all_out = {}
    for label, cfg in [("H2", make_h2_config()), ("LiH(2e,4o)", make_lih_config(2, 4))]:
        all_out[label] = run_molecule(label, cfg)
    out = Path("results/fair_greedy_compare.json")
    out.write_text(json.dumps(all_out, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
