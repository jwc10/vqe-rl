#!/usr/bin/env python3
# brute search: can LiH chem acc work below 7 compiled CNOTs?

from copy import deepcopy
import json
from pathlib import Path

from rl_env import build_action_space, make_lih_config
from raw_prune import circuit_gates_to_raw_records, count_cnots, greedy_raw_prune
from vqe_core import run_vqe_on_circuit

CHEM = 1.6e-3


def best_double(cfg):
    for a in build_action_space(cfg["n_electrons"], cfg["num_qubits"]):
        if a["type"] != "double":
            continue
        vqe = run_vqe_on_circuit(cfg["H"], cfg["num_qubits"], cfg["hf_state"], [a])
        if vqe["energy"] - cfg["fci_energy"] < CHEM:
            return circuit_gates_to_raw_records([{**a, "theta": float(vqe["params"][0])}])
    raise RuntimeError("no double")


def try_remove_one_more(cfg, records):
    fci = cfg["fci_energy"]
    n = len(records)
    best = {"cnots": count_cnots(records), "err": 1e9}
    for i in range(n):
        trial = [deepcopy(records[j]) for j in range(n) if j != i]
        if not trial:
            continue
        from raw_prune import _optimized_energy
        e = _optimized_energy(cfg["H"], cfg["num_qubits"], cfg["hf_state"], trial,
                              extra_restarts=1, maxiter=80)
        err = e - fci
        c = count_cnots(trial)
        if err < CHEM and (c < best["cnots"] or (c == best["cnots"] and err < best["err"])):
            best = {"cnots": c, "err": err, "n_gates": len(trial)}
    return best


def main():
    cfg = make_lih_config(2, 4)
    fci = cfg["fci_energy"]
    recs = best_double(cfg)
    gp = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], recs, fci,
                          chem_acc=CHEM, extra_restarts=1, maxiter=80)
    print(f"greedy: {gp['cnots']} CNOTs, {gp['n_gates']} gates, err={gp['error_vs_fci']*1000:.4f} mHa")

    extra = try_remove_one_more(cfg, gp["records"])
    print(f"greedy + one more removal try: {extra}")

    out = Path("results/beat_greedy/brute_floor.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"greedy": gp, "one_more_removal": extra}, indent=2))


if __name__ == "__main__":
    main()
