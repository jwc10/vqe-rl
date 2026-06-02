#!/usr/bin/env python3
"""Greedy raw-prune from full ADAPT compile: chem-acc vs exact-FCI stopping."""
import json
import time
from pathlib import Path

from adapt import adapt_vqe
from raw_prune import adapt_to_raw_records, count_cnots, greedy_raw_prune
from rl_env import make_lih_config

CHEM = 1.6e-3
EXACT = 1e-6
OUT = Path("results/beat_greedy/greedy_adapt_exact.json")


def main():
    cfg = make_lih_config(2, 4)
    fci = cfg["fci_energy"]
    print("=== Greedy ADAPT prune: chem (1.6 mHa) then exact (1e-6 Ha) ===", flush=True)
    print("Running ADAPT-VQE...", flush=True)
    t0 = time.perf_counter()
    adapt = adapt_vqe(cfg, verbose=False)
    ar = adapt_to_raw_records(adapt["actions"], adapt["params"])
    print(f"ADAPT compile: {len(ar)} gates, {count_cnots(ar)} CNOTs, "
          f"err={(adapt['energy'] - fci) * 1e3:.4f} mHa ({time.perf_counter() - t0:.1f}s)", flush=True)

    rows = []
    for tol, name in [(CHEM, "chem_1.6mHa"), (EXACT, "exact_1e-6_Ha")]:
        print(f"Greedy prune (stop when err < {name})...", flush=True)
        t1 = time.perf_counter()
        gp = greedy_raw_prune(
            cfg["H"], cfg["num_qubits"], cfg["hf_state"], ar, fci,
            chem_acc=tol, extra_restarts=0, maxiter=50, verbose=True,
        )
        row = {
            "target": name,
            "tol_Ha": tol,
            "cnots": gp["cnots"],
            "n_gates": gp["n_gates"],
            "n_evals": gp["n_evals"],
            "err_mHa": gp["error_vs_fci"] * 1e3,
            "within_exact": gp["error_vs_fci"] < EXACT,
            "within_chem_acc": gp["error_vs_fci"] < CHEM,
            "seconds": time.perf_counter() - t1,
        }
        rows.append(row)
        print(f"  -> {row['cnots']} CNOTs, {row['err_mHa']:.6f} mHa, "
              f"exact={row['within_exact']} ({row['seconds']:.1f}s)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"adapt_start_cnots": count_cnots(ar), "fci": fci, "results": rows}
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
