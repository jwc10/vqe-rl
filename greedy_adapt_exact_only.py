#!/usr/bin/env python3
"""Greedy raw-prune from full ADAPT compile, exact-FCI stop only (fair vs RL 25 CNOTs)."""
import json
import time
from pathlib import Path

from adapt import adapt_vqe
from raw_prune import adapt_to_raw_records, count_cnots, greedy_raw_prune
from rl_env import make_lih_config

EXACT = 1e-6
OUT = Path("results/beat_greedy/greedy_adapt_exact_only.json")
LOG = Path("results/beat_greedy/greedy_adapt_exact_only.log")


def main():
    cfg = make_lih_config(2, 4)
    fci = cfg["fci_energy"]
    print("ADAPT-VQE...", flush=True)
    t0 = time.perf_counter()
    adapt = adapt_vqe(cfg, verbose=False)
    ar = adapt_to_raw_records(adapt["actions"], adapt["params"])
    print(f"start: {len(ar)} gates, {count_cnots(ar)} CNOTs, "
          f"err={(adapt['energy'] - fci) * 1e3:.6f} mHa ({time.perf_counter() - t0:.1f}s)", flush=True)

    print(f"Greedy prune (exact: err < {EXACT:.1e} Ha)...", flush=True)
    t1 = time.perf_counter()
    gp = greedy_raw_prune(
        cfg["H"], cfg["num_qubits"], cfg["hf_state"], ar, fci,
        chem_acc=EXACT, extra_restarts=0, maxiter=50, verbose=True,
    )
    row = {
        "method": "greedy_raw_prune_from_ADAPT_compile",
        "target_Ha": EXACT,
        "cnots": gp["cnots"],
        "n_gates": gp["n_gates"],
        "n_evals": gp["n_evals"],
        "err_mHa": gp["error_vs_fci"] * 1e3,
        "within_exact": gp["error_vs_fci"] < EXACT,
        "seconds": time.perf_counter() - t1,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"fci": fci, "adapt_start_cnots": count_cnots(ar), "result": row}, indent=2))
    print(f"DONE: {row['cnots']} CNOTs, {row['err_mHa']:.6f} mHa, exact={row['within_exact']} "
          f"({row['seconds']:.1f}s greedy, {row['n_evals']} evals)", flush=True)
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
