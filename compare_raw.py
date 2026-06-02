# compare_raw.py
# Head-to-head: does RL building a raw-gate circuit FROM SCRATCH reach chemical accuracy
# with fewer compiled CNOTs than rule-based pipelines that start from ADAPT's known-good
# excitation circuit and prune?
#
#   1. ADAPT (excitations)                      -- chemistry baseline
#   2. ADAPT + greedy excitation pruning        -- rule-based, remove whole excitations
#   3. ADAPT compiled to raw gates + greedy raw-gate pruning  -- rule-based, remove raw gates
#   4. raw-gate RL from scratch                 -- learned policy
#
# Metric: compiled CNOT count at chemical accuracy (1.6e-3 Ha).

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adapt import adapt_vqe
from prune_baselines import greedy_backward_elimination
from raw_prune import adapt_to_raw_records, count_cnots, greedy_raw_prune
from train_raw import train
from vqe_core import compiled_cnots_for_actions


def run(molecule="H2", rl_updates=60, max_gates=12, seed=0):
    if molecule.upper() == "H2":
        from rl_env import make_h2_config
        cfg = make_h2_config()
    else:
        from rl_env import make_lih_config
        cfg = make_lih_config(active_electrons=2, active_orbitals=5)

    fci = cfg["fci_energy"]
    print(f"{cfg['name']}: HF {cfg['hf_energy']:.6f}  FCI {fci:.6f}\n")
    rows = []

    # 1. ADAPT (excitations)
    adapt = adapt_vqe(cfg, verbose=False)
    rows.append({
        "method": "ADAPT (excitations)",
        "cnots": compiled_cnots_for_actions(adapt["actions"]),
        "n_units": adapt["n_excitations"],
        "error_vs_fci": adapt["error_vs_fci"],
    })

    # 2. ADAPT + greedy excitation pruning
    exc_prune = greedy_backward_elimination(cfg, adapt["actions"])
    rows.append({
        "method": "ADAPT + greedy excitation prune",
        "cnots": exc_prune["cnots"],
        "n_units": exc_prune["n_excitations"],
        "error_vs_fci": exc_prune["error_vs_fci"],
    })

    # 3. ADAPT compiled to raw gates + greedy raw-gate pruning
    records = adapt_to_raw_records(adapt["actions"], adapt["params"])
    print(f"ADAPT compiled to {len(records)} raw gates ({count_cnots(records)} CNOTs); "
          f"greedy raw-gate pruning...")
    raw_prune = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], records, fci)
    rows.append({
        "method": "ADAPT compiled + greedy raw-gate prune",
        "cnots": raw_prune["cnots"],
        "n_units": raw_prune["n_gates"],
        "error_vs_fci": raw_prune["error_vs_fci"],
    })

    # 3b. ADAPT compiled to raw gates + RL raw-gate pruning (ablation)
    print("RL raw-gate pruning...")
    from prune_env import train_prune
    rl_prune = train_prune(cfg, records, target=1.6e-3, num_updates=40, seed=seed,
                           log_every=0)
    rows.append({
        "method": "ADAPT compiled + RL raw-gate prune",
        "cnots": rl_prune["cnots"],
        "n_units": rl_prune["n_gates"],
        "error_vs_fci": rl_prune["error_vs_fci"],
    })

    # 4. raw-gate RL from scratch (report both chem-acc and exact-FCI best circuits)
    print(f"training raw-gate RL from scratch ({rl_updates} updates)...")
    _, _, _, best = train(cfg, num_updates=rl_updates, max_gates=max_gates, seed=seed,
                          log_every=0, curriculum=True)
    for level, label in (("chem", "raw-gate RL from scratch (chem acc)"),
                         ("exact", "raw-gate RL from scratch (exact FCI)")):
        b = best[level]
        rows.append({
            "method": label,
            "cnots": int(b["cnots"]) if np.isfinite(b["cnots"]) else None,
            "n_units": b.get("n_gates"),
            "error_vs_fci": b.get("error_vs_fci"),
        })

    print(f"\n{'method':42s} {'CNOTs':>6s} {'units':>6s} {'err vs FCI':>12s}")
    print("-" * 70)
    for r in rows:
        c = r["cnots"] if r["cnots"] is not None else "--"
        e = f"{r['error_vs_fci']:+.2e}" if r["error_vs_fci"] is not None else "--"
        print(f"{r['method']:42s} {str(c):>6s} {str(r['n_units']):>6s} {e:>12s}")

    out = Path("results"); out.mkdir(exist_ok=True)
    tag = cfg["name"].lower().split("(")[0]
    with open(out / f"compare_raw_{tag}.json", "w") as f:
        json.dump({"molecule": cfg["name"], "fci": fci, "rows": rows}, f, indent=2)
    print(f"\nSaved results/compare_raw_{tag}.json")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="H2")
    ap.add_argument("--rl-updates", type=int, default=60)
    ap.add_argument("--max-gates", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.molecule, args.rl_updates, args.max_gates, args.seed)


if __name__ == "__main__":
    main()
