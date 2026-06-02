# lih_raw_experiment.py
# LiH (small active space, tractable) experiment driving toward a novel result:
#   (A) method comparison on compiled CNOTs: ADAPT vs greedy raw-gate prune vs raw-gate RL
#   (B) ablation: does ORDER encoding help raw-gate RL? (order_k=0 vs >0)
#   (C) ablation: does a PHYSICS-INSPIRED particle-number penalty help? (number_penalty 0 vs >0)
#
# (B) and (C) are the parts not really settled in the literature -- even a negative result is
# a reportable finding. Uses PPO + curriculum + the L-BFGS inner optimizer.

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adapt import adapt_vqe
from raw_prune import adapt_to_raw_records, count_cnots, greedy_raw_prune
from rl_env import make_lih_config
from train_raw import train
from vqe_core import compiled_cnots_for_actions


def rl_row(label, cfg, **kw):
    t = time.perf_counter()
    _, _, hist, best = train(cfg, log_every=10, **kw)
    dt = time.perf_counter() - t
    b = best["chem"]
    cnots = int(b["cnots"]) if np.isfinite(b["cnots"]) else None
    err = b.get("error_vs_fci")
    print(f"  [{label}] best CNOTs@chem-acc: {cnots}  err {err if err is None else f'{err:+.2e}'}  "
          f"({dt:.0f}s)")
    return {"method": label, "cnots": cnots, "n_units": b.get("n_gates"),
            "error_vs_fci": err, "seconds": dt,
            "cnots_curve": [None if not np.isfinite(c) else int(c) for c in hist["best_cnots_chem"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orbitals", type=int, default=3, help="active orbitals (3->6 qubits)")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--max-gates", type=int, default=14)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--number-penalty", type=float, default=0.2)
    ap.add_argument("--order-k", type=int, default=4)
    ap.add_argument("--inner-maxiter", type=int, default=60)
    ap.add_argument("--skip-greedy-prune", action="store_true",
                    help="skip O(n^2) greedy raw-gate prune (intractable at 10 qubits)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = make_lih_config(active_electrons=2, active_orbitals=args.orbitals)
    print(f"{cfg['name']}: {cfg['num_qubits']} qubits  HF {cfg['hf_energy']:.6f}  "
          f"FCI {cfg['fci_energy']:.6f}\n")
    rows = []

    # (A) classical / rule-based baselines
    adapt = adapt_vqe(cfg, verbose=False)
    rows.append({"method": "ADAPT (excitations)",
                 "cnots": compiled_cnots_for_actions(adapt["actions"]),
                 "n_units": adapt["n_excitations"], "error_vs_fci": adapt["error_vs_fci"]})
    print(f"  [ADAPT] {rows[-1]['cnots']} CNOTs, {adapt['n_excitations']} excitations, "
          f"err {adapt['error_vs_fci']:+.2e}")

    records = adapt_to_raw_records(adapt["actions"], adapt["params"])
    if args.skip_greedy_prune:
        print(f"  ADAPT compiled to {len(records)} raw gates ({count_cnots(records)} CNOTs); "
              f"skipping greedy prune (O(n^2), intractable at this size)")
    else:
        print(f"  ADAPT compiled to {len(records)} raw gates ({count_cnots(records)} CNOTs); "
              f"greedy raw-gate pruning...")
        gp = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], records,
                              cfg["fci_energy"])
        rows.append({"method": "ADAPT + greedy raw-gate prune", "cnots": gp["cnots"],
                     "n_units": gp["n_gates"], "error_vs_fci": gp["error_vs_fci"]})
        print(f"  [greedy raw prune] {gp['cnots']} CNOTs")

    # (B)+(C) RL ablations (fast inner loop for tractability)
    common = dict(num_updates=args.updates, max_gates=args.max_gates,
                  episodes_per_update=args.episodes, curriculum=True, algo="ppo",
                  inner_restarts=0, inner_maxiter=args.inner_maxiter, seed=args.seed)
    print("\nRL from scratch (raw gates):")
    rows.append(rl_row("RL raw bag-of-gates", cfg, gate_set="raw", order_k=0,
                       number_penalty=0.0, **common))
    rows.append(rl_row("RL raw +order", cfg, gate_set="raw", order_k=args.order_k,
                       number_penalty=0.0, **common))
    rows.append(rl_row("RL raw +order +symmetry", cfg, gate_set="raw", order_k=args.order_k,
                       number_penalty=args.number_penalty, **common))
    # the physics-inspired fix: particle-conserving Givens action space (every action is a
    # potential gap-closer, so exploration actually works)
    rows.append(rl_row("RL givens +order", cfg, gate_set="givens", order_k=args.order_k,
                       number_penalty=0.0, **common))

    # summary table
    print(f"\n{'method':32s} {'CNOTs':>6s} {'units':>6s} {'err vs FCI':>12s}")
    print("-" * 62)
    for r in rows:
        c = r["cnots"] if r["cnots"] is not None else "--"
        e = f"{r['error_vs_fci']:+.2e}" if r.get("error_vs_fci") is not None else "--"
        print(f"{r['method']:32s} {str(c):>6s} {str(r['n_units']):>6s} {e:>12s}")

    out = Path("results"); out.mkdir(exist_ok=True)
    tag = cfg["name"].lower().split("(")[0] + f"_{cfg['num_qubits']}q"
    with open(out / f"lih_raw_{tag}.json", "w") as f:
        json.dump({"config": cfg["name"], "qubits": cfg["num_qubits"],
                   "fci": cfg["fci_energy"], "rows": rows}, f, indent=2)

    # bar chart of CNOTs per method
    labels = [r["method"] for r in rows]
    vals = [r["cnots"] if r["cnots"] is not None else 0 for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(labels)), vals, color="steelblue")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("compiled CNOTs @ chemical accuracy")
    ax.set_title(f"{cfg['name']} ({cfg['num_qubits']} qubits): circuit cost by method")
    fig.tight_layout(); fig.savefig(out / f"lih_raw_{tag}.png", dpi=150); plt.close(fig)
    print(f"\nSaved results/lih_raw_{tag}.json and .png")


if __name__ == "__main__":
    main()
