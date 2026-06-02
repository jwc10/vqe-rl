# run_master_table.py
# One-shot baseline + RL comparison table for the final report.
# Includes ADAPT full, ADAPT@chem-acc prefix, greedy exc/raw prune, and optional RL rows.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adapt import adapt_vqe
from prune_baselines import greedy_backward_elimination
from raw_prune import adapt_to_raw_records, count_cnots, greedy_raw_prune
from rl_env import build_action_space, make_h2_config, make_lih_config
from vqe_core import compiled_cnots_for_actions, run_vqe_on_circuit

CHEM = 1.6e-3


def _row(method, cnots, n_units, err, **extra):
    return {"method": method, "cnots": cnots, "n_units": n_units,
            "error_vs_fci": err, "within_chem_acc": err is not None and err < CHEM,
            "within_exact": err is not None and abs(err) < 1e-6, **extra}


def baselines(cfg, skip_raw_prune=False):
    rows = []
    fci = cfg["fci_energy"]
    adapt = adapt_vqe(cfg, verbose=False)
    rows.append(_row("ADAPT full", compiled_cnots_for_actions(adapt["actions"]),
                     adapt["n_excitations"], adapt["error_vs_fci"]))

    if adapt.get("depth_to_chem_acc"):
        n = adapt["depth_to_chem_acc"]
        prefix = adapt["actions"][:n]
        vqe = run_vqe_on_circuit(cfg["H"], cfg["num_qubits"], cfg["hf_state"], prefix)
        rows.append(_row("ADAPT prefix @ chem acc", compiled_cnots_for_actions(prefix),
                         n, vqe["energy"] - fci))

    gp_exc = greedy_backward_elimination(cfg, adapt["actions"])
    rows.append(_row("Greedy excitation prune", gp_exc["cnots"], gp_exc["n_excitations"],
                     gp_exc["error_vs_fci"]))

    pool = [a for a in build_action_space(cfg["n_electrons"], cfg["num_qubits"])
            if a["type"] != "stop"]
    best = None
    for a in pool:
        if a["type"] != "double":
            continue
        vqe = run_vqe_on_circuit(cfg["H"], cfg["num_qubits"], cfg["hf_state"], [a])
        err = vqe["energy"] - fci
        c = compiled_cnots_for_actions([a])
        if err < CHEM and (best is None or c < best[0]):
            best = (c, err)
    if best:
        rows.append(_row("Best single double @ chem", best[0], 1, best[1]))

    if not skip_raw_prune:
        records = adapt_to_raw_records(adapt["actions"], adapt["params"])
        gp_raw = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], records, fci)
        rows.append(_row("Greedy raw-gate prune", gp_raw["cnots"], gp_raw["n_gates"],
                         gp_raw["error_vs_fci"]))
    return rows, adapt


def print_table(rows):
    print(f"\n{'method':36s} {'CNOTs':>6s} {'units':>6s} {'err vs FCI':>12s} {'chem':>5s} {'exact':>5s}")
    print("-" * 78)
    for r in rows:
        c = r["cnots"] if r["cnots"] is not None else "--"
        e = f"{r['error_vs_fci']:+.2e}" if r.get("error_vs_fci") is not None else "--"
        print(f"{r['method']:36s} {str(c):>6s} {str(r['n_units']):>6s} {e:>12s} "
              f"{'Y' if r.get('within_chem_acc') else 'N':>5s} "
              f"{'Y' if r.get('within_exact') else 'N':>5s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="H2", choices=["H2", "LiH4", "LiH5"])
    ap.add_argument("--skip-raw-prune", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.molecule == "H2":
        cfg = make_h2_config()
    elif args.molecule == "LiH4":
        cfg = make_lih_config(active_electrons=2, active_orbitals=4)
    else:
        cfg = make_lih_config(active_electrons=2, active_orbitals=5)

    print(f"{cfg['name']}: {cfg['num_qubits']} qubits  gap={(cfg['hf_energy']-cfg['fci_energy'])*1e3:.2f} mHa")
    rows, _ = baselines(cfg, skip_raw_prune=args.skip_raw_prune)
    print_table(rows)

    out = Path(args.out) if args.out else Path("results") / f"master_{args.molecule.lower()}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"molecule": cfg["name"], "qubits": cfg["num_qubits"], "rows": rows}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
