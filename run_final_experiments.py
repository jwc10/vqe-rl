# baselines + hybrid RL + RL prune for the final table

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from adapt import adapt_vqe
from raw_prune import adapt_to_raw_records, count_cnots, greedy_raw_prune
from rl_env import make_h2_config, make_lih_config
from run_master_table import baselines, print_table
from train_raw import train, pick_device
from prune_env import train_prune


def get_cfg(name):
    if name.upper() == "H2":
        return make_h2_config()
    if name.upper() in ("LIH4", "LIH_8Q"):
        return make_lih_config(active_electrons=2, active_orbitals=4)
    return make_lih_config(active_electrons=2, active_orbitals=5)


def fast_preset(mol):
    lih = mol.upper() != "H2"
    return {
        "updates": 25 if lih else 40,
        "episodes_per_update": 8, "hidden": 128, "log_every": 1,
        "inner_restarts": 0, "inner_maxiter": 30, "ppo_epochs": 2, "order_k": 2,
        "max_gates": 12, "max_givens_phase": 2, "prune_max_steps": 20,
        "prune_order_k": 2, "tier": "fast",
    }


def simul_fast_preset(mol):
    lih = mol.upper() != "H2"
    return {
        "updates": 12 if lih else 30,
        "episodes_per_update": 12 if lih else 12, "hidden": 256, "log_every": 1,
        "inner_restarts": 0, "inner_maxiter": 60, "ppo_epochs": 3, "order_k": 3,
        "max_gates": 16 if lih else 14, "max_givens_phase": 4,
        "prune_max_steps": 20, "prune_order_k": 3, "tier": "simul_fast",
        "max_compile_gates": 56,
    }


def quick20_preset(mol):
    lih = mol.upper() != "H2"
    return {
        "updates": 15 if lih else 25,
        "episodes_per_update": 6 if lih else 8, "hidden": 128, "log_every": 1,
        "inner_restarts": 0, "inner_maxiter": 40, "ppo_epochs": 2, "order_k": 3,
        "max_gates": 12, "max_givens_phase": 3, "prune_max_steps": 15,
        "prune_order_k": 2, "tier": "quick20", "max_compile_gates": 36 if lih else 80,
    }


def medium_preset(mol, simul_prune=False):
    lih = mol.upper() != "H2"
    return {
        "updates": 8 if (lih and simul_prune) else (20 if lih else 60),
        "episodes_per_update": 8 if (lih and simul_prune) else (6 if lih else 16),
        "hidden": 256, "log_every": 1, "inner_restarts": 1, "inner_maxiter": 60,
        "ppo_epochs": 4, "order_k": 4, "max_gates": 12 if lih else 16,
        "max_givens_phase": 3 if lih else 4,
        "prune_max_steps": 25 if (lih and simul_prune) else 30,
        "prune_order_k": 3, "tier": "medium", "max_compile_gates": 36 if lih else 80,
    }


def run_hybrid(cfg, updates, target_mode, seed, cnot_penalty=0.02, preset=None,
               simultaneous=False, simul_prune=False):
    f = preset or {}
    if simul_prune:
        mode = "simul_prune"
    elif simultaneous:
        mode = "simultaneous"
    else:
        mode = "chained"
    tier = f.get("tier", "")
    label = f"RL hybrid {mode} ({target_mode})" + (f" [{tier}]" if tier else "")
    print(f"\n=== {label} on {cfg['name']} | {pick_device()} ===", flush=True)
    t0 = time.perf_counter()
    _, _, _, best = train(
        cfg, num_updates=updates, episodes_per_update=f.get("episodes_per_update", 24),
        max_gates=f.get("max_gates", 20), gate_set="hybrid", order_k=f.get("order_k", 4),
        algo="ppo", hidden=f.get("hidden", 512), ppo_epochs=f.get("ppo_epochs", 4),
        moving_threshold=True, target_mode=target_mode, cnot_penalty=cnot_penalty,
        inner_restarts=f.get("inner_restarts", 0), inner_maxiter=f.get("inner_maxiter", 80),
        max_givens_phase=f.get("max_givens_phase", 6), seed=seed,
        log_every=f.get("log_every", max(1, updates // 20)),
        hybrid_mode=mode, hybrid_simultaneous=(mode == "simultaneous"),
        hybrid_preset=f if simul_prune else None,
    )
    dt = time.perf_counter() - t0
    rows = []
    for level in ("chem", "exact"):
        b = best[level]
        if np.isfinite(b.get("cnots", np.inf)):
            rows.append({
                "method": f"{label} @ {level}",
                "cnots": int(b["cnots"]), "n_units": b["n_gates"],
                "error_vs_fci": b["error_vs_fci"],
                "within_chem_acc": b["error_vs_fci"] < 1.6e-3,
                "within_exact": abs(b["error_vs_fci"]) < 1e-6,
                "seconds": dt,
            })
            print(f"  {level}: {b['cnots']} CNOTs, err {b['error_vs_fci']:+.2e}")
    return rows


def run_prune(cfg, target, updates, seed, skip_greedy=False, run_preset=None):
    f = run_preset or {}
    adapt = adapt_vqe(cfg, verbose=False)
    records = adapt_to_raw_records(adapt["actions"], adapt["params"])
    tgt = 1.6e-3 if target == "chem" else 1e-6
    rows = []
    tag = f" [{f.get('tier', '')}]" if f.get("tier") else ""
    print(f"\n=== Prune{tag} on {cfg['name']}: {len(records)} gates, "
          f"{count_cnots(records)} CNOTs ===", flush=True)

    if not skip_greedy:
        gp = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], records,
                              cfg["fci_energy"], chem_acc=tgt)
        rows.append({"method": f"Greedy raw prune ({target})", "cnots": gp["cnots"],
                     "n_units": gp["n_gates"], "error_vs_fci": gp["error_vs_fci"],
                     "within_chem_acc": gp["error_vs_fci"] < 1.6e-3,
                     "within_exact": gp["error_vs_fci"] < 1e-6})

    t0 = time.perf_counter()
    rl = train_prune(
        cfg, records, target=tgt, num_updates=updates,
        episodes_per_update=f.get("episodes_per_update", 32), strict=False,
        hidden=f.get("hidden", 512), seed=seed,
        log_every=f.get("log_every", max(1, updates // 20)),
        max_steps=f.get("prune_max_steps", 0),
        inner_restarts=f.get("inner_restarts", 1),
        inner_maxiter=f.get("inner_maxiter", 100),
        order_k=f.get("prune_order_k", 3),
    )
    dt = time.perf_counter() - t0
    rows.append({"method": f"RL raw prune ({target})", "cnots": rl["cnots"],
                 "n_units": rl["n_gates"], "error_vs_fci": rl["error_vs_fci"],
                 "within_chem_acc": rl["error_vs_fci"] < 1.6e-3,
                 "within_exact": rl.get("within_exact", rl["error_vs_fci"] < 1e-6),
                 "seconds": dt})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["baselines", "hybrid", "prune", "all"])
    ap.add_argument("--molecule", default="H2")
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--target", default="both", choices=["chem", "exact", "both"])
    ap.add_argument("--prune-target", default="chem", choices=["chem", "exact"])
    ap.add_argument("--skip-raw-prune", action="store_true")
    ap.add_argument("--skip-greedy-prune", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--medium", action="store_true")
    ap.add_argument("--quick20", action="store_true")
    ap.add_argument("--simul-fast", action="store_true")
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--hybrid-simultaneous", action="store_true")
    ap.add_argument("--hybrid-simul-prune", action="store_true")
    args = ap.parse_args()

    cfg = get_cfg(args.molecule)
    preset = None
    if args.simul_fast:
        preset = simul_fast_preset(args.molecule)
    elif args.quick20:
        preset = quick20_preset(args.molecule)
    elif args.fast:
        preset = fast_preset(args.molecule)
    elif args.medium:
        preset = medium_preset(args.molecule, simul_prune=args.hybrid_simul_prune)
    if preset and args.log_every is not None:
        preset["log_every"] = args.log_every
    elif args.log_every is not None:
        preset = {"log_every": args.log_every}

    updates = args.updates
    if preset and args.updates == 150:
        updates = preset["updates"]
    if args.simul_fast and not args.hybrid_simul_prune:
        raise SystemExit("--simul-fast needs --hybrid-simul-prune")

    all_rows = []
    tag = cfg["name"].lower().replace("(", "_").replace(")", "").replace(",", "")

    if args.phase in ("baselines", "all"):
        print(f"\n{'='*60}\nBASELINES: {cfg['name']}\n{'='*60}", flush=True)
        rows, _ = baselines(cfg, skip_raw_prune=args.skip_raw_prune or args.fast or args.medium)
        all_rows.extend(rows)
        print_table(all_rows)

    if args.phase in ("hybrid", "all"):
        modes = ["chem", "exact"] if args.target == "both" else [args.target]
        for m in modes:
            if args.hybrid_simul_prune and args.hybrid_simultaneous:
                raise SystemExit("pick one: --hybrid-simultaneous or --hybrid-simul-prune")
            all_rows.extend(run_hybrid(cfg, updates, m, args.seed, preset=preset,
                                       simultaneous=args.hybrid_simultaneous,
                                       simul_prune=args.hybrid_simul_prune))

    if args.phase in ("prune", "all"):
        skip_g = args.skip_greedy_prune or args.fast or args.medium
        all_rows.extend(run_prune(cfg, args.prune_target, updates, args.seed,
                                  skip_greedy=skip_g, run_preset=preset))

    if all_rows:
        print_table(all_rows)
        out = Path("results") / f"final_{tag}.json"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            json.dump({"molecule": cfg["name"], "device": str(pick_device()), "rows": all_rows}, f, indent=2)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
