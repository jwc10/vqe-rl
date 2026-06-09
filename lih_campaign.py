#!/usr/bin/env python3
# LiH fair-pair campaign: same start circuit, greedy vs RL prune.
# Phases: baselines, focus (1-double + adapt chem), prune, smoke.

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from adapt import adapt_vqe
from lih_campaign_config import (
    CHEM, EXACT, FAIR_PAIRS, PRESETS, TARGET_CHEM_CNOTS,
)
from prune_trainer import train_prune_campaign
from raw_prune import adapt_to_raw_records, count_cnots, greedy_raw_prune
from rl_env import build_action_space, make_lih_config
from vqe_core import run_vqe_on_circuit

OUT = Path("results/lih_campaign")
FAIR_COMPARE_PATH = Path("results/fair_greedy_compare.json")


def one_double_records(cfg):
    for a in build_action_space(cfg["n_electrons"], cfg["num_qubits"]):
        if a["type"] != "double":
            continue
        vqe = run_vqe_on_circuit(cfg["H"], cfg["num_qubits"], cfg["hf_state"], [a])
        if vqe["energy"] - cfg["fci_energy"] < CHEM:
            return adapt_to_raw_records([a], list(vqe["params"]))
    raise RuntimeError("no chem-acc double found")


def _load_cached_greedy():
    if not FAIR_COMPARE_PATH.exists():
        return {}
    rows = json.loads(FAIR_COMPARE_PATH.read_text()).get("LiH(2e,4o)", [])
    out = {}
    for r in rows:
        if "FROM 1-double" in r["method"]:
            out["greedy_1double_chem"] = {
                "cnots": r["final_cnots"], "err_mHa": r["err_mHa"],
                "source": str(FAIR_COMPARE_PATH),
            }
        if "FROM ADAPT" in r["method"]:
            out["greedy_adapt_chem"] = {
                "cnots": r["final_cnots"], "err_mHa": r["err_mHa"],
                "source": str(FAIR_COMPARE_PATH),
            }
    return out


def run_baselines(cfg, *, run_adapt_greedy=True, run_exact_greedy=False):
    # greedy floors: 1-double is fast (~1 min), ADAPT greedy is slow (~25-40 min)
    fci = cfg["fci_energy"]
    rows = _load_cached_greedy()
    od = one_double_records(cfg)

    if "greedy_1double_chem" not in rows:
        g1 = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], od, fci,
                              chem_acc=CHEM, extra_restarts=0, maxiter=50)
        rows["greedy_1double_chem"] = {
            "cnots": g1["cnots"], "err_mHa": g1["error_vs_fci"] * 1e3,
            "n_gates": g1["n_gates"], "source": "computed",
        }
        print(f"  greedy 1-double -> {g1['cnots']} CNOTs", flush=True)

    adapt = adapt_vqe(cfg, verbose=False)
    ar = adapt_to_raw_records(adapt["actions"], adapt["params"])
    rows["adapt_start"] = {"cnots": count_cnots(ar), "n_gates": len(ar)}

    if run_adapt_greedy and "greedy_adapt_chem" not in rows:
        print("  greedy ADAPT (chem), slow, ~25-40 min...", flush=True)
        gp = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], ar, fci,
                              chem_acc=CHEM, extra_restarts=0, maxiter=50, verbose=True)
        rows["greedy_adapt_chem"] = {
            "cnots": gp["cnots"], "err_mHa": gp["error_vs_fci"] * 1e3,
            "n_gates": gp["n_gates"], "source": "computed",
        }

    if run_exact_greedy:
        print("  greedy ADAPT (exact), slow...", flush=True)
        gex = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], ar, fci,
                               chem_acc=EXACT, extra_restarts=0, maxiter=50, verbose=True)
        rows["greedy_adapt_exact"] = {
            "cnots": gex["cnots"], "err_mHa": gex["error_vs_fci"] * 1e3,
            "within_exact": gex["error_vs_fci"] < EXACT, "source": "computed",
        }

    print("Baselines:", json.dumps(rows, indent=2), flush=True)
    return rows, ar, od


def build_fair_comparison(baselines, rl_runs):
    # same start circuit, greedy vs RL prune
    by_pair = {r["pair_id"]: r for r in rl_runs if "pair_id" in r}
    rows = []
    for pair_id, meta in sorted(FAIR_PAIRS.items(), key=lambda x: x[1]["priority"]):
        if pair_id not in by_pair:
            continue  # only pairs actually run in this job
        gkey = meta["greedy_key"]
        if gkey not in baselines:
            continue
        g = baselines[gkey]
        rl = by_pair.get(pair_id)
        rl_cnots = rl.get("cnots") if rl else None
        rl_err = rl.get("err_mHa") if rl else None
        rl_gap = rl.get("error_vs_fci") if rl else None
        goal = meta.get("goal_cnots")
        on_target = rl_gap is not None and rl_gap < meta["target"]
        rows.append({
            "pair": pair_id,
            "start": meta["label"],
            "target": "chem_acc" if meta["target"] == CHEM else "exact_FCI",
            "greedy_cnots": g["cnots"],
            "greedy_err_mHa": g.get("err_mHa"),
            "rl_cnots": rl_cnots,
            "rl_err_mHa": rl_err,
            "rl_beats_greedy_cnots": on_target and rl_cnots is not None and rl_cnots < g["cnots"],
            "rl_ties_greedy": on_target and rl_cnots == g["cnots"],
            "meets_goal": on_target and goal is not None and rl_cnots is not None and rl_cnots <= goal,
            "goal_cnots": goal,
        })
    return rows


def run_prune_pair(cfg, start_records, pair_id, baselines, preset, seed=0):
    meta = FAIR_PAIRS[pair_id]
    kw = preset.as_train_kw()
    n_upd = preset.updates
    if pair_id.startswith("adapt") and preset.updates_adapt is not None:
        n_upd = preset.updates_adapt

    gkey = meta["greedy_key"]
    gb = baselines[gkey]
    greedy_bl = {"cnots": gb["cnots"], "error_vs_fci": gb["err_mHa"] / 1000}

    suffix = "chem" if meta["target"] == CHEM else "exact"
    print(f"\n=== matched pair: {meta['label']} | greedy {gb['cnots']} CNOTs "
          f"-> RL prune ({preset.name}, {n_upd} upd) ===", flush=True)

    res, _, _ = train_prune_campaign(
        cfg, start_records,
        target=meta["target"],
        label=f"RL_prune_{pair_id}_s{seed}",
        seed=seed,
        num_updates=n_upd,
        episodes_per_update=kw["episodes_per_update"],
        hidden=kw["hidden"],
        order_k=kw["order_k"],
        inner_maxiter=kw["inner_maxiter"],
        inner_restarts=kw["inner_restarts"],
        bc_epochs=(
            50 if preset.name == "modal_heavy_1double"
            else (25 if preset.name != "smoke" else 5)
        ),
        use_greedy_bc=getattr(preset, "use_greedy_bc", True),
        greedy_baseline=greedy_bl,
        log_every=1 if preset.name == "smoke" else 5,
        out_dir=OUT,
    )
    res["pair_id"] = pair_id
    res["fair_start"] = meta["label"]
    res["greedy_comparison_cnots"] = gb["cnots"]
    res["goal_cnots"] = meta.get("goal_cnots")
    res["meets_goal"] = (
        res.get("cnots") is not None
        and meta.get("goal_cnots") is not None
        and res["cnots"] <= meta["goal_cnots"]
        and res.get("error_vs_fci", np.inf) < meta["target"]
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="smoke",
                    choices=["baselines", "focus", "prune", "smoke"])
    ap.add_argument("--preset", default="smoke", choices=list(PRESETS.keys()))
    ap.add_argument("--orbitals", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="fair pair ids (default: phase-dependent)")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-adapt-greedy", action="store_true",
                    help="use cached ADAPT greedy; saves ~30 min (Modal default)")
    ap.add_argument("--run-exact-greedy", action="store_true",
                    help="also run greedy ADAPT @ exact (slow, skip for 5h budget)")
    ap.add_argument("--out-dir", default="results/lih_campaign",
                    help="output directory (use per-seed subdirs for parallel Modal runs)")
    args = ap.parse_args()

    global OUT
    OUT = Path(args.out_dir)
    preset = PRESETS[args.preset]
    OUT.mkdir(parents=True, exist_ok=True)

    if args.phase == "focus":
        # ~5h Modal: primary 1-double + secondary ADAPT chem only
        if args.pairs is None:
            args.pairs = ["1double_chem", "adapt_chem"]
        if args.preset == "smoke":
            args.preset = "modal_5h"
            preset = PRESETS["modal_5h"]
        args.skip_adapt_greedy = args.skip_adapt_greedy or True

    elif args.phase == "smoke":
        if args.pairs is None:
            args.pairs = ["1double_chem", "adapt_chem"]

    elif args.pairs is None:
        args.pairs = ["1double_chem", "adapt_chem"]

    cfg = make_lih_config(active_electrons=2, active_orbitals=args.orbitals)
    goal_cnots = 6 if args.orbitals == 3 else TARGET_CHEM_CNOTS
    report = {
        "preset": preset.name,
        "phase": args.phase,
        "orbitals": args.orbitals,
        "goal_cnots": goal_cnots,
        "pairs": args.pairs,
        "runs": [],
        "started": time.time(),
    }
    print(f"{cfg['name']}: HF {cfg['hf_energy']:.6f}  FCI {cfg['fci_energy']:.6f}")
    print(f"Primary goal: {TARGET_CHEM_CNOTS} CNOTs @ chem acc from 1-double compile\n", flush=True)

    baselines_path = OUT / "baselines.json"
    need_bl = args.phase in ("baselines", "focus", "prune", "smoke")
    if need_bl:
        if args.skip_baselines and baselines_path.exists():
            baselines = json.loads(baselines_path.read_text())["rows"]
            adapt = adapt_vqe(cfg, verbose=False)
            ar = adapt_to_raw_records(adapt["actions"], adapt["params"])
            od = one_double_records(cfg)
        else:
            baselines, ar, od = run_baselines(
                cfg,
                run_adapt_greedy=not args.skip_adapt_greedy,
                run_exact_greedy=args.run_exact_greedy,
            )
            baselines_path.write_text(json.dumps({"rows": baselines}, indent=2))
        report["baselines"] = baselines
        if args.phase == "baselines":
            (OUT / "campaign_report.json").write_text(json.dumps(report, indent=2))
            return
    else:
        ar = od = None

    starts = {
        "1double_chem": od,
        "adapt_chem": ar,
        "adapt_exact": ar,
    }

    rl_runs = []
    for pair_id in args.pairs:
        if pair_id not in FAIR_PAIRS:
            print(f"skip unknown pair {pair_id}", flush=True)
            continue
        if pair_id == "adapt_exact" and not args.run_exact_greedy:
            if "greedy_adapt_exact" not in baselines:
                print("skip adapt_exact (no greedy exact baseline; use --run-exact-greedy)", flush=True)
                continue
        res = run_prune_pair(cfg, starts[pair_id], pair_id, baselines, preset, seed=args.seed)
        if pair_id == "1double_chem" and args.orbitals == 3:
            res["goal_cnots"] = goal_cnots
            res["meets_goal"] = (
                res.get("cnots") is not None
                and res["cnots"] <= goal_cnots
                and res.get("error_vs_fci", np.inf) < CHEM
            )
        rl_runs.append(res)
        report["runs"].append(res)

    fair = build_fair_comparison(baselines, rl_runs)
    report["fair_comparison"] = fair
    report["finished"] = time.time()
    report["elapsed_hours"] = (report["finished"] - report["started"]) / 3600

    (OUT / "campaign_report.json").write_text(json.dumps(report, indent=2, default=str))
    (OUT / "fair_comparison.json").write_text(json.dumps(fair, indent=2))

    print("\n=== matched comparison (same start, greedy vs RL) ===", flush=True)
    for row in fair:
        print(f"  {row['start']:22s} | greedy {row['greedy_cnots']:2d} | "
              f"RL {row['rl_cnots']} | beats_greedy={row['rl_beats_greedy_cnots']} | "
              f"goal {row['goal_cnots']} met={row['meets_goal']}", flush=True)
    print(f"\nSaved {OUT / 'fair_comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
