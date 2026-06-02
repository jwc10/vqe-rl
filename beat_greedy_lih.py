#!/usr/bin/env python3
# try to beat greedy on LiH: fewer CNOTs and/or lower error vs FCI

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from adapt import adapt_vqe
from prune_env import train_prune, RawGatePruneEnv
from raw_prune import adapt_to_raw_records, count_cnots, greedy_raw_prune, circuit_gates_to_raw_records
from rl_env import build_action_space, make_lih_config
from train_raw import train
from vqe_core import run_vqe_on_circuit

CHEM = 1.6e-3
EXACT = 1e-6
OUT = Path("results/beat_greedy")


def best_double(cfg):
    best = None
    for a in build_action_space(cfg["n_electrons"], cfg["num_qubits"]):
        if a["type"] != "double":
            continue
        vqe = run_vqe_on_circuit(cfg["H"], cfg["num_qubits"], cfg["hf_state"], [a])
        err = vqe["energy"] - cfg["fci_energy"]
        if best is None or err < best[1]:
            best = (a, err, vqe)
    if best is None or best[1] >= CHEM:
        raise RuntimeError("no chem-acc double")
    a, err, vqe = best
    gate = {**a, "theta": float(vqe["params"][0])}
    recs = circuit_gates_to_raw_records([gate])
    return recs, err, gate


def eval_records(cfg, records):
    from raw_prune import _optimized_energy
    e = _optimized_energy(cfg["H"], cfg["num_qubits"], cfg["hf_state"], records,
                          extra_restarts=1, maxiter=80)
    gap = e - cfg["fci_energy"]
    return {
        "energy": e,
        "error_vs_fci": gap,
        "err_mHa": gap * 1000,
        "cnots": count_cnots(records),
        "n_gates": len(records),
        "within_chem_acc": gap < CHEM,
        "within_exact": gap < EXACT,
    }


def train_prune_tracked(cfg, start_records, label, **kw):
    from train_raw import ActorCritic, discounted_returns
    target = kw.get("target", CHEM)
    num_updates = kw.get("num_updates", 40)
    episodes = kw.get("episodes_per_update", 16)
    seed = kw.get("seed", 0)
    inner_maxiter = kw.get("inner_maxiter", 60)
    order_k = kw.get("order_k", 3)
    hidden = kw.get("hidden", 256)

    torch.manual_seed(seed)
    env = RawGatePruneEnv(cfg, start_records, target=target, strict=False, order_k=order_k,
                          inner_restarts=0, inner_maxiter=inner_maxiter)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ActorCritic(env.state_dim, env.n_actions, hidden=hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    best = {"cnots": env.start_cnots, "error_vs_fci": np.inf, "label": label}
    best_recs = deepcopy(start_records)
    log = []
    start_ev = eval_records(cfg, start_records)
    if start_ev["error_vs_fci"] < target:
        best = {**start_ev, "label": label, "update": 0}
        best_recs = deepcopy(start_records)
    t_run = time.perf_counter()
    tgt_name = "exact" if target <= EXACT else "chem"
    print(f"[{label}] start {len(start_records)} gates / {env.start_cnots} CNOTs | "
          f"target={tgt_name} ({target:.1e} Ha) | {episodes} episodes/update",
          flush=True)

    for upd in range(num_updates):
        t_upd = time.perf_counter()
        S, M, A, R = [], [], [], []
        for _ in range(episodes):
            state, _ = env.reset()
            traj = []
            while True:
                mask = env.valid_action_mask()
                s = torch.tensor(state, dtype=torch.float32, device=dev)
                m = torch.tensor(mask, dtype=torch.bool, device=dev)
                with torch.no_grad():
                    logits, _ = net(s, m)
                    a = torch.distributions.Categorical(logits=logits).sample()
                state, r, done, info = env.step_gym(int(a))
                traj.append({"s": s, "m": m, "a": a, "r": r})
                if done:
                    break
            for t, ret in zip(traj, discounted_returns([t["r"] for t in traj], 0.97)):
                S.append(t["s"]); M.append(t["m"]); A.append(t["a"]); R.append(ret)
            recs = env._alive()
            ev = eval_records(cfg, recs)
            ok = ev["error_vs_fci"] < target
            if ok or ev["error_vs_fci"] < best.get("error_vs_fci", np.inf):
                improved = ok and (
                    ev["cnots"] < best["cnots"]
                    or (ev["cnots"] == best["cnots"] and ev["error_vs_fci"] < best["error_vs_fci"])
                )
                if not ok and ev["error_vs_fci"] < best.get("error_vs_fci", np.inf):
                    improved = True
                if improved:
                    best = {**ev, "label": label, "update": upd + 1}
                    best_recs = deepcopy(recs)

        if S:
            S = torch.stack(S); M = torch.stack(M); A = torch.stack(A)
            R = torch.tensor(R, dtype=torch.float32, device=dev)
            logits, vals = net(S, M)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(A)
            adv = (R - vals).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            loss = -(logp * adv).mean() + 0.5 * torch.nn.functional.mse_loss(vals, R)
            opt.zero_grad(); loss.backward(); opt.step()

        log.append({"update": upd + 1, "best_cnots": best["cnots"],
                    "best_err_mHa": best["error_vs_fci"] * 1000})
        err_mha = best["error_vs_fci"] * 1000
        err_s = f"{err_mha:.4f}" if np.isfinite(err_mha) else "inf"
        print(f"[{label}] upd {upd+1}/{num_updates} | best {best['cnots']} CNOTs "
              f"| {err_s} mHa | exact={best.get('within_exact', False)} | "
              f"upd {time.perf_counter() - t_upd:.1f}s | total {time.perf_counter() - t_run:.0f}s",
              flush=True)

    return best, best_recs, log


def random_prune_search(cfg, start_records, n_trials=80, target=CHEM):
    fci = cfg["fci_energy"]
    n = len(start_records)
    best = None
    rng = np.random.default_rng(0)
    for _ in range(n_trials):
        # random subset: keep each gate with p=0.5, at least 3 gates
        keep = rng.random(n) > 0.45
        if keep.sum() < 3:
            continue
        trial = [deepcopy(start_records[i]) for i in range(n) if keep[i]]
        gp = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], trial, fci,
                              chem_acc=target, extra_restarts=0, maxiter=40)
        if gp["error_vs_fci"] < target:
            rec = {"cnots": gp["cnots"], "error_vs_fci": gp["error_vs_fci"],
                   "n_gates": gp["n_gates"], "method": "random_start+greedy_finish"}
            if best is None or rec["cnots"] < best["cnots"] or (
                rec["cnots"] == best["cnots"] and rec["error_vs_fci"] < best["error_vs_fci"]
            ):
                best = rec
    return best


def beats_greedy(rl, greedy7, greedy_adapt):
    wins = []
    if rl["cnots"] < greedy7["cnots"]:
        wins.append("fewer CNOTs than greedy-1double")
    if rl["error_vs_fci"] < greedy7["error_vs_fci"] - 1e-5:
        wins.append("lower error than greedy-1double (meaningful)")
    if rl.get("within_exact") and rl["cnots"] < 46:
        wins.append("exact FCI with fewer CNOTs than full ADAPT")
    if rl["cnots"] < greedy_adapt["cnots"]:
        wins.append("fewer CNOTs than greedy-ADAPT")
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["baselines", "prune1", "prune_adapt", "adapt2", "hybrid", "random", "all"])
    ap.add_argument("--updates", type=int, default=35)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 42])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = make_lih_config(2, 4)
    fci = cfg["fci_energy"]
    report = {"fci": fci, "runs": [], "greedy_baselines": {}}

    one_double_recs, _, _ = best_double(cfg)
    print(f"1-double: {len(one_double_recs)} gates, {count_cnots(one_double_recs)} CNOTs", flush=True)

    if args.phase in ("baselines", "all"):
        fair = Path("results/fair_greedy_compare.json")
        if fair.exists():
            for r in json.loads(fair.read_text())["LiH(2e,4o)"]:
                if "1-double" in r["method"] and "no prune" in r["method"]:
                    report["greedy_baselines"]["greedy_1double_start"] = r
                if "FROM 1-double" in r["method"]:
                    report["greedy_baselines"]["greedy_1double"] = {
                        "cnots": r["final_cnots"], "err_mHa": r["err_mHa"], "n_gates": r["n_gates"]}
                if "FROM ADAPT" in r["method"]:
                    report["greedy_baselines"]["greedy_adapt"] = {
                        "cnots": r["final_cnots"], "err_mHa": r["err_mHa"]}
        if "greedy_1double" not in report["greedy_baselines"]:
            g7 = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], one_double_recs, fci,
                                  chem_acc=CHEM, extra_restarts=0, maxiter=40)
            report["greedy_baselines"]["greedy_1double"] = {
                "cnots": g7["cnots"], "err_mHa": g7["error_vs_fci"] * 1000, "n_gates": g7["n_gates"]}
        if "greedy_adapt" not in report["greedy_baselines"]:
            adapt = adapt_vqe(cfg, verbose=False)
            ar = adapt_to_raw_records(adapt["actions"], adapt["params"])
            ga = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], ar, fci,
                                  chem_acc=CHEM, extra_restarts=0, maxiter=40)
            report["greedy_baselines"]["greedy_adapt"] = {
                "cnots": ga["cnots"], "err_mHa": ga["error_vs_fci"] * 1000}
            report["greedy_baselines"]["adapt_full"] = {
                "cnots": count_cnots(ar), "err_mHa": (adapt["energy"] - fci) * 1000}
        print("Greedy baselines:", json.dumps(report["greedy_baselines"], indent=2), flush=True)

    g7b = report["greedy_baselines"].get("greedy_1double", {"cnots": 7, "err_mHa": 1.468})
    gab = report["greedy_baselines"].get("greedy_adapt", {"cnots": 23, "err_mHa": 1.468})
    g7 = {"cnots": g7b["cnots"], "error_vs_fci": g7b["err_mHa"] / 1000}
    ga = {"cnots": gab["cnots"], "error_vs_fci": gab["err_mHa"] / 1000}

    if args.phase in ("random", "all"):
        rb = random_prune_search(cfg, one_double_recs, n_trials=100)
        if rb:
            report["runs"].append(rb)
            print("Random+greedy search best:", rb, flush=True)

    if "greedy_1double" not in report["greedy_baselines"]:
        fair = Path("results/fair_greedy_compare.json")
        if fair.exists():
            for r in json.loads(fair.read_text())["LiH(2e,4o)"]:
                if "FROM 1-double" in r["method"]:
                    report["greedy_baselines"]["greedy_1double"] = {
                        "cnots": r["final_cnots"], "err_mHa": r["err_mHa"]}
                if "FROM ADAPT" in r["method"]:
                    report["greedy_baselines"]["greedy_adapt"] = {
                        "cnots": r["final_cnots"], "err_mHa": r["err_mHa"]}

    if args.phase in ("prune1", "all"):
        for seed in args.seeds:
            for tgt, name in [(CHEM, "chem"), (EXACT, "exact")]:
                t0 = time.perf_counter()
                best, recs, log = train_prune_tracked(
                    cfg, one_double_recs, f"RL_prune_1double_{name}_s{seed}",
                    target=tgt, num_updates=args.updates, seed=seed,
                    episodes_per_update=16, inner_maxiter=70)
                wins = beats_greedy(best, g7, ga)
                entry = {**best, "seed": seed, "target": name, "wins_vs_greedy": wins,
                         "seconds": time.perf_counter() - t0}
                report["runs"].append(entry)
                if recs:
                    with open(OUT / f"best_1double_{name}_s{seed}.json", "w") as f:
                        json.dump({"records": recs, **best}, f, indent=2)
                if wins:
                    print("*** WIN:", wins, entry, flush=True)

    if args.phase in ("adapt2", "all"):
        adapt2 = adapt_vqe(cfg, max_iter=2, verbose=False)
        ar2 = adapt_to_raw_records(adapt2["actions"], adapt2["params"])
        print(f"ADAPT-2 compile: {len(ar2)} gates, {count_cnots(ar2)} CNOTs, "
              f"err={(adapt2['energy']-fci)*1000:.4f} mHa", flush=True)
        for seed in [0]:
            for tgt, name in [(EXACT, "exact"), (CHEM, "chem")]:
                best, recs, _ = train_prune_tracked(
                    cfg, ar2, f"RL_prune_ADAPT2_{name}_s{seed}",
                    target=tgt, num_updates=35, seed=seed, episodes_per_update=14,
                    inner_maxiter=60)
                wins = beats_greedy(best, g7, ga)
                entry = {**best, "seed": seed, "wins_vs_greedy": wins}
                report["runs"].append(entry)
                if wins:
                    print("*** WIN:", wins, entry, flush=True)

    if args.phase in ("prune_adapt", "all"):
        adapt = adapt_vqe(cfg, verbose=False)
        ar = adapt_to_raw_records(adapt["actions"], adapt["params"])
        print(f"ADAPT compile: {len(ar)} gates, {count_cnots(ar)} CNOTs", flush=True)
        for tgt, gname in [(CHEM, "chem_1.6mHa"), (EXACT, "exact_1e-6_Ha")]:
            print(f"\n=== Greedy raw-prune from ADAPT ({gname}) ===", flush=True)
            t0 = time.perf_counter()
            gp = greedy_raw_prune(cfg["H"], cfg["num_qubits"], cfg["hf_state"], ar, fci,
                                  chem_acc=tgt, extra_restarts=0, maxiter=50, verbose=True)
            row = {"cnots": gp["cnots"], "err_mHa": gp["error_vs_fci"] * 1000,
                   "n_gates": gp["n_gates"], "n_evals": gp["n_evals"],
                   "within_exact": gp["error_vs_fci"] < EXACT,
                   "seconds": time.perf_counter() - t0}
            report["greedy_baselines"][f"greedy_adapt_{gname}"] = row
            print(f"Greedy ADAPT {gname}: {row['cnots']} CNOTs, {row['err_mHa']:.6f} mHa, "
                  f"exact={row['within_exact']} ({row['seconds']:.1f}s)", flush=True)
        for seed in [0, 42]:
            for tgt, name in [(EXACT, "exact"), (CHEM, "chem")]:
                t0 = time.perf_counter()
                best, recs, log = train_prune_tracked(
                    cfg, ar, f"RL_prune_ADAPT_{name}_s{seed}",
                    target=tgt, num_updates=min(args.updates, 50), seed=seed,
                    episodes_per_update=12, inner_maxiter=50)
                wins = beats_greedy(best, g7, ga)
                entry = {**best, "seed": seed, "target": name, "wins_vs_greedy": wins,
                         "seconds": time.perf_counter() - t0}
                report["runs"].append(entry)
                if wins:
                    print("*** WIN:", wins, entry, flush=True)

    if args.phase in ("hybrid", "all"):
        for seed in args.seeds:
            _, _, _, bests = train(
                cfg, num_updates=20, episodes_per_update=8, gate_set="hybrid",
                hybrid_mode="simul_prune", moving_threshold=True, target_mode="chem",
                seed=seed, max_gates=12, order_k=4, inner_maxiter=60, inner_restarts=1,
                hidden=256, log_every=1,
                hybrid_preset={"tier": "beat", "max_compile_gates": 40,
                               "prune_max_steps": 25, "prune_order_k": 3},
            )
            bc = bests["chem"]
            if np.isfinite(bc.get("cnots", np.inf)):
                entry = {
                    "label": f"simul_prune_s{seed}",
                    "cnots": int(bc["cnots"]),
                    "error_vs_fci": bc["error_vs_fci"],
                    "err_mHa": bc["error_vs_fci"] * 1000,
                    "n_gates": bc["n_gates"],
                    "within_chem_acc": bc["error_vs_fci"] < CHEM,
                    "within_exact": bc["error_vs_fci"] < EXACT,
                }
                wins = beats_greedy(entry, g7, ga)
                entry["wins_vs_greedy"] = wins
                report["runs"].append(entry)
                if wins:
                    print("*** WIN:", wins, entry, flush=True)

    out_path = OUT / "hunt_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved {out_path}")
    any_win = [r for r in report["runs"] if r.get("wins_vs_greedy")]
    if any_win:
        print(f"\n{len(any_win)} run(s) beat greedy on some metric:")
        for r in any_win:
            print(f"  {r.get('label', r)}: {r['wins_vs_greedy']}")
    else:
        print("\nNo run beat greedy yet on CNOTs or error.")


if __name__ == "__main__":
    main()
