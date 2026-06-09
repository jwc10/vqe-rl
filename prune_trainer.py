# Enhanced RL prune trainer: BC warm-start from greedy traces + PPO fine-tune.

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prune_env import RawGatePruneEnv
from raw_prune import count_cnots, greedy_raw_prune
from train_raw import ActorCritic, discounted_returns, ppo_step, pick_device
from greedy_trace import greedy_prune_with_trace

CHEM = 1.6e-3
EXACT = 1e-6


def eval_records(cfg, records, inner_maxiter=80):
    from raw_prune import _optimized_energy
    e = _optimized_energy(cfg["H"], cfg["num_qubits"], cfg["hf_state"], records,
                          extra_restarts=1, maxiter=inner_maxiter)
    gap = e - cfg["fci_energy"]
    return {
        "energy": e,
        "error_vs_fci": gap,
        "cnots": count_cnots(records),
        "n_gates": len(records),
        "within_chem_acc": gap < CHEM,
        "within_exact": gap < EXACT,
    }


def bc_pretrain(net, trace, device, epochs=15, lr=1e-3, batch_size=64):
    if not trace:
        return 0.0
    obs = torch.tensor(np.stack([t["obs"] for t in trace]), dtype=torch.float32, device=device)
    acts = torch.tensor([t["action"] for t in trace], dtype=torch.long, device=device)
    masks = torch.tensor(np.stack([t["mask"] for t in trace]), dtype=torch.bool, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(trace)
    last_loss = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            ix = perm[start:start + batch_size]
            logits, _ = net(obs[ix], masks[ix])
            loss = F.cross_entropy(logits, acts[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
    return last_loss


def train_prune_campaign(
    cfg,
    start_records,
    *,
    target=CHEM,
    label="RL_prune",
    seed=0,
    num_updates=150,
    episodes_per_update=20,
    hidden=512,
    order_k=4,
    inner_maxiter=60,
    inner_restarts=0,
    gamma=0.97,
    lr=3e-3,
    entropy_coef=0.02,
    ppo_epochs=4,
    clip=0.2,
    bc_epochs=20,
    use_greedy_bc=True,
    greedy_baseline=None,
    log_every=1,
    out_dir=None,
):
    # greedy trace seeds behavioral cloning, then PPO explores non-greedy removal orders
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device()
    fci = cfg["fci_energy"]

    env = RawGatePruneEnv(
        cfg, start_records, target=target, strict=False, order_k=order_k,
        inner_restarts=inner_restarts, inner_maxiter=inner_maxiter,
    )
    net = ActorCritic(env.state_dim, env.n_actions, hidden=hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    bc_loss = None
    if use_greedy_bc:
        print(f"[{label}] collecting greedy trace for BC...", flush=True)
        tr = greedy_prune_with_trace(
            cfg, start_records, fci,
            chem_acc=target, order_k=order_k, inner_maxiter=inner_maxiter,
            inner_restarts=inner_restarts, verbose=log_every > 0,
        )
        bc_loss = bc_pretrain(net, tr["trace"], dev, epochs=bc_epochs)
        print(f"[{label}] BC done: {len(tr['trace'])} steps, loss={bc_loss:.4f}, "
              f"greedy trace reached {tr['final'].get('cnots')} CNOTs", flush=True)

    if greedy_baseline is None:
        gp = greedy_raw_prune(
            cfg["H"], cfg["num_qubits"], cfg["hf_state"], start_records, fci,
            chem_acc=target, extra_restarts=0, maxiter=inner_maxiter,
        )
        greedy_baseline = {"cnots": gp["cnots"], "error_vs_fci": gp["error_vs_fci"]}

    start_ev = eval_records(cfg, start_records, inner_maxiter)
    best = {**start_ev, "label": label, "update": 0}
    best_recs = deepcopy(start_records)

    log = []
    t0 = time.perf_counter()
    print(f"[{label}] PPO: {num_updates} updates x {episodes_per_update} ep | "
          f"target={target:.1e} | greedy floor {greedy_baseline['cnots']} CNOTs | "
          f"device={dev}", flush=True)

    for upd in range(num_updates):
        t_upd = time.perf_counter()
        S, M, A, R = [], [], [], []
        for _ in range(episodes_per_update):
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
            for t, ret in zip(traj, discounted_returns([t["r"] for t in traj], gamma)):
                S.append(t["s"]); M.append(t["m"]); A.append(t["a"]); R.append(ret)
            recs = env._alive()
            ev = eval_records(cfg, recs, inner_maxiter)
            ok = ev["error_vs_fci"] < target
            if ok and (ev["cnots"] < best["cnots"] or
                       (ev["cnots"] == best["cnots"] and ev["error_vs_fci"] < best["error_vs_fci"])):
                best = {**ev, "label": label, "update": upd + 1}
                best_recs = deepcopy(recs)
            elif not ok and ev["error_vs_fci"] < best.get("error_vs_fci", np.inf):
                best = {**ev, "label": label, "update": upd + 1}
                best_recs = deepcopy(recs)

        if len(S) >= 2:
            ppo_step(
                net, opt,
                torch.stack(S), torch.stack(M), torch.stack(A),
                torch.tensor(R, dtype=torch.float32, device=dev),
                "ppo", ppo_epochs, clip, entropy_coef,
            )

        beats = (
            best.get("error_vs_fci", np.inf) < target
            and best["cnots"] < greedy_baseline["cnots"]
        )
        row = {
            "update": upd + 1,
            "best_cnots": best["cnots"],
            "best_err_mHa": best["error_vs_fci"] * 1000,
            "beats_greedy": beats,
            "upd_sec": time.perf_counter() - t_upd,
        }
        log.append(row)
        if log_every and (upd + 1) % log_every == 0:
            err = best["error_vs_fci"] * 1000
            print(
                f"[{label}] upd {upd+1}/{num_updates} | best {best['cnots']} CNOTs | "
                f"{err:.4f} mHa | beats_greedy={beats} | "
                f"upd {row['upd_sec']:.1f}s | total {time.perf_counter()-t0:.0f}s",
                flush=True,
            )

    result = {
        **best,
        "err_mHa": best.get("error_vs_fci", np.inf) * 1000,
        "greedy_baseline_cnots": greedy_baseline["cnots"],
        "greedy_baseline_err_mHa": greedy_baseline["error_vs_fci"] * 1000,
        "beats_greedy": (
            best.get("error_vs_fci", np.inf) < target
            and best["cnots"] < greedy_baseline["cnots"]
        ),
        "bc_loss": bc_loss,
        "target": target,
        "seed": seed,
        "seconds": time.perf_counter() - t0,
        "log": log,
    }
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = label.replace("/", "_")
        (out_dir / f"{safe}_result.json").write_text(json.dumps(result, indent=2))
        (out_dir / f"{safe}_records.json").write_text(
            json.dumps({"records": best_recs}, indent=2))
    return result, best_recs, net
