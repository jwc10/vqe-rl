# PPO on raw-gate / Givens / hybrid VQE envs

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from raw_gate_env import RawGateEnv
from curriculum import MovingThreshold
from raw_prune import circuit_gates_to_raw_records
from prune_env import RawGatePruneEnv

MASK_OFF = -1e9
MAX_PRUNE_GATES = 128


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, s, mask):
        h = self.net(s)
        logits = torch.where(mask, self.policy(h), torch.full_like(self.policy(h), MASK_OFF))
        return logits, self.value(h).squeeze(-1)


def discounted_returns(rewards, gamma):
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return list(reversed(out))


def run_episode(env, net, rng, device):
    state, _ = env.reset()
    traj = []
    while True:
        mask = env.valid_action_mask()
        s = torch.tensor(state, dtype=torch.float32, device=device)
        m = torch.tensor(mask, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, val = net(s, m)
            a = torch.distributions.Categorical(logits=logits).sample()
        state, r, done, info = env.step_gym(int(a))
        traj.append({"s": s, "m": m, "a": a, "r": r, "v": float(val)})
        if done:
            return traj, info


def run_hybrid_episode(cfg, net_build, net_prune, rng, device, env_opts, target,
                       simul_build=False):
    skip = ("prune_order_k", "prune_max_steps", "tier", "max_compile_gates")
    kw = {k: v for k, v in env_opts.items() if k not in skip}
    if simul_build:
        build = RawGateEnv(cfg, gate_set="hybrid", hybrid_simultaneous=True, **kw)
    else:
        build = RawGateEnv(cfg, gate_set="givens", **kw)
    chem_floor = build.chem_acc
    build.set_target(target, floor=chem_floor)
    traj_b, info_b = run_episode(build, net_build, rng, device)
    if not info_b.get("within_chem_acc"):
        info_b["hybrid_phase"] = "build_failed"
        return traj_b, [], info_b

    raw_list = circuit_gates_to_raw_records(info_b["gates"])
    if len(raw_list) > MAX_PRUNE_GATES:
        info_b["hybrid_phase"] = "prune_too_large"
        return traj_b, [], info_b
    if len(raw_list) > env_opts.get("max_compile_gates", 48):
        info_b["hybrid_phase"] = "prune_skipped_large_compile"
        info_b["compiled_gates"] = len(raw_list)
        return traj_b, [], info_b

    vqe_kw = {k: env_opts[k] for k in ("inner_restarts", "inner_maxiter") if k in env_opts}
    prune = RawGatePruneEnv(
        cfg, raw_list, target=max(target, chem_floor), strict=False,
        order_k=env_opts.get("prune_order_k", 3), max_n=MAX_PRUNE_GATES,
        max_steps=env_opts.get("prune_max_steps", 0), **vqe_kw)
    state, _ = prune.reset()
    traj_p = []
    info = info_b
    while True:
        mask = prune.valid_action_mask()
        s = torch.tensor(state, dtype=torch.float32, device=device)
        m = torch.tensor(mask, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, val = net_prune(s, m)
            a = torch.distributions.Categorical(logits=logits).sample()
        state, r, done, info = prune.step_gym(int(a))
        traj_p.append({"s": s, "m": m, "a": a, "r": r, "v": float(val)})
        if done:
            break
    info["hybrid_phase"] = "simul+prune" if simul_build else "build+prune"
    info["build_cnots"] = info_b["cnots"]
    info["compiled_gates"] = len(raw_list)
    return traj_b, traj_p, info


def ppo_step(net, opt, S, M, A, R, algo, n_epochs, clip, ent_coef):
    with torch.no_grad():
        old_logits, _ = net(S, M)
        old_logp = torch.distributions.Categorical(logits=old_logits).log_prob(A)
    epochs = n_epochs if algo == "ppo" else 1
    for _ in range(epochs):
        logits, vals = net(S, M)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(A)
        adv = (R - vals).detach()
        if adv.numel() >= 2:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ent = dist.entropy().mean()
        v_loss = F.mse_loss(vals, R)
        if algo == "ppo":
            ratio = torch.exp(logp - old_logp)
            p_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
        else:
            p_loss = -(logp * adv).mean()
        loss = p_loss + 0.5 * v_loss - ent_coef * ent
        opt.zero_grad(); loss.backward(); opt.step()


def anneal_target(upd, n_upd, start, floor, frac=0.6):
    n = max(1, int(frac * n_upd))
    if upd >= n:
        return floor
    return start * (floor / start) ** (upd / n)


def sizing_prune_env(cfg, target):
    # dummy env so prune net has the right action count
    from rl_env import build_action_space
    from vqe_core import run_vqe_on_circuit
    pool = [a for a in build_action_space(cfg["n_electrons"], cfg["num_qubits"])
            if a["type"] == "double"]
    for a in pool:
        vqe = run_vqe_on_circuit(cfg["H"], cfg["num_qubits"], cfg["hf_state"], [a])
        if vqe["energy"] - cfg["fci_energy"] < target:
            recs = circuit_gates_to_raw_records([a])
            return RawGatePruneEnv(cfg, recs, target=target, strict=False, order_k=3,
                                   max_n=MAX_PRUNE_GATES)
    recs = [{"name": "RX", "wires": (0,), "param": 0.0}]
    return RawGatePruneEnv(cfg, recs, target=target, strict=False, order_k=3, max_n=MAX_PRUNE_GATES)


def train_hybrid(cfg, n_upd, eps_per_upd, gamma, lr, max_gates, cnot_pen, ent_coef, seed,
                 log_every, target_mode, num_pen, order_k, inner_restarts, inner_maxiter,
                 algo, ppo_epochs, clip, hidden, reward_mode, max_givens, moving_thr,
                 init_target, simul_build=False, preset=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device()
    env_opts = dict(max_gates=max_gates, cnot_penalty=cnot_pen, number_penalty=num_pen,
                    order_k=order_k, inner_restarts=inner_restarts, inner_maxiter=inner_maxiter,
                    reward_mode=reward_mode, max_givens_phase=max_givens,
                    prune_max_steps=0, prune_order_k=3, tier="", max_compile_gates=48)
    if preset:
        for k in ("prune_max_steps", "prune_order_k", "tier", "max_compile_gates"):
            if k in preset:
                env_opts[k] = preset[k]
    skip = ("prune_order_k", "prune_max_steps", "tier", "max_compile_gates")
    kw = {k: v for k, v in env_opts.items() if k not in skip}
    if simul_build:
        probe = RawGateEnv(cfg, gate_set="hybrid", hybrid_simultaneous=True, **kw)
    else:
        probe = RawGateEnv(cfg, gate_set="givens", **kw)
    floor = probe.exact_tol if target_mode == "exact" else probe.chem_acc
    start = max(init_target, 0.7 * (probe.hf_energy - probe.fci_energy))
    ref = sizing_prune_env(cfg, floor)
    net_b = ActorCritic(probe.state_dim, probe.n_actions, hidden=hidden).to(dev)
    net_p = ActorCritic(ref.state_dim, ref.n_actions, hidden=hidden).to(dev)
    opt_b = torch.optim.Adam(net_b.parameters(), lr=lr)
    opt_p = torch.optim.Adam(net_p.parameters(), lr=lr)
    use_mt = moving_thr and env_opts.get("tier") != "quick20"
    mt = MovingThreshold(initial=start, floor=floor) if use_mt else None
    target = start if moving_thr else floor

    hist = {"return": [], "best_cnots_chem": [], "best_cnots_exact": [], "target": []}
    best_chem = {"cnots": np.inf}
    best_exact = {"cnots": np.inf}

    for upd in range(n_upd):
        if use_mt and mt:
            mt.step_update(upd)
            target = mt.target
        elif env_opts.get("tier") == "quick20":
            target = floor
        hist["target"].append(target)
        Sb, Mb, Ab, Rb = [], [], [], []
        Sp, Mp, Ap, Rp = [], [], [], []
        ep_rets = []
        for ep in range(eps_per_upd):
            t_b, t_p, info = run_hybrid_episode(
                cfg, net_b, net_p, rng, dev, env_opts, target, simul_build=simul_build)
            ret = sum(t["r"] for t in t_b) + sum(t["r"] for t in t_p)
            ep_rets.append(ret)
            if log_every == 1 and cfg["num_qubits"] >= 8:
                print(f"  ep {ep+1}/{eps_per_upd} | return {ret:6.1f} | "
                      f"cnots {info.get('cnots','--')} | gates {info.get('n_gates','--')} | "
                      f"phase {info.get('hybrid_phase','?')}", flush=True)
            if t_b:
                dr = discounted_returns([t["r"] for t in t_b], gamma)
                for t, g in zip(t_b, dr):
                    Sb.append(t["s"]); Mb.append(t["m"]); Ab.append(t["a"]); Rb.append(g)
            if t_p:
                dr = discounted_returns([t["r"] for t in t_p], gamma)
                for t, g in zip(t_p, dr):
                    Sp.append(t["s"]); Mp.append(t["m"]); Ap.append(t["a"]); Rp.append(g)
            if moving_thr and mt:
                mt.note_episode(info.get("error_vs_fci", np.inf),
                                info.get("within_chem_acc", False) or info.get("within_exact", False))
            rec = {"cnots": info["cnots"], "energy": info["energy"],
                   "n_gates": info["n_gates"], "error_vs_fci": info["error_vs_fci"]}
            if info.get("within_chem_acc") and info["cnots"] < best_chem["cnots"]:
                best_chem = rec
            if info.get("within_exact") and info["cnots"] < best_exact["cnots"]:
                best_exact = rec

        if Sb:
            ppo_step(net_b, opt_b, torch.stack(Sb), torch.stack(Mb), torch.stack(Ab),
                     torch.tensor(Rb, dtype=torch.float32, device=dev),
                     algo, ppo_epochs, clip, ent_coef)
        if len(Sp) >= 2:
            ppo_step(net_p, opt_p, torch.stack(Sp), torch.stack(Mp), torch.stack(Ap),
                     torch.tensor(Rp, dtype=torch.float32, device=dev),
                     algo, ppo_epochs, clip, ent_coef)

        hist["return"].append(float(np.mean(ep_rets)))
        hist["best_cnots_chem"].append(best_chem["cnots"] if np.isfinite(best_chem["cnots"]) else np.nan)
        hist["best_cnots_exact"].append(best_exact["cnots"] if np.isfinite(best_exact["cnots"]) else np.nan)
        if log_every and (upd + 1) % log_every == 0:
            bc = best_chem["cnots"] if np.isfinite(best_chem["cnots"]) else "--"
            be = best_exact["cnots"] if np.isfinite(best_exact["cnots"]) else "--"
            print(f"update {upd+1:4d}/{n_upd} | avg return {np.mean(ep_rets):6.3f} | "
                  f"best CNOTs chem:{bc} exact:{be} | target {target:.1e}", flush=True)
    return net_b, probe, hist, {"chem": best_chem, "exact": best_exact}


def train(cfg, num_updates=80, episodes_per_update=16, gamma=0.95, lr=3e-3,
          max_gates=20, cnot_penalty=0.0, entropy_coef=0.02, seed=0, log_every=10,
          curriculum=False, target_mode="chem", number_penalty=0.0,
          order_k=0, inner_restarts=1, inner_maxiter=100,
          algo="ppo", ppo_epochs=4, clip=0.2, hidden=256, gate_set="raw",
          reward_mode="ostaszewski", max_givens_phase=6,
          moving_threshold=False, initial_target=0.005,
          hybrid_mode="chained", hybrid_simultaneous=False, hybrid_preset=None):
    if gate_set == "hybrid" and hybrid_mode in ("chained", "simul_prune"):
        return train_hybrid(
            cfg, num_updates, episodes_per_update, gamma, lr, max_gates,
            cnot_penalty, entropy_coef, seed, log_every, target_mode, number_penalty,
            order_k, inner_restarts, inner_maxiter, algo, ppo_epochs, clip, hidden,
            reward_mode, max_givens_phase, moving_threshold, initial_target,
            simul_build=(hybrid_mode == "simul_prune"), preset=hybrid_preset)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device()
    env = RawGateEnv(cfg, max_gates=max_gates, cnot_penalty=cnot_penalty,
                     number_penalty=number_penalty, order_k=order_k,
                     inner_restarts=inner_restarts, inner_maxiter=inner_maxiter,
                     gate_set=gate_set, reward_mode=reward_mode,
                     max_givens_phase=max_givens_phase,
                     hybrid_simultaneous=hybrid_simultaneous or hybrid_mode == "simultaneous")
    net = ActorCritic(env.state_dim, env.n_actions, hidden=hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    floor = env.exact_tol if target_mode == "exact" else env.chem_acc
    start_gap = max(initial_target, 0.7 * (env.hf_energy - env.fci_energy))
    if moving_threshold:
        env.set_target(start_gap, floor=floor)
        mt = MovingThreshold(initial=start_gap, floor=floor)
    else:
        mt = None
        env.set_target(start_gap if curriculum else floor, floor=floor)

    hist = {"return": [], "best_cnots_chem": [], "best_cnots_exact": [], "target": []}
    best_chem = {"cnots": np.inf}
    best_exact = {"cnots": np.inf}

    for upd in range(num_updates):
        if moving_threshold and mt:
            mt.step_update(upd)
            env.set_target(mt.target, floor=floor)
        elif curriculum:
            env.set_target(anneal_target(upd, num_updates, start_gap, floor), floor=floor)
        hist["target"].append(env.target)
        S, M, A, R = [], [], [], []
        ep_rets = []
        for _ in range(episodes_per_update):
            traj, info = run_episode(env, net, rng, dev)
            dr = discounted_returns([t["r"] for t in traj], gamma)
            ep_rets.append(sum(t["r"] for t in traj))
            for t, g in zip(traj, dr):
                S.append(t["s"]); M.append(t["m"]); A.append(t["a"]); R.append(g)
            if moving_threshold and mt:
                mt.note_episode(info.get("error_vs_fci", np.inf),
                                info.get("within_chem_acc", False) or info.get("within_exact", False))
            rec = {"cnots": info["cnots"], "energy": info["energy"],
                   "n_gates": info["n_gates"], "gates": info["gates"],
                   "error_vs_fci": info["error_vs_fci"]}
            hybrid_ok = (
                gate_set != "hybrid" or hybrid_simultaneous or hybrid_mode == "simultaneous"
                or any(g["type"] in ("RX", "RY", "RZ", "CNOT") for g in info.get("gates", []))
            )
            if hybrid_ok and info.get("within_chem_acc") and info["cnots"] < best_chem["cnots"]:
                best_chem = rec
            if hybrid_ok and info.get("within_exact") and info["cnots"] < best_exact["cnots"]:
                best_exact = rec

        S = torch.stack(S); M = torch.stack(M); A = torch.stack(A)
        R = torch.tensor(R, dtype=torch.float32, device=dev)
        ppo_step(net, opt, S, M, A, R, algo, ppo_epochs, clip, entropy_coef)

        hist["return"].append(float(np.mean(ep_rets)))
        hist["best_cnots_chem"].append(best_chem["cnots"] if np.isfinite(best_chem["cnots"]) else np.nan)
        hist["best_cnots_exact"].append(best_exact["cnots"] if np.isfinite(best_exact["cnots"]) else np.nan)
        if log_every and (upd + 1) % log_every == 0:
            bc = best_chem["cnots"] if np.isfinite(best_chem["cnots"]) else "--"
            be = best_exact["cnots"] if np.isfinite(best_exact["cnots"]) else "--"
            tgt = f" | target {env.target:.1e}" if (curriculum or moving_threshold) else ""
            print(f"update {upd+1:4d}/{num_updates} | avg return {np.mean(ep_rets):6.3f} | "
                  f"best CNOTs chem:{bc} exact:{be}{tgt}", flush=True)

    return net, env, hist, {"chem": best_chem, "exact": best_exact}


# alias for older imports
_device = pick_device
collect_episode = run_episode
collect_hybrid_episode = run_hybrid_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="H2")
    ap.add_argument("--updates", type=int, default=80)
    ap.add_argument("--max-gates", type=int, default=16)
    ap.add_argument("--cnot-penalty", type=float, default=0.0)
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--target", default="chem", choices=["chem", "exact"])
    ap.add_argument("--number-penalty", type=float, default=0.0)
    ap.add_argument("--order-k", type=int, default=0)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "reinforce"])
    ap.add_argument("--inner-restarts", type=int, default=1)
    ap.add_argument("--inner-maxiter", type=int, default=100)
    ap.add_argument("--gate-set", default="raw", choices=["raw", "givens", "hybrid"])
    ap.add_argument("--hybrid-mode", default="chained", choices=["chained", "simultaneous"])
    ap.add_argument("--moving-threshold", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.molecule.upper() == "H2":
        from rl_env import make_h2_config
        cfg = make_h2_config()
    else:
        from rl_env import make_lih_config
        cfg = make_lih_config(active_electrons=2, active_orbitals=5)

    print(f"{cfg['name']}: HF {cfg['hf_energy']:.6f}  FCI {cfg['fci_energy']:.6f}\n")
    net, env, hist, best = train(
        cfg, num_updates=args.updates, max_gates=args.max_gates,
        cnot_penalty=args.cnot_penalty, seed=args.seed, curriculum=args.curriculum,
        target_mode=args.target, number_penalty=args.number_penalty,
        order_k=args.order_k, algo=args.algo, inner_restarts=args.inner_restarts,
        inner_maxiter=args.inner_maxiter, gate_set=args.gate_set,
        moving_threshold=args.moving_threshold,
        hybrid_mode=args.hybrid_mode,
        hybrid_simultaneous=args.hybrid_mode == "simultaneous",
    )
    print(f"device: {pick_device()}")
    for level in ("chem", "exact"):
        b = best[level]
        label = "chem acc" if level == "chem" else "exact FCI"
        print(f"\n=== best @ {label} ===")
        if np.isfinite(b["cnots"]):
            print(f"  {b['cnots']} CNOTs, {b['n_gates']} gates, err {b['error_vs_fci']:+.2e}")
        else:
            print(f"  none (try more updates)")

    out = Path("results"); out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["return"]); ax[0].set_xlabel("update"); ax[0].set_ylabel("return")
    ax[1].plot(hist["best_cnots_chem"], label="chem")
    ax[1].plot(hist["best_cnots_exact"], label="exact")
    ax[1].legend(); ax[1].set_xlabel("update"); ax[1].set_ylabel("best CNOTs")
    fig.tight_layout()
    tag = cfg["name"].lower().split("(")[0]
    path = out / f"raw_gate_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
