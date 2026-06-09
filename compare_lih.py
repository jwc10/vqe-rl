# Compare structure-search methods on LiH vs FCI: ADAPT, PPO, PPO warm-started from ADAPT, random.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adapt import adapt_vqe
from ppo import train_ppo
from rl_env import CircuitStructureEnv, action_key, make_lih_config
from vqe_core import describe_actions, run_vqe_on_circuit


def random_search(config, budget, max_excitations, seed=0):
    """Sample random circuits (deduped), keep the best. budget = #VQE optimizations."""
    rng = np.random.default_rng(seed)
    pool = [a for a in config["actions"] if a["type"] != "stop"]
    H, nq, hf = config["H"], config["num_qubits"], config["hf_state"]
    seen, best = {}, {"reward": -np.inf}

    while len(seen) < budget:
        k = rng.integers(1, max_excitations + 1)
        idx = rng.choice(len(pool), size=k, replace=False)
        actions = [pool[i] for i in idx]
        key = frozenset(action_key(a) for a in actions)
        if key in seen:
            continue
        energy = run_vqe_on_circuit(H, nq, hf, actions)["energy"]
        seen[key] = energy
        reward = config["hf_energy"] - energy
        if reward > best["reward"]:
            best = {"reward": reward, "energy": energy, "actions": actions,
                    "n_excitations": len(actions), "description": describe_actions(actions)}
    best["n_vqe_calls"] = len(seen)
    return best


def summarize(name, energy, n_exc, n_vqe, fci):
    return {"method": name, "energy": energy, "error_vs_fci": energy - fci,
            "n_excitations": n_exc, "n_vqe_calls": n_vqe}


def plot_comparison(rows, fci, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        ax.scatter(r["n_vqe_calls"], max(r["error_vs_fci"], 1e-7), s=80)
        ax.annotate(f"{r['method']}\n({r['n_excitations']} gates)",
                    (r["n_vqe_calls"], max(r["error_vs_fci"], 1e-7)),
                    textcoords="offset points", xytext=(6, 0), fontsize=8, va="center")
    ax.axhline(1.6e-3, color="green", linestyle="--", label="chemical accuracy (1.6 mHa)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("VQE optimizations (compute cost)")
    ax.set_ylabel("Energy error vs FCI (Ha)")
    ax.set_title("LiH: structure-search methods vs FCI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    cfg = make_lih_config(active_electrons=2, active_orbitals=5)
    fci = cfg["fci_energy"]
    print(f"{cfg['name']}: HF {cfg['hf_energy']:.8f}  FCI {fci:.8f}  pool {len(cfg['actions'])-1}\n")

    print("=== ADAPT-greedy ===")
    adapt = adapt_vqe(cfg, verbose=True)
    cap = adapt["n_excitations"]  # RL matches ADAPT's converged depth
    print(f"ADAPT converged at {cap} gates; using max_excitations={cap} for RL\n")

    print("=== PPO from scratch ===")
    _, _, ppo_best, ppo_env = train_ppo(
        cfg, num_updates=40, episodes_per_update=12, max_excitations=cap, seed=0
    )
    ppo_evals = len(ppo_env._vqe_cache)

    print("\n=== PPO warm-started from ADAPT ===")
    _, _, ws_best, ws_env = train_ppo(
        cfg, num_updates=40, episodes_per_update=12, max_excitations=cap, seed=0,
        warm_start_actions=adapt["actions"],
    )
    ws_evals = len(ws_env._vqe_cache)

    print("\n=== Random search (matched budget) ===")
    rand = random_search(cfg, budget=ppo_evals, max_excitations=cap, seed=0)

    rows = [
        summarize("ADAPT", adapt["energy"], adapt["n_excitations"], adapt["n_vqe_calls"], fci),
        summarize("PPO", ppo_best["energy"], ppo_best["n_excitations"], ppo_evals, fci),
        summarize("PPO+warmstart", ws_best["energy"], ws_best["n_excitations"], ws_evals, fci),
        summarize("random", rand["energy"], rand["n_excitations"], rand["n_vqe_calls"], fci),
    ]

    print(f"\n{'method':16s} {'energy':>13s} {'err vs FCI':>12s} {'gates':>6s} {'VQE opt':>8s}")
    for r in rows:
        print(f"{r['method']:16s} {r['energy']:13.8f} {r['error_vs_fci']:12.2e} "
              f"{r['n_excitations']:6d} {r['n_vqe_calls']:8d}")
    print("brute force      infeasible (>1e6 structures over the 24-operator pool)")

    plot_comparison(rows, fci, out_dir / "lih_method_comparison.png")
    with open(out_dir / "lih_comparison.json", "w") as f:
        json.dump({"hf_energy": cfg["hf_energy"], "fci_energy": fci,
                   "cap": cap, "methods": rows,
                   "adapt_circuit": adapt["description"],
                   "ppo_circuit": ppo_best["description"],
                   "warmstart_circuit": ws_best["description"]}, f, indent=2)
    print(f"\nSaved {out_dir/'lih_method_comparison.png'} and {out_dir/'lih_comparison.json'}")


if __name__ == "__main__":
    main()
