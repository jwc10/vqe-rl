# train_lih.py
# RL circuit-structure search on LiH. Uses the frozen-core active space (2e, 5o -> 10
# qubits): full LiH is 12 qubits and one UCCSD VQE call takes ~47s, while the active space
# reaches FCI in ~2s and keeps ~96% of the correlation. Runs PPO and REINFORCE+baseline,
# saves training curves and the best circuit. RUN_PARETO adds the depth reward-shaping sweep.

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ppo import train_ppo
from rl_env import make_lih_config
from train_reinforce import train as train_reinforce

# cap = ADAPT's converged depth (it reaches FCI in 8 gates), so RL isn't held back by an
# arbitrary limit
MAX_EXC = 8
RUN_PARETO = True
# small lambdas: marginal energy per gate is only ~1e-4 Ha at this depth, so bigger
# penalties just collapse the circuit to HF
PARETO_LAMBDAS = [0.0, 0.0001, 0.0003, 0.001]


def moving_avg(x, window=25):
    return [np.mean(x[max(0, i - window + 1): i + 1]) for i in range(len(x))]


def report(tag, best, cfg):
    err = best["energy"] - cfg["fci_energy"]
    print(f"\n[{tag}] best circuit: {best['description']}")
    print(f"  energy: {best['energy']:.8f} Ha")
    print(f"  vs HF:  {best['energy'] - cfg['hf_energy']:+.6f} Ha   vs FCI: {err:+.2e} Ha")
    print(f"  gates:  {best.get('n_excitations', '?')}")


def plot_curves(ppo_hist, rein_hist, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ppo_hist, alpha=0.2, color="C0")
    ax.plot(moving_avg(ppo_hist), color="C0", label="PPO")
    ax.plot(rein_hist, alpha=0.2, color="C1")
    ax.plot(moving_avg(rein_hist), color="C1", label="REINFORCE + baseline")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (E_HF - E)")
    ax.set_title("LiH circuit-structure search")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pareto(points, out_path):
    points = sorted(points, key=lambda p: p["n_excitations"])
    xs = [p["n_excitations"] for p in points]
    ys = [p["energy"] for p in points]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-")
    for p in points:
        ax.annotate(f"lam={p['lam']}", (p["n_excitations"], p["energy"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Number of excitations (circuit depth)")
    ax.set_ylabel("Best energy (Hartree)")
    ax.set_title("LiH energy vs circuit depth (reward-shaping sweep)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("Building LiH active space...")
    cfg = make_lih_config(active_electrons=2, active_orbitals=5)
    n_exc = len(cfg["actions"]) - 1
    print(f"{cfg['name']}: {cfg['num_qubits']} qubits, {n_exc} excitations, "
          f"max_excitations={MAX_EXC}")
    print(f"HF energy:  {cfg['hf_energy']:.8f} Ha")
    print(f"FCI energy: {cfg['fci_energy']:.8f} Ha")
    print(f"correlation to recover: {cfg['hf_energy'] - cfg['fci_energy']:.6f} Ha")

    print("\n=== PPO ===")
    t0 = time.time()
    _, ppo_hist, ppo_best, _ = train_ppo(
        cfg, num_updates=50, episodes_per_update=12, max_excitations=MAX_EXC, seed=0
    )
    print(f"PPO done in {time.time()-t0:.0f}s")
    report("PPO", ppo_best, cfg)

    print("\n=== REINFORCE + baseline ===")
    t0 = time.time()
    _, _, rein_hist, rein_best = train_reinforce(
        num_episodes=600, lr=0.1, config=cfg, max_excitations=MAX_EXC, seed=0
    )
    print(f"REINFORCE done in {time.time()-t0:.0f}s")
    report("REINFORCE", rein_best, cfg)

    plot_curves(ppo_hist, rein_hist, out_dir / "lih_training_curves.png")
    print(f"\nSaved training curves to {out_dir / 'lih_training_curves.png'}")

    overall = ppo_best if ppo_best["reward"] >= rein_best["reward"] else rein_best
    best_out = {
        "molecule": cfg["name"],
        "num_qubits": cfg["num_qubits"],
        "hf_energy": cfg["hf_energy"],
        "fci_energy": cfg["fci_energy"],
        "best_energy": overall["energy"],
        "error_vs_fci": overall["energy"] - cfg["fci_energy"],
        "num_excitations": overall.get("n_excitations"),
        "circuit": overall["description"],
        "excitations": [a for a in overall["actions"]],
    }
    with open(out_dir / "lih_best_circuit.json", "w") as f:
        json.dump(best_out, f, indent=2)
    print(f"Saved best circuit to {out_dir / 'lih_best_circuit.json'}")

    if RUN_PARETO:
        print("\n=== Reward-shaping sweep (energy vs depth) ===")
        points = []
        for lam in PARETO_LAMBDAS:
            _, _, best, _ = train_ppo(
                cfg, num_updates=25, episodes_per_update=12,
                max_excitations=MAX_EXC, lam=lam, seed=0, log_every=0
            )
            points.append({
                "lam": lam,
                "energy": best["energy"],
                "n_excitations": best["n_excitations"],
                "circuit": best["description"],
            })
            print(f"lam={lam:<6} energy={best['energy']:.6f} "
                  f"gates={best['n_excitations']} | {best['description']}")
        plot_pareto(points, out_dir / "lih_pareto.png")
        print(f"Saved Pareto plot to {out_dir / 'lih_pareto.png'}")


if __name__ == "__main__":
    main()
