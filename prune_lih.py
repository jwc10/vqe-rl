# prune_lih.py
# Can RL prune ADAPT's circuit? Warm-start PPO from ADAPT's sequence, then fine-tune with
# a small depth penalty and extra entropy. The penalty makes the tail gates (each worth
# only ~1e-4 Ha) net-negative, shifting the optimum to a shorter circuit. We report the
# best-reward circuit and the shortest circuit within chemical accuracy that RL evaluated.

import json
from pathlib import Path

import numpy as np

from adapt import adapt_vqe
from ppo import train_ppo
from rl_env import make_lih_config

CHEM_ACC = 1.6e-3
# at lam=3e-4 a ~5-gate circuit is reward-optimal (the last few ADAPT gates each buy
# < 3e-4 Ha, below their penalty); lam=5e-4 prunes a touch harder
LAMBDAS = [3e-4, 5e-4]
ENTROPY = 0.1


def describe_key(key):
    parts = []
    for typ, wires in sorted(key):
        if typ == "single":
            parts.append(f"Single({wires[0]},{wires[1]})")
        else:
            parts.append(f"Double({','.join(map(str, wires))})")
    return "HF + " + " + ".join(parts) if parts else "HF only"


def shortest_within_chem_acc(env, fci):
    # shortest evaluated structure within chemical accuracy of FCI
    best = None
    for key, energy in env._vqe_cache.items():
        if energy - fci < CHEM_ACC:
            n = len(key)
            if best is None or (n, energy) < (best["gates"], best["energy"]):
                best = {"gates": n, "energy": energy, "err": energy - fci,
                        "circuit": describe_key(key)}
    return best


def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    cfg = make_lih_config(active_electrons=2, active_orbitals=5)
    fci = cfg["fci_energy"]
    print(f"{cfg['name']}: HF {cfg['hf_energy']:.8f}  FCI {fci:.8f}")

    print("ADAPT (demo + reference)...")
    adapt = adapt_vqe(cfg, verbose=False)
    cap = adapt["n_excitations"]
    print(f"ADAPT: {cap} gates, energy {adapt['energy']:.8f} (err {adapt['error_vs_fci']:.2e})\n")

    # naive hard warm-start just replays ADAPT (BC too peaked to explore), so we compare a
    # from-scratch arm against a soft-BC warm-start (gentle prior) with high entropy
    arms = [
        ("scratch", {}),
        ("warmstart-soft", {"warm_start_actions": adapt["actions"], "bc_epochs": 40}),
    ]

    rows = []
    for lam in LAMBDAS:
        for name, kw in arms:
            print(f"=== {name}, lam={lam}, entropy={ENTROPY} ===")
            _, _, best, env = train_ppo(
                cfg, num_updates=40, episodes_per_update=12, max_excitations=cap,
                lam=lam, entropy_coef=ENTROPY, seed=0, log_every=10, **kw,
            )
            compact = shortest_within_chem_acc(env, fci)
            print(f"  best-reward circuit: {best['n_excitations']} gates, "
                  f"E {best['energy']:.8f} (err {best['energy']-fci:.2e})")
            if compact:
                print(f"  shortest within chem acc: {compact['gates']} gates, "
                      f"E {compact['energy']:.8f} (err {compact['err']:.2e})")
                print(f"    {compact['circuit']}")
            print(f"  structures evaluated: {len(env._vqe_cache)}\n")
            rows.append({"lam": lam, "arm": name, "best_reward_gates": best["n_excitations"],
                         "best_reward_energy": best["energy"], "compact": compact,
                         "n_evals": len(env._vqe_cache)})

    print(f"ADAPT baseline: {cap} gates at FCI ({adapt['error_vs_fci']:.2e})")
    print("\nsummary (shortest circuit within chemical accuracy):")
    print(f"{'lam':>8s} {'arm':>16s} {'gates':>6s} {'energy':>13s} {'err vs FCI':>12s} {'evals':>6s}")
    for r in rows:
        c = r["compact"]
        g = c["gates"] if c else None
        e = c["energy"] if c else float("nan")
        err = c["err"] if c else float("nan")
        print(f"{r['lam']:8.4f} {r['arm']:>16s} {str(g):>6s} {e:13.8f} {err:12.2e} {r['n_evals']:6d}")

    with open(out_dir / "lih_prune.json", "w") as f:
        json.dump({"adapt_gates": cap, "adapt_energy": adapt["energy"],
                   "chem_acc": CHEM_ACC, "results": rows}, f, indent=2)
    print(f"\nSaved {out_dir/'lih_prune.json'}")


if __name__ == "__main__":
    main()
