# prune_baselines.py
# Option B: start from a full circuit (UCCSD pool or ADAPT's output) and remove gates
# while staying within chemical accuracy of FCI. The honest baseline for "can RL prune?"
# is NOT random removal -- it is greedy backward elimination. We provide both so the gap
# between them is visible (RL only matters if it beats greedy, not random).
#
# Metric is compiled CNOT count (the hardware-relevant cost), not raw excitation count.

from __future__ import annotations

import numpy as np

from vqe_core import compiled_cnots_for_actions, run_vqe_on_circuit

CHEM_ACC = 1.6e-3


def _energy(cfg, actions):
    return run_vqe_on_circuit(cfg["H"], cfg["num_qubits"], cfg["hf_state"], actions)["energy"]


def _summary(cfg, actions, n_evals, method):
    e = _energy(cfg, actions)
    return {
        "method": method,
        "energy": e,
        "error_vs_fci": e - cfg["fci_energy"],
        "n_excitations": len(actions),
        "cnots": compiled_cnots_for_actions(actions),
        "n_vqe_calls": n_evals,
        "within_chem_acc": (e - cfg["fci_energy"]) < CHEM_ACC,
        "actions": list(actions),
    }


def greedy_backward_elimination(cfg, start_actions, chem_acc=CHEM_ACC):
    """
    Repeatedly drop the single gate whose removal least increases energy; keep dropping
    while the circuit stays within chemical accuracy. Deterministic, strong baseline.
    """
    actions = list(start_actions)
    n_evals = 0
    improved = True

    while improved and actions:
        improved = False
        best_drop, best_energy = None, np.inf

        for i in range(len(actions)):
            trial = actions[:i] + actions[i + 1:]
            e = _energy(cfg, trial)
            n_evals += 1
            if e < best_energy:
                best_energy, best_drop = e, i

        # accept the best removal only if it keeps us within chemical accuracy
        if best_drop is not None and (best_energy - cfg["fci_energy"]) < chem_acc:
            actions = actions[:best_drop] + actions[best_drop + 1:]
            improved = True

    return _summary(cfg, actions, n_evals, "greedy_backward")


def random_pruning(cfg, start_actions, n_trials=200, chem_acc=CHEM_ACC, seed=0):
    """
    Random-subset baseline: drop a random subset, keep the smallest circuit (fewest CNOTs)
    that stays within chemical accuracy. This is the weak baseline RL should beat trivially.
    """
    rng = np.random.default_rng(seed)
    n_evals = 0
    best = None

    for _ in range(n_trials):
        keep_mask = rng.random(len(start_actions)) > rng.uniform(0.2, 0.8)
        trial = [a for a, k in zip(start_actions, keep_mask) if k]
        if not trial:
            continue
        e = _energy(cfg, trial)
        n_evals += 1
        if (e - cfg["fci_energy"]) < chem_acc:
            cnots = compiled_cnots_for_actions(trial)
            if best is None or cnots < best["cnots"]:
                best = _summary(cfg, trial, n_evals, "random_prune")

    if best is None:
        best = _summary(cfg, start_actions, n_evals, "random_prune")
    else:
        best["n_vqe_calls"] = n_evals
    return best


def main(include_uccsd=False):
    from adapt import adapt_vqe
    from lih_vqe import uccsd_actions
    from rl_env import make_lih_config

    cfg = make_lih_config(active_electrons=2, active_orbitals=5)
    print(f"{cfg['name']}: HF {cfg['hf_energy']:.6f}  FCI {cfg['fci_energy']:.6f}\n")

    # ADAPT's compact circuit is the cheap, meaningful starting point. Pruning from the
    # full UCCSD pool is O(n^2) VQE calls on large circuits (slow), so it's opt-in.
    adapt = adapt_vqe(cfg, verbose=False)["actions"]
    starts = {"from_ADAPT": adapt}
    if include_uccsd:
        starts["from_UCCSD"] = uccsd_actions(cfg["n_electrons"], cfg["num_qubits"])

    def line(tag, s):
        print(f"  {tag:18s} gates={s['n_excitations']:2d} cnots={s['cnots']:3d} "
              f"err_vs_fci={s['error_vs_fci']:+.2e} chem_acc={s['within_chem_acc']} "
              f"vqe_calls={s['n_vqe_calls']}")

    for name, start in starts.items():
        s0 = _summary(cfg, start, 0, "start")
        print(f"=== {name} (start: {s0['n_excitations']} gates, {s0['cnots']} CNOTs, "
              f"err {s0['error_vs_fci']:+.2e}) ===")
        greedy = greedy_backward_elimination(cfg, start)
        rand = random_pruning(cfg, start, n_trials=200)
        line("greedy_backward", greedy)
        line("random_prune", rand)
        print()


if __name__ == "__main__":
    main()
