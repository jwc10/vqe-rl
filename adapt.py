# adapt.py
# ADAPT-VQE greedy baseline. Each step screens every pool operator by its energy gradient
# at theta=0 (chosen operators held at their angles), adds the largest, re-optimizes all
# angles, and repeats until the gradient drops below grad_tol. Same pool as the RL agent,
# but a hand-built selection rule instead of a learned policy.

from __future__ import annotations

import numpy as np
import pennylane as qml

from vqe_core import describe_actions, run_vqe_on_circuit


def _apply_gate(action, theta):
    if action["type"] == "single":
        qml.SingleExcitation(theta, wires=action["wires"])
    else:
        qml.DoubleExcitation(theta, wires=action["wires"])


def _grad_qnode(H, num_qubits, hf_state, chosen, params, candidate):
    # d<H>/d(candidate angle) at 0, with chosen operators fixed at their angles
    dev = qml.device("lightning.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(theta):
        qml.BasisState(hf_state, wires=range(num_qubits))
        for a, p in zip(chosen, params):
            _apply_gate(a, p)
        _apply_gate(candidate, theta)
        return qml.expval(H)

    eps = 1e-4
    return float((circuit(eps) - circuit(-eps)) / (2 * eps))


def adapt_vqe(config, grad_tol=1e-3, max_iter=None, chem_acc=1.6e-3, verbose=True):
    H, nq, hf = config["H"], config["num_qubits"], config["hf_state"]
    pool = [a for a in config["actions"] if a["type"] != "stop"]
    fci = config["fci_energy"]
    if max_iter is None:
        max_iter = len(pool)

    chosen, params = [], np.array([])
    energy = config["hf_energy"]
    used = set()
    history = [energy]
    depth_to_chem_acc = None
    n_vqe_calls = 0
    n_grad_evals = 0

    for it in range(max_iter):
        # screen the pool by gradient magnitude
        grads = []
        for cand in pool:
            key = (cand["type"], tuple(cand["wires"]))
            if key in used:
                grads.append(0.0)
                continue
            grads.append(_grad_qnode(H, nq, hf, chosen, params, cand))
            n_grad_evals += 2
        grads = np.abs(grads)
        best_i = int(np.argmax(grads))
        max_grad = grads[best_i]

        if max_grad < grad_tol:
            if verbose:
                print(f"  converged: max gradient {max_grad:.2e} < {grad_tol}")
            break

        chosen.append(pool[best_i])
        used.add((pool[best_i]["type"], tuple(pool[best_i]["wires"])))

        res = run_vqe_on_circuit(H, nq, hf, chosen, x0=np.append(params, 0.0))
        n_vqe_calls += 1
        params, energy = res["params"], res["energy"]
        history.append(energy)

        if depth_to_chem_acc is None and energy - fci < chem_acc:
            depth_to_chem_acc = len(chosen)

        if verbose:
            print(f"  step {len(chosen):2d} | grad {max_grad:.2e} | E {energy:.8f} "
                  f"| err vs FCI {energy - fci:+.2e}")

    return {
        "actions": chosen,
        "params": params,
        "energy": energy,
        "error_vs_fci": energy - fci,
        "n_excitations": len(chosen),
        "depth_to_chem_acc": depth_to_chem_acc,
        "history": history,
        "n_vqe_calls": n_vqe_calls,
        "n_grad_evals": n_grad_evals,
        "description": describe_actions(chosen),
    }


def main():
    from rl_env import make_lih_config

    cfg = make_lih_config(active_electrons=2, active_orbitals=5)
    print(f"{cfg['name']}: HF {cfg['hf_energy']:.8f}  FCI {cfg['fci_energy']:.8f}  "
          f"pool {len(cfg['actions'])-1} operators")
    print("ADAPT-VQE greedy:")
    res = adapt_vqe(cfg)
    print(f"\nfinal energy: {res['energy']:.8f} Ha  (err vs FCI {res['error_vs_fci']:+.2e})")
    print(f"gates: {res['n_excitations']} | depth to chemical accuracy: {res['depth_to_chem_acc']}")
    print(f"VQE optimizations: {res['n_vqe_calls']} | gradient circuit evals: {res['n_grad_evals']}")
    print(f"circuit: {res['description']}")


if __name__ == "__main__":
    main()
