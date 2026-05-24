"""
Minimal H2 VQE baseline for CS224R milestone.

RL (later) chooses excitation *structure* (actions).
Classical VQE optimizes the continuous angles for that structure.

Run:
    python h2_vqe.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from scipy.optimize import minimize

FCI_ENERGY = -1.13726250  #(STO-3G) H2 ground state energy from PennyLane


def build_h2_hamiltonian():
    """Return (Hamiltonian, num_qubits, hf_state) for H2."""
    symbols = ["H", "H"]
    coordinates = np.array([[-0.70108983, 0.0, 0.0], [0.70108983, 0.0, 0.0]])
    molecule = qml.qchem.Molecule(symbols, coordinates)
    H, num_qubits = qml.qchem.molecular_hamiltonian(molecule)
    hf_state = qml.qchem.hf_state(2, num_qubits)
    return H, num_qubits, hf_state


# circuit created from actions - RL outputs these actions
def excitations_from_actions(actions):
    """
    Maps actions into excitation gates. The 'stop' action is ignored.
    Actions are dicts as seen below:
        {"type": "single", "wires": [0, 2]}
        {"type": "double", "wires": [0, 1, 2, 3]}
        {"type": "stop"}
    """
    return [a for a in actions if a.get("type") in ("single", "double")]

def make_structure_qnode(H, num_qubits, hf_state, actions):
    """
    Builds a QNode/circuit for a given set of excitations, with one variational param for each one.
    """
    excitations = excitations_from_actions(actions)
    dev = qml.device("lightning.qubit", wires=num_qubits)

    if not excitations:
        @qml.qnode(dev)
        def circuit():
            qml.BasisState(hf_state, wires=range(num_qubits))
            return qml.expval(H)
        return circuit, 0

    @qml.qnode(dev)
    def circuit(params):
        qml.BasisState(hf_state, wires=range(num_qubits))
        for i, action in enumerate(excitations):
            if action["type"] == "single":
                qml.SingleExcitation(params[i], wires=action["wires"])
            elif action["type"] == "double":
                qml.DoubleExcitation(params[i], wires=action["wires"])
        return qml.expval(H)
    return circuit, len(excitations)


def run_vqe_on_circuit(H, num_qubits, hf_state, actions, x0=None):
    """
    The inner loop takes in a circuit with various excitations and optimizes its parameters.
    x0 contains the initial parameters before optimization.
    Returns a dict containing the optimized energy, optimal parameters, and optimization history.
    """
    circuit, n_params = make_structure_qnode(H, num_qubits, hf_state, actions)
    history = {"energy": []}

    if n_params == 0:
        energy = np.asarray(circuit()).item()
        history["energy"].append(energy)
        return {
            "energy": energy,
            "params": np.array([]),
            "n_params": 0,
            "history": history,
            "success": True,
        }

    if x0 is None:
        x0 = np.zeros(n_params)

    history["energy"].append(np.asarray(circuit(x0)).item())

    def cost(params):
        energy = np.asarray(circuit(params)).item()
        history["energy"].append(energy)
        return energy

    result = minimize(cost, x0=x0, method="BFGS")
    return {
        "energy": result.fun,
        "params": result.x,
        "n_params": n_params,
        "history": history,
        "success": result.success,
    }


def describe_actions(actions):
    """This creates a readable description of the actions taken in order to create the circuit."""
    excitations = excitations_from_actions(actions)
    if not excitations:
        return "HF only"
    parts = []
    for action in excitations:
        wires = action["wires"]
        if action["type"] == "single":
            parts.append(f"Single({wires[0]},{wires[1]})")
        else:
            parts.append(f"Double({','.join(map(str, wires))})")
    return "HF + " + " + ".join(parts)

# Plots VQE energy vs number of optimization steps
def plot_convergence(history, hf_e, fci_e, title, out_path):
    steps = range(len(history["energy"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, history["energy"], "o-", label="VQE energy")
    ax.axhline(hf_e, color="gray", linestyle="--", label=f"HF = {hf_e:.4f} Ha")
    ax.axhline(fci_e, color="red", linestyle=":", label=f"FCI ≈ {fci_e:.4f} Ha")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    H, num_qubits, hf_state = build_h2_hamiltonian()

    # example structures for h2 molecule vqe circuit to compare
    structures = {
        "hf_only": [],
        "bad_single": [{"type": "single", "wires": [0, 2]}],
        "good_double": [{"type": "double", "wires": [0, 1, 2, 3]}],
    }

    hf_result = run_vqe_on_circuit(H, num_qubits, hf_state, [])
    hf_e = hf_result["energy"]

    print(f"H2 uses {num_qubits} qubits (STO-3G)")
    print(f"Hartree-Fock energy: {hf_e:.8f} Ha\n")

    best = None
    for name, actions in structures.items():
        result = run_vqe_on_circuit(H, num_qubits, hf_state, actions)
        label = describe_actions(actions)
        print(f"[{name}] {label}")
        print(f"  optimized energy: {result['energy']:.8f} Ha")
        print(f"  params ({result['n_params']}): {np.round(result['params'], 4).tolist()}")
        if best is None or result["energy"] < best["energy"]:
            best = {"name": name, "label": label, **result}

    print(f"\nFCI reference: {FCI_ENERGY:.8f} Ha")
    print(f"Best structure: [{best['name']}] {best['label']}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    plot_path = out_dir / "h2_vqe_convergence.png"
    plot_convergence(
        best["history"],
        hf_e,
        FCI_ENERGY,
        f"H2 VQE: {best['label']}",
        plot_path,
    )
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
