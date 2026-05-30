# vqe_core.py
# shared VQE machinery, used by both h2_vqe and lih_vqe

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import scipy.sparse.linalg as spla
from scipy.optimize import minimize


def excitations_from_actions(actions):
    return [a for a in actions if a.get("type") in ("single", "double")]


def make_structure_qnode(H, num_qubits, hf_state, actions):
    """Build the VQE circuit for a set of excitations, one angle per gate."""
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
            else:
                qml.DoubleExcitation(params[i], wires=action["wires"])
        return qml.expval(H)
    return circuit, len(excitations)


def run_vqe_on_circuit(H, num_qubits, hf_state, actions, x0=None):
    """Optimize the gate angles for a fixed structure with BFGS."""
    circuit, n_params = make_structure_qnode(H, num_qubits, hf_state, actions)
    history = {"energy": []}

    if n_params == 0:
        energy = np.asarray(circuit()).item()
        history["energy"].append(energy)
        return {"energy": energy, "params": np.array([]), "n_params": 0,
                "history": history, "success": True}

    if x0 is None:
        x0 = np.zeros(n_params)

    history["energy"].append(np.asarray(circuit(x0)).item())

    def cost(params):
        e = np.asarray(circuit(params)).item()
        history["energy"].append(e)
        return e

    result = minimize(cost, x0=x0, method="BFGS")
    return {"energy": result.fun, "params": result.x, "n_params": n_params,
            "history": history, "success": result.success}


def exact_ground_energy(H, num_qubits):
    """FCI in the active space, i.e. lowest eigenvalue of the qubit Hamiltonian."""
    mat = qml.SparseHamiltonian(H.sparse_matrix(), wires=range(num_qubits)).sparse_matrix()
    if num_qubits <= 4:
        return float(np.linalg.eigvalsh(mat.toarray())[0])
    vals = spla.eigsh(mat, k=1, which="SA", return_eigenvectors=False)
    return float(vals[0])


def describe_actions(actions):
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


def plot_convergence(history, hf_e, fci_e, title, out_path):
    steps = range(len(history["energy"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, history["energy"], "o-", label="VQE energy")
    ax.axhline(hf_e, color="gray", linestyle="--", label=f"HF = {hf_e:.4f} Ha")
    ax.axhline(fci_e, color="red", linestyle=":", label=f"FCI = {fci_e:.4f} Ha")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
