"""
Minimal H2 VQE baseline for CS224R milestone.

Based on PennyLane's VQE tutorial:
https://pennylane.ai/qml/demos/tutorial_vqe/

Run:
    python h2_vqe.py

Outputs:
    - prints HF and optimized energies
    - saves results/h2_vqe_convergence.png
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from scipy.optimize import minimize

# H2 molecule setup with STO-3G basis set and an equilibrium bond length of about 1.4 Bohr
def build_h2_hamiltonian():
    """returns (Hamiltonian, #qubits needed, hf_state (hartree fock state)) for the H2 molecule"""
    symbols = ["H", "H"]
    # coordinates from PennyLane tutorial for H2 molecule
    coordinates = np.array([[-0.70108983, 0.0, 0.0], [0.70108983, 0.0, 0.0]])
    molecule = qml.qchem.Molecule(symbols, coordinates)
    H, num_qubits = qml.qchem.molecular_hamiltonian(molecule)
    electrons = 2
    hf_state = qml.qchem.hf_state(electrons, qubits)
    return H, num_qubits, hf_state

def make_energy_qnodes(H, num_qubits, hf_state):
    """
    Creates 2 circuits: hartree fock only (no variational parameters), and ansatz circuit (hartree fock with double excitation)
    """
    dev = qml.device("lightning.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def hf_energy():
        qml.BasisState(hf_state, wires=range(num_qubits))
        return qml.expval(H)

    @qml.qnode(dev)
    def ansatz_energy(theta):
        qml.BasisState(hf_state, wires=range(num_qubits))
        # adds in electron interaction (this represents bpth electrons being excited)
        qml.DoubleExcitation(theta, wires=[0, 1, 2, 3])
        return qml.expval(H)

    return hf_energy, ansatz_energy


# ---------------------------------------------------------------------------
# Optional: random circuits (for later RL / random-search experiments)
# ---------------------------------------------------------------------------

def random_circuit_energy(H, qubits, hf, depth, rng):
    """
    Build and evaluate a random RX/RY/RZ/CNOT circuit on top of HF.

    TODO (easy extension): call this in a loop over depths {2,4,6,8}
    and plot depth vs energy. Swap this function in place of ansatz_energy
    when you want random-search instead of VQE optimization.
    """
    dev = qml.device("lightning.qubit", wires=qubits)
    gate_pool = ["RX", "RY", "RZ", "CNOT"]

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(hf, wires=range(qubits))
        for _ in range(depth):
            gate = rng.choice(gate_pool)
            if gate == "CNOT":
                control, target = rng.choice(qubits, size=2, replace=False)
                qml.CNOT(wires=[control, target])
            else:
                wire = rng.integers(qubits)
                angle = rng.uniform(0, 2 * np.pi)
                getattr(qml, gate)(angle, wires=wire)
        return qml.expval(H)

    return circuit()


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_vqe(ansatz_energy, theta0=0.0):
    """Minimize ansatz energy; track history for plotting."""
    history = {
        "theta": [theta0],
        "energy": [np.asarray(ansatz_energy(theta0)).item()],
    }

    def cost(theta):
        e = np.asarray(ansatz_energy(theta[0])).item()
        history["theta"].append(float(theta[0]))
        history["energy"].append(e)
        return e

    result = minimize(cost, x0=[theta0], method="BFGS")
    return result, history


def plot_convergence(history, hf_e, fci_e, out_path):
    """Energy vs optimization step."""
    steps = range(len(history["energy"]))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, history["energy"], "o-", label="VQE energy")
    ax.axhline(hf_e, color="gray", linestyle="--", label=f"HF = {hf_e:.4f} Ha")
    ax.axhline(fci_e, color="red", linestyle=":", label=f"FCI ≈ {fci_e:.4f} Ha")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("H2 VQE: HF + double excitation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    H, qubits, hf = build_h2_hamiltonian()
    hf_energy, ansatz_energy = make_energy_qnodes(H, qubits, hf)

    hf_e = hf_energy()
    print(f"H2 uses {qubits} qubits (STO-3G)")
    print(f"Hartree-Fock energy: {hf_e:.8f} Ha")

    result, history = run_vqe(ansatz_energy)
    print(f"Optimized VQE energy: {result.fun:.8f} Ha")
    print(f"Optimal theta: {result.x[0]:.4f} rad")

    # Reference from PennyLane H2 dataset (avoids extra h5py dependency)
    fci_e = -1.13726250
    print(f"FCI reference:        {fci_e:.8f} Ha")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    plot_path = out_dir / "h2_vqe_convergence.png"
    plot_convergence(history, hf_e, fci_e, plot_path)
    print(f"Saved plot to {plot_path}")

    # Uncomment to try one random circuit:
    # rng = np.random.default_rng(0)
    # print("Random circuit (depth=4):", random_circuit_energy(H, qubits, hf, depth=4, rng=rng))


if __name__ == "__main__":
    main()
