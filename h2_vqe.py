# h2_vqe.py
# H2 VQE baseline: RL picks the excitation structure, VQE optimizes the angles.

from pathlib import Path

import numpy as np
import pennylane as qml

from vqe_core import describe_actions, plot_convergence, run_vqe_on_circuit

FCI_ENERGY = -1.13726250  # STO-3G H2 ground state from PennyLane


def build_h2_hamiltonian():
    """Return (Hamiltonian, num_qubits, hf_state) for H2."""
    symbols = ["H", "H"]
    coordinates = np.array([[-0.70108983, 0.0, 0.0], [0.70108983, 0.0, 0.0]])
    molecule = qml.qchem.Molecule(symbols, coordinates)
    H, num_qubits = qml.qchem.molecular_hamiltonian(molecule)
    hf_state = qml.qchem.hf_state(2, num_qubits)
    return H, num_qubits, hf_state


def main():
    H, num_qubits, hf_state = build_h2_hamiltonian()

    structures = {
        "hf_only": [],
        "bad_single": [{"type": "single", "wires": [0, 2]}],
        "good_double": [{"type": "double", "wires": [0, 1, 2, 3]}],
    }

    hf_e = run_vqe_on_circuit(H, num_qubits, hf_state, [])["energy"]
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
    plot_convergence(best["history"], hf_e, FCI_ENERGY, f"H2 VQE: {best['label']}", plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
