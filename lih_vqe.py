# LiH VQE baseline. Full LiH (STO-3G) is 12 qubits, so we use an active space.

import time
from pathlib import Path

import numpy as np
import pennylane as qml

from vqe_core import (
    describe_actions,
    exact_ground_energy,
    plot_convergence,
    run_vqe_on_circuit,
)

BOND_LENGTH = 1.546  # Angstrom, near equilibrium


def build_lih_hamiltonian(bond_length=BOND_LENGTH, active_electrons=None, active_orbitals=None):
    # pass active_electrons/active_orbitals to freeze the core and shrink the qubit count
    symbols = ["Li", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]])
    mol = qml.qchem.Molecule(symbols, coords)
    H, num_qubits = qml.qchem.molecular_hamiltonian(
        mol, active_electrons=active_electrons, active_orbitals=active_orbitals
    )
    n_elec = active_electrons if active_electrons is not None else 4
    hf_state = qml.qchem.hf_state(n_elec, num_qubits)
    return H, num_qubits, hf_state, n_elec


def uccsd_actions(n_electrons, num_qubits):
    # full UCCSD ansatz (all singles + doubles) as a list of actions
    singles, doubles = qml.qchem.excitations(n_electrons, num_qubits)
    actions = [{"type": "single", "wires": list(w)} for w in singles]
    actions += [{"type": "double", "wires": list(w)} for w in doubles]
    return actions


def main(active_electrons=None, active_orbitals=None):
    label = "full" if active_electrons is None else f"active({active_electrons}e,{active_orbitals}o)"
    print(f"=== LiH VQE ({label}, bond {BOND_LENGTH} A) ===")

    t0 = time.time()
    H, num_qubits, hf_state, n_elec = build_lih_hamiltonian(
        active_electrons=active_electrons, active_orbitals=active_orbitals
    )
    singles, doubles = qml.qchem.excitations(n_elec, num_qubits)
    print(f"qubits: {num_qubits} | active electrons: {n_elec} | Hamiltonian build {time.time()-t0:.1f}s")
    print(f"available excitations: {len(singles)} singles, {len(doubles)} doubles "
          f"({len(singles)+len(doubles)} total)")

    hf_e = run_vqe_on_circuit(H, num_qubits, hf_state, [])["energy"]
    fci_e = exact_ground_energy(H, num_qubits)
    print(f"Hartree-Fock energy: {hf_e:.8f} Ha")
    print(f"FCI (exact in space): {fci_e:.8f} Ha")
    print(f"correlation energy:   {hf_e - fci_e:.8f} Ha\n")

    # UCCSD ansatz should land near FCI
    actions = uccsd_actions(n_elec, num_qubits)
    t0 = time.time()
    result = run_vqe_on_circuit(H, num_qubits, hf_state, actions)
    dt = time.time() - t0
    print(f"UCCSD ansatz ({result['n_params']} gates):")
    print(f"  energy: {result['energy']:.8f} Ha")
    print(f"  error vs FCI: {result['energy'] - fci_e:.2e} Ha")
    print(f"  single VQE call took {dt:.1f}s ({len(result['history']['energy'])} evals)\n")

    if dt > 30:
        print("VQE is slow (>30s). For RL, use an active space, e.g.:")
        print("  python -c \"import lih_vqe; lih_vqe.main(active_electrons=2, active_orbitals=5)\"")
        print("  train_lih.py defaults to that frozen-core space.\n")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    plot_path = out_dir / "lih_vqe_convergence.png"
    plot_convergence(result["history"], hf_e, fci_e, f"LiH VQE: UCCSD ({label})", plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
