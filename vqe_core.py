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


_BASIS_GATES = {"CNOT", "RX", "RY", "RZ", "Hadamard", "PhaseShift", "GlobalPhase", "PauliX"}

_DEV_CACHE: dict = {}


def _cached_device(num_qubits):
    dev = _DEV_CACHE.get(num_qubits)
    if dev is None:
        dev = qml.device("lightning.qubit", wires=num_qubits)
        _DEV_CACHE[num_qubits] = dev
    return dev


def optimize_angles(H, num_qubits, apply_fn, n_params, x0=None,
                    extra_restarts=1, maxiter=100, seed=0):
    """
    Minimize <H> over rotation angles using L-BFGS-B with adjoint (analytic) gradients.
    Empirically the best inner optimizer for these small circuits: faster and finds far
    better minima than COBYLA/Adam, and beats QNG (which only pays off in the barren-plateau
    regime of many qubits). apply_fn(params) applies BasisState + the parameterized circuit
    (no measurement). x0 warm-starts from previous angles; extra_restarts adds random tries
    to escape local minima. Returns (min_energy, best_params).
    """
    import pennylane.numpy as pnp
    from scipy.optimize import minimize

    dev = _cached_device(num_qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def cost(p):
        apply_fn(p)
        return qml.expval(H)

    if n_params == 0:
        return float(np.asarray(cost(pnp.array([]))).item()), np.array([])

    rng = np.random.default_rng(seed)
    inits = []
    if x0 is not None and len(np.atleast_1d(x0)) == n_params:
        inits.append(np.asarray(x0, dtype=float))
    inits += [rng.uniform(-np.pi, np.pi, n_params) for _ in range(extra_restarts)]
    if not inits:
        inits = [rng.uniform(-np.pi, np.pi, n_params)]

    def val_grad(p):
        pp = pnp.array(p, requires_grad=True)
        return float(cost(pp)), np.asarray(qml.grad(cost)(pp), dtype=float)

    best_e, best_x = np.inf, inits[0]
    for ini in inits:
        res = minimize(val_grad, ini, method="L-BFGS-B", jac=True,
                       options={"maxiter": maxiter})
        if res.fun < best_e:
            best_e, best_x = float(res.fun), np.asarray(res.x, dtype=float)
    return best_e, best_x


def compiled_resources(ops):
    """
    Decompose a list of pennylane ops into a basic gate set and count resources.
    This is the honest hardware-relevant metric: excitation gates are cheap to write
    but expensive in CNOTs (a DoubleExcitation ~= 14 CNOTs, SingleExcitation ~= 2).

    Returns dict with cnot_count, total_gates, and two_qubit_depth (CNOT-layer depth).
    """
    from collections import Counter

    tape = qml.tape.QuantumScript(list(ops), [])
    (decomposed,), _ = qml.transforms.decompose([tape], gate_set=_BASIS_GATES)
    counts = Counter(op.name for op in decomposed.operations)

    # CNOT depth: greedy layering over CNOTs only (wires busy within a layer)
    cnot_depth, layer_wires = 0, set()
    for op in decomposed.operations:
        if op.name != "CNOT":
            continue
        w = set(op.wires)
        if w & layer_wires:
            cnot_depth += 1
            layer_wires = w
        else:
            layer_wires |= w
    if layer_wires:
        cnot_depth += 1

    return {
        "cnot_count": counts.get("CNOT", 0),
        "total_gates": len(decomposed.operations),
        "cnot_depth": cnot_depth,
        "gate_types": dict(counts),
    }


def excitation_ops(actions):
    """Build the list of excitation ops (no BasisState) for resource counting."""
    ops = []
    for a in excitations_from_actions(actions):
        if a["type"] == "single":
            ops.append(qml.SingleExcitation(0.0, wires=a["wires"]))
        else:
            ops.append(qml.DoubleExcitation(0.0, wires=a["wires"]))
    return ops


def compiled_cnots_for_actions(actions):
    """Convenience: CNOT count for an excitation-based circuit (ignores HF prep)."""
    return compiled_resources(excitation_ops(actions))["cnot_count"]


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
