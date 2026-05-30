# rl_env.py
# RL environment for circuit-structure search, molecule-agnostic via a config dict.
#
# The action space is STOP + singles + doubles (the UCCSD pool). That's the standard
# ansatz but not the whole story for real chemistry: generalized/spin-adapted excitations
# (k-UpCCGSD), adaptive pools that repeat operators (ADAPT-VQE), and gate ordering all
# matter. build_action_space is kept simple so a richer pool can drop in later.

from __future__ import annotations

import numpy as np
import pennylane as qml

from h2_vqe import build_h2_hamiltonian
from lih_vqe import build_lih_hamiltonian
from vqe_core import describe_actions, exact_ground_energy, run_vqe_on_circuit

STOP = {"type": "stop"}


def action_key(action: dict) -> tuple:
    # hashable id, used to mask duplicate excitations
    if action["type"] == "stop":
        return ("stop",)
    return (action["type"], tuple(action["wires"]))


def build_action_space(electrons: int, num_qubits: int) -> list[dict]:
    # STOP + all valid singles/doubles
    singles, doubles = qml.qchem.excitations(electrons, num_qubits)
    actions = [STOP]
    actions += [{"type": "single", "wires": list(w)} for w in singles]
    actions += [{"type": "double", "wires": list(w)} for w in doubles]
    return actions


def build_h2_action_space(num_qubits: int, electrons: int = 2) -> list[dict]:
    return build_action_space(electrons, num_qubits)


def make_config(name, H, num_qubits, hf_state, n_electrons, with_fci=True):
    actions = build_action_space(n_electrons, num_qubits)
    hf_energy = run_vqe_on_circuit(H, num_qubits, hf_state, [])["energy"]
    fci_energy = exact_ground_energy(H, num_qubits) if with_fci else None
    return {
        "name": name,
        "H": H,
        "num_qubits": num_qubits,
        "hf_state": hf_state,
        "n_electrons": n_electrons,
        "actions": actions,
        "hf_energy": hf_energy,
        "fci_energy": fci_energy,
    }


def make_h2_config():
    H, nq, hf = build_h2_hamiltonian()
    return make_config("H2", H, nq, hf, 2)


def make_lih_config(active_electrons=2, active_orbitals=5):
    H, nq, hf, ne = build_lih_hamiltonian(
        active_electrons=active_electrons, active_orbitals=active_orbitals
    )
    name = "LiH" if active_electrons is None else f"LiH({active_electrons}e,{active_orbitals}o)"
    return make_config(name, H, nq, hf, ne)


class CircuitStructureEnv:
    """Pick excitations one at a time until STOP (or max_excitations). Sparse reward at
    STOP: (E_HF - E_vqe) - lam * num_excitations."""

    def __init__(self, config=None, max_excitations: int = 4, lam: float = 0.0):
        if config is None:
            config = make_h2_config()
        self.config = config
        self.H = config["H"]
        self.num_qubits = config["num_qubits"]
        self.hf_state = config["hf_state"]
        self.actions = config["actions"]
        self.hf_energy = config["hf_energy"]
        self.fci_energy = config["fci_energy"]
        self.max_excitations = max_excitations
        self.lam = lam

        self.n_excitation_actions = len(self.actions) - 1
        self._vqe_cache: dict[tuple, float] = {}
        self.reset()

    def reset(self):
        self.action_history: list[dict] = []
        self.used_keys: set[tuple] = set()
        return self._state_vector(), {}

    def _state_vector(self) -> np.ndarray:
        state = np.zeros(self.n_excitation_actions, dtype=np.float64)
        for action in self.action_history:
            idx = self._excitation_index(action)
            if idx is not None:
                state[idx] = 1.0
        return state

    def _excitation_index(self, action: dict):
        key = action_key(action)
        for i, candidate in enumerate(self.actions[1:]):
            if action_key(candidate) == key:
                return i
        return None

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.actions), dtype=bool)
        mask[0] = True  # STOP always legal
        if len(self.action_history) < self.max_excitations:
            for i, action in enumerate(self.actions[1:], start=1):
                if action_key(action) not in self.used_keys:
                    mask[i] = True
        return mask

    def _energy_for(self, action_history) -> float:
        # cache by the unordered set of excitations; the optimized energy doesn't depend on order
        key = frozenset(action_key(a) for a in action_history)
        if key not in self._vqe_cache:
            vqe = run_vqe_on_circuit(self.H, self.num_qubits, self.hf_state, action_history)
            self._vqe_cache[key] = vqe["energy"]
        return self._vqe_cache[key]

    def step_gym(self, action_idx: int):
        action = self.actions[action_idx]
        info: dict = {}

        stopping = action["type"] == "stop" or len(self.action_history) >= self.max_excitations
        if stopping:
            if action["type"] != "stop":
                info["forced_stop"] = True
            energy = self._energy_for(self.action_history)
            n_exc = len(self.action_history)
            reward = (self.hf_energy - energy) - self.lam * n_exc
            info.update({
                "energy": energy,
                "n_excitations": n_exc,
                "description": describe_actions(self.action_history),
                "actions": list(self.action_history),
            })
            return self._state_vector(), reward, True, info

        self.action_history.append(action)
        self.used_keys.add(action_key(action))
        return self._state_vector(), 0.0, False, info
