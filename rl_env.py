"""RL environment and action-space helpers."""

from __future__ import annotations

import numpy as np
import pennylane as qml

from h2_vqe import build_h2_hamiltonian, describe_actions, run_vqe_on_circuit

STOP = {"type": "stop"}


def action_key(action: dict) -> tuple:
    """Hashable id for masking duplicate excitations."""
    if action["type"] == "stop":
        return ("stop",)
    return (action["type"], tuple(action["wires"]))


def build_h2_action_space(num_qubits: int, electrons: int = 2) -> list[dict]:
    """
    STOP + all valid singles/doubles for the molecule.
    For H2 (4 spin orbitals): STOP, 2 singles, 1 double.
    """
    singles, doubles = qml.qchem.excitations(electrons, num_qubits)
    actions = [STOP]
    for wires in singles:
        actions.append({"type": "single", "wires": list(wires)})
    for wires in doubles:
        actions.append({"type": "double", "wires": list(wires)})
    return actions


class CircuitStructureEnv:
    def __init__(self, max_excitations: int = 4):
        self.H, self.num_qubits, self.hf_state = build_h2_hamiltonian()
        self.actions = build_h2_action_space(self.num_qubits)
        self.max_excitations = max_excitations

        hf_result = run_vqe_on_circuit(self.H, self.num_qubits, self.hf_state, [])
        self.hf_energy = hf_result["energy"]

        self.n_excitation_actions = len(self.actions) - 1
        self.reset()

    def reset(self) -> tuple[np.ndarray, dict]:
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

    def _excitation_index(self, action: dict) -> int | None:
        key = action_key(action)
        for i, candidate in enumerate(self.actions[1:], start=0):
            if action_key(candidate) == key:
                return i
        return None

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.actions), dtype=bool)
        mask[0] = True  # STOP always allowed
        if len(self.action_history) < self.max_excitations:
            for i, action in enumerate(self.actions[1:], start=1):
                if action_key(action) not in self.used_keys:
                    mask[i] = True
        return mask

    def step_gym(self, action_idx: int) -> tuple[np.ndarray, float, bool, dict]:
        action = self.actions[action_idx]
        reward = 0.0
        info: dict = {}

        if action["type"] == "stop" or len(self.action_history) >= self.max_excitations:
            if action["type"] != "stop":
                info["forced_stop"] = True
            vqe = run_vqe_on_circuit(
                self.H, self.num_qubits, self.hf_state, self.action_history
            )
            reward = self.hf_energy - vqe["energy"]
            info.update(
                {
                    "energy": vqe["energy"],
                    "description": describe_actions(self.action_history),
                    "actions": list(self.action_history),
                }
            )
            return self._state_vector(), reward, True, info

        self.action_history.append(action)
        self.used_keys.add(action_key(action))
        return self._state_vector(), reward, False, info
