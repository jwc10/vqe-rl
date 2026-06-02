# step-by-step circuit builder: raw gates and/or Givens excitations

from __future__ import annotations

import numpy as np
import pennylane as qml

from vqe_core import compiled_resources, optimize_angles

PARAM_TYPES = ("RX", "RY", "RZ", "single", "double")


def build_raw_action_space(n_qubits):
    acts = [{"type": "stop"}]
    for q in range(n_qubits):
        for rot in ("RX", "RY", "RZ"):
            acts.append({"type": rot, "wire": q})
    for c in range(n_qubits):
        for t in range(n_qubits):
            if c != t:
                acts.append({"type": "CNOT", "control": c, "target": t})
    return acts


def build_givens_action_space(n_elec, n_qubits):
    singles, doubles = qml.qchem.excitations(n_elec, n_qubits)
    acts = [{"type": "stop"}]
    acts += [{"type": "single", "wires": list(w)} for w in singles]
    acts += [{"type": "double", "wires": list(w)} for w in doubles]
    return acts


def build_hybrid_action_space(n_elec, n_qubits):
    acts = [{"type": "stop", "_pool": "any"}]
    for a in build_givens_action_space(n_elec, n_qubits)[1:]:
        acts.append({**a, "_pool": "givens"})
    for a in build_raw_action_space(n_qubits)[1:]:
        acts.append({**a, "_pool": "raw"})
    return acts


def _has_angle(g):
    return g["type"] in PARAM_TYPES


class RawGateEnv:
    def __init__(self, config, max_gates=20, chem_acc=1.6e-3,
                 cnot_penalty=0.0, exact_tol=1e-6, number_penalty=0.0, order_k=0,
                 inner_restarts=1, inner_maxiter=100, gate_set="raw",
                 reward_mode="ostaszewski", max_givens_phase=6, min_raw_before_stop=1,
                 hybrid_simultaneous=False, raw_penalty_pre_chem=0.05, givens_bonus=0.15):
        self.config = config
        self.H = config["H"]
        self.num_qubits = config["num_qubits"]
        self.hf_state = config["hf_state"]
        self.hf_energy = config["hf_energy"]
        self.fci_energy = config["fci_energy"]
        self.max_gates = max_gates
        self.chem_acc = chem_acc
        self.exact_tol = exact_tol
        self.n_electrons = config.get("n_electrons")
        self.number_penalty = number_penalty
        self.target = chem_acc
        self.cnot_penalty = cnot_penalty
        self.order_k = order_k
        self.inner_restarts = inner_restarts
        self.inner_maxiter = inner_maxiter
        self.miss_penalty = True
        self.reward_mode = reward_mode
        self.max_givens_phase = max_givens_phase
        self.min_raw_before_stop = min_raw_before_stop
        self.hybrid_simultaneous = hybrid_simultaneous and gate_set == "hybrid"
        self.raw_penalty_pre_chem = raw_penalty_pre_chem
        self.givens_bonus = givens_bonus
        self.gate_set = gate_set

        if gate_set == "givens":
            self.actions = build_givens_action_space(self.n_electrons, self.num_qubits)
        elif gate_set == "hybrid":
            self.actions = build_hybrid_action_space(self.n_electrons, self.num_qubits)
        else:
            self.actions = build_raw_action_space(self.num_qubits)
        self.n_actions = len(self.actions)
        self.reset()

    def _place(self, g, theta=None):
        t = g["type"]
        if t == "CNOT":
            qml.CNOT(wires=[g["control"], g["target"]])
        elif t == "single":
            qml.SingleExcitation(theta, wires=g["wires"])
        elif t == "double":
            qml.DoubleExcitation(theta, wires=g["wires"])
        else:
            getattr(qml, t)(theta, wires=g["wire"])

    def _energy(self, gate_list):
        rot_ix = [i for i, g in enumerate(gate_list) if _has_angle(g)]

        def circuit(params):
            qml.BasisState(self.hf_state, wires=range(self.num_qubits))
            p = 0
            for g in gate_list:
                if _has_angle(g):
                    self._place(g, params[p])
                    p += 1
                else:
                    self._place(g)

        x0 = np.array([gate_list[i].get("theta", 0.0) for i in rot_ix]) if rot_ix else None
        e, xs = optimize_angles(
            self.H, self.num_qubits, circuit, len(rot_ix), x0=x0,
            extra_restarts=self.inner_restarts, maxiter=self.inner_maxiter)
        for j, i in enumerate(rot_ix):
            gate_list[i]["theta"] = float(xs[j])
        return e

    def _particle_number(self, gate_list):
        from vqe_core import _cached_device
        dev = _cached_device(self.num_qubits)

        @qml.qnode(dev)
        def zexp():
            qml.BasisState(self.hf_state, wires=range(self.num_qubits))
            for g in gate_list:
                self._place(g, g.get("theta", 0.0) if _has_angle(g) else None)
            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]

        z = np.asarray(zexp(), dtype=float)
        return float(np.sum((1.0 - z) / 2.0))

    def set_target(self, target, floor=None):
        if floor is None:
            floor = self.chem_acc
        self.target = max(floor, float(target))

    def reset(self):
        self.gates = []
        self.current_energy = self.hf_energy
        if self.gate_set == "hybrid":
            self.raw_unlocked = self.hybrid_simultaneous
        else:
            self.raw_unlocked = True
        return self._state_vector(), {}

    def _count_givens(self):
        return sum(1 for g in self.gates if g["type"] in ("single", "double"))

    def _count_raw(self):
        return sum(1 for g in self.gates if g["type"] in ("RX", "RY", "RZ", "CNOT"))

    def _unlock_raw_if_ready(self, gap):
        if self.hybrid_simultaneous:
            return False
        if self.gate_set == "hybrid" and not self.raw_unlocked and gap < self.chem_acc:
            self.raw_unlocked = True
            return True
        return False

    def _hit_target(self, gap):
        if gap >= self.target:
            return False
        if self.gate_set == "hybrid" and not self.hybrid_simultaneous:
            if not self.raw_unlocked:
                return False
            return self._count_raw() > 0
        return True

    def _hybrid_ok(self, gap):
        if self.hybrid_simultaneous:
            return gap < self.target
        return self.raw_unlocked and self._count_raw() > 0 and gap < self.target

    def _step_reward(self, prev_e, gate):
        denom = prev_e - self.fci_energy + 1e-12
        r = max((prev_e - self.current_energy) / denom, -1.0)
        pool = gate.get("_pool", "raw" if gate["type"] in ("RX", "RY", "RZ", "CNOT") else "givens")
        gap = self.current_energy - self.fci_energy
        if gate.get("type") == "CNOT":
            r -= self.cnot_penalty
        if self.hybrid_simultaneous and gap >= self.chem_acc:
            if pool == "raw":
                r -= self.raw_penalty_pre_chem
            elif pool == "givens":
                r += self.givens_bonus
        if self.number_penalty > 0.0 and self.n_electrons is not None:
            r -= self.number_penalty * abs(self._particle_number(self.gates) - self.n_electrons)
        return r

    def _end_reward(self, ok):
        if self.reward_mode == "ostaszewski":
            return 5.0 if ok else -5.0
        frac = (self.hf_energy - self.current_energy) / (self.hf_energy - self.fci_energy + 1e-12)
        return 0.0 if ok else (frac - 1.0 if self.miss_penalty else 0.0)

    def _state_vector(self):
        counts = np.zeros(self.n_actions, dtype=np.float64)
        for g in self.gates:
            counts[self._action_index(g)] += 1.0
        frac_corr = (self.hf_energy - self.current_energy) / (
            self.hf_energy - self.fci_energy + 1e-12)
        depth_frac = len(self.gates) / self.max_gates
        parts = [counts, [depth_frac, frac_corr, float(self.raw_unlocked)]]
        if self.order_k > 0:
            window = np.zeros(self.order_k * self.n_actions, dtype=np.float64)
            for slot, g in enumerate(self.gates[-self.order_k:][::-1]):
                window[slot * self.n_actions + self._action_index(g)] = 1.0
            parts.append(window)
        return np.concatenate(parts)

    @property
    def state_dim(self):
        return self.n_actions + 3 + self.order_k * self.n_actions

    def _action_index(self, gate):
        for i, a in enumerate(self.actions):
            if a["type"] != gate["type"]:
                continue
            if a["type"] == "CNOT":
                if a["control"] == gate["control"] and a["target"] == gate["target"]:
                    return i
            elif a["type"] in ("single", "double"):
                if list(a["wires"]) == list(gate["wires"]):
                    return i
            elif a["type"] == "stop":
                return i
            elif a["wire"] == gate["wire"]:
                return i
        raise ValueError(gate)

    def valid_action_mask(self):
        mask = np.ones(self.n_actions, dtype=bool)
        mask[0] = True
        if len(self.gates) >= self.max_gates:
            mask[:] = False
            mask[0] = True
            return mask
        if self.gate_set == "hybrid" and self.hybrid_simultaneous and not self.gates:
            mask[0] = False
        if self.gate_set == "hybrid" and not self.hybrid_simultaneous:
            if not self.raw_unlocked:
                for i, a in enumerate(self.actions):
                    if a.get("_pool") not in ("givens", "any"):
                        mask[i] = False
                if self._count_givens() >= self.max_givens_phase:
                    mask[:] = False
                    mask[0] = True
            elif self._count_raw() < self.min_raw_before_stop:
                mask[0] = False
        if self.gates:
            mask[self._action_index(self.gates[-1])] = False
        return mask

    def step_gym(self, action_idx):
        action = self.actions[action_idx]
        info = {}

        if action["type"] == "stop" or len(self.gates) >= self.max_gates:
            return self._finish(info)

        gate = dict(action)
        if _has_angle(gate):
            gate["theta"] = 0.0
        self.gates.append(gate)

        prev = self.current_energy
        self.current_energy = self._energy(self.gates)
        gap = self.current_energy - self.fci_energy

        if self._unlock_raw_if_ready(gap):
            reward = self._step_reward(prev, gate) + 1.0
        else:
            reward = self._step_reward(prev, gate)

        done = False
        if self._hit_target(gap):
            reward = self._end_reward(True)
            done = True
            info["reached_target"] = True

        if done or len(self.gates) >= self.max_gates:
            info.update(self._final_info())
        return self._state_vector(), reward, done, info

    def _finish(self, info):
        info.update(self._final_info())
        gap = info.get("error_vs_fci", np.inf)
        if self.gate_set == "hybrid":
            ok = self._hybrid_ok(gap)
        elif self.target < self.chem_acc:
            ok = gap < self.target
        else:
            ok = info.get("within_chem_acc", False) if self.target >= self.chem_acc else info.get("within_exact", False)
        return self._state_vector(), self._end_reward(ok), True, info

    def _final_info(self):
        ops = []
        for g in self.gates:
            t = g["type"]
            if t == "CNOT":
                ops.append(qml.CNOT(wires=[g["control"], g["target"]]))
            elif t == "single":
                ops.append(qml.SingleExcitation(g.get("theta", 0.0), wires=g["wires"]))
            elif t == "double":
                ops.append(qml.DoubleExcitation(g.get("theta", 0.0), wires=g["wires"]))
            else:
                ops.append(getattr(qml, t)(g.get("theta", 0.0), wires=g["wire"]))
        res = compiled_resources(ops)
        gap = self.current_energy - self.fci_energy
        return {
            "energy": self.current_energy,
            "error_vs_fci": gap,
            "n_gates": len(self.gates),
            "cnots": res["cnot_count"],
            "cnot_depth": res["cnot_depth"],
            "within_chem_acc": gap < self.chem_acc,
            "within_exact": gap < self.exact_tol,
            "gates": list(self.gates),
        }


if __name__ == "__main__":
    from rl_env import make_h2_config
    env = RawGateEnv(make_h2_config(), max_gates=10)
    print(f"actions={env.n_actions} state_dim={env.state_dim}")
    rng = np.random.default_rng(0)
    state, _ = env.reset()
    tot = 0.0
    for _ in range(10):
        a = rng.choice(np.flatnonzero(env.valid_action_mask()))
        state, r, done, info = env.step_gym(int(a))
        tot += r
        if done:
            break
    print(f"reward={tot:.2f} cnots={info.get('cnots')} err={info.get('error_vs_fci'):+.2e}")
