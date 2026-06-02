# compile excitations to native gates + greedy backward pruning

from __future__ import annotations

import numpy as np
import pennylane as qml

from vqe_core import _BASIS_GATES, optimize_angles

ROT_GATES = {"RX", "RY", "RZ", "PhaseShift"}
CHEM_ACC = 1.6e-3


def _excitation_ops(actions, params):
    out = []
    for a, p in zip(actions, params):
        if a["type"] == "single":
            out.append(qml.SingleExcitation(p, wires=a["wires"]))
        else:
            out.append(qml.DoubleExcitation(p, wires=a["wires"]))
    return out


def adapt_to_raw_records(actions, params):
    return circuit_gates_to_raw_records(_excitation_ops(actions, params))


def circuit_gates_to_raw_records(gates):
    # input: pl ops or env gate dicts
    if gates and isinstance(gates[0], dict):
        ops = []
        for g in gates:
            t = g["type"]
            if t == "CNOT":
                ops.append(qml.CNOT(wires=[g["control"], g["target"]]))
            elif t == "single":
                ops.append(qml.SingleExcitation(g.get("theta", 0.0), wires=g["wires"]))
            elif t == "double":
                ops.append(qml.DoubleExcitation(g.get("theta", 0.0), wires=g["wires"]))
            else:
                ops.append(getattr(qml, t)(g.get("theta", 0.0), wires=g["wire"]))
    else:
        ops = list(gates)
    tape = qml.tape.QuantumScript(ops, [])
    (dec,), _ = qml.transforms.decompose([tape], gate_set=_BASIS_GATES)
    recs = []
    for op in dec.operations:
        if op.name == "GlobalPhase":
            continue
        param = float(op.parameters[0]) if op.name in ROT_GATES and op.parameters else None
        recs.append({"name": op.name, "wires": tuple(op.wires), "param": param})
    return recs


def count_cnots(records):
    return sum(1 for r in records if r["name"] == "CNOT")


def _optimized_energy(H, nq, hf, records, extra_restarts=1, maxiter=100):
    rot_ix = [i for i, r in enumerate(records) if r["param"] is not None]

    def circuit(x):
        qml.BasisState(hf, wires=range(nq))
        p = 0
        for r in records:
            if r["param"] is None:
                getattr(qml, r["name"])(wires=list(r["wires"]))
            else:
                getattr(qml, r["name"])(x[p], wires=list(r["wires"]))
                p += 1

    x0 = np.array([records[i]["param"] for i in rot_ix]) if rot_ix else None
    e, xs = optimize_angles(H, nq, circuit, len(rot_ix), x0=x0,
                            extra_restarts=extra_restarts, maxiter=maxiter)
    for j, i in enumerate(rot_ix):
        records[i]["param"] = float(xs[j])
    return e


def greedy_raw_prune(H, nq, hf, records, fci, chem_acc=CHEM_ACC,
                     extra_restarts=1, maxiter=100):
    records = [dict(r) for r in records]
    n_evals = 0
    changed = True
    while changed and records:
        changed = False
        best_e, best = np.inf, None
        for i in range(len(records)):
            trial = [dict(r) for r in records[:i] + records[i + 1:]]
            e = _optimized_energy(H, nq, hf, trial, extra_restarts=extra_restarts, maxiter=maxiter)
            n_evals += 1
            if e < best_e:
                best_e, best = e, trial
        if best is not None and (best_e - fci) < chem_acc:
            records = best
            changed = True
    final_e = _optimized_energy(H, nq, hf, records, extra_restarts=extra_restarts, maxiter=maxiter)
    return {"records": records, "energy": final_e, "error_vs_fci": final_e - fci,
            "cnots": count_cnots(records), "n_gates": len(records), "n_evals": n_evals}
