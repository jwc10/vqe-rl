# Record greedy removal trajectories for behavioral-cloning warm-start.

from __future__ import annotations

import numpy as np

from prune_env import RawGatePruneEnv
from raw_prune import _optimized_energy, count_cnots


def greedy_prune_with_trace(cfg, start_records, fci, chem_acc=1.6e-3,
                            order_k=4, inner_maxiter=60, inner_restarts=0,
                            verbose=False):
    # greedy backward elimination, emitting (obs, action, mask) steps for the prune env
    H, nq, hf = cfg["H"], cfg["num_qubits"], cfg["hf_state"]
    env = RawGatePruneEnv(
        cfg, start_records, target=chem_acc, strict=False, order_k=order_k,
        inner_restarts=inner_restarts, inner_maxiter=inner_maxiter,
    )
    env.reset()
    trace = []
    n_evals = 0
    round_no = 0

    while env.alive.any():
        round_no += 1
        best_e, best_idx = np.inf, None
        alive_ix = [i for i in range(env.n) if env.alive[i]]
        for i in alive_ix:
            trial_alive = env.alive.copy()
            trial_alive[i] = False
            trial = [dict(env.start[j]) for j in range(env.n) if trial_alive[j]]
            e = _optimized_energy(H, nq, hf, trial,
                                  extra_restarts=inner_restarts, maxiter=inner_maxiter)
            n_evals += 1
            if e < best_e:
                best_e, best_idx = e, i

        err = best_e - fci
        if best_idx is None or err >= chem_acc:
            if verbose:
                print(f"  trace stop round {round_no}: best err={err*1e3:.4f} mHa", flush=True)
            break

        obs = env._obs().copy()
        mask = env.valid_action_mask().copy()
        trace.append({
            "obs": obs,
            "action": int(best_idx),
            "mask": mask,
            "round": round_no,
            "cnots_after": None,
        })
        _, _, done, info = env.step_gym(best_idx)
        if not info:
            info = env._done_info()
        trace[-1]["cnots_after"] = info.get("cnots")
        if verbose:
            print(f"  trace round {round_no}: remove idx {best_idx}, now "
                  f"{info['cnots']} CNOTs, err={info['error_vs_fci']*1e3:.4f} mHa",
                  flush=True)
        if done:
            break

    info = {
        "energy": env.energy,
        "error_vs_fci": env.energy - fci,
        "cnots": count_cnots(env._alive()),
        "n_gates": len(env._alive()),
        "within_target": (env.energy - fci) < chem_acc,
    }
    return {
        "trace": trace,
        "final": info,
        "n_evals": n_evals,
        "records": env._alive(),
    }
