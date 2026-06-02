# Poster summary — RL for short compiled VQE circuits

**Metric:** compiled **CNOT count** at **chemical accuracy** (1.6 mHa vs FCI), unless noted.  
**Stack:** PennyLane + `lightning.qubit`, PPO actor–critic, dense per-step rewards (Ostaszewski-style ±5 terminal bonus).

---

## What we built (three strategies)

1. **Raw-gate RL from scratch** — RX/RY/RZ/CNOT on top of HF; agent builds and optimizes the circuit step by step (`train_raw.py`, `raw_gate_env.py`).
2. **ADAPT → compile → prune** — run ADAPT (or a single double), decompose to native gates, then **greedy** or **RL** gate removal (`raw_prune.py`, `prune_env.py`).
3. **Hybrid** — phase 1: Givens excitations (± raw); phase 2: compile + RL prune. Variants:
   - **Chained:** Givens only until chem acc, then raw unlock / compile / prune (`--quick20`).
   - **Simultaneous:** Givens + raw in one env from step 0.
   - **Simul+prune (poster LiH headline):** simultaneous build → compile → RL prune (`--hybrid-simul-prune --medium`).

**Curriculum:** moving energy-gap threshold (Ostaszewski feedback), not fixed chem acc from step 1 (`curriculum.py`).

---

## H₂ (4 qubits) — main win

| Method | CNOTs | vs FCI | Source |
|--------|-------|--------|--------|
| ADAPT / one double | 14 | exact | `results/compare_raw_h2.json` |
| ADAPT compile + greedy/RL raw prune | 7 | exact | same |
| Chained / greedy from 1-double | 7 | exact | `results/fair_greedy_compare.json` |
| Hybrid simultaneous (medium) | 5–6 | exact | logs |
| **Simul+prune (medium)** | **4** | exact (~0 mHa) | `results/final_h2.json` |
| **Raw-gate RL from scratch** | **3** | exact | `results/compare_raw_h2.json` |

**Takeaway:** Only **raw from scratch** reaches **3 CNOTs**. Hybrid simul+prune (**4**) beats ADAPT→prune (**7**) but not raw search.

---

## LiH (2e, 4o) — 8 qubits

| Method | CNOTs | Chem acc? | Exact FCI? | Notes |
|--------|-------|-----------|------------|--------|
| HF | 0 | no (~2.9 mHa gap) | no | baseline |
| **One double (“14-CNOT floor”)** | 14 | yes (~1.47 mHa) | no | same as ADAPT step 1 |
| ADAPT full | 46 | yes | yes | overkill |
| Greedy raw-prune from **ADAPT** compile | 23 | yes | no | ~hours CPU |
| Greedy raw-prune from **1-double** compile | **7** | yes | no | `fair_greedy_compare.json` |
| Chained hybrid (`--quick20`) | **7** | yes | no | ~20 s, reliable |
| **Simul+prune medium (killed @ upd 7)** | **7** | yes | no | **`results/lih4_simul_prune_partial.json`** |
| Simul+prune reruns (seed sweep) | 8–11 | yes | no | seed 42 → 8 CNOTs |
| RL raw from scratch | — | fail | — | STOP collapse |
| RL Givens only | 14 | yes | no | rediscovers one-double |
| RL prune from ADAPT (long run, killed) | 12 logged | ? | no | `lih4_interrupted_runs.json` |

**Takeaway at chem acc:** excitation floor **14**; compress to **7** via greedy-from-one-double, chained hybrid, or **simul+prune** (same ~1.47 mHa energy). **Not** the same as beating full ADAPT (46→14 is trivial).

---

## What “14-CNOT baseline” means (say this on the poster)

- HF + **one** `DoubleExcitation` → **14 compiled CNOTs**, ~**1.5 mHa** error (chem acc, **not** exact FCI).
- ADAPT’s **first** step, brute-force best double, RL Givens, and greedy excitation prune all land here.
- **≠** full ADAPT (**46 CNOTs**, exact FCI).

---

## Honest negatives

- Raw RL **does not scale** to 8q LiH (exploration / STOP collapse).
- Simul+prune **7 CNOTs** was **stochastic** (update 3–7 of a longer run); reruns often **8–14**; keep the salvaged partial result for the poster.
- Long LiH jobs can **hang** when simultaneous build compiles huge circuits → use `--updates 8`, `max_compile_gates`, kill after update 7 if needed.

---

## Commands (reproduce headline numbers)

```bash
# H2 raw RL → 3 CNOTs
python compare_raw.py --molecule H2

# LiH chained hybrid → 7 CNOTs (~20 s)
python run_final_experiments.py --quick20 --phase hybrid --molecule LiH4 --target chem

# LiH simul+prune (poster method; stop at 8 updates)
python run_final_experiments.py --medium --hybrid-simul-prune --phase hybrid --molecule LiH4 --target chem --seed 42

# Fair greedy table (7 CNOTs from 1-double on both molecules)
python quick_fair_compare.py
```

---

## Key files & figures

| File | Use |
|------|-----|
| `results/compare_raw_h2.json` | H₂ table |
| `results/lih4_simul_prune_partial.json` | LiH simul+prune **7 CNOT** result |
| `results/fair_greedy_compare.json` | Greedy 7 CNOT baseline |
| `results/lih4_simul_prune_seed_sweep.log` | Rerun seeds 0→9, 2→11, 42→8 |
| **`results/poster/*.png`** | **Poster-ready bar charts, learning curves, circuits** |
| `python make_poster_figures.py` | Regenerate poster figures |
| `POSTER_CHECKLIST.md` | CS224R rubric alignment |
| `results/raw_gate_*.png`, `results/reinforce_returns.png` | training curves |

---

## Suggested poster bullets

1. **Problem:** minimize **compiled CNOTs** at **chemical accuracy** for small molecules.
2. **H₂:** unconstrained raw RL finds **3-CNOT** exact solution; hybrid simul+prune gets **4**.
3. **LiH:** symmetry-aware build + RL prune hits **7 CNOTs** @ chem acc (matches greedy from one double); raw-from-scratch fails on 8q.
4. **Compare** three pipelines: scratch raw, ADAPT→prune, hybrid/simul+prune — same metric, different search spaces.
