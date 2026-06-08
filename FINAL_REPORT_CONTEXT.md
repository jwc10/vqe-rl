# CS224R Final Project — Complete Results & Context Document

**Project:** Reinforcement learning for variational quantum eigensolver (VQE) **circuit compression**  
**Molecules:** H₂, LiH (active spaces 6q / 8q / 10q)  
**Primary metric:** **Compiled native CNOT count** at a fixed accuracy target (not excitation count)  
**Generated:** 2026-06-08 (consolidates all local + Modal runs through overnight campaign)

This document is the single source of truth for writing the final report. It explains **what we did**, **what is comparable to what**, and **every major number** with provenance. Figures live in `results/final_report/`; machine-readable aggregation in `results/final_report/master_results.json`.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [What “fair comparison” means](#2-what-fair-comparison-means)
3. [Molecules, Hamiltonians, and accuracy targets](#3-molecules-hamiltonians-and-accuracy-targets)
4. [Methods and pipelines](#4-methods-and-pipelines)
5. [RL algorithm details](#5-rl-algorithm-details)
6. [Results: H₂](#6-results-h₂)
7. [Results: LiH 8q (2e, 4o) — main campaign](#7-results-lih-8q-2e-4o--main-campaign)
8. [Results: LiH 6q (2e, 3o)](#8-results-lih-6q-2e-3o)
9. [Results: LiH 10q (2e, 5o)](#9-results-lih-10q-2e-5o)
10. [Earlier experiments (pre-campaign)](#10-earlier-experiments-pre-campaign)
11. [Modal overnight campaign (Jun 2026)](#11-modal-overnight-campaign-jun-2026)
12. [What failed and why](#12-what-failed-and-why)
13. [Figures for the final report](#13-figures-for-the-final-report)
14. [Recommended report narrative](#14-recommended-report-narrative)
15. [Limitations and honest caveats](#15-limitations-and-honest-caveats)
16. [File index](#16-file-index)
17. [Commands to reproduce](#17-commands-to-reproduce)

---

## 1. Executive summary

### One-sentence thesis

Use RL over **compiled native gates** (RX/RY/RZ/CNOT) to minimize **CNOT count** at chemical accuracy (1.6 mHa) or near-exact FCI, comparing **from-scratch construction**, **ADAPT→compile→prune**, and **hybrid Givens→prune** — always against **greedy pruning baselines on the same starting circuit**.

### Headline findings

| Claim | Strength | Evidence |
|-------|----------|----------|
| **H₂ raw-gate RL from scratch reaches 3 CNOTs (exact FCI)** | ✅ Verified | `results/compare_raw_h2.json` |
| **LiH 8q: RL beats greedy on same ADAPT start @ chem acc** | ✅ Verified (GPU + local) | Greedy **23** → RL **11–20** CNOTs (seeds); local smoke **12** |
| **LiH 8q: RL ties greedy on 1-double start @ chem acc (7 CNOTs)** | ✅ Verified (3 Modal seeds) | Cannot beat global floor from this start |
| **LiH 8q: RL ties greedy on exact FCI from ADAPT (25 CNOTs)** | ✅ Verified | `results/beat_greedy/exact_compare.json` |
| **LiH 6q: RL may dramatically beat greedy (3–5 vs 16)** | ⚠️ Log-only partial | Modal timeout; **not** on volume — treat as promising, not primary |
| **LiH 10q: greedy infeasible; RL scalability inconclusive** | ⚠️ Incomplete | Greedy **60** CNOTs in ~6h CPU; Modal RL timed out at 7h |
| **LiH raw RL from scratch on 8q+** | ❌ Failed in budget | No chem-acc circuit found |
| **Hybrid simul+prune** | ❌ Broken / unreproducible | State-dim bug; 7 CNOTs once, retries failed |

### The most important insight for the report

There are **two different “wins”** that must not be conflated:

1. **Global optimum (best start):** LiH chem-acc floor is **7 CNOTs** from **1-double compile** (greedy and RL **tie**). No method in this project beat 7 CNOTs @ chem acc on 8q.
2. **Same-start improvement:** From the **hard ADAPT compile** (96 gates, 46 CNOTs start), greedy prune stalls at **23 CNOTs** @ chem acc, but RL prune reaches **11–20 CNOTs** — a real **apples-to-apples** win.

Comparing “RL 12 CNOTs” to “greedy 7 CNOTs” is **not** fair — different starting circuits.

---

## 2. What “fair comparison” means

### The fair-pair protocol (`lih_campaign.py`)

Each **pair** fixes:

| Field | Meaning |
|-------|---------|
| **Start circuit** | Exact same gate list before pruning |
| **Target** | `chem_acc` (1.6 mHa) or `exact_FCI` (< 1 µHa) |
| **Greedy baseline** | `greedy_raw_prune()` on that start, cached when possible |
| **RL** | PPO prune with behavioral cloning on greedy removal trace, same `_optimized_energy` inner loop |
| **Metric** | `count_cnots()` after each candidate removal + re-optimization |

**Pairs defined in campaign:**

| Pair ID | Start | Greedy floor (8q) | Question |
|---------|-------|-------------------|----------|
| `1double_chem` | Single double excitation → compile (28 gates, 14 CNOTs start) | **7** CNOTs | Can RL beat greedy from the **best** chem-acc ansatz? |
| `adapt_chem` | Full ADAPT-VQE → compile (96 gates, 46 CNOTs start) | **23** CNOTs | Can RL find **shorter** chem-acc circuits greedy misses? |
| `adapt_exact` | Same ADAPT start | **25** CNOTs | Can RL beat greedy at **exact** FCI? |

### Minor asymmetries (disclose in report)

- RL eval uses `extra_restarts=1` in some configs; greedy uses `0`.
- Greedy `maxiter` 40 vs 50 can shift 8q greedy ADAPT chem between **22** and **23** CNOTs (we cite **23** from cached baseline).
- Modal GPU (lightning.gpu) vs local CPU — same physics, different wall-clock; energies/CNOTs should match if optimization converges.
- Partial runs killed by timeout may have found good circuits mid-training but not saved final `fair_comparison.json`.

### What is NOT a fair comparison

| Bad comparison | Why |
|----------------|-----|
| RL 12 CNOTs (ADAPT start) vs greedy 7 (1-double start) | Different initial circuits |
| ADAPT 46 CNOTs vs pruned 23 | Before/after prune, not method vs method |
| Excitation count vs CNOT count | Compilation blows up excitations |
| Hybrid 8 CNOTs vs greedy 7 | Different pipeline (build+prune vs prune-only) unless same start stated |

---

## 3. Molecules, Hamiltonians, and accuracy targets

### Active spaces

| Label | Electrons | Orbitals | Qubits | FCI (approx) | Use |
|-------|-----------|----------|--------|--------------|-----|
| H₂ | 2 | 2 | 4 | −1.1373 Ha | Proof-of-concept |
| LiH 6q | 2 | 3 | 6 | −7.6362 Ha | Smaller scale / curriculum |
| LiH 8q | 2 | 4 | 8 | −7.6375 Ha | **Main campaign** |
| LiH 10q | 2 | 5 | 10 | −7.6521 Ha | Scalability stress test |

Built via `rl_env.make_lih_config()` / `make_h2_config()` with PennyLane + OpenFermion Jordan–Wigner.

### Accuracy targets

| Name | Threshold | Typical use |
|------|-----------|-------------|
| **Chemical accuracy** | \|E − E_FCI\| < **1.6 mHa** (1.6×10⁻³ Ha) | NISQ-relevant target |
| **Exact FCI** | \|E − E_FCI\| < **10⁻⁶ Ha** (~0.001 mHa) | Stronger constraint; fewer valid pruned circuits |

### The LiH chem-acc plateau (~1.47 mHa)

Many distinct compiled circuits (7, 12, 14, 23 CNOTs on 8q) share **essentially the same energy** ≈ 1.4677 mHa above FCI. This is **not** a bug — it reflects that chem acc is a **loose** target and many structures sit on the same correlation plateau. **Exact FCI** discriminates more (needs 25 CNOTs from ADAPT start on 8q).

---

## 4. Methods and pipelines

### Pipeline A: Excitation-level RL (early project)

- **Files:** `train_lih.py`, `train_reinforce.py`, `compare_lih.py`
- **Action space:** Pick singles/doubles + STOP (UCCSD pool)
- **Metric in training:** excitation count + energy (λ sweep → `results/lih_pareto.png`)
- **Outcome:** Useful for Pareto intuition; pivoted to **compiled CNOTs** for main story

### Pipeline B: ADAPT-VQE baseline

- **File:** `adapt.py`
- Greedy gradient-screened operator selection until convergence
- 8q: 5 excitations → 46 CNOTs compiled; 6q: 4 excitations → 32 CNOTs; 10q: 8 → 88 CNOTs

### Pipeline C: Compile → greedy raw-gate prune

- **Files:** `raw_prune.py`, `quick_fair_compare.py`, `greedy_adapt_exact_only.py`
- Convert excitations to native gate records (`adapt_to_raw_records`)
- Greedily try removing each gate; re-optimize all rotation angles (L-BFGS-B + adjoint)
- **Strong baseline** — slow but deterministic

### Pipeline D: Compile → RL raw-gate prune (main LiH story)

- **Files:** `prune_env.py`, `prune_trainer.py`, `lih_campaign.py`
- PPO removes gates (any-gate removal action space)
- Behavioral cloning warm-start from greedy removal **trace**
- Same inner optimizer as greedy (`_optimized_energy` in `raw_prune.py`)

### Pipeline E: Raw-gate RL from scratch

- **Files:** `train_raw.py`, `raw_gate_env.py`
- Places RX/RY/RZ/CNOT (optional Givens) from empty circuit
- **Works on H₂** (3 CNOTs); **fails on LiH 8q+** in our budget

### Pipeline F: Hybrid Givens → compile → prune

- **Files:** `train_raw.py` (phases: chained, simultaneous, simul+prune)
- Build with structured Givens, compile, then prune
- **Simul+prune:** found 7 CNOTs @ chem once; retries 8–14; medium preset hung; Modal hybrid **crashed** (state dim bug)

---

## 5. RL algorithm details

### PPO actor-critic (`ppo.py`)

- Shared MLP, masked discrete actions, value baseline
- Typical LiH prune: hidden 512–768, 5–6 PPO epochs, 24–40 episodes/update

### State representation (order encoding, not sinusoidal PE)

- Bag-of-gate counts + depth fraction + correlation-energy fraction
- **Order encoding:** one-hot of last `order_k` gate indices (Ostaszewski-style; not full Transformer)

### Rewards (Ostaszewski et al., NeurIPS 2021)

| Component | Location | Formula / value |
|-----------|----------|-----------------|
| Dense step reward | `raw_gate_env._step_reward` | Fractional progress toward FCI |
| Terminal bonus | `raw_gate_env`, `prune_env` | **+5** if within target, **−5** otherwise |
| Moving threshold | `curriculum.MovingThreshold` | Tightens target as agent improves (hybrid runs) |

### Inner VQE optimizer (critical for all pipelines)

- `vqe_core.py` / `raw_prune._optimized_energy`
- L-BFGS-B, PennyLane adjoint gradients, angle warm-start
- Modal: `lightning.gpu` via `pick_vqe_device()` and `VQE_DEVICE` env

### Training presets (`lih_campaign_config.py`)

| Preset | Updates | Use |
|--------|---------|-----|
| `smoke` | 15 | Local CPU validation |
| `modal_5h` | 100 (1-double) + 55 (ADAPT) | Modal overnight |
| `MODAL_10Q_RL` | 20, no greedy BC | 10q scalability (greedy BC infeasible) |

---

## 6. Results: H₂

**Source:** `results/compare_raw_h2.json`, `results/final_h2.json`

| Method | CNOTs | Error vs FCI | Notes |
|--------|-------|--------------|-------|
| ADAPT (excitations) | 14 | ~0 | 1 excitation |
| ADAPT + greedy excitation prune | 14 | ~0 | No reduction |
| ADAPT compile + greedy raw prune | 7 | ~0 | Halves CNOTs |
| ADAPT compile + RL raw prune | 7 | ~0 | Ties greedy |
| **Raw-gate RL from scratch** | **3** | ~0 | **Best result; exact FCI** |

**Figure:** `results/final_report/fig01_h2_cnot_comparison.png`

**Interpretation:** H₂ is small enough that RL from scratch solves the combinatorial gate-ordering problem; prune-RL and greedy agree. This validates the pipeline before LiH scaling.

---

## 7. Results: LiH 8q (2e, 4o) — main campaign

### 7.1 Greedy baselines (cached)

**Source:** `results/fair_greedy_compare.json`, `results/lih_campaign/baselines.json`

| Method | Start CNOTs | Final CNOTs | Err (mHa) | Exact? |
|--------|-------------|-------------|-----------|--------|
| 1-double (no prune) | 14 | 14 | 1.468 | No |
| Greedy prune from 1-double | 14 | **7** | 1.468 | No |
| Full ADAPT compile (no prune) | 46 | 46 | ~0 | Yes |
| Greedy prune from ADAPT @ chem | 46 | **23** | 1.468 | No |
| Greedy prune from ADAPT @ exact | 46 | **25** | 0.0008 | Yes |

ADAPT-k greedy sweep (chem): k=1→7, k=2→13, k=3→19, k=4→23 CNOTs — more ADAPT steps → more pruning headroom before plateau.

### 7.2 Fair pair: `1double_chem` (same 1-double start)

| Run | Device | Seed | RL CNOTs | Greedy | Verdict |
|-----|--------|------|----------|--------|---------|
| Local smoke | CPU | 0 | 7 | 7 | Tie |
| Modal | H100 | 0 | 7 | 7 | Tie |
| Modal | H100 | 1 | 7 | 7 | Tie |
| Modal | H100 | 2 | 7 | 7 | Tie |

**Conclusion:** The **7-CNOT chem-acc floor** is reachable by both methods; RL does not improve on the best single-double ansatz. Beating 7 requires a **different physics** target (exact FCI) or a different start — pruning 1-double cannot reach exact (14 CNOTs, inf error if pushed).

**Figure:** `results/final_report/fig06_lih8q_1double_seeds.png`

### 7.3 Fair pair: `adapt_chem` (same full ADAPT start) — **primary positive result**

Greedy floor: **23 CNOTs** @ 1.468 mHa (same plateau).

| Run | Device | Seed | Updates | RL CNOTs | Err (mHa) | Status |
|-----|--------|------|---------|----------|-----------|--------|
| Local smoke | CPU | 0 | 15/15 | **12** | 1.460 | ✅ Complete, verified |
| Modal adapt_chem | A10G | 0 | 55/55 | **20** | 1.453 | ✅ Volume |
| Modal bundle | H100 | 2 | 55/55 | **11** | 1.460 | ✅ Volume |
| Modal bundle | H100 | 1 | 15/55 | **10** | 1.360 | ⚠️ Timeout partial (logs only) |

**Verified range:** RL **11–20 CNOTs** vs greedy **23** — all seeds that completed beat greedy.

**Learning dynamics:** RL often finds improvements early (e.g. seed 2 hits 11 CNOTs by update 3); seed 0 on A10G progresses 23→21→20 over ~40 updates.

**Figures:** `fig02_lih8q_fair_comparison.png`, `fig03_lih8q_adapt_chem_learning.png`, `fig05_lih8q_energy_cnot_pareto.png`

**Important:** Local 12-CNOT result was initially “unverified”; it is now corroborated by GPU runs (11, 20). Use **“RL achieves 11–20 CNOTs vs greedy 23”** as the conservative claim; cite **11** (best complete GPU seed) and **12** (local) as replicates.

### 7.4 Fair pair: `adapt_exact` (same ADAPT start, exact FCI)

**Source:** `results/beat_greedy/exact_compare.json`

| Method | CNOTs | Err (mHa) | Status |
|--------|-------|-----------|--------|
| Greedy raw prune | **25** | 0.000796 | Complete (~20 min) |
| RL prune | **25** | 0.0008 | Partial (14/40 upd, killed) |

**Conclusion:** **Tie at 25 CNOTs** — RL does not beat greedy on exact FCI from ADAPT. The pruning frontier at exact accuracy is tight; both methods find the same minimum.

### 7.5 Non-fair but informative points

| Method | CNOTs | Target | Note |
|--------|-------|--------|------|
| Givens + order from scratch | 14 | chem | Same as 1-double; not better than 7 with prune |
| Hybrid simul+prune | 7 | chem | Once; same energy as greedy-7; not reproducible |
| RL prune ADAPT-2 only | 13 | chem | `beat_greedy/hunt_report.json` |

---

## 8. Results: LiH 6q (2e, 3o)

### Greedy baselines

**Source:** `results/lih_campaign/baselines_6q.json`, CPU timing run

| Method | CNOTs | Err (mHa) | Wall-clock |
|--------|-------|-----------|------------|
| ADAPT start | 32 | ~0 | — |
| Greedy 1-double prune | **6** | 1.554 | fast |
| Greedy ADAPT prune @ chem | **16** | 1.554 | **514 s (~8.6 min)** |

### Modal `lih6q` campaign

| Job | Result | Status |
|-----|--------|--------|
| `1double_chem` s0 | RL **6** = greedy **6** | ✅ Volume |
| `adapt_chem` s0 | Best **3** CNOTs @ upd 35/55 | ⚠️ Timeout, logs only |
| `adapt_chem` s1 | Best **5** CNOTs @ upd 35/55 | ⚠️ Timeout, logs only |

**Interpretation:** If the partial 3–5 CNOT results reproduce, 6q would be a dramatic same-start win (greedy **16**). **Do not lead the report with this** until re-run or volume checkpoint confirms — cite as “preliminary partial GPU evidence.”

**Figure:** `fig04_multiscale_comparison.png` (includes 6q with asterisk on partial)

---

## 9. Results: LiH 10q (2e, 5o)

### Greedy baseline (CPU timing)

| Start | Greedy result | Time |
|-------|---------------|------|
| 180 gates, 88 CNOTs (ADAPT) | **60 CNOTs** @ 1.45 mHa chem | **21,361 s (~5.9 h)** |

Greedy ADAPT→prune is **prohibitively expensive** at 10q — motivates RL-only exploration.

### RL scalability probe (Modal `lih10q`)

- Config: 20 updates, no greedy BC, `lightning.gpu`
- **Outcome:** 7h timeout before first logged PPO update
- **No result on volume**

### Local smoke (early)

- 2-update CPU smoke reportedly reached **55 CNOTs** @ chem in ~3h (from conversation logs; no JSON in repo root — treat cautiously)

**Report framing:** “At 10q, greedy pruning required ~6h CPU for 60 CNOTs; RL training did not complete within Modal budget, leaving scalability **open**.”

---

## 10. Earlier experiments (pre-campaign)

Chronological arc (see also `POSTER_SESSION_SUMMARY.md`):

| Phase | What we tried | Outcome |
|-------|---------------|---------|
| Excitation PPO | `train_lih.py` λ-sweep | Pareto plot; pivot to CNOTs |
| LiH raw ablations | `lih_raw_experiment.py` bag/order/Givens | 8q from scratch failed |
| Beat-greedy hunt | `beat_greedy_lih.py` | 7 tie, 25 exact tie, ADAPT-2 → 13 |
| Simul+prune | `run_final_experiments.py` | 7 once; retries 8–14 |
| Exact head-to-head | `greedy_adapt_exact_only.py` | RL 25 = greedy 25 |
| Fair compare script | `quick_fair_compare.py` | Greedy 7/23 cached |
| Poster figures | `make_poster_figures.py` | `results/poster/` |

---

## 11. Modal overnight campaign (Jun 2026)

### Infrastructure

- **Platform:** Modal, `modal_lih.py`
- **GPUs:** A10G (single adapt_chem s0), H100 (bundle, lih6q, lih10q)
- **Timeout:** 7h (25200s) per function
- **Volume:** `vqe-rl-results` → downloaded to `results/lih_campaign/lih_campaign/`

### Jobs launched

| Entrypoint | Content | Outcome |
|------------|---------|---------|
| `modal run modal_lih.py` (adapt_chem s0) | 55 upd A10G | ✅ **20 vs 23** |
| `modal run modal_lih.py::bundle` | 1double s0-2 + adapt s1-2 | ⚠️ exit 1; 1double ✅, adapt s2 ✅ **11**, s1 timeout |
| `modal run modal_lih.py::beef` | heavy 1double + hybrid | ❌ hybrid import error; heavy stopped |
| `modal run modal_lih.py::hybrid_only` | hybrid s0-2 | ❌ state-dim 641 vs 513 |
| `modal run modal_lih.py::lih6q` | 6q fair pairs | ⚠️ 1double ✅; adapt timeout |
| `modal run modal_lih.py::lih10q` | 10q RL-only | ❌ 7h timeout |

### Completed Modal RL runs (volume-verified)

**Figure:** `fig07_modal_campaign_table.png`

| System | Pair | Seed | RL CNOTs | vs Greedy |
|--------|------|------|----------|-----------|
| 8q | 1double_chem | 0,1,2 | 7 | Tie |
| 8q | adapt_chem | 0 | 20 | Beat 23 |
| 8q | adapt_chem | 2 | 11 | Beat 23 |
| 6q | 1double_chem | 0 | 6 | Tie |

---

## 12. What failed and why

| Failure | Root cause | Relaunch? |
|---------|------------|-----------|
| Hybrid simul+prune | `RuntimeError` mat1 (1×641) vs (513×768) — prune net sized for initial circuit, not growing hybrid | Needs code fix (pad to max gates) |
| Heavy 1-double 500 upd | Low ROI; won't beat 7 floor | No |
| LiH raw from scratch 8q | Action space + budget; no chem circuit | No |
| 10q Modal RL | VQE per step too slow at 180 gates | Needs lighter preset or fewer episodes |
| adapt_chem s1 timeout | 6h limit during BC/training | Optional retry |
| Greedy “looked stuck” | Silent O(n²) evals per round | Fixed with `verbose=True` |

---

## 13. Figures for the final report

All in `results/final_report/`:

| File | Use in report |
|------|---------------|
| `fig01_h2_cnot_comparison.png` | Hero: H₂ 3 vs 7 vs 14 |
| `fig02_lih8q_fair_comparison.png` | Main LiH fair pairs bar chart |
| `fig03_lih8q_adapt_chem_learning.png` | RL learning curves vs greedy 23 line |
| `fig04_multiscale_comparison.png` | 6q/8q/10q scalability |
| `fig05_lih8q_energy_cnot_pareto.png` | Energy–CNOT tradeoff |
| `fig06_lih8q_1double_seeds.png` | Modal seed reproducibility at floor |
| `fig07_modal_campaign_table.png` | Appendix table of GPU runs |

Legacy poster assets: `results/poster/` (may predate GPU campaign — prefer `final_report/` figures).

**Regenerate:**

```bash
python consolidate_results.py
python make_final_report_figures.py
```

---

## 14. Recommended report narrative

### Abstract-style paragraph

We train PPO agents to minimize **compiled CNOT count** in variational quantum circuits for H₂ and LiH, using dense Ostaszewski-style rewards and greedy-prune baselines on **identical starting circuits**. On H₂, raw-gate RL discovers a **3-CNOT** exact ground-state circuit. On LiH (8 qubits), RL **matches** greedy at the **7-CNOT** chemical-accuracy floor from a minimal double-excitation start, but **beats** greedy from a full ADAPT compile (**11–20 vs 23 CNOTs** at chemical accuracy). At exact FCI, RL and greedy **tie at 25 CNOTs**. Greedy backward elimination remains very strong; RL’s value is **non-greedy exploration** on hard ADAPT starts. Multi-qubit scaling (10q) exposes computational limits of both greedy and RL pruning.

### Suggested section structure

1. **Introduction** — NISQ motivation, CNOTs not excitations  
2. **Background** — VQE, ADAPT, Ostaszewski RL-VQE, greedy prune  
3. **Method** — Pipelines, fair-pair protocol, rewards, order encoding  
4. **Experiments** — H₂, LiH 8q campaign, 6q/10q probes  
5. **Results** — Tables with fair pairs; learning curves  
6. **Discussion** — Two notions of “win”; chem plateau; greedy strength  
7. **Limitations** — Budget, no transfer, hybrid broken, 10q incomplete  
8. **Conclusion** — H₂ success; LiH same-start win; floor tie  

### Numbers safe to put in tables

**Table 1 — H₂ (exact FCI)**

| Method | CNOTs |
|--------|-------|
| ADAPT | 14 |
| Compile + prune (greedy/RL) | 7 |
| Raw RL scratch | **3** |

**Table 2 — LiH 8q fair pairs @ chem acc (1.6 mHa)**

| Start | Greedy | Best RL | n seeds |
|-------|--------|---------|---------|
| 1-double | 7 | 7 (tie) | 4 |
| Full ADAPT | 23 | **11** (best GPU) | 3+ complete/partial |

**Table 3 — LiH 8q @ exact FCI**

| Start | Greedy | RL |
|-------|--------|-----|
| Full ADAPT | 25 | 25 (tie) |

---

## 15. Limitations and honest caveats

1. **Classical simulation only** — no quantum hardware noise.  
2. **Single molecule family** (H₂, LiH) — no cross-molecule transfer.  
3. **Training budget** far below Ostaszewski paper scale (GPU 55–100 updates vs 1000+).  
4. **Chem-acc plateau** — many CNOT counts give ~1.47 mHa; report exact FCI for discrimination.  
5. **Partial runs** — 6q 3/5 CNOTs and 10q incomplete are not volume-verified.  
6. **Inner optimizer asymmetry** — small differences in restarts/maxiter between RL and greedy eval.  
7. **Hybrid pipeline** — not production-ready for report claims.  
8. **Cost** — Modal overnight ~$50–200 total (per user estimate); document if required by course.

---

## 16. File index

### Consolidated outputs (start here)

| Path | Description |
|------|-------------|
| `FINAL_REPORT_CONTEXT.md` | This document |
| `results/final_report/master_results.json` | All JSON aggregated |
| `results/final_report/fig*.png` | Report figures |

### Primary result JSON

| Path | Content |
|------|---------|
| `results/compare_raw_h2.json` | H₂ master table |
| `results/fair_greedy_compare.json` | Greedy 7/23 baselines |
| `results/beat_greedy/exact_compare.json` | Exact 25 tie |
| `results/lih_campaign/fair_comparison.json` | Local smoke fair pairs |
| `results/lih_campaign/RL_prune_ADAPT_chem_s0_result.json` | Local 12-CNOT run |
| `results/lih_campaign/lih_campaign/RL_prune_adapt_chem_s0_result.json` | Modal 20-CNOT |
| `results/lih_campaign/lih_campaign/adapt_chem_s2/` | Modal 11-CNOT seed 2 |
| `results/lih_campaign/baselines_6q.json` | 6q greedy |

### Code entry points

| Path | Role |
|------|------|
| `lih_campaign.py` | Fair-pair orchestration |
| `modal_lih.py` | Modal GPU jobs |
| `prune_trainer.py` | RL prune training |
| `raw_prune.py` | Greedy prune + inner opt |
| `train_raw.py` | From-scratch / hybrid |
| `make_final_report_figures.py` | Figure generation |

### Legacy / supplementary

| Path | Role |
|------|------|
| `POSTER_SESSION_SUMMARY.md` | Jun poster cheat sheet (partially superseded) |
| `LIH_CAMPAIGN.md` | Campaign protocol |
| `FUTURE.md` | Future work ideas |
| `results/poster/` | Earlier poster figures |

---

## 17. Commands to reproduce

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download latest Modal results
modal volume get vqe-rl-results lih_campaign ./results/lih_campaign --force

# Consolidate + figures
python consolidate_results.py
python make_final_report_figures.py

# Local smoke fair comparison
python lih_campaign.py --phase smoke --preset smoke

# Modal full campaign (if re-running)
modal run modal_lih.py
modal run modal_lih.py::bundle
```

---

*End of document. For questions about a specific number, check `master_results.json` first, then the `source` field on that run row.*
