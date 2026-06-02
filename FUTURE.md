# Project thesis & roadmap

## UPDATE 3 (final experiments pipeline) — READ FIRST

### Corrected LiH(2e,4o) 8q results (do NOT over-claim Givens RL)

| Method | CNOTs | chem acc? | exact FCI? |
|--------|-------|-----------|------------|
| ADAPT full | 46 | yes | yes |
| ADAPT prefix @ 1st excitation | 14 | yes | no |
| Greedy excitation prune | 14 | yes | no |
| Best single double (brute force) | 14 | yes | no |
| **Greedy raw-gate prune** (from compiled ADAPT) | **23** | yes | no |
| RL Givens + order | 14 | yes | no |
| RL raw from scratch | FAILED (STOP collapse) | — | — |

**Retracted:** "RL Givens beats ADAPT 46→14" is **not novel** — 14 CNOTs is exactly one double excitation,
found by ADAPT's first step, greedy excitation prune, or brute-force scan.

**Honest thesis:** Three strategies compared on compiled CNOTs:
1. **From-scratch raw RL** — wins on H₂ (3 CNOTs, exact FCI, non-subgraph of ADAPT); fails on LiH 8q+.
2. **ADAPT → compile → raw prune** — 46→23 CNOTs @ chem acc (greedy); excitation-level greedy still wins at 14.
3. **Hybrid Givens→raw** — Phase 1: Givens build to chem acc. Phase 2: compile to raw gates, RL prune (any-index removal). On H₂: **7 CNOTs @ exact FCI** (ties greedy raw-prune; from-scratch raw still wins at 3).

### H₂ headline (unchanged, strongest result)

| Method | CNOTs | exact FCI? |
|--------|-------|------------|
| ADAPT | 14 | yes |
| ADAPT + greedy/RL raw prune | 7 | yes |
| **Raw RL from scratch** | **3** | yes |
| **Hybrid Givens→compile→prune** | **7** | yes (ties greedy prune) |

### Implementation (items 1–4 from final plan)

- `run_master_table.py` — full baseline table
- `run_final_experiments.py` — baselines + hybrid + RL prune phases
- `raw_gate_env.py` — hybrid two-phase (unlock raw @ chem acc, require raw gates before success), Ostaszewski rewards, `order_k`
- `prune_env.py` — any-index gate removal (middle pruning), Ostaszewski +5/−5, order encoding of removed gates
- `train_raw.py` — GPU-ready PPO, moving threshold, hybrid gate set

**Order encoding (`order_k=4`):** keep it — raw circuits are order-dependent; bag-of-gates alone is lossy (Ostaszewski 2021).

**GPU:** use `--hidden 512`, more updates (150–300), CUDA when available.

---

## UPDATE 2 (LiH scaling) — superseded by UPDATE 3 above

## UPDATE (after literature review + compiled-CNOT experiments) — READ FIRST

The earlier roadmap (below) was built around an **excitation-only** action space and
concluded "don't add raw RX/RY/RZ/CNOT gates." **That conclusion is now reversed.** The
RL-for-VQE literature (Ostaszewski et al., NeurIPS 2021; CRLQAS 2024) gets its wins from a
**raw-gate** action space, and the right comparison metric is **compiled CNOT count**, not
the number of excitations.

### Key clarification: UCCSD ≠ ADAPT (we were conflating them)

| | UCCSD | ADAPT |
|--|-------|-------|
| What | ALL singles+doubles, fixed | greedily grows operators one at a time |
| Size (LiH 2e/5o) | 24 excitations (bloated) | 8 excitations (already compact) |
| Reaches FCI | yes | yes |
| Compiled CNOTs | ~300+ | 88 |

The "20× depth reduction" headline from the literature is **RL vs UCCSD**, NOT vs ADAPT.
ADAPT *also* crushes UCCSD on depth, so beating UCCSD is not, by itself, a differentiator.

### Where RL can actually differentiate: compiled CNOT count

Every excitation gate is CNOT-expensive after compilation:
- `SingleExcitation` ≈ 2 CNOTs, `DoubleExcitation` ≈ 14 CNOTs (measured via
  `vqe_core.compiled_cnots_for_actions`).
- So ADAPT's 8-gate LiH circuit = **88 CNOTs**; H2's single DoubleExcitation = **14 CNOTs**.

A **raw-gate** RL agent is not forced to use these expensive blocks — it can chase the same
state with cheap native gates. That is the only honest way RL can beat ADAPT.

### Results so far (this iteration)

**Option A — raw-gate RL (RX/RY/RZ/CNOT), dense reward, actor-critic (`train_raw.py`):**
- **H2: reached chemical accuracy with 3 CNOTs**, vs 14 CNOTs for the excitation circuit
  (~4.7× reduction). Proof of concept that the raw-gate angle is real. (run:
  `python train_raw.py --molecule H2 --updates 60 --max-gates 12`)
- Next: run on LiH (`--molecule LiH`) and compare best CNOTs@chem-acc vs ADAPT's 88.

**RL vs rule-based, head-to-head on H2 (`compare_raw.py` -> `results/compare_raw_h2.json`):**

| Method | CNOTs | err vs FCI |
|--------|------|-----------|
| ADAPT (excitations) | 14 | exact |
| ADAPT + greedy excitation prune | 14 (only 1 excitation, can't prune) | exact |
| ADAPT compiled + greedy raw-gate prune | 7 | exact |
| ADAPT compiled + RL raw-gate prune (ablation) | 7 | exact |
| **raw-gate RL from scratch (chem acc)** | **3** | ~0 |
| **raw-gate RL from scratch (exact FCI)** | **3** | ~1e-16 |

Core novelty evidence: the strongest NON-learned pipeline (compile ADAPT's circuit, greedily
strip raw gates) bottoms at 7 CNOTs. RL from scratch finds a 3-CNOT circuit that reaches
EXACT FCI. RL pruning ties greedy at 7 (on H2 the pruning problem is small enough that greedy
already finds the optimal sub-circuit) -- consistent with the theory that pruning is bounded
by the starting circuit's structure, so RL's value is in building from scratch, not pruning.

**Why the inner optimizer matters (`vqe_core.optimize_angles`):** switched from COBYLA to
L-BFGS-B with adjoint (analytic) gradients + warm-start + random restarts. Benchmark on a
fixed H2 raw circuit: L-BFGS err 2.06e-2 in 143ms vs COBYLA/Adam stuck at 6.0e-1, and QNG
0.64 in 1073ms. QNG only pays off in the barren-plateau regime (many qubits); at our scale
it is slower AND worse. The better optimizer is what let from-scratch RL hit EXACT FCI (1e-16)
at 3 CNOTs instead of 6e-6.

**New knobs added:**
- `train_raw.py --target exact` : reward floor = exact FCI (1e-6) instead of chem acc; tracks
  best circuit at both levels.
- `train_raw.py --curriculum` : anneal success target from ~30% correlation energy to the floor.
- `train_raw.py --number-penalty C` : soft particle-number symmetry penalty (physics-inspired,
  see below).
- `prune_env.py` : RL raw-gate pruning ablation (`RawGatePruneEnv` + `train_prune`).

**Physics-inspired symmetry (the open lever for LiH):** raw RX/RY/RZ/CNOT break
particle-number symmetry, so the agent can waste gates on wrong-electron-count states.
- Hard constraint (only particle-conserving 2-qubit blocks / Givens) would basically recreate
  the excitation/ADAPT ansatz and forfeit the CNOT win -- that's WHY the papers use the
  unconstrained set: the generality is the source of the depth advantage, since the optimal
  short circuit can transiently violate symmetry as long as the FINAL state conserves it.
- Soft constraint (implemented): penalize |<N> - n_electrons| in the reward
  (`number_penalty`). Keeps raw-gate freedom but nudges toward physical states. This is the
  closest thing to a PINN-style physics loss and is a clean LiH experiment to try.

Run on LiH next (`python compare_raw.py --molecule LiH ...`) to see if the 3-vs-7 gap holds.

**Curriculum (`train_raw.py --curriculum`):** anneals the success target from ~70% of the
correlation energy down to chemical accuracy over the first 60% of training, so the agent
gets reward signal early. Verified on H2 (still finds 3 CNOTs); the real test is LiH.

**Option B — pruning (`prune_baselines.py`):**
- Honest baseline is **greedy backward elimination**, NOT random removal.
- LiH from ADAPT: greedy prunes **8 gates/88 CNOTs → 4 gates/56 CNOTs** at chem acc in 30
  VQE calls. Random finds the same 4/56 but needs ~195 calls.
- Takeaway: on LiH, **RL pruning will not beat greedy** (same ceiling, greedy is cheap).
  Pruning is only an interesting RL problem when circuits are large enough that greedy's
  O(n²) becomes costly AND non-greedy prunings exist. So Option A (raw gates) is the
  stronger differentiator; Option B is a baseline/secondary story.

### Why this is learnable (credit assignment)

The worry "a gate is only good in combination with others" is correct for SPARSE reward.
The fixes (all now implemented in `raw_gate_env.py` / `train_raw.py`):
1. **Dense per-step reward** = fractional progress toward FCI each step (Ostaszewski eq. 2),
   clipped at −1. Good gates get immediate credit.
2. **Critic baseline + discounted returns** bootstrap credit back to earlier gates.
3. **Current energy is part of the state** so the policy can judge partial-circuit quality
   (same signal ADAPT exploits via gradients).
4. Curriculum (moving chem-acc threshold) — TODO if training is unstable on LiH.

### New comparison metric (use everywhere): compiled CNOTs

`vqe_core.compiled_resources(ops)` → `cnot_count`, `cnot_depth`, `total_gates`.
Report ALL methods (UCCSD, ADAPT, raw-gate RL, pruned) on compiled CNOTs at chem acc.

### Revised thesis (one sentence)

> **Use RL over a raw native gate set to discover VQE ansätze that reach chemical accuracy
> with fewer compiled CNOTs than excitation-based methods (UCCSD and ADAPT).**

---

## (Older roadmap below — kept for reference; superseded where it says "don't use raw gates")

## What the data actually says (be honest in the writeup)

From `results/lih_comparison.json` (LiH active space 2e/5o, 10 qubits, 24-operator pool):

| Method | Error vs FCI | Gates | VQE evals |
|--------|--------------|-------|-----------|
| **ADAPT** | ~1.7e-6 Ha | 8 | 8 |
| PPO (scratch) | ~1.2e-3 Ha | 8 | 465 |
| PPO + warmstart | ~1.7e-6 Ha | 8 | 1 |
| Random (matched budget) | ~3.2e-4 Ha | 8 | 465 |

**Conclusion:** RL does **not** beat ADAPT at the job "find the most accurate ansatz."
That is expected — ADAPT is literally designed for this and is greedy-optimal per step.

From `results/lih_prune.json` (depth penalty + warm-start from ADAPT):

- ADAPT: **8 gates** at FCI
- RL (warmstart-soft, lam=3e-4): **5 gates** within chemical accuracy (~1.3 mHa error)
- RL explores **different** subsets — not the same circuit as ADAPT

**Emerging RL niche:** not replacing ADAPT, but **compressing** or **re-ranking**
circuits under a **multi-objective** goal (energy + depth), especially when you
already have a good starting point.

---

## Recommended project goal (one sentence)

> **Automate discovery of compact VQE ansätze that reach chemical accuracy with
> fewer excitations than ADAPT, using RL for multi-objective structure search
> where greedy gradient methods do not optimize for depth.**

This is defensible, novel *enough* for CS224R, and matches your actual results.

Do **not** claim: "RL finds lower energy than ADAPT/FCI."
Do claim: "RL finds **shorter** circuits at matched accuracy" or "RL explores
the Pareto frontier between energy and depth."

---

## What NOT to do (probably)

| Idea | Verdict |
|------|---------|
| ~~**Add RX/RY/CNOT (HEA) to action space**~~ | **REVERSED — see top of file.** Raw gates are now the PRIMARY differentiator (compiled-CNOT win). Interpretability matters less than CNOT cost. |
| **"Beat FCI energy"** | Impossible; FCI is exact in basis |
| **"Beat classical CCSD(T)"** | Out of scope; classical wins on small molecules |
| **Full LiH 92-operator pool without budget** | ADAPT grad screening × 92 ops/step is slow; RL training would be brutal |
| **Bigger molecule without active space** | Sim/runtime explodes before you learn anything |

Optional stretch: **HEA as a separate appendix baseline** — not mixed into excitation RL.

---

## Experiments that could show RL value (prioritized)

### Tier 1 — Do these (builds on what you have)

**1. Pareto frontier: energy vs depth (primary figure for poster)**

- Sweep `lam` in reward `(E_HF - E) - lam * n_gates`
- Plot ADAPT point (8 gates, FCI) vs RL-found points
- Highlight: **5–6 gate circuits within 1.6 mHa of FCI** that ADAPT doesn't report
- Script: `train_lih.py` (already has Pareto); polish + annotate chemical accuracy band

**2. Fixed VQE budget comparison**

- Give ADAPT, random, PPO each **exactly N VQE evaluations** (e.g. N=50, 100, 200)
- Metric: best energy vs FCI as function of budget
- Hypothesis: RL may not beat ADAPT at convergence, but might beat **random** and
  find *good-enough* circuits faster than full ADAPT on **larger pools**
- Add to `compare_lih.py` or new `budget_sweep.py`

**3. RL-as-pruner (clear narrative)**

- Pipeline: `ADAPT → warm-start PPO with lam>0 → shortest circuit within chem acc`
- Script: `prune_lih.py` (already exists)
- Report: "ADAPT gives accuracy; RL prunes 8 → 5 gates while staying within 1.6 mHa"

### Tier 2 — Harder instances (where ADAPT gets expensive, RL might shine)

**4. Larger active space on LiH**

- e.g. `(2e, 6o)` or `(3e, 5o)` — bigger pool, ADAPT grad screening cost grows
- Brute force still infeasible; ADAPT still works but is slow
- Question: at **fixed budget**, does RL find better circuits than random?

**5. Stretched bond / harder correlation**

- Same molecule, bond length away from equilibrium (stronger correlation)
- ADAPT may need more gates; pruning vs accuracy tradeoff may change
- One geometry sweep plot is enough

**6. H2O minimal active space**

- Next molecule up; establishes "not just LiH"
- Use same pipeline: `make_h2o_config()` mirroring `make_lih_config()`

### Tier 3 — Mention as future work (don't build unless time)

- Operator **re-use** in action space (ADAPT allows repeats; you currently forbid duplicates)
- Gate **ordering** for hardware depth (currently energy is order-independent)
- Transpiled CNOT count as depth metric instead of excitation count
- Imitation learning from ADAPT + RL fine-tune (you have warm-start already)

---

## Why ADAPT "already wins" and why that's OK

ADAPT is a **greedy gradient oracle** on the same pool. For single-reference,
small molecules it is extremely hard to beat on **energy**.

RL value propositions that remain honest:

1. **Multi-objective** — ADAPT doesn't optimize depth; you do via `lam`
2. **Budget-constrained** — full ADAPT runs all steps; RL might stop early with good-enough
3. **Non-greedy** — RL can skip an operator ADAPT would take if it helps depth later (needs longer horizons / better training)
4. **Scalability narrative** — on larger pools, ADAPT's per-step gradient screening is expensive; learned policies amortize search (needs Tier 2 experiments to demonstrate)

---

## Suggested final report structure

1. **Problem:** automated compact ansatz discovery for VQE (not quantum supremacy)
2. **Baselines:** HF, UCCSD, ADAPT, random
3. **Method:** RL structure search + classical VQE inner loop
4. **Results H2:** pipeline validation
5. **Results LiH:** ADAPT hits FCI; RL finds shorter chem-accurate circuits (Pareto plot)
6. **Results budget sweep (if done):** sample efficiency vs random
7. **Limitations:** RL doesn't beat ADAPT on energy; active space; simulator only
8. **Future:** larger pools, H2O, hardware transpilation

---

## Open questions from team discussion

- **Transpiler:** compresses fixed circuits; does not replace structure search (see top of file)
- **Custom gates beyond excitations:** compilation/synthesis problem; skip for main project
- **CCSD(T):** cite as classical reference, don't implement
