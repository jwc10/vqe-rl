# Poster session summary (Jun 2026)

Consolidated narrative for the CS224R poster: what we built, what we measured, what worked, what did not, and what to say about future work. Does **not** replace `POSTER.md` / `FUTURE.md` — use this as the “everything we just did” cheat sheet.

---

## One-sentence thesis

**Use RL over compiled native gates to minimize CNOT count at chemical accuracy (or exact FCI), comparing from-scratch construction, ADAPT→compile→prune, and hybrid Givens→prune — with honest greedy baselines on compiled circuits.**

---

## Poster-worthy results (verified)

### H₂ (4 qubits) — **RL wins**

| Method | CNOTs | vs FCI | Notes |
|--------|-------|--------|--------|
| ADAPT (excitations) | 14 | exact | Baseline |
| ADAPT compile + greedy/RL prune | 7 | exact | Greedy = RL on H₂ |
| Hybrid Givens→compile→prune | 7 | exact | Ties prune |
| **Raw-gate RL from scratch** | **3** | exact | **Main positive result** |

Source: `results/compare_raw_h2.json`, `results/final_h2.json`.

### LiH (2e, 4o, 8 qubits) — **greedy wins or ties; RL does not beat it**

| Goal | Best greedy | Best RL | Verdict |
|------|-------------|---------|---------|
| **Chem acc (1.6 mHa)** | **7 CNOTs** (prune from 1-double compile) | **7** (tie; same ~1.47 mHa plateau) | Floor, not RL-specific |
| **Chem from full ADAPT** | **23 CNOTs** | 12 logged once, unverified; hybrid often 8–14 | Greedy reliable |
| **Exact FCI from ADAPT** | **25 CNOTs**, ~0.0008 mHa | **25 CNOTs**, ~0.0008 mHa | **Tie** (Jun 2026 head-to-head) |

Sources: `results/fair_greedy_compare.json`, `results/beat_greedy/exact_compare.json`, `results/lih4_simul_prune_partial.json`, `results/final_lih_2e4o.json`.

**14-CNOT story (important):** One double excitation → 14 CNOTs, ~1.47 mHa — chemical accuracy but **not** exact FCI. ADAPT step 1, brute force, RL Givens+order, and greedy excitation prune all find the **same** circuit. Not a novel RL discovery.

**Exact FCI comparison (the run we cared about):**

- RL prune from full ADAPT (`target = 1e-6` Ha): **25 CNOTs** by update 3, plateau to update 14 (then killed).
- Greedy raw-prune from same ADAPT compile (`chem_acc = 1e-6`): **25 CNOTs**, 55 gates, ~0.000796 mHa, ~20 min, ~3500+ inner optimizations.
- **Conclusion:** RL did **not** beat greedy on final CNOTs or accuracy; at best it reached the **same** Pareto point, sometimes faster in wall-clock to first hit 25.

---

## What we built (engineering arc)

1. **Core metric:** compiled **CNOT count** at chem acc and/or exact FCI (`vqe_core.compiled_resources`, `raw_prune.count_cnots`).
2. **Inner optimizer:** L-BFGS-B + adjoint gradients + warm-start (critical for H₂ 3-CNOT exact result).
3. **Three RL pipelines:**
   - **From scratch:** `train_raw.py` + `raw_gate_env.py` (RX/RY/RZ/CNOT, optional Givens, hybrid phases).
   - **Prune compiled circuit:** `prune_env.py` + `raw_prune.greedy_raw_prune`.
   - **Hybrid:** Givens build → compile → prune; variants `chained`, `simultaneous`, `simul+prune` (`train_raw.py`).
4. **Baselines:** ADAPT-VQE (`adapt.py`), greedy prune (`quick_fair_compare.py`, `greedy_adapt_exact_only.py`), ADAPT-k sweep (k=1..4), brute floor (`brute_min_cnots.py`).
5. **Beat-greedy hunt:** `beat_greedy_lih.py`, logs under `results/beat_greedy/`, comparison `exact_compare.json`.
6. **Poster assets:** `make_poster_figures.py` → `results/poster/`, `results/poster_summary.json`.

---

## Reward design: dense rewards (yes, implemented)

Following **Ostaszewski et al. (NeurIPS 2021)** style credit assignment (not sparse terminal-only):

| Component | Where | What |
|-----------|--------|------|
| **Dense per-step reward** | `raw_gate_env._step_reward` | Fractional progress toward FCI: `(E_prev - E_new) / (E_prev - E_FCI)`, clipped at −1 |
| **Terminal bonus** | `raw_gate_env._end_reward`, `prune_env` | **+5** if within target, **−5** otherwise (`reward_mode="ostaszewski"`) |
| **Optional CNOT shaping** | `cnot_penalty` on each CNOT placement | Used in raw build, not main LiH prune story |
| **Moving threshold** | `curriculum.MovingThreshold` + `train_raw` | Target energy gap **tightens** as agent improves (feedback curriculum), used in hybrid/simul+prune |
| **Static curriculum** | `--curriculum` | Anneal success target from ~HF+30% correlation down to chem/exact floor |

**Prune env** uses the same ±5 terminal logic plus removal-step rewards tied to energy/CNOT changes.

**Why dense rewards mattered:** Without them, sparse “only good at end” rewards fail on long gate sequences (combinatorial credit assignment). H₂ from-scratch RL likely **depends** on dense + good inner optimizer.

---

## “Positional encoding” — what we actually did

**We did *not* implement sinusoidal / transformer positional encodings.**

We implemented **order encoding** (`order_k`, default 3–4 in prune, 4 in hybrid):

- State = bag-of-gate counts + depth fraction + correlation-energy fraction + **one-hot indices of the last `order_k` gates/actions**.
- Rationale (from Ostaszewski): raw circuits are **order-dependent**; a pure bag-of-gates Markov state is lossy.

**Ablations run** (`lih_raw_experiment.py`): `order_k=0` (bag only) vs `order_k>0` vs Givens — on LiH 8q, **raw from scratch failed** for all variants (no chem-acc circuit found in budget).

**Not tried:** full sequence encoder (RNN/Transformer over gate sequence) as in some follow-up papers.

---

## Pareto sweep — what it is and what it is *not*

### A) Excitation-level Pareto (`train_lih.py`) — **done early in project**

- **Action space:** ADAPT-style **excitation** pick list (not raw gates).
- **Sweep:** `PARETO_LAMBDAS = [0, 1e-4, 3e-4, 1e-3]` — depth penalty `lam` in PPO reward shaping.
- **Plot:** `results/lih_pareto.png` — energy vs number of excitations for LiH **(2e, 5o) 10 qubits**.
- **Purpose:** Show tradeoff between circuit depth and energy before raw-gate pivot.
- **Note:** Small lambdas only — large `lam` collapses to HF because marginal energy per gate ≈ 1e-4 Ha.

### B) CNOT-level “Pareto” (poster intent, partially done)

The poster framing we want is **energy vs compiled CNOTs** across methods (ADAPT 46, greedy 23, RL 25 exact, 7 chem, etc.). That frontier is a **table/plot of best points per method**, not a single λ-sweep script yet.

**Greedy ADAPT-k sweep (chem target):** k=1→7, k=2→13, k=3→19, k=4→23 CNOTs — shows more ADAPT steps allow **more** greedy pruning before hitting the 1.47 mHa plateau.

---

## How far we copied Ostaszewski / related RL-VQE work

| Idea from literature | Our implementation | Depth |
|---------------------|-------------------|--------|
| Raw native gate set | RX/RY/RZ/CNOT (+ optional Givens) | Full |
| Dense progress reward | `_step_reward` | Full |
| ±5 terminal | `_end_reward` / prune | Full |
| Actor-critic PPO | `train_raw`, `ppo.py` | Full |
| Order of last k gates | `order_k` one-hot window | Partial (not full sequence model) |
| Moving feedback threshold | `MovingThreshold` | Full |
| Paper-scale training | 40–80 updates, CPU, hidden 256 | **Not** — orders of magnitude smaller |
| Multi-molecule generalization | H₂ + LiH 8q mainly | Minimal |
| Particle-number soft penalty | `--number-penalty` | Implemented, **not** decisive on LiH in short runs |

**CRLQAS / other 2024 papers:** cited in `FUTURE.md` for motivation; we did **not** reimplement their full pipelines.

---

## Why things did not work or did not beat greedy (honest explanations)

### LiH raw RL from scratch

- **Action space explosion** on 8 qubits vs 4 (H₂).
- Agent often **STOPs early** or never reaches chem acc in 60 updates.
- Unconstrained raw gates can explore **non–particle-number-conserving** states; Givens fixes structure but caps at **14 CNOTs** (one double).
- **Training budget** far below Ostaszewski-scale (GPU, 150–300 updates, larger net).

### LiH RL prune

- **Greedy is a strong baseline:** each removal is locally optimized with full re-VQE; for 96-gate ADAPT compile, greedy is slow but **deterministic** and finds the same minima RL found.
- **Chem-acc plateau (~1.47 mHa):** many circuits (7, 13, 23 CNOTs) share the **same energy** — RL “wins” in logs are often **fewer gates than 23**, not better physics.
- **12 CNOTs logged** once — never re-saved/verified; likely same plateau if chem acc.
- **Exact prune:** RL and greedy both land at **25 CNOTs** — pruning cannot go to 7 CNOTs and stay exact; different objective than chem-acc floor.

### Hybrid / simul+prune

- **Compile blow-up:** simultaneous Givens+raw → huge decompositions; medium preset **hung** after update 7.
- **7 CNOTs @ chem** achieved once (simul+prune) but **not reproducible** in retries (8–14 typical).
- Chained hybrid `--quick20` can hit 7 but same energy as greedy-7.

### Greedy “looked broken”

- Not broken — **silent + O(n²) VQE evals per round** (~1 min/round on 96 gates, tens of rounds for chem, more for exact).
- Fixed with `verbose=True` in `greedy_raw_prune`.

---

## Experiments run in the “beat greedy” push (chronological)

| Run | Outcome |
|-----|---------|
| `prune1` RL from 1-double, chem | 7 CNOTs, ties greedy |
| `prune1` exact from 1-double | Failed (14 CNOTs, inf error) |
| RL prune ADAPT exact | 25 CNOTs, plateau update 3–14 |
| Greedy ADAPT chem+exact (`869219`) | 23 chem / **25 exact** |
| ADAPT-k greedy (k=1..4) | 7, 13, 19, 23 CNOTs @ chem |
| ADAPT-2 RL prune | 13 chem (ties greedy-2); exact stuck 28 |
| Simul+prune medium | 7 CNOTs once @ kill; retries 8–14 |
| Brute min CNOTs on greedy-7 | Cannot remove another gate @ chem acc |

---

## What to put on the poster

### Figures / tables

1. **H₂ bar chart:** CNOTs by method (3 vs 7 vs 14) — hero figure.
2. **LiH table:** 46 / 23 / 25 / 14 / 7 with chem? and exact? columns.
3. **Exact comparison callout:** RL 25 = Greedy 25 (tie).
4. **14-CNOT “floor” diagram:** one double, chem acc only.
5. Optional: `lih_pareto.png` (excitation-depth sweep, early work).
6. Learning curve: RL prune updates 1–3 dropping 46→25 (wall-clock story).

### Honest bullets

- **Positive:** H₂ raw RL **3 CNOTs exact**; pipeline ADAPT→compile→prune reduces 46→23 (chem) or 46→25 (exact).
- **Negative:** LiH **no RL beat** on CNOTs vs greedy at either target.
- **Insight:** Greedy backward elimination is deceptively strong; RL’s value on LiH is exploration speed, not final circuit quality.
- **Metric:** always **compiled CNOTs**, not excitation count.

---

## Future work: “fast copy” of prior approaches (poster Future Directions)

### Idea

Replicate Ostaszewski-style training **with a reduced budget** that might still beat greedy on LiH:

| Lever | Fast version | Full paper-scale |
|--------|----------------|------------------|
| Updates | 150–300 | 1000+ |
| Hidden dim | 512 | 512–1024 |
| Device | 1× GPU (CUDA) | Multi-GPU |
| Molecule | LiH 6q (2e,3o) then 8q | LiH, BeH₂, … |
| Warm-start | Init policy from greedy-7 trajectory demos | From scratch |
| Prune | RL only after greedy gets to 30 gates | Full 96-gate RL episodes |
| Inner VQE | `maxiter=30`, 0 restarts for training | 50–100, restarts |

### Concrete next experiments (1–2 sentences each on poster)

1. **GPU PPO from scratch** on LiH 6q with `order_k=4`, moving threshold, 200 updates — test if 6q transfers before 8q.
2. **Imitation + RL fine-tune:** behavioral cloning on greedy-7 removal order, then PPO fine-tune for exact target.
3. **Greedy-init RL prune:** start episodes from greedy partial traces (not full ADAPT 96 gates).
4. **Sequence model state** (small Transformer over last 16 gates) instead of fixed `order_k` one-hot — closer to modern CRLQAS encodings.
5. **CNOT Pareto figure:** sweep λ on `cnot_penalty` in `train_raw` and plot best (energy, CNOTs) points per λ.

### Resources needed

| Resource | Estimate | Purpose |
|----------|----------|---------|
| **1 GPU** (NVIDIA, 8–16 GB) | 10–40 hr | LiH 8q PPO 200–300 updates, hidden 512 |
| **CPU cluster** (optional) | 20–60 hr | Greedy exact/chem baselines, ADAPT sweeps |
| **PennyLane Lightning-GPU** | license / install | Faster VQE inner loop if supported for 8q |
| **Storage** | <1 GB | Logs, checkpoints, JSON circuits |
| **Person time** | 2–3 days | Debug simul+prune hangs, verify 12-CNOT claim, polish figures |

**Minimum viable “fast Ostaszewski”:** single Colab/local GPU, `train_raw.py --molecule LiH --updates 200 --hidden 512 --moving-threshold --order-k 4 --device cuda`, ~overnight.

---

## Key file pointers

| File | Role |
|------|------|
| `results/beat_greedy/exact_compare.json` | RL 25 vs greedy 25 exact |
| `results/fair_greedy_compare.json` | Greedy 7 / 23 LiH |
| `results/compare_raw_h2.json` | H₂ 3 vs 7 |
| `results/poster_summary.json` | Short poster JSON |
| `raw_gate_env.py` | Dense + Ostaszewski rewards |
| `curriculum.py` | Moving threshold |
| `train_lih.py` | Excitation Pareto sweep |
| `greedy_adapt_exact_only.py` | Fair greedy exact baseline |

---

## Suggested poster “Future work” slide (copy-paste)

> We implemented Ostaszewski-style dense rewards, ±5 terminals, order encoding, and moving-threshold curriculum, but at **~40 PPO updates on CPU** RL does not beat greedy pruning on LiH. Next: **GPU-scale training** (200+ updates, wider net), **6q curriculum**, and **warm-start from greedy trajectories**; optional sequence encoder vs fixed `order_k`. Goal: test whether paper-scale RL adds value beyond greedy on the **25-CNOT exact** and **7-CNOT chem** frontiers.
