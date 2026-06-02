# CS224R poster / final report checklist (vs course guidelines)

## Poster must answer (Section 6) — you have this

| Question | Where in your story |
|----------|---------------------|
| Problem | Minimize **compiled CNOTs** at **chemical accuracy** for VQE circuits |
| Why interesting | Quantum chemistry cost; RL vs strong greedy baselines |
| Prior work gap | ADAPT uses expensive excitations; raw RL (Ostaszewski) uses dense rewards; pruning often greedy |
| Your approach | PPO + three pipelines: scratch raw, ADAPT→prune, hybrid/simul+prune + moving threshold |
| Key findings | H₂ raw RL **3**; LiH **7** @ chem acc (greedy + RL); raw RL fails LiH; RL prune **12** from ADAPT |
| Lessons / limits | 7 is shared floor with greedy; energies ~1.47 mHa not FCI; simul+prune stochastic |
| Planned before final report | Optional: verify RL-12 energy; 1 short LiH greedy circuit PNG |

## Novelty requirement (Section 1) — OK if framed correctly

You satisfy **application + algorithm modification**:
- Non-trivial envs (raw + Givens + hybrid + prune-any-index)
- Ostaszewski-style rewards + moving threshold
- **Failure modes** are a feature: LiH raw collapse, ADAPT-start vs one-double-start

You do **not** need SOTA. Emphasize **insights**, not “we beat greedy every time.”

## Figures you now have (`results/poster/`)

| File | Use on poster |
|------|----------------|
| `h2_cnot_comparison.png` | H₂ bar chart (3 / 4 / 7 / 14) |
| `lih_cnot_comparison.png` | LiH bar chart (7 / 12 / 14 / 23 / 46) |
| `lih_energy_comparison.png` | Same ~1.47 mHa for compressed circuits |
| `lih_rl_prune_learning.png` | RL prune 46→12 learning curve |
| `lih_simul_prune_learning.png` | Simul+prune medium: return + best CNOTs → 7 |
| `h2_simul_prune_learning.png` | H₂ simul+prune (if present) |
| `h2_adapt_vqe_convergence.png` | Classical VQE/ADAPT energy trace |
| `circuit_h2_greedy7.png` | H₂ 7-CNOT circuit diagram |
| `circuit_lih_schematic.png` | LiH pipeline schematic (or run greedy cache for full diagram) |
| `../raw_gate_h2.png` | H₂ raw RL training |
| `../reinforce_returns.png` | Early H₂ REINFORCE |
| `../lih_vqe_convergence.png` | LiH VQE |
| `../lih_training_curves.png` | LiH PPO (excitation env) |

Regenerate: `python make_poster_figures.py`

## AI disclosure (required in final report)

Add a short section: which tools used for boilerplate/debugging vs you implemented PPO, envs, experiments.

## Submission deadlines (Figure 1)

- Poster print + Gradescope: **June 3** (9:00 AM video if CGOE)
- Final report: **June 8**

## Honest claims to avoid

- ❌ “RL beat ADAPT 46→14” (14 = one double)
- ❌ “Only RL gets 7 on LiH” (greedy from one-double gets 7)
- ❌ “RL has better energy than greedy at 7 CNOTs” (same ~1.47 mHa band)
- ✅ “RL prune beats **greedy from ADAPT** (23→12)”
- ✅ “Raw RL finds **3 CNOTs on H₂**”
- ✅ “Simul+prune RL **matched** 7 CNOTs once; greedy is reliable baseline”
