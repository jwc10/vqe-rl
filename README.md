# RL for compiled-CNOT-efficient VQE circuits

CS224R final project. We use reinforcement learning to design variational quantum eigensolver
(VQE) circuits that reach a fixed accuracy target with as few compiled CNOT gates as possible.
An RL agent (PPO) chooses circuit structure in an outer loop; a classical optimizer (L-BFGS-B
with adjoint gradients) fits the gate angles in an inner loop. The metric is the compiled CNOT
count after decomposing to `{RX, RY, RZ, CNOT}`, not the excitation count. Simulator only
(PennyLane); LiH systems are active spaces, not the full molecule.

Headline results: on H2, raw-gate RL finds a 3-CNOT exact circuit. On LiH (8 qubits) at
chemical accuracy, RL matches greedy at the 7-CNOT floor from a one-double start and beats
greedy (11-20 vs 23 CNOTs) from a full ADAPT start. The written report is submitted separately
on Gradescope and is not tracked in this repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python h2_vqe.py          # H2 VQE baseline (hand-picked structures)
python lih_vqe.py         # LiH VQE baseline, prints HF/FCI
python adapt.py           # ADAPT-VQE greedy baseline
python train_raw.py --molecule H2 --gate-set raw   # raw-gate RL from scratch (the 3-CNOT H2 result)
python lih_campaign.py --phase smoke               # LiH fair-pair campaign (greedy vs RL prune)
python compare_raw.py     # H2 pipeline comparison: ADAPT / prune / raw RL
```

GPU campaign and figures:

```bash
modal run modal_lih.py                 # LiH prune jobs on Modal GPUs
python consolidate_results.py          # aggregate runs into results/final_report/master_results.json
python make_final_report_figures.py    # regenerate report figures
```

## Layout

Core:
- `vqe_core.py` molecule-agnostic VQE: build circuit, L-BFGS-B inner loop, compiled-CNOT counting, exact FCI
- `h2_vqe.py` / `lih_vqe.py` Hamiltonian builders and per-molecule baselines
- `rl_env.py` structure-search env and configs (`make_h2_config`, `make_lih_config`)
- `ppo.py` PPO actor-critic for structure search (masked policy head + value head)
- `train_reinforce.py` REINFORCE baseline with a return baseline

Raw-gate and pruning:
- `raw_gate_env.py` step-by-step raw-gate / Givens / hybrid construction env
- `raw_prune.py` greedy backward-elimination prune and the shared inner optimizer
- `prune_env.py` gate-removal env for RL pruning
- `prune_trainer.py` RL prune trainer (behavioral-cloning warm-start, then PPO)
- `greedy_trace.py` greedy removal trace used for the warm-start
- `train_raw.py` PPO on the raw-gate / hybrid envs
- `adapt.py` ADAPT-VQE greedy baseline
- `prune_baselines.py` greedy vs random excitation-level pruning baselines

Campaign and reporting:
- `lih_campaign.py` + `lih_campaign_config.py` fair-pair campaign (same start, greedy vs RL)
- `quick_fair_compare.py` cached greedy baselines used by the campaign
- `greedy_adapt_exact_only.py` greedy baseline at exact FCI
- `modal_lih.py` Modal GPU jobs
- `consolidate_results.py` + `make_final_report_figures.py` aggregate results and build figures
- `compare_lih.py` / `compare_raw.py` method comparisons

Other:
- `curriculum.py` moving accuracy threshold for curriculum runs
- `results/final_report/` figures and aggregated `master_results.json`
- `FINAL_REPORT_CONTEXT.md` provenance for every number; `LIH_CAMPAIGN.md` campaign protocol

## Notes

- Metric is compiled CNOT count at a fixed accuracy (chemical accuracy 1.6 mHa, or exact FCI
  at 1e-6 Ha). One double excitation compiles to 14 CNOTs.
- LiH active spaces: 6q (2e,3o), 8q (2e,4o, main), 10q (2e,5o). These are not the full molecule.
- This is not a claim that quantum beats classical: `lightning.qubit` simulates a quantum
  algorithm, and classical FCI already solves these systems. The contribution is automated,
  CNOT-efficient ansatz discovery and an honest, same-start comparison against strong greedy
  baselines.
