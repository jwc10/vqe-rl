# Final report assets

Generated 2026-06-08.

| File | Description |
|------|-------------|
| `master_results.json` | Aggregated JSON from all local + Modal runs |
| `fig01_h2_cnot_comparison.png` | H₂ CNOT bar chart |
| `fig02_lih8q_fair_comparison.png` | LiH 8q greedy vs RL (fair pairs) |
| `fig03_lih8q_adapt_chem_learning.png` | ADAPT-start RL learning curves |
| `fig04_multiscale_comparison.png` | 6q / 8q / 10q greedy vs RL |
| `fig05_lih8q_energy_cnot_pareto.png` | Energy error vs CNOTs |
| `fig06_lih8q_1double_seeds.png` | 1-double Modal seed ties |
| `fig07_modal_campaign_table.png` | Completed Modal runs table |

**Full narrative:** `../../FINAL_REPORT_CONTEXT.md`

**Modal raw downloads:** `../lih_campaign/lih_campaign/` (per-seed subdirs)

**Regenerate:**

```bash
modal volume get vqe-rl-results lih_campaign ./results/lih_campaign --force
python consolidate_results.py
python make_final_report_figures.py
```
