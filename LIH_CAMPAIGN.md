# LiH campaign (5-hour Modal budget)

**Primary goal:** **7 CNOTs @ chem acc** from **1-double compile** — same start as greedy’s best LiH result.

## Fair comparisons only

Each row compares **the same starting circuit**:

| Pair | Start | Greedy floor | RL must beat |
|------|-------|--------------|--------------|
| **`1double_chem`** (priority 1) | 1-double compile (28 gates) | **7 CNOTs** | `< 7` CNOTs @ chem |
| **`adapt_chem`** (priority 2) | full ADAPT compile (96 gates) | **23 CNOTs** | `< 23` CNOTs @ chem |
| `adapt_exact` (skip in 5h) | full ADAPT compile | **25 CNOTs** | only if extra time |

Output: `results/lih_campaign/fair_comparison.json`

**Skipped** (poor ROI): scratch raw RL on 8q, hybrid simul+prune, exact-FCI unless you have spare time.

## Presets

| `--preset` | When | RL updates |
|------------|------|------------|
| `smoke` | Local test | 15 |
| **`modal_5h`** | **Modal default** | 100 (1-double) + 55 (ADAPT) |
| `full_scale` | Multi-session only | 250 |

## Time & cost (your ~5h cap)

### Background smoke prune (running now)

Already finished baselines (~45 min) + 1-double RL (tied **7**).  
ADAPT chem RL at **~upd 10/15** → **~5–10 min left**.

### Local

| Job | Time |
|-----|------|
| Smoke (full) | ~60–90 min (greedy ADAPT dominates) |
| 1-double pair only | ~5 min RL + 1 min greedy |

### Modal (`modal_5h`) — fits in **~4–5 hr**, **~$4–7**

| Job | Command | GPU time | Est. cost |
|-----|---------|----------|-----------|
| **Beat-7 only** (recommended first) | `modal run modal_lih.py --pairs 1double_chem` | **~1.5–2.5 hr** | **~$2–3** |
| **Full focus** (default) | `modal run modal_lih.py` | **~3.5–4.5 hr** | **~$4–6** |
| + exact pair | add `--run-exact-greedy` locally first | +1–2 hr | skip for 5h cap |

Uses **cached** ADAPT greedy from `results/fair_greedy_compare.json` (skips ~30 min re-greedy on Modal).

Modal function **hard timeout ~5h 5min**.

## Commands

```bash
# Local smoke (already running / rerun)
python lih_campaign.py --phase smoke --preset smoke

# Modal — beat 7 CNOTs (cheapest serious run)
modal run modal_lih.py --pairs 1double_chem

# Modal — full fair comparison within budget
modal run modal_lih.py

# Download
modal volume get vqe-rl-results lih_campaign ./results/lih_campaign
```

## Success

- **Win:** `rl_beats_greedy_cnots: true` on `1double_chem` with chem acc  
- **Tie at 7:** same as greedy — need `full_scale` or more seeds  
- **ADAPT pair:** RL **12 CNOTs** in smoke beat greedy **23** (same ~1.47 mHa) — interesting but **not** the 7-CNOT goal (different start)
