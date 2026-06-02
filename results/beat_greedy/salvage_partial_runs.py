#!/usr/bin/env python3
"""One-shot: aggregate partial logs into exact_compare_partial.json."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "exact_compare_partial.json"

RL_LOG = ROOT / "prune_adapt_exact.log"
GREEDY_K = """
ADAPT-1: start 14 CNOTs err=1.4677 mHa | greedy prune -> 7 CNOTs err=1.4677 mHa gates=14
ADAPT-2: start 28 CNOTs err=0.1097 mHa | greedy prune -> 13 CNOTs err=1.4677 mHa gates=26
ADAPT-3: start 42 CNOTs err=0.0675 mHa | greedy prune -> 19 CNOTs err=1.4677 mHa gates=38
ADAPT-4: start 44 CNOTs err=0.0326 mHa | greedy prune -> 23 CNOTs err=1.4677 mHa gates=42
""".strip()


def parse_rl_log(path):
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text().splitlines() if "RL_prune_ADAPT_exact" in ln]
    if not lines:
        return None
    last = lines[-1]
    m = re.search(
        r"upd (\d+)/(\d+) \| best (\d+) CNOTs \| ([\d.]+) mHa \| exact=(\w+)", last
    )
    if not m:
        return None
    upd, total, cnots, mha, exact = m.groups()
    return {
        "method": "RL_prune_from_ADAPT_compile",
        "target_Ha": 1e-6,
        "status": "partial" if int(upd) < int(total) else "complete",
        "updates": f"{upd}/{total}",
        "cnots": int(cnots),
        "err_mHa": float(mha),
        "within_exact": exact == "True",
        "note": "from prune_adapt_exact.log; circuit JSON not saved yet",
    }


def main():
    payload = {
        "note": "Partial/aborted runs; greedy exact-only may supersede greedy row",
        "rl_exact_adapt": parse_rl_log(RL_LOG),
        "greedy_adapt_chem_acc": {
            "cnots": 23,
            "err_mHa": 1.4677295661433831,
            "within_exact": False,
            "source": "fair_greedy_compare.json",
        },
        "greedy_adapt_k_chem_only": [
            {"k": 1, "final_cnots": 7, "err_mHa": 1.4677},
            {"k": 2, "final_cnots": 13, "err_mHa": 1.4677},
            {"k": 3, "final_cnots": 19, "err_mHa": 1.4677},
            {"k": 4, "final_cnots": 23, "err_mHa": 1.4677},
        ],
        "aborted_or_stuck": [
            "greedy_adapt_exact_baseline.py stuck on chem pass (no verbose)",
            "RL_prune_1double_exact: inf error",
            "RL_prune_ADAPT2_exact: stuck 28 CNOTs",
        ],
    }
    greedy_only = ROOT / "greedy_adapt_exact_only.json"
    if greedy_only.exists():
        payload["greedy_adapt_exact"] = json.loads(greedy_only.read_text())
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
