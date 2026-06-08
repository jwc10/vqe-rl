#!/usr/bin/env python3
"""Collect all experiment JSON into results/final_report/master_results.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "final_report"
CHEM_MHA = 1.6


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def rl_result_row(path: Path, meta: dict) -> dict | None:
    data = load_json(path)
    if not data or "cnots" not in data:
        return None
    row = {
        "cnots": data["cnots"],
        "n_gates": data.get("n_gates"),
        "err_mHa": data.get("err_mHa", (data.get("error_vs_fci") or 0) * 1000),
        "within_chem_acc": data.get("within_chem_acc"),
        "within_exact": data.get("within_exact"),
        "beats_greedy": data.get("beats_greedy"),
        "greedy_baseline_cnots": data.get("greedy_baseline_cnots"),
        "seed": data.get("seed"),
        "updates_completed": data.get("update"),
        "total_updates_planned": len(data.get("log", [])) or None,
        "seconds": data.get("seconds"),
        "device": meta.get("device"),
        "preset": meta.get("preset"),
        "status": meta.get("status", "complete"),
        "source": str(path.relative_to(ROOT)),
        "log": data.get("log"),
        "label": data.get("label"),
    }
    row.update({k: v for k, v in meta.items() if k not in row})
    return row


def fair_rows(path: Path) -> list[dict]:
    data = load_json(path)
    if not data:
        return []
    if isinstance(data, dict):
        return [data]
    return data


def scan_rl_results() -> list[dict]:
    rows = []
    patterns = [
        ("results/lih_campaign/RL_prune_*_result.json", {"molecule": "LiH(2e,4o)", "qubits": 8, "pair": "local_smoke"}),
        ("results/lih_campaign/lih_campaign/**/RL_prune_*_result.json", {}),
    ]
    seen = set()
    for glob_pat, defaults in patterns:
        for path in sorted(ROOT.glob(glob_pat)):
            if path in seen:
                continue
            seen.add(path)
            rel = str(path.relative_to(ROOT))
            meta = dict(defaults)
            if "lih6q" in rel:
                meta.update(molecule="LiH(2e,3o)", qubits=6)
            if "1double" in path.name:
                meta["pair"] = "1double_chem"
            elif "adapt_chem" in path.name.lower() or "ADAPT_chem" in path.name:
                meta["pair"] = "adapt_chem"
            if "lih_campaign/lih_campaign/" in rel:
                meta["device"] = meta.get("device", "cuda (Modal H100/A10G)")
                meta["preset"] = meta.get("preset", "modal_5h")
            elif "lih_campaign/RL_prune" in rel and "device" not in meta:
                meta["device"] = "cpu (local smoke)"
                meta["preset"] = "smoke"
            if "adapt_chem_s0" in rel and "lih_campaign/lih_campaign/RL_prune_adapt" in rel:
                meta["device"] = "cuda (Modal A10G)"
            row = rl_result_row(path, meta)
            if row:
                rows.append(row)
    return rows


def build_master() -> dict:
    h2_compare = load_json(ROOT / "results/compare_raw_h2.json")
    fair_greedy = load_json(ROOT / "results/fair_greedy_compare.json")
    exact_compare = load_json(ROOT / "results/beat_greedy/exact_compare.json")
    baselines_8q = load_json(ROOT / "results/lih_campaign/baselines.json")
    baselines_6q = load_json(ROOT / "results/lih_campaign/baselines_6q.json")

    fair_comparisons = []
    for path in sorted(ROOT.glob("results/lih_campaign/**/fair_comparison.json")):
        for row in fair_rows(path):
            fair_comparisons.append({**row, "source": str(path.relative_to(ROOT))})

    rl_runs = scan_rl_results()

    # Log-only partial runs not saved to volume
    partial_runs = [
        {
            "molecule": "LiH(2e,4o)", "qubits": 8, "pair": "adapt_chem", "seed": 1,
            "status": "timeout_partial", "device": "cuda (Modal H100)",
            "cnots": 10, "err_mHa": 1.3599, "greedy_baseline_cnots": 23,
            "updates_completed": 15, "total_updates_planned": 55,
            "source": "terminals/273003.txt (bundle adapt_chem_s1)",
            "note": "Killed at 6h; fair_comparison.json not saved",
        },
        {
            "molecule": "LiH(2e,3o)", "qubits": 6, "pair": "adapt_chem", "seed": 0,
            "status": "timeout_partial", "device": "cuda (Modal H100)",
            "cnots": 3, "err_mHa": 1.5544, "greedy_baseline_cnots": 16,
            "updates_completed": 35, "total_updates_planned": 55,
            "source": "terminals/697133.txt (lih6q adapt_chem_s0)",
            "note": "Killed at 7h; not on Modal volume",
        },
        {
            "molecule": "LiH(2e,3o)", "qubits": 6, "pair": "adapt_chem", "seed": 1,
            "status": "timeout_partial", "device": "cuda (Modal H100)",
            "cnots": 5, "err_mHa": 1.5544, "greedy_baseline_cnots": 16,
            "updates_completed": 35, "total_updates_planned": 55,
            "source": "terminals/697133.txt (lih6q adapt_chem_s1)",
            "note": "Killed at 7h; not on Modal volume",
        },
        {
            "molecule": "LiH(2e,5o)", "qubits": 10, "pair": "adapt_chem_rl_only", "seed": 0,
            "status": "timeout_no_result", "device": "cuda (Modal H100)",
            "cnots": None, "greedy_baseline_cnots": 60,
            "updates_completed": 0, "total_updates_planned": 20,
            "source": "terminals/695869.txt (lih10q)",
            "note": "7h timeout before first logged update",
        },
    ]

    greedy_timing = {
        "LiH(2e,3o)": {"greedy_adapt_prune_sec": 514.0, "greedy_adapt_cnots": 16, "err_mHa": 1.5544},
        "LiH(2e,5o)": {"greedy_adapt_prune_sec": 21360.5, "greedy_adapt_cnots": 60, "err_mHa": 1.4546},
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chem_acc_mHa": CHEM_MHA,
        "h2": h2_compare,
        "fair_greedy_baselines": fair_greedy,
        "exact_compare_8q": exact_compare,
        "baselines_8q": baselines_8q,
        "baselines_6q": baselines_6q,
        "fair_comparisons": fair_comparisons,
        "rl_prune_runs": rl_runs,
        "partial_runs_log_only": partial_runs,
        "greedy_timing_cpu": greedy_timing,
        "other": {
            "lih4_simul_prune_partial": load_json(ROOT / "results/lih4_simul_prune_partial.json"),
            "beat_greedy_hunt": load_json(ROOT / "results/beat_greedy/hunt_report.json"),
            "lih_raw_6q": load_json(ROOT / "results/lih_raw_lih_6q.json"),
            "lih_raw_8q": load_json(ROOT / "results/lih_raw_lih_8q.json"),
            "lih_raw_10q": load_json(ROOT / "results/lih_raw_lih_10q.json"),
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    master = build_master()
    out_path = OUT / "master_results.json"
    out_path.write_text(json.dumps(master, indent=2, default=str))
    print(f"Wrote {out_path}")
    print(f"  RL prune runs: {len(master['rl_prune_runs'])}")
    print(f"  Fair comparisons: {len(master['fair_comparisons'])}")


if __name__ == "__main__":
    main()
