#!/usr/bin/env python3
# Modal GPU runners for the LiH prune campaign (each job targets ~2-5h, hard cap 7h).

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).parent
VOL = modal.Volume.from_name("vqe-rl-results", create_if_missing=True)
VOL_PATH = "/vol"
_TIMEOUT = 60 * 60 * 7  # hard cap 7h per job

_BASELINE_FILES = [
    ROOT / "results" / "lih_campaign" / "baselines.json",
    ROOT / "results" / "lih_campaign" / "baselines_6q.json",
    ROOT / "results" / "fair_greedy_compare.json",
]

def _with_project_files(img: modal.Image) -> modal.Image:
    img = img.add_local_dir(
        str(ROOT),
        remote_path="/root/project",
        ignore=[".venv", "datasets", "results", ".git", "__pycache__"],
    )
    for bf in _BASELINE_FILES:
        if bf.exists():
            img = img.add_local_file(
                str(bf),
                remote_path=f"/root/project/{bf.relative_to(ROOT).as_posix()}",
            )
    return img


_core = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
)
_base_image = _with_project_files(_core)
# GPU VQE backend installed before local mounts (Modal build-order rule).
gpu_vqe_image = _with_project_files(
    _core.pip_install("pennylane-lightning-gpu>=0.45")
)

app = modal.App("vqe-rl-lih")


@app.function(
    image=_base_image,
    gpu="A10G",
    timeout=_TIMEOUT,
    volumes={VOL_PATH: VOL},
)
def run_lih_campaign(
    phase: str = "focus",
    preset: str = "modal_5h",
    orbitals: int = 4,
    seed: int = 0,
    pairs: str = "",
    skip_baselines: bool = True,
    skip_adapt_greedy: bool = True,
    run_tag: str = "",
):
    import os
    import shutil

    work = Path("/root/project")
    tag = run_tag or f"{pairs.replace(',', '_')}_s{seed}"
    out = work / "results" / "lih_campaign" / tag
    out.mkdir(parents=True, exist_ok=True)

    shared_bl = work / "results" / "lih_campaign" / "baselines.json"
    if skip_baselines and shared_bl.exists():
        shutil.copy(shared_bl, out / "baselines.json")
    elif skip_baselines and (Path(VOL_PATH) / "lih_campaign" / "baselines.json").exists():
        shutil.copy(Path(VOL_PATH) / "lih_campaign" / "baselines.json", out / "baselines.json")

    cmd = [
        sys.executable, str(work / "lih_campaign.py"),
        "--phase", phase,
        "--preset", preset,
        "--orbitals", str(orbitals),
        "--seed", str(seed),
        "--out-dir", str(out),
        "--skip-adapt-greedy",
    ]
    if pairs.strip():
        cmd.extend(["--pairs", *pairs.split()])
    if skip_baselines:
        cmd.append("--skip-baselines")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    print("CMD:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=work, env=env, check=True)

    vol_out = Path(VOL_PATH) / "lih_campaign" / tag
    vol_out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        if f.is_file():
            shutil.copy(f, vol_out / f.name)
    bl = out / "baselines.json"
    if bl.exists():
        top = Path(VOL_PATH) / "lih_campaign"
        top.mkdir(parents=True, exist_ok=True)
        shutil.copy(bl, top / "baselines.json")
    VOL.commit()

    report = out / "campaign_report.json"
    return {"tag": tag, "report": report.read_text() if report.exists() else "{}"}


@app.function(
    image=gpu_vqe_image,
    gpu="H100",
    timeout=_TIMEOUT,
    volumes={VOL_PATH: VOL},
)
def run_heavy_1double(seed: int = 10):
    """500-update 1-double prune with strong inner optimization."""
    import os
    import shutil

    os.environ["VQE_DEVICE"] = "lightning.gpu"
    work = Path("/root/project")
    tag = f"heavy_1double_s{seed}"
    out = work / "results" / "lih_campaign" / tag
    out.mkdir(parents=True, exist_ok=True)

    bl = work / "results" / "lih_campaign" / "baselines.json"
    if bl.exists():
        shutil.copy(bl, out / "baselines.json")

    cmd = [
        sys.executable, str(work / "lih_campaign.py"),
        "--phase", "focus",
        "--preset", "modal_heavy_1double",
        "--orbitals", "4",
        "--seed", str(seed),
        "--out-dir", str(out),
        "--pairs", "1double_chem",
        "--skip-baselines", "--skip-adapt-greedy",
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    print("CMD:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=work, env=env, check=True)

    vol_out = Path(VOL_PATH) / "lih_campaign" / tag
    vol_out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        if f.is_file():
            shutil.copy(f, vol_out / f.name)
    VOL.commit()
    return {"tag": tag, "fair": (out / "fair_comparison.json").read_text()}


@app.function(
    image=gpu_vqe_image,
    gpu="H100",
    timeout=_TIMEOUT,
    volumes={VOL_PATH: VOL},
)
def run_hybrid_simul_prune(seed: int = 0):
    """Givens+raw build to chem acc, compile, RL prune - end-to-end pipeline."""
    import os
    import shutil

    os.environ["VQE_DEVICE"] = "lightning.gpu"
    work = Path("/root/project")
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))
    tag = f"hybrid_simul_s{seed}"
    out = work / "results" / "lih_campaign" / tag
    out.mkdir(parents=True, exist_ok=True)

    from lih_campaign_config import MODAL_HYBRID_SIMUL, CHEM, TARGET_CHEM_CNOTS
    from rl_env import make_lih_config
    from train_raw import train, pick_device
    from vqe_core import pick_vqe_device

    cfg = make_lih_config(active_electrons=2, active_orbitals=4)
    hp = MODAL_HYBRID_SIMUL
    preset = hp["hybrid_preset"]
    print(f"LiH hybrid simul+prune | seed={seed} | VQE={pick_vqe_device(8)} | "
          f"PPO={pick_device()}", flush=True)

    t0 = time.perf_counter()
    _, _, hist, bests = train(
        cfg,
        num_updates=hp["updates"],
        episodes_per_update=hp["episodes_per_update"],
        hidden=hp["hidden"],
        order_k=hp["order_k"],
        inner_maxiter=hp["inner_maxiter"],
        inner_restarts=hp["inner_restarts"],
        ppo_epochs=hp["ppo_epochs"],
        max_gates=hp["max_gates"],
        max_givens_phase=hp["max_givens_phase"],
        gate_set="hybrid",
        hybrid_mode="simul_prune",
        hybrid_preset=preset,
        moving_threshold=True,
        target_mode="chem",
        seed=seed,
        log_every=hp["log_every"],
        cnot_penalty=0.02,
    )
    elapsed = time.perf_counter() - t0
    bc = bests["chem"]
    result = {
        "method": "hybrid_simul_prune",
        "seed": seed,
        "preset": "modal_hybrid_simul",
        "seconds": elapsed,
        "best_chem": bc,
        "goal_cnots": TARGET_CHEM_CNOTS,
        "meets_goal": (
            bc.get("error_vs_fci", float("inf")) < CHEM
            and bc.get("cnots", float("inf")) <= TARGET_CHEM_CNOTS
        ),
        "hist": {k: [float(x) if x == x else None for x in v]
                 for k, v in hist.items()},
    }
    (out / "hybrid_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str), flush=True)

    vol_out = Path(VOL_PATH) / "lih_campaign" / tag
    vol_out.mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "hybrid_result.json", vol_out / "hybrid_result.json")
    VOL.commit()
    return {"tag": tag, "cnots": bc.get("cnots"), "meets_goal": result["meets_goal"]}


@app.local_entrypoint()
def main(
    phase: str = "focus",
    preset: str = "modal_5h",
    orbitals: int = 4,
    seed: int = 0,
    pairs: str = "adapt_chem",
    skip_baselines: bool = True,
):
    with modal.enable_output():
        out = run_lih_campaign.remote(
            phase, preset, orbitals, seed, pairs, skip_baselines,
            skip_adapt_greedy=True,
        )
    print(out)


@app.local_entrypoint()
def bundle():
    """Parallel: 3x 1-double (modal_5h) + 2x ADAPT replicate."""
    jobs = [
        ("1double_chem", 0), ("1double_chem", 1), ("1double_chem", 2),
        ("adapt_chem", 1), ("adapt_chem", 2),
    ]
    with modal.enable_output():
        handles = [
            run_lih_campaign.spawn(
                "focus", "modal_5h", 4, seed, pairs, True, True,
                f"{pairs}_s{seed}",
            )
            for pairs, seed in jobs
        ]
        for h in handles:
            print(h.get())


@app.local_entrypoint()
def beef():
    """Heavy batch on H100: 3x 500-upd 1-double + 3x hybrid simul+prune."""
    with modal.enable_output():
        handles = []
        for seed in (10, 11, 12):
            handles.append(run_heavy_1double.spawn(seed))
        for seed in (0, 1, 2):
            handles.append(run_hybrid_simul_prune.spawn(seed))
        for h in handles:
            print(h.get())


@app.function(
    image=gpu_vqe_image,
    gpu="H100",
    timeout=_TIMEOUT,
    volumes={VOL_PATH: VOL},
)
def run_lih_6q(pair: str = "adapt_chem", seed: int = 0):
    """LiH(2e,3o) fair prune - faster than 8q; greedy ADAPT floor 16 vs 1-double 6."""
    import os
    import shutil

    os.environ["VQE_DEVICE"] = "lightning.gpu"
    work = Path("/root/project")
    tag = f"lih6q_{pair}_s{seed}"
    out = work / "results" / "lih_campaign" / tag
    out.mkdir(parents=True, exist_ok=True)

    bl6 = work / "results" / "lih_campaign" / "baselines_6q.json"
    if bl6.exists():
        data = json.loads(bl6.read_text())
        (out / "baselines.json").write_text(json.dumps(data, indent=2))

    cmd = [
        sys.executable, str(work / "lih_campaign.py"),
        "--phase", "focus",
        "--preset", "modal_5h",
        "--orbitals", "3",
        "--seed", str(seed),
        "--out-dir", str(out),
        "--pairs", pair,
        "--skip-baselines", "--skip-adapt-greedy",
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    print("CMD:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=work, env=env, check=True)

    vol_out = Path(VOL_PATH) / "lih_campaign" / tag
    vol_out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        if f.is_file():
            shutil.copy(f, vol_out / f.name)
    VOL.commit()
    fc = out / "fair_comparison.json"
    return {"tag": tag, "fair": fc.read_text() if fc.exists() else "{}"}


@app.local_entrypoint()
def lih6q():
    """LiH 6q (2e,3o): ADAPT + 1-double fair pairs in parallel (~2h total)."""
    jobs = [("adapt_chem", 0), ("adapt_chem", 1), ("1double_chem", 0)]
    with modal.enable_output():
        handles = [run_lih_6q.spawn(p, s) for p, s in jobs]
        for h in handles:
            print(h.get())


@app.function(
    image=gpu_vqe_image,
    gpu="H100",
    timeout=_TIMEOUT,
    volumes={VOL_PATH: VOL},
)
def run_lih_10q_rl_prune(seed: int = 0):
    """
    LiH(2e,5o) RL prune from ADAPT - no greedy baseline (too slow at 180 gates).
    Scalability probe: can RL compress to chem acc when greedy is infeasible?
    """
    import os
    import shutil

    os.environ["VQE_DEVICE"] = "lightning.gpu"
    work = Path("/root/project")
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))

    from adapt import adapt_vqe
    from lih_campaign_config import CHEM, MODAL_10Q_RL
    from prune_trainer import train_prune_campaign
    from raw_prune import adapt_to_raw_records, count_cnots
    from rl_env import make_lih_config
    from vqe_core import pick_vqe_device

    tag = f"lih10q_rl_s{seed}"
    out = work / "results" / "lih_campaign" / tag
    out.mkdir(parents=True, exist_ok=True)

    cfg = make_lih_config(active_electrons=2, active_orbitals=5)
    print(f"LiH(2e,5o) | VQE={pick_vqe_device(cfg['num_qubits'])} | seed={seed}", flush=True)
    ad = adapt_vqe(cfg, verbose=False)
    ar = adapt_to_raw_records(ad["actions"], ad["params"])
    start_cnots = count_cnots(ar)
    print(f"ADAPT start: {len(ar)} gates, {start_cnots} CNOTs", flush=True)

    hp = MODAL_10Q_RL
    t0 = time.perf_counter()
    res, _, _ = train_prune_campaign(
        cfg, ar, target=CHEM,
        label=f"RL_prune_10q_adapt_chem_s{seed}",
        seed=seed,
        num_updates=hp["updates"],
        episodes_per_update=hp["episodes_per_update"],
        hidden=hp["hidden"],
        order_k=hp["order_k"],
        inner_maxiter=hp["inner_maxiter"],
        inner_restarts=hp["inner_restarts"],
        use_greedy_bc=hp["use_greedy_bc"],
        greedy_baseline=None,
        log_every=1,
        out_dir=out,
    )
    elapsed = time.perf_counter() - t0
    report = {
        "molecule": "LiH(2e,5o)",
        "qubits": 10,
        "pair": "adapt_chem_rl_only",
        "greedy_adapt_chem_cnots": None,
        "greedy_skipped_reason": "greedy raw-prune infeasible in budget (~180 gates; >10h CPU est.)",
        "rl_result": res,
        "start_cnots": start_cnots,
        "start_gates": len(ar),
        "within_chem_acc": res.get("error_vs_fci", float("inf")) < CHEM,
        "seconds": elapsed,
        "seed": seed,
        "narrative": "RL scalability: automated prune without exhaustive greedy baseline",
    }
    (out / "scalability_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)

    vol_out = Path(VOL_PATH) / "lih_campaign" / tag
    vol_out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        if f.is_file():
            shutil.copy(f, vol_out / f.name)
    VOL.commit()
    return {"tag": tag, "rl_cnots": res.get("cnots"), "chem": report["within_chem_acc"]}


@app.local_entrypoint()
def lih10q():
    """LiH 10q RL-only prune (no greedy BC) - scalability story."""
    with modal.enable_output():
        print(run_lih_10q_rl_prune.remote(0))


@app.local_entrypoint()
def hybrid_only():
    """Relaunch hybrid simul+prune only (3 seeds in parallel, H100)."""
    with modal.enable_output():
        handles = [run_hybrid_simul_prune.spawn(s) for s in (0, 1, 2)]
        for h in handles:
            print(h.get())


if __name__ == "__main__":
    print("modal run modal_lih.py::beef  # heavy 1-double + hybrid on H100")
