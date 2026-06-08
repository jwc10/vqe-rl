# Shared hyperparameters + fair-comparison targets.

from __future__ import annotations

from dataclasses import dataclass, asdict

CHEM = 1.6e-3
EXACT = 1e-6

# Primary LiH goal: match or beat greedy prune from 1-double compile
TARGET_CHEM_CNOTS = 7

# Fair comparison pairs: same start circuit → greedy vs RL prune
FAIR_PAIRS = {
    "1double_chem": {
        "label": "1-double compile",
        "greedy_key": "greedy_1double_chem",
        "rl_label": "RL_prune_1double_chem",
        "target": CHEM,
        "goal_cnots": TARGET_CHEM_CNOTS,
        "priority": 1,
    },
    "adapt_chem": {
        "label": "full ADAPT compile",
        "greedy_key": "greedy_adapt_chem",
        "rl_label": "RL_prune_ADAPT_chem",
        "target": CHEM,
        "goal_cnots": None,  # greedy floor is ~23 @ chem; 7 is a stretch goal
        "priority": 2,
    },
    "adapt_exact": {
        "label": "full ADAPT compile",
        "greedy_key": "greedy_adapt_exact",
        "rl_label": "RL_prune_ADAPT_exact",
        "target": EXACT,
        "goal_cnots": 25,
        "priority": 3,  # skip unless extra time
    },
}


@dataclass
class TrainPreset:
    name: str
    use_greedy_bc: bool = True
    updates: int = 200
    episodes_per_update: int = 24
    hidden: int = 512
    order_k: int = 4
    gamma: float = 0.97
    lr: float = 3e-3
    entropy_coef: float = 0.02
    ppo_epochs: int = 4
    clip: float = 0.2
    inner_restarts: int = 0
    inner_maxiter: int = 60
    max_gates: int = 18
    moving_threshold: bool = True
    curriculum: bool = True
    initial_target: float = 0.005
    number_penalty: float = 0.15
    cnot_penalty: float = 0.0
    gate_set: str = "raw"
    reward_mode: str = "ostaszewski"
    # optional: fewer updates for secondary pair in same job
    updates_adapt: int | None = None

    def as_train_kw(self) -> dict:
        d = asdict(self)
        d.pop("name")
        d.pop("updates_adapt", None)
        return d


SMOKE = TrainPreset(
    name="smoke",
    updates=15,
    episodes_per_update=8,
    hidden=128,
    order_k=2,
    max_gates=14,
    inner_maxiter=40,
)

# ~5 hr Modal budget: focus on prune pairs, skip scratch/hybrid/exact
MODAL_5H = TrainPreset(
    name="modal_5h",
    updates=100,          # 1-double (primary, aim for 7 CNOTs)
    updates_adapt=55,       # ADAPT pair (secondary)
    episodes_per_update=24,
    hidden=512,
    order_k=4,
    inner_maxiter=60,
    max_gates=14,
    moving_threshold=True,
)

# Longer local / multi-session (not for 5h cap)
FULL_SCALE_LIH = TrainPreset(
    name="full_scale_lih",
    updates=250,
    episodes_per_update=32,
    hidden=512,
    order_k=4,
    max_gates=20,
    inner_maxiter=100,
    inner_restarts=1,
)

HYBRID_UNIQUE = TrainPreset(
    name="hybrid_unique",
    updates=80,
    episodes_per_update=16,
    hidden=512,
    order_k=4,
    gate_set="hybrid",
    max_gates=14,
    inner_maxiter=60,
)

# Heavy 1-double prune: ~1-2h GPU, 500 updates, stronger inner opt (still unlikely <7 CNOTs)
MODAL_HEAVY_1DOUBLE = TrainPreset(
    name="modal_heavy_1double",
    updates=500,
    episodes_per_update=40,
    hidden=768,
    order_k=5,
    inner_maxiter=120,
    inner_restarts=2,
    ppo_epochs=6,
    max_gates=14,
    moving_threshold=True,
)

# LiH 2e,5o (10q): skip greedy BC/trace (prohibitive at 180 gates), RL-only scalability probe
MODAL_10Q_RL = {
    "updates": 20,
    "episodes_per_update": 8,
    "hidden": 512,
    "order_k": 3,
    "inner_maxiter": 80,
    "inner_restarts": 1,
    "use_greedy_bc": False,
}

MODAL_HYBRID_SIMUL = {
    "updates": 45,
    "episodes_per_update": 14,
    "hidden": 768,
    "order_k": 4,
    "inner_maxiter": 90,
    "inner_restarts": 1,
    "ppo_epochs": 5,
    "max_gates": 14,
    "max_givens_phase": 5,
    "log_every": 1,
    "hybrid_preset": {
        "tier": "modal_heavy",
        "max_compile_gates": 40,
        "prune_max_steps": 28,
        "prune_order_k": 4,
    },
}

PRESETS = {
    "smoke": SMOKE,
    "modal_5h": MODAL_5H,
    "modal_heavy_1double": MODAL_HEAVY_1DOUBLE,
    "full_scale": FULL_SCALE_LIH,
    "hybrid": HYBRID_UNIQUE,
}
