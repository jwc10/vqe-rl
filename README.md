# CS224R Final Project: RL for Quantum Circuit Construction

RL searches over VQE circuit *structure* (which excitation gates to include); classical
VQE (scipy BFGS) optimizes the gate angles for the chosen structure. Reward is sparse,
given at STOP: `R = (E_HF - E_vqe) - lam * num_excitations`. Simulator only (PennyLane).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python h2_vqe.py          # H2 VQE baseline (hand-picked structures)
python train_reinforce.py # H2: REINFORCE + baseline, vs brute-force search
python lih_vqe.py         # LiH VQE baseline, prints HF/FCI, profiles timing
python adapt.py           # ADAPT-VQE greedy baseline on LiH
python train_lih.py       # LiH: PPO + REINFORCE, best circuits + curves + Pareto
python compare_lih.py     # LiH: ADAPT vs PPO vs PPO+warmstart vs random, all vs FCI
```

## Layout

- `vqe_core.py` molecule-agnostic VQE: build circuit from excitations, BFGS inner loop, exact FCI
- `h2_vqe.py` / `lih_vqe.py` Hamiltonian builders + VQE baselines per molecule
- `rl_env.py` env + molecule configs (`make_h2_config`, `make_lih_config`); action space is
  STOP + singles + doubles
- `train_reinforce.py` linear-softmax REINFORCE with a return baseline
- `ppo.py` PyTorch actor-critic PPO (shared MLP, masked policy head + value head) plus
  ADAPT warm-start via behavioral cloning
- `adapt.py` ADAPT-VQE greedy baseline (gradient-screened operator selection)
- `train_lih.py` LiH RL training driver (PPO vs REINFORCE, Pareto sweep)
- `compare_lih.py` method comparison vs FCI (ADAPT / PPO / PPO+warmstart / random),
  metrics: energy error, circuit depth, #VQE optimizations

## Notes

- Full LiH (STO-3G) is 12 qubits / 4 electrons (92 excitations); a single UCCSD VQE call is
  ~47s. `train_lih.py` defaults to the frozen-core active space (2e, 5o = 10 qubits, 24
  excitations) which reaches FCI in ~2s and keeps ~96% of the correlation energy. Run
  `lih_vqe.main(active_electrons=2, active_orbitals=5)` for the active-space baseline.
- The action space (singles + doubles + STOP) is the standard UCCSD pool. Real chemistry has
  more: generalized/spin-adapted excitations (k-UpCCGSD), adaptive operator pools with
  repetition (ADAPT-VQE), and gate-ordering effects. `build_action_space` is kept simple so a
  richer pool can be dropped in later.
- Framing: this is not a claim that quantum beats classical. `lightning.qubit` is a classical
  simulator of a quantum algorithm, and classical FCI/CCSD(T) already get LiH near-exact. The
  goal is automated ansatz discovery and compact circuits (a NISQ-era workflow). No method
  here is guaranteed to find the ground state except FCI; UCCSD/ADAPT/RL are heuristics that
  happen to do well on small single-reference molecules like H2/LiH. We compare every method
  against simulated FCI on energy error, circuit depth, and #VQE optimizations.
- `max_excitations` is set to ADAPT's converged depth (8 on this active space) so RL can reach
  FCI rather than being capped below what standard methods use.
