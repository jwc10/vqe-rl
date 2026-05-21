# CS224R Final Project: RL for Quantum Circuit Construction

Minimal H2 VQE baseline using PennyLane (simulator only).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python h2_vqe.py
```

Prints Hartree-Fock and optimized VQE energies; saves `results/h2_vqe_convergence.png`.
