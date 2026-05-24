"""
Barebones REINFORCE for H2 circuit-structure search.

Run:
    python train_reinforce.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from h2_vqe import FCI_ENERGY, describe_actions, run_vqe_on_circuit
from rl_env import CircuitStructureEnv, action_key


class LinearSoftmaxPolicy:
    """Simple linear policy: logits = W @ state + b, masked softmax."""

    def __init__(self, state_dim: int, n_actions: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(n_actions, state_dim))
        self.b = np.zeros(n_actions)

    def probs(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits = self.W @ state + self.b
        logits = np.where(mask, logits, -1e9)
        logits -= logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def sample(self, state: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        probs = self.probs(state, mask)
        return int(rng.choice(len(probs), p=probs))

    def update(self, trajectory: list[tuple], return_g: float, lr: float) -> None:
        """One REINFORCE step: grad log pi(a|s) weighted by return G."""
        for state, action, mask in trajectory:
            probs = self.probs(state, mask)
            grad_logits = -probs
            grad_logits[action] += 1.0
            self.W += lr * return_g * np.outer(grad_logits, state)
            self.b += lr * return_g * grad_logits


def run_episode(env: CircuitStructureEnv, policy: LinearSoftmaxPolicy, rng: np.random.Generator):
    state, _ = env.reset()
    trajectory = []

    while True:
        mask = env.valid_action_mask()
        action = policy.sample(state, mask, rng)
        trajectory.append((state.copy(), action, mask.copy()))

        state, reward, done, info = env.step_gym(action)
        if done:
            return reward, trajectory, info


def brute_force_search(env: CircuitStructureEnv) -> dict:
    """
    Try all valid action sequences (order matters, no duplicate excitations).
    Exact baseline for small action spaces like H2.
    """
    best = {"reward": -np.inf, "actions": [], "energy": np.inf, "description": ""}

    def search(action_history: list[dict], used_keys: set[tuple]):
        vqe = run_vqe_on_circuit(env.H, env.num_qubits, env.hf_state, action_history)
        reward = env.hf_energy - vqe["energy"]
        if reward > best["reward"]:
            best.update(
                {
                    "reward": reward,
                    "actions": list(action_history),
                    "energy": vqe["energy"],
                    "description": describe_actions(action_history),
                }
            )

        if len(action_history) >= env.max_excitations:
            return

        for action in env.actions[1:]:
            key = action_key(action)
            if key in used_keys:
                continue
            search(action_history + [action], used_keys | {key})

    search([], set())
    return best


def train(
    num_episodes: int = 800,
    lr: float = 0.05,
    seed: int = 0,
    max_excitations: int = 4,
):
    rng = np.random.default_rng(seed)
    env = CircuitStructureEnv(max_excitations=max_excitations)
    policy = LinearSoftmaxPolicy(
        state_dim=env.n_excitation_actions,
        n_actions=len(env.actions),
        seed=seed,
    )

    returns = []
    best = {"reward": -np.inf}

    for ep in range(num_episodes):
        reward, trajectory, info = run_episode(env, policy, rng)
        returns.append(reward)
        policy.update(trajectory, reward, lr)

        if reward > best["reward"]:
            best = {"reward": reward, **info}

        if (ep + 1) % 100 == 0:
            avg = np.mean(returns[-100:])
            print(f"episode {ep + 1:4d} | avg reward (last 100): {avg:.6f} | best: {best['reward']:.6f}")

    return env, policy, returns, best


def plot_returns(returns: list[float], out_path: Path):
    window = 50
    smoothed = [
        np.mean(returns[max(0, i - window + 1) : i + 1]) for i in range(len(returns))
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(returns, alpha=0.3, label="episode reward")
    ax.plot(smoothed, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (E_HF - E)")
    ax.set_title("REINFORCE training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("=== Brute-force baseline (exhaustive search) ===")
    env = CircuitStructureEnv(max_excitations=4)
    print(f"Action space ({len(env.actions)} actions):")
    for i, a in enumerate(env.actions):
        print(f"  {i}: {a}")
    print(f"HF energy: {env.hf_energy:.8f} Ha\n")

    brute = brute_force_search(env)
    print(f"Best brute-force reward: {brute['reward']:.8f}")
    print(f"  energy: {brute['energy']:.8f} Ha")
    print(f"  circuit: {brute['description']}\n")

    print("=== REINFORCE training ===")
    env, policy, returns, best = train(num_episodes=800, lr=0.05, max_excitations=4)

    print(f"\nBest RL reward: {best['reward']:.8f}")
    print(f"  energy: {best.get('energy', float('nan')):.8f} Ha")
    print(f"  circuit: {best.get('description', '?')}")
    print(f"FCI reference: {FCI_ENERGY:.8f} Ha")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    plot_returns(returns, out_dir / "reinforce_returns.png")
    print(f"Saved plot to {out_dir / 'reinforce_returns.png'}")


if __name__ == "__main__":
    main()
