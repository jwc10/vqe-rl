# PPO actor-critic for structure search. Reward is terminal-only, so the return for every
# step is that terminal reward (gamma=1) and the advantage is return - value.

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_env import CircuitStructureEnv

NEG_INF = -1e9


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state, mask):
        h = self.backbone(state)
        logits = self.policy_head(h)
        logits = torch.where(mask, logits, torch.full_like(logits, NEG_INF))
        value = self.value_head(h).squeeze(-1)
        return logits, value


def _dist(logits):
    return torch.distributions.Categorical(logits=logits)


def collect_episode(env, net, rng):
    state, _ = env.reset()
    steps = []
    while True:
        mask = env.valid_action_mask()
        s = torch.tensor(state, dtype=torch.float32)
        m = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            logits, value = net(s, m)
            dist = _dist(logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        steps.append({
            "state": s, "mask": m, "action": action,
            "logp": logp, "value": value,
        })
        state, reward, done, info = env.step_gym(int(action))
        if done:
            return steps, reward, info


def _action_index(env, action):
    from rl_env import action_key
    key = action_key(action)
    for i, a in enumerate(env.actions):
        if action_key(a) == key:
            return i
    raise ValueError(f"action {action} not in pool")


def behavioral_clone(net, env, demo_actions, epochs=300, lr=5e-3):
    """Warm-start: pretrain the policy to imitate a demo sequence (ADAPT's operator order,
    ending in STOP) via cross-entropy on (state, mask) -> chosen action."""
    # roll the demo through the env so the (state, mask, target) tuples stay consistent
    data = []
    state, _ = env.reset()
    for a in demo_actions:
        mask = env.valid_action_mask()
        idx = _action_index(env, a)
        data.append((state.copy(), mask.copy(), idx))
        state, _, done, _ = env.step_gym(idx)
        if done:
            break
    data.append((state.copy(), env.valid_action_mask().copy(), 0))  # finish with STOP

    states = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
    masks = torch.tensor(np.array([d[1] for d in data]), dtype=torch.bool)
    targets = torch.tensor([d[2] for d in data], dtype=torch.long)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        logits, _ = net(states, masks)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def train_ppo(
    config,
    num_updates: int = 60,
    episodes_per_update: int = 16,
    epochs: int = 4,
    lr: float = 3e-3,
    clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_excitations: int = 4,
    lam: float = 0.0,
    seed: int = 0,
    log_every: int = 10,
    warm_start_actions=None,
    bc_epochs: int = 300,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = CircuitStructureEnv(config=config, max_excitations=max_excitations, lam=lam)

    net = ActorCritic(env.n_excitation_actions, len(env.actions))
    if warm_start_actions is not None:
        behavioral_clone(net, env, warm_start_actions, epochs=bc_epochs)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    returns_hist = []
    best = {"reward": -np.inf}

    for update in range(num_updates):
        states, masks, actions, old_logps, rets = [], [], [], [], []
        batch_rewards = []

        for _ in range(episodes_per_update):
            steps, reward, info = collect_episode(env, net, rng)
            batch_rewards.append(reward)
            returns_hist.append(reward)
            if reward > best["reward"]:
                best = {"reward": reward, **info}
            for st in steps:
                states.append(st["state"])
                masks.append(st["mask"])
                actions.append(st["action"])
                old_logps.append(st["logp"])
                rets.append(reward)  # terminal reward is the return for every step

        states = torch.stack(states)
        masks = torch.stack(masks)
        actions = torch.stack(actions)
        old_logps = torch.stack(old_logps).detach()
        rets = torch.tensor(rets, dtype=torch.float32)

        for _ in range(epochs):
            logits, values = net(states, masks)
            dist = _dist(logits)
            logps = dist.log_prob(actions)
            ratio = torch.exp(logps - old_logps)

            advantage = (rets - values).detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, rets)
            entropy = dist.entropy().mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            opt.zero_grad()
            loss.backward()
            opt.step()

        if log_every and (update + 1) % log_every == 0:
            avg = float(np.mean(batch_rewards))
            print(f"update {update + 1:3d} | avg reward: {avg:.6f} | best: {best['reward']:.6f} "
                  f"| structures seen: {len(env._vqe_cache)}")

    return net, returns_hist, best, env
