# RL picks which gate to drop from a compiled circuit

from __future__ import annotations

import numpy as np
import torch

from raw_prune import _optimized_energy, count_cnots


class RawGatePruneEnv:
    def __init__(self, config, start_records, target=1.6e-3, strict=False,
                 reward_mode="ostaszewski", order_k=3, max_n=None,
                 max_steps=0, inner_restarts=1, inner_maxiter=100):
        self.H = config["H"]
        self.nq = config["num_qubits"]
        self.hf = config["hf_state"]
        self.hf_energy = config["hf_energy"]
        self.fci = config["fci_energy"]
        self.target = target
        self.strict = strict
        self.reward_mode = reward_mode
        self.order_k = order_k
        self.start = [dict(r) for r in start_records]
        self.n = len(self.start)
        self.max_n = max_n if max_n is not None else self.n
        if self.n > self.max_n:
            raise ValueError(f"{self.n} gates > max_n={self.max_n}")
        self.start_cnots = count_cnots(self.start)
        self.max_steps = max_steps  # 0 = no cap
        self.inner_restarts = inner_restarts
        self.inner_maxiter = inner_maxiter
        self.step_count = 0
        self.reset()

    @property
    def n_actions(self):
        return self.max_n + 1  # drop gate i, or stop

    @property
    def state_dim(self):
        return self.max_n + 1 + self.order_k * self.max_n

    def _alive(self):
        return [dict(self.start[i]) for i in range(self.n) if self.alive[i]]

    def _obs(self):
        frac = (self.hf_energy - self.energy) / (self.hf_energy - self.fci + 1e-12)
        mask = np.zeros(self.max_n, dtype=np.float64)
        mask[: self.n] = self.alive.astype(np.float64)
        hist = np.zeros(self.order_k * self.max_n, dtype=np.float64)
        for slot, idx in enumerate(reversed(self.removed[-self.order_k:])):
            if 0 <= idx < self.n:
                hist[slot * self.max_n + idx] = 1.0
        return np.concatenate([mask, [frac], hist])

    def reset(self):
        self.alive = np.ones(self.n, dtype=bool)
        self.removed = []
        self.step_count = 0
        self.energy = _optimized_energy(
            self.H, self.nq, self.hf, self._alive(),
            extra_restarts=self.inner_restarts, maxiter=self.inner_maxiter)
        return self._obs(), {}

    def _vqe(self, recs):
        return _optimized_energy(self.H, self.nq, self.hf, recs,
                                 extra_restarts=self.inner_restarts, maxiter=self.inner_maxiter)

    def valid_action_mask(self):
        m = np.zeros(self.max_n + 1, dtype=bool)
        m[: self.n] = self.alive
        m[self.max_n] = True
        return m

    def _done_info(self):
        recs = self._alive()
        gap = self.energy - self.fci
        return {
            "energy": self.energy,
            "error_vs_fci": gap,
            "cnots": count_cnots(recs),
            "n_gates": len(recs),
            "removed_cnots": self.start_cnots - count_cnots(recs),
            "within_target": gap < self.target,
            "within_chem_acc": gap < 1.6e-3,
            "within_exact": gap < 1e-6,
        }

    def _removal_reward(self, idx, ok, old_c, new_c):
        if not ok:
            return -1.0 if self.reward_mode == "ostaszewski" else -0.5
        saved = old_c - new_c
        extra = 0.5 if self.start[idx]["name"] == "CNOT" else 0.1
        return 2.0 * saved + extra

    def step_gym(self, action_idx):
        self.step_count += 1
        if self.max_steps and self.step_count >= self.max_steps:
            info = self._done_info()
            ok = info["within_target"]
            r = 5.0 if ok and self.reward_mode == "ostaszewski" else -5.0
            return self._obs(), r, True, info

        if action_idx == self.max_n:
            info = self._done_info()
            ok = info["within_target"]
            if self.reward_mode == "ostaszewski":
                r = 5.0 if ok else -5.0
            else:
                r = 1.0 if ok else -5.0
            return self._obs(), r, True, info

        if action_idx >= self.n or not self.alive[action_idx]:
            return self._obs(), -0.1, False, {}

        trial_alive = self.alive.copy()
        trial_alive[action_idx] = False
        trial = [dict(self.start[i]) for i in range(self.n) if trial_alive[i]]
        old_c = count_cnots(self._alive())
        new_e = self._vqe(trial) if trial else self.hf_energy
        new_c = count_cnots(trial) if trial else 0

        if new_e - self.fci < self.target:
            self.alive = trial_alive
            self.energy = new_e
            self.removed.append(action_idx)
            r = self._removal_reward(action_idx, True, old_c, new_c)
            if not self.alive.any():
                info = self._done_info()
                if info["within_target"] and self.reward_mode == "ostaszewski":
                    r += 5.0
                return self._obs(), r, True, info
            return self._obs(), r, False, {}

        if self.strict:
            info = self._done_info()
            r = -5.0 if self.reward_mode == "ostaszewski" else -0.5
            return self._obs(), r, True, info
        return self._obs(), self._removal_reward(action_idx, False, old_c, new_c), False, {}


def train_prune(config, start_records, target=1.6e-3, num_updates=80,
                episodes_per_update=24, gamma=0.97, lr=3e-3, entropy_coef=0.02,
                seed=0, log_every=1, strict=False, hidden=256, device=None,
                max_steps=0, inner_restarts=1, inner_maxiter=100, order_k=3):
    from train_raw import ActorCritic, discounted_returns
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = RawGatePruneEnv(config, start_records, target=target, strict=strict, order_k=order_k,
                          max_steps=max_steps, inner_restarts=inner_restarts,
                          inner_maxiter=inner_maxiter)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(env.state_dim, env.n_actions, hidden=hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best = {"cnots": env.start_cnots, "n_gates": env.n, "error_vs_fci": None}

    for upd in range(num_updates):
        S, M, A, R = [], [], [], []
        removed = []
        for _ in range(episodes_per_update):
            state, _ = env.reset()
            traj = []
            while True:
                mask = env.valid_action_mask()
                s = torch.tensor(state, dtype=torch.float32, device=dev)
                m = torch.tensor(mask, dtype=torch.bool, device=dev)
                with torch.no_grad():
                    logits, _ = net(s, m)
                    a = torch.distributions.Categorical(logits=logits).sample()
                state, r, done, info = env.step_gym(int(a))
                traj.append({"s": s, "m": m, "a": a, "r": r})
                if done:
                    break
            rets = discounted_returns([t["r"] for t in traj], gamma)
            for t, ret in zip(traj, rets):
                S.append(t["s"]); M.append(t["m"]); A.append(t["a"]); R.append(ret)
            if info.get("within_target") and info["cnots"] < best["cnots"]:
                best = {"cnots": info["cnots"], "n_gates": info["n_gates"],
                        "error_vs_fci": info["error_vs_fci"],
                        "within_exact": info.get("within_exact", False)}
            removed.append(info.get("removed_cnots", 0))

        S = torch.stack(S); M = torch.stack(M); A = torch.stack(A)
        R = torch.tensor(R, dtype=torch.float32, device=dev)
        logits, values = net(S, M)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(A)
        adv = (R - values).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss = -(logp * adv).mean() + 0.5 * torch.nn.functional.mse_loss(values, R) \
            - entropy_coef * dist.entropy().mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if log_every and (upd + 1) % log_every == 0:
            err = best.get("error_vs_fci")
            err_s = f"{err:+.2e}" if err is not None else "--"
            print(f"update {upd+1:4d}/{num_updates} | avg removed {np.mean(removed):.1f} | "
                  f"best {best['cnots']} CNOTs | err {err_s}", flush=True)

    return best
