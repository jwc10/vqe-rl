# moving energy target during training (Ostaszewski-style feedback)

from __future__ import annotations


class MovingThreshold:
    def __init__(self, initial, floor, bump=1e-4, shift_every=50,
                 streak_limit=25, bump_decay=1e-5):
        self.target = float(initial)
        self.floor = float(floor)
        self.bump = float(bump)
        self.bump_add = float(bump)
        self.shift_every = shift_every
        self.streak_limit = streak_limit
        self.bump_decay = bump_decay
        self.best_gap = float("inf")
        self.streak = 0
        self.since_shift = 0

    def note_episode(self, gap, ok):
        if gap < self.best_gap:
            self.best_gap = gap
        if ok:
            self.streak += 1
            if self.streak >= self.streak_limit:
                self.bump_add = max(0.0, self.bump_add - self.bump_decay)
                self.streak = 0
        else:
            self.streak = 0
            self.target = min(self.target + self.bump_add, self.best_gap + self.bump_add)

    def step_update(self, idx):
        self.since_shift += 1
        if self.since_shift >= self.shift_every and self.best_gap < float("inf"):
            self.target = max(self.floor, self.best_gap + self.bump_add)
            self.since_shift = 0
