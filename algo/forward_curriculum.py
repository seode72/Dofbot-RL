"""
ForwardCurriculumLearner — RFCL Stage 2 (§4.2)

Prioritized Level Replay (PLR)-style forward curriculum over initial states.
Prioritizes states at the edge of the policy's ability (medium difficulty).

Paper Eq. (1):
  P_S(s | Λ, S) = rank(S_i)^{-1/β} / Σ_j rank(S_j)^{-1/β}
  P_C(s | Λ, C, c) = (c - C_i) / Σ_{C_j} (c - C_j)

Paper Eq. (2):
  P_ρ(s) = P_S(s) + P_C(s)
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SeedState:
    """Per-seed curriculum state."""
    seed: int
    score: float = 2.0        # initial score = 2 (unseen)
    last_sampled_ep: int = 0  # episode count when last sampled (for staleness)
    returns: deque = field(default_factory=lambda: deque([0.0] * 20, maxlen=20))
    successes: deque = field(default_factory=lambda: deque([0] * 20, maxlen=20))


class ForwardCurriculumLearner:
    """
    RFCL Stage 2: Forward curriculum via seed-based initial state prioritization.

    Paper §4.2 methods:
      - compute_score()      : score function q → {1, 2, 3} (Eq. 1 scoring)
      - compute_P_S()        : rank-based score priority (Eq. 1 left)
      - compute_P_C()        : staleness priority (Eq. 1 right)
      - sample_initial_state(): P_S + P_C combined (Eq. 2)
      - record_episode()     : update returns/successes, recompute score
    """

    def __init__(
        self,
        num_seeds: int = 1000,
        omega: float = 0.75,   # ω: threshold for "easy" (score=1 if q ≥ ω)
        beta: float = 0.1,     # β: rank temperature for P_S
        staleness_coef: float = 0.1,   # mix weight for P_C
        staleness_beta: float = 0.1,   # rank temperature for P_C
        nu: float = 0.5,       # prob of sampling seen vs unseen seed
        score_window: int = 5, # last k episodes for q computation
        num_envs: int = 1,
    ):
        self.omega = omega
        self.beta = beta
        self.staleness_coef = staleness_coef
        self.staleness_beta = staleness_beta
        self.nu = nu
        self.score_window = score_window

        seeds = np.arange(num_seeds)
        self.seeds = seeds
        self.seed_to_idx: dict[int, int] = {int(s): i for i, s in enumerate(seeds)}
        self.seed_states: list[SeedState] = [SeedState(seed=int(s)) for s in seeds]

        # Track seen/unseen
        self.seen = np.zeros(num_seeds, dtype=bool)
        # Pre-mark first batch as seen (paper: uniformly sample n initial states first)
        for i in range(0, num_seeds, num_envs):
            for j in range(min(num_envs, num_seeds - i)):
                self.seen[i + j] = True
                self.seed_states[i + j].score = 2.0

        self.total_episodes = 0

    # ------------------------------------------------------------------
    # Paper §4.2: Core methods
    # ------------------------------------------------------------------

    def compute_score(self, seed_state: SeedState) -> float:
        """
        Score function from paper §4.2:
          q = fraction of last k episodes with nonzero return
          q == 0        → score = 2  (never succeeded, medium-hard)
          0 < q < ω     → score = 3  (sometimes succeeds, highest priority)
          q >= ω        → score = 1  (consistently succeeds, easy)
        """
        arr = np.array(list(seed_state.returns)[-self.score_window:])
        q = float((arr > 0).sum()) / len(arr)
        if q == 0.0:
            return 2.0
        if q < self.omega:
            return 3.0
        return 1.0

    def compute_P_S(self, scores: np.ndarray) -> np.ndarray:
        """
        Rank-based score priority (Eq. 1 left):
          P_S(i) ∝ rank(S_i)^{-1/β}
        Higher score → higher rank → higher priority.
        """
        ranks = self._rankmin(scores) + 1  # 1-indexed
        weights = (1.0 / ranks) ** (1.0 / self.beta)
        return weights / weights.sum()

    def compute_P_C(self, last_sampled: np.ndarray, c: int) -> np.ndarray:
        """
        Staleness priority (Eq. 1 right):
          P_C(i) = (c - C_i) / Σ_j (c - C_j)
        Seeds not sampled recently get higher staleness weight.
        """
        staleness = c - last_sampled  # larger = more stale
        staleness = np.maximum(staleness, 0)
        ranks = self._rankmin(staleness) + 1
        weights = (1.0 / ranks) ** (1.0 / self.staleness_beta)
        z = weights.sum()
        return weights / z if z > 0 else np.ones_like(weights) / len(weights)

    def sample_initial_state(self, count: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample initial state seeds using P_ρ = P_S + P_C (Eq. 2).
        Returns (seeds, indices).
        """
        seen_mask = self.seen
        unseen_mask = ~seen_mask
        n_unseen = unseen_mask.sum()

        if n_unseen == 0 or np.random.rand() >= (1.0 - self.nu):
            indices = self._sample_by_priority(count, mask=seen_mask)
        else:
            indices = self._sample_unseen(count)

        self._update_staleness(indices)
        return self.seeds[indices], indices

    def record_episode(self, seed: int, final_return: float, success: bool) -> None:
        """Update seed history and recompute score."""
        idx = self.seed_to_idx[int(seed)]
        ss = self.seed_states[idx]
        self.seen[idx] = True
        ss.returns.append(final_return)
        ss.successes.append(int(success))
        ss.score = self.compute_score(ss)
        self.total_episodes += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_by_priority(self, count: int, mask: np.ndarray) -> np.ndarray:
        """Sample using P_S + P_C combined."""
        scores = np.array([ss.score for ss in self.seed_states])
        last_sampled = np.array([ss.last_sampled_ep for ss in self.seed_states])

        p_s = self.compute_P_S(scores)
        p_c = self.compute_P_C(last_sampled, self.total_episodes)
        weights = (1.0 - self.staleness_coef) * p_s + self.staleness_coef * p_c
        weights = weights * mask.astype(float)

        z = weights.sum()
        weights = weights / z if z > 0 else mask.astype(float) / mask.sum()
        return np.random.choice(len(self.seeds), size=count, p=weights)

    def _sample_unseen(self, count: int) -> np.ndarray:
        unseen_mask = (~self.seen).astype(float)
        probs = unseen_mask / unseen_mask.sum()
        return np.random.choice(len(self.seeds), size=count, p=probs)

    def _update_staleness(self, indices: np.ndarray) -> None:
        for idx in indices:
            self.seed_states[idx].last_sampled_ep = self.total_episodes

    @staticmethod
    def _rankmin(x: np.ndarray) -> np.ndarray:
        """Rank with ties broken by minimum (same value → same rank)."""
        u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        csum = np.zeros_like(counts)
        csum[1:] = counts[:-1].cumsum()
        return csum[inv]
