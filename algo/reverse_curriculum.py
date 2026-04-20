"""
ReverseCurriculumLearner — RFCL Stage 1 (§4.1)

Per-demonstration reverse curriculum via state resets.
Each demo τ_i has its own frontier start_step t_i, initially T_i (end of demo).
The curriculum progresses backward: t_i ← t_i - δ whenever the agent
achieves ≥ threshold success rate from the current frontier.
"""

from __future__ import annotations
import h5py
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass, field


@dataclass
class DemoState:
    """Per-demonstration curriculum state."""
    total_steps: int
    start_step: int = 0       # t_i: current frontier (starts at T_i)
    solved: bool = False
    success_buffer: deque = field(default_factory=deque)

    def __post_init__(self):
        # t_i = T_i initially (start from goal state)
        self.start_step = max(self.total_steps - 1, 0)


class ReverseCurriculumLearner:
    """
    RFCL Stage 1: Per-demonstration reverse curriculum.

    Paper §4.1 methods:
      - sample_demo()          : sample τ_i proportional to t_i / T_i
      - sample_start_state()   : geometric sampling near frontier → s_{i, t+k}
      - get_episode_timelimit(): dynamic horizon = 1 + (T_i - t) * φ^{-1}
      - record_episode()       : update success buffer, advance frontier if threshold met
      - is_complete()          : True when all demos reach t_i = 0
      - generate_reset()       : convenience = sample_demo + sample_start_state
    """

    def __init__(
        self,
        h5_path: str,
        reverse_step_size: int = 8,       # δ: steps to move frontier back
        success_buffer_size: int = 24,     # m: window for success rate
        threshold: float = 0.9,            # advance when success_rate ≥ threshold
        phi: float = 1.0,                  # φ: timelimit scale factor (§4.1)
    ):
        self.delta = reverse_step_size
        self.m = success_buffer_size
        self.threshold = threshold
        self.phi = phi

        # Demo trajectory storage
        self.joint_pos: dict[int, torch.Tensor] = {}   # [T, 7]
        self.joint_vel: dict[int, torch.Tensor] = {}   # [T, 7]
        self.cube_state: dict[int, torch.Tensor] = {}  # [T, 13]
        self.prev_action: dict[int, torch.Tensor] = {} # [T, 6]
        self.demo_state: dict[int, DemoState] = {}

        self._load_demos(h5_path)
        self.demo_ids = list(self.demo_state.keys())
        print(f"[ReverseCurriculumLearner] Loaded {len(self.demo_ids)} demos from {h5_path}")

    # ------------------------------------------------------------------
    # Paper §4.1: Core methods
    # ------------------------------------------------------------------

    def sample_demo(self) -> int:
        """
        Sample demo τ_i with probability proportional to t_i / T_i.
        Demonstrations further from solved (larger t_i) are sampled more.
        """
        weights = np.array([
            self.demo_state[d].start_step / self.demo_state[d].total_steps
            if self.demo_state[d].start_step > 0 else 1e-6
            for d in self.demo_ids
        ], dtype=np.float64)
        weights /= weights.sum()
        return int(np.random.choice(self.demo_ids, p=weights))

    def sample_start_state(self, demo_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Geometric sampling near frontier: k ~ K, reset to s_{i, t_i + k}.
        Returns (joint_pos, joint_vel, cube_state, prev_action, steps_back).
        """
        ds = self.demo_state[demo_id]
        # Geometric-like: probability mass concentrated at frontier
        geo_probs = [0.5, 0.25, 0.125, 0.0625, 0.0625]
        offsets = [min(ds.start_step + k, ds.total_steps - 1) for k in range(5)]
        step = int(np.random.choice(offsets, p=geo_probs))
        steps_back = ds.total_steps - step
        return (
            self.joint_pos[demo_id][step],
            self.joint_vel[demo_id][step],
            self.cube_state[demo_id][step],
            self.prev_action[demo_id][step],
            steps_back,
        )

    def get_episode_timelimit(self, demo_id: int, steps_back: int) -> int:
        """
        Dynamic episode timelimit: H = 1 + (T_i - t_i) * φ^{-1}  (§4.1).
        Shorter horizon when starting near goal; longer when starting far.
        """
        T_i = self.demo_state[demo_id].total_steps
        t_i = T_i - steps_back
        return int(1 + (T_i - t_i) / self.phi)

    def record_episode(self, demo_id: int, steps_back: int, success: bool) -> None:
        """
        Record episode outcome. Advance frontier t_i ← t_i - δ if
        success_rate ≥ threshold over last m episodes at current frontier.
        """
        ds = self.demo_state[demo_id]
        current_steps_back = ds.total_steps - ds.start_step

        if steps_back != current_steps_back:
            return  # stale episode, ignore

        ds.success_buffer.append(1 if success else 0)
        rate = sum(ds.success_buffer) / len(ds.success_buffer)

        if rate >= self.threshold:
            # Reset buffer and advance frontier
            ds.success_buffer = deque([0] * self.m, maxlen=self.m)
            if ds.start_step > 0:
                ds.start_step = max(ds.start_step - self.delta, 0)
                print(f"[RCL] Demo {demo_id}: frontier → step {ds.start_step}")
            elif not ds.solved:
                ds.solved = True
                print(f"[RCL] Demo {demo_id}: SOLVED (frontier reached t=0)")

    def is_complete(self) -> bool:
        """True when all demonstrations have been reverse-solved (t_i = 0)."""
        return all(self.demo_state[d].solved for d in self.demo_ids)

    def solved_fraction(self) -> float:
        """Fraction of demos currently solved."""
        return sum(self.demo_state[d].solved for d in self.demo_ids) / len(self.demo_ids)

    def log_state(self) -> dict:
        """Return per-demo progress for logging (compatible with Dofbot-RL-main format)."""
        fracs, solved_count, details = [], 0, {}
        for d in self.demo_ids:
            ds = self.demo_state[d]
            if ds.solved:
                solved_count += 1
            total = ds.total_steps
            frac = ds.start_step / (total - 1) if total > 1 else 0.0
            fracs.append(frac)
            details[f"demo_{d}"] = frac
        return {
            "solved_frac": solved_count / len(self.demo_ids) if self.demo_ids else 0.0,
            "mean_start_step_frac": sum(fracs) / len(fracs) if fracs else 0.0,
            "details": details,
        }

    def generate_reset(self, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Convenience: sample_demo → sample_start_state.
        Returns (joint_pos, joint_vel, cube_state, prev_action, demo_id, steps_back).
        """
        demo_id = self.sample_demo()
        j_pos, j_vel, c_state, p_act, steps_back = self.sample_start_state(demo_id)
        return (
            j_pos.to(device), j_vel.to(device),
            c_state.to(device), p_act.to(device),
            demo_id, steps_back,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            d: {
                "start_step": self.demo_state[d].start_step,
                "solved": self.demo_state[d].solved,
                "success_buffer": list(self.demo_state[d].success_buffer),
            }
            for d in self.demo_ids
        }

    def load_state_dict(self, sd: dict) -> None:
        for d, v in sd.items():
            if d in self.demo_state:
                self.demo_state[d].start_step = v["start_step"]
                self.demo_state[d].solved = v["solved"]
                self.demo_state[d].success_buffer = deque(v["success_buffer"], maxlen=self.m)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_demos(self, h5_path: str) -> None:
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                env_states = np.array(f[key]["env_states"])
                d = len(self.demo_state)
                self.joint_pos[d] = torch.tensor(env_states[:, 0:7], dtype=torch.float32)
                self.joint_vel[d] = torch.tensor(env_states[:, 7:14], dtype=torch.float32)
                self.cube_state[d] = torch.tensor(env_states[:, 14:27], dtype=torch.float32)
                self.prev_action[d] = torch.tensor(env_states[:, 33:39], dtype=torch.float32)
                ds = DemoState(total_steps=len(env_states))
                ds.success_buffer = deque([0] * self.m, maxlen=self.m)
                self.demo_state[d] = ds
