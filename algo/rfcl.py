"""
RFCLTrainer — Reverse Forward Curriculum Learning (ICLR 2024)

Orchestrates two-stage training:
  Stage 1: ReverseCurriculumLearner  (§4.1) — demo-guided reverse curriculum
  Stage 2: ForwardCurriculumLearner  (§4.2) — PLR-style forward curriculum

Both stages use SAC with a Q-ensemble and hybrid online/offline sampling.
"""

from __future__ import annotations
import os
import numpy as np
import torch
from collections import deque

from dofbot_task.agent.sac import RandomMemory
from dofbot_task.agent.replay_dataset import ReplayDataset
from algo.reverse_curriculum import ReverseCurriculumLearner
from algo.forward_curriculum import ForwardCurriculumLearner


class RFCLTrainer:
    """
    Main RFCL training loop.

    Public interface:
      train()                  — run full training (Stage 1 → Stage 2)
      _collect_step()          — one env step, record transition
      _reset_env_stage1()      — reverse curriculum state injection
      _reset_env_stage2()      — forward curriculum state injection
      _check_stage1_complete() — solved_frac > threshold for STABLE_THRESHOLD steps
      _transition_to_stage2()  — online buffer → offline buffer conversion
      save_checkpoint()        — save agent + metadata
    """

    GRACE_STEPS = 15          # consecutive dual-contact steps = success
    STABLE_THRESHOLD = 10000  # env steps that solved_frac must stay > 0.95

    def __init__(
        self,
        env,
        agent,
        reverse_learner: ReverseCurriculumLearner,
        forward_learner: ForwardCurriculumLearner,
        joint_ids: list[int],
        obs_fn,           # extract_policy_obs(obs_dict) → Tensor
        reward_fn,        # dual_finger_contact(env) → Tensor
        log_dir: str,
        device: str,
        total_timesteps: int = 3_000_000,
    ):
        self.env = env
        self.base_env = env.unwrapped
        self.agent = agent
        self.reverse_learner = reverse_learner
        self.forward_learner = forward_learner
        self.joint_ids = joint_ids
        self.obs_fn = obs_fn
        self.reward_fn = reward_fn
        self.log_dir = log_dir
        self.device = device
        self.total_timesteps = total_timesteps

        self.num_envs = self.base_env.num_envs
        obs_dim = agent.observation_space
        action_dim = agent.action_space

        # Per-env trackers
        self.ep_demo_ids = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.ep_steps_back = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.ep_max_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.ep_reward_sum = torch.zeros(self.num_envs, device=device)
        self.ep_step_count = torch.zeros(self.num_envs, device=device)
        self.ep_success_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=device)

        self.current_stage = 1
        self.total_step = 0
        self.completed_eps = 0
        self.recent_rewards = deque(maxlen=100)
        self.recent_successes = deque(maxlen=100)

        # Stage 1 → 2 transition tracking
        self._solved_frac_stable_steps = 0
        self._last_stable_check = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(self, start_obs: torch.Tensor) -> None:
        """Full training loop: Stage 1 then Stage 2."""
        obs = start_obs
        while self.total_step < self.total_timesteps:
            skip_update = False
            obs, skip_update = self._collect_step(obs)

            if not skip_update:
                self.agent.post_interaction(
                    timestep=self.total_step,
                    timesteps=self.total_timesteps,
                )

            if self.total_step > 0 and self.total_step % 5000 == 0:
                self.save_checkpoint()

            self.total_step += 1

        print("[RFCLTrainer] Training complete.")
        self.env.close()

    # ------------------------------------------------------------------
    # §4.1 / §4.2: Per-step collection
    # ------------------------------------------------------------------

    def _collect_step(self, obs: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """One environment step: act → step → record → handle episode end."""
        skip_update = False

        with torch.no_grad():
            self.agent.pre_interaction(
                timestep=self.total_step,
                timesteps=self.total_timesteps,
            )
            actions, _, _ = self.agent.act(obs, self.total_step, self.total_timesteps)

        next_obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        next_obs = self.obs_fn(next_obs_dict).to(self.device).detach()

        # Success detection: dual-finger contact for GRACE_STEPS consecutive steps
        dual_contact = self.reward_fn(self.base_env).view(self.num_envs).detach()
        is_success = dual_contact > 0.0
        self.ep_success_steps = torch.where(
            is_success, self.ep_success_steps + 1, torch.zeros_like(self.ep_success_steps)
        )
        early_term = self.ep_success_steps >= self.GRACE_STEPS

        terminated_1d = terminated.view(self.num_envs).detach() | early_term
        self.ep_step_count += 1
        truncated_1d = truncated.view(self.num_envs).detach() | (self.ep_step_count >= self.ep_max_steps)
        dones = terminated_1d | truncated_1d

        # Sparse + 1% dense reward mix
        custom_rewards = is_success.float() + 0.01 * rewards.view(self.num_envs).detach()
        self.ep_reward_sum += custom_rewards

        self.agent.record_transition(
            states=obs, actions=actions,
            rewards=custom_rewards.view(self.num_envs, 1),
            next_states=next_obs,
            terminated=terminated_1d.view(self.num_envs, 1),
            truncated=truncated_1d.view(self.num_envs, 1),
            infos=extras,
            timestep=self.total_step,
            timesteps=self.total_timesteps,
        )

        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_idx.numel() > 0:
            is_success_batch = self.ep_success_steps[done_idx] >= self.GRACE_STEPS
            self._record_episode_stats(done_idx, is_success_batch)
            self._reset_done_envs(done_idx, is_success_batch)

            if self.current_stage == 1:
                skip_update = self._check_stage1_complete()

            # Inject fresh observations for reset envs
            self.base_env.scene.write_data_to_sim()
            fresh_obs = self.obs_fn(self.base_env.observation_manager.compute()).to(self.device)
            next_obs[done_idx] = fresh_obs[done_idx]

            self._reset_ep_buffers(done_idx)

        return next_obs, skip_update

    # ------------------------------------------------------------------
    # §4.1: Stage 1 reset
    # ------------------------------------------------------------------

    def _reset_env_stage1(self, env_id: int) -> None:
        """
        Reverse curriculum reset for one env:
        record outcome → advance frontier → inject new state from demo.
        """
        success = (self.ep_success_steps[env_id] >= self.GRACE_STEPS).item()
        d_id = int(self.ep_demo_ids[env_id].item())
        s_back = int(self.ep_steps_back[env_id].item())

        self.reverse_learner.record_episode(d_id, s_back, success)

        j_pos, j_vel, c_state, p_act, next_d_id, next_s_back = self.reverse_learner.generate_reset(self.device)
        self.ep_demo_ids[env_id] = next_d_id
        self.ep_steps_back[env_id] = next_s_back
        self.ep_max_steps[env_id] = self.reverse_learner.get_episode_timelimit(next_d_id, next_s_back) * 6 + 200

        env_ids_t = torch.tensor([env_id], device=self.device)
        self._set_robot_joints(env_ids_t, j_pos.unsqueeze(0), j_vel.unsqueeze(0))
        self._set_cube_state(env_ids_t, c_state.unsqueeze(0))
        self.base_env.action_manager.action[env_id] = p_act

    # ------------------------------------------------------------------
    # §4.2: Stage 2 reset
    # ------------------------------------------------------------------

    def _reset_env_stage2(self, env_id: int) -> None:
        """
        Forward curriculum reset for one env:
        record outcome → sample next seed by P_S + P_C → inject seed-based state.
        """
        success = (self.ep_success_steps[env_id] >= self.GRACE_STEPS).item()
        seed_val = int(self.ep_demo_ids[env_id].item())
        self.forward_learner.record_episode(seed_val, self.ep_reward_sum[env_id].item(), success)

        next_seeds, _ = self.forward_learner.sample_initial_state(1)
        next_seed = int(next_seeds[0])
        self.ep_demo_ids[env_id] = next_seed
        self.ep_max_steps[env_id] = 200

        rng = np.random.RandomState(next_seed)
        j_pos = torch.tensor(
            np.array([0.0, 0.0, -1.1, -1.5, 0.0, 0.0, 0.0])
            + rng.uniform(-0.05, 0.05, 7) * [1, 1, 1, 1, 1, 0.3, 0.3],
            dtype=torch.float32, device=self.device,
        )
        j_vel = torch.zeros(7, dtype=torch.float32, device=self.device)
        cube_state = torch.tensor(
            [rng.uniform(-0.03, 0.03), rng.uniform(0.17, 0.23), 0.03,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=torch.float32, device=self.device,
        )

        env_ids_t = torch.tensor([env_id], device=self.device)
        self._set_robot_joints(env_ids_t, j_pos.unsqueeze(0), j_vel.unsqueeze(0))
        self._set_cube_state(env_ids_t, cube_state.unsqueeze(0))

    # ------------------------------------------------------------------
    # Stage transition
    # ------------------------------------------------------------------

    def _check_stage1_complete(self) -> bool:
        """
        Advance to Stage 2 when solved_frac > 0.95 is maintained for
        STABLE_THRESHOLD env steps. Returns True if update should be skipped.
        """
        solved_frac = self.reverse_learner.solved_fraction()
        steps_elapsed = (self.total_step - self._last_stable_check) * self.num_envs

        if solved_frac > 0.95:
            self._solved_frac_stable_steps += steps_elapsed
        else:
            self._solved_frac_stable_steps = 0
        self._last_stable_check = self.total_step

        if solved_frac > 0.95 and self._solved_frac_stable_steps > self.STABLE_THRESHOLD:
            self._transition_to_stage2()
            return True  # skip update this step (empty online buffer)
        return False

    def _transition_to_stage2(self) -> None:
        """
        Stage 1 → Stage 2 transition (paper §4.2):
        1. Fill offline buffer with Stage 1 online experience
        2. Reset online buffer
        3. Save checkpoint
        """
        print(f"[RFCLTrainer] Stage 1 → Stage 2 at step {self.total_step}")
        self.current_stage = 2

        # 1. Convert online buffer to offline
        self.agent.offline_buffer = ReplayDataset.from_online_buffer(
            self.agent.memory, device=self.device
        )
        buf_path = os.path.join(self.log_dir, f"offline_buffer_stage2_{self.total_step}.pt")
        torch.save(self.agent.offline_buffer.state_dict(), buf_path)
        print(f"[RFCLTrainer] Offline buffer saved → {buf_path}")

        # 2. Reset online buffer
        obs_dim = self.agent.observation_space
        action_dim = self.agent.action_space
        self.agent.memory = RandomMemory(
            memory_size=1_000_000, num_envs=self.num_envs, device=self.device
        )
        for name, size, dtype in [
            ("states", obs_dim, torch.float32), ("next_states", obs_dim, torch.float32),
            ("actions", action_dim, torch.float32), ("rewards", 1, torch.float32),
            ("terminated", 1, torch.bool), ("truncated", 1, torch.bool),
        ]:
            self.agent.memory.create_tensor(name=name, size=size, dtype=dtype)

        # 3. Checkpoint
        self.save_checkpoint(tag=f"stage1_final_{self.total_step}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def init_stage1_envs(self) -> None:
        """Initialize all envs from demo end states before training starts."""
        for env_id in range(self.num_envs):
            j_pos, j_vel, c_state, p_act, d_id, s_back = self.reverse_learner.generate_reset(self.device)
            self.ep_demo_ids[env_id] = d_id
            self.ep_steps_back[env_id] = s_back
            self.ep_max_steps[env_id] = self.reverse_learner.get_episode_timelimit(d_id, s_back) * 6 + 200
            env_ids_t = torch.tensor([env_id], device=self.device)
            self._set_robot_joints(env_ids_t, j_pos.unsqueeze(0), j_vel.unsqueeze(0))
            self._set_cube_state(env_ids_t, c_state.unsqueeze(0))
            self.base_env.action_manager.action[env_id] = p_act
        self.base_env.scene.write_data_to_sim()

    def save_checkpoint(self, tag: str = "") -> None:
        from checkpoint_tools.checkpoint import save_checkpoint
        name = tag or f"checkpoint_{self.total_step}"
        path = os.path.join(self.log_dir, f"{name}.pt")
        save_checkpoint(self.agent, path, self.total_step,
                        self.completed_eps, current_stage=self.current_stage)

    def _reset_done_envs(self, done_idx: torch.Tensor, is_success_batch: torch.Tensor) -> None:
        for i, env_id in enumerate(done_idx.tolist()):
            if self.current_stage == 1:
                self._reset_env_stage1(env_id)
            else:
                self._reset_env_stage2(env_id)

    def _record_episode_stats(self, done_idx: torch.Tensor, is_success_batch: torch.Tensor) -> None:
        self.completed_eps += done_idx.numel()
        self.recent_rewards.append(self.ep_reward_sum[done_idx].mean().item())
        self.recent_successes.append(is_success_batch.float().mean().item())
        if self.completed_eps % 100 == 0:
            succ_pct = sum(self.recent_successes) / len(self.recent_successes) * 100
            rew_avg = sum(self.recent_rewards) / len(self.recent_rewards)
            if self.current_stage == 1:
                log = self.reverse_learner.log_state()
                solved_frac = log.get("solved_frac", 0.0)
                details = log.get("details", {})
                prog_strs = [f"D{k.split('_')[1]}:{int((1.0-v)*100)}%" for k, v in details.items() if k.startswith("demo_")]
                prog_str = " ".join(prog_strs)
                print(
                    f"[EP {self.completed_eps:>6}] "
                    f"step={self.total_step:>8} | "
                    f"rew={rew_avg:>8.4f} | "
                    f"mov_avg={rew_avg:>8.4f} | "
                    f"succ={succ_pct:>5.1f}% | "
                    f"sol={solved_frac:>4.2f} | "
                    f"[{prog_str}]"
                )
            else:
                print(
                    f"[STAGE2 EP {self.completed_eps:>6}] "
                    f"step={self.total_step:>8} | "
                    f"rew={rew_avg:>8.4f} | "
                    f"mov_avg={rew_avg:>8.4f} | "
                    f"succ={succ_pct:>5.1f}%"
                )

    def _reset_ep_buffers(self, done_idx: torch.Tensor) -> None:
        self.ep_reward_sum[done_idx] = 0.0
        self.ep_step_count[done_idx] = 0.0
        self.ep_success_steps[done_idx] = 0

    def _set_robot_joints(self, env_ids, j_pos, j_vel) -> None:
        self.base_env.scene["robot"].write_joint_state_to_sim(
            j_pos.to(self.device), j_vel.to(self.device),
            joint_ids=self.joint_ids, env_ids=env_ids,
        )

    def _set_cube_state(self, env_ids, cube_state) -> None:
        cube = self.base_env.scene["cube1"]
        local_pos = cube_state[:, 0:3].to(self.device)
        world_pos = local_pos + self.base_env.scene.env_origins[env_ids].to(self.device)
        quat = torch.nn.functional.normalize(cube_state[:, 3:7].to(self.device), dim=-1)
        cube.write_root_pose_to_sim(torch.cat([world_pos, quat], dim=-1), env_ids=env_ids)
        cube.write_root_velocity_to_sim(
            torch.cat([cube_state[:, 7:10], cube_state[:, 10:13]], dim=-1).to(self.device),
            env_ids=env_ids,
        )
