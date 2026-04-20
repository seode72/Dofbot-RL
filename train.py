"""
RFCL Training Entry Point
Stage 1 (Reverse Curriculum) → Stage 2 (Forward Curriculum) via SAC + Q-ensemble.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Dofbot with RFCL.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Dofbot-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total_timesteps", type=int, default=3_000_000)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint for resuming.")
parser.add_argument("--start_stage", type=int, default=None, choices=[1, 2])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────────
import glob
import gymnasium as gym
import torch

import dofbot_task
from dofbot_task.dofbot_env_cfg import DofbotEnvCfg
from dofbot_task.agent.sac import SAC, SAC_DEFAULT_CONFIG, RandomMemory
from dofbot_task.agent.replay_dataset import ReplayDataset

from models.policy import PolicyModel
from models.critic import CriticModel
from models.models_cfg import PolicyModelCfg, CriticModelCfg

from checkpoint_tools.checkpoint import find_latest_checkpoint, load_checkpoint
from mdp.reward import reward_dual_finger_contact

from algo.reverse_curriculum import ReverseCurriculumLearner
from algo.forward_curriculum import ForwardCurriculumLearner
from algo.rfcl import RFCLTrainer

# ── constants ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMOS_H5   = os.path.join(SCRIPT_DIR, "demos_1dof", "demos.h5")
LOG_DIR    = os.path.join(SCRIPT_DIR, "logs", "rfcl")
os.makedirs(LOG_DIR, exist_ok=True)

SAC_CFG = {
    "batch_size": 256,
    "gradient_steps": 80,
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "entropy_learning_rate": 3e-4,
    "random_timesteps": 2000,
    "learning_starts": 5000,
    "discount_factor": 0.9,
}

NUM_QS = 10  # Q-ensemble size (paper Appendix C)

JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "Wrist_Twist_RevoluteJoint",
    "Finger_Left_01_RevoluteJoint", "Finger_Right_01_RevoluteJoint",
]


def extract_obs(obs_dict) -> torch.Tensor:
    p = obs_dict["policy"]
    return torch.cat([
        p["joint_pos"],          # [N, 7]
        p["cube1_pos"],          # [N, 3]
        p["cube2_pos"],          # [N, 3]

    ], dim=-1)  # [N, 31]


def build_sac(obs_dim: int, action_dim: int, num_envs: int, device: str) -> SAC:
    """Instantiate SAC with 10-critic ensemble and demo offline buffer."""
    cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
    cfg.update(SAC_CFG)

    policy = PolicyModel(obs_dim, action_dim, PolicyModelCfg(), device)
    critics = [CriticModel(obs_dim, action_dim, CriticModelCfg(), device) for _ in range(NUM_QS)]
    target_critics = [CriticModel(obs_dim, action_dim, CriticModelCfg(), device) for _ in range(NUM_QS)]
    for c, tc in zip(critics, target_critics):
        tc.load_state_dict(c.state_dict())

    memory = RandomMemory(memory_size=1_000_000, num_envs=num_envs, device=device)

    models = {"policy": policy}
    models.update({f"critic_{i+1}": c for i, c in enumerate(critics)})
    models.update({f"target_critic_{i+1}": tc for i, tc in enumerate(target_critics)})

    agent = SAC(models=models, memory=memory,
                observation_space=obs_dim, action_space=action_dim,
                device=device, cfg=cfg)
    agent.offline_buffer = ReplayDataset(DEMOS_H5, device=device)
    agent.init()
    return agent


def get_joint_ids(base_env, joint_names: list[str]) -> list[int]:
    robot = base_env.scene["robot"]
    ids = []
    for name in joint_names:
        jid = robot.find_joints(name)[0]
        if isinstance(jid, (list, tuple)):
            jid = jid[0]
        ids.append(int(jid.item() if isinstance(jid, torch.Tensor) else jid))
    return ids


def detect_stage(log_dir: str, checkpoint: str | None, forced_stage: int | None) -> tuple[int, str | None]:
    """Auto-detect current stage from saved files."""
    if forced_stage is not None:
        return forced_stage, checkpoint

    stage2_files = sorted(glob.glob(os.path.join(log_dir, "offline_buffer_stage2_*.pt")))
    if stage2_files:
        return 2, checkpoint

    return 1, checkpoint


def main():
    # ── environment ───────────────────────────────────────────────────────────
    env_cfg = DofbotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    obs_dict, _ = env.reset()
    obs = extract_obs(obs_dict).to(args_cli.device)

    obs_dim    = obs.shape[-1]
    action_dim = base_env.action_manager.total_action_dim
    num_envs   = base_env.num_envs
    print(f"[INFO] obs_dim={obs_dim}  action_dim={action_dim}  num_envs={num_envs}")

    # ── SAC agent ─────────────────────────────────────────────────────────────
    agent = build_sac(obs_dim, action_dim, num_envs, args_cli.device)

    # ── curriculum learners ───────────────────────────────────────────────────
    reverse_learner = ReverseCurriculumLearner(
        h5_path=DEMOS_H5,
        reverse_step_size=8,
        success_buffer_size=24,
        threshold=0.9,
        phi=1.0,
    )
    forward_learner = ForwardCurriculumLearner(
        num_seeds=1000,
        omega=0.75,
        beta=0.1,
        staleness_coef=0.1,
        num_envs=num_envs,
    )

    # ── joint ids ─────────────────────────────────────────────────────────────
    joint_ids = get_joint_ids(base_env, JOINT_NAMES)

    # ── checkpoint / resume ───────────────────────────────────────────────────
    ckpt_path = args_cli.checkpoint
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(LOG_DIR)

    total_step = 0
    completed_eps = 0
    resumed_stage = None
    if ckpt_path and os.path.exists(ckpt_path):
        result = load_checkpoint(agent, ckpt_path, args_cli.device)
        if isinstance(result, tuple):
            total_step, completed_eps, *rest = result
            resumed_stage = rest[0] if rest else None
        else:
            total_step = result
        print(f"[INFO] Resumed from {ckpt_path}  step={total_step}")

    # ── stage detection ───────────────────────────────────────────────────────
    start_stage, _ = detect_stage(LOG_DIR, ckpt_path, args_cli.start_stage)
    if resumed_stage is not None:
        start_stage = resumed_stage

    if start_stage == 2:
        stage2_files = sorted(glob.glob(os.path.join(LOG_DIR, "offline_buffer_stage2_*.pt")))
        if stage2_files:
            state = torch.load(stage2_files[-1], map_location=args_cli.device)
            agent.offline_buffer.load_state_dict(state)
            print(f"[INFO] Stage 2 offline buffer loaded: {stage2_files[-1]}")
        print("[INFO] Starting at Stage 2.")

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = RFCLTrainer(
        env=env,
        agent=agent,
        reverse_learner=reverse_learner,
        forward_learner=forward_learner,
        joint_ids=joint_ids,
        obs_fn=extract_obs,
        reward_fn=reward_dual_finger_contact,
        log_dir=LOG_DIR,
        device=args_cli.device,
        total_timesteps=args_cli.total_timesteps,
    )
    trainer.current_stage = start_stage
    trainer.total_step = total_step
    trainer.completed_eps = completed_eps

    # ── initial env state injection ───────────────────────────────────────────
    if start_stage == 1:
        trainer.init_stage1_envs()
    else:
        trainer.ep_max_steps[:] = 200
    obs = extract_obs(base_env.observation_manager.compute()).to(args_cli.device)

    # ── train ─────────────────────────────────────────────────────────────────
    trainer.train(obs)


if __name__ == "__main__":
    main()
    simulation_app.close()
