"""
Collect demonstration trajectories using a trained SAC policy.
Saves to demos_1dof/ in the format expected by ReverseCurriculumLearner.

env_states layout (39-dim):
  [0:7]   joint_pos
  [7:14]  joint_vel
  [14:27] cube1 state (pos3 + quat4 + linvel3 + angvel3)
  [27:30] left_contact_force  (net_forces_w current frame, 3-dim)
  [30:33] right_contact_force (net_forces_w current frame, 3-dim)
  [33:39] prev_action         (arm4 + wrist1 + gripper1 = 6-DOF)
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task",         type=str, default="Isaac-Dofbot-v0")
parser.add_argument("--checkpoint",   type=str, required=True,
                    help="SAC checkpoint (best_agent.pt 등)")
parser.add_argument("--num_demos",    type=int, default=10)
parser.add_argument("--max_episodes", type=int, default=500)
parser.add_argument("--out_dir",      type=str, default="demos_1dof")
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--grace_steps",  type=int, default=15)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────────
import copy
import json
import os

import h5py
import numpy as np
import torch
import gymnasium as gym

import dofbot_task
from dofbot_task.dofbot_env_cfg import DofbotEnvCfg
from dofbot_task.agent.sac import SAC, SAC_DEFAULT_CONFIG, RandomMemory
from models.policy import PolicyModel
from models.critic import CriticModel
from models.models_cfg import PolicyModelCfg, CriticModelCfg
from mdp.reward import reward_dual_finger_contact

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ARM_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "Wrist_Twist_RevoluteJoint",
    "Finger_Left_01_RevoluteJoint",
    "Finger_Right_01_RevoluteJoint",
]


# ── obs: dofbot_hong train.py와 동일한 10-dim ─────────────────────────────────
def extract_obs(obs_dict) -> torch.Tensor:
    p = obs_dict["policy"]
    finger_center = 0.5 * (p["left_finger_pos"] + p["right_finger_pos"])
    finger_to_cube1 = p["cube1_pos"] - finger_center
    return torch.cat([
        p["joint_pos"],            # [N, 7]
        p["joint_vel"],            # [N, 7]
        p["cube1_pos"],            # [N, 3]
        p["cube2_pos"],            # [N, 3]
        p["left_finger_pos"],      # [N, 3]
        p["right_finger_pos"],     # [N, 3]
        p["left_finger_contact"],  # [N, 1]
        p["right_finger_contact"], # [N, 1]
        finger_to_cube1,           # [N, 3]
    ], dim=-1)  # [N, 31]


def get_joint_ids(base_env) -> list[int]:
    ids = []
    for name in ARM_JOINT_NAMES:
        jid = base_env.scene["robot"].find_joints(name)[0]
        if isinstance(jid, (list, tuple)):
            jid = jid[0]
        if isinstance(jid, torch.Tensor):
            jid = jid.item()
        ids.append(int(jid))
    return ids


# ── env_states: 39-dim, ReverseCurriculumLearner 형식 ─────────────────────────
def capture_state(base_env, joint_ids: list[int]) -> np.ndarray:
    """
    Layout (39-dim):
      [0:7]   joint_pos
      [7:14]  joint_vel
      [14:17] cube1_local_pos
      [17:21] cube1_quat
      [21:24] cube1_lin_vel
      [24:27] cube1_ang_vel
      [27:30] left_contact_force  (net_forces_w current frame, 3-dim)
      [30:33] right_contact_force (net_forces_w current frame, 3-dim)
      [33:39] prev_action         (6-DOF: arm4 + wrist1 + gripper1)
    """
    robot  = base_env.scene["robot"]
    cube   = base_env.scene["cube1"]                          # dofbot_hong: cube1
    origin = base_env.scene.env_origins[0].cpu().numpy()

    joint_pos  = robot.data.joint_pos[0, joint_ids].cpu().numpy()   # [7]
    joint_vel  = robot.data.joint_vel[0, joint_ids].cpu().numpy()   # [7]
    cube_pos_w = cube.data.root_pos_w[0].cpu().numpy()              # [3]
    cube_quat  = cube.data.root_quat_w[0].cpu().numpy()             # [4]
    cube_lv    = cube.data.root_lin_vel_w[0].cpu().numpy()          # [3]
    cube_av    = cube.data.root_ang_vel_w[0].cpu().numpy()          # [3]

    # contact history: [history_length=6, 3] → flatten → [18]
    left_hist  = base_env.scene["contact_sensor_left_finger"].data.net_forces_w[0].cpu().numpy().flatten()
    right_hist = base_env.scene["contact_sensor_right_finger"].data.net_forces_w[0].cpu().numpy().flatten()

    # prev_action: 6-DOF (arm4 + wrist1 + gripper_1dof)
    action_dim = base_env.action_manager.total_action_dim
    try:
        prev_action = base_env.action_manager.action[0].cpu().numpy()  # [6]
    except Exception:
        prev_action = np.zeros(action_dim, dtype=np.float32)

    return np.concatenate([
        joint_pos, joint_vel,
        cube_pos_w - origin, cube_quat, cube_lv, cube_av,
        left_hist, right_hist,
        prev_action,
    ]).astype(np.float32)  # [39]


def main():
    device = args_cli.device

    # ── 환경 ──────────────────────────────────────────────────────────────────
    env_cfg = DofbotEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = device

    env      = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    obs_dict, _ = env.reset()
    obs = extract_obs(obs_dict).to(device)

    obs_dim    = obs.shape[-1]     # 10
    action_dim = base_env.action_manager.total_action_dim  # 6
    joint_ids  = get_joint_ids(base_env)

    # ── SAC policy 로드 ────────────────────────────────────────────────────────
    NUM_QS = 10
    policy = PolicyModel(obs_dim, action_dim, PolicyModelCfg(), device)
    critics = [CriticModel(obs_dim, action_dim, CriticModelCfg(), device) for _ in range(NUM_QS)]
    target_critics = [CriticModel(obs_dim, action_dim, CriticModelCfg(), device) for _ in range(NUM_QS)]
    memory = RandomMemory(memory_size=1, num_envs=1, device=device)

    models = {"policy": policy}
    models.update({f"critic_{i+1}": c for i, c in enumerate(critics)})
    models.update({f"target_critic_{i+1}": tc for i, tc in enumerate(target_critics)})

    sac_cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
    agent = SAC(models=models, memory=memory,
                observation_space=obs_dim, action_space=action_dim,
                device=device, cfg=sac_cfg)
    agent.init()

    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    print(f"[INFO] SAC policy loaded: {args_cli.checkpoint}")

    # ── 수집 ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(SCRIPT_DIR, args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    collected: list[dict] = []
    ep_count = 0
    grace_steps = args_cli.grace_steps

    def _reset_ep():
        return (
            [obs[0].cpu().numpy()],       # ep_obs
            [capture_state(base_env, joint_ids)],  # ep_states
            [], [],                        # ep_actions, ep_success
        )

    ep_obs, ep_states, ep_actions, ep_success = _reset_ep()
    first_success_step: int | None = None

    print(f"[INFO] Collecting {args_cli.num_demos} demos (max {args_cli.max_episodes} eps)")

    while (
        simulation_app.is_running()
        and len(collected) < args_cli.num_demos
        and ep_count < args_cli.max_episodes
    ):
        with torch.no_grad():
            actions, _, _ = agent.act(states=obs, timestep=0, timesteps=0)

        next_obs_dict, _, terminated, truncated, _ = env.step(actions)
        next_obs = extract_obs(next_obs_dict).to(device)

        dual = reward_dual_finger_contact(base_env)[0].item()
        step_success = float(dual > 0.0)

        ep_actions.append(actions[0].cpu().numpy())
        ep_success.append(step_success)
        ep_states.append(capture_state(base_env, joint_ids))
        ep_obs.append(next_obs[0].cpu().numpy())

        if step_success > 0.5 and first_success_step is None:
            first_success_step = len(ep_actions) - 1

        env_done  = (terminated | truncated)[0].item()
        grace_done = (
            first_success_step is not None
            and (len(ep_actions) - 1 - first_success_step) >= grace_steps
        )

        if env_done or grace_done:
            ep_count += 1
            succeeded = first_success_step is not None

            if succeeded:
                collected.append({
                    "obs":     np.array(ep_obs,     dtype=np.float32),  # [T+1, 10]
                    "actions": np.array(ep_actions, dtype=np.float32),  # [T,   6]
                    "rewards": np.array(ep_success, dtype=np.float32),  # [T]   sparse
                    "states":  np.array(ep_states,  dtype=np.float32),  # [T+1, 69]
                })
                print(f"[EP {ep_count:>4}] SUCCESS {len(collected):>3}/{args_cli.num_demos}"
                      f"  len={len(ep_actions)}")
            else:
                print(f"[EP {ep_count:>4}] fail   len={len(ep_actions)}")

            obs_dict, _ = env.reset()
            obs = extract_obs(obs_dict).to(device)
            ep_obs, ep_states, ep_actions, ep_success = _reset_ep()
            first_success_step = None
        else:
            obs = next_obs

    n = len(collected)
    print(f"\n[INFO] Collected {n}/{ep_count} succeeded.")
    if n == 0:
        print("[WARN] No demos. Not saving.")
        env.close()
        return

    # ── 저장: ReplayDataset + ReverseCurriculumLearner 형식 ───────────────────
    h5_path   = os.path.join(out_dir, "demos.h5")
    json_path = os.path.join(out_dir, "demos.json")

    with h5py.File(h5_path, "w") as f:
        for i, demo in enumerate(collected):
            g = f.create_group(f"traj_{i}")
            g.create_dataset("obs",        data=demo["obs"])      # [T+1, 10]
            g.create_dataset("actions",    data=demo["actions"])  # [T,   6]
            g.create_dataset("rewards",    data=demo["rewards"])  # [T]
            g.create_dataset("env_states", data=demo["states"])   # [T+1, 69]

    meta = {"episodes": [
        {"episode_id": i, "success": True, "reset_kwargs": {}, "episode_seed": None}
        for i in range(n)
    ]}
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved → {h5_path}  ({n} demos, env_states 39-dim)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
