"""
Convert teleop_my dataset → dofbot_hong RFCL format.

teleop_my obs layout (56-dim, from _build_env_state):
  [0:7]   joint_effort
  [7:14]  joint_pos
  [14:21] joint_vel
  [21:24] cube1_pos (local)
  [24:27] cube2_pos (local)
  [27:30] left_finger_pos
  [30:33] right_finger_pos
  [33:34] left_finger_contact
  [34:35] right_finger_contact
  [35:36] cube1_cube2_contact
  [36:39] left_force (current only, no history)
  [39:42] right_force (current only, no history)
  [42:49] prev_action (7-DOF)
  [49:56] desired_joint_pos (7-DOF)

dofbot_hong env_states layout (39-dim):
  [0:7]   joint_pos
  [7:14]  joint_vel
  [14:17] cube1_local_pos
  [17:21] cube1_quat         ← zeros (not in teleop obs)
  [21:24] cube1_lin_vel      ← zeros (not in teleop obs)
  [24:27] cube1_ang_vel      ← zeros (not in teleop obs)
  [27:30] left_contact_force (3-dim, from left_force)
  [30:33] right_contact_force(3-dim, from right_force)
  [33:39] prev_action (6-DOF: arm4+wrist1+gripper1)

Action conversion (7-DOF → 6-DOF):
  teleop: [j1,j2,j3,j4,wrist,finger_left,finger_right]
  dofbot: [j1,j2,j3,j4,wrist, gripper]
  gripper = finger_left  (mimic: left=+g, right=-g)
"""
from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np


def convert_obs(obs: np.ndarray) -> np.ndarray:
    """56-dim teleop obs → 31-dim dofbot_hong SAC obs."""
    joint_pos        = obs[7:14]    # [7]
    joint_vel        = obs[14:21]   # [7]
    cube1_pos        = obs[21:24]   # [3]
    cube2_pos        = obs[24:27]   # [3]
    left_finger_pos  = obs[27:30]   # [3]
    right_finger_pos = obs[30:33]   # [3]
    left_contact     = obs[33:34]   # [1]
    right_contact    = obs[34:35]   # [1]
    finger_center    = 0.5 * (left_finger_pos + right_finger_pos)
    finger_to_cube1  = cube1_pos - finger_center  # [3]
    return np.concatenate([
        joint_pos, joint_vel, cube1_pos, cube2_pos,
        left_finger_pos, right_finger_pos,
        left_contact, right_contact, finger_to_cube1,
    ]).astype(np.float32)  # [31]


def convert_action(action: np.ndarray) -> np.ndarray:
    """7-DOF teleop action → 6-DOF dofbot_hong action."""
    # [j1,j2,j3,j4,wrist, finger_left, finger_right] → [j1,j2,j3,j4,wrist, gripper]
    return np.concatenate([action[:5], action[5:6]]).astype(np.float32)


def convert_env_state(obs: np.ndarray, action_6dof: np.ndarray) -> np.ndarray:
    """
    Build 39-dim env_state from 56-dim obs.
    Fields not captured by teleop (quat, velocities) are set to zero.
    """
    joint_pos  = obs[7:14]      # [7]
    joint_vel  = obs[14:21]     # [7]
    cube1_pos  = obs[21:24]     # [3]
    cube1_quat = np.zeros(4, dtype=np.float32)   # unknown
    cube1_lv   = np.zeros(3, dtype=np.float32)   # unknown
    cube1_av   = np.zeros(3, dtype=np.float32)   # unknown
    left_force  = obs[36:39].astype(np.float32)  # [3]
    right_force = obs[39:42].astype(np.float32)  # [3]

    return np.concatenate([
        joint_pos, joint_vel,
        cube1_pos, cube1_quat, cube1_lv, cube1_av,
        left_force, right_force,
        action_6dof,
    ]).astype(np.float32)  # [39]


def convert(src_h5: str, dst_h5: str, dst_json: str) -> None:
    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        traj_keys = sorted(src.keys())
        n = len(traj_keys)
        print(f"[INFO] {n} trajectories found in {src_h5}")

        for i, key in enumerate(traj_keys):
            grp = src[key]
            success_raw = grp["success"][:].astype(np.float32)  # [T]
            T = len(success_raw)

            # prefer pre-computed dofbot_hong fields (from new test.py)
            if "dofbot_obs" in grp:
                obs_31      = grp["dofbot_obs"][:]         # [T+1, 31]
                actions_6   = grp["dofbot_actions"][:]     # [T, 6]
                env_states  = grp["dofbot_env_states"][:]  # [T+1, 69]
                rewards     = grp["dofbot_rewards"][:]     # [T]
            else:
                # fallback: convert from legacy 56-dim obs
                obs_raw      = grp["obs"][:]        # [T, 56]
                next_obs_raw = grp["next_obs"][:]   # [T, 56]
                actions_raw  = grp["actions"][:]    # [T, 7]

                obs_31 = np.stack(
                    [convert_obs(obs_raw[0])] +
                    [convert_obs(next_obs_raw[t]) for t in range(T)],
                    axis=0,
                ).astype(np.float32)  # [T+1, 31]

                actions_6 = np.stack(
                    [convert_action(actions_raw[t]) for t in range(T)], axis=0
                ).astype(np.float32)  # [T, 6]

                rewards = success_raw

                first_action_6 = np.zeros(6, dtype=np.float32)
                states = [convert_env_state(obs_raw[0], first_action_6)]
                for t in range(T):
                    states.append(convert_env_state(next_obs_raw[t], actions_6[t]))
                env_states = np.stack(states, axis=0).astype(np.float32)  # [T+1, 69]

            g = dst.create_group(f"traj_{i}")
            g.create_dataset("obs",        data=obs_31)
            g.create_dataset("actions",    data=actions_6)
            g.create_dataset("rewards",    data=rewards)
            g.create_dataset("env_states", data=env_states)

            print(f"  traj_{i}: T={T}  success_steps={int(success_raw.sum())}")

    meta = {"episodes": [
        {"episode_id": i, "success": True, "reset_kwargs": {}, "episode_seed": None}
        for i in range(n)
    ]}
    with open(dst_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved → {dst_h5}")
    print(f"[INFO]         {dst_json}")
    print(f"[INFO] env_states: 39-dim  (quat/vel zeros-padded)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str,
                        default="../teleop_my/teleop_dataset/trajectory.state.pd_joint_pos.h5")
    parser.add_argument("--out_dir", type=str, default="demos_1dof")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_h5  = os.path.join(script_dir, args.src)
    out_dir = os.path.join(script_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dst_h5   = os.path.join(out_dir, "demos.h5")
    dst_json = os.path.join(out_dir, "demos.json")

    convert(src_h5, dst_h5, dst_json)


if __name__ == "__main__":
    main()
