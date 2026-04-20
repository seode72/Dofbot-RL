import torch


def terminate_on_cube_dropped(env, grace_steps: int = 20) -> torch.Tensor:
    """한 번 cube contact 달성 후 contact가 사라지면 종료 (pick 태스크용).
    grace_steps: 에피소드 시작 후 이 step 수 동안은 termination 비활성 (RFCL 상태 주입 안정화).
    """
    left = env.scene["contact_sensor_left_finger"]
    right = env.scene["contact_sensor_right_finger"]

    left_contact = (torch.norm(left.data.net_forces_w, dim=-1) > 1e-4).any(dim=1)
    right_contact = (torch.norm(right.data.net_forces_w, dim=-1) > 1e-4).any(dim=1)
    contact = left_contact | right_contact

    if not hasattr(env, "_cube_grasped_flag"):
        env._cube_grasped_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 에피소드 첫 step에서 flag 초기화
    new_episode = env.episode_length_buf <= 1
    env._cube_grasped_flag[new_episode] = False

    env._cube_grasped_flag |= contact
    dropped = env._cube_grasped_flag & (~contact)

    # grace period: 초기 물리 안정화 동안 termination 억제
    in_grace = env.episode_length_buf < grace_steps

    return dropped & (~in_grace)

def terminate_on_excessive_joint_velocity(
    env,
    max_joint_vel: float = 25.0,
) -> torch.Tensor:
    """어떤 joint든 속도가 너무 커지면 종료"""
    robot = env.scene["robot"]
    joint_vel_max = torch.abs(robot.data.joint_vel).max(dim=1).values
    return joint_vel_max > max_joint_vel


# def terminate_on_cube_out_of_bounds(
#     env,
#     max_dist_xy: float = 0.5,
#     min_z: float = -0.05,
#     max_z: float = 0.3,
# ) -> torch.Tensor:
#     """CHANGED: cube가 너무 멀리 가거나 너무 위/아래로 벗어나면 종료."""
#     cube = env.scene["cube"]
#     cube_pos = cube.data.root_pos_w[:, :3]

#     xy_norm = torch.norm(cube_pos[:, :2], dim=1)
#     too_far = xy_norm > max_dist_xy
#     too_low = cube_pos[:, 2] < min_z
#     too_high = cube_pos[:, 2] > max_z

#     return too_far | too_low | too_high