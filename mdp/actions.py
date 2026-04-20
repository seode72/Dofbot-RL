from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class MimicGripperAction(JointAction):
    """1-DOF gripper action term that mirrors a single command to two finger joints.

    실제 도프봇 하드웨어가 1개 모터로 두 finger 를 동시에 구동하므로,
    정책의 1-DOF gripper 명령을 [Left=+a, Right=-a] (signs 가변) 로 분해해
    두 조인트 position target 으로 보낸다.
    """

    cfg: "MimicGripperActionCfg"

    def __init__(self, cfg: "MimicGripperActionCfg", env) -> None:
        # 부모 JointAction 의 __init__ 은 raw/processed action 을 num_joints 차원으로 만든다.
        # 여기서는 raw action 을 1-DOF 로 바꿔치움.
        super().__init__(cfg, env)

        if self._num_joints != len(cfg.signs):
            raise ValueError(
                f"MimicGripperAction: joint_names 개수({self._num_joints}) 와 signs 길이({len(cfg.signs)}) 가 다릅니다."
            )

        # 1-DOF raw action 으로 재할당
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)

        # signs: [num_joints]
        self._signs = torch.tensor(cfg.signs, dtype=torch.float32, device=self.device)

        # offset: use_default_offset 면 articulation 기본 joint pos 사용
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    @property
    def action_dim(self) -> int:
        return 1

    def process_actions(self, actions: torch.Tensor) -> None:
        # raw: [N, 1]
        self._raw_actions[:] = actions

        # scale 은 float 또는 [N, num_joints] tensor 일 수 있음
        if isinstance(self._scale, torch.Tensor):
            scale_per_joint = self._scale  # [N, num_joints]
        else:
            scale_per_joint = float(self._scale)

        # 1-DOF action 을 num_joints 로 broadcast + sign 적용
        per_joint = self._raw_actions * self._signs.unsqueeze(0)  # [N, num_joints]
        per_joint = per_joint * scale_per_joint
        self._processed_actions = per_joint + self._offset

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class MimicGripperActionCfg(JointActionCfg):
    """Config for :class:`MimicGripperAction`.

    joint_names 순서대로 signs 가 적용된다. 예: joint_names=[Left, Right], signs=(1.0, -1.0)
    이면 정책이 +action 을 출력했을 때 Left 는 +action, Right 는 -action 이 적용된다.
    """

    class_type: type[ActionTerm] = MimicGripperAction

    signs: tuple[float, ...] = MISSING
    """각 joint 에 적용할 부호. joint_names 와 길이가 같아야 함."""

    use_default_offset: bool = True
    """True 면 articulation 의 default_joint_pos 를 offset 으로 사용."""
