import os
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg
import isaaclab.envs.mdp as mdp
from mdp.observation import (
    left_contact_binary,
    right_contact_binary,
    cube1_pos_local_obs,
    cube2_pos_local_obs,
    cube1_to_cube2_vec_obs,
    cube_stack_contact_binary,
    left_finger_pos_obs,
    right_finger_pos_obs,
    finger_center_to_cube1_vec_obs,
)
from mdp.termination import terminate_on_excessive_joint_velocity, terminate_on_cube_dropped
from mdp.actions import MimicGripperActionCfg
from mdp.reward import (
    reward_reach_cube_exp,
    reward_per_finger_distance,
    reward_finger_closing,
    reward_close_approach,
    reward_single_finger_contact,
    reward_dual_finger_contact,
)
from cfg.dofbof_cfg import DOFBOT_CFG


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_PLANE_USD = os.path.join(BASE_DIR, "..", "usd", "target_plane.usd")
CUBE_USD1 = os.path.join(BASE_DIR, "..", "usd", "cube1.usd")
CUBE_USD2 = os.path.join(BASE_DIR, "..", "usd", "cube2.usd")



@configclass
class DofBotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75)
        )
    )

    robot = DOFBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cube1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUBE_USD1,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.20, 0.03)
        ),
    )

    cube2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUBE_USD2,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.20, 0.20, 0.03)
        ),
    )


    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link4/Camera",
        data_types=["rgb"],
        width=80,
        height=60,
        spawn=None

    )

    contact_sensor_left_finger = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link5/Finger_Left_03/Finger_Left_03",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube1/Cube"],
    )

    contact_sensor_right_finger = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link5/Finger_Right_03/Finger_Right_03",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube1/Cube"],
    )

    contact_sensor_left_ground = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link5/Finger_Left_03/Finger_Left_03",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        track_air_time=True,
        filter_prim_paths_expr=["/World/defaultGroundPlane"],
    )

    contact_sensor_right_ground = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link5/Finger_Right_03/Finger_Right_03",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        track_air_time=True,
        filter_prim_paths_expr=["/World/defaultGroundPlane"],
    )

    contact_sensor_cube1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cube1/Cube",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube2/Cube"],
    )

    contact_sensor_cube2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cube2/Cube",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube1/Cube"],
    )
        

   


@configclass
class CommandsCfg:

    pass


@configclass
class ActionsCfg:



    arm_pos = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["joint1", "joint2", "joint3", "joint4"],
    scale=0.25,  # 0.15 -> 0.3: 0.5는 joint 폭발, 중간값으로 조정
    use_default_offset=True,
    )

    wrist_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Wrist_Twist_RevoluteJoint"],
        scale=0.25,  # 0.05 -> 0.2: ±2.9° 너무 제한적
        use_default_offset=True,
    )

    # 1-DOF gripper action: 단일 명령을 두 손가락에 부호 반전해서 보냄 (실 하드웨어 single-motor 모사)
    # Left 는 양의 방향, Right 는 음의 방향으로 동일 크기만큼 회전 → 양 손가락이 함께 열고 닫힘
    finger_pos = MimicGripperActionCfg(
        asset_name="robot",
        joint_names=["Finger_Left_01_RevoluteJoint", "Finger_Right_01_RevoluteJoint"],
        signs=(1.0, -1.0),
        scale=0.3,
        use_default_offset=True,
    )




@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):

        # # 카메라 데이터 관측
        # camera_rgb = ObsTerm(
        #     func=mdp.image,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("camera"),
        #         "data_type": "rgb",
        #     },
        # )

        # joints 현재 토크 관측
        
        joint_effort = ObsTerm(
        func=mdp.joint_effort,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "Wrist_Twist_RevoluteJoint",
                    "Finger_Left_01_RevoluteJoint",
                    "Finger_Right_01_RevoluteJoint",
                ],
                preserve_order=True,
                )
            },
        )

        # joints 각도 관측
        joint_pos = ObsTerm(
        func=mdp.joint_pos,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "Wrist_Twist_RevoluteJoint",
                    "Finger_Left_01_RevoluteJoint",
                    "Finger_Right_01_RevoluteJoint",
                ],
                preserve_order=True,
                )
            },
        )

        joint_vel = ObsTerm(
            func = mdp.joint_vel,
            params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "Wrist_Twist_RevoluteJoint",
                    "Finger_Left_01_RevoluteJoint",
                    "Finger_Right_01_RevoluteJoint",
                ],
                preserve_order=True,
                )
            }
            
        )
        cube1_pos = ObsTerm(
            func=cube1_pos_local_obs,
        )

        cube2_pos = ObsTerm(
            func=cube2_pos_local_obs,
        )

        cube1_to_cube2 = ObsTerm(
            func=cube1_to_cube2_vec_obs,
        )

        left_finger_pos = ObsTerm(
            func=left_finger_pos_obs
        )

        right_finger_pos = ObsTerm(
            func=right_finger_pos_obs
        )


        left_finger_contact = ObsTerm(
            func = left_contact_binary
        )

        right_finger_contact = ObsTerm(
            func = right_contact_binary
        )

        cube1_to_finger_vec = ObsTerm(
            func=finger_center_to_cube1_vec_obs
        )

        cube_stack_contact = ObsTerm(
            func=cube_stack_contact_binary
        )


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:

    # reset_target_plane = EventTermCfg(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("target_plane"),
    #         "pose_range": {
    #             "x": (0.1, 0.1),
    #             "y": (-0.1, 0.2),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0.0, 0.0),
    #             "yaw": (0.0, 0.0),
    #         },
    #         "velocity_range": {},
    #     },
    # )

    reset_cube1 = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube1"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
        },
    )

    reset_cube2 = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube2"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
        },
    )

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    reach_cube = RewTerm(func=reward_reach_cube_exp, weight=1.0, params={"temperature": 0.08},)
    per_finger_dist = RewTerm(func=reward_per_finger_distance, weight=15.0, params={"temperature": 0.10},)
    finger_closing = RewTerm(func=reward_finger_closing, weight=10.0, params={"activation_dist": 0.15},)
    close_approach = RewTerm(func=reward_close_approach, weight=30.0)
    single_finger_contact = RewTerm(func=reward_single_finger_contact, weight=10.0)
    dual_finger_contact = RewTerm(func=reward_dual_finger_contact, weight=50.0)
    action_rate_penalty = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_vel_penalty = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    excessive_joint_velocity = DoneTerm(func=terminate_on_excessive_joint_velocity, params={"max_joint_vel": 25.0})


@configclass
class CurriculumCfg:
    pass



@configclass
class DofbotEnvCfg(ManagerBasedRLEnvCfg):
    scene = DofBotSceneCfg(num_envs=1, env_spacing=1.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self) -> None:
        """Post initialization."""

        self.decimation = 4  # 숫자가 커지면 느린 제어
        self.episode_length_s = 10  # x초 동안 에피소드 진행 param: 8 ->
        self.viewer.eye = (8.0, 0.0, 5.0)

        self.sim.dt = 1 / 120   # 시뮬레이션 업데이트 시간 
        self.sim.render_interval = self.decimation