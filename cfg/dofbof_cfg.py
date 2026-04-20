import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOFBOT_USD_PATH = os.path.join(BASE_DIR, "..", "usd", "dofbot.usd")

ARM_SERVO_TORQUE = 3.0
SERVO_VELOCITY_LIMIT = 4.0
FINGER_SERVO_TORQUE = 1.0

DOFBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DOFBOT_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "Wrist_Twist_RevoluteJoint": 0.0,
            "Finger_Left_01_RevoluteJoint": 0.0,
            "Finger_Right_01_RevoluteJoint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "joint1": DCMotorCfg(
            joint_names_expr=["joint1"],
            stiffness=10.0,
            damping=0.5,
            saturation_effort=ARM_SERVO_TORQUE,
            effort_limit_sim=ARM_SERVO_TORQUE,
            velocity_limit=SERVO_VELOCITY_LIMIT,
            velocity_limit_sim=SERVO_VELOCITY_LIMIT,
            armature=0.02,
            friction=0.0,
        ),
        "joint2": DCMotorCfg(
            joint_names_expr=["joint2"],
            stiffness=10.0,
            damping=0.5,
            saturation_effort=ARM_SERVO_TORQUE,
            effort_limit_sim=ARM_SERVO_TORQUE,
            velocity_limit=SERVO_VELOCITY_LIMIT,
            velocity_limit_sim=SERVO_VELOCITY_LIMIT,
            armature=0.02,
            friction=0.0,
        ),
        "joint3": DCMotorCfg(
            joint_names_expr=["joint3"],
            stiffness=10.0,
            damping=0.5,
            saturation_effort=ARM_SERVO_TORQUE,
            effort_limit_sim=ARM_SERVO_TORQUE,
            velocity_limit=SERVO_VELOCITY_LIMIT,
            velocity_limit_sim=SERVO_VELOCITY_LIMIT,
            armature=0.02,
            friction=0.0,
        ),
        "joint4": DCMotorCfg(
            joint_names_expr=["joint4"],
            stiffness=10.0,
            damping=0.5,
            saturation_effort=ARM_SERVO_TORQUE,
            effort_limit_sim=ARM_SERVO_TORQUE,
            velocity_limit=SERVO_VELOCITY_LIMIT,
            velocity_limit_sim=SERVO_VELOCITY_LIMIT,
            armature=0.02,
            friction=0.0,
        ),
        "Wrist_Twist_RevoluteJoint": DCMotorCfg(
            joint_names_expr=["Wrist_Twist_RevoluteJoint"],
            stiffness=10.0,
            damping=0.5,
            saturation_effort=ARM_SERVO_TORQUE,
            effort_limit=ARM_SERVO_TORQUE,
            effort_limit_sim=ARM_SERVO_TORQUE,
            velocity_limit=SERVO_VELOCITY_LIMIT,
            velocity_limit_sim=SERVO_VELOCITY_LIMIT,
            armature=0.02,
            friction=0.0,
        ),
        "Finger_Left_01_RevoluteJoint": DCMotorCfg(
            joint_names_expr=["Finger_Left_01_RevoluteJoint"],
            stiffness=8.0,
            damping=0.5,
            saturation_effort=FINGER_SERVO_TORQUE,
            effort_limit=FINGER_SERVO_TORQUE,
            effort_limit_sim=FINGER_SERVO_TORQUE,
            velocity_limit=4.0,
            velocity_limit_sim=4.0,
            armature=0.05,
            friction=0.0,
        ),
        "Finger_Right_01_RevoluteJoint": DCMotorCfg(
            joint_names_expr=["Finger_Right_01_RevoluteJoint"],
            stiffness=8.0,
            damping=0.5,
            saturation_effort=FINGER_SERVO_TORQUE,
            effort_limit=FINGER_SERVO_TORQUE,
            effort_limit_sim=FINGER_SERVO_TORQUE,
            velocity_limit=4.0,
            velocity_limit_sim=4.0,
            armature=0.05,
            friction=0.0,
        ),
    },
)
