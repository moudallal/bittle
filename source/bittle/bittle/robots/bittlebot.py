import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Actuator names = ["left_back_shoulder_joint", "left_front_shoulder_joint", "right_back_shoulder_joint", "right_front_shoulder_joint", "left_back_knee_joint", "left_front_knee_joint", "right_back_knee_joint", "right_front_knee_joint"]

BITTLE_CONFIG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path = f"assets/bittle_00.usd",
        visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(43/255, 118/255, 240/255))
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        pos = (0.0, 0.0, 0.11),
    ),
    actuators = {
        "lf_shoulder_act": ImplicitActuatorCfg(
            joint_names_expr = ["left_front_shoulder_joint"],
            damping =  100000.0,
            stiffness = 10000000.0,
        ),
        "rf_shoulder_act": ImplicitActuatorCfg(
            joint_names_expr = ["right_front_shoulder_joint"],
            damping =  100000.0,
            stiffness = 10000000.0,
        ),
        "lb_shoulder_act": ImplicitActuatorCfg(
            joint_names_expr = ["left_back_shoulder_joint"],
            damping =  100000.0,
            stiffness = 10000000.0,
        ),
        "rb_shoulder_act": ImplicitActuatorCfg(
            joint_names_expr = ["right_back_shoulder_joint"],
            damping =  100000.0,
            stiffness = 10000000.0,
        ),
        "lf_knee_act": ImplicitActuatorCfg(
            joint_names_expr = ["left_front_knee_joint"],
            damping =  100000.0,
            stiffness = 10000000.0,
        ),
        "rf_knee_act": ImplicitActuatorCfg(
            joint_names_expr = ["right_front_knee_joint"],
            damping =  100000.0,
            stiffness = 10000000.0,
        ),
        "lb_knee_act": ImplicitActuatorCfg(
            joint_names_expr = ["left_back_knee_joint"],
            damping =  100000.0,
            stiffness = 10000000.0,
        ),
        "rb_knee_act": ImplicitActuatorCfg(
            joint_names_expr = ["right_back_knee_joint"],
            damping =  100000.0,
            stiffness = 10000000.0,
        ),
    },
)