from bittle.robots.bittlebot import BITTLE_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class BittleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    # - spaces definition
    action_space = 8  # 8 DOF
    observation_space = 36
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = BITTLE_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=1.0, replicate_physics=True)
    dof_names = [
        "left_front_shoulder_joint",
        "right_front_shoulder_joint",
        "left_back_shoulder_joint",
        "right_back_shoulder_joint",    
        "left_front_knee_joint",
        "right_front_knee_joint",
        "left_back_knee_joint",
        "right_back_knee_joint",]