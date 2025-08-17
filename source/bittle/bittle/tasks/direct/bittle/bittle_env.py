# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Week of the 18th of August 2025 Draft Dissertation

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg

from .bittle_env_cfg import BittleEnvCfg

class BittleEnv(DirectRLEnv):
    cfg: BittleEnvCfg

    def __init__(self, cfg: BittleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self.actions = torch.zeros((self.cfg.scene.num_envs, self.cfg.action_space), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

        # setting aside useful variables for later
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
        self.commands[:, 0] = 0.0
        self.commands[:, 1] = 1.0
        self.commands[:, -1] = 0.0
        self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)

        # offsets to account for atan range and keep things on [-pi, pi]
        ratio = self.commands[:, 1] / (self.commands[:, 0] + 1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:, 0] * gzero[:, 1]
        minus = lzero[:, 0] * lzero[:, 1]
        offsets = torch.pi * plus - torch.pi * minus
        self.yaws = torch.atan(ratio).reshape(-1, 1) + offsets.reshape(-1, 1)

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.1
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions[:] = self.actions  # store previous before overwriting
        self.actions = actions.clone()
        self._apply_action()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions, joint_ids=self.dof_idx)
        # self.robot.set_joint_effort_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        obs = []

        # Base linear velocity
        base_lin_vel = self.robot.data.root_lin_vel_b
        obs.append(base_lin_vel)

        # Base angular velocity
        base_ang_vel = self.robot.data.root_ang_vel_b
        obs.append(base_ang_vel)

        # Orientation angles
        projected_gravity = self.robot.data.projected_gravity_b
        roll_pitch = projected_gravity[:, :2]
        obs.append(roll_pitch)

        # Joint positions
        dof_pos = self.robot.data.joint_pos[:, self.dof_idx]
        obs.append(dof_pos)

        # Joint velocities
        dof_vel = self.robot.data.joint_vel[:, self.dof_idx]
        obs.append(dof_vel)

        # Previous actions
        obs.append(self.actions)

        # Reference velocities
        ref_vel = self.commands
        obs.append(ref_vel)

        # Reference robot altitude
        ref_z = torch.full((self.cfg.scene.num_envs, 1), 0.11, device=self.device)
        obs.append(ref_z)

        # Final observation tensor
        observations = {"policy": torch.cat(obs, dim=-1)}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # --- Reward weights ---
        weights = {
            "lin_vel": 5.0,
            "ang_vel": 0.5,
            "height": 2.0,
            "pose": 0.1,
            "action_rate": 0.01,
            "lin_vel_z": 2.0,
            "stability": 1.0,
            "alive": 0.5,
            "knee-height": 0.5,
            "foot-contact": 0.5,
            "vel_tracking": 5.0,
            "torque_penalty": -2.5e-5,
            "joint_accel": -2.5e-7, 
        }

        # Get needed data for computing the rewards
        lin_vel_b = self.robot.data.root_lin_vel_b
        ang_vel_b = self.robot.data.root_ang_vel_b
        base_pos_w = self.robot.data.root_pos_w
        joint_pos = self.robot.data.joint_pos[:, self.dof_idx]
        joint_vel = self.robot.data.joint_vel[:, self.dof_idx]
        commands = self.commands
        projected_gravity = self.robot.data.projected_gravity_b
        
        # --- Reward terms ---

        # 1. Linear velocity tracking reward
        lin_vel_xy = lin_vel_b[:, :2]
        scale = 0.5  # Scale factor for the linear velocity
        lin_vel_ref = commands[:, :2] * scale  # Scale the reference velocity
        r_lin_vel = -torch.sum((lin_vel_xy - lin_vel_ref) ** 2, dim=1)

        # 2. Angular velocity tracking reward
        # TODO: Implement angular velocity tracking reward
        # ang_vel_ref = commands[:, 2]  # Scale the reference angular velocity
        # r_ang_vel = -(ang_vel_b[:, 2] - ang_vel_ref) ** 2

        # 3. Height penalty
        z = base_pos_w[:, 2]
        z_ref = 0.10  # Reference height
        r_z = -(z - z_ref) ** 2

        # 4. Pose similarity reward
        q_default = torch.zeros_like(joint_pos)  # or set manually to natural stance
        r_pose_similarity = -torch.sum((joint_pos - q_default) ** 2, dim=1)

        # 5. Action rate penalty
        r_action_rate = -torch.sum((self.actions - self.previous_actions) ** 2, dim=1)

        # 6. Vertical velocity penalty
        r_lin_vel_z = -lin_vel_b[:, 2] ** 2

        # 7. Roll and pitch stabilization penalty
        roll_pitch = projected_gravity[:, :2]  # (sin(roll), sin(pitch))
        r_roll_pitch = -torch.sum(roll_pitch ** 2, dim=1)

        # 8. Survival reward
        r_alive = self.episode_length_buf.float() / self.max_episode_length

        # 9. Knee height reward
        knee_link_names = [
            "left_front_knee_link",
            "right_front_knee_link",
            "left_back_knee_link",
            "right_back_knee_link",
        ]
        knee_link_ids = self.robot.find_bodies(knee_link_names)[0]

        knee_heights = self.robot.data.body_link_pos_w[:, knee_link_ids, 2]
        avg_knee_height = torch.mean(knee_heights, dim=1)  # (num_envs,)

        r_knee = -torch.mean((avg_knee_height - 0.055) ** 2)  # Penalize deviation from 0.05m

        # 10. Foot contact reward
        foot_link_names = [
            "left_front_foot_link",
            "right_front_foot_link",
            "left_back_foot_link",
            "right_back_foot_link",
        ]
        foot_link_ids = self.robot.find_bodies(foot_link_names)[0]

        foot_heights = self.robot.data.body_link_pos_w[:, foot_link_ids, 2]  # (num_envs, 4)
        avg_foot_height = torch.mean(foot_heights, dim=1)  # (num_envs,)
        

        # r_foot = -torch.mean((avg_foot_height - 0.01) ** 2)  # Penalize deviation from 0.01m
        r_foot = -torch.mean(torch.abs(avg_foot_height - 0.03))

        # 11. Forward velocity reward
        v_x = lin_vel_b[:, 0]  # Forward velocity in the x direction
        v_y = lin_vel_b[:, 1]  # Forward velocity in the y direction

        v_x_ref = 0.0
        v_y_ref = 0.1
        r_vel = -((v_x - v_x_ref)**2 + (v_y - v_y_ref)**2)
        # print(f"v_x: {v_x}, v_y: {v_y}, r_vel: {r_vel}")

        # 12. Torque penalty
        # joint torques
        r_torques = torch.sum(torch.square(self.robot.data.applied_torque), dim=1)
        # joint acceleration
        r_accel = torch.sum(torch.square(self.robot.data.joint_acc), dim=1)

        # --- Combine with weights ---
        total_reward = (
            # weights["lin_vel"]      * r_lin_vel +
            # weights["ang_vel"]      * r_ang_vel +
            weights["height"]       * r_z +
            weights["pose"]         * r_pose_similarity +
            # weights["action_rate"]  * r_action_rate +
            weights["lin_vel_z"]    * r_lin_vel_z +
            weights["stability"]    * r_roll_pitch +
            # weights["alive"]        * r_alive + 
            # weights["knee-height"]  * r_knee +
            # weights["foot-contact"] * r_foot + 
            weights["vel_tracking"] * r_vel +
            # weights["torque_penalty"] * r_torques +
            weights["joint_accel"] * r_accel
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Termination thresholds ---
        roll_thresh = 0.5
        pitch_thresh = 0.4
        height_thresh = 0.05   # meters (base fell too low)

        # --- Get relevant state ---
        root_quat_w = self.robot.data.root_quat_w
        base_pos_z = self.robot.data.root_pos_w[:, 2]
        roll, pitch, _ = math_utils.euler_xyz_from_quat(root_quat_w)

        # --- Check termination conditions ---
        roll_violation = torch.abs(roll) > roll_thresh
        pitch_violation = torch.abs(pitch) > pitch_thresh
        height_violation = base_pos_z < height_thresh
        progress_done = self.episode_length_buf >= self.max_episode_length - 1

        # --- Combine conditions ---
        reset_due_to_fall = roll_violation | pitch_violation | height_violation
        # reset_due_to_fall = height_violation
        done_envs = progress_done | reset_due_to_fall

        return done_envs, reset_due_to_fall

    def _reset_idx(self, env_ids: Sequence[int] | None):
        
        # 1. Reset robot state
        # self.robot.reset(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # 2. Reset action buffers
        # self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.1 * torch.randn((len(env_ids), self.cfg.action_space), device=self.device)

        # 3. Resample directional command velocities (vx, vy)
        self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
        self.commands[env_ids,0] = 0.0
        self.commands[env_ids,1] = 0.1
        self.commands[env_ids,-1] = 0.0
        self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)

        # Recalculate the orientations for the command markers with the new commands
        ratio = self.commands[env_ids][:, 1] / (self.commands[env_ids][:, 0] + 1E-8)
        gzero = torch.where(self.commands[env_ids] > 0, True, False)
        lzero = torch.where(self.commands[env_ids] < 0, True, False)
        plus = lzero[:, 0] * gzero[:, 1]
        minus = lzero[:, 0] * lzero[:, 1]
        offsets = torch.pi * plus - torch.pi * minus
        self.yaws[env_ids] = torch.atan(ratio).reshape(-1, 1) + offsets.reshape(-1, 1)

        # Set the root state for the reset envs
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        default_q = self.robot.data.default_joint_pos[env_ids]
        default_q_dot = self.robot.data.default_joint_vel[env_ids]

        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self.robot.write_joint_position_to_sim(default_q, self.dof_idx, env_ids)
        self.robot.write_joint_velocity_to_sim(default_q_dot, self.dof_idx, env_ids)

        self._visualize_markers()

    def _visualize_markers(self):
        # Correct bittle orientation quaternion
        # correction_quat = math_utils.euler_angles_to_quat(torch.tensor([0.0, 0.0, -math.pi / 2], device=self.device))  # (3,) tensor
        correction_yaws = torch.ones((self.cfg.scene.num_envs, 1)).cuda() * (math.pi / 2)
        correction_quat = math_utils.quat_from_angle_axis(correction_yaws, self.up_dir).squeeze()

        # Get marker locations and orientations
        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = math_utils.quat_mul(correction_quat, self.robot.data.root_quat_w)
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        # Offset markers so they are above the bittle
        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        # Render the markers
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)


def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.05, 0.05, 0.1),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(44/255, 209/255, 63/255)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.05, 0.05, 0.1),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(209/255, 55/255, 44/255)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)