from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_conjugate, quat_inv, quat_mul, axis_angle_from_quat, saturate
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

from .env_cfg import SharpaWaveEnvCfg
from .tacmap_sensor.sharpa_tacmap_vbts import SharpaTacmap
from .torch_jit_utils import chain_transform, transform_between_frames, unscale


class SharpaWaveInhandRotateTactileAlignEnv(DirectRLEnv):
    cfg: SharpaWaveEnvCfg

    def __init__(self, cfg: SharpaWaveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_pos_prev = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot_prev = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_default_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.obs_buf_lag_history = torch.zeros(
            (self.num_envs, 80, self.cfg.observation_space // 3), device=self.device, dtype=torch.float
        )
        self.at_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.proprio_hist_buf = torch.zeros(
            (self.num_envs, self.cfg.prop_hist_len, self.cfg.observation_space // 3), device=self.device, dtype=torch.float
        )
        self.priv_info_buf = torch.zeros(
            (self.num_envs, self.cfg.priv_info_dim), device=self.device, dtype=torch.float
        )

        self.actuated_dof_indices = sorted(
            [self.hand.joint_names.index(name) for name in cfg.actuated_joint_names]
        )

        self.finger_bodies = [self.hand.body_names.index(name) for name in self.cfg.fingertip_body_names]
        self.num_fingertips = len(self.finger_bodies)

        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0] * self.cfg.dof_limits_scale
        self.hand_dof_upper_limits = joint_pos_limits[..., 1] * self.cfg.dof_limits_scale

        self.p_gain_default = self.hand.data.default_joint_stiffness[:, self.actuated_dof_indices].clone()
        self.d_gain_default = self.hand.data.default_joint_damping[:, self.actuated_dof_indices].clone()
        self.p_gain = self.p_gain_default.clone()
        self.d_gain = self.d_gain_default.clone()

        if self.cfg.torque_control:
            self.hand.data.default_joint_stiffness = torch.zeros_like(self.p_gain_default, device=self.device)
            self.hand.data.default_joint_damping = torch.zeros_like(self.d_gain_default, device=self.device)
            for _, act in self.hand.actuators.items():
                act.stiffness = torch.zeros_like(act.stiffness, device=self.device)
                act.damping = torch.zeros_like(act.damping, device=self.device)

        self._contact_body_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        self.contact_forces = torch.zeros(
            (self.num_envs, len(self._contact_body_ids), 3), dtype=torch.float, device=self.device
        )
        self.contact_pos = torch.zeros(
            (self.num_envs, len(self._contact_body_ids), 3), dtype=torch.float, device=self.device
        )
        self._contact_body_ids_disable = torch.tensor(
            getattr(self.cfg, "disable_tactile_ids", []), dtype=torch.long
        )
        self.last_contacts = torch.zeros(
            (self.num_envs, len(self._contact_body_ids)), dtype=torch.float, device=self.device
        )
        self.elastomer_ids = [
            self.hand.body_names.index(name) for name in
            ["right_thumb_elastomer", "right_index_elastomer", "right_middle_elastomer",
             "right_ring_elastomer", "right_pinky_elastomer"]
        ]

        if (getattr(self.cfg, 'enable_deform', False) or getattr(self.cfg, 'enable_deform_vis', False)) \
                and self.cfg.enable_tactile:
            self.vbts_deform = torch.zeros(
                (self.num_envs, len(self._vbts_sensor),
                 240 // self.cfg.resolution_step, 240 // self.cfg.resolution_step),
                dtype=torch.uint8, device=self.device
            )

        # presser-specific state
        self.action_sequence_id = 0
        self.total_round = 0
        self.force_collect = torch.zeros((self.num_envs, 0, 5, 6), dtype=torch.float32, device=self.device)
        self.pos_diff = torch.zeros((self.num_envs, 0, 22), dtype=torch.float32, device=self.device)
        self.press_direction = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.target_press_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.target_press_rot = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.target_pos_counter = 0
        self.env_step_counter = 0
        self.rb_torques = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.init_finger_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.init_finger_rot = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.first_begin = True

    def _setup_scene(self):
        self.object = RigidObject(self.cfg.object_cfg)
        self.hand = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.filter_collisions()
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object

        if self.cfg.enable_tactile:
            self._contact_sensor = []
            for id in range(len(self.cfg.contact_sensor)):
                self._contact_sensor.append(ContactSensor(self.cfg.contact_sensor[id]))
                self.scene.sensors[f"contact_sensor_{id}"] = self._contact_sensor[id]
            if getattr(self.cfg, 'enable_deform', False) or getattr(self.cfg, 'enable_deform_vis', False):
                self._vbts_sensor = []
                for id in range(len(self.cfg.vbts_sensor)):
                    self._vbts_sensor.append(SharpaTacmap(self.cfg.vbts_sensor[id]))
                    self.scene.sensors[f"vbts_sensor_{id}"] = self._vbts_sensor[id]

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._axes_visualizer = None
        if getattr(self.cfg, 'debug_show_axes', False):
            axes_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/ObjectAxes")
            axes_length = getattr(self.cfg, 'vis_object_axes_length', 0.03)
            axes_marker_cfg.markers["frame"].scale = (axes_length, axes_length, axes_length)
            self._axes_visualizer = VisualizationMarkers(axes_marker_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        elastomer_name = f"right_{self.cfg.action_target_fingertip[1]}_elastomer"
        body_idx = self.hand.body_names.index(elastomer_name)
        target_finger_pos = self.hand.data.body_link_state_w[:, body_idx, :3]
        target_finger_rot = self.hand.data.body_link_state_w[:, body_idx, 3:7]

        target_press_pos_fingerframe = torch.tensor(
            self.cfg.action_target_pos[self.target_pos_counter, :3],
            dtype=torch.float32, device=self.device
        )
        target_press_rot_fingerframe = torch.tensor(
            self.cfg.action_target_pos[self.target_pos_counter, 3:],
            dtype=torch.float32, device=self.device
        )
        target_press_pos, target_press_rot = chain_transform(
            target_press_pos_fingerframe.unsqueeze(0).repeat(self.num_envs, 1),
            target_press_rot_fingerframe.unsqueeze(0).repeat(self.num_envs, 1),
            target_finger_pos, target_finger_rot
        )

        self.target_press_pos[:] = target_press_pos
        self.target_press_rot[:] = target_press_rot
        self.env_step_counter += 1

        object_default_state = self.object.data.default_root_state.clone()
        object_default_state[:, :3] = target_press_pos
        object_default_state[:, 3:7] = target_press_rot
        object_default_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_default_state[:, :7])
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:])

        finger_joint_ids = [
            self.hand.joint_names.index(name)
            for name in self.hand.joint_names
            if self.cfg.action_target_fingertip[1] in name
        ]
        self.cur_targets[:, finger_joint_ids] -= 0.05 * self.hand_dof_pos[:, finger_joint_ids]

        target_finger_pos = self.hand.data.body_link_state_w[:, body_idx, :3]
        target_finger_rot = self.hand.data.body_link_state_w[:, body_idx, 3:7]
        world_quat = torch.zeros_like(target_finger_rot)
        world_quat[..., 0] = 1.0
        real_pose = transform_between_frames(
            target_press_pos - target_finger_pos, world_quat, target_finger_rot
        )
        real_rot = quat_mul(quat_inv(target_finger_rot), target_press_rot)

    def _apply_action(self) -> None:
        self._refresh_lab()
        if self.cfg.torque_control:
            self.torques = self.p_gain * (self.cur_targets - self.hand_dof_pos) - self.d_gain * self.hand_dof_vel
            self.hand.set_joint_effort_target(
                self.torques[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
            )
        else:
            self.hand.set_joint_position_target(
                self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
            )
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

    def _get_observations(self) -> dict:
        self._refresh_lab()
        obs = self.compute_observations()
        observations = {
            "policy": obs,
            "priv_info": self.priv_info_buf,
            "proprio_hist": self.proprio_hist_buf,
        }
        if (getattr(self.cfg, 'enable_deform', False) or getattr(self.cfg, 'enable_deform_vis', False)) \
                and self.cfg.enable_tactile:
            observations['vbts_deform'] = self.vbts_deform
            observations['tactile_forces'] = self.contact_forces
            observations['tactile_points'] = self.contact_pos
        return observations

    def _get_rewards(self) -> torch.Tensor:
        net_contact_forces = torch.cat(
            [self._contact_sensor[id].data.net_forces_w_history[:, 0, 0, :].unsqueeze(1)
             for id in self._contact_body_ids], dim=1
        )
        tactile_frame_pose = self.hand.data.body_link_state_w[:, self.elastomer_ids, :7]
        tactile_frame_pos = tactile_frame_pose[..., :3]
        tactile_frame_quat = tactile_frame_pose[..., 3:7]
        contact_pos = torch.cat(
            [self._contact_sensor[id].data.contact_pos_w[:, 0, 0, :].unsqueeze(1)
             for id in self._contact_body_ids], dim=1
        )

        not_contact_mask = torch.norm(net_contact_forces, dim=-1) < 1.0e-6
        contact_mask = ~not_contact_mask
        contact_pos[not_contact_mask, :] = torch.nan

        world_quat = torch.zeros_like(tactile_frame_quat)
        world_quat[..., 0] = 1.0

        contact_pos[contact_mask, :] = transform_between_frames(
            contact_pos[contact_mask, :] - tactile_frame_pos[contact_mask, :],
            world_quat[contact_mask, :], tactile_frame_quat[contact_mask, :]
        )
        net_contact_forces = transform_between_frames(net_contact_forces, world_quat, tactile_frame_quat)

        collect_data = torch.cat([net_contact_forces, contact_pos], dim=-1).unsqueeze(1)
        self.force_collect = torch.cat([self.force_collect, collect_data], dim=1)
        pos_diff = (self.cur_targets - self.hand_dof_pos).unsqueeze(1)
        self.pos_diff = torch.cat([self.pos_diff, pos_diff], dim=1)

        return torch.tensor([0], dtype=torch.float32, device=self.device)

    def _get_dones(self):
        self._refresh_lab()
        reset_empty = torch.zeros(self.num_envs)
        reset_full = torch.ones(self.num_envs)
        self.action_sequence_id += 1

        if self.action_sequence_id == 5:
            self.action_sequence_id = 0
            self.target_pos_counter += 1

        if self.target_pos_counter == len(self.cfg.action_target_pos):
            self.target_pos_counter = 0
            self.total_round += 1
            print(f"reset env, total round: {self.total_round}")
            return reset_empty, reset_full

        return reset_empty, reset_empty

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        self._refresh_lab()

        if self.first_begin:
            elastomer_name = f"right_{self.cfg.action_target_fingertip[1]}_elastomer"
            body_idx = self.hand.body_names.index(elastomer_name)
            target_finger_pos = self.hand.data.body_link_state_w[:, body_idx, :3]
            target_finger_rot = self.hand.data.body_link_state_w[:, body_idx, 3:7]
            self.first_begin = False
            self.init_finger_pos[env_ids, :] = target_finger_pos
            self.init_finger_rot[env_ids, :] = target_finger_rot
            print(f"env_root_pos: {self.scene.env_origins[env_ids]}")
            print(f"target_finger_pos: {target_finger_pos},{target_finger_rot}")

        super()._reset_idx(env_ids)

        pos_obj_in_finger = torch.tensor(
            self.cfg.presser_init_pos[:3], dtype=torch.float32, device=self.device
        )
        rot_obj_in_finger = torch.tensor(
            self.cfg.presser_init_pos[3:], dtype=torch.float32, device=self.device
        )
        print(f"object in finger frame: {pos_obj_in_finger},{rot_obj_in_finger}")

        pos_obj_in_world, rot_obj_in_world = chain_transform(
            pos_obj_in_finger.unsqueeze(0).repeat(len(env_ids), 1),
            rot_obj_in_finger.unsqueeze(0).repeat(len(env_ids), 1),
            self.init_finger_pos, self.init_finger_rot
        )
        print(f"calculated object_cfg init state: {pos_obj_in_world},{rot_obj_in_world}")

        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, :3] = pos_obj_in_world
        object_default_state[:, 3:7] = rot_obj_in_world
        object_default_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        self.rb_forces[env_ids, :] = 0.0

        self.init_presser_state = object_default_state.clone()

        if 0 in env_ids:
            self.target_pos_counter = 0
            self.env_step_counter = 0

        dof_pos = self.hand.data.default_joint_pos[env_ids]
        dof_pos = saturate(dof_pos, self.hand_dof_lower_limits[env_ids], self.hand_dof_upper_limits[env_ids])
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self._refresh_lab()

        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]

        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def _refresh_lab(self):
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel
        self.hand_dof_torque = self.hand.data.applied_torque

        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        if getattr(self.cfg, 'debug_show_axes', False):
            self._axes_visualizer.visualize(
                translations=self.object.data.root_pos_w, orientations=self.object_rot
            )

    def compute_observations(self):
        tactile_frame_pose = self.hand.data.body_link_state_w[:, self.elastomer_ids, :7]
        tactile_frame_pos = tactile_frame_pose[..., :3]
        tactile_frame_quat = tactile_frame_pose[..., 3:7]
        world_quat = torch.zeros_like(tactile_frame_quat)
        world_quat[..., 0] = 1.0

        net_contact_forces_history = torch.cat(
            [self._contact_sensor[id].data.net_forces_w_history[:, :, 0, :].unsqueeze(2)
             for id in self._contact_body_ids], dim=2
        )
        self.contact_forces = net_contact_forces_history[:, 0, :]
        norm_contact_forces_history = torch.norm(net_contact_forces_history, dim=-1)
        smooth_contact_forces = (
            norm_contact_forces_history[:, 0, :] * self.cfg.contact_smooth
            + norm_contact_forces_history[:, 1, :] * (1 - self.cfg.contact_smooth)
        )
        smooth_contact_forces[:, self._contact_body_ids_disable] = 0.0

        if self.cfg.binary_contact:
            binary_contacts = torch.where(smooth_contact_forces > self.cfg.contact_threshold, 1.0, 0.0)
            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
            self.last_contacts = self.last_contacts * latency + binary_contacts * (1 - latency)
            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.cfg.contact_sensor_noise, 0.0, 1.0)
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
        else:
            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
            self.last_contacts = self.last_contacts * latency + smooth_contact_forces * (1 - latency)
            sensed_contacts = self.last_contacts.clone()

        not_contact_mask = sensed_contacts < 1.0e-6
        not_contact_mask[:, self._contact_body_ids_disable] = True
        contact_mask = ~not_contact_mask

        contact_pos = torch.cat(
            [self._contact_sensor[id].data.contact_pos_w[:, 0, 0, :].unsqueeze(1)
             for id in self._contact_body_ids], dim=1
        )
        nan_mask = torch.isnan(contact_pos)
        contact_pos = torch.nan_to_num(contact_pos, nan=0.0)
        contact_pos[contact_mask, :] = transform_between_frames(
            contact_pos[contact_mask, :] - tactile_frame_pos[contact_mask, :],
            world_quat[contact_mask, :], tactile_frame_quat[contact_mask, :]
        )
        contact_pos[not_contact_mask, :] = 0.0
        contact_pos[nan_mask] = 0.0

        self.contact_pos = contact_pos.clone()
        contact_pos = contact_pos.reshape(self.num_envs, -1)
        if not self.cfg.enable_contact_pos:
            contact_pos[:] = 0.0
        if not self.cfg.enable_contact_force:
            sensed_contacts[:] = 0.0
        if not self.cfg.enable_tactile:
            contact_pos[:] = 0.0
            sensed_contacts[:] = 0.0

        if (getattr(self.cfg, 'enable_deform', False) or getattr(self.cfg, 'enable_deform_vis', False)) \
                and self.cfg.enable_tactile:
            vbts_deform = torch.cat(
                [self._vbts_sensor[id].data.output["distance_along_normal"]
                 .reshape(self.num_envs, 240 // self.cfg.resolution_step, 240 // self.cfg.resolution_step)
                 .unsqueeze(1) for id in range(len(self.cfg.vbts_sensor))], dim=1
            )
            self.vbts_deform = vbts_deform.clone()

        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = (
            (torch.rand(self.hand_dof_pos.shape, device=self.device) * 2.0 - 1.0)
            * self.cfg.joint_noise_scale
        )
        cur_obs_buf = unscale(
            joint_noise_matrix + self.hand_dof_pos,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits
        ).clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        cur_obs_buf = torch.cat(
            [cur_obs_buf, sensed_contacts.clone().unsqueeze(1), contact_pos.clone().unsqueeze(1)], dim=-1
        )
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:22] = unscale(
            self.hand_dof_pos[at_reset_env_ids],
            self.hand_dof_lower_limits[at_reset_env_ids],
            self.hand_dof_upper_limits[at_reset_env_ids],
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 22:44] = self.hand_dof_pos[at_reset_env_ids].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 44:49] = sensed_contacts[at_reset_env_ids].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 49:64] = contact_pos[at_reset_env_ids].unsqueeze(1)
        self.at_reset_buf[at_reset_env_ids] = 0
        obs_buf = self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1).clone()

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.cfg.prop_hist_len:].clone()
        if getattr(self.cfg, 'priv_object_pos_first', False):
            self.priv_info_buf[:, 0:3] = self.object_pos - self.object_default_pose[:, :3]
        else:
            self.priv_info_buf[:, 5:8] = self.object_pos - self.object_default_pose[:, :3]
        if self.cfg.priv_info_dim >= 15:
            self.priv_info_buf[:, 8:12] = self.object_rot
            self.priv_info_buf[:, 12:15] = (
                axis_angle_from_quat(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
                / self.step_dt
            )

        return obs_buf
