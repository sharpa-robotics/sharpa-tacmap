# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster import patterns

from .tacmap_sensor.sharpa_tacmap_cfg import SharpaTacmapCfg



@configclass
class SharpaWaveEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    action_space = 22
    observation_space = 192
    prop_hist_len = 30
    priv_info_dim = 8
    state_space = 0
    # control
    decimation = 12
    clip_obs = 5.0
    clip_actions = 1.0
    torque_control = True
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        gravity=(0.0, 0.0, -0.05),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608, # 2**23
            gpu_max_rigid_patch_count=5*2**18
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"assets/sharpawave/right_sharpa_wave.usda"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.0002,
                rest_offset= -0.0001
            ),
            compliant_contact_stiffness=280.0,
            compliant_contact_damping=20.0,
            physics_material_prim_path=[
                "right_thumb_elastomer",
                "right_index_elastomer",
                "right_middle_elastomer",
                "right_ring_elastomer",
                "right_pinky_elastomer",
            ]
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "right_thumb_CMC_FE": math.pi/180 * 0.0,
                "right_thumb_CMC_AA": math.pi/180 * 0.0,
                "right_thumb_MCP_FE": math.pi/180 * 0.0,
                "right_thumb_MCP_AA": math.pi/180 * 0.0,
                "right_thumb_IP": math.pi/180 * 0.0,
                "right_index_MCP_FE": -math.pi/180 * 0,
                "right_index_MCP_AA": -math.pi/180 * 0,
                "right_index_PIP": math.pi/180 * 0.0,
                "right_index_DIP": math.pi/180 * 0.0,
                "right_middle_MCP_FE": math.pi/180 * 75.1,
                "right_middle_MCP_AA": math.pi/180 * 0.0,
                "right_middle_PIP": math.pi/180 * 84.8,
                "right_middle_DIP": math.pi/180 * 59.8,
                "right_ring_MCP_FE": math.pi/180 * 74.9,
                "right_ring_MCP_AA": math.pi/180 * 0.0,
                "right_ring_PIP": math.pi/180 * 74.7,
                "right_ring_DIP": math.pi/180 * 64.3,
                "right_pinky_CMC": math.pi/180 * 0.1,
                "right_pinky_MCP_FE": math.pi/180 * 79.6,
                "right_pinky_MCP_AA": math.pi/180 * 0.2,
                "right_pinky_PIP": math.pi/180 * 83.5,
                "right_pinky_DIP": math.pi/180 * 64.4,
            },
        ),
        actuators={
            "joints": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    contact_sensor = [
        # elastomer
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_thumb_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=200,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_index_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=200,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_middle_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=200,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_ring_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=200,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_pinky_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=200,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        # DP
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_thumb_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_index_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_middle_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_ring_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_pinky_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        )
    ]

    # presser object
    presser_name="square_2"
    presser_init_pos=[4.0472246564992175e-07, -0.025318063702228378, -0.022740620164500486, 0.9659258262890691, -0.25881904510251796, 9.718005352954428e-17, 9.824732604252323e-17]
    press_direction = [0.99998827, 0.00242228, 0.00419551]
    press_info = presser_name
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"assets/presser/{presser_name}.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.0001, 
                rest_offset= -0.0005
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            scale=(1., 1., 1.),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(6.000e-04, 0, 9.933e-01), rot=( 0.7015, -0.7126, -0.0049, -0.0050)),
    )

    # tacmap sensor
    POINTS_NPY_4F = "assets/tactilesensor_map/tactileSensor_map_4F_point_origin.npy"
    NORMALS_NPY_4F = "assets/tactilesensor_map/tactileSensor_map_4F_normal_origin.npy"
    POINTS_NPY_TH = "assets/tactilesensor_map/tactileSensor_map_TH_point.npy"
    NORMALS_NPY_TH = "assets/tactilesensor_map/tactileSensor_map_TH_normal.npy"
    resolution_step = 1
    vbts_sensor = [
        SharpaTacmapCfg(
            prim_path="/World/envs/env_.*/Robot/right_thumb_elastomer",
            mesh_prim_paths=[SharpaTacmapCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/object/geometry/mesh"),],
            update_period=0.0,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
            offset=SharpaTacmapCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            data_types=["distance_along_normal"],
            points_npy=POINTS_NPY_TH,
            normals_npy=NORMALS_NPY_TH,
            resolution_step=resolution_step,
            max_distance=0.015,
            debug_viz=False,
            correction_scale=1e-3,
        ),
        SharpaTacmapCfg(
            prim_path="/World/envs/env_.*/Robot/right_index_elastomer",
            mesh_prim_paths=[SharpaTacmapCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/object/geometry/mesh"),],
            update_period=0.0,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
            offset=SharpaTacmapCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            data_types=["distance_along_normal"],
            points_npy=POINTS_NPY_4F,
            normals_npy=NORMALS_NPY_4F,
            resolution_step=resolution_step,
            max_distance=0.015,
            debug_viz=True,
            correction_scale=1e-3,
        ),
        SharpaTacmapCfg(
            prim_path="/World/envs/env_.*/Robot/right_middle_elastomer",
            mesh_prim_paths=[SharpaTacmapCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/object/geometry/mesh"),],
            update_period=0.0,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
            offset=SharpaTacmapCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            data_types=["distance_along_normal"],
            points_npy=POINTS_NPY_4F,
            normals_npy=NORMALS_NPY_4F,
            resolution_step=resolution_step,
            max_distance=0.015,
            debug_viz=False,
            correction_scale=1e-3,
        ),
        SharpaTacmapCfg(
            prim_path="/World/envs/env_.*/Robot/right_ring_elastomer",
            mesh_prim_paths=[SharpaTacmapCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/object/geometry/mesh"),],
            update_period=0.0,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
            offset=SharpaTacmapCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            data_types=["distance_along_normal"],
            points_npy=POINTS_NPY_4F,
            normals_npy=NORMALS_NPY_4F,
            resolution_step=resolution_step,
            max_distance=0.015,
            debug_viz=False,
            correction_scale=1e-3,
        ),
        SharpaTacmapCfg(
            prim_path="/World/envs/env_.*/Robot/right_pinky_elastomer",
            mesh_prim_paths=[SharpaTacmapCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/object/geometry/mesh"),],
            update_period=0.0,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
            offset=SharpaTacmapCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            data_types=["distance_along_normal"],
            points_npy=POINTS_NPY_4F,
            normals_npy=NORMALS_NPY_4F,
            resolution_step=resolution_step,
            max_distance=0.015,
            debug_viz=False,
            correction_scale=1e-3,
        ),
    ]

    # joint names
    actuated_joint_names = [
        "right_thumb_CMC_FE",
        "right_thumb_CMC_AA",
        "right_thumb_MCP_FE",
        "right_thumb_MCP_AA",
        "right_thumb_IP",
        "right_index_MCP_FE",
        "right_index_MCP_AA",
        "right_index_PIP",
        "right_index_DIP",
        "right_middle_MCP_FE",
        "right_middle_MCP_AA",
        "right_middle_PIP",
        "right_middle_DIP",
        "right_ring_MCP_FE",
        "right_ring_MCP_AA",
        "right_ring_PIP",
        "right_ring_DIP",
        "right_pinky_CMC",
        "right_pinky_MCP_FE",
        "right_pinky_MCP_AA",
        "right_pinky_PIP",
        "right_pinky_DIP",
    ]
    fingertip_body_names = [
        "right_thumb_fingertip",
        "right_index_fingertip",
        "right_middle_fingertip",
        "right_ring_fingertip",
        "right_pinky_fingertip",
    ]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16384, env_spacing=0.75, replicate_physics=False)
    # noise
    joint_noise_scale = 0.02
    # contact
    enable_tactile = True
    enable_deform = False
    enable_deform_vis = True
    enable_contact_force = True
    binary_contact = False
    enable_contact_pos = True
    disable_tactile_ids = []
    contact_smooth = 0.5
    contact_threshold = 0.05
    contact_latency = 0.005
    contact_sensor_noise = 0.01
    # joint limits
    dof_limits_scale = 0.9
    # debug
    debug_show_axes = False

    # presser action
    action_target_fingertip = (1,'index')
    action_target_pos = [[-0.024952680912017827, -0.018483116497920175, 0.002010279779405183, 0.7071067811801017, -3.019609190115596e-06, 0.7071067811800986, -3.0196091876020132e-06]]

    env_info = {}
