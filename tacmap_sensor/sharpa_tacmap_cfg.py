from dataclasses import MISSING
from typing import Literal, List

from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg


@configclass
class SharpaTacmapCfg(MultiMeshRayCasterCfg):
    """配置 Sharpa Vision-Based Tactile Sensor (VBTS)，支持并行环境及目标物体绑定"""

    # vbts, sampled points and normal directions（numpy formate）
    points_npy: str = MISSING       # "/path/to/surface_points.npy"
    normals_npy: str = MISSING      # "/path/to/surface_normals.npy"
    resolution_step: int = MISSING  # 1 for (240, 240), 2 for (120, 120)
    
    # offsets and coordinate constrain
    @configclass
    class OffsetCfg:
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
        convention: Literal["opengl", "ros", "world"] = "ros"
        
    offset: OffsetCfg = OffsetCfg()

    # output data type（default is "distance_along_normal"）
    data_types: List[str] = ["distance_along_normal"]

    target_rigid_expr = None

    # numpy points use mm as the unit, may different from the sim world
    correction_scale: float = 1e-3
    pts_offsets: float = 0.0000
    max_distance = 0.015

    # contact points detection (CPD inside test) parameters
    # move cpd_max_dist (in world unit) after first surface hit to check whether it will hit the second surface
    cpd_max_dist: float = 0.5  
    debug_viz = False

    # use the pattern config from ray-caster (required but not used)
    pattern_cfg: PinholeCameraPatternCfg = MISSING
