from typing import Tuple

import torch

from isaaclab.utils.math import quat_mul, quat_conjugate, quat_apply, quat_inv

@torch.jit.script
def deform_quantize(deform):
    deform *= 1e3
    mask = deform < 0.5
    deform[mask] /= 5e-3
    deform[~mask] = (deform[~mask] - 0.5) / 3e-2 + 100
    deform[deform > 255] = 255
    deform[deform < 0] = 0
    return deform.to(torch.uint8)

@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) v about the rotation described by quaternion(s) q.

    Args:
        q: Quaternion(s) in (w, x, y, z). Shape (..., 4).
        v: Vector(s). Shape (..., 3).

    Returns:
        Rotated vector(s). Shape (..., 3).
    """
    zeros = torch.zeros_like(v[..., :1])
    v_as_quat = torch.cat([zeros, v], dim=-1)  # (..., 4)
    v_rot = quat_mul(quat_mul(q, v_as_quat), quat_inv(q))
    return v_rot[..., 1:]  # drop scalar part

@torch.jit.script
def transform_between_frames(p_A: torch.Tensor, q_A: torch.Tensor,
                             q_B: torch.Tensor) -> torch.Tensor:
    """Transform a point from frame A to frame B (rotation only).

    Args:
        p_A: Point(s) in frame A, shape (..., 3).
        q_A: Quaternion of frame A in world, shape (..., 4).
        q_B: Quaternion of frame B in world, shape (..., 4).

    Returns:
        Point(s) in frame B, shape (..., 3).
    """
    p_world = quat_rotate(q_A, p_A)
    p_B = quat_rotate(quat_inv(q_B), p_world)
    return p_B

# @torch.jit.script
def chain_transform(p_AB: torch.Tensor, q_AB: torch.Tensor, p_B: torch.Tensor, q_B: torch.Tensor) -> torch.Tensor:
    """
    Chain transform a point from frame A to frame B.
    p_AB: Point(s) of object in frame A, shape (..., 3).
    q_AB: Quaternion of object in frame A , shape (..., 4).
    p_B: Point(s) of frame A in frame B, shape (..., 3).
    q_B: Quaternion of frameA in frame B, shape (..., 4).
    """

    # Rotate offset by q_B (B in world) 
    p_A_world = quat_rotate(q_B, p_AB) + p_B  # (..., 3)
    # Compose quaternions: A_in_world = q_B * q_AB
    q_A_world = quat_mul(q_B, q_AB)
    return p_A_world, q_A_world
