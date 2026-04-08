from __future__ import annotations

"""Script for tactile align presser data collection."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tactile align presser data collection.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--render_deform_env", type=int, default=0, help="Display deformations for specified env index.")
parser.add_argument("--press_info", type=str, default="assets/test_case/cylinder_D4_left_145_20250904193821_40_60.json", help="Path to JSON file containing presser configuration.")
parser.add_argument("--save_exit", action="store_true", default=False, help="Save data and exit after one full round.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Parse presser configuration from JSON
args_cli.presser_name = None
args_cli.presser_init_pos = None
args_cli.presser_init_rot = None
args_cli.presser_direction = None
args_cli.action_target_pos_file = None
args_cli.press_info_pure = None

if args_cli.press_info is not None:
    json_path = os.path.abspath(args_cli.press_info)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Press info JSON file not found: {json_path}")

    with open(json_path, 'r') as f:
        press_config = json.load(f)

    args_cli.presser_name = press_config.get("presser_name")
    args_cli.presser_init_pos = press_config.get("presser_init_pos")
    args_cli.presser_init_rot = press_config.get("presser_init_rot")
    args_cli.presser_direction = press_config.get("presser_direction")
    args_cli.action_target_pos_file = press_config.get("action_target_pos_file")
    args_cli.press_info_pure = os.path.basename(json_path).split(".")[0]

    print(f"  presser_name: {args_cli.presser_name}")
    print(f"  presser_init_pos: {args_cli.presser_init_pos}")
    print(f"  presser_init_rot: {args_cli.presser_init_rot}")
    print(f"  presser_direction: {args_cli.presser_direction}")
    print(f"  action_target_pos_file: {args_cli.action_target_pos_file}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np

from tacmap_sensor.env import SharpaWaveInhandRotateTactileAlignEnv
from tacmap_sensor.env_cfg import SharpaWaveEnvCfg
from tacmap_sensor.tactile_align_wrapper import TactileAlignWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    env_cfg = SharpaWaveEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.save_exit:
        env_cfg.env_info['save_exit'] = True

    if args_cli.presser_name is not None:
        env_cfg.object_cfg.spawn.usd_path = env_cfg.object_cfg.spawn.usd_path.replace(
            env_cfg.presser_name, args_cli.presser_name
        )
        env_cfg.presser_name = args_cli.presser_name

    if args_cli.presser_direction is not None:
        env_cfg.press_direction = args_cli.presser_direction

    if args_cli.press_info_pure is not None:
        env_cfg.press_info = args_cli.press_info_pure

    if args_cli.presser_init_pos is not None:
        env_cfg.presser_init_pos[:3] = args_cli.presser_init_pos

    if hasattr(env_cfg, "press_direction") and env_cfg.press_direction is not None:
        direction = np.array(env_cfg.press_direction, dtype=np.float64)
        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)
            delta = -0.004 * direction
            env_cfg.presser_init_pos[:3] = (
                np.array(env_cfg.presser_init_pos[:3], dtype=np.float64) + delta
            ).tolist()

    if args_cli.presser_init_rot is not None:
        env_cfg.presser_init_pos[3:] = args_cli.presser_init_rot

    if args_cli.action_target_pos_file is not None:
        action_target_pos_path = args_cli.action_target_pos_file
        if not os.path.isabs(action_target_pos_path):
            if args_cli.press_info is not None:
                press_info_dir = os.path.dirname(os.path.abspath(args_cli.press_info))
                action_target_pos_path = os.path.join(press_info_dir, action_target_pos_path)
            action_target_pos_path = os.path.abspath(action_target_pos_path)

        if not os.path.exists(action_target_pos_path):
            raise FileNotFoundError(f"action_target_pos file not found: {action_target_pos_path}")

        env_cfg.action_target_pos = np.load(action_target_pos_path)
    else:
        env_cfg.action_target_pos = np.array(env_cfg.action_target_pos)

    env = SharpaWaveInhandRotateTactileAlignEnv(env_cfg)

    if args_cli.render_deform_env is not None:
        show = not getattr(args_cli, "headless", False)
        env = TactileAlignWrapper(env, show=show, env_idx=[args_cli.render_deform_env])

    env.reset()
    while True:
        actions = torch.zeros(env_cfg.scene.num_envs, env_cfg.action_space, device=env.device)
        env.step(actions)


if __name__ == "__main__":
    main()
    simulation_app.close()
