# tacmap_sensor


> For the core principle and evaluation results of this tactile sensor, please refer to:  
> Lei Su, Zhijie Peng, Renyuan Ren, Shengping Mao, Juan Du, Kaifeng Zhang, Xuezhou Zhu, **"Tacmap: Bridging the Tactile Sim-to-Real Gap via Geometry-Consistent Penetration Depth Map"**, arXiv:2602.21625, 2026.  
> https://arxiv.org/abs/2602.21625

`tacmap_sensor` provides a TacMap-based tactile sensor that can be integrated into an Isaac Lab project and used like a regular simulator sensor.

This repository includes:

- Sensor implementation in `tacmap_sensor/` (`SharpaTacmap` and `SharpaTacmapCfg`)
- An Isaac Lab environment and configuration example (`env.py`, `env_cfg.py`)
- A runnable test/visualization script (`run.py`) showing end-to-end usage

## What This Repository Is For

The main goal is to make `tacmap_sensor` easy to plug into Isaac Lab, then provide a practical reference for:

- How to configure the sensor in environment config
- How to run tactile pressing tests against different object shapes
- How to visualize TacMap deformation/contact outputs
- How to save simulated tactile test data

## Key Files

- `tacmap_sensor/sharpa_tacmap_vbts.py`: TacMap sensor runtime implementation.
- `tacmap_sensor/sharpa_tacmap_cfg.py`: TacMap sensor config class.
- `env_cfg.py`: Example Isaac Lab environment config (robot, object, tactile/contact sensor config).
- `env.py`: Example environment logic that consumes tactile outputs.
- `tactile_align_wrapper.py`: Visualization/update wrapper plus data saving logic.
- `run.py`: Main script to launch simulation, load test case, and execute pressing.

## Running the Example

Run from the repository root:

```bash
python run.py --num_envs 1 --press_info assets/test_case/cylinder_D4_left_145_20250904193821_40_60.json
```

Common options:

- `--press_info`: JSON file describing which test shape/case to run (object name, initial pose, direction, and target trajectory file).
- `--render_deform_env`: Environment index for deformation rendering (default `0`).
- `--save_exit`: Save one full test round and exit automatically.
- `--num_envs`, `--seed`: Standard simulation controls.

## How `press_info` Selects a Shape/Test Case

Each JSON file in `assets/test_case/` defines one pressing scenario.  
Example fields:

- `presser_name`: object shape name (for example `cylinder_D4`)
- `presser_init_pos`: initial object position in finger frame
- `presser_init_rot`: initial object orientation
- `presser_direction`: pressing direction
- `action_target_pos_file`: trajectory/target positions used by the test

By changing `--press_info`, you switch to a different shape or pressing sequence.

## Visualization and Testing

During simulation, the wrapper can visualize:

- TacMap deformation map (`vbts_deform`)
- Tactile/contact forces
- Contact positions projected to sensor map

This makes it easy to inspect whether the sensor response matches the expected contact behavior for each test object.

## Saving Test Data with `save_exit`

When `--save_exit` is enabled, the script finishes one complete pressing round, saves outputs, then exits.

Saved files:

- `sim_vbts_deform.npy`
- `sim_tactile_force.npy`
- `sim_tactile_contact_pos.npy`

Output directory pattern:

`outputs/VBTS_results/<press_info_name>/`

where `<press_info_name>` is derived from the JSON filename used in `--press_info`.

## Typical Workflow

1. Pick a test case JSON in `assets/test_case/`.
2. Run `run.py` with that file via `--press_info`.
3. Watch deformation/contact visualization during the run.
4. Add `--save_exit` when you want to export one round of tactile data.

