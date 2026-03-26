import numpy as np
import cv2
import torch
from collections import deque
import os


class _VBTSPanel:
    def __init__(self, H: int, W: int, resolution_step: int, title: str = "VBTS Deform (env 0)"):
        import omni.ui as ui
        self.ui = ui
        self.H, self.W = int(H), int(W)
        self.resolution_step = resolution_step
        self._win = ui.Window(title, width=self.W + 24, height=self.H + 90)
        with self._win.frame:
            with ui.VStack(spacing=6):
                self._stats = ui.Label("min=0.0  max=0.0  mean=0.0")
                self._prov = ui.ByteImageProvider()
                ui.ImageWithProvider(self._prov, width=self.W, height=self.H)

    @staticmethod
    def _to_rgba(arr2d_np, clip_max=None):
        a = np.nan_to_num(arr2d_np, nan=0.0, posinf=0.0, neginf=0.0)
        if clip_max <= 1e-12:
            g = np.zeros_like(a, dtype=np.uint8)
        else:
            g = np.clip(a, 0.0, clip_max).astype(np.uint8)
        return np.stack([g, g, g, np.full_like(g, 255, dtype=np.uint8)], axis=-1)

    def update(self, arr2d_np, f=None, clip_max=None, points:list|None=None):
        rgba = self._to_rgba(arr2d_np, clip_max=clip_max)
        rgb = rgba[:, :, :3]
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)     # (H, W, 3), C-contiguous
        if f is not None:
            z = 20 // self.resolution_step
            for var in f:
                cv2.putText(rgb, f'{var:.3f}', (5//self.resolution_step, z), cv2.FONT_HERSHEY_SIMPLEX, 0.6/self.resolution_step, (0, 255, 0), 2)
                z += 20 // self.resolution_step
        
        if points:
            for pt in points:
                if pt is None:
                    continue
                x = int(round(pt[0]))
                y = int(round(pt[1]))
                if x<0 or x>=self.W or y<0 or y>=self.H:
                    continue
                cv2.circle(rgb, (x, y), radius=4//self.resolution_step, color=(0, 0, 255))

        rgba[:, :, :3] = rgb
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)   # (H, W, 4), C-contiguous
        flat = rgba.reshape(-1)  # 1-D view (avoids multi-dimensional memoryview)
        self._prov.set_bytes_data(memoryview(flat), (self.W, self.H))
        self._stats.text = f"min={arr2d_np.min():.6f}  max={arr2d_np.max():.6f}  mean={arr2d_np.mean():.6f}"

        


class VBTSVizWrapper:
    def __init__(self, env, show=True, env_idx=[0]):
        self._env = env
        self._show = bool(show)
        self._env_idx = env_idx                    # which env to display
        self._sensor_idx = []
        for i, vbts_cfg in enumerate(self.cfg.vbts_sensor):
            if vbts_cfg.debug_viz:
                self._sensor_idx.append(i)
        self._panel = None
        self._H = 240 / self.cfg.resolution_step   # you can change these if your VBTS uses a different resolution
        self._W = 240 / self.cfg.resolution_step
        self._clip_max = 255                       # set to your sensor's max distance if you want fixed brightness

        POINTS_NPY_4F = "assets/tactilesensor_map/tactileSensor_map_4F_point.npy"
        NORMALS_NPY_4F = "assets/tactilesensor_map/tactileSensor_map_4F_normal.npy"
        POINTS_NPY_TH = "assets/tactilesensor_map/tactileSensor_map_TH_point.npy"
        NORMALS_NPY_TH = "assets/tactilesensor_map/tactileSensor_map_TH_normal.npy"

        self.th_points = np.load(POINTS_NPY_TH)/1000.0
        self.th_normals = np.load(NORMALS_NPY_TH)/1000.0
        self.f4_points = np.load(POINTS_NPY_4F)/1000.0
        self.f4_normals = np.load(NORMALS_NPY_4F)/1000.0



    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        self._maybe_init_panel(obs)
        self._update_panel(obs)
        return out

    def step(self, action):
        out = self._env.step(action)
        obs = out[0] if isinstance(out, tuple) else out
        self._update_panel(obs)
        return out

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _maybe_init_panel(self, observations):
        if not self._show or self._panel is not None: 
            print("[VBTSVizProxy] Warning: VBTSVizProxy already initialized. Skipping...")
            return
        if not isinstance(observations, dict) or "vbts_deform" not in observations: 
            print("[VBTSVizProxy] Warning: 'vbts_deform' not found in observations during init. Skipping...")
            return

        vbts = observations["vbts_deform"]
        assert isinstance(vbts, torch.Tensor)

        # Expect [num_envs, H*W, num_sensors]
        if vbts.ndim != 4 or vbts.shape[2] != self._H or vbts.shape[3] != self._W:
            print(
                "[VBTSVizProxy] Error: shape mismatch in vbts_deform during init.\n"
                f"  Expected: [num_envs, num_sensors, H, W] "
                f"(H={self._H}, W={self._W})\n"
                f"  Got:      {tuple(vbts.shape)}\n"
                "  Please adjust H and W in VBTSVizProxy or change how vbts_deform is flattened."
            )
            return

        self._panel = {env_id: {sensor_id: _VBTSPanel(
            self._H,
            self._W,
            self.cfg.resolution_step,
            title=f"VBTS Deform (env {env_id}, sensor {sensor_id})",
        ) for sensor_id in self._sensor_idx} for env_id in self._env_idx}

    def _get_2d_point(self, point_3d, sensor_id):
        if sensor_id == 0:
            if point_3d is not None and hasattr(self, "th_points") and np.linalg.norm(point_3d)>1e-3:
                points = self.th_points.reshape(-1, 3)  # (H*W, 3)
                dists = ((points - point_3d) ** 2).sum(axis=1)
                min_idx = dists.argmin()
                i, j = divmod(min_idx, self._W)
                point_2d = [j, i]
            else:
                point_2d = None
        else:
            if point_3d is not None and hasattr(self, "f4_points") and np.linalg.norm(point_3d)>1e-3:
                points = self.f4_points.reshape(-1, 3)  # (H*W, 3)
                dists = ((points - point_3d) ** 2).sum(axis=1)
                min_idx = dists.argmin()
                i, j = divmod(min_idx, self._W)
                point_2d = [j, i]
                
            else:
                point_2d = None
            
        return point_2d



    def _update_panel(self, observations):

        if not isinstance(observations, dict) or "vbts_deform" not in observations: return

        vbts = observations["vbts_deform"]
        forces = observations["tactile_forces"] if "tactile_forces" in observations else None
        contact_pos = observations["tactile_points"] if "tactile_points" in observations else None
       

        if not isinstance(vbts, torch.Tensor): return

        # guard env index
        num_sensors = vbts.shape[1]
        for env_id in self._env_idx:
            for sensor_id in self._sensor_idx:
                # UPDATED: handle multi-sensor output [N, H*W, S] strictly
                if vbts.ndim == 4 or vbts.shape[2] == self._H or vbts.shape[3] == self._W:
                    # sensor index check
                    if sensor_id >= num_sensors:
                        print(
                            "[VBTSVizProxy] Error: requested sensor index out of range.\n"
                            f"  --vbts_index = {sensor_id}\n"
                            f"  But vbts_deform has only {num_sensors} sensors "
                            f"(last dimension of shape {tuple(vbts.shape)}).\n"
                            f"  Please choose --vbts_index in [0, {num_sensors - 1}]."
                        )
                        return

                    v = (
                        vbts[env_id, sensor_id, :]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    f = forces[env_id, sensor_id, :].detach().cpu().numpy() if forces is not None else None
                    point_3d = contact_pos[env_id, sensor_id, :].detach().cpu().numpy() if contact_pos is not None else None

                    point_2d = self._get_2d_point(point_3d, sensor_id)

                else:
                    # Shape mismatch: not [N, H*W, S]
                    print(
                        "[VBTSVizProxy] Error: shape mismatch in vbts_deform during update.\n"
                        f"  Expected: [num_envs, num_sensors, H, W] "
                        f"(H={self._H}, W={self._W})\n"
                        f"  Got:      {tuple(vbts.shape)}\n"
                        "  Please adjust H and W in VBTSVizProxy or change how vbts_deform is flattened."
                    )
                    return
                if not self._show or self._panel is None: return
                self._panel[env_id][sensor_id].update(v, f, clip_max=self._clip_max, points=None)
