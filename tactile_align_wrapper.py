import numpy as np
import cv2
import torch
from collections import deque
import os
from .vbts_viz_wrapper import VBTSVizWrapper

class TactileAlignWrapper(VBTSVizWrapper):
    def __init__(self, env, show=True, env_idx=[0]):
        super().__init__(env, show, env_idx)

        if 'action_target_pos' in self.cfg.__dict__:
            self.current_deform_stack = deque(maxlen=len(self.cfg.action_target_pos)*5)
            self.current_force_stack = deque(maxlen=len(self.cfg.action_target_pos)*5)
            self.current_contact_pos_stack = deque(maxlen=len(self.cfg.action_target_pos)*5)

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



                    if sensor_id==1 and 'action_target_pos' in self.cfg.__dict__:


                        self.current_deform_stack.append(vbts[env_id, sensor_id, :].detach().cpu().numpy())
                        self.current_force_stack.append(f)
                        
                        if point_2d is not None:
                            self.current_contact_pos_stack.append(point_2d)
                        else:
                            self.current_contact_pos_stack.append([0,0])

                        if 'target_pos_counter' in self.unwrapped.__dict__:
                            print(f"self.unwrapped.target_pos_counter {self.unwrapped.target_pos_counter}, length {len(self.cfg.action_target_pos)}")

                        if 'env_info' in self.unwrapped.cfg.__dict__ and 'save_exit' in self.unwrapped.cfg.env_info.keys() and self.unwrapped.cfg.env_info['save_exit']:
                        
                            if 'target_pos_counter' in self.unwrapped.__dict__ and self.unwrapped.target_pos_counter == len(self.cfg.action_target_pos)-1 and self.unwrapped.action_sequence_id==4:
                                print(f"Saving data to npy files at counter {self.unwrapped.target_pos_counter} ")

                                name = self.cfg.press_info
                                base_dir = os.path.dirname(os.path.abspath(__file__))
                                output_dir = os.path.join(base_dir, "outputs", "VBTS_results", name)
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir, exist_ok=True)
                                np.save(os.path.join(output_dir, "sim_vbts_deform.npy"), np.array(self.current_deform_stack))
                                np.save(os.path.join(output_dir, "sim_tactile_force.npy"), np.array(self.current_force_stack))
                                np.save(os.path.join(output_dir, "sim_tactile_contact_pos.npy"), np.array(self.current_contact_pos_stack))
                                self.current_deform_stack.clear()
                                self.current_force_stack.clear()
                                self.current_contact_pos_stack.clear()
                                print("finish")
                                exit()
                        else:
                            pass


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
                self._panel[env_id][sensor_id].update(v, f, clip_max=self._clip_max, points=[point_2d])
