# Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from isaacgym import gymtorch

import rofunc as rf
# from rofunc.learning.RofuncRL.tasks.isaacgym.hotu.humanoid_hotu import HumanoidHOTU
from ClothHang import ClothHang


class ClothHangViewMotionTask(ClothHang):
    def __init__(
            self,
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
    ):
        self.cfg = cfg
        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        #
        # cfg["env"]["controlFrequencyInv"] = 1
        # cfg["env"]["pdControl"] = False

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
        )

        self._motion_dt = 0.05

        num_motions = self._motion_lib.num_motions()
        # self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        # self._motion_ids = torch.remainder(self._motion_ids, num_motions)

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        self.pts = self.sim.getStateInfo().x.reshape(-1, 3)
        grasp_point1 = self.pts[self.gp1].copy()
        grasp_point2 = self.pts[self.gp2].copy()

        init = torch.cat(grasp_point1, grasp_point2)
        position_control = self.get_grasp_traj(init, actions, 1)

        x = self.sim.getStateInfo().x
        v = self.sim.getStateInfo().v
        x, v = self.sim_mod(x, v, position_control)

        return

    def post_physics_step(self):
        super().post_physics_step()
        self._motion_sync()  # Read the real action from the motion data and actuate the robot

    # def _get_humanoid_collision_filter(self):
    #     return 1  # disable self collisions

    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        # motion_ids = self._motion_ids
        motion_times = self.progress_buf * self._motion_dt

        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
            f0l, f1l
        ) = self._motion_lib.get_motion_state(motion_ids, motion_times)

        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        if self.object_names is not None:
            object_poses = self._object_motion_lib.get_motion_state(motion_ids, motion_times)
            for object_name, object_pose in object_poses.items():
                # 13-dim for the actor root state: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
                self._root_states[self._object_actor_ids[object_name][0], 0:7] = object_pose
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self._root_states),
                    gymtorch.unwrap_tensor(self._object_actor_ids[object_name]),
                    len(self._object_actor_ids[object_name]),
                )

            if "base" in self.object_names:
                base_pose = object_poses["base"]
                root_pos = base_pose[0:3].unsqueeze(0)
                # opti_base_ori = rf.robolab.euler_from_quaternion(base_pose[3:7].cpu().numpy())
                # origin_ori = rf.robolab.euler_from_quaternion(self._root_states[0, 3:7].cpu().numpy())
                # ori = torch.tensor([origin_ori[0], origin_ori[1], opti_base_ori[2]]).to(self.device)
                # root_rot = ori.unsqueeze(0)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
        )

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _compute_reset(self):
        motion_lengths = self._motion_lib.get_motion_length()
        self.reset_buf[:], self._terminate_buf[:] = compute_view_motion_reset(
            self.reset_buf, motion_lengths, self.progress_buf, self._motion_dt
        )
        return

    def _reset_actors(self, env_ids):
        return

    # def _reset_env_tensors(self, env_ids):
    #     num_motions = self._motion_lib.num_motions()
    #     self._motion_ids[env_ids] = torch.remainder(
    #         self._motion_ids[env_ids] + self.num_envs, num_motions
    #     )
    #
    #     self.progress_buf[env_ids] = 0
    #     self.reset_buf[env_ids] = 0
    #     self._terminate_buf[env_ids] = 0
    #     return


@torch.jit.script
def compute_view_motion_reset(reset_buf, motion_lengths, progress_buf, dt):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt
    reset = torch.where(
        motion_times > motion_lengths, torch.ones_like(reset_buf), terminated
    )
    return reset, terminated