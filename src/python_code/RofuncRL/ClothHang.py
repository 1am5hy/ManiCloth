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

import os
from enum import Enum
import numpy as np

import torch
from gym import spaces
# from isaacgym.torch_utils import *

import rofunc as rf
from src.python_code.RofuncRL.ClothBase import Cloth
from motion_lib import MotionLib, ObjectMotionLib
# from rofunc.learning.RofuncRL.tasks.isaacgym.hotu.motion_lib import MotionLib, ObjectMotionLib
from rofunc.learning.RofuncRL.tasks.utils import torch_jit_utils as torch_utils


class ClothHang(Cloth):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id):
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = ClothHang.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert self._num_amp_obs_steps >= 2
        self.randomize = False

        # self._reset_default_env_ids = []
        # self._reset_ref_env_ids = []

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id)

        # Load motion file
        motion_file = "/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/gp_dyndemo.npy"
        # if rf.oslab.is_absl_path(motion_file):
        #     motion_file_path = motion_file
        # elif motion_file.split("/")[0] == "examples":
        #     motion_file_path = os.path.join(
        #         os.path.dirname(os.path.abspath(__file__)),
        #         "../../../../../../" + motion_file,
        #     )
        # else:
        #     motion_file_path = os.path.join(
        #         os.path.dirname(os.path.abspath(__file__)),
        #         "../../../../../../examples/data/amp/" + motion_file,
        #     )
        self._load_motion(motion_file)

        # Load object motion file
        object_motion_file = "/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/marker_path_dyndemo.npy"
        # if object_motion_file is not None:
        #     if rf.oslab.is_absl_path(object_motion_file):
        #         object_motion_file_path = object_motion_file
        #     elif object_motion_file.split("/")[0] == "examples":
        #         object_motion_file_path = os.path.join(
        #             os.path.dirname(os.path.abspath(__file__)),
        #             "../../../../../../" + object_motion_file,
        #         )
        #     else:
        #         raise ValueError("Unsupported object motion file path")
        self._load_object_motion(object_motion_file)
        # Set up the observation space for AMP
        self._amp_obs_space = spaces.Box(np.ones(self.get_num_amp_obs()) * -np.Inf,
                                         np.ones(self.get_num_amp_obs()) * np.Inf)
        self._amp_obs_buf = torch.zeros((1, self._num_amp_obs_steps, self._num_amp_obs_per_step),
                                        device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[0]
        self._hist_amp_obs_buf = self._amp_obs_buf[1:]

        self._amp_obs_demo_buf = None

    def post_physics_step(self):

        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

    def get_num_amp_obs(self):
        # print(self._num_amp_obs_steps)
        # print(self._num_amp_obs_per_step)
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    @property
    def amp_observation_space(self):
        # print(self._amp_obs_space)
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):

        return self.fetch_amp_obs_demo_other()

    def fetch_amp_obs_demo_other(self):
        dt = self.dt

        if self._amp_obs_demo_buf is None:
            self._build_amp_obs_demo_buf()
        # print(self._amp_obs_demo_buf.shape)

        amp_obs_demo = self.build_amp_obs_demo()
        # print(amp_obs_demo.shape)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self):
        dt = self.dt

        motion = self._motion_lib.get_motion_state()
        # print(type(motion))
        motion = torch.Tensor(motion[0])
        amp_obs_demo = build_amp_observations(motion)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self):
        self._amp_obs_demo_buf = torch.zeros((self._num_amp_obs_steps, self._num_amp_obs_per_step),
                                             device=self.device, dtype=torch.float32)

    def _setup_character_props(self):

        self._num_amp_obs_per_step = 6


    def _load_motion(self, motion_file):
        self._motion_lib = MotionLib(
            motion_file=motion_file,
            device=self.device,
        )

    def _load_object_motion(self, object_motion_file):
        self._object_motion_lib = ObjectMotionLib(
            object_motion_file=object_motion_file,
            device=self.device,
        )

    def reset_idx(self):
        # self._reset_default_env_ids = []
        # self._reset_ref_env_ids = []

        super().reset_idx()
        return

    # def _reset_actors(self, env_ids):
    #     if self._state_init == ClothHang.StateInit.Default:
    #         self._reset_default(env_ids)
    #     elif (
    #             self._state_init == ClothHang.StateInit.Start
    #             or self._state_init == ClothHang.StateInit.Random
    #     ):
    #         self._reset_ref_state_init(env_ids)
    #     elif self._state_init == ClothHang.StateInit.Hybrid:
    #         self._reset_hybrid_state_init(env_ids)
    #     else:
    #         assert False, "Unsupported state initialization strategy: {:s}".format(
    #             str(self._state_init)
    #         )
    #     return
    #
    # def _reset_default(self, env_ids):
    #     self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[
    #         env_ids
    #     ]
    #     self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
    #     self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
    #     self._reset_default_env_ids = env_ids
    #     return
    #
    # def _reset_ref_state_init(self, env_ids):
    #     num_envs = env_ids.shape[0]
    #     motion_ids = self._motion_lib.sample_motions(num_envs)
    #
    #     if (
    #             self._state_init == HumanoidHOTU.StateInit.Random
    #             or self._state_init == HumanoidHOTU.StateInit.Hybrid
    #     ):
    #         motion_times = self._motion_lib.sample_time(motion_ids)
    #     elif self._state_init == HumanoidHOTU.StateInit.Start:
    #         motion_times = torch.zeros(num_envs, device=self.device)
    #     else:
    #         assert (
    #             False
    #         ), f"Unsupported state initialization strategy: {self._state_init}"
    #
    #     (
    #         root_pos,
    #         root_rot,
    #         dof_pos,
    #         root_vel,
    #         root_ang_vel,
    #         dof_vel,
    #         key_pos,
    #     ) = self._motion_lib.get_motion_state(motion_ids, motion_times)
    #
    #     self._set_env_state(
    #         env_ids=env_ids,
    #         root_pos=root_pos,
    #         root_rot=root_rot,
    #         dof_pos=dof_pos,
    #         root_vel=root_vel,
    #         root_ang_vel=root_ang_vel,
    #         dof_vel=dof_vel,
    #     )
    #
    #     self._reset_ref_env_ids = env_ids
    #     self._reset_ref_motion_ids = motion_ids
    #     self._reset_ref_motion_times = motion_times
    #     return
    #
    # def _reset_hybrid_state_init(self, env_ids):
    #     num_envs = env_ids.shape[0]
    #     ref_probs = to_torch(
    #         np.array([self._hybrid_init_prob] * num_envs), device=self.device
    #     )
    #     ref_init_mask = torch.bernoulli(ref_probs) == 1.0
    #
    #     ref_reset_ids = env_ids[ref_init_mask]
    #     if len(ref_reset_ids) > 0:
    #         self._reset_ref_state_init(ref_reset_ids)
    #
    #     default_reset_ids = env_ids[torch.logical_not(torch.tensor(ref_init_mask))]
    #     if len(default_reset_ids) > 0:
    #         self._reset_default(default_reset_ids)
    #
    #     return
    #
    # def _init_amp_obs(self, env_ids):
    #     self._compute_amp_observations(env_ids)
    #
    #     if len(self._reset_default_env_ids) > 0:
    #         self._init_amp_obs_default(self._reset_default_env_ids)
    #
    #     if len(self._reset_ref_env_ids) > 0:
    #         self._init_amp_obs_ref(
    #             self._reset_ref_env_ids,
    #             self._reset_ref_motion_ids,
    #             self._reset_ref_motion_times,
    #         )
    #     return
    #
    # def _init_amp_obs_default(self, env_ids):
    #     curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
    #     self._hist_amp_obs_buf[env_ids] = curr_amp_obs
    #     return
    #
    # def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
    #     dt = self.dt
    #     motion_ids = torch.tile(
    #         motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1]
    #     )
    #     motion_times = motion_times.unsqueeze(-1)
    #     time_steps = -dt * (
    #             torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1
    #     )
    #     motion_times = motion_times + time_steps
    #
    #     motion_ids = motion_ids.view(-1)
    #     motion_times = motion_times.view(-1)
    #     (
    #         root_pos,
    #         root_rot,
    #         dof_pos,
    #         root_vel,
    #         root_ang_vel,
    #         dof_vel,
    #         key_pos,
    #     ) = self._motion_lib.get_motion_state(motion_ids, motion_times)
    #     amp_obs_demo = build_amp_observations(
    #         root_pos,
    #         root_rot,
    #         root_vel,
    #         root_ang_vel,
    #         dof_pos,
    #         dof_vel,
    #         key_pos,
    #         self._local_root_obs,
    #         self._root_height_obs,
    #         self._dof_obs_size,
    #         self._dof_offsets,
    #     )
    #     self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(
    #         self._hist_amp_obs_buf[env_ids].shape
    #     )
    #     return


    def _update_hist_amp_obs(self):
        for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
            # print(i)
            self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]


        return

    def _compute_amp_observations(self):
        # print(self._motion_lib._motions[0]))

        mot = torch.zeros((len(self._motion_lib._motions), len(self._motion_lib._motions[0]), 6), device=self.device)
        for i in range(len(self._motion_lib._motions)):
            mot[i] = torch.Tensor(self._motion_lib._motions[i])
        mot = mot[0]
        # print(self._curr_amp_obs_buf.shape)
        # print(mot.shape)
        self._curr_amp_obs_buf[:] = build_amp_observations(mot)
        return


@torch.jit.script
def build_amp_observations(
        motion,
):
    # type: (Tensor) -> Tensor
    obs = motion

    return obs