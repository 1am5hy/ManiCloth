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

from enum import Enum

import gym
import numpy as np
import rofunc as rf
import torch
from gym import spaces

from ork.tasks.motion_lib import MotionLib
from gym.spaces import Box

class ClothRelease(gym.Env):
    def __init__(self, cfg, sim_device, cloth_env):
        self.cfg = cfg
        self.device = sim_device
        # self.observation_space = Box(-np.inf, np.inf, (675,))
        self.observation_space = Box(-np.inf, np.inf, (30,))
        # self.observation_space = Box(-np.inf, np.inf, (6,))
        self.action_space = Box(0, 100, (1,), dtype=int)
        # self.action_space = Box(75, 150, (1,), dtype=int)

        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        self._num_amp_obs_per_step = 30   # Observed particles + tool pose
        # self._num_amp_obs_per_step = 6
        # self._num_amp_obs_per_step = 675
        self._num_amp_target_obs = 30  # Target particle states size
        # self._num_amp_target_obs = 6
        # self._num_amp_target_obs = 675
        self._amp_hindsight_target_flag = cfg["env"][
            "AMPHindsightTargetFlag"]  # Whether to use hindsight target for AMP observation
        self._random_initial_flag = cfg["env"][
            "RandomInitialFlag"]  # Whether to use random initial state for the task in one episode
        self._random_target_flag = cfg["env"][
            "RandomTargetFlag"]  # Whether to use random target state for the task in one episode
        self._use_target_flag = cfg["env"][
            "UseTargetFlag"]  # Whether to use target state for the task in one episode
        assert self._num_amp_obs_steps >= 2

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        # motion_file = "/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/marker_path_dyndemo.npy"
        # motion_file = "/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/obs_hang.npy"
        # motion_file = "/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_30.npy"
        # motion_file = "/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/relative_marker_hang_task_30.npy"
        # motion_file = "/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_hang.npy"
        # print(np.load(motion_file).shape)
        # new = np.load(motion_file).reshape(len(np.load(motion_file)), -1)
        # print(new.shape)
        # np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_30.npy", new)
        # for i in range()

        # motion_file = "/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/gp_dyndemo.npy"
        # if rf.oslab.is_absl_path(motion_file):
        #     motion_file_path = motion_file
        # else:
        #     raise ValueError(f"Unsupported motion file path: {motion_file}")

        # self._load_motion(motion_file)

        # self._amp_obs_space = spaces.Box(np.ones(self.get_num_amp_obs()) * -np.Inf,
        #                                  np.ones(self.get_num_amp_obs()) * np.Inf)
        # self._amp_obs_buf = torch.zeros((1, self._num_amp_obs_steps, self._num_amp_obs_per_step),
        #                                 device=self.device, dtype=torch.float)
        # self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]  # Current observation
        # self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]  # History observations

        # self.target_obs = self._motion_lib.get_motion_state(0, int(self._motion_lib._motion_lengths[0] - 1))[0]
        # interest1 = np.array(
        #     [0, 1, 2, 21, 22, 23, 42, 43, 44, 189, 190, 191, 360, 361, 362, 399, 400, 401, 519, 520, 521, 765, 766, 767,
        #      786, 787, 788, 807, 808, 809])
        self.target_obs = np.load("/home/ubuntu/Github/ManiCloth/src/python_code/ReleaseRL/np_files/relative_table_release.npy")
        # print(self.target_obs.shape)
        # print(self)
        # print("self.target_obs = {}".format(self.target_obs))
        self._target_obs_space = spaces.Box(np.ones(self._num_amp_target_obs) * -np.Inf,
                                            np.ones(self._num_amp_target_obs) * np.Inf)

        # self._amp_obs_demo_buf = None

        self.cloth_env = cloth_env

    def reset(self):
        obs = self.cloth_env.reset()
        # print(obs.shape)
        # self._compute_amp_observations(obs)
        # self._init_amp_obs_default([0])

        return obs

    def get_obs(self):
        return self.cloth_env.get_obs()

    def step(self, action):
        next_states, rewards, terminated, infos = self.cloth_env.step(action)
        # self._update_hist_amp_obs()
        # self._compute_amp_observations(next_states)

        # amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        #
        # infos["amp_obs"] = amp_obs_flat
        # infos["target_obs"] = self.target_obs
        return next_states, rewards, terminated, infos

    # def render(self, mode="human", **kwargs):
    #     return self.plast_env.render(mode, **kwargs)

    # def get_num_amp_obs(self):
    #     return self._num_amp_obs_steps * self._num_amp_obs_per_step

    @property
    # def amp_observation_space(self):
    #     return self._amp_obs_space

    @property
    def target_observation_space(self):
        return self._target_obs_space

    # def fetch_amp_obs_demo(self, num_samples):
    #     dt = 1
    #     motion_ids = self._motion_lib.sample_motions(num_samples)
    #
    #     if self._amp_obs_demo_buf is None:
    #         self._build_amp_obs_demo_buf(num_samples)
    #     else:
    #         assert self._amp_obs_demo_buf.shape[0] == num_samples
    #
    #     # since negative times are added to these values in build_amp_obs_demo,
    #     # we shift them into the range [0 + truncate_time, end of clip]
    #     truncate_time = dt * (self._num_amp_obs_steps - 1)
    #     motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
    #     motion_times0 += truncate_time
    #
    #     amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
    #     return amp_obs_demo

    # def build_amp_obs_demo(self, motion_ids0, motion_times0):
    #     dt = 1
    #
    #     motion_ids = torch.tile(motion_ids0.unsqueeze(-1), [1, self._num_amp_obs_steps])
    #     motion_times = motion_times0.unsqueeze(-1)
    #     time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
    #     motion_times = motion_times + time_steps
    #
    #     motion_ids = motion_ids.view(-1)
    #     motion_times = motion_times.view(-1)
    #     # obs, r, tool_pose, target_obs = self._motion_lib.get_motion_state(motion_ids, motion_times)
    #     # obs = torch.cat([obs, tool_pose], dim=-1)
    #     # self._amp_obs_demo_buf[:] = obs.view(self._amp_obs_demo_buf.shape)
    #     amp_obs = self._motion_lib.get_motion_state(motion_ids, motion_times)
    #     # amp_obs = torch.cat([amp_obs, amp_tool_pose], dim=-1)
    #     self._amp_obs_demo_buf = amp_obs.view(self._amp_obs_demo_buf.shape)
    #
    #     amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
    #     return amp_obs_demo_flat

    # def _build_amp_obs_demo_buf(self, num_samples):
    #     self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step),
    #                                          device=self.device, dtype=torch.float32)

    # def _load_motion(self, motion_file):
    #     self._motion_lib = MotionLib(motion_file=motion_file, device=self.device)
    #
    # def _init_amp_obs_default(self, env_ids):
    #     curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
    #     self._hist_amp_obs_buf[env_ids] = curr_amp_obs
    #     return
    #
    # def _update_hist_amp_obs(self):
    #     for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
    #         self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]

    # def _compute_amp_observations(self):
    #
    #     mot = torch.zeros((len(self._motion_lib._motions), len(self._motion_lib._motions[0]), 6), device=self.device)
    #     for i in range(len(self._motion_lib._motions)):
    #         mot[i] = torch.Tensor(self._motion_lib._motions[i])
    #     mot = mot[0]
    #     self._curr_amp_obs_buf[:] = mot
    #     return

    # def _compute_amp_observations(self, next_states):
    #     self._curr_amp_obs_buf[:] = torch.tensor(next_states[:self._num_amp_obs_per_step]).to(self.device)