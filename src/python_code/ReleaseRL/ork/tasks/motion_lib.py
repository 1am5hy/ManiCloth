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

import numpy as np
import rofunc as rf
import json
import torch
import yaml
import tqdm


class MotionLib:
    def __init__(self, motion_file, device):
        self._device = device

        self._load_motions(motion_file)
        self.process_data()

    def process_data(self):
        """
        Process the data to be used for training.
        """
        motions = self._motions

        self.obs = torch.cat([torch.tensor([data_one_step]) for m in motions for data_one_step in m],
                             dim=0).float().to(self._device)
        self.r = torch.cat([torch.tensor([data_one_step]) for m in motions for data_one_step in m],
                           dim=0).float().to(self._device)
        self.tool_pose = torch.cat([torch.tensor([np.array([*data_one_step[0:3], *data_one_step[6:9]])]) for m in motions for data_one_step in m],
                                   dim=0).float().to(self._device)
        self.target_obs = torch.cat(
            [torch.tensor([m[-1]]) for m in motions for _ in m],
            dim=0).float().to(self._device)
        self.state = torch.cat([torch.tensor([data_one_step]) for m in motions for data_one_step in m],
                               dim=0).float().to(self._device)
        # self.grid = torch.cat([torch.tensor([data_one_step["grid"]]) for m in motions for data_one_step in m],
        #                       dim=0).float().to(self._device)
        # self.obs = torch.tensor(motions[0].to(self._device))
        # # print(self.obs)
        # # self.obs = torch.cat([torch.tensor([data_one_time["obs"]]) for m in motions for data_one_time in m],
        # #                      dim=0).float().to(self._device)
        # # self.r = torch.cat([torch.tensor([data_one_time["r"]]) for m in motions for data_one_time in m],
        # #                    dim=0).float().to(self._device)
        # self.tool_pose = torch.zeros([len(self.obs), 2, 3]).to(self._device)
        # # self.tool_pose[:, 0] = self.obs[:, 0, :]
        # # self.tool_pose[:, 1] = self.obs[:, 2, :]
        # self.tool_pose[:, 0] = self.obs[:, 0:3]
        # self.tool_pose[:, 1] = self.obs[:, 6:9]
        # # self.tool_pose = torch.hstack([self.obs[:, 0, :], self.obs[:, 2, :]])
        # # self.tool_pose = torch.cat([torch.tensor([data_one_time["tool_pose"]]) for m in motions for data_one_time in m],
        # #                            dim=0).float().to(self._device)
        # self.target_obs = self.obs[-1]

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        motion_ids = torch.multinomial(
            self._motion_weights, num_samples=n, replacement=True
        )

        # m = self.num_motions()
        # motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)
        # motion_ids = torch.tensor(motion_ids, device=self._device, dtype=torch.long)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        """

        :param motion_ids: [amp_batch_size]
        :param truncate_time:
        :return:
        """
        phase = torch.rand(motion_ids.shape, device=self._device)

        motion_len = self._motion_lengths[motion_ids]
        motion_lengths2 = torch.tensor([70], dtype=torch.int64, device=self._device)
        motion_len2 = motion_lengths2[motion_ids]

        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time
        motion_time = torch.tensor(phase * motion_len, dtype=torch.int64, device=self._device)
        motion_time2 = torch.tensor(phase * motion_len2, dtype=torch.int64, device=self._device)
        # motion_time = torch.cat([motion_time, motion_time2])
        ids = torch.randint(0, 512, (256,), device=self._device)
        # motion_time = motion_time[ids]
        motion_time = torch.clamp(motion_time, min=torch.tensor(0, device=self._device, dtype=torch.int64),
                                  max=motion_len - 1)
        return torch.tensor(motion_time, dtype=torch.int64, device=self._device)

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times):
        """
        Get the state of the motion at the given time.

        :param motion_ids: which motion_file to sample from
        :param motion_times: times sampled by sample_time
        :return:
        """
        obs = self.obs[motion_ids * 50 + motion_times]
        # r = self.r[motion_ids * 50 + motion_times]
        # tool_pose = self.tool_pose[motion_ids * 50 + motion_times]
        # state = self.state[motion_ids * 50 + motion_times]
        # target_obs = self.target_obs[motion_ids * 50 + motion_times]
        # self.target_obs = self.target_obs.reshape(-1)
        # return obs.to("cuda:0"), tool_pose.to("cuda:0"), self.target_obs.to("cuda:0")
        return obs

    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        # self._motion_fps = []
        # self._motion_dt = []
        # self._motion_num_frames = []
        self._motion_files = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        # import tqdm
        #
        # curr_file = motion_files
        # # curr_motion = SkeletonMotion.from_file(curr_file)
        # curr_motion = np.load(curr_file[0])
        # curr_motion = curr_motion.reshape(len(curr_motion), -1)
        # print(curr_motion)
        # print(curr_motion.shape)
        # motion_fps = 20
        # curr_dt = 1.0 / motion_fps
        #
        # num_frames = len(curr_motion)
        # curr_len = 1.0 / motion_fps * (num_frames - 1)

        # self._motion_fps.append(motion_fps)
        # self._motion_dt.append(curr_dt)
        # self._motion_num_frames.append(num_frames)
        #
        # """
        # Change to velocity later
        # """
        # # curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
        # curr_dof_vels = curr_motion
        #
        # # curr_motion.dof_vels = curr_dof_vels
        #
        # # Moving motion tensors to the GPU
        # # if USE_CACHE:
        # #     curr_motion = DeviceCache(curr_motion, self._device)
        # # else:
        # curr_motion = torch.Tensor(curr_motion)
        # curr_motion.tensor = curr_motion.to(self._device)
        #
        # # curr_mot = curr_motion.detach().clone()
        # self._motions.append(curr_motion)
        # self._motion_lengths.append(len(curr_motion))
        #
        # curr_weight = motion_weights
        # self._motion_weights.append(curr_weight)
        # self._motion_files.append(curr_file)
        #
        # self._motion_lengths = torch.tensor(
        #     self._motion_lengths, device=self._device, dtype=torch.float32
        # )
        #
        # self._motion_weights = torch.tensor(
        #     self._motion_weights, dtype=torch.float32, device=self._device
        # )
        #
        # num_motions = self.num_motions()
        # total_len = self.get_total_length()
        # print(
        #     "Loaded {:d} motions with a total length of {:.3f}s.".format(
        #         num_motions, total_len
        #     )
        # )

        with tqdm.trange(num_motion_files, ncols=100, colour="green") as t_bar:
            for f in t_bar:
                curr_file = motion_files[f]
                t_bar.set_postfix_str("Loading: {:s}".format(curr_file.split("/")[-1]))
                # with open(motion_files[f], "r") as mf:
                #     curr_motion = json.load(mf)

                curr_motion = np.load(curr_file, allow_pickle=True)
                curr_motion = curr_motion[:150]

                self._motions.append(curr_motion)
                self._motion_lengths.append(len(curr_motion))

                curr_weight = motion_weights[f]
                self._motion_weights.append(curr_weight)
                self._motion_files.append(curr_file)

        self._motion_lengths = torch.tensor(
            self._motion_lengths, device=self._device, dtype=torch.float32
        )

        self._motion_weights = torch.tensor(
            self._motion_weights, dtype=torch.float32, device=self._device
        )

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

    @staticmethod
    def _fetch_motion_files(motion_file):
    #     ext = os.path.splitext(motion_file)[1]
    #     if ext == ".yaml":
    #         dir_name = os.path.dirname(motion_file)
    #         motion_files = []
    #         motion_weights = []
    #
    #         with open(os.path.join(os.getcwd(), motion_file), "r") as f:
    #             motion_config = yaml.load(f, Loader=yaml.SafeLoader)
    #
    #         motion_list = motion_config["motions"]
    #         for motion_entry in motion_list:
    #             curr_file = motion_entry["file"]
    #             curr_weight = motion_entry["weight"]
    #             assert curr_weight >= 0
    #
    #             curr_file = os.path.join(dir_name, curr_file)
    #             motion_weights.append(curr_weight)
    #             motion_files.append(curr_file)
    #     else:
        motion_files = [motion_file]
        motion_weights = [1.0]

        return motion_files, motion_weights



if __name__ == '__main__':
    demo_json_path = ("/home/ubuntu/Github/ORK/ork/utils/runs/RollingpinParticleaction_2023-12-05-21-02-49/"
                      "Rollingpin-Particle-v1_demo_data.json")
    motion_lib = MotionLib(motion_file=demo_json_path, device="cuda:0")
    motion_ids = motion_lib.sample_motions(1)
    motion_times = motion_lib.sample_time(motion_ids)
    obs, r, tool_pose = motion_lib.get_motion_state(motion_ids, motion_times)
