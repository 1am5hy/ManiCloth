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
import time

import torch
import yaml
from isaacgym.torch_utils import *

import rofunc as rf
from rofunc.learning.RofuncRL.tasks.utils import torch_jit_utils as torch_utils
from rofunc.utils.datalab.poselib.poselib.core.rotation3d import *
from rofunc.utils.datalab.poselib.poselib.skeleton.skeleton3d import SkeletonMotion

USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy


    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)


    torch.Tensor.numpy = Patch.numpy


class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                # print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1

        # print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLib:
    def __init__(self, motion_file, device):
        """

        Args:
            motion_file:
            dof_body_ids:
            dof_offsets:
            key_body_ids:
            device:
        """

        self._device = device

        self._object_poses = torch.zeros((), device=device)

        self._load_motions(motion_file)

        motions = self._motions
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(
            len(self._motions), dtype=torch.long, device=self._device
        )

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        # motion_id = 0
        return self._motions[motion_id]

    def sample_motions(self):
        motion_ids = torch.multinomial(
            self._motion_weights, replacement=True
        )

        # m = self.num_motions()
        # motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)
        # motion_ids = torch.tensor(motion_ids, device=self._device, dtype=torch.long)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)

        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self):
        return len(self._motions)

    # def get_object_pose(self, frame_id):
    #     """Return object pose at frame=id, where id is recorded in motion_ids
    #
    #     Args:
    #         frame_id [list]: Same frame id * num_envs, where num_envs is the environment number.
    #
    #     Returns:
    #
    #     """
    #     if self._object_poses.ndim == 0:
    #         return None
    #     object_pose = self._object_poses[frame_id][0]
    #     return object_pose

    def get_motion_state(self):
        """

        Args:
            motion_ids:
            motion_times:

        Returns:

        """
        motion_len = self._motion_lengths
        num_frames = self._motion_num_frames
        dt = self._motion_dt

        return self._motions

    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        total_len = 0.0

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        import tqdm



        curr_file = motion_files
        # curr_motion = SkeletonMotion.from_file(curr_file)
        curr_motion = np.load(curr_file)
        motion_fps = 20
        curr_dt = 1.0 / motion_fps

        num_frames = len(curr_motion)
        curr_len = 1.0 / motion_fps * (num_frames - 1)

        self._motion_fps.append(motion_fps)
        self._motion_dt.append(curr_dt)
        self._motion_num_frames.append(num_frames)

        """
        Change to velocity later
        """
        # curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
        curr_dof_vels = curr_motion

        # curr_motion.dof_vels = curr_dof_vels

        # Moving motion tensors to the GPU
        # if USE_CACHE:
        #     curr_motion = DeviceCache(curr_motion, self._device)
        # else:
        curr_motion = torch.Tensor(curr_motion)
        curr_motion.tensor = curr_motion.to(self._device)

        # curr_mot = curr_motion.detach().clone()
        self._motions.append(curr_motion)
        self._motion_lengths.append(curr_len)

        curr_weight = motion_weights
        self._motion_weights.append(curr_weight)
        self._motion_files.append(curr_file)


        self._motion_lengths = torch.tensor(
            self._motion_lengths, device=self._device, dtype=torch.float32
        )

        self._motion_weights = torch.tensor(
            self._motion_weights, dtype=torch.float32, device=self._device
        )
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(
            self._motion_fps, device=self._device, dtype=torch.float32
        )
        self._motion_dt = torch.tensor(
            self._motion_dt, device=self._device, dtype=torch.float32
        )
        self._motion_num_frames = torch.tensor(
            self._motion_num_frames, device=self._device
        )

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print(
            "Loaded {:d} motions with a total length of {:.3f}s.".format(
                num_motions, total_len
            )
        )

    @staticmethod
    def _fetch_motion_files(motion_file):
        # ext = os.path.splitext(motion_file)[1]
        # if ext == ".yaml":
        #     dir_name = os.path.dirname(motion_file)
        #     motion_files = []
        #     motion_weights = []
        #
        #     with open(os.path.join(os.getcwd(), motion_file), "r") as f:
        #         motion_config = yaml.load(f, Loader=yaml.SafeLoader)
        #
        #     motion_list = motion_config["motions"]
        #     for motion_entry in motion_list:
        #         curr_file = motion_entry["file"]
        #         curr_weight = motion_entry["weight"]
        #         assert curr_weight >= 0
        #
        #         curr_file = os.path.join(dir_name, curr_file)
        #         motion_weights.append(curr_weight)
        #         motion_files.append(curr_file)
        # else:
        motion_files = motion_file
        motion_weights = [1.0]

        return motion_files, motion_weights

    @property
    def motions(self):
        return self._motions


class ObjectMotionLib:
    def __init__(self, object_motion_file, device, humanoid_start_time=None):
        self.object_motion_file = object_motion_file
        self.object_poses_w_time = []
        self.device = device
        # self.humanoid_start_time = humanoid_start_time  # in second, for the motion sync
        # if self.humanoid_start_time is not None:
        #     assert len(self.humanoid_start_time) == len(self.object_motion_file)

        self._load_motions()

    def _load_motions(self):
        # objs_list, meta_list = rf.optitrack.get_objects(self.object_motion_file)
        # self.scales = self._get_scale(meta_list)
        # self.dts = self._get_dt(meta_list)
        self.dts = 1 / 20

        self.tds = [0]
        self.object_poses_w_time = np.load(self.object_motion_file, allow_pickle=True)
        # for i in range(self.num_motions):
        #     # data is a numpy array of shape (n_samples, n_features)
        #     # labels is a list of strings corresponding to the name of the features
        #     data, labels = rf.optitrack.data_clean(self.object_motion_file, legacy=False, objs=objs_list[i])[i]
        #
        #     # Accessing the position and attitude of an object over all samples:
        #     # Coordinates names and order: ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        #     object_poses_dict = {}
        #     for object_name in self.object_names:
        #         data_ptr = labels.index(f'{object_name}.pose.x')
        #         assert data_ptr + 6 == labels.index(f'{object_name}.pose.qw')
        #         pose = data[:, data_ptr:data_ptr + 7]
        #         pose[:, :3] *= self.scales[i]  # convert to meter
        #         pose = self._motion_transform(torch.tensor(pose, dtype=torch.float))
        #
        #         pose_w_time = torch.hstack((torch.tensor(data[:, 1], dtype=torch.float).unsqueeze(-1), pose))
        #         object_poses_dict[object_name] = pose_w_time.to(self.device)
        #     self.object_poses_w_time.append(object_poses_dict)  # [num_motions, num_objects, num_samples, 7]


    def get_motion_state(self):
        """

        :param motion_ids:
        :param motion_times:
        :return:
        """

        return self.object_poses_w_time