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

import torch
from isaacgym.torch_utils import *

import rofunc as rf
from rofunc.learning.RofuncRL.tasks.isaacgym.base.vec_task import VecTask
from rofunc.learning.RofuncRL.tasks.utils import torch_jit_utils as torch_utils
from rofunc.utils.oslab.path import get_rofunc_path
import diffcloth_py as dfc
# from diffcloth_env import run_result
from src.python_code.pySim.pysim2 import pySim

class Cloth(VecTask):
    def __init__(self, config, rl_device, sim_device, graphics_device_id):
        self.cfg = config

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        # self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        # key_bodies = self.cfg["env"]["keyBodies"]
        # constant = 1
        self._setup_character_props()
        self._dof_obs_size = 30  # 10 * 3 = 30
        self._num_actions = 6
        self._num_obs = 30
        # print("Nani?")

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id)

        dt = self.cfg["sim"]["dt"]
        self.dt = 0.05

        # # create some wrapper tensors for different slices
        # self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # dofs_per_env = self._dof_state.shape[0] // self.num_envs
        # self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        # self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]
        #
        # self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        # self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)


        self._terminate_buf = torch.ones(1, device=self.device, dtype=torch.long)

        self.scene = self.get_default_scene()
        self.sim = dfc.makeSimFromConf(self.scene)
        self.x = torch.tensor(self.sim.getStateInfo().x)
        self.v = torch.tensor(self.sim.getStateInfo().v)

        self.sim.gradientClippingThreshold, self.sim.gradientClipping = 500.0, False
        np.set_printoptions(precision=5)

        dfc.enableOpenMP(n_threads=10)

        self.helper = dfc.makeOptimizeHelper("inverse_design")
        self.helper.taskInfo.dL_dx0 = True
        self.helper.taskInfo.dL_density = False

        self.sim.forwardConvergenceThreshold = 1e-6

        self.sim_mod = pySim(self.sim, self.helper, True)
        self.step_num = 0
        self.randomize = False
        self.gp1 = 0
        self.gp2 = 14

        return

    def get_obs_size(self):
        # print(self._num_obs)
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def create_sim(self):
        # self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.scene = self.get_default_scene()
        self.sim = dfc.makeSimFromConf(self.scene)

        # If randomizing, apply once immediately on startup before the fist sim step

        if self.randomize:
            # self.apply_randomizations(self.randomization_params)
            self.gp1 = 0
            self.gp2 = 14
            self.scene.scene.customAttachmentVertexIdx = [(0.0, [self.gp1, self.gp2])]

        return

    def reset_idx(self):
        self._reset_env_tensors()
        # self._refresh_sim_tensors()
        self._compute_observations()
        return

    def _reset_env_tensors(self):

        self.progress_buf = 0
        self.reset_buf = 0
        self._terminate_buf = 0

    def _setup_character_props(self):
        """
        dof_body_ids records the ids of the bodies that are connected to their parent bodies with joints
        The order of these ids follows the define order of the body in the MJCF. The id start from 0, and
        the body with id:0 is pelvis, which is not considered in the list.

        dof_offset's length is always len(dof_body_ids) + 1, and it always start from 0.
        Each 2 values' minus in the list represents how many dofs that corresponding body have.

        dof_observation_size is equal to dof * 6, where 6 stands for position and rotation observations, dof is the
        number of actuated dofs, it equals to the length of dof_body_ids

        num_actions is equal to the number of actuatable joints' number in the character. It does not include the
        joint connecting the character to the world.
        dof_observation_size

        num_observations is composed by 3 parts, the first observation is the height of the CoM of the character; the
        second part is the observations for all bodies. The body number is multiplied by (3 position values, 6
        orientation values, 3 linear velocity, and 3 angular velocity); finally, -3 stands for

        :param key_bodies:
        """

        self._num_actions = 6
        self._num_obs = 30


    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_cloth_reward(self.obs_buf)
        # print()
        return

    def _compute_reset(self):

        self.reset_buf[:], self._terminate_buf[:] = compute_cloth_reset(
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
        )
        return

    def _compute_observations(self):
        obs = self.sim.getStateInfo().x
        # print(type(obs))
        obs = self._compute_marker_observation(obs)
        self.obs_buf[:] = obs

        return

    def _compute_marker_observation(self, obs):

        new_obs = np.zeros(30)
        new_obs[0:3] = obs[0:3]
        new_obs[3:6] = obs[21:24]
        new_obs[6:9] = obs[42:45]
        new_obs[9:12] = obs[252:255]
        new_obs[12:15] = obs[315:318]
        new_obs[15:18] = obs[357:360]
        new_obs[18:21] = obs[468:471]
        new_obs[21:24] = obs[630:633]
        new_obs[24:27] = obs[651:654]
        new_obs[27:30] = obs[672:675]

        return torch.tensor(new_obs)

    def get_grasp_traj(self, init, target, length):

        tar = target.detach().clone().cpu().numpy()
        traj = init + (tar * np.linspace(0, 1, length)[:, None])
        traj = traj[1]
        return torch.tensor(traj)

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        self.pts = self.sim.getStateInfo().x.reshape(-1, 3)
        grasp_point1 = self.pts[self.gp1].copy()
        grasp_point2 = self.pts[self.gp2].copy()
        print(grasp_point1, grasp_point2)
        init = np.array([*grasp_point1, *grasp_point2])

        position_control = self.get_grasp_traj(init, actions, 2)
        # k = torch.tensor([1, 10000, self.scene.fabric.k_stiff_stretching, self.scene.fabric.k_stiff_bending])

        self.x, self.v = self.sim_mod(self.x, self.v, position_control)

        dfc.render(self.sim, renderPosPairs=True, autoExit=True)

        self.x = torch.tensor(self.sim.getStateInfo().x)
        self.v = torch.tensor(self.sim.getStateInfo().v)


        return

    def post_physics_step(self):
        self.progress_buf += 1
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        return

    def get_default_scene(self):
        fabric = dfc.FabricConfiguration()
        fabric.clothDimX = 3
        fabric.clothDimY = 3
        fabric.k_stiff_stretching = 2442.31
        fabric.k_stiff_bending = 0.1
        fabric.gridNumX = 15
        fabric.gridNumY = 15
        fabric.density = 0.3
        fabric.keepOriginalScalePoint = True
        fabric.isModel = False
        fabric.custominitPos = False
        fabric.initPosFile = ""
        fabric.fabricIdx = 0
        fabric.color = np.array([0.9, 0., 0.1])
        fabric.name = "test"
        # fabric.

        scene = dfc.SceneConfiguration()
        scene.fabric = fabric
        scene.orientation = dfc.Orientation.CUSTOM_ORIENTATION
        scene.upVector = np.array([0, 1, 0])
        scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
        scene.customAttachmentVertexIdx = [(0.0, [0, 14])]
        # scene.customAttachmentVertexIdx = [(0.0, [1838])]
        scene.trajectory = dfc.TrajectoryConfigs.PER_STEP_TRAJECTORY
        scene.primitiveConfig = dfc.PrimitiveConfiguration.NONE
        scene.windConfig = dfc.WindConfig.NO_WIND
        # scene.camPos = np.array([-90, 30, 60])
        scene.camPos = np.array([-12.67, 12, 13.67])
        scene.camFocusPos = np.array([0, 12, 0])
        # scene.camPos = np.array([-21.67, 15.40, -10.67])
        scene.sockLegOrientation = np.array([0., 0., 0.])
        scene.camFocusPointType = dfc.CameraFocusPointType.POINT
        # scene.sceneBbox = dfc.AABB(np.array([-70, -70, -70]), np.array([70, 70, 70]))
        # scene.sceneBbox = dfc.AABB(np.array([-7, 3, -7]), np.array([7, 17, 7]))
        scene.timeStep = 0.05
        scene.forwardConvergenceThresh = 1e-6
        scene.backwardConvergenceThresh = 5e-4
        scene.name = "Test scene"

        scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
        gp1_id = np.array([0])
        gp2_id = np.array([14])
        scene.customAttachmentVertexIdx = [(0, [0, 14])]
        # scene.stepNum = num_step
        scene.stepNum = 1

        return scene


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_cloth_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward

def compute_cloth_reset(reset_buf,
        progress_buf,
        max_episode_length,
):
    # type: (Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    print(max_episode_length)
    terminated = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    return reset, terminated
