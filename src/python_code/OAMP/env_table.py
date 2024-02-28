import math
import gym
from gym.spaces import Box
import os
import yaml
import numpy as np

import time
import torch
import numpy as np
import diffcloth_py as dfc
import random

from src.python_code.pySim.pySim_table import pySim

class cloth_env:
    def __init__(self):
        """
        Environment Configs
        """
        self.fabric = dfc.FabricConfiguration()
        self.fabric.clothDimX = 7
        self.fabric.clothDimY = 8
        self.fabric.k_stiff_stretching = 2442.31
        self.fabric.k_stiff_bending = 0.1
        self.fabric.gridNumX = 15
        self.fabric.gridNumY = 18
        self.fabric.density = 0.27
        self.fabric.keepOriginalScalePoint = True
        self.fabric.isModel = False
        self.fabric.custominitPos = False
        # self.fabric.initPosFile = "/home/ubuntu/Github/DiffCloth/src/python_code/x_init.txt"
        # self.fabric.initPosFile = "/../../python_code/x_init_lift.txt"
        self.fabric.fabricIdx = 0
        self.fabric.color = np.array([0.9, 0., 0.1])
        self.fabric.name = "test"

        self.scene = dfc.SceneConfiguration()
        self.scene.fabric = self.fabric
        self.scene.orientation = dfc.Orientation.CUSTOM_ORIENTATION
        self.scene.upVector = np.array([0, 1, 0])
        self.scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
        self.scene.customAttachmentVertexIdx = [(0.0, [0, 14])]
        # self.scene.customAttachmentVertexIdx = [(0.0, [1838])]
        self.scene.trajectory = dfc.TrajectoryConfigs.PER_STEP_TRAJECTORY
        self.scene.primitiveConfig = dfc.PrimitiveConfiguration.TABLE_LOW
        self.scene.sockLegOrientation = np.array([0., 0., 0.])
        self.scene.windConfig = dfc.WindConfig.NO_WIND
        # self.scene.camPos = np.array([-21.67, 12.40, -10.67])
        self.scene.camPos = np.array([-2, 10.40, -20])
        # self.scene.sockLegOrientation = np.array([0., 0., 0.])
        self.scene.camFocusPointType = dfc.CameraFocusPointType.POINT
        self.scene.camFocusPos = np.array([-5, 8.40, 0])
        # self.scene.sceneBbox = dfc.AABB(np.array([-7, -7, -7]), np.array([7, 7, 7]))
        self.scene.timeStep = 0.025
        self.scene.forwardConvergenceThresh = 1e-6
        self.scene.backwardConvergenceThresh = 5e-6
        self.scene.name = "Test scene"
        self.scene.stepNum = 1

        self.sim = dfc.makeSimFromConf(self.scene)
        self.sim.forwardConvergenceThreshold = 1e-6
        self.sim.backwardConvergenceThreshold = 1e-6
        # self.sim.resetSystemWithParams(self.x)
        self.paramInfo = dfc.ParamInfo()
        self.paramInfo.set_k_pertype(1, 10, self.fabric.k_stiff_stretching, self.fabric.k_stiff_bending)

        self.x = self.sim.getStateInfo().x
        self.v = self.sim.getStateInfo().v
        self.gp1_id = np.array([0])
        self.gp2_id = np.array([14])
        self.gp1_loc = self.x[0:3]
        self.gp2_loc = self.x[42:45]


        self.sim.gradientClippingThreshold, self.sim.gradientClipping = 100.0, True

        np.set_printoptions(precision=5)
        dfc.enableOpenMP(n_threads=10)

        self.helper = dfc.makeOptimizeHelper("inverse_design")
        self.helper.taskInfo.dL_dx0 = True
        self.helper.taskInfo.dL_density = False
        self.reset_clock = 0
        self.render_clock = 0
        state_info_init = self.sim.getStateInfo()
        self.action_space = Box(-1, 1, (6,))
        # self.observation_space = Box(-np.inf, np.inf, (30,))
        self.observation_space = Box(-np.inf, np.inf, (30,))
        # self.interest1 = np.array(
        #     [0, 1, 2, 21, 22, 23, 42, 43, 44, 330, 331, 332, 405, 406, 407, 489, 490, 491, 654, 655, 656, 900, 901, 902,
        #      921, 922, 923, 942, 943, 944])
        self.interest1 = np.array(
                [0, 1, 2, 21, 22, 23, 42, 43, 44, 189, 190, 191, 360, 361, 362, 399, 400, 401, 519, 520, 521, 765, 766, 767,
                 786, 787, 788, 807, 808, 809])
        # self.interest1 = np.array([0, 1, 2, 21, 22, 23, 42, 43, 44, 156, 157, 158, 315, 316, 317, 357, 358, 359, 465, 466, 467, 630, 631, 632, 651, 652, 653, 672, 673, 674])
        self.interest2 = np.array([0, 1, 2, 42, 43, 44])
        self.render = False
        self.loss = 0

        # self.similiarity = []
        # self.loss = []
        self.action_save = []
        self.weight = 0.5
        self.example = np.zeros([150, 30])

    def reset(self):
        print("reset")
        observe = self.sim.getStateInfo().x
        # obs = observe[self.interest1]
        obs = self.get_obs()
        # obs = self.sim.getStateInfo().x

        # print(sum(self.loss)/55)
        # self.loss = []
        #
        if self.reset_clock != 0:
             self.action_save = np.array(self.action_save)
             np.save("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/exp_txt_files/table_task_inference_28_2.npy", self.action_save)
             print(self.action_save.shape)
        #     print(self.example.shape)
        #     print(self.example)
        #     np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_table_task_obs_30.npy", self.example)

        self.reset_clock = 0
        self.sim = dfc.makeSimFromConf(self.scene)
        self.sim_mod = pySim(self.sim, self.helper, True)

        self.paramInfo = dfc.ParamInfo()
        x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/x_init_table_task.npy')
        self.paramInfo.x0 = x.flatten()
        # self.paramInfo.v0 = np.zeros_like(x).flatten()\

        self.sim.resetSystemWithParams(self.helper.taskInfo, self.paramInfo)
        self.sim.gradientClipping = False
        self.sim.forwardConvergenceThreshold = 1e-6
        self.sim.backwardConvergenceThreshold = 5e-6
        new_x = self.sim.getStateInfo().x
        self.x = self.sim.getStateInfo().x
        self.v = self.sim.getStateInfo().v
        self.gp1_loc = new_x[0:3]
        self.gp2_loc = new_x[42:45]

        self.loss = 0

        return obs

    def get_obs(self):
        observe = self.sim.getStateInfo().x
        obs = observe[self.interest1]
        bar_pos = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/table_low_pos.npy")
        # print(bar_pos)
        obs = bar_pos - obs
        # obs = self.sim.getStateInfo().x
        return obs

    def get_rew(self, obs):
        traj = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_table_task_obs_30.npy", allow_pickle=True)

        # print(traj[self.reset_clock])
        # Include IOU and include the distance between AMP and current
        loss = 0
        rew = 0

        if self.reset_clock > 110:
            obs = obs.clone().detach().numpy()
            loss = np.linalg.norm(traj[self.reset_clock] - obs[self.interest1])
            rew = 1/loss

        if 85 > self.reset_clock > 75:
            obs = obs.clone().detach().numpy()
            loss = np.linalg.norm(traj[self.reset_clock] - obs[self.interest1])
            rew = 1/loss

        return rew, loss

    def step(self, action):
        """
        initialize
        """
        action = action * 0.5

        gp_loc = torch.tensor([*self.gp1_loc, *self.gp2_loc])
        new_loc = gp_loc + action

        self.action_save.append(new_loc.clone().detach().numpy())

        # gp = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_table_task.npy')
        # self.x, self.v = self.sim_mod(torch.tensor(self.x), torch.tensor(self.v), torch.tensor(gp[self.reset_clock]))
        #
        self.x, self.v = self.sim_mod(torch.tensor(self.x), torch.tensor(self.v), new_loc)
        self.gp1_loc = self.x[0:3]
        self.gp2_loc = self.x[42:45]

        terminated = False
        self.render = False

        observe = self.x
        obs = self.get_obs()
        # self.example[self.reset_clock] = observe[self.interest1]

        reward, loss = self.get_rew(observe)
        info = {'loss': loss, 'succeed': terminated, 'reward': reward}
        #
        self.reset_clock = self.reset_clock + 1
        if self.reset_clock == 150:
            self.render = True
            terminated = True

        if self.render == True:
            dfc.render(self.sim, renderPosPairs=True, autoExit=True)

        return obs, reward, terminated, info

