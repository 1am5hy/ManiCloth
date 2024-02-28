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

from src.python_code.pySim.pySim import pySim

class cloth_env:
    def __init__(self):
        """
        Environment Configs
        """
        self.fabric = dfc.FabricConfiguration()
        self.fabric.clothDimX = 7
        self.fabric.clothDimY = 8
        self.fabric.k_stiff_stretching = 3139.41
        self.fabric.k_stiff_bending = 0.57
        self.fabric.gridNumX = 15
        self.fabric.gridNumY = 18
        self.fabric.density = 0.3
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
        self.scene.camPos = np.array([-2, 16.40, -20])
        # self.scene.sockLegOrientation = np.array([0., 0., 0.])
        self.scene.camFocusPointType = dfc.CameraFocusPointType.POINT
        self.scene.camFocusPos = np.array([-5, 8.40, 0])
        # self.scene.sceneBbox = dfc.AABB(np.array([-7, -7, -7]), np.array([7, 7, 7]))
        self.scene.timeStep = 0.025
        self.scene.forwardConvergenceThresh = 1e-6
        self.scene.backwardConvergenceThresh = 5e-6
        self.scene.name = "Test scene"
        self.scene.stepNum = 150

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


        self.sim.gradientClippingThreshold, self.sim.gradientClipping = 100.0, False

        np.set_printoptions(precision=5)
        dfc.enableOpenMP(n_threads=10)

        self.helper = dfc.makeOptimizeHelper("inverse_design")
        self.helper.taskInfo.dL_dx0 = True
        self.helper.taskInfo.dL_density = False
        self.reset_clock = 0
        self.render_clock = 0
        state_info_init = self.sim.getStateInfo()
        self.action_space = Box(0, 100, (1,), dtype=int)
        # self.observation_space = Box(-np.inf, np.inf, (30,))
        self.observation_space = Box(-np.inf, np.inf, (270,))
        self.interest1 = np.array(
            [0, 1, 2, 21, 22, 23, 42, 43, 44, 189, 190, 191, 360, 361, 362, 399, 400, 401, 519, 520, 521, 765, 766, 767,
             786, 787, 788, 807, 808, 809])
        self.interest2 = np.array([0, 1, 2, 42, 43, 44])
        self.render = False
        self.loss = 0
        self.action_save = []
        # self.similiarity = []
        # self.loss = []
        # self.example = np.zeros([150, 30])

    def reset(self):
        # print("reset")
        observe = self.sim.getStateInfo().x
        # obs = observe[self.interest1]
        obs = self.get_obs()
        # obs = self.sim.getStateInfo().x

        # self.loss = []

        # if self.reset_clock != 0:
        #     self.action_save = np.array(self.action_save)
        #     np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/exp_txt_files/hang_inference_test.npy", self.action_save)
        # #     print(self.action_save.shape)
        #     print(self.example.shape)
        #     print(self.example)
        #     np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task_obs_30.npy", self.example)

        self.reset_clock = 0
        self.sim.resetSystem()
        self.scene.customAttachmentVertexIdx = [(0, [0, 14])]
        self.sim = dfc.makeSimFromConf(self.scene)
        self.sim_mod = pySim(self.sim, self.helper, True)

        self.paramInfo = dfc.ParamInfo()
        x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/x_init_table_task.npy')
        self.paramInfo.x0 = x.flatten()
        # self.paramInfo.v0 = np.zeros_like(x).flatten()\

        self.sim.resetSystemWithParams(self.helper.taskInfo, self.paramInfo)
        self.sim.gradientClipping = True
        self.sim.gradientClippingThreshold = 100.0
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
        obs = observe.flatten()[self.interest1]
        table_pos = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/table_low_pos.npy")
        # # print(bar_pos)
        # print(obs.shape)
        obs = table_pos - obs
        # obs = self.sim.getStateInfo().x
        return obs

    def get_rew(self, obs, time):
        # pose = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/ReleaseRL/np_files/table_release_pose.npy")
        #
        # y_value = []
        # for i in range(len(pose)):
        #     y_value.append(pose[i, 1])
        # y_value = np.array(y_value)
        # min_y_pose = np.min(y_value)
        # print(min_y_pose)

        obs = obs.clone().detach().numpy()

        y_value = []
        for i in range(len(obs)):
            y_value.append(obs[i, 1])
        y_value = np.array(y_value)
        min_y_obs = np.min(y_value)
        # print(min_y_obs)
        if min_y_obs < 0:
            min_y_obs = 0

        # loss = min_y_obs - min_y_pose
        loss = min_y_obs

        # Normalize time to range 0 to 1
        time = (150 - time) / 150

        rew = loss + time

        # rew = (1 / loss) + 0.001 * (150 - time)

        print("Reward")
        print(loss)
        print(time)
        print(rew)

        return rew, loss

    def step(self, action):
        """
        initialize
        """
        # To set up the action range
        action = 50 + action
        action = int(action)
        # If the cloth is release in the first half of the manipulation, set a minimum start time for the policy to start its release

        self.scene.stepNum = action
        self.sim = dfc.makeSimFromConf(self.scene)
        self.sim_mod = pySim(self.sim, self.helper, True)
        self.paramInfo = dfc.ParamInfo()
        x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/x_init_table_task.npy')
        self.paramInfo.x0 = x.flatten()
        self.sim.resetSystemWithParams(self.helper.taskInfo, self.paramInfo)

        gp = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_table_task.npy')
        # gp = gp[:action]
        self.x, self.v = self.sim_mod(torch.tensor(self.x), torch.tensor(self.v), torch.tensor(gp), torch.tensor([([1, 10, 3139.41, 0.57])]))

        # dfc.render(self.sim, renderPosPairs=True, autoExit=True)

        self.x = self.x[-1].flatten()
        self.v = self.v[-1].flatten()

        self.sim.resetSystem()
        self.scene.customAttachmentVertexIdx = [(0, [])]
        self.scene.stepNum = 200
        self.sim = dfc.makeSimFromConf(self.scene)

        self.sim_mod = pySim(self.sim, self.helper, True)
        self.paramInfo = dfc.ParamInfo()
        self.paramInfo.x0 = self.x
        self.paramInfo.v0 = self.v
        self.sim.resetSystemWithParams(self.helper.taskInfo, self.paramInfo)

        gp = np.load('/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/gp_hang_task.npy')
        self.x, self.v = self.sim_mod(torch.tensor(self.x), torch.tensor(self.v), torch.tensor(gp), torch.tensor([([1, 10, 3139.41, 0.57])]))
        self.x = self.x[-1]
        self.v = self.v[-1]

        terminated = False
        self.render = False

        observe = self.x
        obs = torch.tensor(self.x.flatten()[self.interest1])
        # print(obs)
        # print(obs.shape)
        # self.example[self.reset_clock] = observe[self.interest1]
        # obs = self.sim.getStateInfo().x
        reward, loss = self.get_rew(observe, action)
        info = {'loss': loss, 'succeed': terminated, 'reward': reward}

        self.reset_clock = self.reset_clock + 1
        if self.reset_clock == 1:
            # self.render = True
            terminated = True

        # print(self.render)
        if self.render == True:
            dfc.render(self.sim, renderPosPairs=True, autoExit=True)

        # info = {'succeed': terminated, 'reward': reward}


        return obs, reward, terminated, info

