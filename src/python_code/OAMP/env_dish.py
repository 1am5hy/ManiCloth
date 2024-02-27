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
from src.python_code.OAMP.utils.gp_2_quat import edge_to_quat, quat_to_edge

from src.python_code.pySim.pySim_dish_OAMP import pySim

class cloth_env:
    def __init__(self):
        """
        Environment Configs
        """
        self.fabric = dfc.FabricConfiguration()
        self.fabric.clothDimX = 3
        self.fabric.clothDimY = 3
        self.fabric.k_stiff_stretching = 3738.24
        self.fabric.k_stiff_bending = 2.88
        self.fabric.gridNumX = 15
        self.fabric.gridNumY = 15
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
        self.scene.customAttachmentVertexIdx = [(0.0, [80, 82, 110, 112])]
        # self.scene.customAttachmentVertexIdx = [(0.0, [1838])]
        self.scene.trajectory = dfc.TrajectoryConfigs.PER_STEP_TRAJECTORY
        self.scene.primitiveConfig = dfc.PrimitiveConfiguration.DISH
        self.scene.sockLegOrientation = np.array([0., 0., 0.])
        self.scene.windConfig = dfc.WindConfig.NO_WIND
        self.scene.camPos = np.array([-21.67, 12.40, -10.67])
        # self.scene.sockLegOrientation = np.array([0., 0., 0.])
        self.scene.camFocusPointType = dfc.CameraFocusPointType.CLOTH_CENTER
        # self.scene.sceneBbox = dfc.AABB(np.array([-7, -7, -7]), np.array([7, 7, 7]))
        self.scene.timeStep = 0.025
        self.scene.forwardConvergenceThresh = 1e-10
        self.scene.backwardConvergenceThresh = 5e-10
        self.scene.name = "Test scene"
        self.scene.stepNum = 1

        self.sim = dfc.makeSimFromConf(self.scene)
        self.sim.forwardConvergenceThreshold = 1e-8
        self.sim.backwardConvergenceThreshold = 1e-8
        # self.sim.resetSystemWithParams(self.x)
        self.paramInfo = dfc.ParamInfo()
        self.paramInfo.set_k_pertype(1, 10, self.fabric.k_stiff_stretching, self.fabric.k_stiff_bending)

        self.x = self.sim.getStateInfo().x
        self.v = self.sim.getStateInfo().v
        # self.gp1_id = np.array([0])
        # self.gp2_id = np.array([14])
        self.gp1_loc = self.x[240:243]
        self.gp2_loc = self.x[246:249]
        self.gp3_loc = self.x[330:333]
        self.gp4_loc = self.x[336:339]
        p = np.array([*self.gp1_loc, *self.gp2_loc, *self.gp3_loc, *self.gp4_loc])
        pose, self.w1, self.w2 = edge_to_quat(p)

        self.sim.gradientClippingThreshold, self.sim.gradientClipping = 100.0, False

        np.set_printoptions(precision=5)
        dfc.enableOpenMP(n_threads=10)

        self.helper = dfc.makeOptimizeHelper("inverse_design")
        self.helper.taskInfo.dL_dx0 = True
        self.helper.taskInfo.dL_density = False
        self.reset_clock = 0
        self.render_clock = 0
        state_info_init = self.sim.getStateInfo()
        self.action_space = Box(-1, 1, (7,))
        self.observation_space = Box(-np.inf, np.inf, (24,))
        # self.observation_space = Box(-np.inf, np.inf, (6,))
        self.interest1 = np.array([0, 1, 2, 42, 43, 44, 240, 241, 242, 246, 247, 248, 330, 331, 332, 336, 337, 338, 630, 631, 632, 672, 673, 674])
        # self.interest2 = np.array([0, 1, 2, 42, 43, 44])
        self.render = False
        self.loss = 0
        self.action_save = []
        self.last_x = np.zeros(675)
        # self.similiarity = []
        # self.loss = []
        self.example = np.zeros([100, 675])

    def reset(self):
        print("reset")
        observe = self.sim.getStateInfo().x
        # obs = observe[self.interest1]
        obs = self.get_obs()
        # obs = self.sim.getStateInfo().x

        # print(sum(self.loss)/55)
        # self.loss = []

        if self.reset_clock != 0:
            print(self.example.shape)
            print(self.example)
        #     # print(self.example[-1])
        #     # np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/obs_hang.npy", self.example)
        #     actions = np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/actions_hang.npy", self.action_save)

        self.reset_clock = 0
        self.scene.customAttachmentVertexIdx = [(0.0, [80, 82, 110, 112])]
        self.sim = dfc.makeSimFromConf(self.scene)
        self.sim_mod = pySim(self.sim, self.helper, True)

        self.paramInfo = dfc.ParamInfo()
        x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/x_init_dish.npy')
        self.paramInfo.x0 = x.flatten()
        # self.paramInfo.v0 = np.zeros_like(x).flatten()\

        self.sim.resetSystemWithParams(self.helper.taskInfo, self.paramInfo)
        self.sim.gradientClipping = False
        self.sim.forwardConvergenceThreshold = 1e-10
        self.sim.backwardConvergenceThreshold = 5e-10
        new_x = self.sim.getStateInfo().x
        self.x = self.sim.getStateInfo().x
        self.v = self.sim.getStateInfo().v
        self.gp1_loc = self.x[240:243]
        self.gp2_loc = self.x[246:249]
        self.gp3_loc = self.x[330:333]
        self.gp4_loc = self.x[336:339]
        self.loss = 0
        self.example = np.zeros([100, 675])

        return obs

    def get_obs(self):
        observe = self.sim.getStateInfo().x
        obs = observe[self.interest1]
        # bar_pos = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/bar_pos.npy")
        # obs = bar_pos - obs
        # obs = self.sim.getStateInfo().x
        return obs

    def get_rew(self, obs):
        traj = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/obs_hang.npy", allow_pickle=True)

        # Include IOU and include the distance between AMP and current
        loss = 1
        rew = 1

        if self.reset_clock > 130:
            obs = obs.clone().detach().numpy()
            loss = np.linalg.norm(traj[-1][self.interest1] - obs[self.interest1])
            rew = 1/loss

        if 90 > self.reset_clock > 80:
            obs = obs.clone().detach().numpy()
            loss = np.linalg.norm(traj[self.reset_clock][self.interest1] - obs[self.interest1])
            rew = 1/loss

        return rew, loss

    def step(self, action):
        """
        initialize
        """

        action = action * 0.5
        ori_norm = np.linalg.norm(action[3:])
        action[3:] = action[3:] / ori_norm
        displacement = quat_to_edge(action, self.w1, self.w2)

        gp_loc = torch.tensor([*self.gp1_loc, *self.gp2_loc, *self.gp3_loc, *self.gp4_loc])
        new_loc = gp_loc + displacement

        # self.action_save.append(action)

        gp = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_dish_modified.npy')
        # print(gp[self.reset_clock])

        self.x, self.v = self.sim_mod(torch.tensor(self.x), torch.tensor(self.v), torch.tensor(gp[self.reset_clock]))
        self.x = self.x.reshape(-1)
        # self.x, self.v = self.sim_mod(torch.tensor(self.x), torch.tensor(self.v), new_loc)
        self.gp1_loc = self.x[240:243]
        self.gp2_loc = self.x[246:249]
        self.gp3_loc = self.x[330:333]
        self.gp4_loc = self.x[336:339]

        terminated = False
        self.render = False

        observe = self.x
        obs = self.get_obs()
        # obs = self.sim.getStateInfo().x
        # self.example[self.reset_clock] = observe
        reward, loss = self.get_rew(observe)
        info = {'loss': loss, 'succeed': terminated, 'reward': reward}

        self.reset_clock = self.reset_clock + 1
        if self.reset_clock == 100:
            self.render = True
            terminated = True
        #
        # self.render_clock = self.render_clock + 1
        # if self.render_clock == 6000:
        #     self.render = True
        #     terminated = True

        # print(self.render)
        if self.render == True:
            dfc.render(self.sim, renderPosPairs=True, autoExit=True)

        # info = {'succeed': terminated, 'reward': reward}


        return obs, reward, terminated, info

