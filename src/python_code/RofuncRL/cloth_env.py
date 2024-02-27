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

class ClothEnv(gym.Env):
    def __init__(self):
        """
        Environment Configs
        """
        self.loss = 0
        self.loss2 = 0
        self.loss3 = 0
        self.fabric = dfc.FabricConfiguration()
        self.fabric.clothDimX = 4
        self.fabric.clothDimY = 4
        self.fabric.k_stiff_stretching = 500
        self.fabric.k_stiff_bending = 0.025
        self.fabric.gridNumX = 25
        self.fabric.gridNumY = 25
        self.fabric.density = 0.24
        self.fabric.keepOriginalScalePoint = True
        self.fabric.isModel = False
        self.fabric.custominitPos = False
        self.fabric.initPosFile = ""
        self.fabric.fabricIdx = 0
        self.fabric.color = np.array([0.9, 0., 0.1])
        self.fabric.name = "dim6x6-grid25x25-dens0.32-k50"

        self.scene = dfc.SceneConfiguration()
        self.scene.fabric = self.fabric
        self.scene.orientation = dfc.Orientation.CUSTOM_ORIENTATION
        self.scene.upVector = np.array([0, 1, 0])
        self.scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
        self.scene.customAttachmentVertexIdx = [(0.0, [0, 24])]
        # self.scene.customAttachmentVertexIdx = [(0.0, [1838])]
        self.scene.trajectory = dfc.TrajectoryConfigs.PER_STEP_TRAJECTORY
        self.scene.primitiveConfig = dfc.PrimitiveConfiguration.PLANE_BUST_WEARHAT
        self.scene.windConfig = dfc.WindConfig.NO_WIND
        self.scene.camPos = np.array([-21.67, 5.40, -10.67])
        self.scene.sockLegOrientation = np.array([0., 0., 0.])
        self.scene.camFocusPointType = dfc.CameraFocusPointType.CLOTH_CENTER
        self.scene.sceneBbox = dfc.AABB(np.array([-7, -7, -7]), np.array([7, 7, 7]))
        self.scene.timeStep = 0.1
        self.scene.forwardConvergenceThresh = 1e-3
        self.scene.backwardConvergenceThresh = 1e-3
        self.scene.name = "Test scene"

        self.sim = dfc.makeSimFromConf(self.scene)
        # self.old_obs = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/x_np.npy')
        self.old_obs = np.array([np.load("/home/ubuntu/Github/DiffCloth/src/python_code/x_init_dyndemo.npy"), np.load("/home/ubuntu/Github/DiffCloth/src/python_code/v_np.npy")])
        # self.old_obs[1] = np.load("v_np.npy")
        self.gp1_id = np.array([0])
        self.gp2_id = np.array([24])

        self.state_info_init = self.sim.getStateInfo()
        self.pts = self.state_info_init.x.reshape(-1, 3)
        self.grasp_point1 = self.pts[self.gp1_id].copy()
        self.grasp_point2 = self.pts[self.gp2_id].copy()
        self.restart = 0
        self.inference = 0
        self.visual_step = 1
        self.stage1 = False
        self.stage2 = False
        self.stage3 = False
        self.indicate1 = 0
        self.indicate2 = 0
        self.indicate3 = 0
        self.steps = 0
        self.target_points = np.zeros(6)

        self.sim.gradientClippingThreshold, self.gradientClipping = 100.0, False
        np.set_printoptions(precision=1)
        dfc.enableOpenMP(n_threads=50)

        self.helper = dfc.makeOptimizeHelper("wear_hat")
        self.helper.taskInfo.dL_dx0 = False
        self.helper.taskInfo.dL_density = False
        # self.helper
        self.sim_mod = pySim(self.sim, self.helper, True)
        self.reset_clock = 0

        state_info_init = self.sim.getStateInfo()
        self.action_space = Box(-1, 1, (6,))

        # self.observation_space = state_info_init.x.reshape(-1, 3)
        observation_space = state_info_init.x
        self.observation_space = Box(-np.inf, np.inf, observation_space.shape)

    def reset(self):
        self.sim.resetSystem()
        self.old_obs = np.array([np.load("x_np.npy"), np.load("v_np.npy")])
        # gp1_id = np.random.randint(0, 624)
        # gp2_id = np.random.randint(0, 624)
        gp1_id = 0
        gp2_id = 24
        # print("reset")
        self.state_info_init = self.sim.getStateInfo()
        self.pts = self.state_info_init.x.reshape(-1, 3)
        self.gp1_id = np.array([gp1_id])
        self.gp2_id = np.array([gp2_id])
        self.scene.customAttachmentVertexIdx = [(0, [self.gp1_id, self.gp2_id])]
        self.grasp_point1 = self.pts[self.gp1_id].copy()
        self.grasp_point2 = self.pts[self.gp2_id].copy()
        # self.sim.resetSystem()
        self.record = np.zeros([10])

        self.restart = self.restart + 1
        self.reset_clock = 0

        return self.state_info_init.x


    def get_obs(self, t = 0):
        x = self.sim.getStateInfo().x
        v = self.sim.getStateInfo().v

        return np.concatenate((x, v))

    def get_target_points(self, action):
        vel1 = np.array(action[0:2])
        vel2 = np.array(action[2:])

        target1 = np.zeros(3)
        target2 = np.zeros(3)

        target1[0] = 0
        target2[0] = 0

        target1[1] = vel1[0] * 3
        target2[1] = target1[1] + 0.1 * vel2[0]

        target1[2] = vel1[1] * 3
        target2[2] = target1[2] + 0.1 * vel2[1]

        target_points = np.array([[*target1, *target2]])

        return target_points

    def get_rew(self, state_x):
        import matplotlib.pyplot as plt
        # Extract downsampled data
        index = np.load("index.npy")

        x_all = np.zeros([25, 3])

        for step in range(25):
            ind = index[step]
            x_all[step] = state_x[int(ind)]

        # Goal State
        g = np.load("goal_inv.npy")

        xgbar = np.zeros(25)
        xbar = np.zeros(25)

        # barl1 = 0
        # barl2 = 0
        # barl3 = 0
        # # To Bar error
        # for i in range(19, 25):
            # barl1 = barl1 + 1/2 * ((2 - x_all[i][1])**2 + (1 - x_all[i][2]**2))
            # barl2 = barl2 + 1/2 * ((1 - x_all[i][1])**2 + (1.5 - x_all[i][2]**2))
            # barl3 = barl3 + (g[i][:] - x_all[i][:])
        barl1 = 1/2 * (2.25 - x_all[19][1])**2
        barl2 = 1/2 * (1.25 - x_all[19][2])**2

        barl3 = 1/2 * (2.25 - x_all[24][1])**2
        barl4 = 1/2 * (1.25 - x_all[24][2])**2

        mid1 = 1/2 * (2.5 - x_all[12][1])**2
        mid2 = 1/2 * (1 - x_all[12][2])**2

        bar1l = barl1 + barl2 + barl3 + barl4

        barl5 = 1/2 * (1.25 - x_all[19][1])**2
        barl6 = 1/2 * (1.25 - x_all[19][2])**2

        barl7 = 1/2 * (1.25 - x_all[24][1])**2
        barl8 = 1/2 * (1.25 - x_all[24][2])**2

        bar2l = barl5 + barl6 + barl7 + barl8

        # Connectivity Error
        y_gcon = np.zeros(20)
        z_gcon = np.zeros(20)
        for i in range(0, 19):
            y_gcon[i] = g[i + 5][1] - g[i][1]
            z_gcon[i] = g[i + 5][2] - g[i][2]

        y_xcon = np.zeros(20)
        z_xcon = np.zeros(20)
        for i in range(0, 19):
            y_xcon[i] = x_all[i + 5][1] - x_all[i][1]
            z_xcon[i] = x_all[i + 5][2] - x_all[i][2]

        y_con = abs(y_gcon - y_xcon)
        z_con = abs(z_gcon - z_xcon)

        # stage 1 reward
        loss = bar1l + mid1 + mid2

        if self.reset_clock == 0:
            reward = -1 * (self.loss - loss)

        else:
            reward = (self.loss - loss) * (10 - self.reset_clock)

        # print(reward)
        self.loss = loss

        if barl1/4 < 0.1:
            self.stage1 = True

        if self.stage1 is True:
            loss = bar2l
            reward = (self.loss - loss) * (10 - self.reset_clock)
            self.loss = loss

            # print("stage 2")

        loss3 = loss

        return reward, loss3

    def get_grasp_traj(self, init, target, length):
        # print(init)
        # print(target)
        traj_all = 0
        for i in range(len(target)):
            if i == 0:
                traj = init + (target[i] * np.linspace(0, 1, length)[:, None])
                traj_all = traj

                for j in range(length):
                    if traj[j][2] > 0.75:
                        traj[j][2] = 0.75

                for j in range(length):
                    if traj[j][1] > 7:
                        traj[j][1] = 7

                for j in range(length):
                    if traj[j][2] < -0.75:
                        traj[j][2] = -0.75

                for j in range(length):
                    if traj[j][1] < 2:
                        traj[j][1] = 2
            else:
                traj = traj_all[-1] + (target[i] * np.linspace(0, 1, length)[:, None])

                for j in range(length):
                    if traj[j][2] > 0.75:
                        traj[j][2] = 0.75

                for j in range(length):
                    if traj[j][1] > 6:
                        traj[j][1] = 6

                for j in range(length):
                    if traj[j][2] < -0.5:
                        traj[j][2] = -0.5

                for j in range(length):
                    if traj[j][1] < 2:
                        traj[j][1] = 2

                traj_all = np.vstack([traj_all, traj])

        # print(traj_all)

        for i in range(150):
            if i == 0:
                null_traj = traj_all[-1]
            else:
                null_traj = np.vstack([null_traj, traj_all[-1]])

        traj_all = np.vstack([traj_all, null_traj])

        return torch.tensor(traj_all)

    def get_new_traj(self, init, target, length):
        traj = np.zeros([length, 3])
        traj[:15] = init + (target * np.linspace(0, 1, int(length / 10))[:, None])  #
        traj[15:] = init + target
        return torch.tensor(traj)

    def step(self, action):
        """
        initialize
        """
        num_step = 150

        if self.steps == 0:
            self.target_points = self.get_target_points(action)
        else:
            self.target_points = np.vstack([self.target_points, self.get_target_points(action)])

        # print(self.target_points)
        self.old_obs = np.array([np.load("x_np.npy"), np.load("v_np.npy")])

        x = torch.tensor(self.old_obs[0])
        v = torch.tensor(self.old_obs[1])

        # print(self.target_points)

        x_list, v_list = [x.reshape(-1, 3)], [v.reshape(-1, 3)]

        # if self.restart == self.restart + 1:
        #     # self.paramInfo = dfc.ParamInfo()
        #     # self.paramInfo.x0 = x.flatten().detach().cpu().numpy()
        #
        #     self.sim.resetSystem()
        #     self.sim_mod = pySim(self.sim, self.helper, True)
        #
        # elif self.restart == 0:
        #     # self.paramInfo = dfc.ParamInfo()
        #     # self.paramInfo.x0 = x.flatten().detach().cpu().numpy()

        self.sim.resetSystem()
        self.sim_mod = pySim(self.sim, self.helper, True)

        self.restart = 1

        # print(self.target_points[:, 0:3])
        grasp1_traj = self.get_grasp_traj(self.grasp_point1, self.target_points[:, 0:3], 10)
        grasp2_traj = self.get_grasp_traj(self.grasp_point2, self.target_points[:, 3:], 10)


        pos = torch.cat((grasp1_traj, grasp2_traj), dim=1)

        final_pos = np.array(pos[0])
        self.grasp_point1 = np.array(final_pos[0:3])
        self.grasp_point2 = np.array(final_pos[3:])

        for step in range(num_step + (self.steps * 10)):
            p = pos[step]
            x, v = self.sim_mod(x, v, p)
            x_list.append(x.reshape(-1, 3))
            v_list.append(v.reshape(-1, 3))

        self.steps = self.steps + 1

        render = True
        #
        if self.inference == 9:
            render = True

        self.inference = self.inference + 1

        if render:
            dfc.render(self.sim, renderPosPairs=True, autoExit=True)

        # self.old_obs = np.array([np.array(x), np.array(v)])

        xl = np.array(self.sim.getStateInfo().x).reshape(-1, 3)
        obs = np.array(self.sim.getStateInfo().x).reshape(-1)
        # print(self.sim.perStepGradient)
        # print(xl)

        # Reward Function

        reward, loss = self.get_rew(xl)
        # print(reward)

        terminated = loss < 0.01

        info = {'succeed': terminated, 'reward': reward}

        truncated = False

        self.reset_clock = self.reset_clock + 1

        if self.reset_clock == 10:
            self.stage1 = False
            self.stage2 = False
            self.stage3 = False
            self.steps = 0
            truncated = True

        return obs, reward, terminated, truncated, info

if __name__ == '__main__':
    cenv = ClothEnv()

    start = time.time()

    for i in range(10):
        if i == 0:
            obs, reward, terminated, truncated, info = cenv.step(np.array([1, 0, 0, 0]))
        if i == 1:
            obs, reward, terminated, truncated, info = cenv.step(np.array([0, 1, 0, 0]))
        if i == 2:
            obs, reward, terminated, truncated, info = cenv.step(np.array([-0.15, 0, 0, 0]))
        if i == 3:
            obs, reward, terminated, truncated, info = cenv.step(np.array([-0.15, 0, -0, 0]))
        if i == 4:
            obs, reward, terminated, truncated, info = cenv.step(np.array([-0.15, 0, -0, 0]))
        if i == 5:
            obs, reward, terminated, truncated, info = cenv.step(np.array([-0.15, 0, -0, 0]))
        if i == 6:
            obs, reward, terminated, truncated, info = cenv.step(np.array([-0.15, 0, -0, 0]))
        else:
            obs, reward, terminated, truncated, info = cenv.step(np.array([0, 0, 0, 0, 0, 0]))
    # obs, reward, terminated, truncated, info = cenv.step(np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0]))
    cenv.reset()