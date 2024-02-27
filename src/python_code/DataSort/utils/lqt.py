"""
    Linear Quadratic Tracker

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""
from math import factorial
from typing import Tuple

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

import rofunc as rf
from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.config.utils import omegaconf_to_dict, get_config

import matplotlib.pyplot as plt


class LQT:
    def __init__(self, all_points, cfg: DictConfig = None):
        # self.cfg = get_config(absl_config_path='/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/utils/config/lqt.yaml',
        #                       config_name='lqt', args=None) if cfg is None else cfg
        self.cfg = get_config(config_path='../../../DiffCloth/src/python_code/DataSort/utils/config/', config_name='lqt')
        self.all_points = all_points
        self.start_point, self.via_points = self._data_process(all_points)

    def _data_process(self, data):
        if len(data[0]) == self.cfg.nbVar:
            all_points = data
        else:
            all_points = np.zeros((len(data), self.cfg.nbVar))
            all_points[:, :self.cfg.nbVarPos] = data
        start_point = all_points[0]
        via_points = all_points[1:]
        return start_point, via_points

    def get_matrices(self):
        self.cfg.nbPoints = len(self.via_points)

        # Control cost matrix
        R = np.identity((self.cfg.nbData - 1) * self.cfg.nbVarPos, dtype=np.float32) * self.cfg.rfactor

        tl = np.linspace(0, self.cfg.nbData, self.cfg.nbPoints + 1)
        tl = np.rint(tl[1:]).astype(np.int64) - 1
        idx_slices = [slice(i, i + self.cfg.nbVar, 1) for i in (tl * self.cfg.nbVar)]

        # Target
        mu = np.zeros((self.cfg.nbVar * self.cfg.nbData, 1), dtype=np.float32)
        # Task precision
        Q = np.zeros((self.cfg.nbVar * self.cfg.nbData, self.cfg.nbVar * self.cfg.nbData), dtype=np.float32)

        for i in range(len(idx_slices)):
            slice_t = idx_slices[i]
            x_t = self.via_points[i].reshape((self.cfg.nbVar, 1))
            mu[slice_t] = x_t

            Q[slice_t, slice_t] = np.diag(
                np.hstack((np.ones(self.cfg.nbVarPos), np.zeros(self.cfg.nbVar - self.cfg.nbVarPos))))
        return mu, Q, R, idx_slices, tl

    def set_dynamical_system(self):
        A1d = np.zeros((self.cfg.nbDeriv, self.cfg.nbDeriv), dtype=np.float32)
        B1d = np.zeros((self.cfg.nbDeriv, 1), dtype=np.float32)
        for i in range(self.cfg.nbDeriv):
            A1d += np.diag(np.ones(self.cfg.nbDeriv - i), i) * self.cfg.dt ** i * 1 / factorial(i)
            B1d[self.cfg.nbDeriv - i - 1] = self.cfg.dt ** (i + 1) * 1 / factorial(i + 1)

        A = np.kron(A1d, np.identity(self.cfg.nbVarPos, dtype=np.float32))
        B = np.kron(B1d, np.identity(self.cfg.nbVarPos, dtype=np.float32))

        # Build Sx and Su transfer matrices
        Su = np.zeros((self.cfg.nbVar * self.cfg.nbData, self.cfg.nbVarPos * (self.cfg.nbData - 1)))
        Sx = np.kron(np.ones((self.cfg.nbData, 1)), np.eye(self.cfg.nbVar, self.cfg.nbVar))

        M = B
        for i in range(1, self.cfg.nbData):
            Sx[i * self.cfg.nbVar:self.cfg.nbData * self.cfg.nbVar, :] = np.dot(
                Sx[i * self.cfg.nbVar:self.cfg.nbData * self.cfg.nbVar, :], A)
            Su[self.cfg.nbVar * i:self.cfg.nbVar * i + M.shape[0], 0:M.shape[1]] = M
            M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]
        return Su, Sx

    def get_u_x(self, mu: np.ndarray, Q: np.ndarray, R: np.ndarray, Su: np.ndarray, Sx: np.ndarray, **kwargs) -> \
            Tuple[np.ndarray, np.ndarray]:
        x0 = self.start_point.reshape((self.cfg.nbVar, 1))
        # Equ. 18
        u_hat = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ (mu - Sx @ x0)
        # x= S_x x_1 + S_u u
        x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, self.cfg.nbVar))
        return u_hat, x_hat

    def solve(self):
        beauty_print("Planning smooth trajectory via LQT", type='module')

        mu, Q, R, idx_slices, tl = self.get_matrices()
        Su, Sx = self.set_dynamical_system()
        u_hat, x_hat = self.get_u_x(mu, Q, R, Su, Sx)
        return u_hat, x_hat, mu, idx_slices

def plot_3d_uni(x_hat, muQ=None, idx_slices=None, ori=False, save=False, save_file_name=None, g_ax=None, title=None,
                legend=None, for_test=False):
    if g_ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d', fc='white')
    else:
        ax = g_ax

    if muQ is not None and idx_slices is not None:
        for slice_t in idx_slices:
            ax.scatter(muQ[slice_t][0], muQ[slice_t][1], muQ[slice_t][2], c='red', s=10)

    if not isinstance(x_hat, list):
        if len(x_hat.shape) == 2:
            x_hat = np.expand_dims(x_hat, axis=0)

    title = 'Unimanual trajectory' if title is None else title
    rf.visualab.traj_plot(x_hat, legend=legend, title=title, mode='3d', ori=ori, g_ax=ax)

    if save:
        assert save_file_name is not None
        np.save(save_file_name, np.array(x_hat))
    if g_ax is None and not for_test:
        plt.show()