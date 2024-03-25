import numpy as np
from src.python_code.DataSort.utils import lqt as lqt


def uni_lqt():
    via_points = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_hang.npy')
    # filter_indices = [0, 1, 5, 10, 22, 36]
    # via_points = via_points[filter_indices]

    controller = lqt.LQT(via_points)
    u_hat, x_hat, mu, idx_slices = controller.solve()
    lqt.plot_3d_uni(x_hat, mu, idx_slices, ori=False, save=False, for_test=False)


if __name__ == '__main__':
    uni_lqt()