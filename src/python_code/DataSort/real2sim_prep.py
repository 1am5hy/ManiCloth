import numpy as np
from src.python_code.DataSort.utils.rearrange_idx import rearrange
from src.python_code.DataSort.utils.pt_idx import pt_indentify_hang, pt_indentify_table
from src.python_code.DataSort.utils.downsample import downsample

traj = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task.npy")

"""Downsample"""
traj_3, traj_30 = downsample(traj)

"""Rearrange to marker pose to match simulation particle index"""
# Traj input must be a nx30 array
traj_30 = rearrange(traj_30)

"""Extract grasping trajectory"""
gp = np.hstack([traj_30[:, :3], traj_30[:, 6:9]])

"""Extract particle index"""
pt_indentify_hang(traj_30)
# pt_indentify_table(traj)

