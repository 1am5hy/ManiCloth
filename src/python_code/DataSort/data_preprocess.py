import numpy as np

"""Downsample"""

# traj = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task_3.npy")
# # print(len(traj[0][0]))
# traj_new = np.zeros((int(len(traj)/3), len(traj[0]), len(traj[0][0])))
# for i in range(int(len(traj)/3)):
#     traj_new[i] = traj[i*3]
# print(traj_new.shape)
# np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task_3.npy", traj_new)
# traj_new = traj_new.reshape(-1, len(traj[0])*len(traj[0][0]))
# print(traj_new.shape)
# np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task_30.npy", traj_new)

# print(traj_new[-1])

"""Grasp Point"""
traj = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task_3.npy")

particle2 = traj[:, 0]
particle3 = traj[:, 2]
# particle4 = traj[:, 4]
# particle5 = traj[:, 5]

gp = np.hstack([particle2, particle3])
print(gp.shape)
print(gp[0])

gp_spread = np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_hang_task.npy", gp)