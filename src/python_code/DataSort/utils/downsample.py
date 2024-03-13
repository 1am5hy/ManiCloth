import numpy as np

def downsample(traj):
    traj = traj.reshape(-1, 10, 3)

    traj_new = np.zeros((int(len(traj)/3), len(traj[0]), len(traj[0][0])))
    for i in range(int(len(traj)/3)):
        traj_new[i] = traj[i*3]

    traj_new_30 = traj_new.reshape(-1, len(traj[0])*len(traj[0][0]))

    return traj_new, traj_new_30