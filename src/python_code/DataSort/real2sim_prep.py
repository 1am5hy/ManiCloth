import numpy as np
from src.python_code.DataSort.utils.rearrange_idx import rearrange
from src.python_code.DataSort.utils.pt_idx import pt_indentify_hang, pt_indentify_table
from src.python_code.DataSort.utils.downsample import downsample
material = "cotton"
task = "hang"
# demo_id = 2
idx_all = []

for i in range(10):
    demo_id = i + 1
    print(demo_id)
    load_path = "/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/marker_{}_{}_demo_{}.npy".format(material, task, demo_id)
    traj = np.load(load_path)

    """Downsample"""
    traj_3, traj_30 = downsample(traj)
    # print(traj_30.shape)
    # np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task_3.npy", traj_3)
    # np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task_30.npy", traj_30)

    """Rearrange to marker pose to match simulation particle index"""
    # Traj input must be a nx30 array
    traj_30 = rearrange(traj_30)
    # print(traj_30[0])

    traj_3 = traj_30.reshape(-1, 10, 3)

    traj_30 = traj_30.reshape(-1, 30)

    traj_3_path = "/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/marker_{}_{}_demo_{}_3.npy".format(material, task, demo_id)
    traj_30_path = "/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/marker_{}_{}_demo_{}_30.npy".format(material, task, demo_id)
    np.save(traj_3_path, traj_3)
    np.save(traj_30_path, traj_30)

    """Extract grasping trajectory"""
    gp = np.hstack([traj_30[:, :3], traj_30[:, 6:9]])

    gp_path = "/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/gp_{}_{}_demo_{}.npy".format(material, task, demo_id)
    np.save(gp_path, gp)

    """Extract particle index"""
    id_all = pt_indentify_hang(traj_30)
    # pt_indentify_table(traj_30)
    idx_all.append(id_all)

idx_all = np.array(idx_all)
np.save("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/idx_all_{}_{}.npy".format(material, task), idx_all)

