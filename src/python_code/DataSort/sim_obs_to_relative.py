import numpy as np

marker_hang_30 = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_table_task_obs_30.npy")

bar_pos = np.array([-3.68522, 3.768829, 3.926864])
for i in range(9):
    bar_pos = np.hstack([bar_pos, np.array([-3.68522, 3.768829, 3.926864])])

relative_marker_hang_30 = np.zeros(marker_hang_30.shape)
# print(relative_marker_hang_30.shape)
for i in range(len(marker_hang_30)):
    relative_marker_hang_30[i] = bar_pos - marker_hang_30[i]

np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/relative_marker_table_task_30.npy", relative_marker_hang_30)
np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/table_low_pos.npy", bar_pos)