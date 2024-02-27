import numpy as np

init = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task_30.npy")

init_state = init[0].reshape(10, 3)
print(init_state)
#
particle1 = init[:, 9:12]
particle2 = init[:, 24:27]
particle3 = init[:, 0:3]
particle4 = init[:, 6:9]
particle5 = init[:, 27:30]
particle6 = init[:, 18:21]
particle7 = init[:, 3:6]
particle8 = init[:, 12:15]
particle9 = init[:, 15:18]
particle10 = init[:, 21:24]
#
new = np.hstack([particle1, particle2, particle3, particle4, particle5, particle6, particle7, particle8, particle9, particle10])
# print(new[0].reshape(10, 3))
# np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_real_table_30.npy", new)
# new = new.reshape(-1, 10, 3)
# print(new.shape)
# np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_real_table_3.npy", new)
#
# new = np.zeros((len(init), 30))
# for i in range(10):
#     new[:, i*3:i*3+3] = init[:, i*3:i*3+3]