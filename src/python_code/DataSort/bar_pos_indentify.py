import numpy as np

init_gp = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_hang_task.npy")[0]
print(init_gp)

robot_init = np.array([-19.1, 11.28, -1.1, -19.1, 11.28, 2.1])
bar_pos = np.array([-18.84, 7.355, 0.4, -18.84, 7.355, 0.4])
diff = robot_init - bar_pos
print(robot_init)
print(diff)

bar_pos_new = init_gp - diff
bar_pos_new = bar_pos_new.reshape(2, 3)
print(bar_pos_new)