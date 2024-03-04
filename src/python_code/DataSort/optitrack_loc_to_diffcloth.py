import numpy as np

"""
Hang Task - Bar Position Identification
"""

# robot_gripper_init = np.array([-19.1, 11.28, -1.1, -19.1, 11.28, 2.1])
# bar_opti_pos = np.array([-18.84, 7.355, 0.4, -18.84, 7.355, 0.4])
#
# diff = robot_gripper_init - bar_opti_pos
#
# init_gp = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_hang_task.npy")[0]
#
# bar_pos_new = init_gp - diff
# bar_pos_new = bar_pos_new.reshape(2, 3)
# print("The bar pose in diffcloth is ", format(bar_pos_new))

"""
Hang Task - diffcloth2opti - Bar Position Identification
"""

# init_gp = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_hang_task.npy")[0]
# # print(init_gp)
# cur_bar_pos = np.array([-5.29989, 6.811906, -0.285244, -5.29989, 6.811906, -0.285244])
#
# diff = init_gp - cur_bar_pos
#
# robot_gripper_init = np.array([-20.30, 9.725, -1.02, -20.30, 9.725, 2.11])
# opti_bar_pos = robot_gripper_init - diff
#
# print("The bar pose in real life is", format(opti_bar_pos))

"""
Table Cloth Spread Task - Table Position Identification
"""

robot_gripper_init = np.array([-20.1, 10.67, -3.17, -19.1, 10.87, 3.57])
table_opti_pos = np.array([-17.05, 2.90, 0.4, -18.84, 7.355, 0.4])

diff = robot_gripper_init - table_opti_pos

init_gp = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_table_task.npy")[0]
print(init_gp)

table_pos_new = init_gp - diff
table_pos_new = table_pos_new.reshape(2, 3)

print("The table pose in diffcloth is", format(table_pos_new))

"""
Table Cloth Spread Task - diffcloth to optitrack - Table Position Identification
"""

# init_gp = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_table_task.npy")[0]
# cur_table_pos = np.array([--3.68522, 3.768829, 3.926864, -3.68522, 3.768829, 3.926864])
#
# diff = init_gp - cur_table_pos
#
# robot_gripper_init = np.array([-19.1, 11.28, -1.1, -19.1, 11.28, 2.1])
# opti_table_pos = robot_gripper_init - diff
#
# print("The table pose in real life is", format(opti_table_pos))