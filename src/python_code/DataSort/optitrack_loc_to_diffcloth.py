import numpy as np

"""
Hang Task - Bar Position Identification
"""

robot_gripper_init = np.array([-19.13, 10.255, -1.07, -18.88, 10.2775, 2.1607])
bar_opti_pos = np.array([-16.21, 7.3670, -1.4170, -18.5670, 7.3670, 5.26])


diff = robot_gripper_init - bar_opti_pos

init_gp = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_hang_task.npy")[0]
print(init_gp)

bar_pos_new = init_gp - diff
bar_anchor_1 = bar_pos_new[0:3]
bar_anchor_2 = bar_pos_new[3:6]
bar_len = np.linalg.norm(bar_anchor_2 - bar_anchor_1)
print(bar_len)

joint_axis = (bar_anchor_2 - bar_anchor_1)/bar_len
print(joint_axis)
print(np.linalg.norm(joint_axis))

# print()
# bar_pos_new = bar_pos_new.reshape(2, 3)
# bar_pos_new = np.array([np.mean([bar_pos_new[0,0], bar_pos_new[1,0]]), np.mean([bar_pos_new[0,1], bar_pos_new[1,1]]), np.mean([bar_pos_new[0,2], bar_pos_new[1,2]])])
print("The bar anchor 1 pose in diffcloth is ", format(bar_anchor_1))
print("The bar joint axis in diffcloth is ", format(joint_axis))

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

# robot_gripper_init = np.array([-20.1, 10.67, -3.17, -19.1, 10.87, 3.57])
# table_opti_pos = np.array([-17.05, 2.90, 0.4, -18.84, 7.355, 0.4])
#
# diff = robot_gripper_init - table_opti_pos
#
# init_gp = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_table_task.npy")[0]
# print(init_gp)
#
# table_pos_new = init_gp - diff
# table_pos_new = table_pos_new.reshape(2, 3)
#
# print("The table pose in diffcloth is", format(table_pos_new))

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