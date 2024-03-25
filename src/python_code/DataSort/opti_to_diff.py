import numpy as np
import math

"""
For Cloth Hang Task
"""

robot_gripper_init = np.array([-19.13, 10.255, -1.07, -18.88, 10.255, 2.1607])
bar_opti_pos = np.array([-16.21, 7.3670, -1.4170, -18.5670, 7.3670, 5.26])
diff = bar_opti_pos - robot_gripper_init

init_gp = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_hang_task.npy")[0]

bar_real_offset = diff + init_gp

bar_diffcloth = np.array([-5.29989, 6.811906, -0.285244, -5.29989, 6.811906, 6.93326354])

# Calculate the translation
bar_opti_anchor1 = bar_real_offset[0:3]
bar_opti_anchor2 = bar_real_offset[3:6]
bar_opti_center = (bar_opti_anchor1 + bar_opti_anchor2)/2

bar_diffcloth_anchor1 = bar_diffcloth[0:3]
bar_diffcloth_anchor2 = bar_diffcloth[3:6]
bar_diffcloth_center = (bar_diffcloth_anchor1 + bar_diffcloth_anchor2)/2

print("Translation =", -1 * (bar_diffcloth_center - bar_opti_center))

# Calculate the angle
bar_opti_anchor1 = bar_real_offset[0:3]
bar_opti_anchor2 = bar_real_offset[3:6]

bar_len = np.linalg.norm(bar_opti_anchor2 - bar_opti_anchor1)

joint_axis = (bar_opti_anchor2 - bar_opti_anchor1)/bar_len
angle = math.acos(joint_axis[2])
angle2 = math.asin(-joint_axis[0])
print("The angle in radians is", angle)
print("The angle in degrees is", angle * 180 / math.pi)
print("To confirm the angle:", angle2 * 180 / math.pi)

"""
For Table Cloth Spread Task
"""

# robot_gripper_init = np.array([-20.1, 10.67, -3.17, -19.1, 10.87, 3.57])
# table_opti_pos = np.array([-17.05, 7.355, 0.4, -18.84, 7.355, 0.4])
#
# table_top_right = table_opti_pos[0:3]
# table_bottom_left = table_opti_pos[3:6]
#
# angle = math.atan((table_top_right[2] - table_bottom_left[2])/(table_top_right[0] - table_bottom_left[0]))
#
# table_diffcloth_center = np.array([-3.68522, 3.768829, 3.926864])
# table_opti_center = np.array([-3.68522, 3.768829, 3.926864])
#
# print("The Translation is", table_diffcloth_center - table_opti_center)
# print("The angle in radians is", -angle)
# print("The angle in degrees is", -angle * 180 / math.pi)
