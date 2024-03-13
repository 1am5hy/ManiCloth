import numpy as np

marker_hang_30 = np.load("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/marker_hang_task_30.npy")

marker_hang_60 = np.hstack([marker_hang_30, marker_hang_30])

bar_anchor_1 = np.array([-3.893591, 7.0981565, 0.2083674])
bar_len = 7.218507544889505
joint_axis = np.array([-0.34292087, -0.00287866, 0.93935988])

bar_anchor_2 = bar_anchor_1 + (bar_len * joint_axis)
print(bar_anchor_2)
bar_pos = np.hstack([bar_anchor_1, bar_anchor_2])

bar_new_len = np.linalg.norm(bar_anchor_2 - bar_anchor_1)
# print(bar_new_len)

for i in range(9):
    bar_pos = np.hstack([bar_pos, np.hstack([bar_anchor_1, bar_anchor_2])])


relative_marker_hang_60 = np.zeros(marker_hang_60.shape)

for i in range(len(marker_hang_60)):
    relative_marker_hang_60[i] = bar_pos - marker_hang_60[i]

np.save("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/relative_marker_hang_task_60_inference2.npy", relative_marker_hang_60)
np.save("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/bar_pos_60_inference2.npy", bar_pos)

"""
Table Task
"""
# marker_table_30 = np.load("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/marker_table_task_30.npy")
#
# marker_table_60 = np.hstack([marker_table_30, marker_table_30])
#
# table_pos = np.array([-6.56042, 3.768829, 0.0349113, -0.79241, 3.768829, 0.0562581])
#
# for i in range(9):
#     table_pos = np.hstack([table_pos, np.array([-6.56042, 3.768829, 0.0349113, -0.79241, 3.768829, 0.0562581])])
# #
# relative_table_task_60 = np.zeros(marker_table_60.shape)
#
# for i in range(len(marker_table_60)):
#     relative_table_task_60[i] = table_pos - marker_table_60[i]
#
# # print(relative_table_task_60.shape)
# # print(table_pos.shape)
# np.save("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/relative_marker_table_task_60.npy",
#         relative_table_task_60)
# np.save("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/table_pos_60.npy", table_pos)