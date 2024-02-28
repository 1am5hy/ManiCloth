import numpy as np
#
# interest = np.array([0, 1, 2, 21, 22, 23, 42, 43, 44, 147, 148, 149, 270, 271, 272, 312, 313, 314, 432, 433, 434, 630, 631, 632, 651, 652, 653, 672, 673, 674])
# release_pose = np.load("/home/ubuntu/Github/ManiCloth/src/python_code/ReleaseRL/np_files/hang_pose.npy").flatten()[interest]
#
# bar_pos = np.load("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/bar_pos.npy")
#
# relative_release_pose = bar_pos - release_pose
# # print(relative_release_pose.shape)
# np.save("/home/ubuntu/Github/ManiCloth/src/python_code/ReleaseRL/np_files/relative_hang_release.npy", relative_release_pose)

"""Table"""

interest = np.array([0, 1, 2, 21, 22, 23, 42, 43, 44, 189, 190, 191, 360, 361, 362, 399, 400, 401, 519, 520, 521, 765, 766, 767,
                 786, 787, 788, 807, 808, 809])
release_pose = np.load("/home/ubuntu/Github/ManiCloth/src/python_code/ReleaseRL/np_files/table_release_pose.npy").flatten()[interest]

table_pos = np.load("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/table_low_pos.npy")

relative_release_pose = table_pos - release_pose
# print(relative_release_pose.shape)
np.save("/home/ubuntu/Github/ManiCloth/src/python_code/ReleaseRL/np_files/relative_table_release.npy", relative_release_pose)