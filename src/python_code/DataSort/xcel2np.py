"""
Optitrack Export
================

This example shows how to clean the .csv files outputted by Optitrack.
"""

import rofunc as rf
import os
import numpy as np

# If input_path points to a folder, each element of objs and meta is the data corresponding to one file.
# In a folder, only the file with the following name format are considered: 'Take*.csv'
# If input_file points to a file, objs and meta are lists with only one element.
input_path = '/src/python_code/DataSort/xcel_extract/dish_washing.csv'
parent_dir = os.path.dirname(input_path)
objs_list, meta_list = rf.optitrack.get_objects(input_path)
# print(objs_list[0])
labels = rf.optitrack.data_clean(parent_dir, legacy=True, objs=objs_list[0])[0]
print(labels.to_numpy())
print(labels.shape)
print(type(labels))


# data_rigid = labels.index('bar.pose.x')
# assert data_rigid + 6 == labels.index('bar.pose.qw')
# bar_pos_x = data[:, data_rigid:data_rigid + 7] / 10

# print(bar_pos_x[0])

data_marker1 = labels.index('unlabeled 1111')
assert data_marker1 + 2 == labels.index('cloth:marker 001.pose.z')
marker_1 = data[:, data_marker1:data_marker1 + 3]

marker_all = []
no_of_marker = 10

for mark_idx in range(no_of_marker):
    mark_idx = mark_idx + 1
    data_marker = labels.index('cloth:marker 00{}.pose.x'.format(mark_idx))
    assert data_marker + 2 == labels.index('cloth:marker 00{}.pose.z'.format(mark_idx))
    marker_x = data[:, data_marker:data_marker + 3]
    marker_all.append(marker_x)

# Shift the trajectory to the origin
x_init = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/x_init_dyndemo.npy')
marker_all = np.array(marker_all)
marker_all = marker_all.transpose(1, 0, 2)
marker_all = marker_all / 10

shift = x_init[0] - marker_all[0][0]
marker_all = marker_all + shift
# bar_pos_x[0, 0:3] = bar_pos_x[0, 0:3] + shift

# data_rigid = labels.index('bar:marker 001.pose.x')
# assert data_rigid + 2 == labels.index('bar:marker 001.pose.z')
# bar_marker1 = data[:, data_rigid:data_rigid + 3]
#
# data_rigid = labels.index('bar:marker 002.pose.x')
# assert data_rigid + 2 == labels.index('bar:marker 002.pose.z')
# bar_marker2 = data[:, data_rigid:data_rigid + 3]
#
# radius = np.linalg.norm(bar_marker1[0] - bar_marker2[0]) / 2
# # print(radius)
# marker_all = marker_all.reshape(-1, no_of_marker * 3)
# print(bar_pos_x[0])

# print(gp.shape)
markers = np.zeros([int(len(marker_all)/3), no_of_marker, 3])
# print(marker_all.shape)
for i in range(int(len(marker_all)/3)):
    markers[i] = marker_all[i*3]

gp = markers[:, [0, 2]].reshape(len(markers), -1)
gp[0, 3:] = x_init[14]

# print(markers.shape)
# print(markers)
# print(markers.shape)
# np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_hang.npy", gp)

# np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang2.npy", markers)