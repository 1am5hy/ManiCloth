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
material = "leather"
task = "hang"


for i in range(10):
    demo_id = i + 1
    print(demo_id)
    input_path = '/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/xcelfiles/5hy_21_03/{}_{}/{}_{}_exp_{}.csv'.format(material, task, material, task, demo_id)
    parent_dir = os.path.dirname(input_path)
    objs_list, meta_list = rf.optitrack.get_objects(input_path)

    labels = rf.optitrack.data_clean(input_path, legacy=True, objs=objs_list[0])[0]

    labels = labels.to_numpy()
    # print(labels.shape)
    # labels = labels[:, 21:51]
    labels = labels[:, 33:63]
    # labels = labels[:, 2:]
    marker_table_new = np.array(labels)/10

    print(marker_table_new[0].shape)
    print(marker_table_new[100])

    save_path = "/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/marker_{}_{}_demo_{}.npy".format(material, task, demo_id)
    np.save(save_path, marker_table_new)
# primitive = labels[:, 30:33]
# print(primitive[0]/10)