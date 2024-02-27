"""
Optitrack Export
================

This example shows how to clean the .csv files outputted by Optitrack.
"""

import rofunc as rf
import os
import numpy as np
import pandas as pd
#
input_path = '/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/Take 2023-11-14 02.00.39 PM.csv'
parent_dir = os.path.dirname(input_path)
objs_list, meta_list = rf.optitrack.get_objects(input_path)

data, labels = rf.optitrack.data_clean(parent_dir, legacy=False, objs=objs_list)[0]
print(labels)
print(data[0])
#
particle = np.array(data[:, 21:])
# print(particle[0])
print(particle.shape)
# np.save('lifting', particle)

# print(df)