"""
Optitrack Export
================

This example shows how to clean the .csv files outputted by Optitrack.
"""
import math

import rofunc as rf
import os
import numpy as np

input_path = '/src/python_code/DataSort/xcelfiles/dish_washing.csv'
parent_dir = os.path.dirname(input_path)
objs_list, meta_list = rf.optitrack.get_objects(input_path)
labels = rf.optitrack.data_clean(parent_dir, legacy=True, objs=objs_list[0])[0]

"""Table Identification"""
# labels = labels.to_numpy()
# table_center = labels[:, 6:9]
# table_length = (abs(labels[0, 9] - labels[0, 15]) + abs(labels[0, 12] - labels[0, 18])) / 2
# table_width = (abs(labels[0, 11] - labels[0, 14]) + abs(labels[0, 17] - labels[0, 20])) / 2
#
# print(table_center[0]/10)
# print(table_length/10)
# print(table_width/10)

"""Dish Identification"""
labels = labels.to_numpy()
dish_center = labels[0, 6:9]
radius = labels[0, 9:12] - labels[0, 12:15]
radius = math.sqrt(radius[0] ** 2 + radius[2] ** 2)

print(dish_center/10)
print(radius/10)
