"""
Image segmentation using SAM with prompt
============================================================

This example runs an interactive demo which allows user to select a region of interest and generate a mask for it.
"""

import os

import cv2
import matplotlib.pyplot as plt

import rofunc as rf
from sam_seg import sam_predict

image_path = "/home/ubuntu/Github/ManiCloth/src/python_code/eval/SAM_figs/small_cloth_crumpled.JPG"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()
masks, scores, logits = sam_predict(image,
                                    use_point=False,
                                    use_box=True,
                                    choose_best_mask=True)

masks = masks[0].flatten()

num_of_true = 0
num_of_false = 0

for i in range(len(masks)):
    if masks[i] == True:
        num_of_true += 1
    else:
        num_of_false += 1

area_coverage_table = num_of_true

masks2, scores2, logits2 = sam_predict(image,
                                       use_point=False,
                                       use_box=True,
                                       choose_best_mask=True)

masks2 = masks2[0].flatten()

num_of_true = 0
num_of_false = 0

for i in range(len(masks2)):
    if masks2[i] == True:
        num_of_true += 1
    else:
        num_of_false += 1

area_coverage_cloth = num_of_true

area_coverage_percent = (area_coverage_cloth / (area_coverage_cloth + area_coverage_table)) * 100
print("Coverage Percentile = ", area_coverage_percent, "%")