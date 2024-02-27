import numpy as np

data_l = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/exp_txt_files/table_demo_left_lqt.npy")

frequency = 0.001
length = len(data_l)*frequency
length = length / 0.025

new_data_l = []
for i in range(int(length)):
    new_data_l.append(data_l[int(i*(frequency/0.025))][0:3])

new_data_l = np.array(new_data_l)
print(new_data_l.shape)

data_r = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/exp_txt_files/table_demo_right_lqt.npy")

frequency = 0.001
length = len(data_r)*frequency
length = length / 0.025

new_data_r = []
for i in range(int(length)):
    new_data_r.append(data_r[int(i*(frequency/0.025))][0:3])

new_data_r = np.array(new_data_r)
print(new_data_r.shape)