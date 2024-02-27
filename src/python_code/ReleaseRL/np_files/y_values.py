import numpy as np

data = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/ReleaseRL/np_files/hang_pose.npy")
# print(data.shape)
y_value = []
for i in range(len(data)):
    y_value.append(data[i, 1])

y_value = np.array(y_value)
print(min(y_value))
print(y_value.shape)