import numpy as np

data = np.load("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/exp_txt_files/hang_inference_19_3_test.npy", allow_pickle=True)[:, 3:]/10
# print(len(data))

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# print(np.max(x) - np.min(x))
# print(np.max(y) - np.min(y))
# print(np.max(z) - np.min(z))


vel = []
speed = []
for i in range(len(data)-1):
    i = i + 1
    diff = (data[i] - data[i-1])/0.025
    vel.append(diff)
    speed.append(abs(diff))

vel = np.array(vel)
speed = np.array(speed)
init = np.zeros([1, 3])
# print(vel.shape)
vel = np.vstack([init, vel])

# print(max(vel[:, 0]))
# print(max(vel[:, 1]))
# print(max(vel[:, 2]))
print(max(speed[:, 0]))
print(max(speed[:, 1]))
print(max(speed[:, 2]))
print("..."*50)

vel_all = []
for i in range(len(vel)):
    velocity = np.sqrt(vel[i, 0]**2 + vel[i, 1]**2 + vel[i, 2]**2)
    vel_all.append(velocity)

vel_all = np.array(vel_all)
print(max(vel_all))
# print()
#
# for i in range(len(vel)):
#     if vel_all[i] == max(vel_all):
#         print(i)