import numpy as np
import math
import matplotlib.pyplot as plt

def random_reset(data, plot=False, no_angle=False):
    random_angle = np.random.uniform(-1/4 * math.pi, 1/4 * math.pi)
    # random_angle = 1/4 * math.pi
    print(random_angle)
    if no_angle:
        random_angle = 0
    ori_array = np.array([math.sin(random_angle), 0, math.cos(random_angle)])

    distance = []
    data_ori = data.copy()
    data_ori[:, 1] = 0

    for i in range(len(data)):
        distance.append(np.linalg.norm(data_ori[0] - data_ori[i]))

    distance = np.array(distance)
    ori_distance = np.zeros([1, 3])

    for i in range(len(data)-1):
        ori_array_now = ori_array * distance[i+1]
        # print(data[i])
        ori_array_now[1] = data[i+1][1]
        ori_array_now = ori_array_now + data_ori[0]
        ori_distance = np.vstack([ori_distance, ori_array_now])

    ori_distance[0] = data[0]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1.0, 1.0, 1.0])

        for i in range(len(ori_distance)):
            # Init point of the bar
            ax.scatter(-ori_distance[i][0], ori_distance[i][2], ori_distance[i][1], c='r', marker='o')
            ax.scatter(-data[i][0], data[i][2], data[i][1], c='b', marker='o')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Z-axis')
        ax.set_zlabel('Y-axis')

        ax.set_xlim(5, 12)

        plt.show()

    return ori_distance

if __name__ == "__main__":
    data = np.load("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/x_init_table_task.npy")
    x_new = random_reset(data, plot=True)
    # print(x_new.shape)
    # np.save('/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/x_init_hang_ori.npy', x_new)
# np.save('/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/x_init_hang_ori.npy', ori_distance)

