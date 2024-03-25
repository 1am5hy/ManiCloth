import numpy as np
import math
import matplotlib.pyplot as plt

def random_reset(data, plot=False, angle=False, translate=False):
    if angle is not False:
        random_angle = angle
    else:
        random_angle = np.random.uniform(-1 / 4 * math.pi, 1 / 4 * math.pi)

    print(random_angle)

    ori_array = np.array([math.sin(random_angle), 0, math.cos(random_angle)])

    distance = []
    data_ori = data.copy()
    data_ori[:, 1] = 0

    for i in range(len(data)):
        dis = data_ori[i] - data_ori[7]

        if dis[2] < 0:
            distance.append(-np.linalg.norm(data_ori[7] - data_ori[i]))
        else:
            distance.append(np.linalg.norm(data_ori[7] - data_ori[i]))
        # distance.append(np.linalg.norm(data_ori[7] - data_ori[i]))

    distance = np.array(distance)
    ori_distance = np.zeros([1, 3])

    for i in range(len(data)-1):
        ori_array_now = ori_array * distance[i+1]
        # print(data[i])
        ori_array_now[1] = data[i+1][1]
        ori_array_now = ori_array_now + data_ori[7]
        ori_distance = np.vstack([ori_distance, ori_array_now])

    ori_distance[7] = data[7]
    ori_distance[0] = data[0]

    if translate is not False:
        random_translate = translate
    else:
        random_x = np.random.uniform(-1.5, -0.5)
        random_y = np.random.uniform(-1, 1)

        if random_angle > 0:
            random_z = np.random.uniform(0, 1)
        else:
            random_z = np.random.uniform(-1, 0)

        random_translate = np.array([random_x, random_y, random_z])

    # for i in range()
    ori_distance = ori_distance + random_translate

    print(random_translate)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1.0, 1.0, 1.0])

        for i in range(len(ori_distance)):
            # Init point of the bar
            ax.scatter(ori_distance[i][0], ori_distance[i][2], ori_distance[i][1], c='r', marker='o')
            ax.scatter(data[i][0], data[i][2], data[i][1], c='b', marker='o')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Z-axis')
        ax.set_zlabel('Y-axis')

        ax.set_xlim(-3, -7)

        plt.show()


    return ori_distance

if __name__ == "__main__":
    data = np.load("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/x_init_hang.npy")
    # x_new = random_reset(data, plot=True, angle=False, translate=False)
    x_new = random_reset(data, plot=True, angle=-0.3500261433323031, translate=np.array([0.16861055, 0.2871107, -1.27474583]))
    print(x_new[0])
    print(x_new[14])

    # print(x_new)
    # print(data-x_new)
    # print(x_new[7])
    # y = x_new[:, 1]
    # print(min(y))
    # np.save('/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/x_init_hang_ori.npy', x_new)
# np.save('/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/x_init_hang_ori.npy', ori_distance)

