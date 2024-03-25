import numpy as np
import math

def pt_indentify_table(traj):
    init = traj[0].reshape(10, 3)
    "Table Task"

    # init[:, ]

    particle1 = init[0]
    particle3 = init[2]
    particle4 = init[3]
    particle5 = init[4]
    particle6 = init[5]
    particle7 = init[6]
    particle8 = init[7]

    len1 = abs(particle3[2] - particle1[2])
    len2 = abs(particle1[1] - particle8[1])
    print(len1)
    print(len2)

    x1 = abs(particle4[2] - particle1[2])
    y1 = abs(particle1[1] - particle4[1])

    idx1 = int(x1 / len1 * 15)
    idy1 = math.floor(y1 / len2 * 18)
    id1 = (idy1 - 1) * 15 + idx1

    x2 = abs(particle5[2] - particle1[2])
    y2 = abs(particle1[1] - particle5[1])

    idx2 = int(x2 / len1 * 15)
    idy2 = int(y2 / len2 * 18)
    id2 = int(idy2 - 1) * 15 + idx2

    x3 = abs(particle6[2] - particle1[2])
    y3 = abs(particle1[1] - particle6[1])

    idx3 = int(x3 / len1 * 15)
    idy3 = int(y3 / len2 * 18)
    id3 = int((idy3 - 1)) * 15 + idx3

    x4 = abs(particle7[2] - particle1[2])
    y4 = abs(particle1[1] - particle7[1])

    idx4 = int(x4 / len1 * 15)
    idy4 = int(y4 / len2 * 18)
    id4 = int((idy4 - 1)) * 15 + idx4

    print("0, 7, 14, {}, {}, {}, {}, 255, 262, 269".format(id1, id2, id3, id4))

def pt_indentify_hang(traj):
    init = traj[0].reshape(10, 3)

    "Hang Task"

    particle1 = init[0]
    particle3 = init[2]
    particle4 = init[3]
    particle5 = init[4]
    particle6 = init[5]
    particle7 = init[6]
    particle8 = init[7]

    len1 = abs(particle3[2] - particle1[2])
    len2 = abs(particle1[1] - particle8[1])
    # print(len1)
    # print(len2)

    x1 = abs(particle4[2] - particle1[2])
    y1 = abs(particle1[1] - particle4[1])

    idx1 = int(x1 / len1 * 15)
    idy1 = math.floor(y1 / len2 * 15)
    id1 = (idy1 - 1) * 15 + idx1

    x2 = abs(particle5[2] - particle1[2])
    y2 = abs(particle1[1] - particle5[1])

    idx2 = int(x2 / len1 * 15)
    idy2 = int(y2 / len2 * 15)
    id2 = int(idy2 - 1) * 15 + idx2

    x3 = abs(particle6[2] - particle1[2])
    y3 = abs(particle1[1] - particle6[1])

    idx3 = int(x3 / len1 * 15)
    idy3 = int(y3 / len2 * 15)
    id3 = int((idy3 - 1)) * 15 + idx3

    x4 = abs(particle7[2] - particle1[2])
    y4 = abs(particle1[1] - particle7[1])

    idx4 = int(x4 / len1 * 15)
    idy4 = int(y4 / len2 * 15)
    id4 = int((idy4 - 1)) * 15 + idx4

    print("0, 7, 14, {}, {}, {}, {}, 210, 217, 224".format(id1, id2, id3, id4))
    id_all = np.array([0, 7, 14, id1, id2, id3, id4, 210, 217, 224])

    return id_all

if __name__ == "__main__":
    traj = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task_3.npy")
    pt_indentify_hang(traj)