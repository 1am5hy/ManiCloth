import numpy as np
import math

from scipy.spatial.transform import Rotation as R

data1 = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_dish.npy")[0]
print(data1)

# data1 = np.array([3, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3])
# data1 = np.array([3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 3, 0])
# data1 = np.array([-1.5, 0, -1.5, -1.5, -3, -1.5, 1.5, 0, -1.5, 1.5, -3, -1.5])

def edge_to_quat(data):
    p1 = data[0:3]
    p2 = data[3:6]
    p3 = data[6:9]
    p4 = data[9:12]

    center_x = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
    center_y = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
    center_z = (p1[2] + p2[2] + p3[2] + p4[2]) / 4
    center = np.array([center_x, center_y, center_z])

    p1 = p1 - center
    p2 = p2 - center
    p3 = p3 - center

    line1 = (p1 - p3)/np.linalg.norm(p1 - p3)
    line3 = (p2 - p1)/np.linalg.norm(p2 - p1)
    line2 = np.cross(line3, line1)

    w1 = np.linalg.norm(p1 - p3)/2
    w2 = np.linalg.norm(p2 - p1)/2

    rot = np.array([line3, line1, line2])

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [center_x, center_y, center_z]

    r = R.from_matrix(T[:3, :3])
    quat = r.as_quat()

    pose = np.array([center_x, center_y, center_z, quat[0], quat[1], quat[2], quat[3]])

    return pose, w1, w2


def quat_to_edge(quat, w1, w2):
    # w1 = 1.5
    # w2 = 1.5
    center = quat[0:3]
    quat = R.from_quat(quat[3:])
    rot = quat.as_matrix()
    p1 = center + np.array([-w1, w2, 0]) @ rot
    p2 = center + np.array([w1, w2, 0]) @ rot
    p3 = center + np.array([-w1, -w2, 0]) @ rot
    p4 = center + np.array([w1, -w2, 0]) @ rot

    return np.array([p1, p2, p3, p4]).flatten()

if __name__ == '__main__':
    pose, w1, w2 = edge_to_quat(data1)
    p = quat_to_edge(pose, w1, w2)

    data = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_spread.npy")
    gp_modified = []
    for i in range(len(data)):
        pose, w1, w2 = edge_to_quat(data[i])
        p = quat_to_edge(pose, w1, w2)
        gp_modified.append(p)

    gp_modified = np.array(gp_modified)

    # diff = np.max(gp_modified - data)
    # print(diff)
    # print(gp_modified.shape)
    # np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_spread_modified.npy", gp_modified)
