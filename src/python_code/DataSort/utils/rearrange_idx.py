import numpy as np

def rearrange(data):
    # init = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task_30.npy")
    init = data
    init = init.reshape(-1, 10, 3)

    init_state = init[0].copy()

    stop = False
    mem = np.zeros_like(init[0].copy())
    # for k in range(10):
    while stop == False:
        for i in range(len(init_state) - 1):
            """
            Segment based on y-axis
            """
            if init[0][i+1][1] > init[0][i][1]:
                temp = init[:, i].copy()
                init[:, i] = init[:, i+1]
                init[:, i+1] = temp

        if np.linalg.norm([mem - init[0]]) == 0:
            stop = True
        else:
            mem = init[0].copy()

    # print(init[0])
    init_state = init[0].copy()
    """
    Segment based on z-axis, row by row
    """
    # for row 1 - 3
    for j in range(2):
        if init[0][j + 1][2] < init[0][j][2]:
            temp = init[:, j].copy()
            init[:, j] = init[:, j + 1].copy()
            init[:, j + 1] = temp

    # for row 5 - 6
    if init[0][5][2] < init[0][4][2]:
        temp = init[:, 4].copy()
        init[:, 4] = init[:, 5].copy()
        init[:, 5] = temp

    for j in range(2):
        if init[0][j + 8][2] < init[0][j + 7][2]:
            temp = init[:, j + 7].copy()
            init[:, j + 7] = init[:, j + 8].copy()
            init[:, j + 8] = temp

    # print(init[0])
    new = init.reshape(-1, 30)

    return new

if __name__ == "__main__":
    data = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task_30.npy")
    data = rearrange(data)
    print(data[0].reshape(-1, 3))