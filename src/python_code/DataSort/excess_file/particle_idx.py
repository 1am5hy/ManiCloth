import numpy as np

init = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang_task_3.npy")[0]
# init = init.reshape(8, 3)
print(init[0])
# print(init[1])
"Spread Task"

# particle1 = init[21:24]
# particle2 = init[0:3]
# particle3 = init[6:9]
# particle4 = init[3:6]
# particle5 = init[15:18]
# particle6 = init[9:12]
# particle7 = init[18:21]
# particle8 = init[12:15]
#
# len1 = abs(particle1[0] - particle2[0])
# len2 = abs(particle1[2] - particle7[2])
# print(len1)
#
# x1 = abs(particle1[0] - particle3[0])
# z1 = abs(particle1[2] - particle3[2])
#
# idx1 = int(x1 / len1 * 15)
# idz1 = int(z1 / len2 * 15)
# id1 = int((idz1 - 1)) * 15 + int((idx1-1))
# print(id1)
#
# x2 = abs(particle1[0] - particle4[0])
# z2 = abs(particle1[2] - particle4[2])
#
# idx2 = int(x2 / len1 * 15)
# idz2 = int(z2 / len2 * 15)
# id2 = int((idz2 - 1)) * 15 + int((idx2 - 1))
# print(id2)
#
# x3 = abs(particle1[0] - particle5[0])
# z3 = abs(particle1[2] - particle5[2])
#
# idx3 = int(x3 / len1 * 15)
# idz3 = int(z3 / len2 * 15)
# id3 = int((idz3 - 1)) * 15 + int((idx3 - 1))
# print(id3)
#
# x4 = abs(particle1[0] - particle6[0])
# z4 = abs(particle1[2] - particle6[2])
#
# idx4 = int(x4 / len1 * 15)
# idz4 = int(z4 / len2 * 15)
# id4 = int((idz4 - 1)) * 15 + int((idx4 - 1))
# print(id4)
#
# print("0, 14, {}, {}, {}, {}, 210, 224".format(id1, id2, id3, id4))

"Hang Task"

# init[:, ]

particle1 = init[0]
particle2 = init[1]
particle3 = init[2]
particle4 = init[3]
particle5 = init[4]
particle6 = init[5]
particle7 = init[6]
particle8 = init[7]
particle9 = init[8]
particle10 = init[9]
#
len1 = abs(particle3[2] - particle1[2])
len2 = abs(particle1[1] - particle8[1])
#
print(len1)
print(len2)
#
x1 = abs(particle4[2] - particle1[2])
y1 = abs(particle1[1] - particle4[1])
#
idx1 = int(x1 / len1 * 15)
idy1 = int(y1 / len2 * 15)
id1 = int((idy1 - 1)) * 15 + int((idx1-1))
print(id1)
#
x2 = abs(particle5[2] - particle1[2])
y2 = abs(particle1[1] - particle5[1])

idx2 = int(x2 / len1 * 15)
idy2 = int(y2 / len2 * 15)
id2 = int((idy2 - 1)) * 15 + int((idx2-1))
print(id2)
#
x3 = abs(particle6[2] - particle1[2])
y3 = abs(particle1[1] - particle6[1])

idx3 = int(x3 / len1 * 15)
idy3 = int(y3 / len2 * 15)
id3 = int((idy3 - 1)) * 15 + int((idx3-1))
print(id3)

x4 = abs(particle7[2] - particle1[2])
y4 = abs(particle1[1] - particle7[1])

idx4 = int(x4 / len1 * 15)
idy4 = int(y4 / len2 * 15)
id4 = int((idy4 - 1)) * 15 + int((idx4-1))
print(id4)
#
print("0, 7, 14, {}, {}, {}, {}, 255, 262, 269".format(id1, id2, id3, id4))


# init

# coord4 = init[0] - init[3]
# coord7 = init[0] - init[6]
# grid_length = init[0]-init[-1]
#
# coord4 = coord4 / grid_length
# coord4 = coord4 * 15
# idx = int((coord4[1] - 1)) * 15 + int((coord4[0]-1))
#
# coord7 = coord7 / grid_length
# coord7 = coord7 * 15
# idx2 = int((coord7[1] - 1)) * 15 + int((coord7[0]-1))

# print("0, 7, 14, {}, 105, 119, {}, 210, 217, 224".format(idx, idx2))