import numpy as np
import torch

particles = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/lifting.npy')
par1 = particles[:, 12:]
par2 = particles[:, 9:12]
par3 = particles[:, 6:9]
par4 = particles[:, :3]
par5 = particles[:, 3:6]
# par6 = particles[:, 24:27]
# par7 = particles[:, 15:18]
# par8 = particles[:, 3:6]
# par9 = particles[:, 18:21]
# par10 = particles[:, 21:24]

new = np.zeros([int(len(particles)/6), 5, 3])
for i in range(int(len(particles)/6)):
    new[i] = np.array([par1[i*6],
                       par2[i*6],
                       par3[i*6],
                       par4[i*6],
                       par5[i*6]])
                       # par6[i*6],
                       # par7[i*6],
                       # par8[i*6],
                       # par9[i*6],
                       # par10[i*6]])

# np.save("lifting_005", new)

particles = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/x_init_lift.npy')
# print(particles[0])
#
markers = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/lifting_005.npy')
markers = markers * np.array([1, 1, -1])

markers = markers/10
particles[112][1] = 0
distance = particles[112] - markers[0, 2]
# distance[1] = 0

markers = markers + distance

# markers[0, 0] = particles[0]
# markers[0, 1] = particles[7]
# markers[0, 2] = particles[14]
# markers[0, 3] = particles[84]
# markers[0, 4] = particles[105]
# markers[0, 5] = particles[119]
# markers[0, 6] = particles[156]
# markers[0, 7] = particles[210]
# markers[0, 8] = particles[217]
# markers[0, 9] = particles[224]

# print(markers[0,0])
# print(markers[0,-3])
# markers = markers + [0, 0.1, 0]
# print(markers[:, 2, 1])
np.save("marker_path_lifting", markers)
markers = torch.tensor(markers)
gp = markers[:, 2]
gp = np.array(gp)
# print(gp)
np.save("gp_lifting.npy", gp)