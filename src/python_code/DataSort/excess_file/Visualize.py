from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation


fig = plt.figure()
ax = p3.Axes3D(fig)
ax.axes.set_xlim3d(left=-8, right=8)
ax.axes.set_ylim3d(bottom=-8, top=8)
ax.axes.set_zlim3d(bottom=2, top=12)

case1 = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_table_task.npy")

no_of_particles = 10

x = np.zeros([len(case1), no_of_particles])
y = np.zeros([len(case1), no_of_particles])
z = np.zeros([len(case1), no_of_particles])
# print(case1[:, 0].shape)

for i in range(no_of_particles):
    print((3*i)+(5-no_of_particles)*3)

    x[:, i] = case1[:, (3*i)+(5-no_of_particles)*3]
    y[:, i] = case1[:, ((3*i)+(5-no_of_particles)*3 + 1)]
    z[:, i] = case1[:, ((3*i)+(5-no_of_particles)*3 + 2)]

x_int = x[0]
y_int = y[0]
z_int = z[0]

print(x.shape)

points, = ax.plot(x_int, y_int, z_int, '*')
txt = fig.suptitle('')
i = 0
def update_points(num, x_int, y_int, z_int, points):
    txt.set_text('num={:d}'.format(num)) # for debug purposes

    # calculate the new sets of coordinates here. The resulting arrays should have the same shape
    # as the original x,y,z
    new_x = x[num*6]
    new_y = z[num*6]
    new_z = y[num*6]

    # update properties
    points.set_data(new_x,new_y)
    points.set_3d_properties(new_z, 'z')

    # return modified artists
    return points,txt

num = int(len(case1)/6)-1
ani = animation.FuncAnimation(fig, update_points, frames=num, fargs=(x, y, z, points))

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel('z')
plt.show()

# markers = np.zeros()