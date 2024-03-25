import isaacgym
import numpy as np
import rofunc as rf
import math

print("Data Simulation Testing")

data = np.load("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/exp_txt_files/hang_inference_19_3_test.npy", allow_pickle=True)/10
print(data[0])

data_l = data[:, 3:]
data_r = data[:, :3]
data_mean = (data_l + data_r) / 2
# print(data_mean.shape)

"""Angle rotate
"""
# angle = 0.3500261433323031
# rot_matrix = np.array([[math.cos(angle), 0, -math.sin(angle)],
#                        [0, 1, 0],
#                        [math.sin(angle), 0, math.cos(angle)]])
#
# data_l = np.dot(data_l, rot_matrix)
# data_r = np.dot(data_r, rot_matrix)


# r = data[:, :3]
# l = data[:, 3:]
r = data_r
l = data_l
traj_l = np.zeros([len(l), 7])
traj_r = np.zeros([len(r), 7])
traj_l[:, :3] = l
traj_r[:, :3] = r
traj_l[:, 3:] = [1, 0, 0, 0]
traj_r[:, 3:] = [1, 0, 0, 0]

temp = traj_l[:, 1].copy()
traj_l[:, 1] = -1 * traj_l[:, 2]
traj_l[:, 2] = temp

traj_l[:, 0] += 1.5
traj_l[:, 1] -= 0.5

temp = traj_r[:, 1].copy()
traj_r[:, 1] = -1 * traj_r[:, 2]
traj_r[:, 2] = temp
traj_r[:, 0] += 1.5
traj_r[:, 1] -= 0.5

rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)

args = rf.config.get_sim_config("CURI")
# CURIsim = rf.sim.CURISim(args)
# CURIsim.run_traj(traj=[traj_l, traj_r],
#                  update_freq=0.001)
#
# """
# LQT Left
# """
# print("LQT Left processing")
#
# data = traj_l
# print("Time = {}".format(len(data)*0.025))
# print("LQT no. of steps = {}".format((len(data)*0.025)/0.001))
# print("LQT nb_data = {}".format(((len(data)*0.025)/0.001)/25))
#
# via_points = data
# cfg = rf.config.utils.get_config("./planning", "lqt")
# controller = rf.planning_control.lqt.LQTHierarchical(via_points, cfg, interval=3)
# u_hat_l, x_hat_l, mu_l, idx_slices_l = controller.solve()
# rf.lqt.plot_3d_uni(x_hat_l, mu_l, idx_slices_l, ori=False, save=False)
# print(len(x_hat_l))
#
# # np.save("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/exp_txt_files/hang_inference_left_11_3_no_2_lqt.npy", x_hat_l)
# # np.savetxt("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/exp_txt_files/hang_inference_left_11_3_no_2_lqt.txt", x_hat_l)
#
# """
# LQT right
# """
# print("LQT right processing")
#
# data = traj_r
# print("Time = {}".format(len(data)*0.025))
# print("LQT no. of steps = {}".format((len(data)*0.025)/0.001))
# print("LQT nb_data = {}".format(((len(data)*0.025)/0.001)/25))
#
# via_points = data
# cfg = rf.config.utils.get_config("./planning", "lqt")
# controller = rf.planning_control.lqt.LQTHierarchical(via_points, cfg, interval=3)
# u_hat_r, x_hat_r, mu_r, idx_slices_r = controller.solve()
# rf.lqt.plot_3d_uni(x_hat_r, mu_r, idx_slices_r, ori=False, save=False)
# print(len(x_hat_r))
#
# # np.save("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/exp_txt_files/hang_inference_right_11_3_no_2_lqt.npy", x_hat_r)
# # np.savetxt("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/exp_txt_files/hang_inference_right_11_3_no_2_lqt.txt", x_hat_r)
#
# """
# Test LQT on Simulated Curi
# """
# print("LQT Simulation Testing")
#
# rf.lqt.plot_3d_bi(x_hat_l, x_hat_r, ori=False)
#
# args = rf.config.get_sim_config("CURI")
# CURIsim = rf.sim.CURISim(args)
# CURIsim.run_traj(traj=[x_hat_l, x_hat_r],
#                  update_freq=0.001)