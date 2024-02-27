import random
import time
import torch
import numpy as np
import diffcloth_py as dfc

from pySim.pySim import pySim


def get_default_scene():
    fabric = dfc.FabricConfiguration()
    fabric.clothDimX = 3
    fabric.clothDimY = 3
    fabric.k_stiff_stretching = 600
    fabric.k_stiff_bending = 1.06
    fabric.gridNumX = 15
    fabric.gridNumY = 15
    fabric.density = 0.3
    fabric.keepOriginalScalePoint = True
    fabric.isModel = False
    fabric.custominitPos = False
    fabric.initPosFile = ""
    fabric.fabricIdx = 0
    fabric.color = np.array([0.9, 0., 0.1])
    fabric.name = "dim6x6-grid25x25-dens0.32-k50"

    scene = dfc.SceneConfiguration()
    scene.fabric = fabric
    scene.orientation = dfc.Orientation.CUSTOM_ORIENTATION
    scene.upVector = np.array([0, 1, 0])
    scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
    # scene.customAttachmentVertexIdx = [(0.0, [0, 24])]
    # scene.customAttachmentVertexIdx = [(0.0, [1838])]
    scene.trajectory = dfc.TrajectoryConfigs.PER_STEP_TRAJECTORY
    scene.primitiveConfig = dfc.PrimitiveConfiguration.NONE
    scene.windConfig = dfc.WindConfig.NO_WIND
    # scene.camPos = np.array([-60, 20, -30.67])
    scene.camPos = np.array([-21.67, 5.40, -10.67])
    scene.sockLegOrientation = np.array([0., 0., 0.])
    scene.camFocusPointType = dfc.CameraFocusPointType.CLOTH_CENTER
    # scene.sceneBbox = dfc.AABB(np.array([-70, -70, -70]), np.array([70, 70, 70]))
    scene.sceneBbox = dfc.AABB(np.array([-7, -7, -7]), np.array([7, 7, 7]))
    scene.timeStep = 0.05
    scene.forwardConvergenceThresh = 1e-3
    scene.backwardConvergenceThresh = 1e-3
    scene.name = "Test scene"

    scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
    gp1_id = np.array([0])
    gp2_id = np.array([14])
    scene.customAttachmentVertexIdx = [(0, [0, 7, 14, 84, 105, 119, 156, 210, 217, 224])]
    # scene.stepNum = num_step
    scene.stepNum = 50

    return scene


def stepSim(simModule, sim, init, pos):
    start = time.time()
    x = init[0]
    v = init[1]
    x_list, v_list = [x.reshape(-1, 3)], [v.reshape(-1, 3)]
    print(sim.sceneConfig.stepNum)
    print(sim.sceneConfig.timeStep)
    for step in range(sim.sceneConfig.stepNum-2):
        p = pos[step]
        x, v = simModule(x, v, p)
        x_list.append(x.reshape(-1, 3))
        v_list.append(v.reshape(-1, 3))


    # print("stepSim took {} sec".format(time.time() - start))
    render = True
    if render:
        dfc.render(sim, renderPosPairs=True, autoExit=True)

    return x_list, v_list

def get_grasp_traj(init, target, length):
    traj = init + (target - init) * np.linspace(0, 1, length)[:, None]
    return torch.tensor(traj)

scene = get_default_scene()
sim = dfc.makeSimFromConf(scene)

sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
np.set_printoptions(precision=5)

dfc.enableOpenMP(n_threads=50)

helper = dfc.makeOptimizeHelper("inverse_design")
helper.taskInfo.dL_dx0 = True
helper.taskInfo.dL_density = False

sim.forwardConvergenceThreshold = 1e-3

sim_mod = pySim(sim, helper, True)

def run_sim_step(step_num, target1, target2):
    # sim = dfc.makeSimFromConf(scene)

    scene = get_default_scene()

    scene.customAttachmentVertexIdx = [(0, [0, 7, 14, 84, 105, 119, 156, 210, 217, 224])]
    # scene.customAttachmentVertexIdx = [(0, [0, 14])]
    scene.stepNum = step_num
    sim = dfc.makeSimFromConf(scene)

    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=50)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-3

    sim.resetSystem()
    paramInfo = dfc.ParamInfo()
    x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/x_init_dyndemo.npy')
    #
    # translation = np.zeros(3)
    # translation[0] = random.uniform(-8, 8)
    # translation[1] = random.uniform(-8, 8)
    # translation[2] = random.uniform(-8, 8)
    #
    # translation_mat = np.tile(translation, (len(x), 1) )
    # x_shifted = x + translation_mat
    # x_shifted = torch.from_numpy(x_shifted)
    #
    paramInfo.x0 = x.flatten()
    sim.resetSystemWithParams(helper.taskInfo, paramInfo)
    # sim.resetSystem()


    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    gp1_id = np.array([1])
    gp2_id = np.array([14])

    # grasp_point_1 = pts[gp1_id].copy()
    # grasp1_traj = get_grasp_traj(grasp_point_1, target1, step_num)
    # # Fix this
    # #
    # grasp_point_2 = pts[gp2_id].copy()
    # grasp2_traj = get_grasp_traj(grasp_point_2, target2, step_num)
    position_control = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/marker_path_dyndemo.npy')
    if len(position_control) > 150:
        position_control = position_control[:150]
    #
    pos0 = torch.tensor(position_control[:, 0])
    pos1 = torch.tensor(position_control[:, 1])
    pos2 = torch.tensor(position_control[:, 2])
    pos3 = torch.tensor(position_control[:, 3])
    pos4 = torch.tensor(position_control[:, 4])
    pos5 = torch.tensor(position_control[:, 5])
    pos6 = torch.tensor(position_control[:, 6])
    pos7 = torch.tensor(position_control[:, 7])
    pos8 = torch.tensor(position_control[:, 8])
    pos9 = torch.tensor(position_control[:, 9])

    db_pos = torch.cat((pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9), dim=1)
    #
    # grasp_trajs = torch.cat((grasp1_traj, grasp2_traj), dim=1)
    #
    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), db_pos)
    # x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), grasp_trajs)

    return x, v

def run_sim_step2(x_fin1, v_fin1):
    scene = get_default_scene()
    scene.stepNum = 200
    scene.customAttachmentVertexIdx = [(0, [0, 14])]

    sim = dfc.makeSimFromConf(scene)
    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=15)

    dfc.enableOpenMP(n_threads=10)

    sim.forwardConvergenceThreshold = 1e-3

    sim_mod = pySim(sim, helper, True)

    paramInfo = dfc.ParamInfo()
    paramInfo.x0 = x_fin1.flatten().detach().cpu().numpy()

    sim.resetSystemWithParams(helper.taskInfo, paramInfo)

    state_info_init = sim.getStateInfo()

    pts = state_info_init.x.reshape(-1, 3)

    gp_id = scene.customAttachmentVertexIdx[0][1][0]
    print("gp_id: ", gp_id)

    # Grasp trajectory is filled to prevent crash in stepSim, but values will be ignored
    grasp_point = pts[gp_id].copy()
    grasp_traj = np.tile(grasp_point, (scene.stepNum, 1))
    grasp_traj = torch.tensor(grasp_traj, dtype=torch.float32)

    x, v = stepSim(sim_mod, sim, (x_fin1.flatten(), v_fin1.flatten()), grasp_traj)

    # sim.resetSystem()

    return x, v

def run_step1():
    scene = get_default_scene()
    scene.attachmentPoints = dfc.AttachmentConfigs.NO_ATTACHMENTS
    scene.stepNum = 100

    sim = dfc.makeSimFromConf(scene)
    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)

    dfc.enableOpenMP(n_threads=15)

    helper = dfc.makeOptimizeHelper("inverse_design")

    sim.forwardConvergenceThreshold =  1e-3

    sim_mod = pySim(sim, helper, True)

    sim.resetSystem()
    state_info_init = sim.getStateInfo()

    pts = state_info_init.x.reshape(-1, 3)

    gp_id = scene.customAttachmentVertexIdx[0][1][0]
    # print("gp_id: ", gp_id)

    # Grasp trajectory is filled to prevent crash in stepSim, but values will be ignored
    grasp_point = pts[gp_id].copy()
    grasp_traj = np.tile(grasp_point, (scene.stepNum, 1))
    grasp_traj = torch.tensor(grasp_traj, dtype=torch.float32)

    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x),
                                  torch.tensor(state_info_init.v)), grasp_traj)

    return x, v


def run_step2(x_fin1, v_fin1):
    scene = get_default_scene()

    sim = dfc.makeSimFromConf(scene)
    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)

    dfc.enableOpenMP(n_threads=10)

    helper = dfc.makeOptimizeHelper("inverse_design")
    helper.taskInfo.dL_dx0 = True
    helper.taskInfo.dL_density = False

    sim.forwardConvergenceThreshold = 1e-3

    sim_mod = pySim(sim, helper, True)

    paramInfo = dfc.ParamInfo()
    paramInfo.x0 = x_fin1.flatten().detach().cpu().numpy()

    sim.resetSystemWithParams(helper.taskInfo, paramInfo)

    state_info_init = sim.getStateInfo()

    pts = state_info_init.x.reshape(-1, 3)

    gp_id = scene.customAttachmentVertexIdx[0][1][0]
    # print("gp_id: ", gp_id)

    grasp_point = pts[gp_id].copy()
    target = grasp_point.copy()
    target[1] = grasp_point[1] + 8
    grasp_traj = get_grasp_traj(grasp_point, target, scene.stepNum)

    x, v = stepSim(sim_mod, sim, (x_fin1.flatten(), v_fin1.flatten()), grasp_traj)

    return x, v, gp_id


def get_gp2():
    gp2 = np.random.randint(2500, size=1)
    if gp2 == gp_id:
        gp2 = get_gp2()

    # print("gp2_id: ", gp2[0])
    return gp2


def run_step3(gp1_id, gp2_id, x_fin2, v_fin2):
    scene = get_default_scene()
    gp2_id = np.array([gp2_id])
    # gp2_id = get_gp2()
    scene.customAttachmentVertexIdx = [(0.0, [gp1_id, gp2_id])]

    sim = dfc.makeSimFromConf(scene)
    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)

    dfc.enableOpenMP(n_threads=5)

    helper = dfc.makeOptimizeHelper("inverse_design")
    helper.taskInfo.dL_dx0 = True
    helper.taskInfo.dL_density = False

    sim.forwardConvergenceThreshold = 1e-3

    sim_mod = pySim(sim, helper, True)

    paramInfo = dfc.ParamInfo()
    paramInfo.x0 = x_fin2.flatten().detach().cpu().numpy()

    sim.resetSystemWithParams(helper.taskInfo, paramInfo)

    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    grasp_point_1 = pts[gp1_id].copy()
    grasp1_traj = np.tile(grasp_point_1, (scene.stepNum, 1))
    grasp1_traj = torch.tensor(grasp1_traj, dtype=torch.float32)

    grasp_point_2 = pts[gp2_id].copy()
    target = grasp_point_2.copy()[0]
    target[1] = grasp_point_1[1]
    grasp2_traj = get_grasp_traj(grasp_point_2, target, scene.stepNum)

    grasp_trajs = torch.cat((grasp1_traj, grasp2_traj), dim=1)

    x, v = stepSim(sim_mod, sim, (x_fin2.flatten(), v_fin2.flatten()), grasp_trajs)

    return x, v, gp2_id

def run_step4(gp2_id, x_fin2, v_fin2):

    scene = get_default_scene()
    scene.stepNum = 100

    scene.customAttachmentVertexIdx = [(0.0, [gp2_id, gp2_id])]

    sim = dfc.makeSimFromConf(scene)
    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)

    dfc.enableOpenMP(n_threads=5)

    helper = dfc.makeOptimizeHelper("inverse_design")
    helper.taskInfo.dL_dx0 = True
    helper.taskInfo.dL_density = False

    sim.forwardConvergenceThreshold = 1e-3

    sim_mod = pySim(sim, helper, True)

    paramInfo = dfc.ParamInfo()
    paramInfo.x0 = x_fin2.flatten().detach().cpu().numpy()

    sim.resetSystemWithParams(helper.taskInfo, paramInfo)

    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    grasp_point_1 = pts[gp2_id].copy()
    grasp1_traj = np.tile(grasp_point_1, (scene.stepNum, 1))
    grasp1_traj = torch.tensor(grasp1_traj, dtype=torch.float32)

    grasp_trajs = torch.cat((grasp1_traj, grasp1_traj), dim=1)

    x, v = stepSim(sim_mod, sim, (x_fin2.flatten(), v_fin2.flatten()), grasp_trajs)

    return x, v, gp2_id

def get_gp3():
    gp3 = np.random.randint(2500, size=1)
    if gp3 == gp_id:
        gp3 = get_gp3()

    # print("gp3_id: ", gp3[0])
    return gp3

def run_step5(gp2_id, gp3_id, x_fin2, v_fin2):
    scene = get_default_scene()
    gp3_id = np.array([gp3_id])
    # gp3_id = get_gp3()
    scene.customAttachmentVertexIdx = [(0.0, [gp3_id, gp2_id])]

    sim = dfc.makeSimFromConf(scene)
    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)

    dfc.enableOpenMP(n_threads=5)

    helper = dfc.makeOptimizeHelper("inverse_design")
    helper.taskInfo.dL_dx0 = True
    helper.taskInfo.dL_density = False

    sim.forwardConvergenceThreshold = 1e-3

    sim_mod = pySim(sim, helper, True)

    paramInfo = dfc.ParamInfo()
    paramInfo.x0 = x_fin2.flatten().detach().cpu().numpy()

    sim.resetSystemWithParams(helper.taskInfo, paramInfo)

    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    grasp_point_2 = pts[gp2_id].copy()
    grasp2_traj = np.tile(grasp_point_2, (scene.stepNum, 1))
    grasp2_traj = torch.tensor(grasp2_traj, dtype=torch.float32)

    grasp_point_3 = pts[gp3_id].copy()
    target = grasp_point_3.copy()[0]
    target[1] = grasp_point_2[0][1]
    grasp3_traj = get_grasp_traj(grasp_point_3, target, scene.stepNum)

    grasp_trajs = torch.cat((grasp3_traj, grasp2_traj), dim=1)

    x, v = stepSim(sim_mod, sim, (x_fin2.flatten(), v_fin2.flatten()), grasp_trajs)
    # print(gp3_id, gp2_id)

    return x, v, (gp3_id, gp2_id)

def run_step6(gp_ids, x_fin3, v_fin3):
    scene = get_default_scene()

    scene.customAttachmentVertexIdx = [(0.0, gp_ids)]

    sim = dfc.makeSimFromConf(scene)
    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)

    dfc.enableOpenMP(n_threads=5)

    helper = dfc.makeOptimizeHelper("inverse_design")
    helper.taskInfo.dL_dx0 = True
    helper.taskInfo.dL_density = False

    sim.forwardConvergenceThreshold = 1e-3

    sim_mod = pySim(sim, helper, True)

    paramInfo = dfc.ParamInfo()
    paramInfo.x0 = x_fin3.flatten().detach().cpu().numpy()

    sim.resetSystemWithParams(helper.taskInfo, paramInfo)

    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    grasp_point_1 = pts[gp_ids[0]].copy()
    target = grasp_point_1.copy()
    # print(target, grasp_point_1)
    target[0][2] = grasp_point_1[0][2] - 5
    grasp1_traj = get_grasp_traj(grasp_point_1, target, scene.stepNum)

    grasp_point_2 = pts[gp_ids[1]].copy()[0]
    target = grasp_point_2.copy()
    target[2] = grasp_point_2[2] - 5
    grasp2_traj = get_grasp_traj(grasp_point_2, target, scene.stepNum)

    grasp_trajs = torch.cat((grasp1_traj, grasp2_traj), dim=1)

    x, v = stepSim(sim_mod, sim, (x_fin3.flatten(), v_fin3.flatten()), grasp_trajs)

    return x, v, gp_ids

if __name__=="__main__":
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import os

    # graph_timestep = 100
    # no_of_simulations = 1
    #
    # for j in range(no_of_simulations):
    #
    #     start = time.time()
    #     print("Beginning Simulation number {num}".format(num=j))
    #     x1, v1 = run_sim_step(900, [-2, 0, 6.5], [2, 0, 6.5])
    #
    #     x2, v2 = run_sim_step2(x1[-1], v1[-1])
    #
    #     x1_np = torch.stack(x1).numpy()
    #     x2_np = torch.stack(x2).numpy()
    #
    #     full_x = np.concatenate((x1_np, x2_np))
    #
    #     x_pos = np.zeros((int(len(full_x) / graph_timestep), len(full_x[0]), 3))
    #
    #     os.mkdir("/home/ubuntu/Github/DiffCloth/src/python_code/graphs/simulation_no_{number}".format(number=j))
    #
    #     for i in range(int(len(full_x) / graph_timestep)):
    #
    #         x_pos[i] = full_x[i * graph_timestep]
    #
    #         x = x_pos[i, :, 0].flatten()
    #         y = x_pos[i, :, 2].flatten()
    #         z = x_pos[i, :, 1].flatten()
    #
    #         fig = plt.figure(figsize=(10, 10))
    #         ax = plt.axes(projection='3d')
    #         ax.grid()
    #
    #         ax.scatter(x, y, z, c='r', s=50)
    #         ax.set_title('Cloth Graph Model at timestep {timestep}'.format(timestep=(i * graph_timestep)))
    #
    #         # Set axes label
    #         ax.set_xlabel('x', labelpad=20)
    #         ax.set_ylabel('y', labelpad=20)
    #         ax.set_zlabel('z', labelpad=20)
    #
    #         plt.savefig("/home/ubuntu/Github/DiffCloth/src/python_code/graphs/simulation_no_{number}/{timestep}.png".format(number=j, timestep=(i * graph_timestep)))
    #         # plt.savefig("/home/ubuntu/Github/DiffCloth/src/python_code/graphs/goal/{timestep}.png".format(timestep=(i * graph_timestep)))
    #         # plt.show()
    #         plt.close()
    #         # np.save("/home/hengyi/GitHub/DiffCloth/src/python_code/graphs/simulation_no_{number}/x_pos.npy".format(number=j), x_pos)
    #         # np.save("/home/hengyi/GitHub/DiffCloth/src/python_code/graphs/goal/x_pos.npy", x_pos)
    #
    #     now = time.time()
    #
    #     print("Simulation number {num} complete".format(num=j))
    #     print("Time taken for simulation number {num} is {time}".format(num=j, time=(now - start)))
    #
    #     print(full_x[0].shape)
    x, v = run_sim_step(150, [0, 3, 0], [3, 3, 0])

    # x_traj = np.zeros([200, 225, 3])
    #
    # for i in range(len(x)):
    #     x_traj[i] = x[i]
    # np.save("x_demo", x_traj)

    # run_sim_step2(x[-1], v[-1])
    # x = np.array(x[-1])
    # np.save("x_init_dyndemo", x)
    # np.savetxt("x_init_dyndemo.txt", x)
