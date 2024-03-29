import random
import time
import torch
import numpy as np
import diffcloth_py as dfc

from pySim.pySim_spread import pySim


def get_default_scene():
    fabric = dfc.FabricConfiguration()
    fabric.clothDimX = 3
    fabric.clothDimY = 3
    # fabric.k_stiff_stretching = 1000
    # fabric.k_stiff_bending = 0
    fabric.k_stiff_stretching = 768.15
    fabric.k_stiff_bending = 0.28
    fabric.gridNumX = 15
    fabric.gridNumY = 15
    fabric.density = 0.3
    fabric.keepOriginalScalePoint = True
    fabric.isModel = False
    fabric.custominitPos = False
    fabric.initPosFile = ""
    fabric.fabricIdx = 0
    fabric.color = np.array([0.9, 0., 0.1])
    fabric.name = "test"

    scene = dfc.SceneConfiguration()
    scene.fabric = fabric
    scene.orientation = dfc.Orientation.CUSTOM_ORIENTATION
    scene.upVector = np.array([0, 0, 1])
    scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
    # scene.customAttachmentVertexIdx = [(0.0, [0, 24])]
    # scene.customAttachmentVertexIdx = [(0.0, [1838])]
    scene.trajectory = dfc.TrajectoryConfigs.PER_STEP_TRAJECTORY
    scene.primitiveConfig = dfc.PrimitiveConfiguration.TABLE_HIGH
    # scene.primitiveConfig = dfc.PrimitiveConfiguration.NONE
    scene.windConfig = dfc.WindConfig.NO_WIND
    # scene.camPos = np.array([-90, 30, 60])
    # scene.camPos = np.array([0, 35.0, 0])
    scene.camPos = np.array([-21.67, 17.40, -10.67])
    # scene.camFocusPos = np.array([0, 12, 0])
    # scene.camPos = np.array([-21.67, 15.40, -10.67])
    scene.sockLegOrientation = np.array([0., 0., 0.])
    scene.camFocusPointType = dfc.CameraFocusPointType.CLOTH_CENTER
    # scene.sceneBbox = dfc.AABB(np.array([-70, -70, -70]), np.array([70, 70, 70]))
    # scene.sceneBbox = dfc.AABB(np.array([-7, 3, -7]), np.array([7, 17, 7]))
    scene.timeStep = 0.025
    scene.forwardConvergenceThresh = 1e-4
    scene.backwardConvergenceThresh = 5e-4
    scene.name = "Test scene"

    scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
    gp1_id = np.array([0])
    gp2_id = np.array([14])
    scene.customAttachmentVertexIdx = [(0, [0, 14])]
    # scene.stepNum = num_step
    scene.stepNum = 150

    return scene


def stepSim(simModule, sim, init, pos, helper):
    start = time.time()
    x = torch.tensor(sim.getStateInfo().x)
    v = torch.tensor(sim.getStateInfo().v)
    x, v = simModule(x, v, pos, torch.tensor([1, 10, 768.15, 0.28]))

    # x, v = simModule(x, v, pos, next_step=False, v)

    render = True
    if render:
        dfc.render(sim, renderPosPairs=True, autoExit=True)

    return x, v

def get_grasp_traj(init, target, length):
    traj = init + (target - init) * np.linspace(0, 1, length)[:, None]
    return torch.tensor(traj)

def upsample(marker, particle):

    """
    Input: - the marker points
           - the simulation point cloud

    Output: Point Cloud of marker points mixed into the simulation point cloud
    """

    # particle = particle + 0.001
    particle[:, 0] = marker[:, 0]
    particle[:, 7] = marker[:, 1]
    particle[:, 14] = marker[:, 2]
    particle[:, 84] = marker[:, 3]
    particle[:, 105] = marker[:, 4]
    particle[:, 119] = marker[:, 5]
    particle[:, 156] = marker[:, 6]
    particle[:, 210] = marker[:, 7]
    particle[:, 217] = marker[:, 8]
    particle[:, 224] = marker[:, 9]
    # print(marker[:, 7])
    # print(particle[:, 210])


    return particle

scene = get_default_scene()
sim = dfc.makeSimFromConf(scene)

sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
np.set_printoptions(precision=5)

dfc.enableOpenMP(n_threads=10)

helper = dfc.makeOptimizeHelper("inverse_design")
helper.taskInfo.dL_dx0 = True
helper.taskInfo.dL_density = False

# sim.forwardConvergenceThreshold = 1e-6

sim_mod = pySim(sim, helper, True)

def run_sim_visualize(step_num):
    scene = get_default_scene()
    scene.customAttachmentVertexIdx = [(0, [0, 14, 80, 82, 110, 112, 210, 224])]
    # scene.customAttachmentVertexIdx = [(0, [80, 82, 110, 112])]
    scene.stepNum = 57
    sim = dfc.makeSimFromConf(scene)

    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=50)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-5
    sim.backwardConvergenceThreshold = 5e-5

    sim.resetSystem()
    paramInfo = dfc.ParamInfo()
    x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/x_init_spread.npy')
    #
    paramInfo.x0 = x.flatten()
    sim.resetSystemWithParams(helper.taskInfo, paramInfo)
    # sim.resetSystem()

    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    position_control = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_spread_3.npy')
    print(position_control[0])
    if len(position_control) > step_num:
        position_control = position_control[:step_num]

    pos0 = torch.tensor(position_control[:, 0])
    pos1 = torch.tensor(position_control[:, 1])
    pos2 = torch.tensor(position_control[:, 2])
    pos3 = torch.tensor(position_control[:, 3])
    pos4 = torch.tensor(position_control[:, 4])
    pos5 = torch.tensor(position_control[:, 5])
    pos6 = torch.tensor(position_control[:, 6])
    pos7 = torch.tensor(position_control[:, 7])

    # position_control = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_spread_modified.npy')
    #
    # pos2 = torch.tensor(position_control[:, 0:3])
    # pos3 = torch.tensor(position_control[:, 3:6])
    # pos4 = torch.tensor(position_control[:, 6:9])
    # pos5 = torch.tensor(position_control[:, 9:12])


    if len(position_control) > step_num:
        position_control = position_control[:step_num]

    # db_pos = torch.cat((pos2, pos3, pos4, pos5), dim=1)
    db_pos = torch.cat((pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7), dim=1)
    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), db_pos, helper)

    scene.customAttachmentVertexIdx = [(0, [])]
    scene.stepNum = step_num - 57
    sim = dfc.makeSimFromConf(scene)

    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=50)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-6
    sim.backwardConvergenceThreshold = 5e-6

    sim.resetSystem()
    paramInfo = dfc.ParamInfo()
    x = x[-1]
    v = v[-1]
    # print(v)
    paramInfo.x0 = x.flatten()
    paramInfo.v0 = v.flatten()
    sim.resetSystemWithParams(helper.taskInfo, paramInfo)

    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), db_pos, helper)

    return x, v

def run_result(step_num):
    scene = get_default_scene()
    # scene.customAttachmentVertexIdx = [(0, [0, 14, 80, 82, 110, 112, 210, 224])]
    scene.customAttachmentVertexIdx = [(0, [80, 82, 110, 112])]
    scene.stepNum = 57
    sim = dfc.makeSimFromConf(scene)

    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=50)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-5
    sim.backwardConvergenceThreshold = 5e-5

    sim.resetSystem()
    paramInfo = dfc.ParamInfo()
    x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/x_init_spread.npy')
    #
    paramInfo.x0 = x.flatten()
    sim.resetSystemWithParams(helper.taskInfo, paramInfo)
    # sim.resetSystem()

    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    # position_control = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_spread_3.npy')
    # print(position_control[0])
    # if len(position_control) > step_num:
    #     position_control = position_control[:step_num]
    #
    # pos0 = torch.tensor(position_control[:, 0])
    # pos1 = torch.tensor(position_control[:, 1])
    # pos2 = torch.tensor(position_control[:, 2])
    # pos3 = torch.tensor(position_control[:, 3])
    # pos4 = torch.tensor(position_control[:, 4])
    # pos5 = torch.tensor(position_control[:, 5])
    # pos6 = torch.tensor(position_control[:, 6])
    # pos7 = torch.tensor(position_control[:, 7])

    position_control = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_spread_modified.npy')

    pos2 = torch.tensor(position_control[:, 0:3])
    pos3 = torch.tensor(position_control[:, 3:6])
    pos4 = torch.tensor(position_control[:, 6:9])
    pos5 = torch.tensor(position_control[:, 9:12])

    if len(position_control) > step_num:
        position_control = position_control[:step_num]

    db_pos = torch.cat((pos2, pos3, pos4, pos5), dim=1)
    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), db_pos, helper)

    scene.customAttachmentVertexIdx = [(0, [])]
    scene.stepNum = step_num - 57
    sim = dfc.makeSimFromConf(scene)

    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=50)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-6
    sim.backwardConvergenceThreshold = 5e-6

    sim.resetSystem()
    paramInfo = dfc.ParamInfo()
    x = x[-1]
    v = v[-1]
    # print(v)
    paramInfo.x0 = x.flatten()
    paramInfo.v0 = v.flatten()
    sim.resetSystemWithParams(helper.taskInfo, paramInfo)

    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), db_pos, helper)

    return x, v

def run_sim_target(step_num, target1, target2, target3, target4, target5, target6, target7, target8):
    # sim = dfc.makeSimFromConf(scene)

    scene = get_default_scene()

    scene.customAttachmentVertexIdx = [(0, [0, 14, 80, 82, 110, 112, 210, 224])]
    scene.stepNum = step_num
    sim = dfc.makeSimFromConf(scene)

    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=50)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-3

    sim.resetSystem()

    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    gp1_id = np.array([0])
    gp2_id = np.array([14])
    gp3_id = np.array([82])
    gp4_id = np.array([84])
    gp5_id = np.array([110])
    gp6_id = np.array([112])
    gp7_id = np.array([210])
    gp8_id = np.array([224])

    grasp_point_1 = pts[gp1_id].copy()
    grasp1_traj = get_grasp_traj(grasp_point_1, target1, (step_num))

    grasp_point_2 = pts[gp2_id].copy()
    grasp2_traj = get_grasp_traj(grasp_point_2, target2, (step_num))

    grasp_point_3 = pts[gp3_id].copy()
    grasp3_traj = get_grasp_traj(grasp_point_3, target3, (step_num))

    grasp_point_4 = pts[gp4_id].copy()
    grasp4_traj = get_grasp_traj(grasp_point_4, target4, (step_num))

    grasp_point_5 = pts[gp5_id].copy()
    grasp5_traj = get_grasp_traj(grasp_point_5, target5, (step_num))

    grasp_point_6 = pts[gp6_id].copy()
    grasp6_traj = get_grasp_traj(grasp_point_6, target6, (step_num))

    grasp_point_7 = pts[gp7_id].copy()
    grasp7_traj = get_grasp_traj(grasp_point_7, target7, (step_num))

    grasp_point_8 = pts[gp8_id].copy()
    grasp8_traj = get_grasp_traj(grasp_point_8, target8, (step_num))

    grasp_trajs = torch.cat((grasp1_traj, grasp2_traj, grasp3_traj, grasp4_traj, grasp5_traj, grasp6_traj, grasp7_traj, grasp8_traj), dim=1)
    # print(grasp_trajs.shape)
    empty = grasp_trajs[-1].reshape(1, -1)

    for i in range(150):
        grasp_trajs = torch.cat((grasp_trajs, empty), dim=0)

    print(grasp_trajs.shape)

    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), grasp_trajs, helper)

    return x, v

if __name__=="__main__":
    x, v = run_result(130)
    # x, v = run_sim_visualize(100)
    # init = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_spread.npy")[0]
    # print(init)
    # x, v = run_sim_target(300, [-0.29508, 12.5739, 0.72649], [-0.22013, 12.60516, 0.40306], [0.05515, 12.53844,0.83549], [ 0.07695, 12.6428, 0.4739])
    # x, v = run_sim_target(300, [-4.43617e-01, 1.05689e+01, 8.17779e-01], [3.31579e-03, 1.05914e+01, 0],[-0.29508, 12.5739, 0.72649], [-0.22013, 12.60516, 0.40306],
    #                       [0.05515, 12.53844, 0.83549], [0.07695, 12.6428, 0.4739], [-1.20677e-01, 1.05394e+01, 1.05336e+00], [4.71524e-01, 1.05676e+01, 4.92854e-01])
    # x_traj = np.zeros([200, 225, 3])
    #
    # for i in range(len(x)):
    #     x_traj[i] = x[i]
    # np.save("x_demo", x_traj)

    # run_sim_step2(x[-1], v[-1])
    # x = np.array(x[-1])
    # print(x.shape)
    # np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/x_init_spread.npy", x)
    # np.savetxt("x_init_dyndemo.txt", x)