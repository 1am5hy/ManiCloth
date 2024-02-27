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
    fabric.k_stiff_stretching = 2442.31
    fabric.k_stiff_bending = 0.1
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
    scene.primitiveConfig = dfc.PrimitiveConfiguration.DISH
    scene.windConfig = dfc.WindConfig.NO_WIND
    # scene.camPos = np.array([-90, 30, 60])
    # scene.camPos = np.array([0, 35.0, 0])
    # scene.camPos = np.array([-21.67, 12.40, -10.67])
    # scene.camFocusPos = np.array([0, 12, 0])
    scene.camPos = np.array([-21.67, 15.40, -10.67])
    scene.sockLegOrientation = np.array([0., 0., 0.])
    scene.camFocusPointType = dfc.CameraFocusPointType.CLOTH_CENTER
    # scene.sceneBbox = dfc.AABB(np.array([-70, -70, -70]), np.array([70, 70, 70]))
    # scene.sceneBbox = dfc.AABB(np.array([-7, 3, -7]), np.array([7, 17, 7]))
    scene.timeStep = 0.025
    scene.forwardConvergenceThresh = 1e-9
    scene.backwardConvergenceThresh = 5e-9
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
    x, v = simModule(x, v, pos, torch.tensor([1, 10, 2442.31, 0.21]))

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
    scene.forwardConvergenceThresh = 1e-9
    # scene.customAttachmentVertexIdx = [(0, [0, 14, 80, 82, 110, 112, 210, 224])]
    scene.customAttachmentVertexIdx = [(0, [80, 82, 110, 112])]
    # scene.customAttachmentVertexIdx = [(0, [0, 14, 210, 224])]
    scene.stepNum = 101
    sim = dfc.makeSimFromConf(scene)

    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=50)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-9

    sim.resetSystem()
    paramInfo = dfc.ParamInfo()
    x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/x_init_dish.npy')
    #
    paramInfo.x0 = x.flatten()
    sim.resetSystemWithParams(helper.taskInfo, paramInfo)
    # sim.resetSystem()

    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    position_control = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_dish_3.npy')
    print(position_control[-1])
    if len(position_control) > step_num:
        position_control = position_control[:step_num]

    # pos0 = torch.tensor(position_control[:, 0])
    # pos1 = torch.tensor(position_control[:, 1])
    pos2 = torch.tensor(position_control[:, 2])
    pos3 = torch.tensor(position_control[:, 3])
    pos4 = torch.tensor(position_control[:, 4])
    pos5 = torch.tensor(position_control[:, 5])
    # pos6 = torch.tensor(position_control[:, 6])
    # pos7 = torch.tensor(position_control[:, 7])
    position_control = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_dish_modified.npy')
    pos2 = torch.tensor(position_control[:, 0:3])
    pos3 = torch.tensor(position_control[:, 3:6])
    pos4 = torch.tensor(position_control[:, 6:9])
    pos5 = torch.tensor(position_control[:, 9:12])

    # db_pos = torch.cat((pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7), dim=1)
    db_pos = torch.cat((pos2, pos3, pos4, pos5), dim=1)
    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), db_pos, helper)

    return x, v

def run_result(step_num):
    scene = get_default_scene()
    scene.customAttachmentVertexIdx = [(0, [0, 14])]
    scene.stepNum = step_num
    sim = dfc.makeSimFromConf(scene)


    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=10)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-4

    # sim.resetSystem()
    paramInfo = dfc.ParamInfo()
    x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/x_init_dyndemo.npy')

    paramInfo.x0 = x.flatten()
    sim.resetSystemWithParams(helper.taskInfo, paramInfo)
    # sim.resetSystem()


    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    position_control = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_hang.npy')
    print(position_control[0])
    print(x[0])
    print(x[14])
    if len(position_control) > step_num:
        position_control = position_control[:step_num]

    pos0 = torch.tensor(position_control[:, :3])
    pos1 = torch.tensor(position_control[:, 3:])

    db_pos = torch.cat((pos0, pos1), dim=1)

    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), db_pos, helper)

    x_sim = np.zeros([step_num, 225, 3])

    for i in range(len(x)):
        x_sim[i] = x[i]
        # print(x[i].shape)

    # mark = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/OAMP/demo/dyn_sim_obs.npy")
    markers_traj = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/marker_hang.npy")

    markers_traj = torch.tensor(markers_traj[:150]).float()
    x = x_sim.copy()

    markers = upsample(markers_traj, x)

    x_sim = torch.tensor(x_sim)
    markers = torch.tensor(markers)
    marker_loss = markers - x_sim

    y_sim = x_sim[:, 150:, 1].reshape(-1)
    y_marker = markers[:, 150:, 1].reshape(-1)
    # print(max(y_sim))
    # print(max(y_marker))


    loss = 1 * torch.nn.functional.mse_loss(markers, x_sim, reduction='sum')
    print(loss)

    return x, v

def run_sim_target(step_num, target1, target2, target3, target4):
    # sim = dfc.makeSimFromConf(scene)

    scene = get_default_scene()

    # scene.customAttachmentVertexIdx = [(0, [80, 82, 110, 112])]
    scene.customAttachmentVertexIdx = [(0, [0, 14, 210, 224])]
    scene.stepNum = step_num
    sim = dfc.makeSimFromConf(scene)

    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=50)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-4

    sim.resetSystem()

    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    gp1_id = np.array([80])
    gp2_id = np.array([82])
    gp3_id = np.array([110])
    gp4_id = np.array([112])

    grasp_point_1 = pts[gp1_id].copy()
    grasp1_traj = get_grasp_traj(grasp_point_1, target1, (step_num))

    grasp_point_2 = pts[gp2_id].copy()
    grasp2_traj = get_grasp_traj(grasp_point_2, target2, (step_num))

    grasp_point_3 = pts[gp3_id].copy()
    grasp3_traj = get_grasp_traj(grasp_point_3, target3, (step_num))

    grasp_point_4 = pts[gp4_id].copy()
    grasp4_traj = get_grasp_traj(grasp_point_4, target4, (step_num))

    grasp_trajs = torch.cat((grasp1_traj, grasp2_traj, grasp3_traj, grasp4_traj), dim=1)
    print(grasp_trajs.shape)
    empty = grasp_trajs[-1].reshape(1, -1)

    for i in range(150):
        grasp_trajs = torch.cat((grasp_trajs, empty), dim=0)

    print(grasp_trajs.shape)

    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), grasp_trajs, helper)

    return x, v

if __name__=="__main__":
    # x, v = run_result(150)
    x, v = run_sim_visualize(101)
    # init = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/gp_spread.npy")[0]
    # print(init)
    # x, v = run_sim_target(300, [-0.29508, 12.5739, 0.72649], [-0.22013, 12.60516, 0.40306], [0.05515, 12.53844,0.83549], [ 0.07695, 12.6428, 0.4739])
    # x, v = run_sim_target(300, [3, 0, 0], [ 3, 0, 3],
    #                       [0, 0, 0], [0, 0, 3])
    # x_traj = np.zeros([200, 225, 3])
    #
    # for i in range(len(x)):
    #     x_traj[i] = x[i]
    # np.save("x_demo", x_traj)

    # run_sim_step2(x[-1], v[-1])
    # x = np.array(x[-1])
    # print(x.shape)
    # np.save("/home/ubuntu/Github/DiffCloth/src/python_code/DataSort/npfiles/x_init_dish.npy", x)
    # np.savetxt("x_init_dyndemo.txt", x)