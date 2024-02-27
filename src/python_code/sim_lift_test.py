import random
import time
import torch
import numpy as np
import diffcloth_py as dfc

from pySim.pySim import pySim
from pySim.functional import SimFunction


def get_default_scene():
    fabric = dfc.FabricConfiguration()
    fabric.clothDimX = 3
    fabric.clothDimY = 3
    # fabric.k_stiff_stretching = 2500
    fabric.k_stiff_stretching = 10000
    # fabric.k_stiff_bending = 1.5
    fabric.k_stiff_bending = 0.01
    fabric.gridNumX = 15
    fabric.gridNumY = 15
    fabric.density = 0.3
    fabric.keepOriginalScalePoint = True
    fabric.isModel = False
    fabric.custominitPos = False
    # fabric.initPosFile = "/home/ubuntu/Github/DiffCloth/src/python_code/x_init_lift.txt"
    fabric.fabricIdx = 0
    fabric.color = np.array([0.9, 0., 0.1])
    fabric.name = "test"

    scene = dfc.SceneConfiguration()
    scene.fabric = fabric
    scene.orientation = dfc.Orientation.DOWN
    scene.upVector = np.array([0, 0, 1])
    scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
    # scene.customAttachmentVertexIdx = [(0.0, [0, 24])]
    # scene.customAttachmentVertexIdx = [(0.0, [1838])]
    scene.trajectory = dfc.TrajectoryConfigs.NO_TRAJECTORY
    scene.primitiveConfig = dfc.PrimitiveConfiguration.Y0PLANE
    scene.windConfig = dfc.WindConfig.NO_WIND
    # Back
    # scene.camPos = np.array([11.67, 14, -15.67])
    # Front
    scene.camPos = np.array([-11.67, 14, 15.67])
    # scene.camPos = np.array([-21.67, 15.40, -10.67])
    # scene.camPos = np.array([-21.67, 15.4, -10.67])
    scene.sockLegOrientation = np.array([0., 0., 0.])
    scene.camFocusPointType = dfc.CameraFocusPointType.CLOTH_CENTER
    # scene.sceneBbox = dfc.AABB(np.array([-70, -70, -70]), np.array([70, 70, 70]))
    # scene.sceneBbox = dfc.AABB(np.array([-7, 3, -7]), np.array([7, 17, 7]))
    scene.timeStep = 0.05
    scene.forwardConvergenceThresh = 1e-3
    scene.backwardConvergenceThresh = 5e-5
    scene.name = "Test scene"

    # scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
    # gp1_id = np.array([0])
    # gp2_id = np.array([14])
    # scene.customAttachmentVertexIdx = [(0, [0, 14])]
    # scene.stepNum = num_step
    scene.stepNum = 50

    return scene


def stepSim(simModule, sim, init, pos, phy_params, helper):
    start = time.time()
    x = torch.tensor(sim.getStateInfo().x)
    v = torch.tensor(sim.getStateInfo().v)

    x, v = simModule.apply(x, v, pos, phy_params, sim, helper)

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
    particle[:, 14] = marker[:, 1]
    particle[:, 112] = marker[:, 2]
    particle[:, 210] = marker[:, 3]
    particle[:, 224] = marker[:, 4]
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

sim.forwardConvergenceThreshold = 1e-6

sim_mod = pySim(sim, helper, True)

def run_sim_visualize(step_num):
    scene = get_default_scene()
    scene.forwardConvergenceThresh = 1e-5
    scene.backwardConvergenceThresh = 5e-5
    scene.customAttachmentVertexIdx = [(0, [0, 14, 112, 210, 224])]
    scene.stepNum = step_num
    sim = dfc.makeSimFromConf(scene)

    sim.gradientClippingThreshold, sim.gradientClipping = 200.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=10)
    # sim_mod = pySim(sim, helper, True)\
    sim_mod = SimFunction()

    sim.forwardConvergenceThreshold = 1e-5
    sim.backwardConvergenceThreshold = 5e-5

    # sim.resetSystem()
    paramInfo = dfc.ParamInfo()
    x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/x_init_lift.npy')
    # print(x[0])
    phy_params = torch.tensor([1, 10000, 2000, 0.05])
    paramInfo.x0 = x.flatten()
    paramInfo.set_k_pertype(phy_params[0], phy_params[1], phy_params[2], phy_params[3])
    sim.resetSystemWithParams(helper.taskInfo, paramInfo)

    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    position_control = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/marker_path_lifting.npy')
    # print(position_control[0, 0])

    if len(position_control) > step_num:
        position_control = position_control[:step_num]
    #
    pos0 = torch.tensor(position_control[:, 0])
    pos1 = torch.tensor(position_control[:, 1])
    pos2 = torch.tensor(position_control[:, 2])
    pos3 = torch.tensor(position_control[:, 3])
    pos4 = torch.tensor(position_control[:, 4])

    db_pos = torch.cat((pos0, pos1, pos2, pos3, pos4), dim=1)
    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), db_pos, phy_params, helper)

    return x, v

def run_result(step_num):
    scene = get_default_scene()
    scene.customAttachmentVertexIdx = [(0, [112])]
    scene.stepNum = step_num
    scene.forwardConvergenceThresh = 1e-8
    scene.backwardConvergenceThresh = 5e-5
    sim = dfc.makeSimFromConf(scene)
    sim_mod = SimFunction()

    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=10)
    # sim_mod = pySim(sim, helper, True)


    sim.forwardConvergenceThreshold = 1e-6
    sim.backwardConvergenceThreshold = 5e-5


    # sim.resetSystem()
    paramInfo = dfc.ParamInfo()

    x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/x_init_lift.npy')

    paramInfo.x0 = x.flatten()
    phy_params = torch.tensor([1, 10000, 800, 0.01])
    paramInfo.set_k_pertype(phy_params[0], phy_params[1], phy_params[2], phy_params[3])
    sim.resetSystemWithParams(helper.taskInfo, paramInfo)
    # sim.resetSystem()


    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    position_control = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/gp_lifting.npy')
    if len(position_control) > step_num:
        position_control = position_control[:step_num]

    pos0 = torch.tensor(position_control[:, :3])
    # db_pos = torch.cat((pos0, pos1), dim=1)

    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), pos0, phy_params,helper)

    x_sim = np.zeros([step_num, 225, 3])

    for i in range(len(x)):
        x_sim[i] = x[i]
        # print(x[i].shape)

    markers_traj = np.load("/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/marker_path_lifting.npy")
    # print(markers_traj[0, 0])
    markers_traj = torch.tensor(markers_traj).float()
    x = x_sim.copy()
    markers = upsample(markers_traj, x)

    x_sim = torch.tensor(x_sim)
    markers = torch.tensor(markers)
    marker_loss = markers - x_sim

    y_sim = x_sim[:, 210:, 1].reshape(-1)
    y_marker = markers[:, 210:, 1].reshape(-1)
    # print(max(y_sim))
    # print(max(y_marker))


    loss = 1 * torch.nn.functional.mse_loss(markers, x_sim, reduction='sum')
    print(loss)

    return x, v

def run_sim_target(step_num, target1, target2, target3, target4, target5):
    # sim = dfc.makeSimFromConf(scene)

    scene = get_default_scene()

    scene.customAttachmentVertexIdx = [(0, [0, 14, 112, 210, 224])]
    scene.stepNum = step_num
    sim = dfc.makeSimFromConf(scene)

    sim.gradientClippingThreshold, sim.gradientClipping = 300.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=5)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-3


    paramInfo = dfc.ParamInfo()
    # x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/x_init_lift.npy')
    #
    # paramInfo.x0 = x.flatten()
    # sim.resetSystemWithParams(helper.taskInfo, paramInfo)
    sim.resetSystem()
    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    gp1_id = np.array([7])
    gp2_id = np.array([14])
    gp3_id = np.array([112])
    gp4_id = np.array([210])
    gp5_id = np.array([224])

    grasp_point_1 = pts[gp1_id].copy()
    grasp1_traj = get_grasp_traj(grasp_point_1, target1, step_num)

    grasp_point_2 = pts[gp2_id].copy()
    grasp2_traj = get_grasp_traj(grasp_point_2, target2, step_num)

    grasp_point_3 = pts[gp3_id].copy()
    grasp3_traj = get_grasp_traj(grasp_point_3, target3, step_num)

    grasp_point_4 = pts[gp4_id].copy()
    grasp4_traj = get_grasp_traj(grasp_point_4, target4, step_num)

    grasp_point_5 = pts[gp5_id].copy()
    grasp5_traj = get_grasp_traj(grasp_point_5, target5, step_num)
    
    grasp_trajs = torch.cat((grasp1_traj, grasp2_traj, grasp3_traj, grasp4_traj, grasp5_traj), dim=1)
    # print(grasp_trajs.shape)

    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), grasp_trajs, helper)

    return x, v

if __name__=="__main__":
    x, v = run_result(200)
    # x, v = run_sim_visualize(200)
    # x, v = run_sim_target(200, [0, 0, 0], [3, 0, 0], [1.5, 0, 1.5], [0, 0, 3], [3, 0, 3])

    # x_traj = np.zeros([200, 225, 3])
    #
    # for i in range(len(x)):
    #     x_traj[i] = x[i]
    # np.save("x_demo", x_traj)

    # run_sim_step2(x[-1], v[-1])
    x = np.array(x[1])
    # np.save("x_init_lift", x)
    # np.savetxt("x_init_lift.txt", x)