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
    fabric.k_stiff_stretching = 689.16
    fabric.k_stiff_bending = 0.09
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
    scene.upVector = np.array([0, 1, 0])
    scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
    # scene.customAttachmentVertexIdx = [(0.0, [0, 24])]
    # scene.customAttachmentVertexIdx = [(0.0, [1838])]
    scene.trajectory = dfc.TrajectoryConfigs.PER_STEP_TRAJECTORY
    scene.primitiveConfig = dfc.PrimitiveConfiguration.BIG_SPHERE
    # scene.primitiveConfig = dfc.PrimitiveConfiguration.NONE
    # sce
    # scene.primitive.center = dfc.Primitive.center = np.array([100, 100, 100])
    # scene.primitiveConfig = dfc.Primitive
    scene.windConfig = dfc.WindConfig.NO_WIND
    # scene.camPos = np.array([-90, 30, 60])
    # scene.camPos = np.array([-21.67, 12.40, -10.67])
    scene.camPos = np.array([-2, 16.40, -20])
    # scene.camPos = np.array([-5, 7, -15.67])
    # scene.camFocusPos = np.array([0, 12, 0])
    # scene.camPos = np.array([-21.67, 15.40, -10.67])
    scene.sockLegOrientation = np.array([0., 0., 0.])
    scene.camFocusPointType = dfc.CameraFocusPointType.POINT
    # scene.camFocusPos = np.array([0, 7, -10])
    scene.camFocusPos = np.array([-5, 8.40, 0])
    # scene.sceneBbox = dfc.AABB(np.array([-70, -70, -70]), np.array([70, 70, 70]))
    # scene.sceneBbox = dfc.AABB(np.array([-7, 3, -7]), np.array([7, 17, 7]))
    scene.timeStep = 0.025
    scene.forwardConvergenceThresh = 1e-5
    scene.backwardConvergenceThresh = 5e-5
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
    x, v = simModule(x, v, pos, torch.tensor([([1, 10, 2442.31, 0.1])]))

    # x, v = simModule(x, v, pos, next_step=False, v)

    render = True
    if render:
        dfc.render(sim, renderPosPairs=True, autoExit=True)

    return x, v

def get_grasp_traj(init, target, length):
    traj = init + (target - init) * np.linspace(0, 1, length)[:, None]
    return torch.tensor(traj)

def upsample(marker, particle, particle_idx):

    """
    Input: - the marker points
           - the simulation point cloud

    Output: Point Cloud of marker points mixed into the simulation point cloud
    """
    particle_idx = np.array(particle_idx)
    # particle = particle + 0.001
    particle[:, particle_idx[0]] = marker[:, 0]
    particle[:, particle_idx[1]] = marker[:, 1]
    particle[:, particle_idx[2]] = marker[:, 2]
    particle[:, particle_idx[3]] = marker[:, 3]
    particle[:, particle_idx[4]] = marker[:, 4]
    particle[:, particle_idx[5]] = marker[:, 5]
    particle[:, particle_idx[6]] = marker[:, 6]
    particle[:, particle_idx[7]] = marker[:, 7]
    particle[:, particle_idx[8]] = marker[:, 8]
    particle[:, particle_idx[9]] = marker[:, 9]
    # print(marker[:, 7])
    # print(particle[:, 210])


    return particle

scene = get_default_scene()
sim = dfc.makeSimFromConf(scene)

sim.gradientClippingThreshold, sim.gradientClipping = 100.0, True
np.set_printoptions(precision=5)

dfc.enableOpenMP(n_threads=10)

helper = dfc.makeOptimizeHelper("inverse_design")
helper.taskInfo.dL_dx0 = True
helper.taskInfo.dL_density = False

# sim.forwardConvergenceThreshold = 1e-6

sim_mod = pySim(sim, helper, True)

def run_sim_visualize(step_num, material, demo_id, particle_idx):
    scene = get_default_scene()

    scene.customAttachmentVertexIdx = [(0, particle_idx)]
    scene.stepNum = step_num
    sim = dfc.makeSimFromConf(scene)

    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, True
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=50)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-3

    # sim.resetSystem()
    paramInfo = dfc.ParamInfo()
    x = np.load('/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/x_init_hang.npy')

    paramInfo.x0 = x.flatten()
    sim.resetSystemWithParams(helper.taskInfo, paramInfo)

    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    load_pth = "/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/marker_{}_hang_demo_{}_3.npy".format(material, demo_id)
    position_control = np.load(load_pth)
    print(position_control.shape)
    if len(position_control) > step_num:
        position_control = position_control[:step_num]
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

    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), db_pos, helper)

    return x, v

def run_result(step_num, material, demo_id, particle_idx):
    scene = get_default_scene()
    scene.customAttachmentVertexIdx = [(0, [0, 14])]
    scene.stepNum = step_num
    # dfc.Primitive.center = np.array([100, 100, 100])
    sim = dfc.makeSimFromConf(scene)
    # sim.primitives[0].center = np.array([100, 100, 100])
    # dfc.Primitive.primitives.type

    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, True
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=10)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-6
    sim.backwardConvergenceThreshold = 5e-6

    # sim.resetSystem()
    paramInfo = dfc.ParamInfo()
    init_load_pth = "/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/x_{}_hang_init_{}.npy".format(material, demo_id)
    x = np.load(init_load_pth)

    paramInfo.x0 = x.flatten()
    sim.resetSystemWithParams(helper.taskInfo, paramInfo)
    # sim.resetSystem()


    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    gp_load_pth = "/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/gp_{}_hang_demo_{}.npy".format(material, demo_id)
    position_control = np.load(gp_load_pth)

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

    marker_load_pth = "/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/marker_{}_hang_demo_{}_3.npy".format(material, demo_id)
    markers_traj = np.load(marker_load_pth)

    markers_traj = torch.tensor(markers_traj[:150]).float()
    x = x_sim.copy()

    markers = upsample(markers_traj, x, particle_idx)

    x_sim = torch.tensor(x_sim)
    markers = torch.tensor(markers)
    marker_loss = markers - x_sim

    y_sim = x_sim[:, 150:, 1].reshape(-1)
    y_marker = markers[:, 150:, 1].reshape(-1)

    loss = 1 * torch.nn.functional.mse_loss(markers, x_sim, reduction='sum')
    print("The loss is:", loss)
    print(loss)

    # scene.customAttachmentVertexIdx = [(0, [])]
    # scene.stepNum = 150
    # # scene.s
    # sim = dfc.makeSimFromConf(scene)
    #
    # sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    # np.set_printoptions(precision=5)
    # dfc.enableOpenMP(n_threads=50)
    # sim_mod = pySim(sim, helper, True)
    #
    # sim.forwardConvergenceThreshold = 1e-6
    # sim.backwardConvergenceThreshold = 5e-6
    #
    # sim.resetSystem()
    # paramInfo = dfc.ParamInfo()
    # x = x[-1]
    # v = v[-1]
    # # print(v)
    # paramInfo.x0 = x.flatten()
    # paramInfo.v0 = v.flatten()
    # sim.resetSystemWithParams(helper.taskInfo, paramInfo)
    #
    # x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), db_pos, helper)

    return x, v

def run_sim_target(step_num, array, particle_idx):
    # sim = dfc.makeSimFromConf(scene)

    scene = get_default_scene()

    scene.customAttachmentVertexIdx = [(0, particle_idx)]
    # scene.customAttachmentVertexIdx = [(0, [0, 14, 210, 224])]
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

    target1 = array[0:3]
    target2 = array[3:6]
    target3 = array[6:9]
    target4 = array[9:12]
    target5 = array[12:15]
    target6 = array[15:18]
    target7 = array[18:21]
    target8 = array[21:24]
    target9 = array[24:27]
    target10 = array[27:30]
    # target1 = array[0:3]
    # target3 = array[3:6]
    # target8 = array[6:9]
    # target10 = array[9:12]

    gp1_id = np.array([0])
    gp2_id = np.array([7])
    gp3_id = np.array([14])
    gp4_id = np.array([49])
    gp5_id = np.array([90])
    gp6_id = np.array([104])
    gp7_id = np.array([144])
    gp8_id = np.array([210])
    gp9_id = np.array([217])
    gp10_id = np.array([224])

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

    grasp_point_9 = pts[gp9_id].copy()
    grasp9_traj = get_grasp_traj(grasp_point_9, target9, (step_num))

    grasp_point_10 = pts[gp10_id].copy()
    grasp10_traj = get_grasp_traj(grasp_point_10, target10, (step_num))

    grasp_trajs = torch.cat(
        (grasp1_traj, grasp2_traj, grasp3_traj, grasp4_traj, grasp5_traj, grasp6_traj, grasp7_traj, grasp8_traj,
         grasp9_traj, grasp10_traj), dim=1)
    # grasp_trajs = torch.cat(
    #     (grasp1_traj, grasp3_traj, grasp8_traj, grasp10_traj), dim=1)
    empty = grasp_trajs[-1].reshape(1, -1)

    for i in range(2000):
        grasp_trajs = torch.cat((grasp_trajs, empty), dim=0)

    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), grasp_trajs,
                   helper)
    return x, v

if __name__=="__main__":
    material = "cotton"
    idx_all = np.load("/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/idx_all_{}_hang.npy".format(material))

    for i in range(1):
        # demo_id = i + 1
        demo_id = 6
        particle_idx = idx_all[i]

        target_load_pth = "/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/marker_{}_hang_demo_{}_30.npy".format(material, demo_id)
        target = np.load(target_load_pth)[0]
        # print(target)
        #
        # target = np.array([0, 8., 0, 0, 8, 3, 0, 6, 0, 0, 6, 3.])
        # print(target)

        x, v = run_sim_visualize(150, material, demo_id, particle_idx)

        x, v = run_sim_target(500, target, particle_idx)
        x = np.array(x[-1])
        init_save_pth = "/home/ubuntu/Github/ManiCloth/src/python_code/DataSort/npfiles/x_{}_hang_init_{}.npy".format(material, demo_id)
        np.save(init_save_pth, x)

        x, v = run_result(150, material, demo_id, particle_idx)