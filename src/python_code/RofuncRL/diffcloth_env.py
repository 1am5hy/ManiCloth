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
    scene.upVector = np.array([0, 1, 0])
    scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
    # scene.customAttachmentVertexIdx = [(0.0, [0, 24])]
    # scene.customAttachmentVertexIdx = [(0.0, [1838])]
    scene.trajectory = dfc.TrajectoryConfigs.PER_STEP_TRAJECTORY
    scene.primitiveConfig = dfc.PrimitiveConfiguration.NONE
    scene.windConfig = dfc.WindConfig.NO_WIND
    # scene.camPos = np.array([-90, 30, 60])
    scene.camPos = np.array([-12.67, 12, 13.67])
    scene.camFocusPos = np.array([0, 12, 0])
    # scene.camPos = np.array([-21.67, 15.40, -10.67])
    scene.sockLegOrientation = np.array([0., 0., 0.])
    scene.camFocusPointType = dfc.CameraFocusPointType.POINT
    # scene.sceneBbox = dfc.AABB(np.array([-70, -70, -70]), np.array([70, 70, 70]))
    # scene.sceneBbox = dfc.AABB(np.array([-7, 3, -7]), np.array([7, 17, 7]))
    scene.timeStep = 0.05
    scene.forwardConvergenceThresh = 1e-6
    scene.backwardConvergenceThresh = 5e-4
    scene.name = "Test scene"

    scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
    gp1_id = np.array([0])
    gp2_id = np.array([14])
    scene.customAttachmentVertexIdx = [(0, [0, 14])]
    # scene.stepNum = num_step
    scene.stepNum = 50

    return scene

def stepSim(simModule, sim, init, pos, helper):
    start = time.time()
    x = torch.tensor(sim.getStateInfo().x)
    v = torch.tensor(sim.getStateInfo().v)
    x, v = simModule(x, v, pos, )

    # x, v = simModule(x, v, pos, next_step=False, v)

    render = True
    if render:
        dfc.render(sim, renderPosPairs=True, autoExit=True)

    return x, v

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
step_num = 150

def run_result(action):
    scene = get_default_scene()
    scene.customAttachmentVertexIdx = [(0, [0, 14])]
    scene.stepNum = step_num
    sim = dfc.makeSimFromConf(scene)

    sim.gradientClippingThreshold, sim.gradientClipping = 200.0, False
    np.set_printoptions(precision=5)
    dfc.enableOpenMP(n_threads=10)
    sim_mod = pySim(sim, helper, True)

    sim.forwardConvergenceThreshold = 1e-7

    sim.resetSystem()
    paramInfo = dfc.ParamInfo()
    x = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/x_init_dyndemo.npy')

    paramInfo.x0 = x.flatten()
    sim.resetSystemWithParams(helper.taskInfo, paramInfo)
    # sim.resetSystem()


    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    gp1_id = np.array([1])
    gp2_id = np.array([14])
    grasp_point_1 = pts[gp1_id].copy()
    grasp_point_2 = pts[gp2_id].copy()
    init = torch.cat(grasp_point_1, grasp_point_2)

    position_control = get_grasp_traj(init, action)

    pos0 = torch.tensor(position_control[:, :3])
    pos1 = torch.tensor(position_control[:, 3:])

    db_pos = torch.cat((pos0, pos1), dim=1)

    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x), torch.tensor(state_info_init.v)), db_pos, helper)

    return x, v

def get_grasp_traj(init, target):
    # print(init)
    # print(target)
    length = 1
    traj = init + (target * np.linspace(0, 1, length)[:, None])

    return torch.tensor(traj)