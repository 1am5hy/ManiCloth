import numpy as np
import diffcloth_py as dfc

def get_default_fabric_config():
    fabric = dfc.FabricConfiguration()
    fabric.clothDimX = 3
    fabric.clothDimY = 3
    fabric.k_stiff_stretching = 70
    fabric.k_stiff_bending = .1
    fabric.gridNumX = 15
    fabric.gridNumY = 15
    fabric.density = 0.3
    fabric.keepOriginalScalePoint = True
    fabric.isModel = False
    fabric.custominitPos = True
    # fabric.initPosFile = "/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/x.txt"
    fabric.initPosFile = "/../../python_code/x_init_lift.txt"
    fabric.fabricIdx = 0
    fabric.color = np.array([0.9, 0., 0.1])
    fabric.name = "test"
    return fabric

def get_default_scene_config(fabric):
    scene = dfc.SceneConfiguration()
    scene.fabric = fabric
    scene.orientation = dfc.Orientation.DOWN
    scene.upVector = np.array([0, 1, 0])
    # scene.attachmentPoints = dfc.AttachmentConfigs.NO_ATTACHMENTS
    scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
    scene.customAttachmentVertexIdx = [(0.0, [0, 14])]
    # scene.trajectory = dfc.TrajectoryConfigs.CORNERS_2_WEARHAT
    scene.trajectory = dfc.TrajectoryConfigs.PER_STEP_TRAJECTORY
    scene.primitiveConfig = dfc.PrimitiveConfiguration.Y0PLANE
    scene.windConfig = dfc.WindConfig.NO_WIND
    # scene.camPos = np.array([-90, 30, 60])
    scene.camPos = np.array([-21.67, 15.40, -10.67])
    scene.sockLegOrientation = np.array([0., 0., 0.])
    scene.camFocusPointType = dfc.CameraFocusPointType.CLOTH_CENTER
    # scene.sceneBbox = dfc.AABB(np.array([-70, -70, -70]), np.array([70, 70, 70]))
    scene.sceneBbox = dfc.AABB(np.array([-7, 3, -7]), np.array([7, 17, 7]))
    scene.timeStep = 0.05
    scene.stepNum = 150
    scene.forwardConvergenceThresh = 1e-3
    scene.backwardConvergenceThresh = 1e-3
    scene.name = "Test scene"

    return scene