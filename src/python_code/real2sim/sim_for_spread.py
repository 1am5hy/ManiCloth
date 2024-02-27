import torch
import torch.nn as nn
import numpy as np
import diffcloth_py as dfc

from sim_params import *
from sim_func_spread import SimFunction

class SimDeform(nn.Module):
    def __init__(self,
                 n_time_steps,
                 time_step,
                 n_threads,
                 feat_size,
                 gridNum,
                 scale=1e2,
                 model_file=None
                ):
        super().__init__()

        self.time_step = time_step
        self.n_time_steps = n_time_steps
        self.gridNumX, self.gridNumY = gridNum
        # self.phy_params = nn.Parameter(torch.tensor([70., 10000, 3000., 0.1, .24]))
        # phy_scales = torch.tensor([1e2, 1e5, 1e2, 1e1, 1e0])
        # self.phy_params = nn.Parameter(torch.tensor([1, 10000, 200., 5., 1.]) + 2 * (torch.rand(5) - 0.5))
        # self.local_features = nn.Parameter(2 * torch.rand((self.gridNumX * self.gridNumY, feat_size)) - 1)

        # phy_scales = torch.tensor([1e2, 1e2, 8e-1, 5e-2, 1e-1])
        # self.phy_params = nn.Parameter(torch.log(torch.tensor([1, 10, 29.31, 1.01, 0.3]) + (torch.rand(5) - 0.5)*phy_scales))
        phy_scales = torch.tensor([1e2, 1e2, 8e0, 5e-1, 1e-1])
        # self.phy_params = nn.Parameter(torch.log(torch.tensor([1, 10, 2442.31, 0.21, 0.3]) + (torch.rand(5) - 0.5) * phy_scales))
        self.phy_params = nn.Parameter(torch.log(torch.tensor([1, 10, 1000, 0, 0.3]) + (torch.rand(5) - 0.5) * phy_scales))
        # self.scale = float(scale)
        self.model_file = model_file
        # if self.model_file is not False:
        #     self.local_features = nn.Parameter(2 * torch.rand((289, feat_size)) - 1)
        # else:
        #     self.local_features = nn.Parameter(2 * torch.rand((self.gridNumX * self.gridNumY, feat_size)) - 1)

        dfc.enableOpenMP(n_threads=n_threads)

        self.helper = dfc.makeOptimizeHelper("inverse_design")
        self.helper.taskInfo.dL_dx0 = True
        self.helper.taskInfo.dL_dmu = False
        self.helper.taskInfo.dL_density = False
        self.helper.taskInfo.set_dL_dk_pertype(False, False, True, True)
        # self.offsets = [0, 0]

        self.sim_func = SimFunction()

        self.scale_mat = None


    def forward(self, coordinates, position_control, decay):

        "Coordinates means the initial position of all the particles"
        "Position Control means the trajectory"
        "Play with Decay\ value"

        print(f"Phy params: {self.phy_params}")
        # print(self.phy_params.grad)

        fabric = get_default_fabric_config()
        phy_params = torch.exp(self.phy_params)

        fabric.k_stiff_stretching = phy_params[2].item()
        fabric.k_stiff_bending = phy_params[3].item()

        fabric.density = 0.3
        fabric.custominitPos = False
        # print(fabric.density)
        # Adapt resolution
        fabric.clothDimX = 3
        fabric.clothDimY = 3
        fabric.gridNumX = 15
        fabric.gridNumY = 15

        scene = get_default_scene_config(fabric)
        scene.timeStep = self.time_step
        scene.stepNum = self.n_time_steps
        scene.forwardConvergenceThresh = 1e-5
        scene.backwardConvergenceThresh = 5e-5

        position_control = position_control
        pos0 = torch.tensor(position_control[:, :3])
        pos1 = torch.tensor(position_control[:, 3:6])
        pos2 = torch.tensor(position_control[:, 6:9])
        pos3 = torch.tensor(position_control[:, 9:12])
        db_pos = torch.cat((pos0, pos1, pos2, pos3), dim=1)

        scene.primitiveConfig = dfc.PrimitiveConfiguration.TABLE_HIGH
        # scene.customAttachmentVertexIdx = [(0.0, [torch.argmin(torch.norm(coordinates - position_control[0], dim=1)).item()])]
        # scene.customAttachmentVertexIdx = [(0.0, [torch.argmin(torch.norm(coordinates - db_pos[0][:3], dim=1)).item(),
        #                                           torch.argmin(torch.norm(coordinates - db_pos[0][3:], dim=1)).item()
        #                                    ])]

        scene.stepNum = 57
        scene.customAttachmentVertexIdx = [(0.0, [80, 82, 110, 112])]
        sim = dfc.makeSimFromConf(scene)
        sim.gradientClippingThreshold, sim.gradientClipping = 100.0, True

        coordinates = torch.tensor(coordinates)
        paramInfo = dfc.ParamInfo()
        paramInfo.x0 = coordinates.flatten().detach().cpu().numpy()

        paramInfo.density = 0.3
        paramInfo.set_k_pertype(phy_params[0].item(), phy_params[1].item(), phy_params[2].item(), phy_params[3].item())
        sim.resetSystemWithParams(self.helper.taskInfo, paramInfo)

        sim.forwardConvergenceThreshold = 1e-6
        sim.backwardConvergenceThreshold = 5e-6

        state_info_init = sim.getStateInfo()

        x = torch.tensor(state_info_init.x)
        # print(x.shape)
        v = torch.tensor(state_info_init.v)
        x, v = self.sim_func.apply(x, v, db_pos,
                                   phy_params[:4], phy_params[4], sim, self.helper, decay)
        # dfc.render(sim, renderPosPairs=True, autoExit=True)
        x_1 = x.clone()

        scene.stepNum = 75 - 57
        scene.customAttachmentVertexIdx = [(0.0, [])]
        sim = dfc.makeSimFromConf(scene)
        sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False

        coordinates = torch.tensor(coordinates)
        paramInfo = dfc.ParamInfo()
        paramInfo.x0 = x[-1].flatten().detach().cpu().numpy()
        paramInfo.v0 = v[-1].flatten().detach().cpu().numpy()
        paramInfo.density = 0.3
        paramInfo.set_k_pertype(phy_params[0].item(), phy_params[1].item(), phy_params[2].item(), phy_params[3].item())
        sim.resetSystemWithParams(self.helper.taskInfo, paramInfo)

        x, v = self.sim_func.apply(x[-1].flatten().detach().cpu(), v[-1].flatten().detach().cpu(), db_pos,
                                   phy_params[:4], phy_params[4], sim, self.helper, decay)

        # dfc.render(sim, renderPosPairs=True, autoExit=True)

        x = torch.cat((x_1, x), dim=0)

        return x
#
# if __name__ == '__main__':
#
#     run = Runner("/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/cloth.conf", case='cloth')
#     run.train()