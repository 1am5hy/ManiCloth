# from typing import Any, Optional, Mapping, Tuple
#
# import torch
# import torch.autograd as autograd
# from torch import Tensor
#
#
# import diffcloth_py as dfc
# import numpy as np
# from numpy import linalg
# import time
#
# from diffcloth_py import Simulation, ForwardInformation
#
#
# class SimFunction(autograd.Function):
#
#     @staticmethod
#     def forward(
#             ctx: Any,
#             x0: Tensor,
#             v0: Tensor,
#             ctrl_traj: Tensor,
#             k_pertype: Tensor,
#             density: Tensor,
#             cppSim: dfc.Simulation,
#             helper: dfc.OptimizeHelper,
#             decay: float
#     ) -> Tuple[Tensor, Tensor]:
#         ctx.helper = helper
#         ctx.simulation = cppSim
#         ctx.pastRecord = cppSim.getStateInfo()
#         ctx.decay = decay
#         argX = np.float64(x0.contiguous().detach().cpu().numpy())
#         argV = np.float64(v0.contiguous().detach().cpu().numpy())
#         argC = np.float64(ctrl_traj.contiguous().detach().cpu().numpy())
#
#         out_x = [argX.reshape(-1, 3).copy()]
#         out_v = [argV.reshape(-1, 3).copy()]
#         fw_records = []
#         for step in range(cppSim.sceneConfig.stepNum - 1):
#             cppSim.stepNN(step + 1, argX, argV, argC[step+1])
#
#             newRecord = cppSim.getStateInfo()
#             fw_records.append(newRecord)
#             argX = newRecord.x
#             argV = newRecord.v
#             out_x.append(argX.reshape(-1, 3).copy())
#             out_v.append(argV.reshape(-1, 3).copy())
#
#         ctx.fwRecords = fw_records
#
#
#         x_np = np.stack(out_x, axis=0)
#         v_np = np.stack(out_v, axis=0)
#         # try cat abd move one tensor
#         x_next = torch.as_tensor(x_np).float()
#         v_next = torch.as_tensor(v_np).float()
#
#         return x_next, v_next
#
#     @staticmethod
#     def backward(
#             ctx: Any,
#             dL_dx_next: Tensor,
#             dL_dv_next: Tensor
#     ) -> Tuple[Tensor, Tensor, None, Tensor, Tensor, None, None]:
#         cppSim = ctx.simulation
#         decay = ctx.decay
#         # grad_scales = np.array([1e2, 1e5, 1e4, 1e0])
#         grad_scales = np.array([1e0, 1e0, 1e0, 1e0])
#
#         dL_dx_next_np = dL_dx_next.contiguous().detach().cpu().numpy()
#         dL_dv_next_np = dL_dv_next.contiguous().detach().cpu().numpy()
#
#         dL_dxcur = dL_dx_next_np[-1].reshape(-1)
#         dL_dvcur = dL_dv_next_np[-1].reshape(-1)
#         isLast = True
#         fwRecords = ctx.fwRecords
#
#         dL_dd = []
#         dL_dk = []
#
#         for step in range(cppSim.sceneConfig.stepNum - 1, 0, -1):
#             if isLast:
#                 backRecord = cppSim.stepBackwardNN(
#                     ctx.helper.taskInfo,
#                     np.zeros_like(dL_dxcur),
#                     np.zeros_like(dL_dvcur),
#                     fwRecords[step - 1],
#                     fwRecords[step - 1].stepIdx == 1,
#                     dL_dxcur,
#                     dL_dvcur)
#                 isLast = False
#             else:
#                 backRecord = cppSim.stepBackwardNN(
#                     ctx.helper.taskInfo,
#                     dL_dxcur,
#                     dL_dvcur,
#                     fwRecords[step - 1],
#                     fwRecords[step - 1].stepIdx == 1,
#                     dL_dx_next_np[step].flatten(),
#                     dL_dv_next_np[step].flatten()
#                 )
#             # dL_dxcur = 1 / decay * backRecord.dL_dx + dL_dx_next_np[step - 1].flatten()
#             # dL_dvcur = 1 / decay * backRecord.dL_dv + dL_dv_next_np[step - 1].flatten()
#             dL_dd.append(backRecord.dL_ddensity)
#             dL_dk.append(backRecord.numpy_dL_dk_pertype())
#             dL_dxcur = backRecord.dL_dx
#             dL_dvcur = backRecord.dL_dv
#
#         # dL_dd_np = np.stack(dL_dd, axis=0).mean(axis=0)
#         # dL_dk_np = np.stack(dL_dk, axis=0).mean(axis=0)
#         # dL_dk_np = dL_dk_np * grad_scales
#         dL_dd_np = np.asarray(backRecord.dL_ddensity).copy()
#         dL_dk_np = np.asarray(backRecord.numpy_dL_dk_pertype()).copy()
#
#         dL_dx = torch.as_tensor(np.asarray(dL_dxcur).copy())
#         dL_dv = torch.as_tensor(np.asarray(dL_dvcur).copy())
#         dL_dx += dL_dx_next[0].flatten()
#         dL_dv += dL_dv_next[0].flatten()
#         # dL_dmu = torch.as_tensor(backRecord.dL_dmu)
#         dL_ddensity = torch.as_tensor(dL_dd_np).to(torch.device('cuda'))
#         dL_dk_pertype = torch.as_tensor(dL_dk_np).to(torch.device('cuda'))
#
#         dL_dx = dL_dx.to(torch.device('cuda'))
#         dL_dv = dL_dv.to(torch.device('cuda'))
#
#         return dL_dx, dL_dv, None, dL_dk_pertype, dL_ddensity, None, None, None
#
from typing import Any, Optional, Mapping, Tuple

import torch
import torch.autograd as autograd
from torch import Tensor


import diffcloth_py as dfc
import numpy as np
import math
from numpy import linalg
import time

from diffcloth_py import Simulation, ForwardInformation


class SimFunction(autograd.Function):

    @staticmethod
    def forward(
            ctx: Any,
                x0: Tensor,
                v0: Tensor,
                ctrl_traj: Tensor,
                k_pertype: Tensor,
                density: Tensor,
                cppSim: dfc.Simulation,
                helper: dfc.OptimizeHelper,
                decay: float
    ) -> Tuple[Tensor, Tensor]:
        ctx.helper = helper
        ctx.simulation = cppSim
        ctx.pastRecord = cppSim.getStateInfo()
        ctx.decay = 0.1
        argX = np.float64(x0.contiguous().detach().cpu().numpy())
        argV = np.float64(v0.contiguous().detach().cpu().numpy())
        argC = np.float64(ctrl_traj.contiguous().detach().cpu().numpy())
        x_init = argX

        out_x = [argX.reshape(-1, 3).copy()]
        out_v = [argV.reshape(-1, 3).copy()]
        fw_records = []
        for step in range(cppSim.sceneConfig.stepNum - 1):
            v = np.array(argV)
            for i in range(int(len(argV) / 3) - 1):

                h = argX[i * 3 + 1] - x_init[i * 3 + 1]
                a = 1
                if h < 0:
                    h = 0
                    a = -1
                v[i * 3 + 1] = argV[i * 3 + 1] - 2 * a * math.sqrt(2 * h * 9.81 * 0.01)
                # v[i * 3 + 1] = argV[i * 3 + 1] - math.sqrt(2 * h * 9.81 * 0.01)

            cppSim.stepNN(step + 1, argX, v, argC[step+1])

            newRecord = cppSim.getStateInfo()
            fw_records.append(newRecord)
            argX = newRecord.x
            argV = newRecord.v
            out_x.append(argX.reshape(-1, 3).copy())
            out_v.append(argV.reshape(-1, 3).copy())

        ctx.fwRecords = fw_records


        x_np = np.stack(out_x, axis=0)
        v_np = np.stack(out_v, axis=0)
        # try cat abd move one tensor
        x_next = torch.as_tensor(x_np).float()
        v_next = torch.as_tensor(v_np).float()

        return x_next, v_next

    @staticmethod
    def backward(
            ctx: Any,
            dL_dx_next: Tensor,
            dL_dv_next: Tensor
    ) -> Tuple[Tensor, Tensor, None, Tensor, Tensor, None, None]:
        cppSim = ctx.simulation
        decay = ctx.decay
        # grad_scales = np.array([1e2, 1e5, 1e4, 1e0])
        grad_scales = np.array([1e0, 1e0, 1e0, 1e0])

        dL_dx_next_np = dL_dx_next.contiguous().detach().cpu().numpy()
        dL_dv_next_np = dL_dv_next.contiguous().detach().cpu().numpy()

        dL_dxcur = dL_dx_next_np[-1].reshape(-1)
        dL_dvcur = dL_dv_next_np[-1].reshape(-1)
        isLast = True
        fwRecords = ctx.fwRecords

        dL_dd = []
        dL_dk = []

        for step in range(cppSim.sceneConfig.stepNum - 1, 0, -1):
            if isLast:
                backRecord = cppSim.stepBackwardNN(
                    ctx.helper.taskInfo,
                    np.zeros_like(dL_dxcur),
                    np.zeros_like(dL_dvcur),
                    fwRecords[step - 1],
                    fwRecords[step - 1].stepIdx == 1,
                    dL_dxcur,
                    dL_dvcur)
                isLast = False
            else:
                backRecord = cppSim.stepBackwardNN(
                    ctx.helper.taskInfo,
                    dL_dxcur,
                    dL_dvcur,
                    fwRecords[step - 1],
                    fwRecords[step - 1].stepIdx == 1,
                    dL_dx_next_np[step].flatten(),
                    dL_dv_next_np[step].flatten()
                )
            # dL_dxcur = 1 / decay * backRecord.dL_dx + dL_dx_next_np[step - 1].flatten()
            # dL_dvcur = 1 / decay * backRecord.dL_dv + dL_dv_next_np[step - 1].flatten()
            dL_dd.append(backRecord.dL_ddensity)
            dL_dk.append(backRecord.numpy_dL_dk_pertype())
            dL_dxcur = backRecord.dL_dx
            dL_dvcur = backRecord.dL_dv

        # dL_dd_np = np.stack(dL_dd, axis=0).mean(axis=0)
        # dL_dk_np = np.stack(dL_dk, axis=0).mean(axis=0)
        # dL_dk_np = dL_dk_np * grad_scales
        dL_dd_np = np.asarray(backRecord.dL_ddensity).copy()
        dL_dk_np = np.asarray(backRecord.numpy_dL_dk_pertype()).copy()

        dL_dx = torch.as_tensor(np.asarray(dL_dxcur).copy())
        dL_dv = torch.as_tensor(np.asarray(dL_dvcur).copy())
        dL_dx += dL_dx_next[0].flatten()
        dL_dv += dL_dv_next[0].flatten()
        # dL_dmu = torch.as_tensor(backRecord.dL_dmu)
        dL_ddensity = torch.as_tensor(dL_dd_np).to(torch.device('cuda'))
        dL_dk_pertype = torch.as_tensor(dL_dk_np).to(torch.device('cuda'))

        dL_dx = dL_dx.to(torch.device('cuda'))
        dL_dv = dL_dv.to(torch.device('cuda'))

        return dL_dx, dL_dv, None, dL_dk_pertype, dL_ddensity, None, None, None