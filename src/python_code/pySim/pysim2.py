from typing import Optional, Mapping, Tuple, Sequence, NoReturn, Union

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

import diffcloth_py as diffcloth

from .functional2 import SimFunction


class pySim(nn.Module):

    def __init__(self,
    cppSim: diffcloth.Simulation,
    optimizeHelper: diffcloth.OptimizeHelper,
    useFixedPoint: bool
    ) -> NoReturn:
        super().__init__()
        self.cppSim = cppSim
        self.optimizeHelper = optimizeHelper

        self.cppSim.useCustomRLFixedPoint = useFixedPoint

    def forward(
            self,
            x: Tensor,
            v: Tensor,
            a: np.ndarray,
    ) -> Tuple[Tensor, Tensor, np.ndarray, Tensor]:

        return SimFunction.apply(
            x, v, a, self.cppSim, self.optimizeHelper)
