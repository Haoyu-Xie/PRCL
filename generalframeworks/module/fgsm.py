from matplotlib.colors import TwoSlopeNorm
import torch.nn as nn
from torch import Tensor
from typing import Tuple
class FGSMGenerator(object):

    def __init__(self, net: nn.Module, eplision: float =0.05) -> None:
        super().__init__()
        self.net = net
        self.eplision = eplision

    def __call__(self, img: Tensor, gt: Tensor, criterion: nn.Module) -> Tuple[Tensor, Tensor, Tensor]:
        assert img.shape.__len__() == 4
        assert img.shape[0] >= gt.shape[0]
        img.requires_grad = True
        if img.grad is not None:
            img.grad.zero_()
        self.net.zero_grad()
        pred = self.net(img)
        
