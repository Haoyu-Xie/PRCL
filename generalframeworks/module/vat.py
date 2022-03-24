
from dataclasses import dataclass
from doctest import OutputChecker
from turtle import backward
import torch
import torch.nn as nn
import warnings
from typing import Tuple
import torch.nn.functional as F

from zmq import device
from generalframeworks.utils import simplex
class VATGenerator(object):
    def __init__(self, net: nn.Module, xi=1e-6, eplision=10, ip=1):
        ''' VAT generator '''
        super(VATGenerator, self).__init__()
        self.xi = xi
        self.eps = eplision
        self.ip = ip
        self.net = net
        self.entropy = Entropy_2d()

    @staticmethod
    def _l2_normalize(d) -> torch.Tensor:
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-16
        assert torch.allclose(d.view(d.shape[0], -1).norm(dim=1), torch.ones(d.shape[0]).to(d.device), rtol=1e-3)
        return d

    def kl_div_with_logit(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        logq = F.log_softmax(q_logit, dim=1)
        logp = F.log_softmax(p_logit, dim=1)

        qlogq = (q * logq).sum(dim=1)
        qlogp = (q * logp).sum(dim=1)
        return qlogq - qlogp

    def __call__(self, img: torch.Tensor, loss_name='kl') -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            output = self.net(img)
        d = torch.Tensor(img.size()).normal_()
        d = self._l2_normalize(d).to(img.device)
        assert torch.allclose(d.view(d.shape[0], -1).norm(dim=1), torch.ones(d.shape[0]).to(img.device), rtol=1e-3), 'The L2 normal fails'

        self.net.zero_grad()
        for _ in range(self.ip):
            d = self.xi * self._l2_normalize(d).to(img.device)
            d.requires_grad = True
            y_hat = self.net(img + d)

            delta_kl = self.kl_div_with_logit(output[0].detach(), y_hat[0])
            delta_kl.mean().backward()
            #d = d.grad.data.clone().cpu()
            d = d.grad.data.clone().to(img.device)
            self.net.zero_grad()

        d = self._l2_normalize(d).to(img.device)
        r_adv = self.eps * d
        img_adv = img + r_adv.detach()  
        img_adv = torch.clamp(img_adv, 0, 1)

        return img_adv.detach(), r_adv.detach()          

class Entropy_2d(nn.Module):
    '''self entropy, on dim c'''
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor):
        assert input.shape.__len__() == 4
        b, _, h, w= input.shape
        assert simplex(input)
        e = input * (input + 1e-16).log()
        e = -1.0 * e.sum(1)
        assert e.shape == torch.Size([b, h, w])
        return e