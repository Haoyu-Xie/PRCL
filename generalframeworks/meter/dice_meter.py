import os
from pickle import GLOBAL
import sys

from torch._C import default_generator
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import torch.nn.functional as F
from generalframeworks.utils import one_hot, intersection, class2one_hot, probs2class, probs2one_hot
import torch
from torch import einsum
from functools import partial

def toOneHot(pred_logit, label):
    '''
    pred_logit: (B, C, H, W) containing 0, 1
    label: (B, H, W) containing (0, 1, ..., num_class)
    '''
    oh_predmask = probs2one_hot(F.softmax(pred_logit, dim=1)) # [b, w, h] -> [b, c, w, h]
    oh_label = class2one_hot(label.squeeze(1), pred_logit.shape[1]) # [b, c, w, h]
    assert oh_predmask.shape == oh_label.shape, oh_label.shape
    return oh_predmask, oh_label
    

def meta_dice(sum_str: str, label: torch.Tensor, pred: torch.Tensor, smooth: float = 1e-18) -> float:
    assert label.shape == pred.shape, 'label shape must be the same as pred'
    assert one_hot(label), 'label must be one hot coding'
    assert one_hot(pred), 'pred must be one hot coding'

    inter_size: torch.Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: torch.Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: torch.Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)
    
    return dices

dice_coef = partial(meta_dice, "bcwh->bc") # sum on dim w*h return dim (b, c)
dice_batch = partial(meta_dice, 'bcwh->c') # sum on dim b*w*h return dim (c)

class DiceMeter(object):
    def __init__(self, method='2d', report_axises='all', num_class=4) -> None:
        assert method in ('2d', '3d')
        assert report_axises == 'all' or isinstance(report_axises, list)
        self.method = method
        self.diceCall = dice_coef if self.method =='2d' else dice_batch
        self.report_axis = report_axises
        self.diceLog = []
        self.num_class = num_class

    def reset(self):
        self.diceLog = []

    

    @property
    def log(self) -> torch.Tensor:
        try:
            log = torch.cat(self.diceLog)
        except:
            log = torch.Tensor([0 for _ in range(self.num_class)])

        if len(log.shape) == 1:
            log = log.unsqueeze(0)
        assert len(log.shape) == 2
        return log

    def add(self, pred_logit, label):
        dice_value = self.diceCall(*toOneHot(pred_logit, label))
        if dice_value.shape.__len__() == 1:
            dice_value = dice_value.unsqueeze(0)
        assert dice_value.shape.__len__() == 2
        self.diceLog.append(dice_value)

    def value(self, **kwargs):
        log = self.log
        means = log.mean(0)
        stds = log.std(0)
        report_means = log.mean(1) if self.report_axis == 'all' else log[:, self.report_axis].means(1)
        report_std = report_means.std()
        report_mean = report_mean.mean()

        return (report_mean, report_std), (means, stds)

        

if __name__ == '__main__':
    c = torch.tensor([[[[0.5, 0.3], #logit 
                        [0.1, 0.9]],

                        [[0.1, 0.6],
                        [0.0, 0.4]],

                        [[0.1, 0.2],
                        [0.7, 0.5]],]])
    a = torch.softmax(c, dim=1) # prob
    e = probs2class(a) # class
    l, k = toOneHot(c, e)
    print(l)
    print(k)
    