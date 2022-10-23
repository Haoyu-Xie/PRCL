from multiprocessing.sharedctypes import Value
import torch
from generalframeworks.utils import class2one_hot
import numpy as np


class Meter(object):

    def reset(self):
        # Reset the Meter to default settings
        pass

    def add(self, pred_logits, label):
        # Log a new value to the meter
        pass

    def value(self):
        # Get the value of the meter in the current state
        pass

    def summary(self) -> dict:
        raise NotImplementedError

    def detailed_summary(self) -> dict:
        raise NotImplementedError


class AverageValueMeter(Meter):
    def __init__(self, num_class):  # HD is await.
        super(AverageValueMeter, self).__init__()
        self.num_class = num_class
        self.reset()
        self.dice = []
        self.HD = []

    def add(self, pred_logits: torch.Tensor, label: torch.Tensor) -> None:
        '''
        Args:
            pred_logits: pred_logits from network unthrough sotmax (b, c, w, h)
            label: label from the dataset (b, w, h) containing (0, 1, ..., num_class)
        '''
        assert pred_logits.shape.__len__() == 4, 'pred_logits must be 4 dim, but now is {} dim!'.format(
            pred_logits.shape.__len__())
        assert label.shape.__len__() == 3, 'label must be 3 dim, but now is {} dim!'.format(label.shape.__len__())
        batch_size = pred_logits.shape[0]
        pred_probs, pred_class = torch.max(torch.softmax(pred_logits, dim=1), dim=1)
        # one_hot coding
        label_oh = class2one_hot(label, num_class=self.num_class).permute(1, 0, 2, 3)  # [c, b, h, w]
        pred_oh = class2one_hot(pred_class, num_class=self.num_class).permute(1, 0, 2, 3)  # [c, b, h, w]
        class_dice = [compute_dice(pred_oh[i], label_oh[i]) for i in range(self.num_class)]  # log dice of each class
        self.dice.append(class_dice)

    def reset(self):
        self.dice = []
        self.HD = []

    def value(self, index, mode='mean'):
        if mode == 'mean':
            return np.array(self.dice[index]).mean()
        elif mode == 'all':
            return self.dice[index]  # list with length is num_class containing dice of each num_class
        else:
            raise ValueError("mode must be in (all, mean)")

    def summary(self) -> dict:
        Dice_dct: dict = {}
        for c in range(self.num_class):
            if c != 0:
                Dice_dct['DSC_{}'.format(c)] = np.array([self.value(i, mode='all') for i in range(len(self.dice))])[:,
                                               c].mean()
        return Dice_dct


class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
    
    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)
    
    
    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        up = torch.diag(h)
        down = h.sum(1) + h.sum(0) - torch.diag(h)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-6)
        return torch.mean(iu).item(), acc.item()

    def get_valid_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        up = torch.diag(h)
        down = h.sum(1) + h.sum(0) - torch.diag(h)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-6)
        num_no_zero = (iu == 0).sum()
        return iu.sum() / (len(iu) - num_no_zero).item(), acc.item()
        


class My_ConfMatrix(Meter):
    def __init__(self, num_classes):
        super(ConfMatrix, self).__init__()
        self.num_classes = num_classes
        self.mat = None
        self.reset()
        self.mIOU = []
        self.Acc = []

    def add(self, pred_logits, label):
        pred_logits = pred_logits.argmax(1).flatten()
        label = label.flatten()
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred_logits.device)
        with torch.no_grad():
            k = (label >= 0) & (label < n)
            inds = n * label[k].to(torch.int64) + pred_logits[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def value(self, mode='mean'):
        h = self.mat.float()
        self.acc = torch.diag(h).sum() / h.sum()
        self.iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        if mode == 'mean':
            return torch.mean(self.iu).item(), self.acc.item()
        else:
            raise ValueError("mode must be in (mean)")

    def reset(self):
        self.mIOU = []
        self.Acc = []

    def summary(self) -> dict:
        mIOU_dct: dict = {}
        Acc_dct: dict = {}
        for c in range(self.num_classes):
            if c != 0:
                mIOU_dct['mIOU_{}'.format(c)] = np.array([self.value(i, mode='all')[0] for i in range(len(self.mIOU))])[
                                                :, c].mean()
                Acc_dct['Acc_{}'.format(c)] = np.array([self.value(i, mode='all')[1] for i in range(len(self.mIOU))])[:,
                                              c].mean()
        return mIOU_dct, Acc_dct


def compute_dice(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    intersection = (mask1 * mask2).sum()
    union = (mask1 + mask2).sum()
    if union != 0:
        dice = float((2 * intersection) / union)
    else:
        dice = 0

    return dice


if __name__ == '__main__':
    pred = torch.Tensor([[[[0.3026, 0.0367],
                           [0.9102, 0.7437]],

                          [[0.5027, 0.2508],
                           [0.1371, 0.0762]],

                          [[0.7781, 0.2338],
                           [0.5623, 0.9338]]],

                         [[[0.0077, 0.5065],
                           [0.7355, 0.0411]],

                          [[0.0916, 0.0701],
                           [0.2530, 0.1168]],

                          [[0.6059, 0.6115],
                           [0.5303, 0.0733]]]])  # [2, 2, 2, 2][[0, 1], [1, 1]] # [[[1, 0],[1, 1]], [[0, 1], [1, 1]]]
    mask = torch.Tensor([[[2, 1],
                          [0, 2]],

                         [[2, 2],
                          [0, 1]]])
    pred_1 = torch.Tensor([[[[0.2689, 0.8217],
                             [0.9374, 0.9823]],

                            [[0.8852, 0.7713],
                             [0.1072, 0.5964]],

                            [[0.6984, 0.5038],
                             [0.4828, 0.6572]]],

                           [[[0.6826, 0.1830],
                             [0.7507, 0.4995]],

                            [[0.2060, 0.0187],
                             [0.3212, 0.3637]],

                            [[0.2623, 0.9202],
                             [0.4473, 0.7727]]]])
    mask_1 = torch.Tensor([[[1, 0],
                            [0, 0]],

                           [[0, 2],
                            [0, 0]]])
    '''
        mask_1 = torch.Tensor([[[1, 0],
                          [0, 0]],

                         [[0, 2],
                          [0, 2]]])'''
    meter = AverageValueMeter(num_class=3)
    meter.reset()
    meter.add(pred, mask)
    meter.add(pred, mask)
    print(meter.dice)
    meter.add(pred_1, mask_1)
    print(meter.dice)
    # for i in range(2):
    # print(meter.value(i, mode='all'))

    print(meter.summary())










