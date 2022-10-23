import copy
import torch.nn as nn
class EMA(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))

    def update(self, model):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

class OOD_EMA(nn.Module):
    def __init__(self, model, alpha):
        super(OOD_EMA, self).__init__()
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))

    def update(self, model):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1