import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class Uncertainty_head(nn.Module):   # feature -> log(sigma^2)
    def __init__(self, in_feat=304, out_feat=256):
        super(Uncertainty_head, self).__init__()
        self.fc1 = Parameter(torch.Tensor(out_feat, in_feat))
        self.bn1 = nn.BatchNorm2d(out_feat, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = Parameter(torch.Tensor(out_feat, out_feat))
        self.bn2 = nn.BatchNorm2d(out_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta = Parameter(torch.Tensor([0.0]))

        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = F.linear(x, F.normalize(self.fc1, dim=-1)) # [B, W, H, D]
        x = x.permute(0, 3, 1, 2) # [B, W, H, D] -> [B, D, W, H]
        x = self.bn1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = F.linear(x, F.normalize(self.fc2, dim=-1))
        x = x.permute(0, 3, 1, 2)
        x = self.bn2(x)
        x = self.gamma * x + self.beta
        x =  torch.log(torch.exp(x) + 1e-6)
        x = torch.sigmoid(x)
        
        return x