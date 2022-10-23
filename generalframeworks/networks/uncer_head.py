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
        # x = torch.exp(x)
        
        return x


class Uncertainty_head_conv(nn.Module):   # feature -> log(sigma^2)
    def __init__(self, in_feat=304, out_feat=256):
        super(Uncertainty_head_conv, self).__init__()
        # self.uncertainty = nn.Sequential(
        #     nn.Conv2d(in_feat, out_feat, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_feat, affine=True),
        #     nn.ReLU(),
        #     nn.Conv2d(out_feat, out_feat, 1),
        #     nn.BatchNorm2d(out_feat, affine=True)
        # )
        self.uncertainty = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_feat, 1)
        )
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta = Parameter(torch.Tensor([0.0]))

    def forward(self, x: torch.Tensor):
        x = self.uncertainty(x)
        x = self.gamma * x + self.beta
        x =  torch.log(torch.exp(x) + 1e-6)
        x = torch.sigmoid(x)
        return x

##### Test #####
if __name__ == '__main__':
    uncer = Uncertainty_head_conv()
    a = torch.randn(4, 304, 81, 81)
    b = uncer(a)
    b = torch.exp(b)
    print(b.shape)
    print(b.sum(1))

class Uncertainty_head_sig(nn.Module):   # feature -> log(sigma^2)
    def __init__(self, in_feat=304, out_feat=256):
        super(Uncertainty_head_sig, self).__init__()
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
        x = torch.exp(x)
        x = F.sigmoid(x) # (1/2, 1)

        return x
