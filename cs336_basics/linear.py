import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.std = math.sqrt(2/(in_features + out_features)) 
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=self.std, a=-3*self.std, b=3*self.std)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T)

