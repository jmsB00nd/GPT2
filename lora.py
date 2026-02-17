import torch
import torch.nn as nn
import math


class LoraLayer(nn.Module):
    def __init__(self,in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        
        
    def forward(self, x):
        return self.alpha * x @ self.A @ self.B
 
 
 
class LinearWithLora(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoraLayer(linear.in_features, linear.out_features, rank, alpha)
        
    def forward(self, x):
        return self.linear(x) + self.lora(x)       