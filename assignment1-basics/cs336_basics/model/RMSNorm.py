import math

import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Purpose:
        Normalizes input by root mean square of features along the last dimension.
        More computationally efficient than LayerNorm as it doesn't subtract mean.
        Formula: output = x / sqrt(mean(x^2) + eps) * weight
    
    Input shape:
        x: (..., d_model) - arbitrary batch dimensions with d_model features
    
    Output shape:
        output: (..., d_model) - same shape as input, normalized and scaled
    
    Attributes:
        d_model: dimension of feature vectors to normalize
        eps: small constant for numerical stability (prevents division by zero)
    """
    def __init__(self,d_model: int, eps: float = 1e-5,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.info = {'device':device, 'dtype':dtype}
        self.weight = nn.Parameter(torch.ones(d_model,**self.info))

    def forward(self,x):
        in_dtype = x.dtype
        x_float32 = x.to(torch.float32)

        rms = torch.rsqrt((x_float32.pow(2).mean(dim = -1,keepdim = True)) + self.eps)
        result = x * rms * self.weight
        
        return result.to(in_dtype)

    def load_weight(self, weight):
        with torch.no_grad():
            self.weight.copy_(weight)
