import math
import torch
import torch.nn as nn



class Linear(nn.Module):
    """
    Linear transformation layer without bias term.
    

    Input shape:
        x: (..., in_features)
    
    Output shape:
        y: (..., out_features)
    
    Attributes:
        in_features: size of input feature dimension
        out_features: size of output feature dimension
    """

    def __init__(self, in_features: int, out_features: int, device :torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.info = {'device': device, 'dtype':dtype}
        self.weight = nn.Parameter(torch.empty((self.out_features,self.in_features),**self.info))
        self.initialize_parameters()

    def initialize_parameters(self):
        sigma = math.sqrt(2 /(self.in_features + self.out_features))

        nn.init.trunc_normal_(self.weight, mean = 0.0, std = sigma , a = - 3 * sigma , b = 3 * sigma )

    def load_weight(self, weight: torch.Tensor):
        with torch.no_grad():
            self.weight.copy_(weight)

    def forward(self, x: torch.Tensor):
        return torch.einsum("...i, oi -> ... o",x,self.weight)