import torch.nn as nn

from cs336_basics.Linear import Linear
from cs336_basics.Multihead_self_attention import Multihead_self_attention
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Rope import Rope
import torch

from cs336_basics.SwiGLU import SwiGLU


class TransformerBlock(nn.Module):
    """


    """
    def __init__(self, d_model : int, num_heads:int, d_ff: int,
                 use_rope:bool = True,rope_theta:float = 1000.0,max_seq_len:int = 2048,
                 device:torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.info = {'device' : device, 'dtype': dtype}
        self.attn = Multihead_self_attention(d_model,num_heads,rope_theta,max_seq_len,use_rope,**self.info)

        self.ln1 = RMSNorm(d_model,eps = 1e-5,**self.info)
        self.ln2 = RMSNorm(d_model,eps = 1e-5,**self.info)

        self.ffn = SwiGLU(d_model,d_ff,**self.info)

    def forward(self,x:torch.Tensor):
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x1 = x + residual
        residual1 = x1
        x1 = self.ln2(x1)
        x1 = self.ffn(x1)
        output = x1 + residual1
        return output



