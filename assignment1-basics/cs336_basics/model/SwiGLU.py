import torch
import torch.nn as nn
from cs336_basics.Linear import Linear
from cs336_basics.Utils import SiLU


class SwiGLU(nn.Module):
    """
    SwiGLU activation-based feedforward network.
    
    Purpose:
        Gated feedforward layer using SiLU (Swish) activation and element-wise gating.
        Provides better performance than standard ReLU-based FFN in transformers.
        Formula: output = W2(SiLU(W1(x)) * W3(x))
    
    Input shape:
        x: (batch_size, seq_len, d_model) - input sequence embeddings
    
    Output shape:
        output: (batch_size, seq_len, d_model) - transformed sequence embeddings
    
    Attributes:
        d_model: dimension of input/output embeddings
        d_ff: dimension of intermediate hidden layer (typically 4 * d_model)
    """
    def __init__(self, d_model: int, d_ff: int, device:torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        # W1: Gate projection (d_model -> d_ff)
        self.w1 = Linear(d_model, d_ff, device, dtype)
        # W2: Down projection (d_ff -> d_model)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        # W3: Up projection (d_model -> d_ff)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x):

        return self.w2(SiLU(self.w1(x)) * self.w3(x))

    def load_weight(self, w1, w2, w3):

        self.w1.load_weight(w1)
        self.w2.load_weight(w2)
        self.w3.load_weight(w3)