from typing import Mapping, Any

import torch
import torch.nn as nn
import einops
from cs336_basics.Linear import Linear
from cs336_basics.Rope import Rope
from cs336_basics.Utils import scaled_dot_product_attention



class Multihead_self_attention(nn.Module):
    """
    Multi-head self-attention mechanism with optional Rotary Position Embedding (RoPE).
    
    Purpose:
        Implements scaled dot-product attention across multiple heads in parallel.
        Projects input to Q, K, V matrices, applies optional RoPE for positional encoding,
        computes attention weights with masking, and projects output back to d_model dimensions.
    
    Input shape:
        x: (batch_size, seq_len, d_model) - input sequence embeddings
        mask: (seq_len, seq_len) - optional boolean attention mask (default: lower triangular)
        token_positions: (batch_size, seq_len) - optional position indices for RoPE
    
    Output shape:
        output: (batch_size, seq_len, d_model) - attended output sequence
    
    Attributes:
        d_model: dimension of input/output embeddings
        num_heads: number of attention heads
        use_rope: whether to apply rotary position embeddings
    """

    def __init__(self, d_model : int , num_heads : int, theta:float = 1000, max_seq_len:int = 2048,
                 use_rope:bool = True,device : torch.device | None = None, dtype: torch.dtype = None):
        super().__init__()
        self.q_proj = Linear(d_model, d_model, device = device, dtype = dtype)
        self.k_proj = Linear(d_model, d_model, device = device, dtype = dtype)
        self.v_proj = Linear(d_model, d_model, device = device, dtype = dtype)
        self.output_proj = Linear(d_model, d_model, device = device, dtype = dtype)
        self.use_rope = use_rope
        self.num_heads = num_heads
        d_k =int (d_model / num_heads)
        if self.use_rope:
            self.rope = Rope(theta,d_k,max_seq_len)
        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, x,mask = None, token_positions = None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (batch,seq,d_model) -> (batch,num_head,seq,d_k)
        q = einops.rearrange(q,'b s (h d) -> b h s d',h = self.num_heads)
        k = einops.rearrange(k,'b s (h d) -> b h s d',h = self.num_heads)
        v = einops.rearrange(v,'b s (h d) -> b h s d',h = self.num_heads)
        batch_size, seq_len, _ =  x.shape

        # apply rope for query and key
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            q = self.rope(q,token_positions)
            k = self.rope(k,token_positions)

        # if mask is None, use the default lower triangular
        if mask is None:
            mask = torch.tril(torch.ones((seq_len,seq_len), device = self.device,dtype = torch.bool))
        attention_weight = scaled_dot_product_attention(q,k,v,mask)
        attention_weight = einops.rearrange(attention_weight,"b h s d -> b s (h d)")
        return self.output_proj(attention_weight)
