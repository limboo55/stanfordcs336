import torch.nn as nn
import torch
import einops

def split_rotate_half(x:torch.Tensor):
    x1,x2 = x.chunk(2,dim = -1)
    return torch.cat((-x2,x1),dim = -1)


# class Rope(nn.Module):
#
#     def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
#         super().__init__()
#         self.theta = theta
#         self.d_k = d_k
#         self.max_seq_len = max_seq_len
#
#         # theta_i = 1 / (theta ** (2j / d_k)) [theta_0, theta_1, ... , theta_(d_k - 1)]
#         thetas = 1 / theta ** (torch.arange(0,d_k,2,device = device).float() / d_k)
#         # [0,max_seq_len - 1]
#         pos = torch.arange(0,max_seq_len, device = device)
#         pos_thetas = torch.einsum('i,j ->ij', pos,thetas)
#         freqs = torch.cat((pos_thetas,pos_thetas),dim = -1)
#
#         self.register_buffer("cos_cache", torch.cos(freqs), persistent= False)
#         self.register_buffer("sin_cache", torch.sin(freqs), persistent= False)
#
#     def forward(self, x: torch.Tensor, token_positions: torch.Tensor ) -> torch.Tensor:
#
#         cos_pos = self.cos_cache[token_positions]
#         sin_pos = self.sin_cache[token_positions]
#
#         return x * cos_pos + split_rotate_half(x) * sin_pos



class Rope(nn.Module):
    """
    Rotary Position Embedding (RoPE) for encoding positional information.
    
    Purpose:
        Encodes positional information by rotating query/key vectors in complex space.
        Provides better relative position encoding than absolute positional embeddings.
        Pre-computes sine and cosine rotation matrices for efficiency.
    
    Input shape:
        x: (..., seq_len, d_k) - vectors to apply positional encoding (typically Q or K)
        token_positions: (batch_size, seq_len) - position indices for each token
    
    Output shape:
        output: (..., seq_len, d_k) - same shape as input with rotary position applied
    
    Attributes:
        theta: base frequency for rotation (default 10000)
        d_k: dimension of key/query vectors per head
        max_seq_len: maximum sequence length supported
    """

    def __init__(self, theta: float , d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # theta_i = 1 / (theta ** (2j / d_k)) [theta_0, theta_1, ... , theta_(d_k - 1)]
        thetas = 1 / theta ** (torch.arange(0,d_k,2,device = device).float() / d_k)
        # [0,max_seq_len - 1]
        pos = torch.arange(0,max_seq_len, device = device)
        pos_thetas = torch.einsum('i,j ->ij', pos,thetas)
        freqs = pos_thetas.repeat_interleave(2,dim = -1)

        self.register_buffer("cos_cache", torch.cos(freqs), persistent= False)
        self.register_buffer("sin_cache", torch.sin(freqs), persistent= False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor ) -> torch.Tensor:
        x_pair = einops.rearrange(x, '... (d c) -> c ... d', c=2)
        # [x1,x2....] * cos_pos + [-x2,x1,-x4,x3] * sin_pos
        x_evens, x_odds = x_pair[0], x_pair[1]
        x_rotated = einops.rearrange([-x_odds, x_evens], 'c ... d -> ... (d c)')
        cos_pos = self.cos_cache[token_positions]
        sin_pos = self.sin_cache[token_positions]

        return x * cos_pos + x_rotated * sin_pos


