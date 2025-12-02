import torch.nn as nn
import torch
from cs336_basics.Embedding import Embedding
from cs336_basics.Linear import Linear
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.TransformerBlock import TransformerBlock


class TransformerLM(nn.Module):

    def __init__(self, vocab_size, context_length, num_layers,
                 d_model:int,num_heads:int,d_ff:int,rope_theta:float,use_rope:bool,
                 device:torch.device = None, dtype:torch.dtype = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.info = {'device': device, 'dtype':dtype}
        self.token_embeddings = Embedding(vocab_size,d_model,**self.info)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model,num_heads,d_ff,use_rope,rope_theta,context_length,**self.info) for i in range(num_layers)])
        self.ln_final = RMSNorm(d_model,eps = 1e-5,**self.info)
        self.lm_head = Linear(d_model,vocab_size,**self.info)

    def forward(self,x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        output = self.lm_head(x)

        return output
