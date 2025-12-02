import torch.nn as nn
import torch


class Embedding(nn.Module):
    """
    Token embedding layer that maps token IDs to dense vectors.
    
    Purpose:
        Converts discrete token IDs into continuous vector representations.
        Embedding matrix initialized with N(µ = 0, σ2 = 1)truncated at [−3, 3].
    
    Input shape:
        token_ids: (*batch_dims,) - arbitrary batch dimensions with integer token IDs
    
    Output shape:
        embeddings: (*batch_dims, embedding_dim) - same batch dimensions with embedding vectors
    
    Attributes:
        num_embeddings: vocabulary size
        embedding_dim: dimension of embedding vectors
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.info = {'device': device, 'dtype': dtype}
        self.embedding_matrix = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **self.info))

    def initialize_parameter(self):
        nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=1, a=-3, b=3)

    def load_weight(self, weight: torch.Tensor):
        with torch.no_grad():
            self.embedding_matrix.copy_(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]
