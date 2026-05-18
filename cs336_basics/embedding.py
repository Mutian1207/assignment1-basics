import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.std = 1
        nn.init.trunc_normal_(self.embedding_matrix, 0, std = 1, a = -3, b = 3)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]

