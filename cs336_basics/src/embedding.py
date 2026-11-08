import math
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 存成 [num_embeddings, embedding_dim]:每一行是一个 token 的向量
        self.W = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # N(0, 1),截断在 [-3, 3]
        nn.init.trunc_normal_(self.W, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [...] (整数 id) -> [..., embedding_dim]
        return self.W[token_ids]
