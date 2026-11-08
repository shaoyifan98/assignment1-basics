import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 存成 W,形状 [out_features, in_features],不是 W^T
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # N(0, σ²),σ² = 2 / (d_in + d_out),截断在 [-3σ, 3σ]
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features] -> [..., out_features]
        return x.matmul(self.W.T)