from cs336_basics.src.linear import Linear
import torch
import torch.nn as nn


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None,  device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = int (8 / 3 * d_model)
            # to better levereage the hardware
            d_ff = (d_ff + 63) // 64 * 64  # round up to multiple of 64
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)   # 门支路
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)   # 内容支路
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)   # 降维

    def silu(self, x: torch.Tensor) -> torch.Tensor:
            # SiLU(x) = x * sigmoid(x)
            return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
        