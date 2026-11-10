import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self,  d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor ) -> torch.Tensor:
        # x = [batch, sequence_length, d_model]
        input_type = x.dtype
        # avoid overflow when squaring x
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        res = x / rms * self.weight
        return res.to(input_type)
