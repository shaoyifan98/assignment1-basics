import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int,
                 device: torch.device | None = None):
        super().__init__()
        assert d_k % 2 == 0, "d_k 必须是偶数,因为要两两配对"
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 每一对的频率:1 / theta^(2k/d_k),k = 0, 1, ..., d_k/2 - 1
        k = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (k / d_k))                    # [d_k/2]

        # 每个位置、每一对的旋转角度
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]          # [max_seq_len, d_k/2]

        # 预计算 cos / sin;不是参数,也不存进 state_dict
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: [..., seq_len, d_k]
        # token_positions: [..., seq_len](整数)
        cos = self.cos[token_positions]    # [..., seq_len, d_k/2]
        sin = self.sin[token_positions]

        x_even = x[..., 0::2]              # 每一对的第 1 个分量 [..., seq_len, d_k/2]
        x_odd = x[..., 1::2]               # 每一对的第 2 个分量

        # 如果 x 比 token_positions 多一个头维度,给 cos/sin 补上,便于广播
        while cos.dim() < x_even.dim():
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # 交错放回:[..., seq_len, d_k/2, 2] -> [..., seq_len, d_k]
        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
        return out.to(x.dtype)