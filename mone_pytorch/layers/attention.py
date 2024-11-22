# adapted from dinov2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nested_linear import NestedLinearExpand, NestedLinearContract


class NestedAttention(nn.Module):
    """
    Nested Attention layer with token-wise expert assignment.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_experts: int = 4,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = len(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = NestedLinearExpand(dim, dim * 3, num_experts, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = NestedLinearContract(dim, dim, num_experts, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, input_tokens: torch.Tensor, expert_mask: torch.Tensor
    ) -> torch.Tensor:
        B, N, _ = input_tokens.shape

        qkv = self.qkv(input_tokens, expert_mask)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.dim // self.num_heads)

        q, k, v = qkv.unbind(dim=-3)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop, scale=self.scale
        )

        x = x.reshape([B, N, self.dim])
        x = self.proj(x, expert_mask)
        x = self.proj_drop(x)

        return x
