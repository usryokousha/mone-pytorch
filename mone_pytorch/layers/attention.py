# adapted from dinov2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .nested_linear import NestedLinearExpand, NestedLinearContract


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    dim: int,
    attn_drop: float,
    scale: float,
) -> torch.Tensor:
    B, N, _ = q.shape
    q = q.reshape(B, N, num_heads, dim // num_heads)
    k = k.reshape(B, N, num_heads, dim // num_heads)
    v = v.reshape(B, N, num_heads, dim // num_heads)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop, scale=scale)
    x = x.reshape([B, N, dim])
    return x


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
        qk_scale: Optional[float] = None,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5 if qk_scale is None else qk_scale

        self.qkv = NestedLinearExpand(dim, dim * 3, qkv_bias, num_experts)
        self.attn_drop = attn_drop
        self.proj = NestedLinearContract(dim, dim, proj_bias, num_experts)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, input_tokens: torch.Tensor, expert_mask: torch.Tensor
    ) -> torch.Tensor:
        B, N, _ = input_tokens.shape

        qkv = self.qkv(input_tokens, expert_mask)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        x = _attention(q, k, v, self.num_heads, self.dim, self.attn_drop, self.scale)
        x = self.proj(x, expert_mask)
        x = self.proj_drop(x)

        return x
