# adapted from dinov2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nested_linear import NestedLinearExpand, NestedLinearContract

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")

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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = NestedLinearContract(dim, dim, num_experts, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, input_tokens: torch.Tensor, expert_mask: torch.Tensor, attn_bias=None
    ) -> torch.Tensor:
        B, N, _ = input_tokens.shape

        qkv = self.qkv(input_tokens, expert_mask)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.dim // self.num_heads)

        if XFORMERS_AVAILABLE:
            q, k, v = unbind(qkv, dim=2)
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        else:
            q, k, v = qkv.unbind(dim=-3)
            if attn_bias is not None:
                raise ValueError("Only support xFormers for now")
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.reshape([B, N, self.dim])
        x = self.proj(x, expert_mask)
        x = self.proj_drop(x)

        return x