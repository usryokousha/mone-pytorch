# adapted from dinov2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from mone_pytorch.layers.nested_experts_linear import (
    nested_linear_expand,
    nested_linear_contract,
)
from mone_pytorch.layers.experts_choose_linear import (
    ExpertsChooseContract,
    ExpertsChooseExpand,
)
from mone_pytorch.layers.natten_mask import create_natten_mask
from typing import Optional, Tuple

# copied from timm
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        kernel_size: Optional[Tuple[int, int]] = None,
        canvas_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.kernel_size = kernel_size
        self.canvas_size = canvas_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.kernel_size is not None and self.canvas_size is not None:
            assert math.prod(self.canvas_size) == N, \
                "canvas_size area must match sequence length"
            w, h = self.canvas_size
            block_mask = create_natten_mask(
                B=B,
                H=self.num_heads,
                canvas_w=N,
                canvas_h=N,
                kernel_w=self.kernel_size[0],
                kernel_h=self.kernel_size[1],
            )
            x = flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=self.scale,
            )
        else:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop if self.training else 0.0,
                scale=self.scale,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NestedExpertsAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        num_experts: int = 4,
        kernel_size: Optional[Tuple[int, int]] = None,
        canvas_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            qkv_bias,
            qk_norm,
            attn_drop,
            proj_drop,
            norm_layer,
        )
        self.num_experts = num_experts

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            nested_linear_expand(
                x,
                self.qkv.weight,
                expert_mask,
                self.qkv.bias,
                num_experts=self.num_experts,
            )
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.kernel_size is not None and self.canvas_size is not None:
            assert math.prod(self.canvas_size) == N, \
                "canvas_size area must match sequence length"
            w, h = self.canvas_size
            block_mask = create_natten_mask(
                B=B,
                H=self.num_heads,
                canvas_w=w,
                canvas_h=h,
                kernel_w=self.kernel_size[0],
                kernel_h=self.kernel_size[1],
            )
            x = flex_attention(q, k, v, block_mask=block_mask, scale=self.scale)
        else:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = nested_linear_contract(
            x,
            self.proj.weight,
            expert_mask,
            self.proj.bias,
            num_experts=self.num_experts,
        )
        x = self.proj_drop(x)
        return x


class ExpertsChooseAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_experts: int = 4,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.head_dim = dim // num_experts * num_heads
        self.heads_per_expert = num_heads // num_experts
        self.scale = self.head_dim**-0.5
        assert (
            self.head_dim % num_heads == 0
        ), "head_dim should be divisible by num_heads"
        self.qkv = ExpertsChooseContract(dim, dim * 3, num_experts, bias=qkv_bias)
        self.proj = ExpertsChooseExpand(dim, dim, num_experts)
        self.proj_drop = nn.Dropout(proj_drop)
        self.q_norm = norm_layer(dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop

    def forward(
        self, x: torch.Tensor, combine_array: torch.Tensor, dispatch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x, dispatch_mask)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )

        x = x.transpose(1, 2).reshape(B, N, self.num_experts, C // self.num_experts)
        x = self.proj(x, combine_array)
        x = self.proj_drop(x)
        return x
