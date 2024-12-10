# Attention block using MoNE components
import torch
import torch.nn as nn
import torch.nn.functional as F
from mone_pytorch.layers.mlp import NestedMLP, MLP
from mone_pytorch.layers.attention import Attention, BlockMask
from mone_pytorch.layers.nested_linear import (
    nested_linear_expand,
    nested_linear_contract,
)
from torch.nn.attention.flex_attention import flex_attention

from typing import Optional, Callable, Union, Tuple


# https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/drop.py
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = MLP,
        attn_fn: Callable = F.scaled_dot_product_attention,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            attn_fn=attn_fn,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[Union[torch.Tensor, BlockMask]] = None,
        expert_mask: Optional[torch.Tensor] = None,
        num_experts: int = 1,
    ) -> torch.Tensor:
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), mask, expert_mask, num_experts))
        )
        x = x + self.drop_path2(
            self.ls2(self.mlp(self.norm2(x), expert_mask, num_experts))
        )
        return x


class ParallelBlock(nn.Module):
    """Parallel ViT block (MLP & Attention in parallel)
    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: Optional[nn.Module] = None,  # placeholder for nested mlp
        attn_fn: Callable = F.scaled_dot_product_attention,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer("qkv_bias", None)
            self.register_parameter("mlp_bias", None)
        else:
            self.register_buffer("qkv_bias", torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(mlp_hidden_dim + dim, 2 * dim)

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()

        self.ls = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn_fn = attn_fn

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[Union[torch.Tensor, BlockMask]] = None,
        expert_mask: Optional[torch.Tensor] = None,
        num_experts: int = 1,
    ) -> torch.Tensor:
        B, N, C = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        # Combine the bias terms
        bias = torch.cat((self.qkv_bias, self.mlp_bias)) if self.mlp_bias is not None else self.qkv_bias
        
        # Apply linear transformation with appropriate function based on num_experts
        if num_experts > 1:
            y = nested_linear_expand(
                y,
                self.in_proj.weight,
                expert_mask,
                bias,
                num_experts,
            )
        else:
            y = F.linear(y, self.in_proj.weight, bias) if bias is not None else self.in_proj(y)
            
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.attn_fn is F.scaled_dot_product_attention:
            assert isinstance(
                mask, torch.Tensor
            ), "block_mask must be a tensor for scaled_dot_product_attention"
            x_attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        elif self.attn_fn is flex_attention:
            assert isinstance(
                mask, BlockMask
            ), "block_mask must be a BlockMask for flex_attention"
            x_attn = flex_attention(q, k, v, block_mask=mask, scale=self.scale)
            x_attn = F.dropout(x_attn, p=self.attn_drop, training=self.training)
        else:
            raise ValueError(f"Unsupported attention function: {self.attn_fn}")

        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)

        # MLP activation, dropout, fc2
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp_attn = torch.cat([x_mlp, x_attn], dim=-1)
        if num_experts > 1:
            x_mlp_attn = nested_linear_contract(
                x_mlp_attn,
                self.out_proj.weight,
                expert_mask,
                self.out_proj.bias,
                num_experts,
            )
        else:
            x_mlp_attn = self.out_proj(x_mlp_attn)
        x_mlp_out, x_attn_out = torch.chunk(x_mlp_attn, 2, dim=-1)

        # Add residual w/ drop path & layer scale applied
        y = self.drop_path(self.ls(x_attn_out + x_mlp_out))
        x = x + y
        return x


