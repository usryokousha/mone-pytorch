# Attention block using MoNE components
import torch
import torch.nn as nn
import torch.nn.functional as F
from mone_pytorch.layers.mlp import NestedMLP, MLP
from mone_pytorch.layers.attention import NestedAttention, Attention, _attention
from mone_pytorch.layers.nested_linear import (
    nested_linear_expand,
    nested_linear_contract,
)

from typing import Optional, Callable


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


class SequentialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Callable = nn.GELU(),
        norm_layer: Callable = nn.RMSNorm,
        mlp_layer: nn.Module = MLP,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.proj_bias = proj_bias
        self.mlp_bias = mlp_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.mlp_layer = mlp_layer
        self.norm1 = norm_layer(dim)
        self.attention = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            mlp_ratio=mlp_ratio,
            activation=act_layer,
            drop_rate=drop_prob,
            bias=mlp_bias,
        )
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NestedSequentialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Callable = nn.GELU(),
        norm_layer: Callable = nn.LayerNorm,
        mlp_layer: nn.Module = NestedMLP,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.norm1 = norm_layer(dim)
        self.attention = NestedAttention(
            dim,
            num_heads,
            num_experts,
            qkv_bias,
            qk_scale,
            proj_bias,
            attn_drop=attn_drop,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            dim,
            mlp_ratio,
            num_experts=num_experts,
            activation=act_layer,
            drop_rate=proj_drop,
            bias=mlp_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: torch.Tensor,
        expert_probs: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        # As described in the paper
        z = x + self.attention(self.norm1(x), expert_mask)
        z_prime = self.mlp(self.norm2(z), expert_mask)

        if alpha is None:
            output_tokens = z + expert_probs.unsqueeze(-1) * z_prime
        else:
            output_tokens = z + (alpha * expert_probs.unsqueeze(-1) + 1) * z_prime
        return output_tokens


class ParallelBlock(nn.Module):
    """Naive parallel block as in https://arxiv.org/pdf/2302.05442"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        proj_bias: bool = True,
        drop_prob: float = 0.0,
        act_layer: Callable = nn.GELU(),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable = nn.RMSNorm,
        **kwargs
    ):
        super().__init__()
        expand_dim = 3 * dim + mlp_ratio * dim
        contract_dim = 2 * dim
        concat_dim = mlp_ratio * dim + dim
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.proj_bias = proj_bias
        self.drop_prob = drop_prob
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(2 * dim)
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.act_layer = act_layer
        self.expand_weight = nn.Parameter(torch.zeros((expand_dim, dim)))
        self.contract_weight = nn.Parameter(torch.zeros((contract_dim, concat_dim)))
        self.mlp_bias = nn.Parameter(torch.zeros((mlp_ratio * dim,)))
        self.contract_bias = nn.Parameter(torch.zeros((contract_dim,)))
        if qkv_bias:
            self.qkv_bias = nn.Parameter(torch.zeros((3 * dim,)))
        else:
            self.qkv_bias = None
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.expand_weight, std=0.02)
        nn.init.trunc_normal_(self.contract_weight, std=0.02)
        nn.init.zeros_(self.mlp_bias)
        nn.init.zeros_(self.contract_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.qkv_bias is None:
            input_bias = torch.cat(
                [
                    self.mlp_bias,
                    torch.zeros(
                        (3 * self.dim,),
                        dtype=self.mlp_bias.dtype,
                        device=self.mlp_bias.device,
                    ),
                ]
            )
        else:
            input_bias = torch.cat([self.mlp_bias, self.qkv_bias], dim=-1)
        qkv, mlp_hidden = torch.split(
            F.linear(self.norm1(x), self.expand_weight, bias=input_bias),
            [3 * self.dim, self.mlp_ratio * self.dim],
            dim=-1,
        )
        mlp_hidden = F.dropout(
            self.act_layer(mlp_hidden), self.proj_drop, self.training
        )
        q, kv = torch.split(qkv, [self.dim, 2 * self.dim], dim=-1)
        k, v = torch.chunk(self.norm2(kv), 2, dim=-1)
        attn_output = _attention(
            q, k, v, self.num_heads, self.dim, self.attn_drop, self.scale
        )
        mlp_output, attn_output = torch.chunk(
            F.linear(
                torch.cat([mlp_hidden, attn_output], dim=-1),
                self.contract_weight,
                bias=self.contract_bias,
            ),
            2,
            dim=-1,
        )
        attn_output = F.dropout(attn_output, self.proj_drop, self.training)
        output = self.drop_path(attn_output + mlp_output)
        output = output + residual
        return output


class NestedParallelBlock(nn.Module):
    """Nested parallel block adapted from https://arxiv.org/pdf/2302.05442"""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_heads: int = 8,
        act_layer: Callable = nn.GELU(),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs
    ):
        super().__init__()
        expand_dim = 3 * dim + mlp_ratio * dim
        self.dim = dim
        self.num_experts = num_experts
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.act_layer = act_layer
        self.expand_weight = nn.Parameter(torch.zeros((expand_dim, dim)))
        self.mlp_bias = nn.Parameter(torch.zeros((mlp_ratio * dim,)))
        if qkv_bias:
            self.qkv_bias = nn.Parameter(torch.zeros((3 * dim,)))
        else:
            self.qkv_bias = None
        contract_dim = 2 * dim
        concat_dim = mlp_ratio * dim + dim
        self.contract_weight = nn.Parameter(torch.zeros((contract_dim, concat_dim)))
        self.contract_bias = nn.Parameter(torch.zeros((contract_dim,)))
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(2 * dim)
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.expand_weight, std=0.02)
        nn.init.trunc_normal_(self.contract_weight, std=0.02)
        nn.init.zeros_(self.mlp_bias)
        nn.init.zeros_(self.contract_bias)

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: torch.Tensor,
        expert_probs: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        residual = x
        if self.qkv_bias is None:
            input_bias = torch.cat(
                [
                    self.mlp_bias,
                    torch.zeros(
                        (3 * self.dim,),
                        dtype=self.mlp_bias.dtype,
                        device=self.mlp_bias.device,
                    ),
                ],
                dim=-1,
            )
        else:
            input_bias = torch.cat([self.mlp_bias, self.qkv_bias], dim=-1)
        qkv, mlp_hidden = torch.split(
            nested_linear_expand(
                self.norm1(x), self.expand_weight, expert_mask, input_bias
            ),
            [3 * self.dim, self.mlp_ratio * self.dim],
            dim=-1,
        )
        q, kv = torch.split(qkv, [self.dim, 2 * self.dim], dim=-1)
        k, v = torch.chunk(self.norm2(kv), 2, dim=-1)
        attn_output = _attention(
            q, k, v, self.num_heads, self.dim, self.attn_drop, self.scale
        )

        concat_output = nested_linear_contract(
            torch.cat(
                [
                    F.dropout(
                        self.act_layer(mlp_hidden), self.proj_drop, self.training
                    ),
                    attn_output,
                ],
                dim=-1,
            ),
            self.contract_weight,
            expert_mask,
            self.contract_bias,
        )
        # add residual to concat_output
        mlp_output, attn_output = torch.chunk(
            concat_output + residual.repeat(1, 1, 2), 2, dim=-1
        )
        if alpha is None:
            output = attn_output + expert_probs.unsqueeze(-1) * mlp_output
        else:
            output = attn_output + (alpha * expert_probs.unsqueeze(-1) + 1) * mlp_output
        return output
