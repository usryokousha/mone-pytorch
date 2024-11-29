# Attention block using MoNE components
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mone_pytorch.layers.mlp import NestedMLP
from mone_pytorch.layers.attention import NestedAttention, _attention
from mone_pytorch.layers.nested_linear import (
    nested_linear_expand,
    nested_linear_contract,
)

from typing import Optional, List, Callable


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
        act_layer: Callable = nn.GELU,
        capacity_dist: Optional[List[float]] = None,
        norm_layer: Callable = nn.LayerNorm,
        mlp_layer: nn.Module = NestedMLP,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.capacity_dist = capacity_dist

        self.norm1 = norm_layer(dim)
        self.attention = NestedAttention(
            dim,
            num_heads,
            num_experts,
            qkv_bias,
            proj_bias,
            qk_scale,
            attn_drop=attn_drop,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            dim,
            mlp_ratio,
            num_experts,
            activation=act_layer,
            drop_rate=proj_drop,
            bias=mlp_bias,
        )
        self.alpha = nn.Parameter(torch.zeros((1,)))

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: torch.Tensor,
        expert_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # As described in the paper
        z = x + self.attention(self.norm1(x), expert_mask)
        z_prime = self.mlp(self.norm2(z), expert_mask)

        output_tokens = z + (self.alpha * expert_probs + 1) * z_prime
        return output_tokens


class NestedParallelBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        proj_bias: bool = True, # placeholder to make block compatible with other blocks
        num_heads: int = 8,
        act_layer: Callable = nn.GELU(),
        mlp_bias: bool = True, # placeholder to make block compatible with other blocks
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
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
        self.contract_weight = nn.Parameter(torch.zeros((2 * dim, expand_dim)))
        self.contract_bias = nn.Parameter(torch.zeros((2 * dim,)))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.zeros((1,)))
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
            input_bias = torch.cat(
                [self.mlp_bias, 
                 self.qkv_bias], 
                dim=-1
            )
        qkv, mlp_hidden = torch.split(
            nested_linear_expand(
                self.norm1(x), 
                self.expand_weight, 
                expert_mask, 
                input_bias
            ),
            [3 * self.dim, self.mlp_ratio * self.dim],
            dim=-1,
        )
        mlp_hidden += self.mlp_bias
        q, kv = torch.chunk(qkv, 2, dim=-1)
        k, v = torch.chunk(self.norm2(kv), 2, dim=-1)
        attn_output = _attention(
            q, 
            k, 
            v, 
            self.num_heads, 
            self.dim, 
            self.attn_drop, 
            self.scale
        )
        attn_output, mlp_output = torch.chunk(
            nested_linear_contract(
                torch.cat([attn_output, 
                           self.act_layer(mlp_hidden)], 
                          dim=-1),
                self.contract_weight,
                self.contract_bias,
            ),
            2,
            dim=-1,
        )
        attn_output = attn_output + residual
        output = attn_output + (self.alpha * expert_probs + 1) * mlp_output
        return output
