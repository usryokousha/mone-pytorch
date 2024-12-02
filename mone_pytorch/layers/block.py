# Attention block using MoNE components
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
        act_layer: Callable = nn.GELU(),
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


class NestedParallelBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        proj_bias: bool = True, # placeholder to make block compatible with other blocks
        num_heads: int = 8,
        act_layer: Callable = nn.GELU(),
        mlp_bias: bool = True, # placeholder to make block compatible with other blocks
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_layer: nn.Module = None,
        norm_layer: nn.Module = nn.LayerNorm,
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
        q, kv = torch.split(qkv, [self.dim, 2 * self.dim], dim=-1)
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
        
        concat_output = nested_linear_contract(
                torch.cat([F.dropout(self.act_layer(mlp_hidden), self.proj_drop, self.training),
                          attn_output],  
                          dim=-1),
                self.contract_weight,
                expert_mask,
            self.contract_bias,
        )
        # add residual to concat_output
        mlp_output, attn_output = torch.chunk(concat_output + residual.repeat(1, 1, 2), 2, dim=-1)
        if alpha is None:
            output = attn_output + expert_probs.unsqueeze(-1) * mlp_output
        else:
            output = attn_output + (alpha * expert_probs.unsqueeze(-1) + 1) * mlp_output
        return output
