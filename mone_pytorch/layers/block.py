# Attention block using MoNE components
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mone_pytorch.layers.feedforward import NestedFeedForward, NestedSwiGLUFeedForward
from mone_pytorch.layers.attention import NestedAttention, _attention
from mone_pytorch.layers.nested_linear import (
    nested_linear_expand,
    nested_linear_contract,
)
from mone_pytorch.layers import routing

from typing import Optional, List, Callable


class NestedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Callable = nn.GELU,
        capacity_dist: Optional[List[float]] = None,
        norm_layer: Callable = nn.LayerNorm,
        ffn_layer: nn.Module = NestedFeedForward,
        router_layer: nn.Module = routing.ExpertPreferredRouter,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.capacity_dist = capacity_dist

        self.norm1 = norm_layer(dim)
        self.router = None
        if capacity_dist is not None:
            self.router = router_layer(dim, capacity_dist)
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
        self.mlp = ffn_layer(
            dim,
            mlp_ratio,
            num_experts,
            activation=act_layer,
            drop_rate=proj_drop,
            bias=ffn_bias,
        )
        self.alpha = nn.Parameter(torch.zeros((1,)))

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
        router_probs: Optional[torch.Tensor] = None,
        jitter_noise: float = 0.0,
    ) -> torch.Tensor:
        if self.router is not None:
            expert_mask, router_probs = self.router(x, router_probs, jitter_noise)
        else:
            router_probs = None

        expert_probs = self.router.get_expert_probs(router_probs, expert_mask)

        # As described in the paper
        z = x + self.attention(self.norm1(x), expert_mask)
        z_prime = self.mlp(self.norm2(z), expert_mask)

        output_tokens = z + (self.alpha * expert_probs + 1) * z_prime
        return output_tokens, expert_mask, router_probs


# TODO: possibly implement a parallel version of the block
class NestedParallelBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        num_heads: int = 8,
        act_layer: Callable = nn.GELU(),
        ffn_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        capacity_dist: Optional[List[float]] = None,
        router_layer: nn.Module = routing.ExpertPreferredRouter,
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
        self.contract_weight = nn.Parameter(torch.zeros((2 * dim, expand_dim)))
        self.contract_bias = nn.Parameter(torch.zeros((2 * dim,)))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.router = None
        if capacity_dist is not None:
            self.router = router_layer(dim, capacity_dist)
        self.alpha = nn.Parameter(torch.zeros((1,)))
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.expand_weight, std=0.02)
        nn.init.trunc_normal_(self.contract_weight, std=0.02)
        nn.init.zeros_(self.ffn_bias)
        nn.init.zeros_(self.contract_bias)

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
        router_probs: Optional[torch.Tensor] = None,
        jitter_noise: float = 0.0,
    ) -> torch.Tensor:
        residual = x
        if self.router is not None:
            expert_mask, router_probs = self.router(x, router_probs, jitter_noise)
        else:
            router_probs = None
        expert_probs = self.router.get_expert_probs(router_probs, expert_mask)
        qkv, mlp_hidden = torch.split(
            nested_linear_expand(self.norm1(x), self.expand_weight, expert_mask, None),
            [3 * self.dim, self.mlp_ratio * self.dim],
            dim=-1,
        )
        mlp_hidden += self.mlp_bias
        q, kv = torch.chunk(qkv, 2, dim=-1)
        k, v = torch.chunk(self.norm2(kv), 2, dim=-1)
        attn_output = _attention(q, k, v, self.num_heads, self.dim, self.attn_drop, self.scale)
        attn_output, mlp_output = torch.chunk(
            nested_linear_contract(
                torch.cat((attn_output, self.act_layer(mlp_hidden)), dim=-1),
                self.contract_weight,
                self.contract_bias,
            ),
            2,
            dim=-1,
        )
        attn_output = attn_output + residual
        output = attn_output + (self.alpha * expert_probs + 1) * mlp_output
        return output, expert_mask, router_probs
