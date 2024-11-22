# Attention block using MoNE components

import torch
import torch.nn as nn

from mone_pytorch.layers.feedforward import NestedFeedForward, NestedSwiGLUFeedForward
from mone_pytorch.layers.attention import NestedAttention
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
            dim, num_heads, num_experts, qkv_bias, proj_bias, qk_scale, attn_drop=attn_drop
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
        router_weights: Optional[torch.Tensor] = None,
        jitter_noise: float = 0.0,
    ) -> torch.Tensor:
        if self.router is not None:
            expert_mask, router_weights = self.router(x, router_weights, jitter_noise)
        else:
            router_weights = None

        if isinstance(self.router, routing.ConditionedEPR):
            router_probs = self.router.get_expert_probs(router_weights, expert_mask)
        else:
            router_probs = router_weights

        # As described in the paper
        z = x + self.attention(self.norm1(x), expert_mask)
        z_prime = self.mlp(self.norm2(z), expert_mask)

        output_tokens = z + (self.alpha * router_probs + 1) * z_prime
        return output_tokens, expert_mask, router_weights
