# Attention block using MoNE components

import torch
import torch.nn as nn

from mone_pytorch.layers.feedforward import NestedFeedForward, NestedSwiGLUFeedForward
from mone_pytorch.layers.attention import NestedAttention
from mone_pytorch.layers.routing import ExpertPreferredRouter, NestedCombine

from typing import Optional, List, Callable


class NestedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Callable = nn.GELU,
        capacity_distribution: Optional[List[float]] = None,
        ffn_layer: nn.Module = NestedFeedForward,
        jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.capacity_distribution = capacity_distribution
        self.jitter_noise = jitter_noise

        self.norm1 = nn.LayerNorm(dim)
        self.router = None
        if capacity_distribution is not None:
            self.router = ExpertPreferredRouter(
                dim, capacity_distribution, jitter_noise
            )
        self.attention = NestedAttention(
            dim, num_heads, num_experts, qkv_bias, proj_bias, attn_drop=attn_drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ffn_layer(
            dim,
            mlp_ratio,
            num_experts,
            act_layer=act_layer,
            drop_rate=proj_drop,
            bias=ffn_bias,
        )
        self.alpha = nn.Parameter(torch.zeros(1.0))
        self.combine = NestedCombine(dim)

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
        router_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.router is not None:
            expert_mask, router_probs = self.router(x)
        else:
            router_probs = None
        # As described in the paper
        z = x + self.attention(self.norm1(x), expert_mask)
        z_prime = self.mlp(self.norm2(z), expert_mask)

        output_tokens = z + (self.alpha * router_probs + 1) * z_prime
        return output_tokens