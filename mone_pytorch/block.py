# Attention block using MoNE components

import torch
import torch.nn as nn

from mone_pytorch.layers import NestedFeedForward, NestedAttention
from mone_pytorch.routing import ExpertPreferredRouter, NestedCombine

from typing import List

class NestedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int,
        capacity_distribution: List[float],
        jitter_noise: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.capacity_distribution = capacity_distribution
        self.jitter_noise = jitter_noise
        self.dtype = dtype

        self.norm1 = nn.LayerNorm(dim, dtype=dtype)
        self.router = ExpertPreferredRouter(dim, capacity_distribution, jitter_noise, dtype)
        self.attention = NestedAttention(dim, num_heads, num_experts, dtype)
        self.norm2 = nn.LayerNorm(dim, dtype=dtype)
        self.feed_forward = NestedFeedForward(dim, dtype)
        self.combine = NestedCombine(dim, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_mask, router_probs = self.router(x)
        z = x + self.attention(self.norm1(x), token_mask)
        z_prime = self.feed_forward(self.norm2(z))
        output_tokens = self.combine(z, z_prime, router_probs)
        return output_tokens
