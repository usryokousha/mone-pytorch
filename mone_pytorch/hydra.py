# Copyright (c) 2024, Sukjun Hwang, Aakash Lahoti, Ratish Puduppully, Tri Dao, Albert Gu.
# Based on https://github.com/goombalab/hydra/blob/main/hydra/modules/hydra.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from layers import NestedLinear

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    RMSNormGated = None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from typing import Tuple, Callable, Optional

class NestedHydra(nn.Module):
    def __init__(
            self,
            dim: int,
            state_dim: int = 64,
            kernel_size: int = 7,
            expansion_factor: int = 2,
            num_heads: int = 8,
            num_groups: int = 1,
            num_experts: int = 4,
            delta_range: Tuple[float, float] = (0.001, 0.1),
            delta_limit: Optional[Tuple[float, float]] = None,
            delta_init_cutoff: float = 1e-4,
            learnable_init_state: bool = False,
            activation: Callable = F.silu,
            linear_bias: bool = False,
            conv_bias: bool = True,
            conv_init_scale: Optional[float] = None,
            chunk_size: int = 256,

    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.kernel_size = kernel_size
        self.num_experts = num_experts
        self.expansion_factor = expansion_factor
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.delta_range = delta_range
        self.delta_limit = delta_limit
        self.delta_init_cutoff = delta_init_cutoff
        self.learnable_init_state = learnable_init_state
        self.chunk_size = chunk_size
        self.inner_dim = dim * expansion_factor
        self.head_dim = self.inner_dim // num_heads
        self.activation = activation
        assert self.inner_dim % num_heads == 0, "Inner dim must be divisible by num_heads"

        self.z_proj = NestedLinear(dim, self.inner_dim, bias=linear_bias, num_experts=num_experts, expert_expansion=False)
        self.x_proj = NestedLinear(dim, self.inner_dim, bias=linear_bias, num_experts=num_experts, expert_expansion=False)
        self.B_proj = NestedLinear(dim, self.num_groups * self.state_dim, bias=linear_bias, num_experts=num_experts, expert_expansion=False)
        self.C_proj = NestedLinear(dim, self.num_groups * self.state_dim, bias=linear_bias, num_experts=num_experts, expert_expansion=False)
        self.delta_proj = NestedLinear(dim, 2 * self.num_heads, bias=linear_bias, num_experts=num_experts, expert_expansion=False)

        conv_dim = self.inner_dim + 2 * (2 * self.num_groups * self.state_dim) + 2 * self.num_heads
        self.conv = nn.Conv1d(conv_dim, conv_dim, kernel_size, bias=conv_bias, padding=kernel_size // 2)
        self.delta_bias = nn.Parameter(self._init_delta_bias())
        self.delta_bias._no_weight_decay = True

        # A parameter in state space equation
        self.log_A = nn.Parameter(torch.ones(self.num_heads).log())
        self.log_A._no_weight_decay = True

        # D parameter in state space equation
        self.D_proj = nn.Linear(self.inner_dim, self.num_heads, bias=True)
        self.D_proj.bias._no_weight_decay = True

        self.norm = RMSNormGated(self.inner_dim, eps=1e-5, norm_before_gate=True)
        self.output_proj = NestedLinear(self.inner_dim, self.dim, bias=linear_bias, num_experts=num_experts, expert_expansion=False)

        if learnable_init_state:
            self.init_state = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, self.state_dim))
            self.init_state._no_weight_decay = True

        self._init_weights()

    def _init_delta_bias(self):
        delta = torch.exp(torch.rand(self.num_heads) * 
                          (math.log(self.delta_range[1]) - math.log(self.delta_range[0])) + 
                          math.log(self.delta_range[0]))
        delta = torch.clamp(delta, min=self.delta_init_cutoff)
        delta = delta + torch.log(-torch.expm1(delta))
        return delta
                          

    def _init_weights(self):
        if self.conv_init_scale is not None:
            nn.init.uniform_(self.conv.weight, -self.conv_init_scale, self.conv_init_scale)

    def forward(self, u: torch.Tensor, token_mask: torch.Tensor,seq_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            u: Input tensor of shape (batch_size, seq_len, dim)
            token_mask: Mask tensor of shape (batch_size, seq_len)
            seq_idx: Sequence index tensor of shape (batch_size, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = u.shape
        assert token_mask.shape == (batch_size, seq_len)

        z = self.z_proj(u)
        x = self.x_proj(u)
        B = self.B_proj(u)
        C = self.C_proj(u)
        delta = self.delta_proj(u)

        A = -torch.exp(self.log_A)
        init_state = self.init_state.repeat(batch_size, 1, 1)
        delta_limit = {} if self.delta_limit is None else {"dt_limit": self.delta_limit}

        delta = torch.cat([delta[:, :, :self.num_heads], 
                           torch.flip(delta[:, :, self.num_heads:], dims=(1,))], dim=0)
        delta = F.softplus(delta + self.delta_bias)

        # 1D Convolution over tokens
        x = self.activation(self.conv(x.transpose(1, 2)).transpose(1, 2))
        
       
        # Apply the state space equation
        
        
        