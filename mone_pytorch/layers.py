import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.nested_linear_triton import (
    nested_linear_contract as nested_linear_contract_triton,
)
from ops.nested_linear_triton import (
    nested_linear_expand as nested_linear_expand_triton,
)


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")

CUDA_AVAILABLE = torch.cuda.is_available()


class NestedLinearExpand(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_experts=4,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if CUDA_AVAILABLE:
            return nested_linear_expand_triton(
                x, self.weight, self.bias, token_mask, self.num_experts
            )
        else:
            return nested_linear_expand(
                x, self.weight, self.bias, token_mask, self.num_experts
            )


class NestedLinearContract(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_experts=4,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if CUDA_AVAILABLE:
            return nested_linear_contract_triton(
                x, self.weight, self.bias, token_mask, self.num_experts
            )
        else:
            return nested_linear_contract(
                x, self.weight, self.bias, token_mask, self.num_experts
            )


@torch.compile
def nested_linear_expand(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    token_mask: torch.Tensor,
    num_experts: int = 4,
) -> torch.Tensor:
    batch, seq_len, in_dim = x.shape
    in_dim = w.shape[0]
    out_dim = w.shape[1]
    output = torch.zeros((batch * seq_len, in_dim), device=x.device, dtype=x.dtype)
    x = x.reshape(batch * seq_len, in_dim)
    for m in range(num_experts):
        # get the valid mask for the m-th expert
        valid_mask = (token_mask == m).view(batch * seq_len)
        N_m = valid_mask.sum().item()

        # skip if no tokens are assigned to the m-th expert
        if N_m == 0:
            continue

        D_m = in_dim >> (num_experts - m - 1)

        # slice the input and weight
        x_m = x[valid_mask, :D_m]
        w_m = w[:, :D_m]

        # project up to the expert dim
        output[valid_mask, :] = F.linear(x_m, w_m, b)

    return output.reshape(batch, seq_len, out_dim)


@torch.compile
def nested_linear_contract(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    token_mask: torch.Tensor,
    num_experts: int = 4,
) -> torch.Tensor:
    batch, seq_len, in_dim = x.shape
    in_dim = w.shape[0]
    out_dim = w.shape[0]
    output = torch.zeros((batch * seq_len, out_dim), device=x.device, dtype=x.dtype)
    x = x.reshape(batch * seq_len, in_dim)
    for m in range(num_experts):
        # get the valid mask for the m-th expert
        valid_mask = (token_mask == m).view(batch * seq_len)
        N_m = valid_mask.sum().item()

        # skip if no tokens are assigned to the m-th expert
        if N_m == 0:
            continue

        D_m = out_dim >> (num_experts - m - 1)

        # slice the input and weight
        x_m = x[valid_mask]
        w_m = w[:D_m, :]

        # project down to the expert dim
        if b is not None:
            b_m = b[:D_m]
        else:
            b_m = None
        pad_width = (0, out_dim - D_m)
        y = F.linear(x_m, w_m, b_m)
        output[valid_mask, :] = F.pad(y, pad_width)

    return output.reshape(batch, seq_len, out_dim)


class NestedFeedForward(nn.Module):
    """
    Nested FeedForward layer with token-wise expert assignment.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: int = 4,
        activation: nn.Module = nn.GELU(),
        num_experts: int = 4,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expansion_factor = expansion_factor
        self.activation = activation
        self.proj1 = NestedLinearExpand(dim, dim * expansion_factor, num_experts)
        self.proj2 = NestedLinearContract(dim * expansion_factor, dim, num_experts)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x, token_mask)
        x = self.activation(x)
        x = self.proj2(x, token_mask)
        return x


class NestedAttention(nn.Module):
    """
    Nested Attention layer with token-wise expert assignment.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_experts: int = 4,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = len(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = NestedLinearExpand(dim, dim, num_experts, bias=qkv_bias)
        self.k = NestedLinearExpand(dim, dim, num_experts, bias=qkv_bias)
        self.v = NestedLinearExpand(dim, dim, num_experts, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = NestedLinearContract(dim, dim, num_experts, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, input_tokens: torch.Tensor, token_mask: torch.Tensor, attn_bias=None
    ) -> torch.Tensor:
        B, N, _ = input_tokens.shape

        q = self.q(input_tokens, token_mask).reshape(
            B, N, self.num_heads, self.dim // self.num_heads
        )
        k = self.k(input_tokens, token_mask).reshape(
            B, N, self.num_heads, self.dim // self.num_heads
        )
        v = self.v(input_tokens, token_mask).reshape(
            B, N, self.num_heads, self.dim // self.num_heads
        )

        if XFORMERS_AVAILABLE:
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        else:
            if attn_bias is not None:
                raise ValueError("Only support xFormers for now")
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.reshape([B, N, self.dim])
        x = self.proj(x, token_mask)
        x = self.proj_drop(x)

        return x
