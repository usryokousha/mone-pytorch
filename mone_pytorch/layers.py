
import os
import time
import warnings
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

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

class NestedLinear(nn.Linear):
    """
    Nested Linear layer with token-wise expert assignment.

    This layer extends the standard Linear layer to support multiple experts,
    where each expert operates on a subset of the input or output features.
    The expert assignment is determined by a token mask.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        num_experts (int): Number of experts.
        expert_expansion (bool, optional): If True, experts operate on input features.
                                    If False, experts operate on output features.
                                    Defaults to True.
        bias (bool, optional): If set to False, the layer will not learn an additive bias.
                               Defaults to True.
        **kwargs: Additional arguments passed to nn.Linear.

    Attributes:
        num_experts (int): Number of experts.
        expert_dim (List[int]): List of feature dimensions for each expert.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        expert_expansion: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(in_features, out_features, bias, **kwargs)
        self.num_experts = num_experts
        self.expert_expansion = expert_expansion

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_features).
            token_mask (torch.Tensor): Tensor of shape (batch_size, seq_len) containing
                                       expert assignments for each token.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, out_features).
        """
    
        if self.expert_expansion:
            return linear_expert_expansion(
                x, self.weight, self.bias, token_mask, self.num_experts
            )
        else:
            return linear_expert_contraction(
                x, self.weight, self.bias, token_mask, self.num_experts
            )


def linear_expert_expansion(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    token_mask: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    batch, seq_len, in_dim = x.shape
    out_dim = w.shape[1]
    output = torch.zeros((batch * seq_len, in_dim), device=x.device, dtype=x.dtype)
    x = x.reshape(batch * seq_len, in_dim)
    for m in range(1, num_experts + 1):
        # get the valid mask for the m-th expert
        valid_mask = (token_mask == m).view(batch * seq_len)
        N_m = valid_mask.sum().item()
        exponent = num_experts - m

        # skip if no tokens are assigned to the m-th expert
        if N_m == 0:
            continue

        D_m = (in_dim >> exponent)

        # slice the input and weight
        x_m = x[valid_mask, :D_m]
        w_m = w[:, :D_m]

        # project up to the expert dim
        output[valid_mask, :] = F.linear(x_m, w_m, b)

    return output.reshape(batch, seq_len, out_dim)


def linear_expert_contraction(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    token_mask: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    batch, seq_len, in_dim = x.shape
    out_dim = w.shape[0]
    output = torch.zeros((batch * seq_len, out_dim), device=x.device, dtype=x.dtype)
    x = x.reshape(batch * seq_len, in_dim)
    for m in range(1, num_experts + 1):
        # get the valid mask for the m-th expert
        valid_mask = (token_mask == m).view(batch * seq_len)
        N_m = valid_mask.sum().item()

        # skip if no tokens are assigned to the m-th expert
        if N_m == 0:
            continue

        exponent = num_experts - m
        D_m = (in_dim >> exponent)

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
    def __init__(self, in_features: int, num_experts: int, expansion_factor: int = 4, activation: nn.Module = nn.GELU()):
        super().__init__()
        self.num_experts = num_experts
        self.expansion_factor = expansion_factor
        self.activation = activation
        self.proj1 = NestedLinear(in_features, in_features * expansion_factor, num_experts, expert_expansion=True)
        self.proj2 = NestedLinear(in_features * expansion_factor, in_features, num_experts, expert_expansion=False)

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
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_experts: int = 4,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = NestedLinear(dim, dim, num_experts, expert_expansion=True, bias=qkv_bias)
        self.k = NestedLinear(dim, dim, num_experts, expert_expansion=True, bias=qkv_bias)
        self.v = NestedLinear(dim, dim, num_experts, expert_expansion=True, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = NestedLinear(dim, dim, num_experts, expert_expansion=False, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, input_tokens: torch.Tensor, token_mask: torch.Tensor, attn_bias=None) -> torch.Tensor:
        B, N, C = input_tokens.shape

        q = self.q(input_tokens, token_mask).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k(input_tokens, token_mask).reshape(B, N, self.num_heads, C // self.num_heads)
        v = self.v(input_tokens, token_mask).reshape(B, N, self.num_heads, C // self.num_heads)

        if XFORMERS_AVAILABLE:
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        else:
            if attn_bias is not None:
                raise ValueError("Only support xFormers for now")
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.reshape([B, N, C])
        x = self.proj(x, token_mask)
        x = self.proj_drop(x)

        return x + input_tokens
    

def remove_padding(x: torch.Tensor, num_experts: int, token_mask: torch.Tensor) -> torch.Tensor:
    batch, seq_len, in_dim = x.shape
    expert_dim = [
                x.shape[-1] // 2**i for i in reversed(range(num_experts))
            ]
    
    x = x.reshape(batch * seq_len, in_dim)
    output = []
    for m in range(1, num_experts + 1):
        valid_mask = (token_mask == m).view(batch * seq_len)
        N_m = valid_mask.sum().item()
        if N_m == 0:
            continue
        output.append(x[valid_mask, :expert_dim[m-1]].reshape(batch, N_m // batch, expert_dim[m-1]))
    return output

# Test the layers
num_experts = 4
in_features = 32
out_features = 32

x = torch.randn(1, 10, in_features).cuda()
token_mask = torch.randint(1, num_experts, (1, 10)).cuda()

linear_expansion = NestedLinear(in_features, out_features, num_experts, expert_expansion=True).cuda()
y_expansion = linear_expansion(x, token_mask)
print("x:", x.cpu())
print("y_expansion:", y_expansion.cpu())
print("all tokens are assigned to an expert:", torch.all(y_expansion != 0.))

linear_contraction = NestedLinear(in_features, out_features, num_experts, expert_expansion=False).cuda()
y_contraction = linear_contraction(x, token_mask)
print("y_contraction:", y_contraction.cpu())
print("all tokens are assigned to an expert:", torch.all(y_contraction != 0.))
y_contraction_unpad = remove_padding(y_contraction, num_experts, token_mask)
print("Shape of each unpadded output:", [y_i.shape for y_i in y_contraction_unpad])

def benchmark_nested_linear(linear_expansion, input_tokens, token_mask):
    start_time = time.time()
    linear_expansion(input_tokens, token_mask)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    end_time = time.time()
    return end_time - start_time

def benchmark_default_linear(input_tokens, weight, bias):
    start_time = time.time()
    F.linear(input_tokens, weight, bias)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    end_time = time.time()
    return end_time - start_time

print(f"Execution time: {benchmark_nested_linear(linear_expansion, x, token_mask):.6f} seconds")
print(f"Execution time: {benchmark_default_linear(x, linear_expansion.weight, linear_expansion.bias):.6f} seconds")
