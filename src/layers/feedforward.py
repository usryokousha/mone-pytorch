import torch
import torch.nn as nn
import torch.nn.functional as F
from .nested_linear import NestedLinearExpand, NestedLinearContract

from typing import Optional, Callable


class NestedFeedForward(nn.Module):
    """
    Nested FeedForward layer with token-wise expert assignment.
    """

    def __init__(
        self,
        in_features: int,
        mlp_ratio: int = 4,
        out_features: Optional[int] = None,
        activation: Callable = nn.GELU,
        num_experts: int = 4,
        drop_rate: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.mlp_ratio = mlp_ratio
        self.activation = activation
        if out_features is None:
            out_features = in_features
        self.proj1 = NestedLinearExpand(
            in_features, in_features * mlp_ratio, num_experts, bias=bias
        )
        self.proj2 = NestedLinearContract(
            in_features * mlp_ratio, out_features, num_experts, bias=bias
        )
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x, token_mask)
        x = self.activation(x)
        x = self.drop(x)
        x = self.proj2(x, token_mask)
        x = self.drop(x)
        return x


class NestedSwiGLUFeedForward(nn.Module):
    """
    Nested SwiGLU FeedForward layer with token-wise expert assignment.
    """

    def __init__(
        self,
        in_features: int,
        mlp_ratio: int = 2,
        out_features: Optional[int] = None,
        activation: Callable = None,
        num_experts: int = 4,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.mlp_ratio = mlp_ratio
        if out_features is None:
            out_features = in_features
        self.proj1 = NestedLinearExpand(
            in_features, in_features * mlp_ratio, num_experts
        )
        self.proj2 = NestedLinearContract(
            in_features * mlp_ratio, out_features, num_experts
        )

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x, token_mask)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.proj2(x, token_mask)
        return x
