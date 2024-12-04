import torch
import torch.nn as nn
import torch.nn.functional as F
from .nested_linear import nested_mlp, nested_swiglu_mlp

from typing import Optional, Callable


class MLP(nn.Module):
    """
    MLP layer.
    """

    def __init__(
        self,
        in_features: int,
        mlp_ratio: int = 4,
        out_features: Optional[int] = None,
        activation: Callable = nn.GELU(),
        drop_rate: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.activation = activation
        if out_features is None:
            out_features = in_features
        self.proj1 = nn.Linear(in_features, in_features * mlp_ratio, bias=bias)
        self.proj2 = nn.Linear(in_features * mlp_ratio, out_features, bias=bias)
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.proj2(x)
        return x


class SwiGLUMLP(nn.Module):
    """
    SwiGLU activation function.
    """

    def __init__(
        self,
        in_features: int,
        mlp_ratio: int = 2,
        out_features: Optional[int] = None,
        activation: Callable = nn.SiLU(),
        drop_rate: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        if out_features is None:
            out_features = in_features
        self.activation = activation
        self.proj1 = nn.Linear(in_features, in_features * mlp_ratio, bias=bias)
        self.proj2 = nn.Linear(in_features * mlp_ratio, out_features, bias=bias)
        self.drop_rate = drop_rate
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x = x1 * self.activation(x2)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.proj2(x)
        return x


class NestedMLP(nn.Module):
    """
    Nested MLP layer with token-wise expert assignment.
    """

    def __init__(
        self,
        in_features: int,
        mlp_ratio: int = 4,
        out_features: Optional[int] = None,
        activation: Callable = nn.GELU(),
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
        self.proj1 = nn.Linear(in_features, in_features * mlp_ratio, bias=bias)
        self.proj2 = nn.Linear(in_features * mlp_ratio, out_features, bias=bias)
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        x = nested_mlp(
            x,
            self.proj1.weight,
            self.proj2.weight,
            expert_mask,
            self.proj1.bias,
            self.proj2.bias,
            self.activation,
            self.drop_rate,
            self.num_experts,
            training=self.training,
        )
        return x


class NestedSwiGLUMLP(nn.Module):
    """
    Nested SwiGLU MLP layer with token-wise expert assignment.
    """

    def __init__(
        self,
        in_features: int,
        mlp_ratio: int = 2,
        out_features: Optional[int] = None,
        activation: Callable = None,
        num_experts: int = 4,
        drop_rate: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.mlp_ratio = mlp_ratio
        if out_features is None:
            out_features = in_features
        self.proj1 = nn.Linear(in_features, in_features * mlp_ratio, bias=bias)
        self.proj2 = nn.Linear(in_features * mlp_ratio, out_features, bias=bias)

    def forward(self, x: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        x = nested_swiglu_mlp(
            x,
            self.proj1.weight,
            self.proj2.weight,
            expert_mask,
            self.proj1.bias if self.proj1.bias is not None else None,
            self.proj2.bias if self.proj2.bias is not None else None,
            self.num_experts,
        )
        return x
