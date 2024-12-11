import torch
import torch.nn as nn
import torch.nn.functional as F
from .nested_linear import nested_mlp, nested_swiglu_mlp, _check_nested_linear

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
        act_layer: Callable = nn.GELU(),
        drop_rate: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.act_layer = act_layer()
        if out_features is None:
            out_features = in_features
        self.proj1 = nn.Linear(in_features, in_features * mlp_ratio, bias=bias)
        self.proj2 = nn.Linear(in_features * mlp_ratio, out_features, bias=bias)
        self.drop_rate = drop_rate

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
        num_experts: int = 1,
    ) -> torch.Tensor:
        if _check_nested_linear(expert_mask, num_experts):
            x = nested_mlp(
                x,
                self.proj1.weight,
                self.proj2.weight,
                expert_mask,
                self.proj1.bias,
                self.proj2.bias,
                self.act_layer,
                self.drop_rate,
                num_experts,
                training=self.training,
            )
        else:
            x = self.proj1(x)
            x = self.act_layer(x)
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
        act_layer: Callable = nn.SiLU,
        drop_rate: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        if out_features is None:
            out_features = in_features
        self.act_layer = act_layer()
        self.proj1 = nn.Linear(in_features, in_features * mlp_ratio, bias=bias)
        self.proj2 = nn.Linear(in_features * mlp_ratio, out_features, bias=bias)
        self.drop_rate = drop_rate

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
        num_experts: int = 1,
    ) -> torch.Tensor:
        if _check_nested_linear(expert_mask, num_experts):
            x = nested_swiglu_mlp(
                x,
                self.proj1.weight,
                self.proj2.weight,
                expert_mask,
                self.proj1.bias,
                self.proj2.bias,
                num_experts,
                self.act_layer,
                self.drop_rate,
            )
        else:
            x1, x2 = torch.chunk(x, 2, dim=-1)
            x = x1 * self.act_layer(x2)
            x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.proj2(x)
        return x