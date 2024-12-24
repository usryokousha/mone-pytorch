from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mone_linear import nested_mlp
from .nmoe_linear import (
    ExpertsChooseMaskedContract,
    ExpertsChooseMaskedExpand,
)


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = make_2tuple(bias)
        drop_probs = make_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class NestedExpertsMlp(Mlp):
    def forward(
        self, x: torch.Tensor, expert_mask: torch.Tensor, num_experts: int = 4
    ) -> torch.Tensor:
        if self.norm is not None:
            x = nested_mlp(
                x,
                w1=self.fc1.weight,
                w2=self.fc2.weight,
                expert_mask=expert_mask,
                b1=self.fc1.bias,
                b2=self.fc2.bias,
                ln_weight=self.norm.weight,
                ln_bias=self.norm.bias,
                num_experts=num_experts,
                drop=self.drop1.p,
                training=self.training,
            )
        else:
            x = nested_mlp(
                x,
                w1=self.fc1.weight,
                w2=self.fc2.weight,
                expert_mask=expert_mask,
                b1=self.fc1.bias,
                b2=self.fc2.bias,
                num_experts=num_experts,
                drop=self.drop1.p,
                training=self.training,
            )
        return x


class ExpertsChooseMlp(nn.Module):
    """MLP with experts choice routing"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        num_experts: int = 4,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = make_2tuple(bias)
        drop_probs = make_2tuple(drop)

        self.fc1 = ExpertsChooseMaskedContract(  
            in_features=in_features,
            out_features=hidden_features,
            num_experts=num_experts,
            bias=bias[0],
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features // num_experts)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = ExpertsChooseMaskedExpand(
            in_features=hidden_features,
            out_features=out_features,
            num_experts=num_experts,
            bias=bias[1],
        )
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(
        self, x: torch.Tensor, dispatch_mask: torch.Tensor, combine_array: torch.Tensor
    ) -> torch.Tensor:
        x = self.fc1(x, dispatch_mask)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x, combine_array)
        x = self.drop2(x)
        return x
