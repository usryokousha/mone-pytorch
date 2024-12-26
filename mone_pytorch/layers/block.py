# Attention block using MoNE components
import torch
import torch.nn as nn
import torch.nn.functional as F

from mone_pytorch.layers.mlp import NestedExpertsMlp, ExpertsChooseMaskedMlp, Mlp
from mone_pytorch.layers.attention import (
    Attention,
    NestedExpertsAttention,
    ExpertsChooseAttention,
)
from mone_pytorch.layers.experts_choose_linear import (
    experts_choose_masked_contract,
    experts_choose_masked_expand,
)
from mone_pytorch.layers.routing import ExpertsChooseMaskedRouter

from typing import Optional


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# Copied from timm's LayerScale
class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# Copied from timm's Block
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        **kwargs,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            **kwargs,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class NestedExpertsBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = NestedExpertsMlp,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.norm1 = norm_layer(dim)
        self.attn = NestedExpertsAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: torch.Tensor,
        expert_probs: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        x_attn = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), expert_mask, self.num_experts))
        )
        x_mlp = self.drop_path2(
            self.ls2(self.mlp(self.norm2(x_attn), expert_mask, self.num_experts))
        )
        x = x_attn + (alpha * expert_probs + 1) * x_mlp
        return x


class ExpertsChooseBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        router_bias: bool = False,
        capacity_factor: float = 1.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = ExpertsChooseMaskedMlp,
        router_layer: nn.Module = ExpertsChooseMaskedRouter,
    ) -> None:
        super().__init__()
        self.capacity_factor = capacity_factor
        self.router = router_layer(dim, num_experts=num_experts, bias=router_bias)
        self.norm1 = norm_layer(dim)
        self.attn = ExpertsChooseAttention(
            dim,
            num_heads=num_heads,
            num_experts=num_experts,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            num_experts=num_experts,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        capacity = int(self.capacity_factor * x.shape[1])
        dispatch_mask, combine_array = self.router(x, capacity)
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), combine_array, dispatch_mask))
        )
        x = x + self.drop_path2(
            self.ls2(self.mlp(self.norm2(x), combine_array, dispatch_mask))
        )
        return x


class ExpertsChooseParallelBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        router_bias: bool = False,
        capacity_factor: float = 1.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        router_layer: nn.Module = ExpertsChooseMaskedRouter,
    ):
        super().__init__()
        attn_in_dim = 3 * dim
        mlp_in_dim = dim
        self.mlp_hidden_dim = mlp_ratio * dim
        self.qkv_dim = 3 * dim
        self.fc1_out_features = self.mlp_hidden_dim + self.qkv_dim
        self.fc2_in_features = self.mlp_hidden_dim + dim
        self.fc2_out_features = 2 * dim

        self.norm1 = norm_layer(self.dim)

        if qk_norm:
            self.q_norm = norm_layer(attn_in_dim)
            self.k_norm = norm_layer(attn_in_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.fc1 = nn.Linear(attn_in_dim, self.fc1_out_features, bias=qkv_bias)
        self.fc2 = nn.Linear(
            in_features=self.fc2_in_features,
            out_features=self.fc2_out_features,
            bias=qkv_bias,
        )
        self.mlp_act = act_layer()
        self.mlp_drop = proj_drop

        if not qkv_bias:
            self.mlp_bias = nn.Parameter(torch.zeros(dim), requires_grad=False)
            self.qkv_bias = nn.Parameter(torch.zeros(3 * dim), requires_grad=False)
        else:
            self.qkv_bias = None
            self.mlp_bias = None

        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.ls = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.router = router_layer(dim, num_experts=num_experts, bias=router_bias)
        self.capacity_factor = capacity_factor

    def update_capacity(self, capacity_factor: float):
        self.capacity_factor = capacity_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        capacity = int(self.capacity_factor * N)
        dispatch_mask, combine_array = self.router(x, capacity)

        # Prepare inputs for MLP and QKV
        y = self.norm1(x)
        if self.qkv_bias is not None:
            fc1_bias = torch.cat([self.mlp_bias, self.qkv_bias], dim=-1)
        else:
            fc1_bias = self.fc1.bias

        # Project MLP input and QKV input
        x_mlp, x_qkv = torch.split(
            experts_choose_masked_contract(
                y,
                self.fc1.weight,
                fc1_bias,
                dispatch_mask,
                num_experts=self.num_experts,
            ),
            [self.mlp_hidden_dim, self.qkv_dim],
            dim=-1,
        )
        # Perform attention on QKV input
        x_qkv = x_qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = x_qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x_attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            scale=self.scale,
        )

        x_attn = x_attn.transpose(1, 2).reshape(B, N, self.num_experts, C // self.num_experts)

        # Combine MLP and attention outputs and project
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = F.dropout(x_mlp, p=self.mlp_drop, training=self.training)
        y = torch.cat([x_mlp, x_attn], dim=-1)
        x_mlp, x_attn = torch.chunk(
            experts_choose_masked_expand(
                x,
                self.fc2.weight,
                self.fc2.bias,
                combine_array,
                num_experts=self.num_experts,
            ),
            2,
            dim=-1,
        )
        y = self.drop_path(self.ls(x_mlp + x_attn))
        x = x + y

        return x
