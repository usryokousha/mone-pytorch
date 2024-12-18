# adapted from dinov2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mone_pytorch.layers.patch_embed import PatchEmbed
from mone_pytorch.layers.block import Block, ParallelBlock
from mone_pytorch.layers.attention import attention_fn
from mone_pytorch.layers.pooling import AttentionPooling
from mone_pytorch.layers.mlp import MLP, SwiGLUMLP
from mone_pytorch.layers.routing import EPR
from mone_pytorch.utils.initialization import lecun_normal_
from typing import Union, Tuple, Set, Optional, Callable

mlp_layer = {
    "mlp": MLP,
    "swiglu": SwiGLUMLP,
}

block_layer = {
    "sequential": Block,
    "parallel": ParallelBlock,
}


def build_vit(
    block_type: str = "sequential",
    mlp_type: str = "mlp",
    attention_type: str = "flash",
    **kwargs,
):
    return VisionTransformer(
        block_layer=block_layer[block_type],
        mlp_layer=mlp_layer[mlp_type],
        attention_fn=attention_fn[attention_type],
        **kwargs,
    )


# copied from timm
def resample_pos_embed(
    pos_embed: torch.Tensor,
    new_size: Tuple[int, int],
    old_size: Optional[Tuple[int, int]] = None,
    num_prefix_tokens: int = 1,
    interpolation: str = "bicubic",
    antialias: bool = True,
    verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = pos_embed.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return pos_embed

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        pos_embed_prefix, pos_embed = (
            pos_embed[:, :num_prefix_tokens],
            pos_embed[:, num_prefix_tokens:],
        )
    else:
        pos_embed_prefix, pos_embed = None, pos_embed

    # do the interpolation
    embed_dim = pos_embed.shape[-1]
    orig_dtype = pos_embed.dtype
    pos_embed = pos_embed.float()  # interpolate needs float32
    pos_embed = pos_embed.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(
        pos_embed, size=new_size, mode=interpolation, antialias=antialias
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    pos_embed = pos_embed.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if pos_embed_prefix is not None:
        pos_embed = torch.cat([pos_embed_prefix, pos_embed], dim=1)

    return pos_embed


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        mlp_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.RMSNorm,
        mlp_layer=MLP,
        block_layer=Block,
        cls_token=False,
        global_pooling="avg",
        num_routers=0,
        num_experts=1,
        router_bias=True,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Add cls token
        if cls_token and global_pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        # Position embedding for patches + cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.Sequential(
            *[
                block_layer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_bias=mlp_bias,
                    qk_scale=qk_scale,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    mlp_layer=mlp_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # Router
        self.num_routers = num_routers
        self.num_experts = num_experts
        self.routers = (
            nn.ModuleList(
                [
                    EPR(embed_dim, num_experts, bias=router_bias)
                    for _ in range(num_routers)
                ]
            )
            if num_routers >= 1
            else None
        )

        # Initialize weights
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def init_weights(self) -> None:
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_jax, self)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token"}

    def set_input_size(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        self.patch_embed.set_input_size(img_size, patch_size)

    def resample_pos_embed(self, x, w, h):
        # Calculate new size based on input dimensions and patch size
        new_size = (
            w // self.patch_embed.patch_size,
            h // self.patch_embed.patch_size,
        )

        # Use the more robust resample function
        return resample_pos_embed(
            pos_embed=self.pos_embed,
            new_size=new_size,
            num_prefix_tokens=1,  # for cls token
            verbose=True,
        ).to(x.dtype)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.resample_pos_embed(x, w, h)

        return self.pos_drop(x)

    def forward_features(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        x = self.prepare_tokens(x)
        expert_mask = None
        expert_probs = None
        for i, blk in enumerate(self.blocks):
            if self.routers is not None:
                router_idx = i % self.num_routers
                expert_mask, expert_probs = self.routers[router_idx](x, c=c)
            x = blk(
                x,
                attn_mask=attn_mask,
                expert_mask=expert_mask,
                expert_probs=expert_probs,
                num_experts=self.num_experts,
            )
        x = self.norm(x)
        return x

    def pooling(self, x):
        if self.global_pooling == "avg":
            x = x[:, 0]
        elif self.global_pooling == "cls":
            x = x[:, 1:]
        elif self.global_pooling == "attn":
            x = AttentionPooling(
                dim=self.dim,
                latent_size=1,
                num_heads=self.num_heads,
                qkv_bias=self.qkv_bias,
                proj_bias=self.proj_bias,
                drop_rate=self.drop_rate,
            )(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        x = self.forward_features(x, c=c, attn_mask=attn_mask)
        x = self.pooling(x)
        output = self.head(x)
        return output


def named_apply(
    fn: Callable,
    module: nn.Module,
    name="",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


# modified from timm
def init_weights_jax(module: nn.Module, name: str = "", head_bias: float = 0.0) -> None:
    """ViT weight initialization matching Jax"""
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                (
                    nn.init.normal_(module.bias, std=1e-6)
                    if "mlp" in name
                    else nn.init.zeros_(module.bias)
                )
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()
