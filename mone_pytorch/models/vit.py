"""
Adapted from from timm library.
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
"""

import math

import torch
import torch.nn as nn

from mone_pytorch.layers.patch_embed import PatchEmbed
from mone_pytorch.layers.block import SequentialBlock, ParallelBlock
from mone_pytorch.layers.mlp import MLP, SwiGLUMLP

mlp_layer = {
    "mlp": MLP,
    "swiglu": SwiGLUMLP,
}

block_layer = {
    "sequential": SequentialBlock,
    "parallel": ParallelBlock,
}


def vit(
    block_type: str = "sequential",
    mlp_type: str = "mlp",
    **kwargs
):
    return VisionTransformer(
        block_layer=block_layer[block_type],
        mlp_layer=mlp_layer[mlp_type],
        **kwargs
    )


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
        block_layer=SequentialBlock,
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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Position embedding for patches + cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(
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
            )

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # Initialize weights
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1  # Remove cls token
        N = self.pos_embed.shape[1] - 1  # Remove cls token position
        if npatch == N and w == h:
            return self.pos_embed.to(x.dtype)
        
        # Only interpolate patch position embeddings, handle cls token separately
        cls_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((cls_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # Use cls token for classification
        output = self.head(x[:, 0])
        return output
