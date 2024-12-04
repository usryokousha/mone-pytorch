import torch
from lightning import Fabric
from omegaconf import DictConfig
from mone_pytorch.layers.routing import compute_capacity_distribution
from mone_pytorch.models.nested_vit import nested_vit
from mone_pytorch.models.vit import vit
def initialize_model(cfg: DictConfig, fabric: Fabric):
    if "nested" in cfg.model.name:
        if fabric.is_global_zero:
            capacity_distribution = compute_capacity_distribution(
                e_c=cfg.mone.effective_capacity,
                E=cfg.mone.num_experts,
                delta=cfg.mone.delta,
                beta=cfg.mone.beta
            )
            # convert to tensor
            capacity_distribution = torch.tensor(capacity_distribution)
        else:
            capacity_distribution = torch.zeros(cfg.mone.num_experts)

        # send to other process
        capacity_distribution = fabric.broadcast(capacity_distribution)

        # Initialize the ViT model with the computed capacity distribution
        model = nested_vit(
            img_size=cfg.model.img_size,
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            capacity_dist=capacity_distribution.tolist(),
            num_experts=cfg.mone.num_experts,
            qkv_bias=cfg.model.qkv_bias,
            mlp_ratio=cfg.model.mlp_ratio,
            mlp_type=cfg.model.mlp_type,
            block_type=cfg.model.block_type,
            num_classes=cfg.data.num_classes,
        )
    else:
        model = vit(
            img_size=cfg.model.img_size,
            patch_size=cfg.model.patch_size,
            num_classes=cfg.data.num_classes,
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            qkv_bias=cfg.model.qkv_bias,
            mlp_type=cfg.model.mlp_type,
            block_type=cfg.model.block_type,
        )

    fabric.print(f"{cfg.model.name} model initialized")

    return model
