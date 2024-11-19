import torch
from lightning import Fabric
from omegaconf import DictConfig
from mone_pytorch.layers.routing import compute_capacity_distribution
from mone_pytorch.models.nested_vit import NestedVisionTransformer

def initialize_mone_model(cfg: DictConfig, fabric: Fabric):
    # Only calculate capacity distribution on main process
    if fabric.global_rank == 0:
        capacity_distribution = compute_capacity_distribution(
            e_c=cfg.mone.effective_capacity,
            E=cfg.mone.num_experts,
            delta=cfg.mone.delta,
            beta=cfg.mone.beta
        )
    else:
        capacity_distribution = None
    
    # Broadcast capacity distribution to all processes
    if fabric.world_size > 1:
        if fabric.global_rank == 0:
            tensor = torch.tensor(capacity_distribution, dtype=torch.float32)
        else:
            tensor = torch.zeros(cfg.mone.num_experts, dtype=torch.float32)
        fabric.broadcast(tensor, src=0)
        capacity_distribution = tensor.numpy().tolist()

    # Initialize the ViT model with the computed capacity distribution
    model = NestedVisionTransformer(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        blocks_per_router=cfg.model.blocks_per_router,
        capacity_distribution=capacity_distribution,
        jitter_noise=cfg.mone.jitter_noise,
        chunk_blocks=cfg.model.chunk_blocks,
        num_register_tokens=cfg.model.num_register_tokens,
    )
    
    return model 