import math
import logging
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from lightning.fabric import Fabric
from mone_pytorch.optimizer.flora import Flora

from typing import Collection

logger = logging.getLogger(__name__)


def scale_lr(
    fabric: Fabric,
    base_lr: float,
    batch_size: int,
    grad_accum: int,
    base_batch_size: int = 256,
) -> float:
    global_batch_size = batch_size * grad_accum * fabric.world_size
    batch_size_ratio = global_batch_size / base_batch_size
    lr = base_lr * batch_size_ratio
    fabric.print(
        f"Learning rate is scaled linearly from {base_lr} to {lr} "
        f"for effective batch size {global_batch_size}"
    )
    return lr


# taken from https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/_param_groups.py
def param_groups_weight_decay(
    model: nn.Module,
    weight_decay: float = 1e-5,
    no_weight_decay_list: Collection[str] = (),
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


# adapted from Torchtune implementation
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:

    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # cosine
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )

        cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        )
        return max(0.0, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def build_optimizer(
    cfg: DictConfig,
    model: nn.Module,
    fabric: Fabric,
) -> torch.optim.Optimizer:
    """
    Build optimizer with parameter groups.

    Args:
        model: The neural network model
        cfg: Configuration object containing optimizer settings

    Returns:
        Configured optimizer
    """

    # Group parameters
    weight_decay = cfg.optimizer.hparams.weight_decay
    no_weight_decay = getattr(model, "no_weight_decay", lambda: set())()
    param_groups = param_groups_weight_decay(
        model,
        weight_decay=weight_decay,
        no_weight_decay_list=no_weight_decay,
    )

    # Initialize optimizer - modify this line
    fabric.print(f"Using optimizer: {cfg.optimizer.name}")
    if cfg.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(param_groups, **cfg.optimizer.hparams)
        fabric.print("AdamW hyperparameters:")
        for k, v in cfg.optimizer.hparams.items():
            fabric.print(f"  {k}: {v}")
    elif cfg.optimizer.name == "Flora":
        optimizer = Flora(param_groups, **cfg.optimizer.hparams)
        fabric.print("Flora hyperparameters:")
        for k, v in cfg.optimizer.hparams.items():
            fabric.print(f"  {k}: {v}")
    else:
        raise ValueError(f"Optimizer {cfg.optimizer.name} not supported")

    effective_lr = scale_lr(
        fabric,
        cfg.optimizer.hparams.lr,
        cfg.training.batch_size,
        cfg.training.gradient_accumulation,
    )
    optimizer.param_groups[0]["lr"] = effective_lr

    effective_batch_size = (
        cfg.training.batch_size * cfg.training.gradient_accumulation * fabric.world_size
    )
    steps_per_epoch = cfg.data.training_size // effective_batch_size
    training_steps = cfg.training.epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(cfg.scheduler.warmup_epochs * steps_per_epoch),
        training_steps,
    )
    fabric.print(f"Total training steps: {training_steps:,}")
    return optimizer, scheduler
