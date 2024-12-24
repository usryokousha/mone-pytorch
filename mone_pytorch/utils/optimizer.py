import math
import logging
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from typing import Collection, Optional, Union, Tuple

logger = logging.getLogger(__name__)


def scale_lr(
    lr_base: float, 
    batch_size: int, 
    grad_accum_steps: int, 
    world_size: int, 
    base_size: int = 256
) -> Tuple[float, int]:
    global_batch_size = batch_size * grad_accum_steps * world_size
    batch_ratio = global_batch_size / base_size
    lr = lr_base * batch_ratio
    return lr, global_batch_size


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

def create_scheduler(optimizer,
                     num_warmup_epochs: int,
                     num_training_epochs: int,
                     num_cycles: float = 0.5,
                     updates_per_epoch: int = 1,
                     ) -> LambdaLR:
    num_warmup_steps = num_warmup_epochs * updates_per_epoch
    num_training_steps = num_training_epochs * updates_per_epoch
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles)


# taken from Timm implementation
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


def create_optimizer(
    model_or_params: Union[nn.Module, dict],
    lr: Optional[float] = None,
    weight_decay: float = 0.0,
    betas: Tuple[float, float] = (0.9, 0.999),
    foreach: Optional[bool] = None,
):
    """Create an optimizer instance.

    Args:
        model_or_params: Model or parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay factor
        betas: Betas for AdamW optimizer
        foreach: Enable/disable foreach operation

    Returns:
        Configured optimizer instance
    """

    # Get parameters to optimize
    if isinstance(model_or_params, nn.Module):
        # Extract parameters from a nn.Module, build param groups w/ weight-decay and/or layer-decay applied
        no_weight_decay = getattr(model_or_params, "no_weight_decay", lambda: set())()

        params = param_groups_weight_decay(
            model_or_params,
            weight_decay=weight_decay,
            no_weight_decay_list=no_weight_decay,
        )
        weight_decay = 0.
    else:
        # pass parameters / parameter groups through to optimizer
        params = model_or_params

    # Create optimizer
    optimizer = torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay, betas=betas, foreach=foreach
    )

    return optimizer
