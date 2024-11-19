import math
import hydra
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig
from typing import List, Dict, Any


def scale_lr(cfg: DictConfig) -> float:
    return cfg.optimizer.lr * cfg.data.batch_size / 256


def group_params(
    model: nn.Module,
    weight_decay: float = 0.0,
    decay_modules: List[nn.Module] = [nn.Linear],
    no_decay_modules: List[nn.Module] = [nn.LayerNorm, nn.Embedding],
) -> List[Dict[str, Any]]:
    """
    Group model parameters into parameter groups.
    """
    decay = set()
    no_decay = set()
    for mod_name, mod in model.named_modules():
        for param_name, param in mod.named_parameters():
            full_param_name = f"{mod_name}.{param_name}" if mod_name else param_name
            if full_param_name.endswith("bias"):
                no_decay.add(param)
            elif full_param_name.endswith("weight") and isinstance(mod, decay_modules):
                decay.add(param)
            elif full_param_name.endswith("weight") and isinstance(
                mod, no_decay_modules
            ):
                no_decay.add(param)
    param_dict = {param_name: param for param_name, param in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), f"parameters {inter_params} are in both decay and no decay!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {param_dict.keys() - union_params} not separated"
    optim_groups = [
        {
            "params": [
                param_dict[param_name] for param_name in sorted(list(union_params))
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    return optim_groups

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
    model: nn.Module,
    cfg: DictConfig,
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
    param_groups = group_params(
        model,
        weight_decay=cfg.optimizer.weight_decay,
    )

    # Initialize optimizer and scale learning rate
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=param_groups)
    optimizer.param_groups[0]['lr'] = scale_lr(cfg)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        cfg.scheduler.num_warmup_steps,
        cfg.scheduler.num_training_steps,
    )

    return optimizer, scheduler