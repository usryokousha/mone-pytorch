import math
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from lightning.fabric import Fabric
from mone_pytorch.optimizer.flora import Flora

from typing import List, Dict, Any, Tuple


def scale_lr(
    lr: float,
    batch_size: int,
    grad_accum: int,
    num_gpus: int,
    base_batch_size: int = 256,
) -> float:
    return lr * (batch_size * grad_accum * num_gpus) / base_batch_size


def group_params(
    model: nn.Module,
    weight_decay: float = 0.0,
    decay_modules: Tuple[nn.Module] = (nn.Linear,),
    no_decay_modules: Tuple[nn.Module] = (nn.LayerNorm, nn.Embedding),
) -> List[Dict[str, Any]]:
    """
    Group model parameters into parameter groups.
    """
    decay = set()
    no_decay = set()
    special = set()
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    for mod_name, mod in model.named_modules():
        for param_name, param in mod.named_parameters():
            full_param_name = f"{mod_name}.{param_name}" if mod_name else param_name
            if not param.requires_grad or full_param_name not in param_dict:
                continue  # frozen weights
            if hasattr(param, "_optim"):
                special.add(full_param_name)
            if full_param_name.endswith("bias"):
                no_decay.add(full_param_name)
            elif full_param_name.endswith("weight") and isinstance(mod, decay_modules):
                decay.add(full_param_name)
            elif full_param_name.endswith("weight") and isinstance(
                mod, no_decay_modules
            ):
                no_decay.add(full_param_name)
            elif getattr(param, "_no_weight_decay", False):
                no_decay.add(full_param_name)

    decay |= param_dict.keys() - no_decay - special
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), f"parameters {inter_params} are in both decay and no decay!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {param_dict.keys() - special - union_params} not separated"

    if weight_decay == 0.0 or not no_decay:
        param_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay | decay))],
                "weight_decay": weight_decay,
            }
        ]
    else:
        param_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
    # Add parameters with special hyperparameters
    # Unique dicts
    hparams = [
        dict(s) for s in set(frozenset(param_dict[pn]._optim.items()) for pn in special)
    ]
    for hparam in hparams:
        params = [
            param_dict[pn]
            for pn in sorted(list(special))
            if param_dict[pn]._optim == hparam
        ]
        param_groups.append({"params": params, **hparam})
    return param_groups


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
    param_groups = group_params(
        model,
        weight_decay=cfg.optimizer.hparams.weight_decay,
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

    optimizer.param_groups[0]["lr"] = scale_lr(
        cfg.optimizer.hparams.lr,
        cfg.training.batch_size,
        cfg.training.gradient_accumulation,
        fabric.world_size,
    )

    training_steps = (
        cfg.training.epochs
        * cfg.data.training_size
        // (
            cfg.training.batch_size
            * cfg.training.gradient_accumulation
            * fabric.world_size
        )
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(cfg.scheduler.warmup_fraction * training_steps),
        training_steps,
    )
    fabric.print(f"Total training steps: {training_steps:,}")

    return optimizer, scheduler
