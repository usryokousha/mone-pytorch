import os
import math
import json
import time
from collections import OrderedDict
from typing import Collection, Optional, Tuple, Union, Callable
from argparse import Namespace
import heapq
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import swa_utils
from torch.optim.lr_scheduler import LambdaLR
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from wandb.integration.lightning.fabric import WandbLogger

from mone_pytorch.data.dataloader import build_dataloaders
from mone_pytorch.data.augmentation import CutMixup
from mone_pytorch.models import build_model
from utils import (
    accuracy,
    AverageMeter,
    create_scheduler,
    create_optimizer,
)


def train_one_epoch(
    fabric: Fabric,
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    cfg: DictConfig,
    lr_scheduler: Optional[LambdaLR] = None,
    model_ema: Optional[swa_utils.AveragedModel] = None,
    mixup_fn: Optional[CutMixup] = None,
):
    update_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    accum_steps = cfg.train.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if mixup_fn is not None:
            input, target = mixup_fn(input, target)

        data_time_m.update(accum_steps * (time.time() - data_start_time))

        with fabric.no_backward_sync(model, enabled=not need_update):
            output = model(input)
            loss = loss_fn(output, target)
            if accum_steps > 1:
                loss = loss / accum_steps

            fabric.backward(loss)

        if need_update:
            if cfg.train.clip_grad is not None:
                fabric.clip_gradients(
                    model,
                    optimizer,
                    max_norm=cfg.train.clip_grad,
                    mode=cfg.train.clip_mode,
                )
            optimizer.step()
            optimizer.zero_grad()

        losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        if model_ema is not None:
            model_ema.update_parameters(model.module)

        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % cfg.train.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            loss_avg, loss_now = losses_m.avg, losses_m.val
            update_sample_count = update_sample_count * fabric.world_size

            if fabric.is_global_zero:
                fabric.print(
                    f"Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} "
                    f"({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  "
                    f"Loss: {loss_now:#.3g} ({loss_avg:#.3g})  "
                    f"Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  "
                    f"({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  "
                    f"LR: {lr:.3e}  "
                    f"Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})"
                )

        if lr_scheduler is not None:
            lr_scheduler.step()

        update_sample_count = 0
        data_start_time = time.time()

    return OrderedDict([("loss", losses_m.avg, "lr", lr_scheduler.get_last_lr())])


def validate(
    fabric: Fabric,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    args: Namespace,
):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses_m.update(loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()

            if fabric.is_global_zero and (
                last_batch or batch_idx % args.log_interval == 0
            ):
                fabric.print(
                    f"Test: [{batch_idx:>4d}/{last_idx}]  "
                    f"Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  "
                    f"Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  "
                    f"Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  "
                    f"Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})"
                )

    metrics = OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
    )

    # Synchronize metrics across processes
    if fabric.world_size > 1:
        metrics = {k: v.item() for k, v in metrics.items()}

    return metrics


class CheckpointManager:
    def __init__(self, save_dir: str, topk: int = 10):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.topk = topk
        self.best_checkpoints = []  # Will store tuples of (metric, epoch, filename)

    def save_checkpoint(
        self, fabric: Fabric, state_dict: dict, metric: float, epoch: int
    ):
        filename = f"epoch_{epoch}_acc_{metric:.3f}.pth"
        save_path = self.save_dir / filename

        # Save the new checkpoint
        fabric.save(save_path, state_dict)

        # Add to best checkpoints (negative metric for max-heap of top-k best values)
        heapq.heappush(self.best_checkpoints, (-metric, epoch, str(save_path)))

        # Remove excess checkpoints
        while len(self.best_checkpoints) > self.topk:
            _, _, filepath_to_remove = heapq.heappop(self.best_checkpoints)
            try:
                Path(filepath_to_remove).unlink()  # Delete the file
            except FileNotFoundError:
                pass

    def get_topk_checkpoints(self):
        return heapq.nlargest(self.topk, self.best_checkpoints)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):

    # Set up logging
    # Pickup from previous Wandb run
    loggers = []
    if cfg.logging.wandb.enabled:
        logger = WandbLogger(
            project=cfg.wandb.project,
            save_dir=os.path.join(cfg.train.log_dir, "wandb"),
            id=cfg.train.checkpoint.id,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        loggers.append(logger)

    elif cfg.logging.csv.enabled:
        logger = CSVLogger(save_dir=os.path.join(cfg.train.log_dir, "csv"))
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        loggers.append(logger)

    # Set up CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    cfg.train.grad_accum_steps = max(1, cfg.train.grad_accum_steps)

    accelerator = cfg.train.get("accelerator", "auto")
    fabric = Fabric(
        accelerator=accelerator,
        devices=cfg.train.get("num_gpus", "auto"),
        precision=cfg.train.get("precision", "bf16-mixed"),
        loggers=loggers,
    )
    fabric.launch()

    # Indicate what distributed backends is being used
    fabric.print(f"Distributed backend: {fabric.strategy.__class__.__name__}")
    fabric.print(f"Using {cfg.train.get('precision', 'bf16-mixed')} precision")

    # Set up model
    model = build_model(cfg)

    if cfg.train.gradient_checkpointing:
        model.set_gradient_checkpointing(True)

    nested_name = cfg.nested.get("arch", "vit")
    fabric.print(f"Model: {cfg.model.name} created with {nested_name}")

    # Set up learning rate
    if cfg.train.lr is None:
        global_batch_size = (
            cfg.train.batch_size * fabric.world_size * cfg.train.grad_accum_steps
        )
        batch_ratio = global_batch_size / cfg.train.lr_batch_base
        cfg.train.lr = cfg.train.lr_base * batch_ratio

    fabric.print(
        f"Learning rate: {cfg.train.lr} calculated from base learning rate {cfg.train.lr_base}"
        f" and effective global batch size {global_batch_size} with linear scaling"
    )

    optimizer = create_optimizer(model, cfg.train.lr, cfg.train.weight_decay)

    train_loader, val_loader = build_dataloaders(cfg)
    fabric.print(
        f"Created dataloaders with {len(train_loader)} training samples "
        f"and {len(val_loader)} validation samples"
    )

    # Set up learning rate scheduler
    if cfg.train.scheduler.enabled:
        updates_per_epoch = (
            len(train_loader) + cfg.train.grad_accum_steps - 1
        ) // cfg.train.grad_accum_steps
        lr_scheduler = create_scheduler(
            optimizer,
            cfg.train.scheduler.num_warmup_epochs,
            cfg.train.scheduler.num_training_epochs,
            cfg.train.scheduler.num_cycles,
            updates_per_epoch,
        )
        fabric.print(
            f"Created LR scheduler with warmup steps {cfg.train.scheduler.num_warmup_epochs} "
            f"and total steps {cfg.train.scheduler.num_training_epochs}"
        )

    # Set up EMA
    if cfg.train.ema.enabled:
        ema_model = swa_utils.AveragedModel(
            model, multi_avg_fn=swa_utils.get_ema_multi_avg_fn
        )
        ema_model.update_parameters(model.module)
        fabric.print(f"Created EMA model with decay {cfg.train.ema.decay}")

    # Apply torch compile
    if cfg.train.compile.enabled:
        model = torch.compile(
            model, mode=cfg.train.compile.mode, backend=cfg.train.compile.backend
        )
        if ema_model is not None:
            ema_model = torch.compile(
                ema_model.module,
                mode=cfg.train.compile.mode,
                backend=cfg.train.compile.backend,
            )

    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # Create mixup function
    mixup_fn = None
    if cfg.train.mixup.enabled:
        mixup_fn = CutMixup(
            mixup_alpha=cfg.train.mixup.mixup_alpha,
            cutmix_alpha=cfg.train.mixup.cutmix_alpha,
        )

    # Create loss function
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing).to(
        fabric.device
    )

    # Resume from checkpoint
    start_epoch = None
    if cfg.train.checkpoint.path:
        state = {
            "model": model,
            "optimizer": optimizer,
            "ema_model": ema_model,
            "lr_scheduler": lr_scheduler,
            "epoch": 0,
        }
        fabric.print(f"Resuming from checkpoint {cfg.train.checkpoint.path}")
        fabric.load(cfg.train.checkpoint.path, state)
        start_epoch = state["epoch"] + 1
        fabric.print(f"Resuming from epoch {start_epoch}")

    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(
        save_dir=cfg.train.checkpoint.path,
        topk=cfg.train.checkpoint.get("keep_topk", 3),
    )

    try:
        for epoch in range(start_epoch, cfg.train.epochs):
            # Nested model training
            if cfg.nested.enabled:
                effective_capacity = cfg.nested.effective_capacity
                model.module.update_capacity(effective_capacity)

            # Train the model
            train_metrics = train_one_epoch(
                fabric,
                epoch,
                model,
                train_loader,
                optimizer,
                loss_fn,
                cfg,
                lr_scheduler,
                ema_model,
                mixup_fn,
            )
            fabric.log_dict(train_metrics, step=epoch)

            # Validate the model
            val_metrics = validate(fabric, model, val_loader, loss_fn, cfg)
            fabric.log_dict(val_metrics, step=epoch)

            # Save checkpoint using the checkpoint manager
            if epoch % cfg.train.checkpoint.save_interval == 0:
                ckpt_manager.save_checkpoint(
                    fabric,
                    {
                        "model": model,
                        "optimizer": optimizer,
                        "ema_model": ema_model,
                        "lr_scheduler": lr_scheduler,
                        "epoch": epoch,
                    },
                    val_metrics["top1"],  # Using top1 accuracy as the metric
                    epoch,
                )

    except KeyboardInterrupt:
        pass

    topk_checkpoints = ckpt_manager.get_topk_checkpoints()
    topk_metrics = {}
    for metric, epoch, _ in topk_checkpoints:
        topk_metrics.update({"validation_accuracy": metric, "epoch": epoch})

    fabric.print(
        "Training finished with top metrics: \n" f"{json.dumps(topk_metrics, indent=4)}"
    )


if __name__ == "__main__":
    main()
