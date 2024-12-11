import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from lightning.fabric import Fabric, seed_everything
from torchmetrics import MetricCollection, Accuracy, Precision
from torch.optim.swa_utils import get_ema_multi_avg_fn, AveragedModel
from mone_pytorch.utils.flops import profile_fvcore

from mone_pytorch.data.dataloader import build_dataloaders
from mone_pytorch.train.initialize import initialize_model
from mone_pytorch.utils.optimizer import build_optimizer
from mone_pytorch.layers.routing import CapacityScheduler, compute_capacity_distribution
from mone_pytorch.utils.augmentation import CutMixup

from typing import Optional

def train_one_epoch(
    fabric: Fabric,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: MetricCollection,
    cfg: DictConfig,
    epoch: int,
    capacity_distribution: Optional[torch.Tensor] = None,
):
    model.train()
    cutmixup = CutMixup(
        cutmix_alpha=cfg.augmentation.cutmix.alpha,
        mixup_alpha=cfg.augmentation.mixup.alpha,
        num_classes=cfg.model.num_classes,
    )

    for batch_idx, (images, targets) in enumerate(train_loader):
        with fabric.no_backward_sync(
            model,
            enabled=(batch_idx + 1) % cfg.training.gradient_accumulation != 0,
        ):
            # Apply mixup/cutmix transforms
            images, targets = cutmixup(images, targets)
            outputs = model(images, c=capacity_distribution.to(fabric.device))

            loss = torch.nn.functional.cross_entropy(
                outputs, targets, label_smoothing=cfg.augmentation.label_smoothing
            )

            fabric.backward(loss)

        # Update metrics for every batch, not just after accumulation
        metrics.update(outputs.detach(), targets)

        # Only update optimizer when gradient accumulation is complete
        if (batch_idx + 1) % cfg.training.gradient_accumulation == 0:
            if hasattr(cfg.training, "grad_clip"):
                fabric.clip_gradients(model, optimizer, max_norm=cfg.training.grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if cfg.training.ema.enabled and epoch >= cfg.training.ema.start_epoch:
                if batch_idx % cfg.training.ema.update_interval == 0:
                    ema_model.update_parameters(model.module)

        # Only print when we complete a gradient accumulation step AND it's a 10th effective batch
        if (
            batch_idx + 1
        ) % cfg.training.gradient_accumulation == 0:  # Only on accumulation steps
            effective_batch = batch_idx // cfg.training.gradient_accumulation
            if fabric.is_global_zero and effective_batch % 10 == 0:
                total_effective_batches = (
                    len(train_loader) // cfg.training.gradient_accumulation
                )
                fabric.print(
                    f"Epoch: {epoch} | Effective Batch: {effective_batch}/{total_effective_batches} "
                    f"| Loss: {loss.item():.4f}"
                )

            # Log training metrics (moved inside gradient accumulation check)
            if (effective_batch + 1) % cfg.training.logging.interval == 0:
                # Adjust global step to account for gradient accumulation
                global_step = (
                    epoch * len(train_loader) + batch_idx
                ) // cfg.training.gradient_accumulation

                metrics_dict = metrics.compute()
                metrics_dict["train/learning_rate"] = scheduler.get_last_lr()[0]
                metrics_dict["train/loss"] = loss.detach()

                fabric.log_dict(metrics_dict, step=global_step)

                # Reset metrics after logging
                metrics.reset()


def validate(
    fabric: Fabric,
    model: torch.nn.Module,
    val_loader,
    metrics: MetricCollection,
    cfg: DictConfig,
    epoch: int,
    capacity_distribution: Optional[torch.Tensor] = None,
):
    """Run validation loop and log metrics."""
    model.eval()
    if capacity_distribution is not None:
        capacity_distribution = torch.tensor(
            [0.0] * (cfg.mone.num_experts - 1) + [1.0], 
            dtype=torch.float32,
        )
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images, c=capacity_distribution.to(fabric.device))
            loss = torch.nn.functional.cross_entropy(outputs, targets)

            # Update validation metrics
            metrics.update(outputs.detach(), targets)

    # Compute and log validation metrics
    metrics_dict = metrics.compute()
    metrics_dict["val/loss"] = loss.detach()

    if fabric.is_global_zero:
        fabric.print(
            f"Validation Results - Epoch: {epoch}\n"
            f"Top-1 Accuracy: {metrics_dict['val/acc/top1']:.2f}%\n"
            f"Top-5 Accuracy: {metrics_dict['val/acc/top5']:.2f}%\n"
            f"Precision: {metrics_dict['val/prec/top1']:.2f}%\n"
            f"Loss: {metrics_dict['val/loss']:.4f}"
        )

    fabric.log_dict(metrics_dict, step=epoch)

    # Reset validation metrics
    metrics.reset()

    return metrics_dict


@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    seed_everything(cfg.seed)

    # log experiment configuration
    loggers = []
    if "csv_logger" in cfg.loggers:
        csv_logger = hydra.utils.instantiate(cfg.loggers.csv_logger)
        loggers.append(csv_logger)
    if "wandb" in cfg.loggers:
        wandb_logger = hydra.utils.instantiate(cfg.loggers.wandb)
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg))
        loggers.append(wandb_logger)
    if "tensorboard" in cfg.loggers:
        tensorboard_logger = hydra.utils.instantiate(cfg.loggers.tensorboard)
        loggers.append(tensorboard_logger)

    # Initialize Fabric with loggers
    fabric = Fabric(
        accelerator="cuda",
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        strategy=cfg.training.strategy,
        loggers=loggers,
    )
    # set accumulation rank for Flora strategy
    if cfg.training.strategy.startswith("flora"):
        fabric._strategy.accumulation_rank = cfg.training.accumulation_rank
    fabric.launch()

    # Get dataloaders and augmentation
    train_loader, val_loader = build_dataloaders(cfg)

    train_steps = len(train_loader) // cfg.training.gradient_accumulation
    val_steps = len(val_loader)
    fabric.print(f"Effective training steps / epoch: {train_steps}")
    fabric.print(f"Effective validation steps / epoch: {val_steps}")

    # Create model with MoNE initialization
    with fabric.init_module():
        model = initialize_model(cfg, fabric)

    # Create optimizer and scheduler
    optimizer, scheduler = build_optimizer(cfg, model, fabric)

    # Setup with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # apply EMA optionally
    if cfg.training.ema.enabled:
        ema_avg_fn = get_ema_multi_avg_fn(cfg.training.ema.decay)
        ema_model = AveragedModel(model.module, avg_fn=ema_avg_fn)
        fabric.print(f"Using EMA with decay {cfg.training.ema.decay}")
    else:
        ema_model = None

    # Initialize metrics using ModuleDict and MetricCollection
    train_metrics = MetricCollection(
        {
            "acc/top1": Accuracy(
                task="multiclass", num_classes=cfg.model.num_classes, top_k=1
            ),
            "acc/top5": Accuracy(
                task="multiclass", num_classes=cfg.model.num_classes, top_k=5
            ),
            "prec/top1": Precision(
                task="multiclass", num_classes=cfg.model.num_classes, top_k=1
            ),
        },
        prefix="train/",
    )
    val_metrics = train_metrics.clone(prefix="val/")
    train_metrics = train_metrics.to(fabric.device)
    val_metrics = val_metrics.to(fabric.device)

    # Capacity scheduler
    if cfg.mone is not None:
        fabric.print("Using capacity scheduler for MoNE")
        capacity_scheduler = CapacityScheduler(
            patch_size=cfg.model.patch_size,
            image_size=cfg.model.img_size,
            max_epochs=cfg.training.epochs,
            min_capacity=cfg.mone.min_capacity,
            max_capacity=cfg.mone.max_capacity,
            num_experts=cfg.mone.num_experts,
            delta=cfg.mone.delta,
            beta=cfg.mone.beta,
            annealing_type=cfg.mone.annealing_type,
        )

    # Handle checkpoint resuming
    start_epoch = 0
    if cfg.training.checkpoints.resume_from_checkpoint:
        checkpoint = fabric.load(cfg.training.checkpoints.resume_from_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        train_metrics.load_state_dict(checkpoint["train_metrics_state_dict"])
        val_metrics.load_state_dict(checkpoint["val_metrics_state_dict"])
        if capacity_scheduler is not None:
            capacity_scheduler.load_state_dict(checkpoint["capacity_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # print settings
    fabric.print(f"Training for {cfg.training.epochs} epochs")
    fabric.print(f"Starting from epoch {start_epoch}")
    fabric.print(f"Using {cfg.training.precision} precision")
    fabric.print(f"Using {cfg.training.devices} devices")

    if cfg.training.logging.flops:
        if fabric.is_global_zero:
            import warnings
            import logging

            # Suppress fvcore logging
            logging.getLogger("fvcore").setLevel(logging.ERROR)
            if cfg.mone is not None:
                capacity_distribution = torch.tensor(
                    [0.0] * (cfg.mone.num_experts - 1) + [1.0], 
                    dtype=torch.float32,
                    device=fabric.device,
                )
                # Suppress warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, fca_total, _, aca_total = profile_fvcore(
                        model,
                    input_size=(3, cfg.model.img_size, cfg.model.img_size),
                    input_dtype=torch.bfloat16,
                )
            fabric.print("Profiling FLOPs ...")
            fabric.print(f"MoNE is {["enabled", "disabled"][cfg.mone is None]}")
            fabric.print(f"FLOPs: {fca_total * 1e-9:.5f} GFLOPs")
            fabric.print(f"Activation FLOPs: {aca_total * 1e-9:.5f} GFLOPs")

            if cfg.mone is not None:
                capacity_distribution = compute_capacity_distribution(
                    e_c=cfg.mone.effective_capacity,
                    E=cfg.mone.num_experts,
                    delta=cfg.mone.delta,
                    beta=cfg.mone.beta
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, fca_total, _, aca_total = profile_fvcore(
                        model,
                        input_size=(3, cfg.model.img_size, cfg.model.img_size),
                        input_dtype=torch.bfloat16,
                        other_inputs=(capacity_distribution.to(fabric.device),),
                    )
                fabric.print("Profiling FLOPs at 50% Effective Capacity ...")
                fabric.print(f"FLOPs: {fca_total * 1e-9:.5f} GFLOPs")
                fabric.print(f"Activation FLOPs: {aca_total * 1e-9:.5f} GFLOPs")

    # Training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        capacity_distribution = None
        adjusted_patch_size = None
        # Update capacity scheduler
        if capacity_scheduler is not None:
            if fabric.is_global_zero:
                fabric.print("Using capacity scheduler for MoNE")
                capacity_distribution, adjusted_patch_size = capacity_scheduler.update()
            else:
                capacity_distribution = torch.zeros(cfg.mone.num_experts)
                adjusted_patch_size = torch.zeros(1)
            capacity_distribution = fabric.broadcast(capacity_distribution)
            adjusted_patch_size = fabric.broadcast(adjusted_patch_size)

            model.module.set_input_size(
                img_size=cfg.model.img_size,
                patch_size=adjusted_patch_size.item(),
            )
        train_one_epoch(
            fabric=fabric,
            model=model,
            ema_model=ema_model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=train_metrics,  # Pass metrics to train_one_epoch
            cfg=cfg,
            epoch=epoch,
            capacity_distribution=capacity_distribution,
        )

        # Validation
        if epoch % cfg.training.val_interval == 0:
            validate(
                fabric=fabric,
                model=model,
                val_loader=val_loader,
                metrics=val_metrics,
                epoch=epoch,
                capacity_distribution=capacity_distribution,
            )

        # Save checkpoint
        if fabric.is_global_zero and epoch % cfg.training.save_interval == 0:
            fabric.save(
                cfg.training.checkpoints.path / f"epoch_{epoch}.pt",
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics_state_dict": train_metrics.state_dict(),
                    "val_metrics_state_dict": val_metrics.state_dict(),
                },
            )


if __name__ == "__main__":
    main()
