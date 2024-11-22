import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from lightning.fabric import Fabric, seed_everything
from torchmetrics import MetricCollection, Accuracy, MeanMetric
from torch.nn import ModuleDict
from torch.optim.swa_utils import get_ema_multi_avg_fn, AveragedModel
from fvcore.nn.flop_count import flop_count

from mone_pytorch.data.dataloader import build_dataloaders
from mone_pytorch.train.initialize import initialize_model
from mone_pytorch.utils.optimizer import build_optimizer
from mone_pytorch.utils.augmentation import CutMixup

def train_one_epoch(
    fabric: Fabric,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: ModuleDict,
    cfg: DictConfig,
    epoch: int
):
    model.train()
    train_metrics = metrics['train']
    cutmixup = CutMixup(alpha=cfg.augmentation.cutmixup.alpha, num_classes=cfg.model.num_classes)
    
    for batch_idx, (images, targets) in enumerate(train_loader):        
        with fabric.no_backward_sync(model, enabled=(batch_idx + 1) % cfg.gradient_accumulation != 0):
            # Apply mixup/cutmix transforms
            images, targets = cutmixup(images, targets)
            outputs = model(images)
            
            if isinstance(targets, (list, tuple)):
                # Handle mixed labels from mixup/cutmix
                targets_a, targets_b, lam = targets
                loss = lam * torch.nn.functional.cross_entropy(
                    outputs, targets_a, label_smoothing=cfg.augmentation.label_smoothing
                ) + (1 - lam) * torch.nn.functional.cross_entropy(
                    outputs, targets_b, label_smoothing=cfg.augmentation.label_smoothing
                )
            else:
                loss = torch.nn.functional.cross_entropy(
                    outputs, targets, label_smoothing=cfg.augmentation.label_smoothing
                )
            
            fabric.backward(loss)
        
        # Only update metrics and optimizer when gradient accumulation is complete
        if (batch_idx + 1) % cfg.gradient_accumulation == 0:
            if hasattr(cfg.training, 'grad_clip'):
                fabric.clip_gradients(model, optimizer, max_norm=cfg.training.grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if cfg.ema.enabled and epoch >= cfg.ema.start_epoch:
                if batch_idx % cfg.ema.update_interval == 0:
                    ema_model.update_parameters(model.module)
            
            # Update metrics only after accumulation
            if not isinstance(targets, (list, tuple)):
                train_metrics.update('loss', loss)
                train_metrics.update('top1', outputs, targets)
                train_metrics.update('top5', outputs, targets)

            if cfg.profile.enabled:
                train_metrics.update('flops', flop_count(model, images).total() * 1e-9)
        
        if fabric.is_global_zero and batch_idx % 100 == 0:
            fabric.print(f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}")
        
        # Log training metrics
        if batch_idx % cfg.log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            # Adjust global step to account for gradient accumulation
            global_step = (epoch * len(train_loader) + batch_idx) // cfg.gradient_accumulation
            
            computed_metrics = train_metrics.compute()
            metrics_dict = {
                "train/loss": computed_metrics['loss'],
                "train/top1_accuracy": computed_metrics['top1'],
                "train/top5_accuracy": computed_metrics['top5'],
                "train/learning_rate": current_lr
            }
            if cfg.profile.enabled:
                metrics_dict['train/flops'] = computed_metrics['flops']
                
            fabric.log_dict(metrics_dict, step=global_step)
            
            # Reset metrics after logging
            train_metrics.reset()

def validate(fabric: Fabric, model: torch.nn.Module, val_loader, metrics: ModuleDict, epoch: int):
    """Run validation loop and log metrics."""
    model.eval()
    val_metrics = metrics['val']
    
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            
            # Update validation metrics
            val_metrics.update('loss', loss)
            val_metrics.update('top1', outputs, targets)
            val_metrics.update('top5', outputs, targets)
    
    # Compute and log validation metrics
    computed_metrics = val_metrics.compute()
    val_metrics_dict = {
        "val/loss": computed_metrics['loss'],
        "val/top1_accuracy": computed_metrics['top1'],
        "val/top5_accuracy": computed_metrics['top5']
    }
    
    if fabric.is_global_zero:
        fabric.print(
            f"Validation Results - Epoch: {epoch}\n"
            f"Top-1 Accuracy: {computed_metrics['top1']:.2f}%\n"
            f"Top-5 Accuracy: {computed_metrics['top5']:.2f}%\n"
            f"Loss: {computed_metrics['loss']:.4f}"
        )
    
    fabric.log_dict(val_metrics_dict, step=epoch)
    
    # Reset validation metrics
    val_metrics.reset()
    
    return computed_metrics

@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    seed_everything(cfg.seed)

    # log experiment configuration
    loggers = []
    if 'csv_logger' in cfg.logging:
        csv_logger = hydra.utils.instantiate(cfg.logging.csv_logger)
        loggers.append(csv_logger)
    if 'wandb' in cfg.logging:
        wandb_logger = hydra.utils.instantiate(cfg.logging.wandb)
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg))
        loggers.append(wandb_logger)
    if 'tensorboard' in cfg.logging:
        tensorboard_logger = hydra.utils.instantiate(cfg.logging.tensorboard)
        loggers.append(tensorboard_logger)

    # Initialize Fabric with loggers
    fabric = Fabric(
        accelerator="cuda",
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        strategy=cfg.training.strategy,
        loggers=loggers
    )
    # set accumulation rank for Flora strategy
    if cfg.training.strategy.startswith('flora'):
        fabric._strategy.accumulation_rank = cfg.training.accumulation_rank
    fabric.launch()

    # Get dataloaders and augmentation
    train_loader, val_loader = build_dataloaders(cfg)
    
    # Create model with MoNE initialization
    model = initialize_model(cfg, fabric)
    
    # Create optimizer and scheduler
    optimizer, scheduler = build_optimizer(cfg, model, fabric.world_size)
    
    # Setup with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # apply EMA optionally
    if cfg.ema.enabled:
        ema_avg_fn = get_ema_multi_avg_fn(cfg.ema.decay)
        ema_model = AveragedModel(model.module, avg_fn=ema_avg_fn)
        fabric.print(f"Using EMA with decay {cfg.ema.decay}")
    else:
        ema_model = None

    # Initialize metrics using ModuleDict and MetricCollection
    metrics = ModuleDict({
        'train': MetricCollection({
            'loss': MeanMetric(),
            'top1': Accuracy(task='multiclass', num_classes=cfg.model.num_classes, top_k=1),
            'top5': Accuracy(task='multiclass', num_classes=cfg.model.num_classes, top_k=5)
        }),
        'val': MetricCollection({
            'loss': MeanMetric(),
            'top1': Accuracy(task='multiclass', num_classes=cfg.model.num_classes, top_k=1),
            'top5': Accuracy(task='multiclass', num_classes=cfg.model.num_classes, top_k=5)
        })
    })
    if cfg.profile.enabled:
        metrics['train'].add_module('flops', MeanMetric())
    metrics = metrics.to(fabric.device)

    # Handle checkpoint resuming
    start_epoch = 0
    if cfg.get('resume_from_checkpoint'):
        checkpoint = fabric.load(cfg.resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        metrics.load_state_dict(checkpoint['metrics_state_dict'])  # Load metrics state
        start_epoch = checkpoint['epoch'] + 1

    # print settings
    fabric.print(f"Training for {cfg.train.epochs} epochs")
    fabric.print(f"Starting from epoch {start_epoch}")
    fabric.print(f"Effective batch size: {cfg.train.batch_size * cfg.train.devices * cfg.gradient_accumulation}")
    fabric.print(f"Effective learning rate: {cfg.train.learning_rate * cfg.train.devices * cfg.gradient_accumulation}")
    fabric.print(f"Using {cfg.train.precision} precision")
    fabric.print(f"Using {cfg.train.devices} devices")

    # Training loop
    for epoch in range(start_epoch, cfg.train.epochs):
        train_one_epoch(
            fabric=fabric,
            model=model,
            ema_model=ema_model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,  # Pass metrics to train_one_epoch
            cfg=cfg,
            epoch=epoch
        )
        
        # Validation
        if epoch % cfg.val_interval == 0:
            validate(
                fabric=fabric,
                model=model,
                val_loader=val_loader,
                metrics=metrics,
                epoch=epoch
            )
        
        # Save checkpoint
        if fabric.is_global_zero and epoch % cfg.save_interval == 0:
            fabric.save(cfg.checkpoints.path / f"epoch_{epoch}.pt", {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics_state_dict': metrics.state_dict(),  # Save metrics state
            })

if __name__ == "__main__":
    main() 