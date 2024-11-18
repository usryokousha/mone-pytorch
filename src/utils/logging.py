from typing import List
from lightning.fabric.loggers import Logger, TensorBoardLogger, CSVLogger
from wandb.integration.lightning.fabric import WandbLogger
from omegaconf import DictConfig
from pathlib import Path

def get_loggers(cfg: DictConfig) -> List[Logger]:
    """Initialize loggers based on config.
    
    Supports TensorBoard, CSV, and Weights & Biases loggers.
    
    Args:
        cfg: Configuration containing logger settings
        
    Returns:
        List of initialized loggers
    """
    loggers = []
    
    if not cfg.get("logging"):
        return loggers
        
    log_dir = Path(cfg.logging.get("log_dir", "logs"))
    experiment_name = cfg.logging.get("name", "")
    
    # Add TensorBoard logger if specified
    if "tensorboard" in cfg.logging.get("loggers", []):
        loggers.append(TensorBoardLogger(
            root_dir=log_dir / "tensorboard",
            name=experiment_name,
            default_hp_metric=False  # Don't log a placeholder metric
        ))
    
    # Add CSV logger if specified
    if "csv" in cfg.logging.get("loggers", []):
        loggers.append(CSVLogger(
            root_dir=log_dir / "csv",
            name=experiment_name
        ))
    
    # Add WandB logger if specified
    if "wandb" in cfg.logging.get("loggers", []):
        wandb_config = cfg.logging.get("wandb", {})
        loggers.append(WandbLogger(
            project=wandb_config.get("project", "my-project"),
            name=experiment_name,
            save_dir=str(log_dir / "wandb"),
            log_model=wandb_config.get("log_model", False),
            **wandb_config.get("kwargs", {})
        ))
    
    return loggers 