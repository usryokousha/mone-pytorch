from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, ImageNet

from mone_pytorch.utils.augmentation import ClassificationAugmentation
def build_dataloaders(cfg):
    # Create augmentation instance
    augmentation = ClassificationAugmentation(
        img_size=cfg.model.img_size,
        randaugment_num_ops=cfg.augmentation.randaugment.num_ops,
        randaugment_magnitude=cfg.augmentation.randaugment.magnitude,
        randaugment_num_layers=cfg.augmentation.randaugment.num_layers,
        mixup_alpha=cfg.augmentation.mixup.alpha if cfg.augmentation.mixup.enabled else 0.,
        cutmix_alpha=cfg.augmentation.cutmix.alpha if cfg.augmentation.cutmix.enabled else 0.,
        random_erase_prob=cfg.augmentation.random_erase.probability,
        label_smoothing=cfg.augmentation.label_smoothing,
    )
    if cfg.dataset.dataset_selected == "imagenet1k":
        # Create datasets
        train_dataset = ImageNet(
            root=cfg.paths.train_dir,
            split="train",
            transform=augmentation
        )
    
        val_dataset = ImageNet(
            root=cfg.paths.val_dir,
            split="val",
            transform=augmentation
        )
    elif cfg.dataset.dataset_selected == "imagenet21k":
        train_dataset = ImageFolder(
            root=cfg.paths.train_dir,
            transform=augmentation
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, augmentation 