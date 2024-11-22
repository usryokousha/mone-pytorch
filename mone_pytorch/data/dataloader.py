from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, ImageNet

from mone_pytorch.utils.augmentation import ClassificationAugmentation
def build_dataloaders(cfg):
    # Create augmentation instance
    train_augmentation = ClassificationAugmentation(
        img_size=cfg.model.img_size,
        randaugment_num_ops=cfg.augmentation.randaugment.num_ops,
        randaugment_magnitude=cfg.augmentation.randaugment.magnitude,
        random_erase_prob=cfg.augmentation.random_erase.probability,
        is_train=True
    )
    val_augmentation = ClassificationAugmentation(
        img_size=cfg.model.img_size,
        is_train=False
    )

    if 'imagenet1k' == cfg.data.dataset_name:
        # Create datasets
        train_dataset = ImageNet(
            root=cfg.data.imagenet1k.train_path,
            split="train",
            transform=train_augmentation
        )
    
        val_dataset = ImageNet(
            root=cfg.data.imagenet1k.val_path,
            split="val",
            transform=val_augmentation
        )
    elif 'imagenet21k' == cfg.data.dataset_name:
        train_dataset = ImageFolder(
            root=cfg.data.imagenet21k.train_path,
            transform=train_augmentation
        )
        val_dataset = ImageFolder(
            root=cfg.data.imagenet21k.val_path,
            transform=val_augmentation
        )
    else:
        raise ValueError(f"Config file does not contain valid dataset")

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

    return train_loader, val_loader